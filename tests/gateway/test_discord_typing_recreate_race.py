"""Regression tests for the Discord typing-indicator RECREATE race.

Distinct from ``test_discord_typing_stop.py`` (which guards the 429-driven
stuck-bubble, PR #34). This file guards a SECOND stuck-"is typing…" bug with
ZERO rate-limiting involved:

Field report (2026-06-20, #voice-assitant): an "Apollo is typing…" bubble
stayed lit ~26 min after the turn was fully delivered, with no 429s on the
channel. Root cause: a check-then-act race across teardown. Under concurrent
same-channel turns, a late ``send_typing`` (from base ``_keep_typing``'s
refresh tick, or the follow-up restart at gateway/run.py:16006) can RECREATE
the per-channel ``_typing_loop`` AFTER the owning turn's ``stop_typing``
already popped + cancelled it — leaving an orphaned loop that re-POSTs the
typing indicator forever with no live owner to cancel it.

``test_orphaned_loop_outlives_stop`` is the AUTHORITATIVE red-on-``main``
proof: it drives the REAL, unmodified ``DiscordAdapter`` send_typing /
stop_typing / _typing_loop through the field interleaving and asserts that
after the owning turn's stop, NO typing task is still POSTing. On pre-fix
code it FAILS (orphan survives + keeps POSTing); post-fix it PASSES.

The remaining tests are post-fix structural guards for the ownership-token
primitive (they exercise the ``token=`` seam that only exists post-fix, so
they are green-only by construction — not claimed red-pre-fix).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Minimal mock ``discord`` module exposing ``http.Route`` for the loop."""
    existing = sys.modules.get("discord")
    if existing is not None and getattr(existing, "_typing_test_ready", False):
        return
    discord_mod = existing if existing is not None else MagicMock()
    discord_mod.http = SimpleNamespace(Route=lambda *a, **k: SimpleNamespace(**k))
    discord_mod._typing_test_ready = True
    sys.modules["discord"] = discord_mod


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _make_adapter():
    """Real DiscordAdapter with a POST-counting mocked http client."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    posts = {"n": 0}

    async def _count_post(*_a, **_k):
        posts["n"] += 1
        return None  # success — no 429 anywhere in this file

    http = SimpleNamespace(request=AsyncMock(side_effect=_count_post))
    adapter._client = SimpleNamespace(http=http)
    return adapter, posts


def _supports_token() -> bool:
    """True once send_typing accepts the post-fix ``token`` kwarg."""
    import inspect
    return "token" in inspect.signature(DiscordAdapter.send_typing).parameters


# --------------------------------------------------------------------------
# AUTHORITATIVE red-on-main repro (drives the real unmodified code path).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orphaned_loop_outlives_stop():
    """The recreate race: a late send_typing after stop_typing orphans a loop.

    Interleaving (no 429s):
      1. Turn A starts typing  -> loop A registered, POSTing.
      2. Turn A completes; stop_typing() pops + cancels loop A.
      3. A late send_typing() lands (base _keep_typing tick / run.py:16006
         follow-up) for the SAME chat AFTER step 2 -> recreates a loop.
      4. The turn is over: nobody calls stop_typing again.

    PRE-FIX: step 3 unconditionally arms a new loop (the duplicate-guard is
    empty because step 2 cleared the dict), so an orphan survives and keeps
    POSTing -> assertion FAILS (this is the red proof).

    POST-FIX: the late send_typing carries turn-A's now-superseded token (or,
    for the tokenless legacy path, ownership was relinquished), so it refuses
    to arm -> no orphan -> assertion PASSES.
    """
    adapter, posts = _make_adapter()
    chat = "voice-assitant"

    # --- Turn A: start typing, let the loop make a POST. ---
    if _supports_token():
        token_a = adapter._issue_typing_token(chat)
        await adapter.send_typing(chat, token=token_a)
    else:
        token_a = None
        await adapter.send_typing(chat)
    await asyncio.sleep(0)  # let loop A run its first POST
    assert adapter._typing_tasks.get(chat) is not None

    # --- Turn A completes: stop typing (owning turn's teardown). ---
    if _supports_token():
        # Mirror _stop_typing_refresh: relinquish ownership for this token.
        if getattr(adapter, "_typing_owner", {}).get(chat) == token_a:
            adapter._typing_owner.pop(chat, None)
        await adapter.stop_typing(chat, token=token_a)
    else:
        await adapter.stop_typing(chat)
    assert chat not in adapter._typing_tasks

    # --- A LATE send_typing for turn A lands AFTER stop (the race). ---
    # This models base _keep_typing's in-flight refresh tick, or the
    # follow-up restart at gateway/run.py:16006, firing for the just-stopped
    # turn. It carries turn-A's (now superseded/relinquished) token.
    if _supports_token():
        await adapter.send_typing(chat, token=token_a)
    else:
        await adapter.send_typing(chat)

    # Drive the event loop and count POSTs over a window.
    posts_after_stop_start = posts["n"]
    for _ in range(5):
        await asyncio.sleep(0.01)
    posts_during_window = posts["n"] - posts_after_stop_start

    orphan = adapter._typing_tasks.get(chat)
    # Clean up any orphan so the test process doesn't leak a live task.
    if orphan is not None:
        orphan.cancel()
        try:
            await orphan
        except (asyncio.CancelledError, Exception):
            pass

    assert orphan is None, (
        "Orphaned _typing_loop survived the owning turn's stop_typing "
        "(recreate race) — the 'is typing…' bubble would never clear."
    )
    assert posts_during_window == 0, (
        f"Orphaned loop kept POSTing typing ({posts_during_window} POSTs) "
        "after the turn ended — this is the stuck-bubble field symptom."
    )


# --------------------------------------------------------------------------
# Post-fix structural guards (green-only; require the token seam).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_late_send_typing_after_stop_does_not_rearm():
    """A send_typing carrying a superseded token must NOT arm a loop."""
    if not _supports_token():
        pytest.skip("pre-fix: no token seam")
    adapter, posts = _make_adapter()
    chat = "c"
    tok = adapter._issue_typing_token(chat)
    await adapter.send_typing(chat, token=tok)
    await asyncio.sleep(0)
    # Owning teardown relinquishes ownership, then stop.
    adapter._typing_owner.pop(chat, None)
    await adapter.stop_typing(chat, token=tok)
    # Late stale arm with the OLD token -> must be refused.
    await adapter.send_typing(chat, token=tok)
    await asyncio.sleep(0)
    assert chat not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_supersede_does_not_false_stop_newer_turn():
    """Turn B's loop must survive turn A's owner-matched teardown."""
    if not _supports_token():
        pytest.skip("pre-fix: no token seam")
    adapter, posts = _make_adapter()
    chat = "c"
    tok_a = adapter._issue_typing_token(chat)
    await adapter.send_typing(chat, token=tok_a)
    await asyncio.sleep(0)
    # Turn B supersedes A (new turn) and arms its own loop.
    tok_b = adapter._issue_typing_token(chat)
    # stop A's loop the way a recreate would, then B arms.
    await adapter.stop_typing(chat, token=tok_a)
    await adapter.send_typing(chat, token=tok_b)
    await asyncio.sleep(0)
    b_task = adapter._typing_tasks.get(chat)
    assert b_task is not None and not b_task.done(), "newer turn B was false-stopped"
    # Now B's real teardown clears it.
    adapter._typing_owner.pop(chat, None)
    await adapter.stop_typing(chat, token=tok_b)
    assert chat not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_owner_dict_is_bounded():
    """N full turn lifecycles on one chat -> owner dict stays one-per-chat."""
    if not _supports_token():
        pytest.skip("pre-fix: no token seam")
    adapter, posts = _make_adapter()
    chat = "c"
    for _ in range(5):
        tok = adapter._issue_typing_token(chat)
        await adapter.send_typing(chat, token=tok)
        await asyncio.sleep(0)
        adapter._typing_owner.pop(chat, None)
        await adapter.stop_typing(chat, token=tok)
    assert len(adapter._typing_owner) <= 1
