"""
DM-path verification for in_channel continuable cron (Option A scoping).

Option A: `cron_continuable_surface` is a CHANNEL feature. For a 1:1 DM the
governing knob is the pre-existing `dm_top_level_threads_as_sessions` — a DM has
no thread-vs-timeline split, so DM continuation works ONLY when top-level DMs
share one flat session (`dm_top_level_threads_as_sessions: false`).

This harness PROVES that scoping against the REAL inbound handler
(`SlackAdapter._handle_slack_message`) — no hard-coded thread_id assumption (the
mistake that made the earlier E2E falsely pass):

  SCENARIO 1 (the supported config, false): a top-level DM reply keys to the
    flat `…:dm:<chat>` session — the SAME key the cron seed
    (`_seed_cron_channel_session`, is_dm=True) creates. → continuation works.

  SCENARIO 2 (the default, True — CONTROL): a top-level DM reply keys to a
    per-message `…:dm:<chat>:<ts>` session — DIVERGES from the flat seed. → this
    is exactly why in_channel does NOT give DM continuation under the default,
    and why Option A documents the requirement rather than pretending otherwise.

Run from INSIDE the worktree:
    cd <worktree>
    PYTHONPATH="$PWD" ../../.venv/bin/python tests/manual/cron_inchannel_dm_e2e.py

No real names. Uses a throwaway HERMES_HOME.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="cron_dm_e2e_")

import cron.scheduler as sched  # noqa: E402
from gateway.config import PlatformConfig, Platform  # noqa: E402
from gateway.session import build_session_key, SessionSource  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

DM_CHAT = "D_TESTDM"
BOT = "U_TESTBOT"
USER = "U_TESTER"


async def _inbound_dm_reply_key(dm_threads_as_sessions: bool):
    """Drive the REAL _handle_slack_message for a top-level DM message and
    return (session_key, source) the dispatched MessageEvent resolves to."""
    cfg = PlatformConfig(enabled=True, token="xoxb-test-not-real")
    cfg.extra["dm_top_level_threads_as_sessions"] = dm_threads_as_sessions
    a = SlackAdapter(cfg)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT
    a._running = True

    captured = []
    a.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))

    event = {
        "channel": DM_CHAT,
        "channel_type": "im",          # 1:1 DM
        "user": USER,
        "text": "how many items in that brief?",
        "ts": "1782999999.000100",      # a NEW top-level DM message (no thread_ts)
    }
    with patch.object(a, "_resolve_user_name", new=AsyncMock(return_value="tester")):
        await a._handle_slack_message(event)

    assert len(captured) == 1, "DM reply was dropped by the handler"
    src = captured[0].source
    return build_session_key(src), src


def _seed_key() -> str:
    """The session key the cron in_channel DM seed creates (is_dm=True, flat)."""
    seed_source = SessionSource(
        platform=Platform.SLACK, chat_id=DM_CHAT, chat_type="dm",
        user_id=USER, thread_id=None,
    )
    return build_session_key(seed_source)


def main():
    print(f"adapter module: {SlackAdapter.__module__} ({sched.__file__.rsplit('/',2)[0]})")
    seed_key = _seed_key()
    print(f"\ncron in_channel DM seed key: {seed_key}")

    # SCENARIO 1 — supported config (false): reply MUST converge on the seed.
    key_false, src_false = asyncio.run(_inbound_dm_reply_key(False))
    print(f"\n[dm_top_level_threads_as_sessions=false]  reply key: {key_false}")
    print(f"    thread_id on reply source: {src_false.thread_id!r}")
    assert key_false == seed_key, (
        f"FAIL: with the supported config, reply key {key_false} != seed {seed_key}"
    )
    print("    ✓ CONVERGES with the seed → DM continuation works")

    # SCENARIO 2 — default (true): reply DIVERGES (this is why A documents the req).
    key_true, src_true = asyncio.run(_inbound_dm_reply_key(True))
    print(f"\n[dm_top_level_threads_as_sessions=true (default)]  reply key: {key_true}")
    print(f"    thread_id on reply source: {src_true.thread_id!r}")
    assert key_true != seed_key, (
        "unexpected: default DM keying matched the flat seed — the control is wrong"
    )
    print("    ✓ DIVERGES from the seed (per-message session) → in_channel gives")
    print("      NO DM continuation under the default; false is required (Option A)")

    print(
        "\nPASS: Option A verified against the REAL inbound handler.\n"
        "  • DM continuable cron works IFF dm_top_level_threads_as_sessions: false\n"
        "    (reply and seed converge on the flat …:dm:<chat> session).\n"
        "  • Under the default (true) they diverge — documented, not silently broken."
    )


if __name__ == "__main__":
    main()
