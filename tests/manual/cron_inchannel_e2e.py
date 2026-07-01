"""
Offline E2E harness for continuable in-channel cron (specs/cron-inchannel-continuable).

Drives BOTH legs of the feature against the REAL code paths — no network, no
Slack contact, no Socket Mode — and asserts they converge on the same
shared-channel session key ``(slack, <channel>, None)``:

  LEG 1 (delivery):  cron.scheduler._deliver_result(...) with a live Slack
    adapter + cron_continuable_surface=in_channel  →  the thread-open branch is
    SKIPPED and the shipped origin-mirror seeds (slack, C, None) with
    thread_id=None (F5). Asserted via the mirror_to_session call.

  LEG 2 (reply):  SlackAdapter._handle_slack_message(...) for a plain top-level
    channel message under reply_in_thread=false  →  the inbound session keying
    stamps thread_id=None, i.e. the SAME (slack, C, None) bucket the seed landed
    in. Asserted via the dispatched MessageEvent.source.thread_id.

If both legs report thread_id=None for the same channel, a plain channel reply
after an in_channel cron delivery resolves to the seeded session with the brief
in context (G3) — with NO visible thread (G2).

Run from INSIDE the worktree so the worktree's code loads, not the editable
main-checkout install:

    cd <worktree>
    PYTHONPATH="$PWD" ../../.venv/bin/python tests/manual/cron_inchannel_e2e.py

No real names anywhere (synthetic channel C_TEST / user U_TESTER / bot U_TESTBOT).
"""

import asyncio
import sys
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

# --- confirm we are running the WORKTREE's code, not the main checkout --------
import cron.scheduler as _sched_mod
import plugins.platforms.slack.adapter as _slack_mod

CHANNEL = "C_TEST"
BOT_UID = "U_TESTBOT"
USER_UID = "U_TESTER"
BRIEF = "Your daily brief: 3 PRs need review."


def leg1_delivery_seeds_flat_channel_session():
    """Real _deliver_result down the live-adapter path, in_channel mode."""
    from gateway.config import Platform

    # A Slack pconfig opting into in_channel.
    pconfig = MagicMock()
    pconfig.enabled = True
    pconfig.extra = {"cron_continuable_surface": "in_channel", "reply_in_thread": False}
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.SLACK: pconfig}

    # A live Slack-like adapter that advertises the capability + sends OK.
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True)
    adapter.supports_inchannel_continuable = True

    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        fut = Future()
        try:
            fut.set_result(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    job = {
        "id": "brief-job",
        "name": "Daily Brief",
        "deliver": "origin",
        "origin": {"platform": "slack", "chat_id": CHANNEL},  # channel origin, no thread
        "attach_to_session": True,
    }

    open_thread_calls = []
    real_open = _sched_mod._open_continuable_cron_thread

    def _spy_open(*a, **k):
        open_thread_calls.append((a, k))
        return real_open(*a, **k)

    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler._open_continuable_cron_thread", side_effect=_spy_open), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
         patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
        _sched_mod._deliver_result(
            job, BRIEF, adapters={Platform.SLACK: adapter}, loop=loop,
        )

    assert not open_thread_calls, "LEG1 FAIL: thread-open was attempted in in_channel mode (G2)"
    assert mirror_mock.call_count == 1, "LEG1 FAIL: brief was not mirrored/seeded"
    kw = mirror_mock.call_args
    seeded_platform = kw.args[0]
    seeded_chat = kw.args[1]
    seeded_text = kw.args[2]
    seeded_thread = kw.kwargs.get("thread_id")
    assert seeded_platform == "slack" and seeded_chat == CHANNEL, "LEG1 FAIL: wrong seed target"
    assert seeded_thread is None, f"LEG1 FAIL: seed carried a thread_id ({seeded_thread!r}), not flat"
    assert BRIEF in seeded_text, "LEG1 FAIL: brief text missing from seed"
    return ("slack", seeded_chat, seeded_thread)


async def _leg2_reply_keys_flat_channel_session():
    """Real _handle_slack_message for a plain channel reply, reply_in_thread=false."""
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, token="xoxb-test-not-a-real-token")
    config.extra["reply_in_thread"] = False
    # A channel where flat continuable-cron makes sense is one the bot answers
    # ambiently — otherwise the user must @-mention on every reply (that is a
    # pre-existing, orthogonal channel-gating choice, not part of this feature).
    config.extra["require_mention"] = False
    a = _slack_mod.SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT_UID
    a._running = True

    captured = []
    a.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))

    event = {
        "channel": CHANNEL,
        "channel_type": "channel",
        "user": USER_UID,
        # A plain channel reply — the user just types back, no @mention, no thread.
        "text": "thanks, show me the first one",
        "ts": "1700000000.000900",
    }

    with patch.object(a, "_resolve_user_name", new=AsyncMock(return_value="tester")):
        await a._handle_slack_message(event)

    assert len(captured) == 1, "LEG2 FAIL: plain channel reply was dropped (not continued)"
    src = captured[0].source
    assert src.thread_id is None, (
        f"LEG2 FAIL: reply keyed thread_id={src.thread_id!r}, not the flat "
        "channel session — a threaded reply would NOT resolve to the seed"
    )
    return ("slack", src.chat_id, src.thread_id)


def main():
    print(f"scheduler module: {_sched_mod.__file__}")
    print(f"slack adapter module: {_slack_mod.__file__}")
    if "cron-inchannel" not in _sched_mod.__file__:
        print("WARNING: not running the worktree's scheduler — set PYTHONPATH=$PWD", file=sys.stderr)

    seed_key = leg1_delivery_seeds_flat_channel_session()
    print(f"LEG 1 (delivery seed) → session key {seed_key}")

    reply_key = asyncio.run(_leg2_reply_keys_flat_channel_session())
    print(f"LEG 2 (inbound reply) → session key {reply_key}")

    # Convergence: both legs must land on (slack, CHANNEL, None).
    assert seed_key[0] == reply_key[0], "platform mismatch"
    assert str(seed_key[1]) == str(reply_key[1]), (
        f"channel mismatch: seed {seed_key[1]} vs reply {reply_key[1]}"
    )
    assert seed_key[2] is None and reply_key[2] is None, "one leg was threaded"
    print(
        f"\nPASS: both legs converge on (slack, {CHANNEL}, None) — a plain "
        "channel reply after an in_channel cron delivery continues the job "
        "in-context, with no visible thread."
    )


if __name__ == "__main__":
    main()
