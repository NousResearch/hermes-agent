"""Slack flat mode (``reply_in_thread: false``): the typing indicator along the real call path.

Inbound Slack message -> the adapter's own ``MessageEvent`` (session thread derivation) ->
``_thread_metadata_for_event`` -> ``_start_typing_refresh`` / ``_keep_typing`` -> ``send_typing``
-> ``stop_typing``, plus the runner's progress/status lane and the turn-end stop. A flat turn has
no thread, so the inbound message's own ts anchors the status through ``agents.sessions.setStatus``
(no thread created); the legacy ``assistant.threads.setStatus`` would open one, so it is never hit.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import _mark_notify_metadata, _reply_anchor_for_event, _thread_metadata_for_event
from gateway.platforms.base_status_anchor import STATUS_ANCHOR_MESSAGE_ID
from gateway.run import GatewayRunner
from tests.gateway.slack_status_clock import SimulatedTurnClock
from tests.gateway.test_slack import SlackAdapter, _slack_mod
from tests.waiting import async_wait_until

TEAM = "T0FLAT"
MSG_TS = "1761300000.000100"
PARENT_TS = "1761299000.000001"
QUEUED_TS = "1761300050.000300"
LATER_TS = "1761300040.000200"


@pytest.fixture(autouse=True)
def _sessions_api_available():
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = True
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev


def _adapter(**extra):
    config = PlatformConfig(enabled=True, token="xoxb-fake-token", extra={"reply_in_thread": False, **extra})
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.agents_sessions_setStatus = AsyncMock()
    a._app.client.assistant_threads_setStatus = AsyncMock()
    a._app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1761300001.000200"})
    a._resolve_user_name = AsyncMock(return_value="Ada")
    a._resolve_channel_name = AsyncMock(return_value="general")
    a._humanize_user_mentions = AsyncMock(side_effect=lambda text, **_kw: text)
    a._set_assistant_thread_title = AsyncMock()
    return a


async def _inbound(adapter, *, channel, is_dm, thread_ts=None, user="U1", ts=MSG_TS):
    """The MessageEvent the adapter's inbound handler builds for this Slack message."""
    raw = {"type": "message", "channel": channel, "user": user, "team": TEAM, "ts": ts, "text": "hi",
           "channel_type": "im" if is_dm else "channel", **({"thread_ts": thread_ts} if thread_ts else {})}
    session_thread = adapter._session_thread_ts(raw, ts, is_dm, {})
    return await adapter._build_message_event(
        raw, text="hi", original_text="hi", command_probe_text="hi", is_command_text=False,
        channel_id=channel, team_id=TEAM, ts=ts, user_id=user, thread_ts=session_thread, is_dm=is_dm,
        media_urls=[], media_types=[], media_text_inlined=[], channel_context=None)


def _owned_metadata(adapter, event):
    """What ``_process_message_background`` hands ``_keep_typing``: anchored and session-owned."""
    return _thread_metadata_for_event(event, adapter._event_session_key(event))


def _status_calls(adapter):
    return [(c.kwargs["thread_ts"], c.kwargs["status"]) for c in adapter._app.client.agents_sessions_setStatus.call_args_list]


FLAT_TOP_LEVEL = [
    pytest.param("D0DM", True, {"dm_top_level_threads_as_sessions": False}, id="dm-one-session-per-channel"),
    pytest.param("D0DM", True, {}, id="dm-session-per-top-level-message"),
    pytest.param("C0CHAN", False, {}, id="channel"),
]


@pytest.mark.parametrize("channel,is_dm,extra", FLAT_TOP_LEVEL)
@pytest.mark.asyncio
async def test_base_typing_refresh_throttles_the_heartbeat(channel, is_dm, extra, monkeypatch):
    """The base ``_keep_typing`` heartbeat ticks every 2s, but the sessions-API
    ``processing`` payload never changes, so a 3-minute turn (91 ticks, t=0..180) calls
    ``setStatus`` exactly at t=0 plus the 60s keep-alives at t=60/120/180, then clears — not ~90."""
    adapter = _adapter(**extra)
    event = await _inbound(adapter, channel=channel, is_dm=is_dm)
    metadata = _thread_metadata_for_event(event)
    clock = SimulatedTurnClock(turn_seconds=180.0).install(monkeypatch)
    sent = clock.record(adapter._app.client.agents_sessions_setStatus)
    # stop_event=None: the ``asyncio.sleep(interval)`` branch, which the simulated clock paces; the
    # timeout only guards against a hang — the turn ends by tick count in ``SimulatedTurnClock``.
    await asyncio.wait_for(
        SlackAdapter._keep_typing(adapter, channel, interval=2.0, metadata=metadata, stop_event=None),
        timeout=30.0)

    assert clock.now == 180.0 and len(clock.sleeps) == 90
    assert sent == [(0.0, "processing"), (60.0, "processing"), (120.0, "processing"),
                    (180.0, "processing"), (180.0, "active")]
    assert {c[0] for c in _status_calls(adapter)} == {MSG_TS}
    assert adapter._active_status_threads == {}
    adapter._app.client.assistant_threads_setStatus.assert_not_called()


@pytest.mark.parametrize("channel,is_dm,extra", FLAT_TOP_LEVEL)
@pytest.mark.asyncio
async def test_flat_reply_send_stays_top_level(channel, is_dm, extra):
    adapter = _adapter(**extra)
    event = await _inbound(adapter, channel=channel, is_dm=is_dm)
    metadata = _mark_notify_metadata(_thread_metadata_for_event(event))
    result = await adapter.send(channel, "done", reply_to=_reply_anchor_for_event(event), metadata=metadata)
    assert result.success
    assert adapter._app.client.chat_postMessage.call_args.kwargs.get("thread_ts") is None


@pytest.mark.asyncio
async def test_flat_typing_never_falls_back_to_legacy_when_sessions_fail():
    adapter = _adapter()
    adapter._app.client.agents_sessions_setStatus.side_effect = Exception("not_allowed")
    event = await _inbound(adapter, channel="C0CHAN", is_dm=False)
    metadata = _thread_metadata_for_event(event)
    await adapter.send_typing("C0CHAN", metadata=metadata)
    await adapter.stop_typing("C0CHAN", metadata=metadata)
    adapter._app.client.assistant_threads_setStatus.assert_not_called()


@pytest.mark.asyncio
async def test_real_thread_reply_keeps_thread_status_and_legacy_fallback():
    adapter = _adapter()
    adapter._app.client.agents_sessions_setStatus.side_effect = Exception("invalid_arguments")
    event = await _inbound(adapter, channel="C0CHAN", is_dm=False, thread_ts=PARENT_TS)
    metadata = _thread_metadata_for_event(event)
    await adapter.send_typing("C0CHAN", metadata=metadata)
    assert _status_calls(adapter) == [(PARENT_TS, "processing")]
    adapter._app.client.assistant_threads_setStatus.assert_called_once()
    assert adapter._app.client.assistant_threads_setStatus.call_args.kwargs["thread_ts"] == PARENT_TS
    await adapter.send(
        "C0CHAN", "done", reply_to=_reply_anchor_for_event(event), metadata=_mark_notify_metadata(metadata))
    assert adapter._app.client.chat_postMessage.call_args.kwargs.get("thread_ts") == PARENT_TS


@pytest.mark.asyncio
async def test_reply_in_thread_true_keeps_the_synthetic_thread_root():
    adapter = _adapter(reply_in_thread=True)
    adapter._app.client.agents_sessions_setStatus.side_effect = Exception("invalid_arguments")
    event = await _inbound(adapter, channel="C0CHAN", is_dm=False)
    await adapter.send_typing("C0CHAN", metadata=_thread_metadata_for_event(event))
    assert adapter._app.client.assistant_threads_setStatus.call_args.kwargs["thread_ts"] == MSG_TS


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda _source: adapter
    return runner


@pytest.mark.parametrize("channel,is_dm,extra", FLAT_TOP_LEVEL)
@pytest.mark.asyncio
async def test_runner_progress_restore_and_turn_end_stop_use_the_flat_anchor(channel, is_dm, extra):
    adapter = _adapter(**extra)
    event = await _inbound(adapter, channel=channel, is_dm=is_dm)
    runner = _runner(adapter)
    progress_md, progress_reply_to, status_md = runner._run_agent_progress_threading(
        event.source, _reply_anchor_for_event(event), False)
    for md in (progress_md, status_md):
        assert "thread_id" not in md and md[STATUS_ANCHOR_MESSAGE_ID] == MSG_TS
    await adapter.send_typing(channel, metadata=progress_md)  # _progress_restore_typing
    # recursive-turn typing restart: same target, same instant — throttled, so this
    # does not re-call setStatus (the sessions-API "processing" payload would be unchanged anyway).
    await adapter.send_typing(channel, metadata=status_md)
    assert _status_calls(adapter) == [(MSG_TS, "processing")]
    await adapter.send(channel, "working on it", reply_to=progress_reply_to, metadata=progress_md)
    assert adapter._app.client.chat_postMessage.call_args.kwargs.get("thread_ts") is None
    await runner._hmwa_stop_typing_for_turn(event, event.source)
    assert _status_calls(adapter)[-1] == (MSG_TS, "active")
    assert adapter._active_status_threads == {}
    adapter._app.client.assistant_threads_setStatus.assert_not_called()


# Modes where a later top-level message joins the running session (and so can be ``/queue``d).
FLAT_SHARED_SESSION = [p for p in FLAT_TOP_LEVEL if p.id != "dm-session-per-top-level-message"]


@pytest.mark.parametrize("channel,is_dm,extra", FLAT_SHARED_SESSION)
@pytest.mark.asyncio
async def test_queued_followup_flat_status_is_cleared_at_turn_end(channel, is_dm, extra):
    """A ``/queue`` follow-up's progress lane anchors on the queued message; the chain's single
    turn-end stop (on the opening message) must clear that status too, or it stays ``processing``."""
    adapter = _adapter(**extra)
    event = await _inbound(adapter, channel=channel, is_dm=is_dm)
    runner = _runner(adapter)
    await adapter.send_typing(channel, metadata=_owned_metadata(adapter, event))
    queued_progress_md, _, _ = runner._run_agent_progress_threading(event.source, QUEUED_TS, False)
    await adapter.send_typing(channel, metadata=queued_progress_md)  # _progress_restore_typing, queued turn
    assert (QUEUED_TS, "processing") in _status_calls(adapter)
    await runner._hmwa_stop_typing_for_turn(event, event.source)
    calls = _status_calls(adapter)
    assert (MSG_TS, "active") in calls and (QUEUED_TS, "active") in calls
    assert adapter._active_status_threads == {}
    adapter._app.client.assistant_threads_setStatus.assert_not_called()


# Two flat turns in one channel, in separate sessions; B's message is the LATER one.
CONCURRENT_SESSIONS = [
    pytest.param("C0CHAN", False, {}, "U2", id="channel-per-user-group-sessions"),
    pytest.param("D0DM", True, {}, "U1", id="dm-session-per-top-level-message"),
]


@pytest.mark.parametrize("channel,is_dm,extra,user_b", CONCURRENT_SESSIONS)
@pytest.mark.asyncio
async def test_turn_end_leaves_a_concurrent_sessions_flat_status_alone(channel, is_dm, extra, user_b):
    """Finishing turn A must not send ``active`` for turn B, still running in its own session."""
    adapter = _adapter(**extra)
    runner = _runner(adapter)
    event_a = await _inbound(adapter, channel=channel, is_dm=is_dm)
    event_b = await _inbound(adapter, channel=channel, is_dm=is_dm, user=user_b, ts=LATER_TS)
    assert adapter._event_session_key(event_a) != adapter._event_session_key(event_b)
    for event in (event_a, event_b):
        await adapter.send_typing(channel, metadata=_owned_metadata(adapter, event))
        progress_md, _, _ = runner._run_agent_progress_threading(
            event.source, _reply_anchor_for_event(event), False)
        await adapter.send_typing(channel, metadata=progress_md)
    await runner._hmwa_stop_typing_for_turn(event_a, event_a.source)
    assert (MSG_TS, "active") in _status_calls(adapter)
    assert (LATER_TS, "active") not in _status_calls(adapter)
    assert {key[2] for key in adapter._active_status_threads} == {LATER_TS}
    await runner._hmwa_stop_typing_for_turn(event_b, event_b.source)
    assert _status_calls(adapter)[-1] == (LATER_TS, "active")
    assert adapter._active_status_threads == {}
