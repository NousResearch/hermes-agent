"""Runtime notices never reach a bot peer (#137027): the busy path, the failure copy, progress lanes.

Each of these is a message the runtime wrote, not the model; delivered to another bot it is a new
inbound turn there, which answers it, which is a new turn here. The model's own reply still goes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.platforms.base import MessageEvent
from hermes_agent.gateway.run_turn import GatewayTurnMixin
from hermes_agent.gateway.session import Platform, SessionSource


def _source(is_bot):
    return SessionSource(platform=Platform.MATTERMOST, chat_id="dm", chat_type="dm", user_id="peer", is_bot=is_bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot, sent", [(True, False), (False, True)], ids=["bot peer", "human control"])
async def test_every_busy_path_reply_skips_a_bot_peer(is_bot, sent):
    """The drain notice, skill reply and approval echo share ``_send_busy_reply`` with the ack."""
    from hermes_agent.gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._reply_anchor_for_event = lambda event: None
    runner._thread_metadata_for_source = lambda source, anchor=None: None
    adapter = SimpleNamespace(_send_with_retry=AsyncMock())
    event = MessageEvent(text="x", message_id="m", source=_source(is_bot))

    await GatewayRunner._send_busy_reply(runner, event, adapter, "Gateway is restarting; your message is queued.")

    assert adapter._send_with_retry.await_count == (1 if sent else 0)


class _Runner(GatewayTurnMixin):
    def _silence_allowed_for_source(self, source, display_kind, reply_expected, agent_result):
        from hermes_agent.gateway.response_filters import silence_allowed
        return silence_allowed(display_kind, reply_expected)

    def __init__(self):
        self.async_session_store = SimpleNamespace(clear_resume_pending=self._noop)

    async def _noop(self, *_a, **_k):
        return None

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None


async def _shape(agent_result, source):
    return await _Runner()._hmwa_shape_agent_response(
        agent_result, source, history=[], session_entry=SimpleNamespace(session_id="s"), session_key=None,
        _quick_key=None, run_generation=0, _run_start_session_id="s", _platform_name="mattermost",
        _msg_start_time=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_result", [
    {"final_response": "", "interrupted": True, "api_calls": 0, "messages": []},
    {"final_response": "(empty)", "model": "m", "api_calls": 2, "messages": []},
    {"final_response": "", "failed": True, "error": "boom", "api_calls": 1, "messages": []},
], ids=["interrupted before start", "empty after retries", "failed turn"])
async def test_runtime_failure_copy_is_not_the_reply_to_a_bot(agent_result):
    response, _silent, _messages = await _shape(dict(agent_result), _source(True))
    assert response == ""
    human, _silent, _messages = await _shape(dict(agent_result), _source(False))
    assert human  # the human control still gets the explanation


@pytest.mark.asyncio
async def test_the_models_reply_to_a_bot_is_still_delivered_and_its_silence_is_honoured():
    response, silent, _ = await _shape({"final_response": "4", "api_calls": 1, "messages": []}, _source(True))
    assert (response, silent) == ("4", False)
    response, silent, _ = await _shape({"final_response": "[SILENT]", "api_calls": 1, "messages": []}, _source(True))
    assert silent is True and response != _unexpected()


def _unexpected():
    from hermes_agent.gateway.run_turn import _unexpected_silence_reply
    return _unexpected_silence_reply()


def test_status_and_notice_callbacks_stay_quiet_for_a_bot_peer():
    from hermes_agent.gateway.run_turn_runner import TurnRunner

    turn_runner = object.__new__(TurnRunner)
    turn_runner._schedule = MagicMock()
    turn_runner._ctx = SimpleNamespace(
        mute_notification_reply=False, source=_source(True), _status_adapter=object(),
        _run_still_current=lambda: True, user_config={}, _status_chat_id="dm",
        _status_thread_metadata=None, _cleanup_progress=False)

    turn_runner._status_callback_sync("lifecycle", "⚠️ Model fallback: primary unavailable")
    turn_runner._notice_callback_sync(SimpleNamespace(key="credits.low", text="Credits are low", level="warn"))

    turn_runner._schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_expected", [True, None], ids=["addressed (DM / mention)", "unaddressed"])
async def test_an_addressed_bot_may_still_end_the_exchange_in_silence(reply_expected):
    """Greptile 4467193953: Telegram/Slack set reply_expected=True for a DM or mention, bots included;
    the silence fallback would then be runtime text the peer answers."""
    runner = _Runner()
    response, silent, _ = await runner._hmwa_shape_agent_response(
        {"final_response": "[SILENT]", "api_calls": 1, "messages": []}, _source(True), history=[],
        session_entry=SimpleNamespace(session_id="s"), session_key=None, _quick_key=None, run_generation=0,
        _run_start_session_id="s", _platform_name="telegram", _msg_start_time=0.0, reply_expected=reply_expected)
    assert silent is True and response != _unexpected()
    human, silent, _ = await runner._hmwa_shape_agent_response(
        {"final_response": "[SILENT]", "api_calls": 1, "messages": []}, _source(False), history=[],
        session_entry=SimpleNamespace(session_id="s"), session_key=None, _quick_key=None, run_generation=0,
        _run_start_session_id="s", _platform_name="telegram", _msg_start_time=0.0, reply_expected=True)
    assert silent is False and human == _unexpected()  # the human control keeps the visible fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot, sent", [(True, 0), (False, 1)], ids=["bot peer", "human control"])
async def test_platform_notices_and_the_inactivity_warning_skip_a_bot_peer(is_bot, sent):
    """Greptile 4467193888: the delegation-failure notice (``_deliver_platform_notice``) and the
    inactivity warning sent straight to the adapter both go through the one bot-peer policy."""
    from hermes_agent.gateway.run import GatewayRunner

    adapter = SimpleNamespace(send=AsyncMock(), emit_warning=AsyncMock(), config=None)
    runner = object.__new__(GatewayRunner)
    runner.config = None
    runner._delivery_adapter_for = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: None
    source = _source(is_bot)

    await GatewayRunner._deliver_platform_notice(runner, source, "⚠️ Subagent failed: timeout")
    worker = SimpleNamespace(agent_warning=600, agent_timeout=1200)
    await GatewayRunner._run_agent_inactivity_warning(runner, worker, source, None)

    assert adapter.send.await_count == sent and adapter.emit_warning.await_count == sent



@pytest.mark.parametrize("is_bot", [True, False], ids=["bot peer", "human control"])
def test_the_runtime_footer_is_not_sent_to_a_bot_peer(monkeypatch, is_bot):
    import hermes_agent.gateway.runtime_footer as footer

    monkeypatch.setattr(footer, "build_footer_line", lambda **kw: "— model · 3s")
    line = _Runner()._hmwa_runtime_footer_line({"model": "m"}, _source(is_bot), 3.0)
    assert line == ("" if is_bot else "— model · 3s")
