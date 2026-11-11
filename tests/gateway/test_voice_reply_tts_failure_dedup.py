"""A ``text_to_speech`` call that produced no audio must not dedup the runner's voice reply (#151862).

``GatewayRunner._should_send_voice_reply`` skips the runner's own auto voice reply when the turn
already called the TTS tool, so the agent's audio is not spoken twice. The dedup read only the
CALL, never its result: a call that failed (provider length limit, a bad argument, a tool error)
left the chat with the text reply and no voice note. Only an explicit failure result exempts the
turn; a call whose result has not arrived yet keeps the historical dedup.
"""

import json

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

CHAT_ID = "123"


def _make_runner(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {"telegram:" + CHAT_ID: "all"}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    return runner


def _make_event(message_type=MessageType.TEXT):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT_ID,
                           user_id="user1", user_name="user")
    return MessageEvent(text="hi", message_type=message_type, source=source)


def _tts_call(call_id="call_1"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": "text_to_speech", "arguments": json.dumps({"text": "It passed."})}}]}


def _tool_result(call_id, payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _turn(*tail):
    return [{"role": "user", "content": "how did the deploy go?"}, *tail]


def test_failed_tts_call_still_sends_the_runner_voice_reply(tmp_path):
    runner = _make_runner(tmp_path)
    turn = _turn(
        _tts_call(),
        _tool_result("call_1", {"success": False, "error": "text length 12000 exceeds 10000"}),
        {"role": "assistant", "content": "It passed."},
    )
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is True


def test_tool_error_shaped_failure_still_sends_the_runner_voice_reply(tmp_path):
    """``tool_error`` writes ``{"error": ...}`` and no ``success`` key."""
    runner = _make_runner(tmp_path)
    turn = _turn(
        _tts_call(),
        _tool_result("call_1", {"error": "ELEVENLABS_API_KEY not set."}),
        {"role": "assistant", "content": "It passed."},
    )
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is True


def test_successful_tts_call_keeps_the_dedup(tmp_path):
    runner = _make_runner(tmp_path)
    turn = _turn(
        _tts_call(),
        _tool_result("call_1", {"success": True, "file_path": "/tmp/a.ogg", "provider": "edge"}),
        {"role": "assistant", "content": "It passed."},
    )
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is False


def test_one_successful_tts_call_dedups_after_another_failed(tmp_path):
    runner = _make_runner(tmp_path)
    turn = _turn(
        _tts_call("call_1"), _tool_result("call_1", {"success": False, "error": "boom"}),
        _tts_call("call_2"), _tool_result("call_2", {"success": True, "file_path": "/tmp/b.ogg"}),
        {"role": "assistant", "content": "It passed."},
    )
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is False


def test_tts_call_without_a_result_keeps_the_dedup(tmp_path):
    """A turn still in flight (the call has not returned) keeps the historical dedup."""
    runner = _make_runner(tmp_path)
    turn = _turn(_tts_call(), {"role": "assistant", "content": ""})
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is False


def test_unparseable_tts_result_keeps_the_dedup(tmp_path):
    """A body we cannot read as a failure is treated as audio, not as a gap to fill."""
    runner = _make_runner(tmp_path)
    turn = _turn(_tts_call(), _tool_result("call_1", "ok"),
                 {"role": "assistant", "content": "It passed."})
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is False


def _bridged_tts_call(call_id="call_1"):
    """A ``text_to_speech`` call made through the ``tool_call`` bridge (deferred tools).

    The assistant message records the bridge name, so a lookup keyed on the producer's own
    name never matches its result — ``gateway.media_repair.tool_name_by_call_id`` resolves it.
    """
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": "tool_call",
                      "arguments": json.dumps({"name": "text_to_speech",
                                               "arguments": {"text": "It passed."}})}}]}


def test_bridged_tts_call_with_a_failed_result_still_sends_the_runner_voice_reply(tmp_path):
    runner = _make_runner(tmp_path)
    turn = _turn(
        _bridged_tts_call(),
        _tool_result("call_1", {"success": False, "error": "text length 12000 exceeds 10000"}),
        {"role": "assistant", "content": "It passed."},
    )
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is True


def test_bridged_tts_call_with_audio_keeps_the_dedup(tmp_path):
    runner = _make_runner(tmp_path)
    turn = _turn(
        _bridged_tts_call(),
        _tool_result("call_1", {"success": True, "file_path": "/tmp/a.ogg"}),
        {"role": "assistant", "content": "It passed."},
    )
    assert runner._should_send_voice_reply(_make_event(), "It passed.", turn) is False
