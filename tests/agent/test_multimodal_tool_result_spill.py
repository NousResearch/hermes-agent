"""Oversized TEXT parts inside a multimodal tool envelope must go through the same persistence
policy as string results (#95429): a ``browser_exec`` call that captured a screenshot bakes its
whole stdout into the envelope's text part, and that used to bypass ``maybe_persist_tool_result``
and ride every later provider request inline (760K chars -> multi-megabyte requests)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.tool_result_storage import PERSISTED_OUTPUT_TAG
from tests.agent.test_tool_call_incremental_persistence import _make_agent, _mock_tool_call


def _run_sequential(agent, function_result):
    tool_calls = [_mock_tool_call(name="browser_exec", call_id="call_browser")]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    agent._flush_messages_to_session_db = MagicMock()
    # Vision-capable route (the reporter's case): the envelope stays a part LIST in history, which
    # the per-turn aggregate budget cannot measure — so only per-part persistence bounds it.
    with (patch("model_tools.handle_function_call", return_value=function_result),
          patch.object(type(agent), "_model_supports_vision", lambda self: True),
          patch.object(type(agent), "_provider_supports_vision_tool_messages", lambda self: True)):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")
    return [m for m in messages if m.get("role") == "tool"]


def _envelope(text: str) -> dict:
    # Real browser_exec shape: the content text carries an extra screenshot note the summary lacks.
    return {"_multimodal": True, "text_summary": text, "meta": {"screenshot_path": "/tmp/shot.png"},
            "content": [{"type": "text", "text": text + "\nscreenshot captured"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]}


def test_oversized_multimodal_text_part_is_spilled_and_recoverable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path, raising=False)
    big = "x" * 760_396
    agent = _make_agent()
    (tool_msg,) = _run_sequential(agent, _envelope(big))
    content = tool_msg["content"]
    assert isinstance(content, list)
    texts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
    assert len(texts) == 1 and PERSISTED_OUTPUT_TAG in texts[0] and len(texts[0]) < 10_000
    assert any(p.get("type") == "image_url" for p in content)  # image part untouched
    # One spill file holding the full content text (not overwritten by a second text_summary write),
    # and the duplicate-result stub guard knows where it lives.
    (spilled,) = Path(tmp_path, "cache", "spillover").glob("*")
    assert spilled.read_text() == big + "\nscreenshot captured"
    assert agent._tool_guardrails._persisted_result_paths == {"call_browser": str(spilled)}


def test_normal_multimodal_result_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    small = "snapshot ok"
    (tool_msg,) = _run_sequential(_make_agent(), _envelope(small))
    assert tool_msg["content"][0] == {"type": "text", "text": small + "\nscreenshot captured"}
    assert not Path(tmp_path, "cache", "spillover").exists()
