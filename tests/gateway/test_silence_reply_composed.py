"""Real provider-scripted AIAgent through TurnRunner and gateway response shaping.

External provider and platform IO are doubles; the agent loop, tool execution,
result finalizer, TurnRunner and final warning/deduplication decision are real.
"""
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
import hermes_yaml as yaml
import run_agent
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.agent import test_empty_response_recovery_persistence as recovery
from tests.gateway.test_gateway_silence_tokens import _runner
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter

real_loop = recovery.real_loop
ANSWER = "Your meeting is at 3pm."
WARNING = "The model returned only a silence marker"


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.WHATSAPP])
@pytest.mark.parametrize("interim", [False, True])
@pytest.mark.parametrize("housekeeping", [False, True], ids=["marker-only", "written-reply"])
async def test_real_loop_silence_delivery(monkeypatch, tmp_path, real_loop, platform, interim, housekeeping):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "display": {"interim_assistant_messages": interim, "tool_progress": "off"},
    }))
    agent = real_loop.agent
    agent.valid_tool_names.add("todo_list")
    agent.tools.append({"type": "function", "function": {"name": "todo_list", "parameters": {}}})
    script = []
    if housekeeping:
        script.append(recovery._response(ANSWER, finish_reason="tool_calls", tool_calls=[
            recovery._call("todo_list", {"todos": []}, "call_todo"),
        ]))
    script.append(recovery._response("NO_REPLY"))
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = lambda **_kw: script.pop(0)

    # Supply the real, isolated agent to the runtime constructor seam. Do not
    # mock run_conversation, _run_agent, finalization or gateway shaping.
    class IsolatedAgent(run_agent.AIAgent):
        def __new__(cls, **kwargs):
            for key, value in kwargs.items():
                if key.endswith("callback"):
                    setattr(agent, key, value)
            return agent

    monkeypatch.setattr(run_agent, "AIAgent", IsolatedAgent)
    adapter = ProgressCaptureAdapter(platform=platform)
    runner = _runner(monkeypatch, tmp_path)
    runner.adapters = {platform: adapter}
    runner.session_store.get_model_override.return_value = None
    source = SessionSource(platform=platform, chat_id="dm-composed", chat_type="dm", user_id="human")
    key = f"agent:main:{platform.value}:dm:dm-composed"
    entry = runner.session_store.get_or_create_session.return_value
    entry.session_key, entry.session_id, entry.platform, entry.chat_type = key, real_loop.sid, platform, "dm"
    event = MessageEvent(text="when is my meeting?", source=source, message_id="m1", reply_expected=True)
    reply = await runner._handle_message_with_agent(event, source, key, 1)
    sent = [call["content"] for call in adapter.sent]
    if housekeeping:
        assert not any(WARNING in text for text in sent + [reply or ""])
        assert sent.count(ANSWER) + int(reply == ANSWER) == 1
    else:
        assert WARNING in (reply or "")
    assert not script, "the real agent must consume the entire scripted provider sequence"
