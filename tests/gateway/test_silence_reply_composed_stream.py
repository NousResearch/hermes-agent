"""Streamed variant: a real AIAgent streams through TurnRunner into the real stream consumer.

The provider streams the reply beside a housekeeping tool, then a bare marker. Only the
provider and the platform IO are doubles; the agent loop, tool execution, finalizer,
TurnRunner, stream consumer and the final send/dedupe decision are real.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_yaml as yaml
import run_agent
from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.agent import test_empty_response_recovery_persistence as recovery
from tests.gateway.test_gateway_silence_tokens import _runner
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter

real_loop = recovery.real_loop
ANSWER = "Your meeting is at 3pm."
WARNING = "The model returned only a silence marker"


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    return SimpleNamespace(model="test/model", usage=None,
                           choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)])


def _tool_delta(call_id, name, arguments):
    return SimpleNamespace(index=0, id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


class _Stream:
    """Token stream stand-in; no ``choices`` attribute, so it is read as a stream."""

    response = None

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


class EditingAdapter(ProgressCaptureAdapter):
    """Edit-capable adapter: each send is a new message id, deletes retract it."""

    def __init__(self, platform):
        super().__init__(platform=platform)
        self.deleted = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id=f"m{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        self.deleted.append(message_id)
        return True

    def visible(self):
        shown = {f"m{i}": call["content"] for i, call in enumerate(self.sent, 1)}
        for edit in self.edits:
            shown[edit["message_id"]] = edit["content"]
        for message_id in self.deleted:
            shown.pop(message_id, None)
        return list(shown.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.WHATSAPP])
@pytest.mark.parametrize("transform", [False, True])
async def test_streamed_reply_then_marker_shows_the_reply_once(monkeypatch, tmp_path, real_loop, platform, transform):
    raw = "Answer PRIVATE_VALUE" if transform else ANSWER
    expected = "Answer [REDACTED]." if transform else ANSWER

    def output_hook(name, **kwargs):
        if transform and name == "transform_llm_output" and kwargs["response_text"] == raw:
            return [expected]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", output_hook)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "display": {"interim_assistant_messages": False, "tool_progress": "off"},
    }))
    agent = real_loop.agent
    agent.valid_tool_names.add("todo_list")
    agent.tools.append({"type": "function", "function": {"name": "todo_list", "parameters": {}}})
    script = [
        [_chunk(content=raw), _chunk(tool_calls=[_tool_delta("call_todo", "todo_list", '{"todos": []}')]),
         _chunk(finish_reason="tool_calls")],
        [_chunk(content="NO_REPLY"), _chunk(finish_reason="stop")],
    ]
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = (
        lambda **kw: _Stream(script.pop(0)) if kw.get("stream") else pytest.fail("expected a streamed request")
    )

    class IsolatedAgent(run_agent.AIAgent):
        def __new__(cls, **kwargs):
            for key, value in kwargs.items():
                if key.endswith("callback"):
                    setattr(agent, key, value)
            return agent

    monkeypatch.setattr(run_agent, "AIAgent", IsolatedAgent)
    adapter = EditingAdapter(platform)
    runner = _runner(monkeypatch, tmp_path)
    runner._hmwa_first_contact_notes = AsyncMock()
    runner.config.streaming = StreamingConfig(enabled=True, transport="edit", edit_interval=0.01,
                                              buffer_threshold=1, cursor="")
    runner.adapters = {platform: adapter}
    runner.session_store.get_model_override.return_value = None
    source = SessionSource(platform=platform, chat_id="dm-composed", chat_type="dm", user_id="human")
    key = f"agent:main:{platform.value}:dm:dm-composed"
    entry = runner.session_store.get_or_create_session.return_value
    entry.session_key, entry.session_id, entry.platform, entry.chat_type = key, real_loop.sid, platform, "dm"
    event = MessageEvent(text="when is my meeting?", source=source, message_id="m0", reply_expected=True)

    reply = await runner._handle_message_with_agent(event, source, key, 1)

    assert not script, "the real agent must consume the whole streamed provider script"
    shown = adapter.visible() + ([reply] if reply else [])
    assert shown == [expected], (adapter.sent, adapter.edits, adapter.deleted, reply)
    history = real_loop.db.get_messages_as_conversation(real_loop.sid)
    assert history[-1]["content"] == expected
    assert not any(WARNING in text or "NO_REPLY" in text for text in shown)
