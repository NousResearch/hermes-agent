"""Late unconsumed completions must permit explicit silence through Codex recovery.

The LLM transport below is a deterministic instruction follower, not a substitute
for real-model acceptance.
Notification admission, preprocessing, conversation recovery and delivery are real.
"""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext
from tools.process_registry_notifications import format_process_notification


class RecordingTelegram(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def stop_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def codex_response(text=None):
    if text is None:
        output = [SimpleNamespace(
            type="reasoning", id="rs_late", encrypted_content="opaque-test-state",
            summary=[SimpleNamespace(type="summary_text", text="This result was already reported.")],
            status="completed",
        )]
    else:
        output = [SimpleNamespace(
            type="message", content=[SimpleNamespace(type="output_text", text=text)],
        )]
    return SimpleNamespace(
        output=output, status="completed", model="gpt-5-codex",
        usage=SimpleNamespace(input_tokens=50, output_tokens=10, total_tokens=60),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count,new_failure", [(1, False), (3, False), (3, True)])
async def test_late_completion_recovery_and_delivery(monkeypatch, tmp_path, count, new_failure):
    import gateway.run as gateway_run
    from run_agent import AIAgent

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0.0)
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingTelegram()
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_message_handler(runner._handle_message)
    source = adapter.build_source("123", user_id="123")
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    agent = AIAgent(
        model="gpt-5-codex", api_mode="codex_responses", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_key="test-token",
        quiet_mode=True, max_iterations=4, skip_context_files=True, skip_memory=True,
    )
    history = [
        {"role": "user", "content": "Report the build and test results."},
        {"role": "assistant", "content": "Build and tests succeeded. I read the artifacts; work is complete."},
    ]
    entries = []
    for index in range(count):
        failure = new_failure and index == count - 1
        evt = dict(
            type="completion", session_id=f"proc_late_{index}", started_at=100 + index,
            session_key=key, platform="telegram", chat_id="123", chat_type="dm", user_id="123",
            exit_code=7 if failure else 0,
            output="NEW FAILURE: deployment verification failed" if failure else "Build and tests succeeded",
        )
        entries.append((format_process_notification(evt), evt))

    # Admit through the actual internal-event route, then recover the queued unit.
    # No consumed mark: the prior report came from artifacts, not registry wait/log.
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    try:
        text = entries[0][0] if count == 1 else runner._format_coalesced_process_completions(
            [(text, evt, None) for text, evt in entries]
        )
        envelope = dict(entries[0][1], process_completion_entries=entries)
        assert await runner._inject_watch_notification(text, envelope) is True
        event = adapter.get_pending_message(key)
        assert event is not None
        assert event.internal is True
        assert len(event.metadata["process_completion_entries"]) == count
    finally:
        adapter._session_tasks.pop(key, None)
        adapter._active_sessions.pop(key, None)

    requests = []
    human_turn = False
    consumer = None
    stream_task = None
    if count == 3 and not new_failure:
        consumer = GatewayStreamConsumer(
            adapter, source.chat_id, StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        stream_task = asyncio.create_task(consumer.run())

    def transport(api_kwargs):
        requests.append(copy.deepcopy(api_kwargs))
        if len(requests) <= 2:
            return codex_response()
        # The final user item is the real recovery nudge, after two incomplete
        # responses. Test the instruction the model receives, not source spelling.
        wire = api_kwargs.get("messages", api_kwargs.get("input", []))
        users = [item for item in wire if item.get("role") == "user"]
        nudge = json.dumps(users[-1].get("content", ""))
        if human_turn:
            return codex_response("NO_REPLY" if "NO_REPLY" in nudge else "Here is your requested update.")
        if new_failure:
            return codex_response("Deployment verification failed (exit 7); this is a new failure.")
        final = "NO_REPLY" if "NO_REPLY" in nudge else "The result was already accounted for."
        if consumer:
            for chunk in (final[:2], final[2:5], final[5:]):
                consumer.on_delta(chunk)
        return codex_response(final)

    monkeypatch.setattr(agent, "_interruptible_api_call", transport)
    queued_contexts = []

    async def run_queued_agent(message, context_prompt, history, source, session_id, **kwargs):
        # Replace orchestration only; keep the production recursive caller, its
        # provenance kwargs, actual AIAgent facade/recovery, and send boundary.
        kwargs.pop("message_type", None)  # Used by outer voice setup, not TurnContext.
        queued_ctx = TurnContext(
            message=message, context_prompt=context_prompt, history=history,
            source=source, session_id=session_id, **kwargs,
        )
        queued_contexts.append(queued_ctx)
        queued_runner = TurnRunner(runner, queued_ctx)
        return await asyncio.to_thread(
            queued_runner._run_conversation_with_approval, agent, history, [], None, None,
        )

    monkeypatch.setattr(runner, "_run_agent", run_queued_agent)
    ctx = TurnContext(
        source=source, session_key=key, session_id="late-completion-test", history=history, context_prompt="",
    )
    prior = {"final_response": "NO_REPLY", "messages": history}
    result = await runner._run_agent_queued_followup(
        ctx, adapter, event.text, event, prior, prior, None,
    )
    ctx = queued_contexts[-1]
    turn = TurnRunner(runner, ctx)
    assert result["completed"] is True
    assert len(requests) == 3, "Recovery must stay bounded and actually execute"
    assert any(
        row.get("role") == "assistant" and row.get("finish_reason") == "incomplete"
        for row in result["messages"]
    ), "Exercise real reasoning-only recovery before delivery"
    if consumer:
        turn._finish_stream_consumer(result, history, consumer)
        await asyncio.wait_for(stream_task, 10)
        assert adapter.sent == [], "Even a partial NO_REPLY marker must never reach Telegram"
    await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
    assert len(adapter.sent) == int(new_failure), adapter.sent
    if new_failure:
        assert "new failure" in adapter.sent[0]
    else:
        assert result["final_response"] == "NO_REPLY"
        nudge = [row for row in result["messages"] if row.get("role") == "user"][-1]
        assert "NO_REPLY" in nudge["content"]

    # Cached AIAgent reuse: a queued human can copy both notification text and its
    # public metadata. Its own provenance must win over the preceding silent turn.
    assert not agent._completion_silence_allowed
    human_turn = True
    first_requests = list(requests)
    requests.clear()
    human = MessageEvent(text=event.text, source=source, metadata=dict(event.metadata))
    human.metadata["completion_silence_allowed"] = True
    assert runner._completion_silence_metadata(human) == {}
    # The prior result has already crossed its delivery boundary above.
    delivered = dict(result, final_response="NO_REPLY")
    adapter._pending_messages[key] = human
    pending_event, pending = await runner._run_agent_drain_pending(delivered, adapter, source, key)
    assert pending_event is human
    followup = await runner._run_agent_queued_followup(
        ctx, adapter, pending, pending_event, delivered, delivered, None,
    )
    assert followup["completed"] is True
    assert len(requests) == 3
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE
    assert [row for row in followup["messages"] if row.get("role") == "user"][-1]["content"] == _CODEX_INCOMPLETE_NUDGE
    await runner._run_agent_deliver_first_response(queued_contexts[-1], adapter, followup, followup, None)
    assert adapter.sent == (["Deployment verification failed (exit 7); this is a new failure."] if new_failure else []) + [
        "Here is your requested update."
    ]
    assert not agent._completion_silence_allowed
    # The system prompt remains byte-stable across notification recovery and human reuse.
    def system_input(request):
        return request.get("instructions") or [
            row for row in request.get("messages", []) if row.get("role") == "system"
        ]
    assert system_input(first_requests[0])
    assert all(system_input(request) == system_input(first_requests[0]) for request in first_requests + requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "media", "command", "consumed_sibling"])
async def test_completion_permission_tracks_admitted_input(monkeypatch, tmp_path, kind):
    """A retained internal flag cannot authorize merged user content as a completion."""
    import gateway.run as gateway_run
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageType

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingTelegram()
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.set_message_handler(runner._handle_message)
    source = adapter.build_source("123", user_id="123")
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    entries = [("Old result", dict(type="completion", session_id="old")),
               ("Other result", dict(type="completion", session_id="other"))]
    text = runner._format_coalesced_process_completions([(text, evt, None) for text, evt in entries])
    try:
        assert await runner._inject_watch_notification(text, dict(
            type="completion", session_key=key, platform="telegram", chat_id="123", chat_type="dm",
            process_completion_entries=entries,
        )) is True
        event = adapter.get_pending_message(key)
        assert event is not None
    finally:
        adapter._session_tasks.pop(key, None)
        adapter._active_sessions.pop(key, None)
    assert runner._completion_silence_metadata(event) == {"completion_silence_allowed": True}
    if kind == "consumed_sibling":
        monkeypatch.setattr(runner, "_process_completion_consumed", lambda evt: evt["session_id"] == "old")
        assert runner._refresh_process_completion_event(event)
        assert event.text == "Other result"
        assert runner._completion_silence_metadata(event) == {"completion_silence_allowed": True}
    else:
        human = MessageEvent(
            text="/status" if kind == "command" else "Please explain the result", source=source,
            message_type=MessageType.PHOTO if kind == "media" else MessageType.TEXT,
            media_urls=["/tmp/photo.png"] if kind == "media" else [],
        )
        slot = {key: event}
        merge_pending_message_event(slot, key, human, merge_text=True)
        assert human.text in slot[key].text
        assert runner._completion_silence_metadata(slot[key]) == {}
        assert slot[key].internal is False
        monkeypatch.setattr(runner, "_process_completion_consumed", lambda evt: True)
        assert runner._refresh_process_completion_event(slot[key])
        assert human.text in slot[key].text

@pytest.mark.parametrize("kind", ["codex", "post_tool"])
def test_completion_recovery_is_synthetic_after_metadata_loss(kind):
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE, _EMPTY_TOOL_RESPONSE_NUDGE
    from agent.context_compressor import ContextCompressor
    from agent.turn_facade import completion_recovery_nudge

    default = _CODEX_INCOMPLETE_NUDGE if kind == "codex" else _EMPTY_TOOL_RESPONSE_NUDGE
    scoped = completion_recovery_nudge(SimpleNamespace(_completion_silence_allowed=True), default)
    row = {"role": "user", "content": scoped}  # persisted history has no transient metadata
    assert ContextCompressor._is_synthetic_compression_user_turn(row)
    assert not ContextCompressor._transcript_has_real_user_turn([row])
    assert ContextCompressor._derive_auto_focus_topic([row]) is None
    assert completion_recovery_nudge(SimpleNamespace(), default) == default
    assert ContextCompressor._transcript_has_real_user_turn([
        row, {"role": "user", "content": "Please explain the result"},
    ])


def test_completion_permission_is_restored_after_exception(monkeypatch):
    from run_agent import AIAgent
    import agent.conversation_loop as loop

    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    agent = AIAgent(model="gpt-5-codex", api_mode="codex_responses", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_key="test-token",
        quiet_mode=True, skip_context_files=True, skip_memory=True)
    def fail(*args, **kwargs):
        assert agent._completion_silence_allowed is True
        raise RuntimeError("injected failure before model request")
    monkeypatch.setattr(loop, "run_conversation", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        agent.run_conversation("Completion", persist_user_display_metadata={"completion_silence_allowed": True})
    assert agent._completion_silence_allowed is False
