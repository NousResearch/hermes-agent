"""A topic policy stop crosses turn boundaries as failure, never as delivered content."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.base_thread_metadata import _thread_metadata_for_event
from gateway.platforms.event import MessageEvent
from gateway.run_turn_runner import TurnRunner, _ExecApprovalDeclined
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway.test_decline_fallback_suppression import APPROVAL, _Adapter, _runner
from tests.gateway.test_run_progress_topics import FakeAgent, ProgressCaptureAdapter, _make_runner, _run_with_agent


def _terminal():
    return SendResult(success=False, error="topic policy suppressed original", retry_suppressed=True)


def _source(platform=Platform.FEISHU):
    return SessionSource(platform=platform, chat_id="oc_group", chat_type="group", thread_id="omt_topic")


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.FEISHU, Platform.TELEGRAM])
async def test_event_state_reaches_progress_status_and_stream_metadata(platform):
    adapter = ProgressCaptureAdapter(platform=platform)
    runner = _make_runner(adapter)
    runner.config.streaming = StreamingConfig(enabled=True)
    source = _source(platform)
    event = MessageEvent(text="question", source=source, message_id="om_question")
    metadata = runner._event_thread_metadata(event, source)
    ctx = TurnContext(
        source=source, user_config={}, resolve_display_setting=lambda *args: True,
        _run_still_current=lambda: True,
    )
    worker = TurnRunner(runner, ctx)

    runner._run_agent_bind_turn_wiring(ctx, worker, source, "om_question", False, metadata)
    worker._setup_stream_consumer(platform.value)
    consumer = ctx.stream_consumer_holder[0]
    assert consumer is not None

    if platform == Platform.FEISHU:
        state = _thread_metadata_for_event(event)["_feishu_topic_delivery"]
        assert ctx._progress_metadata["_feishu_topic_delivery"] is state
        assert ctx._status_thread_metadata["_feishu_topic_delivery"] is state
        assert consumer.metadata["_feishu_topic_delivery"] is state
        another = MessageEvent(text="next", source=source, message_id="om_next")
        assert runner._event_thread_metadata(another, source)["_feishu_topic_delivery"] is not state
    else:
        assert "_feishu_topic_delivery" not in ctx._progress_metadata
        assert "_feishu_topic_delivery" not in ctx._status_thread_metadata
        assert "_feishu_topic_delivery" not in consumer.metadata


class _PolicyProgressAdapter(ProgressCaptureAdapter):
    terminal = True

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        result = _terminal() if self.terminal else SendResult(success=False, error="temporary", retryable=True)
        if self.terminal:
            metadata["_feishu_topic_delivery"]["terminal"] = result
        return result


class _TransientProgressAdapter(_PolicyProgressAdapter):
    terminal = False


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls,suppressed", [(_PolicyProgressAdapter, True), (_TransientProgressAdapter, False)])
async def test_real_agent_turn_only_propagates_explicit_progress_policy_stop(monkeypatch, tmp_path, adapter_cls, suppressed):
    adapter, response = await _run_with_agent(
        monkeypatch, tmp_path, FakeAgent, session_id="feishu-policy",
        platform=Platform.FEISHU, adapter_cls=adapter_cls,
        config_data={"display": {"tool_progress": "all"}},
    )
    assert adapter.sent
    assert response["final_response"] == "done"
    assert bool(response.get("delivery_retry_suppressed")) is suppressed
    assert not response.get("already_sent")
    if suppressed:
        assert response["_delivery_retry_suppressed_result"] is adapter.sent[0]["metadata"]["_feishu_topic_delivery"]["terminal"]
        assert len(adapter.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_stream_terminal_suppresses_final_even_when_agent_failed(failed):
    runner = _make_runner(ProgressCaptureAdapter(platform=Platform.FEISHU))
    terminal = _terminal()
    consumer = SimpleNamespace(retry_suppressed_result=terminal)
    ctx = TurnContext(source=_source(), stream_consumer_holder=[consumer])
    response = {"final_response": "original", "failed": failed}

    await runner._run_agent_mark_streamed_delivery(response, ctx)

    assert response["delivery_retry_suppressed"] is True
    assert response["_delivery_retry_suppressed_result"] is terminal
    assert not response.get("already_sent")


@pytest.mark.asyncio
async def test_completion_preserves_original_for_ledger_but_skips_voice_media_and_footer():
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    runner._send_voice_reply = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    source = _source()
    event = MessageEvent(text="question", source=source, message_id="om_question")
    terminal = _terminal()
    runner._event_thread_metadata(event, source)["_feishu_topic_delivery"]["terminal"] = terminal
    response = {"final_response": "original MEDIA: /tmp/private.png"}

    content = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="s1"), "sk1", 1,
        response, [], response["final_response"], "footer", False,
    )

    assert content == response["final_response"]
    assert event._delivery_retry_suppressed_result is terminal
    assert not response.get("already_sent")
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()
    assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery,delivered,suppressed", [(True, True, False), (False, False, False), (_terminal(), False, True)])
async def test_queued_return_distinguishes_terminal_failure_from_delivered(delivery, delivered, suppressed):
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    runner._deliver_queued_first_response = AsyncMock(return_value=delivery)
    response = {"final_response": "original", "messages": []}
    result = dict(response)
    ctx = TurnContext(source=_source(), session_key="sk1")

    await runner._run_agent_deliver_first_response(ctx, adapter, response, result, None)

    assert bool(result.get("already_sent")) is delivered
    assert bool(result.get("delivery_retry_suppressed")) is suppressed
    assert bool(response.get("delivery_retry_suppressed")) is suppressed
    if suppressed:
        assert result["_delivery_retry_suppressed_result"] is delivery
        assert not result.get("media_already_delivered")


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [True, False])
async def test_progress_edit_stops_policy_fallback_but_keeps_transient_retry(terminal):
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    adapter.edit_message = AsyncMock(return_value=_terminal() if terminal else SendResult(success=False, error="timeout", retryable=True))
    ctx = TurnContext(source=_source(), _progress_metadata={"thread_id": "omt_topic"})
    worker = TurnRunner(_make_runner(adapter), ctx)
    state = worker._progress_edit_state(adapter)
    state.progress_msg_id = "om_existing"
    state.progress_lines = ["progress"]

    finished = await worker._progress_send_or_edit(state, "progress")
    await worker._flush_progress_edit(state)

    assert finished is terminal
    assert adapter.sent == []
    assert state.can_edit is True
    assert adapter.edit_message.await_count == (1 if terminal else 2)


def test_native_approval_policy_stop_unblocks_without_plaintext_retry():
    adapter = _Adapter(_terminal())
    worker = _runner(adapter)
    with pytest.raises(_ExecApprovalDeclined):
        worker._approval_notify_sync(dict(APPROVAL))
    assert adapter.text_sends == []


def test_text_approval_policy_stop_does_not_register_timeout_notice(monkeypatch):
    class TextAdapter:
        typed_command_prefix = "/"
        pause_typing_for_chat = lambda self, chat_id: None
        send = AsyncMock(return_value=_terminal())

    register = SimpleNamespace(called=False)
    def timeout_notice(*args, **kwargs):
        register.called = True
    monkeypatch.setattr("gateway.run_turn_runner_approval_settle.register_timeout_notice", timeout_notice)
    worker = _runner(TextAdapter())
    with pytest.raises(_ExecApprovalDeclined):
        worker._approval_notify_sync(dict(APPROVAL))
    assert not register.called


class _FirstTurnPolicyAdapter(ProgressCaptureAdapter):
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        delivered = await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        if len(self.sent) == 1:
            terminal = _terminal()
            metadata["_feishu_topic_delivery"]["terminal"] = terminal
            return terminal
        return delivered


@pytest.mark.asyncio
async def test_real_queued_followup_gets_fresh_scope_after_first_turn_policy_stop(monkeypatch, tmp_path):
    adapter, response = await _run_with_agent(
        monkeypatch, tmp_path, FakeAgent, session_id="feishu-queued-scope", pending_text="next question",
        platform=Platform.FEISHU, adapter_cls=_FirstTurnPolicyAdapter,
        config_data={"display": {"tool_progress": "all"}},
    )
    first = adapter.sent[0]["metadata"]["_feishu_topic_delivery"]
    next_state = response["_queued_terminal_delivery_metadata"]["_feishu_topic_delivery"]
    assert next_state is not first
    assert first["terminal"].retry_suppressed is True
    assert "terminal" not in next_state
    assert adapter.sent[-1]["metadata"]["_feishu_topic_delivery"] is next_state
    assert response["final_response"] == "done"
    assert not response.get("delivery_retry_suppressed")
    assert not response.get("already_sent")


@pytest.mark.asyncio
async def test_poststream_media_shares_event_scope_and_policy_stop_suppresses_footer():
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    source = _source()
    event = MessageEvent(text="question", source=source, message_id="om_question")
    state = runner._event_thread_metadata(event, source)["_feishu_topic_delivery"]
    state["destination"] = "parent_chat"
    terminal = _terminal()

    async def media_delivery(response, received_event, received_adapter, *, thread_metadata):
        assert received_event is event
        assert received_adapter is adapter
        assert thread_metadata["_feishu_topic_delivery"] is state
        assert state["destination"] == "parent_chat"
        state["terminal"] = terminal

    runner._deliver_media_from_response = AsyncMock(side_effect=media_delivery)
    result = {"final_response": "answer MEDIA: /tmp/photo.png", "already_sent": True}
    content = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="s1"), "sk1", 1,
        result, [], result["final_response"], "footer", False,
    )
    assert content is None
    assert event._delivery_retry_suppressed_result is terminal
    assert result["delivery_retry_suppressed"] is True
    assert result["already_sent"] is True
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_background_terminal_text_does_not_upload_attachments(monkeypatch):
    from unittest.mock import MagicMock, patch
    from tests.gateway.test_background_command import _make_runner as background_runner

    runner = background_runner()
    source = _source()
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    adapter.send = AsyncMock(return_value=_terminal())
    adapter.send_image = AsyncMock()
    adapter.extract_media = lambda response: ([], response)
    adapter.extract_images = lambda response: ([("https://example.test/image.png", "caption")], "answer")
    runner.adapters[Platform.FEISHU] = adapter
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
         patch("gateway.run._load_gateway_config", return_value={}), \
         patch("run_agent.AIAgent") as agent:
        agent.return_value = MagicMock()
        agent.return_value.run_conversation.return_value = {"final_response": "answer", "messages": []}
        await runner._run_background_task("question", source, "bg_policy")
    adapter.send.assert_awaited_once()
    assert isinstance(adapter.send.await_args.kwargs["metadata"]["_feishu_topic_delivery"], dict)
    adapter.send_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_policy_stop_blocks_final_text_and_media():
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    runner._should_send_voice_reply = lambda *args, **kwargs: True
    runner._deliver_media_from_response = AsyncMock()
    source = _source()
    event = MessageEvent(text="question", source=source, message_id="om_question")
    terminal = _terminal()

    async def failed_voice(received_event, response):
        runner._event_thread_metadata(received_event, source)["_feishu_topic_delivery"]["terminal"] = terminal

    runner._send_voice_reply = AsyncMock(side_effect=failed_voice)
    result = {"final_response": "answer MEDIA: /tmp/photo.png"}
    content = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="s1"), "sk1", 1,
        result, [], result["final_response"], "footer", False,
    )
    assert content == result["final_response"]
    assert event._delivery_retry_suppressed_result is terminal
    assert result["delivery_retry_suppressed"] is True
    assert not result.get("already_sent")
    runner._deliver_media_from_response.assert_not_awaited()
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_queued_media_terminal_preserves_delivered_body_at_completion():
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner = _make_runner(adapter)
    terminal = _terminal()
    terminal._text_already_delivered = True
    runner._deliver_queued_first_response = AsyncMock(return_value=terminal)
    response = {"final_response": "original", "messages": []}
    result = dict(response)
    source = _source()
    ctx = TurnContext(source=source, session_key="sk1")
    await runner._run_agent_deliver_first_response(ctx, adapter, response, result, None)
    assert result["already_sent"] is True
    assert result["delivery_retry_suppressed"] is True
    assert not result.get("media_already_delivered")
    event = MessageEvent(text="question", source=source, message_id="om_question")
    content = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="s1"), "sk1", 1,
        result, [], result["final_response"], "footer", False,
    )
    assert content is None
    assert event._delivery_retry_suppressed_result is terminal
    assert result["already_sent"] is True
    assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [True, False])
async def test_footer_policy_failure_retains_delivered_text_and_reports_failure(terminal):
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    result = _terminal() if terminal else SendResult(success=False, error="network failure")
    adapter.send = AsyncMock(return_value=result)
    runner = _make_runner(adapter)
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    source = _source()
    event = MessageEvent(text="question", source=source, message_id="om_question")
    response = {"final_response": "answer", "already_sent": True, "media_already_delivered": True}
    content = await runner._hmwa_deliver_turn_response(
        event, source, SimpleNamespace(session_id="s1"), "sk1", 1,
        response, [], response["final_response"], "footer", False,
    )
    assert content is None
    assert response["already_sent"] is True
    assert bool(response.get("delivery_retry_suppressed")) is terminal
    if terminal:
        assert event._delivery_retry_suppressed_result is result
