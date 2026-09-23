"""A failed bounded final is recorded, not duplicated by corrective delivery."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run_turn import GatewayTurnMixin
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_failed_stream_handoff_is_ledgered_without_corrective_send(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    text = "the unchanged final"
    failure = SendResult(success=False, retryable=True, error="send_path_degraded",
                         raw_response={"defer_final_delivery": True})
    consumer = SimpleNamespace(deferred_final_delivery=lambda value: failure if value == text else None)
    runner = GatewayTurnMixin()
    source = SimpleNamespace(chat_id="555")
    turn = SimpleNamespace(stream_consumer_holder=[consumer], source=source, session_key="session")
    response = {"final_response": text}
    await runner._run_agent_mark_streamed_delivery(response, turn)
    assert response["deferred_final_delivery"] is failure
    assert not response.get("already_sent")
    adapter = DiscordAdapter(PlatformConfig())
    runner._delivery_adapter_for = lambda _: adapter
    runner._should_send_voice_reply = lambda *a, **kw: False
    runner._streaming_tts_turn_completed = lambda *a: False
    runner._hmwa_surface_delivery_gaps = AsyncMock()
    event = SimpleNamespace(source=source)
    result = await runner._hmwa_deliver_turn_response(
        event, source, None, "session", 1, response, [], text, "", False,
    )
    assert result == text
    adapter._final_delivery_adapter = lambda _: adapter
    adapter._record_delivery_obligation = AsyncMock(return_value=12)
    adapter._finalize_delivery_obligation = AsyncMock()
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
    delivered, owner = await adapter.send_final_ledgered(event, "session", text, {}, reply_to="123")
    assert delivered is failure and owner is adapter
    adapter._send_with_retry.assert_not_awaited()
    adapter._record_delivery_obligation.assert_awaited_once()
    adapter._finalize_delivery_obligation.assert_awaited_once_with(12, failure, event, adapter)
    # The handoff is per-turn and consumed once; a real later retry is not suppressed.
    delivered, _ = await adapter.send_final_ledgered(event, "session", text, {}, reply_to="123")
    assert delivered.success
    adapter._send_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_handoff_does_not_suppress_changed_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = DiscordAdapter(PlatformConfig())
    failure = SendResult(success=False, retryable=True)
    event = SimpleNamespace(source=SimpleNamespace(chat_id="555"),
                            _deferred_final_delivery=("old final", failure))
    adapter._final_delivery_adapter = lambda _: adapter
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
    result, _ = await adapter.send_final_ledgered(event, "session", "new final", {}, reply_to="123")
    assert result.success
    adapter._send_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_queued_failed_final_is_ledgered_without_reconcile_or_send():
    from gateway.run_notifications import GatewayNotificationsMixin
    from gateway.session import SessionSource
    from gateway.config import Platform
    adapter = DiscordAdapter(PlatformConfig())
    adapter._final_delivery_adapter = lambda _: adapter
    adapter._record_delivery_obligation = AsyncMock(return_value=12)
    adapter._finalize_delivery_obligation = AsyncMock()
    adapter._send_with_retry = AsyncMock()
    adapter.edit_message = AsyncMock()
    failure = SendResult(success=False, retryable=True, error="send_path_degraded")
    text = "failed full final"
    consumer = SimpleNamespace(message_id="42", _turn_split_delivery=False,
                               deferred_final_delivery=lambda final: failure if final == text else None)
    runner = GatewayNotificationsMixin()
    await runner._deliver_queued_first_response(
        text, SessionSource(platform=Platform.DISCORD, chat_id="555"), adapter,
        stream_consumer=consumer, session_key="s", inbound_message_id="123", deliver_media=False,
    )
    adapter._send_with_retry.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
    assert adapter._finalize_delivery_obligation.call_args.args[1] is failure


@pytest.mark.asyncio
async def test_transformed_reconciliation_does_not_claim_failed_upload():
    failure = SendResult(success=False, retryable=True,
                         raw_response={"defer_final_delivery": True})
    consumer = SimpleNamespace(message_id="42", adapter=SimpleNamespace(
        edit_message=AsyncMock(return_value=failure)))
    response = {"final_response": "transformed final"}
    await GatewayTurnMixin()._run_agent_edit_streamed_message(
        consumer, SimpleNamespace(chat_id="555"), response, "transformed final",
        _sk="s", ok=("ok",), fail_result=None, fail_exc="%s %s",
    )
    assert not response.get("already_sent")
    assert response.get("deferred_final_delivery") is failure
