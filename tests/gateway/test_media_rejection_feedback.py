"""Rejected attachments must reach both the human and next model turn."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.session import AsyncSessionStore, SessionStore, build_session_key
from gateway.run import GatewayRunner
from run_agent import AIAgent
from tests.gateway.test_background_command import _make_runner
from tests.gateway.test_tool_response_drop_recovery import _DummyAdapter, _make_event


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["normal", "streamed", "background"])
async def test_rejected_media_is_not_silent_and_is_in_session_history(tmp_path, delivery):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    async_store = AsyncSessionStore(store)
    event = _make_event(Platform.DISCORD)
    adapter = _DummyAdapter(Platform.DISCORD)
    runner = _make_runner()
    assert isinstance(runner, GatewayRunner)
    runner.session_store = store
    adapter.gateway_runner = runner
    session = await async_store.get_or_create_session(event.source)

    await async_store.append_to_transcript(session.session_id, {"role": "assistant", "content": "Sent it."})
    try:
        response = "Sent it.\nMEDIA:/nonexistent/private report.pdf"
        if delivery == "normal":
            result = await adapter._extract_response_content(
                response, event, build_session_key(event.source), is_ephemeral_response=False,
            )
            assert not result.media_files
            assert "was not attached" in result.text_content
            assert "private report" not in result.text_content
        elif delivery == "streamed":
            await runner._deliver_media_from_response(response, event, adapter, thread_metadata={})
            assert any("was not attached" in sent["content"] for sent in adapter.sent)
        else:
            runner.adapters[Platform.DISCORD] = adapter
            background_source = _make_event(Platform.DISCORD, chat_id="background-fixture").source
            background = await async_store.get_or_create_session(background_source)
            agent = MagicMock(spec=AIAgent)
            agent.run_conversation.return_value = {"final_response": response, "messages": []}
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
                 patch("gateway.run._load_gateway_config", return_value={}), \
                 patch("run_agent.AIAgent", return_value=agent):
                await runner._run_background_task("send report", event.source, background.session_id)
            assert any("was not attached" in sent["content"] for sent in adapter.sent)
        original_history = await async_store.load_transcript(session.session_id)
        assert original_history[0]["content"] == "Sent it."
        if delivery == "background":
            assert len(original_history) == 1
            history = await async_store.load_transcript(background.session_id)
        else:
            history = original_history
        assert history[-1]["role"] == "user"
        assert "was not attached" in history[-1]["content"]
        assert "private report" not in history[-1]["content"]
    finally:
        store._db.close()


@pytest.mark.asyncio
async def test_failed_warning_does_not_abort_media_extraction(monkeypatch):
    from gateway.media_delivery_feedback import extract_media_with_feedback

    adapter = _DummyAdapter(Platform.DISCORD)
    monkeypatch.setattr(adapter, "emit_warning", AsyncMock(side_effect=RuntimeError("transport offline")))
    media, text = await extract_media_with_feedback(
        None, adapter, "MEDIA:/nonexistent/private.pdf", _make_event(Platform.DISCORD).source, notify=True,
    )
    assert media == []
    assert "was not attached" in text
