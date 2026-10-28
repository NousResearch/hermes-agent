"""Interim assistant commentary carries a semantic marker adapters can act on.

``_interim_send`` only says "not the turn-final" (approval prompts and stuck warnings carry it
too), so an adapter that wants to suppress optional commentary on some routes needs
``interim_assistant_message``. When it does suppress, ``SendResult.delivered=False`` keeps the
invisible text out of the consumer's delivered-text ledger so a matching final is still sent.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_run_progress_topics import CommentaryAgent, _run_with_agent


def _adapter(*results):
    adapter = MagicMock()
    adapter.send = AsyncMock(side_effect=list(results))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.MAX_MESSAGE_LENGTH = 4096
    return adapter


@pytest.mark.asyncio
async def test_commentary_is_marked_and_final_is_not():
    adapter = _adapter(
        SimpleNamespace(success=True, message_id="m1"),
        SimpleNamespace(success=True, message_id="m2"),
    )
    consumer = GatewayStreamConsumer(
        adapter, "chat_123", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        metadata={"thread_id": "thread_7"},
    )

    consumer.on_commentary("I'll inspect the repository first.")
    consumer.on_delta("Done.")
    consumer.finish()
    await consumer.run()

    commentary_md, final_md = (c.kwargs["metadata"] for c in adapter.send.call_args_list)
    assert commentary_md["interim_assistant_message"] is True
    assert commentary_md["thread_id"] == "thread_7"
    assert "interim_assistant_message" not in final_md
    assert final_md["thread_id"] == "thread_7"


@pytest.mark.asyncio
async def test_suppressed_commentary_is_not_recorded_as_delivered():
    adapter = _adapter(SimpleNamespace(success=True, message_id="dropped", delivered=False))
    new_messages = []
    consumer = GatewayStreamConsumer(
        adapter, "chat_123", StreamConsumerConfig(), on_new_message=lambda: new_messages.append(1),
    )

    assert await consumer._send_commentary("Final answer") is True
    # Invisible text must not dedup a later final with the same content.
    assert consumer.has_delivered_text("Final answer") is False
    assert new_messages == []


@pytest.mark.asyncio
async def test_fallback_commentary_send_is_marked_when_consumer_setup_fails(monkeypatch, tmp_path):
    gateway_run = importlib.import_module("gateway.run")
    called = []

    def _fail(self, *args, **kwargs):
        called.append(True)
        raise RuntimeError("synthetic stream consumer setup failure")

    monkeypatch.setattr(gateway_run.GatewayRunner, "_build_stream_consumer_config", _fail)
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-interim-fallback-marker",
        config_data={
            "display": {"interim_assistant_messages": True},
            "streaming": {"enabled": False},
        },
    )

    assert called, "consumer setup must have failed so the fallback path ran"
    assert result["final_response"] == "done"
    assert [c["content"] for c in adapter.sent] == ["I'll inspect the repo first."]
    assert adapter.sent[0]["metadata"]["interim_assistant_message"] is True
    assert adapter.sent[0]["metadata"]["thread_id"] == "17585"
