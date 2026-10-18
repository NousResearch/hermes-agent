"""Gateway-owned CLI expansion carries file bytes to the actual consuming turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_cli_turn import cli_turn, cli_turn_scope
from gateway.session_context_refs import turn_references
from gateway.source_context import bind_inbound_source_slices
from agent.source_provenance import SourceProvenanceError


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_cli_api_reference_keeps_raw_slice_for_consumer(tmp_path, monkeypatch, changed):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "gateway.session_policy.policy_for_source",
        lambda *_args: SimpleNamespace(cwd=str(tmp_path)),
    )
    runner = SimpleNamespace(_inbound_model_context_length=AsyncMock(return_value=128_000))
    text = "Inspect @file:source.py:1-1"
    event = MessageEvent(
        text=text, source=SessionSource(platform=Platform.LOCAL, chat_id="local"),
        message_id="source-turn",
    )
    event._gateway_source_slices = []
    with cli_turn_scope(event.message_id, {"api_text": "[voice] " + text}):
        assert await turn_references(runner, event, event.source, "local", text) == text
        expanded = cli_turn(event.message_id)["api_text"]
        assert expanded.startswith("[voice] " + text) and expanded.count("value = 1") == 1
    assert len(event._gateway_source_slices) == 1
    consumer = SimpleNamespace(session_id="consumer")
    if changed:
        source.write_text("value = 2\n", encoding="utf-8")
        with pytest.raises(SourceProvenanceError, match="content_mismatch"):
            bind_inbound_source_slices(consumer, event._gateway_source_slices)
        assert consumer._source_provenance_pending_turn_id is None
    else:
        bind_inbound_source_slices(consumer, event._gateway_source_slices)
        turn = consumer._source_provenance_pending_turn_id
        [grant] = consumer._source_provenance_registry.grants_for_request(f"{turn}:api:1")
        assert grant.canonical_path == source and grant.session_id == "consumer"
