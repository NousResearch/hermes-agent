"""Inbound source grants belong to the actual consuming gateway turn, not its cache."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.gateway.test_context_ref_expansion_runtime import _make_runner, _patch_runtime_resolution, _source
from gateway.platforms.event import MessageEvent
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from agent.source_provenance import DEFAULT_POLICY_DIGEST, SourceProvenanceError, SourceProvenanceRegistry


async def _prepare(tmp_path, monkeypatch):
    runner = _make_runner()
    _patch_runtime_resolution(monkeypatch)
    source = tmp_path / "source.py"
    source.write_text("safe = 1\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    foreign = SimpleNamespace(session_id="foreign", _current_turn_id="previous")
    runner._agent_cache["context-scope"] = (foreign, "route", 0, "foreign")
    event = MessageEvent(text="Inspect @file:source.py:1-1", source=_source())
    event.reply_to_text = "literal @file:never-read.py:1-1"
    event.reply_to_message_id = "quoted"
    message = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[], session_key="context-scope",
    )
    ctx = TurnContext(source=event.source, session_key="context-scope", message=message)
    ctx.context_source_slices = tuple(getattr(event, "_gateway_source_slices", ()))
    return runner, source, foreign, ctx


def test_inbound_slices_bind_to_new_consuming_turn_and_clear_afterward(tmp_path, monkeypatch):
    runner, source, foreign, ctx = asyncio.run(_prepare(tmp_path, monkeypatch))
    consumed = []
    agent = SimpleNamespace(
        session_id="consumer", _current_turn_id="previous", _current_api_request_id="previous:api:4",
        _source_provenance_registry=SourceProvenanceRegistry(),
    )
    def consume(message, **kwargs):
        turn = getattr(agent, "_source_provenance_pending_turn_id", None)
        assert turn and turn != "previous"
        grants = agent._source_provenance_registry.grants_for_request(f"{turn}:api:1")
        assert len(grants) == 1
        grant = grants[0]
        assert grant.session_id == "consumer" and grant.turn_id == turn
        assert grant.canonical_path == source and grant.policy_digest == DEFAULT_POLICY_DIGEST
        assert "literal @file:never-read.py:1-1" in message
        assert "safe = 1" in message
        consumed.append(grant)
        return {"final_response": "ok"}
    agent.run_conversation = consume
    result = TurnRunner(runner, ctx)._run_conversation_with_approval(agent, [], {}, None, None)
    assert result["final_response"] == "ok" and len(consumed) == 1
    assert not hasattr(foreign, "_source_provenance_registry")
    assert agent._source_provenance_registry.grants_for_request(consumed[0].request_id) == ()
    assert agent._source_provenance_pending_turn_id is None


def test_changed_inbound_source_is_refused_before_consumer_io(tmp_path, monkeypatch):
    runner, source, foreign, ctx = asyncio.run(_prepare(tmp_path, monkeypatch))
    source.write_text("changed = 2\n", encoding="utf-8")
    consume = Mock()
    agent = SimpleNamespace(session_id="consumer", run_conversation=consume)
    with pytest.raises(SourceProvenanceError, match="content_mismatch"):
        TurnRunner(runner, ctx)._run_conversation_with_approval(agent, [], {}, None, None)
    consume.assert_not_called()
    assert agent._source_provenance_pending_turn_id is None
    assert not hasattr(foreign, "_source_provenance_registry")
