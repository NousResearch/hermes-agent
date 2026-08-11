"""Regression coverage for internal delegation-event transcript projection."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest


def _event(delegation_id: str = "deleg_0123456789abcdef0123456789abcdef") -> dict:
    from tools.async_delegation import _internal_event_envelope

    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        **_internal_event_envelope(delegation_id),
    }


def test_internal_event_persistence_requires_canonical_envelope():
    from tools.async_delegation import internal_event_persistence

    event = _event()
    display_kind, metadata = internal_event_persistence(event)

    assert display_kind == "internal_event"
    assert metadata == {
        "event_schema": "hermes.internal_event.v1",
        "event_id": f"async_delegation:{event['delegation_id']}:terminal",
        "event_kind": "workflow.async_delegation.terminal",
        "workflow_id": f"delegation:{event['delegation_id']}",
        "delegation_id": event["delegation_id"],
        "user_originated": False,
        "terminal": True,
    }

    malformed = dict(event, event_id="async_delegation:other:terminal")
    assert internal_event_persistence(malformed) == (None, None)


@pytest.mark.asyncio
async def test_gateway_run_wrapper_forwards_internal_projection_fields():
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    captured = {}

    async def _inner(*_args, **kwargs):
        captured.update(kwargs)
        return {"final_response": "ok"}

    runner._run_agent_inner = _inner
    metadata = {"event_id": "evt", "user_originated": False}
    await runner._run_agent(
        message="synthetic",
        context_prompt="",
        history=[],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        session_id="sid",
        persist_user_display_kind="internal_event",
        persist_user_display_metadata=metadata,
    )

    assert captured["persist_user_display_kind"] == "internal_event"
    assert captured["persist_user_display_metadata"] is metadata


def test_cli_drain_queues_internal_event_with_projection(monkeypatch):
    import tools.async_delegation as delegation
    from cli import HermesCLI, _InternalEventInput
    from tools.process_registry import process_registry

    event = _event()
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "sid"
    cli._pending_input = queue.Queue()
    cli._owns_process_notification = lambda _event: True

    monkeypatch.setattr(
        process_registry,
        "drain_notifications",
        lambda **_kwargs: [(event, "[ASYNC DELEGATION COMPLETE]")],
    )
    monkeypatch.setattr(delegation, "claim_event_delivery", lambda *_args: "claim")
    completed = []
    monkeypatch.setattr(
        delegation,
        "complete_event_delivery",
        lambda evt, claim: completed.append((evt, claim)),
    )

    cli._drain_process_notifications("cli-idle")

    queued = cli._pending_input.get_nowait()
    assert isinstance(queued, _InternalEventInput)
    assert queued.text == "[ASYNC DELEGATION COMPLETE]"
    assert queued.display_kind == "internal_event"
    assert queued.display_metadata["event_id"] == event["event_id"]
    assert queued.display_metadata["user_originated"] is False
    assert completed == [(event, "claim")]
