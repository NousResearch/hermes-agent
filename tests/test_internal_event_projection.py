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

    assert display_kind == "async_delegation_complete"
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


def test_trusted_generic_internal_event_gets_deterministic_non_human_projection():
    from tools.async_delegation import internal_event_persistence

    event = {
        "type": "completion",
        "session_id": "proc-42",
        "completed_at": "2026-08-11T03:00:00Z",
        "text": "sensitive process output must not be copied to metadata",
    }

    assert internal_event_persistence(event) == (None, None)
    display_kind, metadata = internal_event_persistence(
        event,
        trusted_internal=True,
    )
    display_kind_again, metadata_again = internal_event_persistence(
        dict(reversed(list(event.items()))),
        trusted_internal=True,
    )

    assert display_kind == display_kind_again == "internal_event"
    assert metadata == metadata_again
    assert metadata["event_schema"] == "hermes.internal_event.v1"
    assert metadata["event_id"].startswith("internal_event:")
    assert metadata["event_kind"] == "workflow.completion.internal"
    assert metadata["user_originated"] is False
    assert "text" not in metadata


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
        persist_user_display_kind="async_delegation_complete",
        persist_user_display_metadata=metadata,
    )

    assert captured["persist_user_display_kind"] == "async_delegation_complete"
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
    assert queued.display_kind == "async_delegation_complete"
    assert queued.display_metadata["event_id"] == event["event_id"]
    assert queued.display_metadata["user_originated"] is False
    assert completed == [(event, "claim")]


def test_cli_drain_projects_trusted_process_completion_as_internal(monkeypatch):
    import tools.async_delegation as delegation
    from cli import HermesCLI, _InternalEventInput
    from tools.process_registry import process_registry

    event = {"type": "completion", "session_id": "proc-42", "exit_code": 0}
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "sid"
    cli._pending_input = queue.Queue()
    cli._owns_process_notification = lambda _event: True
    monkeypatch.setattr(
        process_registry,
        "drain_notifications",
        lambda **_kwargs: [(event, "[SYSTEM: process completed]")],
    )
    monkeypatch.setattr(delegation, "claim_event_delivery", lambda *_args: "claim")
    monkeypatch.setattr(delegation, "complete_event_delivery", lambda *_args: None)

    cli._drain_process_notifications("cli-idle")

    queued = cli._pending_input.get_nowait()
    assert isinstance(queued, _InternalEventInput)
    assert queued.display_kind == "internal_event"
    assert queued.display_metadata["user_originated"] is False
    assert queued.display_metadata["event_kind"] == "workflow.completion.internal"


def test_tui_notification_prompt_preserves_internal_projection(monkeypatch):
    from tui_gateway import server

    event = _event()
    event.update(
        results=[
            {"status": "completed"},
            {"status": "failed"},
        ],
        total_duration_seconds=4.5,
    )
    captured = {}
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, session, text, **kwargs: captured.update(
            rid=rid,
            sid=sid,
            session=session,
            text=text,
            kwargs=kwargs,
        ),
    )
    session = {"session_key": "sid"}

    server._run_notification_prompt(
        "rid",
        "sid",
        session,
        event,
        "[ASYNC DELEGATION COMPLETE]",
    )

    assert captured["text"] == "[ASYNC DELEGATION COMPLETE]"
    assert captured["kwargs"]["display_kind"] == "async_delegation_complete"
    assert captured["kwargs"]["display_metadata"]["event_id"] == event["event_id"]
    assert captured["kwargs"]["display_metadata"]["user_originated"] is False
    assert captured["kwargs"]["display_metadata"]["task_count"] == 2
    assert captured["kwargs"]["display_metadata"]["completed_count"] == 1
    assert captured["kwargs"]["display_metadata"]["failed_count"] == 1
    assert captured["kwargs"]["display_metadata"]["duration_seconds"] == 4.5


def test_tui_notification_prompt_projects_trusted_process_completion(monkeypatch):
    from tui_gateway import server

    captured = {}
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, sid, session, text, **kwargs: captured.update(kwargs=kwargs),
    )

    server._run_notification_prompt(
        "rid",
        "sid",
        {"session_key": "sid"},
        {"type": "completion", "session_id": "proc-42", "exit_code": 0},
        "[SYSTEM: process completed]",
    )

    assert captured["kwargs"]["display_kind"] == "internal_event"
    assert captured["kwargs"]["display_metadata"]["user_originated"] is False
