from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.routing_decision import build_routing_decision, record_agent_fallback, record_fallback
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_routing import build_dispatch_routing_context, persist_run_routing_decision


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _decision():
    return build_routing_decision(
        task_id="task-route",
        board="default",
        task_type="implementation",
        capability="implement",
        risk="normal",
        code_change=True,
        independent_review=True,
        preferred_profile="rozmilo-codex",
        reviewer_profile="rozmilo-claude",
        selected_profile="rozmilo-codex",
        selected_provider="openai-codex",
        selected_model="gpt-5.6-sol",
        human_gate_required=False,
        independence_valid=True,
        policy_digest=None,
        selected_at="2026-09-09T12:00:00Z",
        selected_by="dispatcher",
        run_id=1,
        session_id="session-route",
    )


def test_ready_dispatch_lane_preserves_implementation_routing_facts():
    task = SimpleNamespace(id="task-ready", current_run_id=7, assignee="rozmilo-codex")

    context = build_dispatch_routing_context(task, board="default", lane="ready")

    assert context["task_type"] == "implementation"
    assert context["capability"] == "implement"
    assert context["reviewer_profile"] == "rozmilo-claude"


def test_routing_persistence_updates_run_metadata_and_event_without_lifecycle_change(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="route audit", assignee="rozmilo-codex")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        run_id = claimed.current_run_id
        decision = {**_decision(), "task_id": task_id, "run_id": run_id}

        assert persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision=decision,
            event_kind="routing_selected",
        ) is True

        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert run.status == "running"
        assert run.metadata["routing_decision"] == decision
        route_events = [event for event in kb.list_events(conn, task_id) if event.kind == "routing_selected"]
        assert len(route_events) == 1
        assert route_events[0].payload == {
            "decision_id": decision["decision_id"],
            "profile": "rozmilo-codex",
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "fallback_used": False,
            "fallback_reason": None,
        }


def test_fallback_persistence_records_effective_route_and_survives_completion_metadata(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="fallback audit", assignee="rozmilo-codex")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        run_id = claimed.current_run_id
        initial = {**_decision(), "task_id": task_id, "run_id": run_id}
        persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision=initial,
            event_kind="routing_selected",
        )
        fallback = record_fallback(
            initial,
            from_provider="openai-codex",
            from_model="gpt-5.6-sol",
            to_provider="copilot",
            to_model="gpt-5.6-luna",
            reason="rate_limit",
            recorded_at="2026-09-09T12:01:00Z",
        )

        assert persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision=fallback,
            event_kind="routing_fallback",
        ) is True
        assert kb.complete_task(
            conn,
            task_id,
            result="done",
            metadata={"validation": "passed"},
            expected_run_id=run_id,
        ) is True

        run = kb.get_run(conn, run_id)
        assert run.metadata == {
            "routing_decision": fallback,
            "validation": "passed",
        }
        event = [event for event in kb.list_events(conn, task_id) if event.kind == "routing_fallback"][-1]
        assert event.payload["provider"] == "copilot"
        assert event.payload["model"] == "gpt-5.6-luna"
        assert event.payload["fallback_reason"] == "rate_limit"


def test_stale_or_mismatched_routing_decision_cannot_rewrite_run_history(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="append-only route", assignee="rozmilo-codex")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        run_id = claimed.current_run_id
        initial = {**_decision(), "task_id": task_id, "run_id": run_id}
        fallback = record_fallback(
            initial,
            from_provider="openai-codex",
            from_model="gpt-5.6-sol",
            to_provider="copilot",
            to_model="gpt-5.6-luna",
            reason="rate_limit",
            recorded_at="2026-09-09T12:01:00Z",
        )
        assert persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision=fallback,
            event_kind="routing_fallback",
        ) is True

        assert persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision=initial,
            event_kind="routing_selected",
        ) is False
        assert persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision={**fallback, "task_id": "different-task"},
            event_kind="routing_fallback",
        ) is False
        assert persist_run_routing_decision(
            conn,
            task_id=task_id,
            run_id=run_id,
            decision={**fallback, "human_gate_required": True},
            event_kind="routing_fallback",
        ) is False
        assert kb.get_run(conn, run_id).metadata["routing_decision"] == fallback


def test_rejected_stale_writer_adopts_authoritative_run_decision(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="competing route", assignee="rozmilo-codex")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        run_id = claimed.current_run_id
    db_path = kb.kanban_db_path()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    initial = {**_decision(), "task_id": task_id, "run_id": run_id}

    first = SimpleNamespace(
        provider="copilot",
        model="gpt-5.6-luna",
        session_id="session-route",
        routing_decision=initial,
        _session_db=None,
        _session_init_model_config={},
    )
    authoritative = record_agent_fallback(
        first,
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        reason="rate_limit",
        recorded_at="2026-09-09T12:01:00Z",
    )

    stale = SimpleNamespace(
        provider="zai",
        model="glm-5.2",
        session_id="session-route",
        routing_decision=initial,
        _session_db=None,
        _session_init_model_config={},
    )
    adopted = record_agent_fallback(
        stale,
        from_provider="openai-codex",
        from_model="gpt-5.6-sol",
        reason="provider_error",
        recorded_at="2026-09-09T12:01:01Z",
    )

    assert adopted == authoritative
    assert stale.routing_decision == authoritative
    with kbc.connect() as conn:
        assert kb.get_run(conn, run_id).metadata["routing_decision"] == authoritative
