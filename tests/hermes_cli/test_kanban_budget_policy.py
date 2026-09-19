"""Focused contracts for Kanban worker turn-budget calibration."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.kanban_budget_policy import POLICY_VERSION, resolve_task_budget
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _task(title, body=None, goal_max_turns=None):
    return SimpleNamespace(title=title, body=body, goal_max_turns=goal_max_turns)


def test_classifies_known_archetypes_and_reserves_handoff_turns():
    budget = resolve_task_budget(_task("Implement retry-safe dispatcher"))

    assert budget["policy_version"] == POLICY_VERSION
    assert budget["archetype"] == "implementation"
    assert budget["phase_turns"] == {"plan": 6, "implement": 24, "verify": 8, "handoff": 2}
    assert sum(budget["phase_turns"].values()) == budget["max_turns"]


def test_unknown_task_uses_explicit_conservative_fallback():
    budget = resolve_task_budget(_task("Handle this"))

    assert budget["archetype"] == "unknown"
    assert budget["fallback_reason"] == "no_archetype_signal"
    assert budget["max_turns"] == 24


def test_title_signal_wins_over_body_signal():
    budget = resolve_task_budget(_task("Implement parser", "Research alternatives first"))

    assert budget["archetype"] == "implementation"


def test_bounded_operator_override_preserves_phase_reservations():
    budget = resolve_task_budget(_task("Research API behavior", goal_max_turns=40))

    assert budget["archetype"] == "research"
    assert budget["max_turns"] == 40
    assert budget["override_turns"] == 40
    assert budget["phase_turns"]["handoff"] >= 2


def test_invalid_override_fails_open_to_policy_default():
    budget = resolve_task_budget(_task("Verify test suite", goal_max_turns=1))

    assert budget["archetype"] == "verification"
    assert budget["override_turns"] is None
    assert budget["fallback_reason"] == "invalid_override"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_creation_claim_retry_and_spawn_preserve_selected_policy(kanban_home, monkeypatch):
    captured = {}

    class FakeProc:
        pid = 7

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: captured.update(kwargs) or FakeProc())
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda task, command: command)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Implement retry-safe dispatch", assignee="worker")
        created = kb.list_events(conn, task_id)[0].payload["budget_policy"]
        first = kb.claim_task(conn, task_id)
        first_run = kb.get_run(conn, first.current_run_id).metadata
        conn.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, current_run_id = NULL WHERE id = ?", (task_id,))
        second = kb.claim_task(conn, task_id)
        second_run = kb.get_run(conn, second.current_run_id).metadata

    kbd._default_spawn(second, str(kanban_home / "workspace"))
    assert created == first_run == second_run
    assert captured["env"]["HERMES_MAX_ITERATIONS"] == "40"
    assert json.loads(captured["env"]["HERMES_KANBAN_BUDGET_POLICY"]) == second_run
