"""Regression coverage for deterministic Kanban watchdog alerts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.kanban_watchers_notifier import TERMINAL_KINDS
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd
from hermes_cli import kanban_watchdog as kw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, task_id):
    return list(conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
    ))


def test_watchdog_cli_reports_quiet_healthy_output(kanban_home):
    from hermes_cli import kanban as kc

    import json
    assert json.loads(kc.run_slash("watchdog --json")) == {
        "new_alerts": [], "resolved_count": 0, "pruned_count": 0,
    }


def test_watchdog_alerts_are_routed_through_existing_notifier_pipeline():
    assert "watchdog_alert" in TERMINAL_KINDS


def test_review_without_a_claim_is_diagnosed_as_stranded(kanban_home):
    now = 1_000_000
    task = {"id": "t_review", "status": "review", "assignee": "reviewer", "claim_lock": None, "created_at": now - 3600}
    diags = kd.compute_task_diagnostics(
        task, [{"kind": "review_requested", "created_at": now - 3600, "payload": {}}], [], now=now
    )
    assert [diag.kind for diag in diags] == ["stranded_in_review"]


def test_watchdog_emits_one_alert_then_suppresses_duplicate_for_stranded_ready(kanban_home):
    now = 1_000_000
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="safe fixture", assignee="worker")
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (now - 3600, task_id))
        conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'created'", (now - 3600, task_id))
        conn.commit()

        first = kw.run_watchdog(conn, now=now)
        second = kw.run_watchdog(conn, now=now + 60)

        assert [(a.task_id, a.kind) for a in first.new_alerts] == [(task_id, "stranded_in_ready")]
        assert second.new_alerts == []
        assert [row["kind"] for row in _events(conn, task_id)].count("watchdog_alert") == 1


def test_watchdog_realerts_only_after_condition_clears_then_recurs(kanban_home):
    now = 1_000_000
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="safe fixture", assignee="worker")
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (now - 3600, task_id))
        conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'created'", (now - 3600, task_id))
        conn.commit()
        assert len(kw.run_watchdog(conn, now=now).new_alerts) == 1

        conn.execute("UPDATE tasks SET claim_lock = 'live' WHERE id = ?", (task_id,))
        conn.commit()
        assert kw.run_watchdog(conn, now=now + 1).resolved_count == 1

        conn.execute("UPDATE tasks SET claim_lock = NULL WHERE id = ?", (task_id,))
        conn.commit()
        assert len(kw.run_watchdog(conn, now=now + 2).new_alerts) == 1
        assert [row["kind"] for row in _events(conn, task_id)].count("watchdog_alert") == 2


def test_watchdog_alerts_completed_deployment_required_task_without_activation_evidence(kanban_home):
    now = 1_000_000
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="release", assignee="ops", body="deployment-required: publish the release"
        )
        conn.execute("UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?", (now - 3600, task_id))
        conn.commit()

        result = kw.run_watchdog(conn, now=now)

        assert [(a.task_id, a.kind) for a in result.new_alerts] == [(task_id, "activation_missing")]
        payload = _events(conn, task_id)[-1]["payload"]
        assert "activation" in payload


def test_watchdog_treats_live_verified_completion_metadata_as_activation_evidence(kanban_home):
    now = 1_000_000
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="release", assignee="ops", body="deployment-required: publish the release"
        )
        conn.execute("UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?", (now - 3600, task_id))
        kb._append_event(conn, task_id, "completed", {"metadata": {"live_verified": True}})
        conn.commit()

        assert kw.run_watchdog(conn, now=now).new_alerts == []
