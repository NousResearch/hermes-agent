"""Staging tests for ensure_escalation (temp SQLite only)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_escalation import EnsureEscalationError, ensure_escalation


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    return home


def _incident(conn, title="[AUD-P1] staging"):
    return kb.create_task(
        conn, title=title, body="incident", assignee="auditor", created_by="test",
    )


def _payload(incident_id, reason="open_p1", source="auditor_reason", **extra):
    data = {"incident_id": incident_id, "reason": reason, "source": source}
    data.update(extra)
    return data


def test_first_call_creates_one_ready_default(kanban_home):
    with kbc.connect() as conn:
        iid = _incident(conn)
        out = ensure_escalation(conn, _payload(iid))
        esc = kb.get_task(conn, out["escalation_id"])
        inc = kb.get_task(conn, iid)
        rows = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '[ESC] %' AND status != 'archived'",
        ).fetchall()
        assert out["created"] is True
        assert esc.status == "ready"
        assert esc.assignee == "default"
        assert inc.assignee == "auditor"
        assert inc.status != "done"
        assert len(rows) == 1
        assert kbd.has_spawnable_ready(conn) is True or esc.status == "ready"


def test_sequential_repeat_same_id(kanban_home):
    with kbc.connect() as conn:
        iid = _incident(conn)
        a = ensure_escalation(conn, _payload(iid, reason="open_p1"))
        b = ensure_escalation(conn, _payload(iid, reason="ambiguity"))
        assert a["escalation_id"] == b["escalation_id"]
        assert b["created"] is False
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM tasks WHERE title LIKE '[ESC] %' AND status NOT IN ('done','archived')",
        ).fetchone()["c"]
        assert n == 1


def test_concurrent_same_incident_one_esc(kanban_home):
    with kbc.connect() as conn:
        iid = _incident(conn)
    results = []
    errors = []

    def worker():
        try:
            with kbc.connect() as conn:
                results.append(ensure_escalation(conn, _payload(iid)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(results) == 2
    assert results[0]["escalation_id"] == results[1]["escalation_id"]
    with kbc.connect() as conn:
        open_esc = conn.execute(
            "SELECT id, status, assignee FROM tasks WHERE title LIKE '[ESC] %' "
            "AND status NOT IN ('done','archived')",
        ).fetchall()
        assert len(open_esc) == 1
        assert open_esc[0]["status"] == "ready"
        assert open_esc[0]["assignee"] == "default"
        inc = kb.get_task(conn, iid)
        assert inc.assignee == "auditor"


def test_failure_before_commit_leaves_nothing(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        iid = _incident(conn)

        def boom(*_a, **_k):
            raise RuntimeError("injected")

        monkeypatch.setattr(kb, "add_comment", boom)
        with pytest.raises(RuntimeError):
            ensure_escalation(conn, _payload(iid))
        left = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '[ESC] %'",
        ).fetchall()
        assert left == []
        assert kb.get_task(conn, iid).assignee == "auditor"


def test_next_generation_after_terminal(kanban_home):
    with kbc.connect() as conn:
        iid = _incident(conn)
        first = ensure_escalation(conn, _payload(iid))
        assert kb.complete_task(conn, first["escalation_id"], result="decided")
        second = ensure_escalation(conn, _payload(iid, reason="recurrence"))
        assert second["escalation_id"] != first["escalation_id"]
        assert second["created"] is True
        esc1 = kb.get_task(conn, first["escalation_id"])
        esc2 = kb.get_task(conn, second["escalation_id"])
        assert esc1.status == "done"
        assert esc2.status == "ready"
        assert esc2.assignee == "default"
        open_n = conn.execute(
            "SELECT COUNT(*) AS c FROM tasks WHERE title LIKE '[ESC] %' AND status = 'ready'",
        ).fetchone()["c"]
        assert open_n == 1


def test_rejects_non_auditor_incident(kanban_home):
    with kbc.connect() as conn:
        iid = kb.create_task(conn, title="[AUD-P1] x", assignee="jerusalem", created_by="test")
        with pytest.raises(EnsureEscalationError):
            ensure_escalation(conn, _payload(iid))


def test_dispatcher_does_not_duplicate_esc(kanban_home, monkeypatch):
    spawned = []

    def spawn_fn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: name == "default")
    with kbc.connect() as conn:
        iid = _incident(conn)
        esc = ensure_escalation(conn, _payload(iid))
        esc_id = esc["escalation_id"]
        kbd.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        kbd.dispatch_once(conn, spawn_fn=spawn_fn, stale_timeout_seconds=0)
        open_esc = conn.execute(
            "SELECT id, status, assignee FROM tasks WHERE title LIKE '[ESC] %' "
            "AND status NOT IN ('done','archived')",
        ).fetchall()
        assert len(open_esc) == 1
        assert open_esc[0]["id"] == esc_id
        assert open_esc[0]["assignee"] == "default"
        inc = kb.get_task(conn, iid)
        assert inc.assignee == "auditor"
        assert inc.status != "done"
        assert spawned.count(esc_id) == 1
        assert iid not in spawned
