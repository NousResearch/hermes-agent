"""A human closing a card from the board must not need to type a summary.

A drag (or a kebab-menu close) sends neither ``result`` nor ``summary``.
``complete_task`` refuses a non-review completion that would leave the card with no
evidence, and its only exemption is ``status == "review"`` — so dragging any card in
``blocked``/``ready``/``todo`` to Done used to fail with no visible explanation
(the board only recorded ``completion_blocked_empty_result``).

The card's own conversation is the evidence: the owner's last words, or — when they said
nothing — a line that attributes and timestamps the human action.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_drag_completion_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _parked_card(conn) -> str:
    """A decision card: blocked, never worked, no stored result."""
    tid = kb.create_task(conn, title="stage gate")
    kb.block_task(conn, tid, reason="waiting for the owner", kind="needs_input")
    return tid


def _last_run_summary(conn, tid: str) -> str:
    row = conn.execute(
        "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    return (row["summary"] if row else "") or ""


def test_drag_to_done_uses_the_last_comment_as_evidence(client):
    with kbc.connect() as conn:
        tid = _parked_card(conn)
        kb.add_comment(conn, tid, "owner", "approved: works for me")

    r = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"status": "done"})

    assert r.status_code == 200, r.text
    with kbc.connect() as conn:
        status = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"]
        assert status == "done"
        assert "approved: works for me" in _last_run_summary(conn, tid)


def test_drag_to_done_without_a_comment_still_closes_and_attributes_the_human(client):
    with kbc.connect() as conn:
        tid = _parked_card(conn)

    r = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={"status": "done"})

    assert r.status_code == 200, r.text
    with kbc.connect() as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
        summary = _last_run_summary(conn, tid)
        assert "Closed from the dashboard by a human" in summary


def test_a_summary_from_the_payload_still_wins(client):
    with kbc.connect() as conn:
        tid = _parked_card(conn)
        kb.add_comment(conn, tid, "owner", "approved: nice")

    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "done", "summary": "the operator typed this"},
    )

    assert r.status_code == 200, r.text
    with kbc.connect() as conn:
        assert _last_run_summary(conn, tid) == "the operator typed this"
