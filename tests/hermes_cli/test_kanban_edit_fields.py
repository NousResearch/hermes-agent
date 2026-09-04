"""Regression tests for issue #52371: hermes kanban edit cannot update task
body/title/priority after creation, despite the documented contract at
website/docs/user-guide/features/kanban.md ("edit task title / body /
priority in place").

Store-layer coverage lives here alongside the other kanban_db mutator tests;
CLI/gateway (run_slash) coverage lives in tests/hermes_cli/test_kanban_cli.py.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb.Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _mk_task(conn, *, title: str = "original title", body: str = "original body"):
    return kb.create_task(conn, title=title, body=body)


def _last_event(conn, task_id, kind):
    events = kb.list_events(conn, task_id)
    matches = [e for e in events if e.kind == kind]
    return matches[-1] if matches else None


class TestEditTaskFieldsStore:
    def test_edit_body_in_place(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            assert kb.edit_task_fields(conn, tid, body="updated body") is True
            task = kb.get_task(conn, tid)
            assert task.body == "updated body"
            assert task.title == "original title"
            assert task.status in {"ready", "running"}

    def test_edit_title_and_priority_in_place(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            assert kb.edit_task_fields(
                conn, tid, title="new title", priority=5
            ) is True
            task = kb.get_task(conn, tid)
            assert task.title == "new title"
            assert task.priority == 5
            assert task.body == "original body"

    def test_edit_records_edited_event_with_changed_fields(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            kb.edit_task_fields(conn, tid, body="updated body")
            ev = _last_event(conn, tid, "edited")
            assert ev is not None
            payload = ev.payload
            assert payload["fields"] == ["body"]

    def test_edit_allowed_in_all_non_archived_states(self, kanban_home):
        with kb.connect_closing() as conn:
            for status in ("ready", "todo", "blocked", "running", "review", "done"):
                tid = _mk_task(conn, title=f"t-{status}")
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?", (status, tid)
                )
                assert kb.edit_task_fields(conn, tid, body=f"b-{status}") is True
                assert kb.get_task(conn, tid).body == f"b-{status}"

    def test_edit_refuses_archived_task(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            conn.execute(
                "UPDATE tasks SET status = 'archived' WHERE id = ?", (tid,)
            )
            with pytest.raises(RuntimeError, match="archived"):
                kb.edit_task_fields(conn, tid, body="nope")

    def test_edit_unknown_task_returns_false(self, kanban_home):
        with kb.connect_closing() as conn:
            assert kb.edit_task_fields(conn, "t_nope", body="x") is False

    def test_edit_blank_title_rejected(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            with pytest.raises(ValueError, match="title cannot be blank"):
                kb.edit_task_fields(conn, tid, title="   ")

    def test_edit_noop_records_no_event(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            assert kb.edit_task_fields(conn, tid, body="original body") is True
            ev = _last_event(conn, tid, "edited")
            assert ev is None

    def test_edit_noop_fires_no_task_updated_observer(self, kanban_home, monkeypatch):
        # A no-op edit commits no row change — the RFC #58548 observer
        # contract ("fired for a committed task-row mutation") says it must
        # not fire, same as the event log staying silent.
        fired = []
        monkeypatch.setattr(
            kb, "notify_task_updated",
            lambda conn, task_id, fields, **kw: fired.append((task_id, list(fields))),
        )
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            assert kb.edit_task_fields(conn, tid, body="original body") is True
        assert fired == []

    def test_edit_does_not_disturb_running_claim_state(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'lock-1', "
                "claim_expires = 9999999999, current_run_id = NULL WHERE id = ?",
                (tid,),
            )
            assert kb.edit_task_fields(conn, tid, body="edited mid-run") is True
            row = conn.execute(
                "SELECT status, claim_lock, claim_expires FROM tasks WHERE id = ?",
                (tid,),
            ).fetchone()
            assert row["status"] == "running"
            assert row["claim_lock"] == "lock-1"
            assert row["claim_expires"] == 9999999999

    def test_edit_body_fires_task_updated_observer(self, kanban_home, monkeypatch):
        fired = []
        monkeypatch.setattr(
            kb, "notify_task_updated",
            lambda conn, task_id, fields, **kw: fired.append((task_id, list(fields))),
        )
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            kb.edit_task_fields(conn, tid, body="observer body")
        assert fired == [(tid, ["body"])]

    def test_edit_triage_task_does_not_promote(self, kanban_home):
        with kb.connect_closing() as conn:
            tid = _mk_task(conn)
            conn.execute(
                "UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,)
            )
            assert kb.edit_task_fields(conn, tid, body="triage body") is True
            # In-place edit must NOT auto-promote — specify_triage_task owns
            # the triage -> todo transition.
            assert kb.get_task(conn, tid).status == "triage"
