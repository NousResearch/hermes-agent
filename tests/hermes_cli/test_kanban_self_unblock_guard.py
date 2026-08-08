"""Tests for the kanban self-unblock caller-identity guard (#75319).

A worker must not be able to silently lift its own human-gate block:
``unblock_task`` refuses when the caller's identity matches the recorded
blocker on a ``needs_input`` / ``capability`` / legacy (NULL-kind) block
unless the caller explicitly acknowledges the self-unblock. Programmatic
(``transient``) blocks proceed with a recorded warning. Tasks with no
recorded blocker (legacy rows) fail open. Every unblock records the actor
on the audit trail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _blocked_task(conn, *, kind=None, author: str | None = "alice", reason="review-required: please check"):
    """Create a task and block it from ``ready`` with a recorded author."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    ok = kb.block_task(conn, tid, reason=reason, kind=kind, author=author)
    assert ok is True
    return tid


def _events(conn, tid):
    return [
        {"kind": e.kind, "payload": e.payload}
        for e in kb.list_events(conn, tid)
    ]


# ---------------------------------------------------------------------------
# DB layer: blocked_by recording
# ---------------------------------------------------------------------------


def test_block_records_blocked_by(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.blocked_by == "alice"


def test_block_without_author_leaves_null_blocker(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.block_task(conn, tid, reason="x")
        assert kb.get_task(conn, tid).blocked_by is None


# ---------------------------------------------------------------------------
# DB layer: self-unblock guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["needs_input", "capability", None])
def test_self_unblock_refused_on_human_gate(kanban_home, kind):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind=kind, author="alice")
        ok, err = kb.unblock_task(conn, tid, actor="alice")
        assert ok is False
        assert "blocked by alice" in (err or "")
        # The task stays blocked — the guard must not partially transition.
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.get_task(conn, tid).blocked_by == "alice"


def test_self_unblock_allowed_with_explicit_override(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
        ok, err = kb.unblock_task(conn, tid, actor="alice", override_self=True)
        assert ok is True
        assert err is None
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.blocked_by is None
        # Audit trail distinguishes the acknowledged self-unblock.
        unblocked = [e for e in _events(conn, tid) if e["kind"] == "unblocked"]
        assert unblocked
        assert unblocked[-1]["payload"] == {
            "actor": "alice",
            "override_self": True,
        }


def test_different_actor_unblocks_without_flag(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
        ok, err = kb.unblock_task(conn, tid, actor="bob")
        assert ok is True
        assert err is None
        assert kb.get_task(conn, tid).status == "ready"
        unblocked = [e for e in _events(conn, tid) if e["kind"] == "unblocked"]
        assert unblocked[-1]["payload"] == {"actor": "bob"}


def test_transient_self_unblock_proceeds_with_warning(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="transient", author="alice")
        ok, warning = kb.unblock_task(conn, tid, actor="alice")
        assert ok is True
        assert warning is not None
        assert "proceeded but is recorded" in warning
        assert kb.get_task(conn, tid).status == "ready"
        unblocked = [e for e in _events(conn, tid) if e["kind"] == "unblocked"]
        assert unblocked[-1]["payload"] == {"actor": "alice"}


def test_legacy_null_blocker_fails_open(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author=None)
        assert kb.get_task(conn, tid).blocked_by is None
        ok, err = kb.unblock_task(conn, tid, actor="alice")
        assert ok is True
        assert err is None
        # The audit trail must say the block predates identity recording.
        unblocked = [e for e in _events(conn, tid) if e["kind"] == "unblocked"]
        assert unblocked[-1]["payload"] == {"actor": "alice", "legacy_blocker": True}


def test_scheduled_unblock_not_affected_by_guard(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.schedule_task(conn, tid) is True
        ok, err = kb.unblock_task(conn, tid, actor="alice")
        assert ok is True
        assert err is None


def test_unblock_clears_blocked_by(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
        kb.unblock_task(conn, tid, actor="bob")
        assert kb.get_task(conn, tid).blocked_by is None


def test_complete_clears_blocked_by(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
        assert kb.complete_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.blocked_by is None


def test_promote_clears_blocked_by(kanban_home):
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
        ok, err = kb.promote_task(conn, tid, actor="bob")
        assert ok is True
        assert err is None
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.blocked_by is None


def test_blocked_event_still_has_no_actor_payload(kanban_home):
    """The blocked event payload is unchanged; identity lives in blocked_by."""
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="capability", author="alice")
        blocked = [e for e in _events(conn, tid) if e["kind"] == "blocked"]
        assert blocked[-1]["payload"] == {
            "reason": "review-required: please check",
            "kind": "capability",
            "recurrences": 1,
        }


# ---------------------------------------------------------------------------
# CLI layer: `hermes kanban unblock` + --override-self
# ---------------------------------------------------------------------------


def _unblock_ns(task_id, *, reason=None, override_self=False):
    return argparse.Namespace(
        task_ids=[task_id],
        reason=reason,
        override_self=override_self,
    )


def test_cli_self_unblock_refused_without_override(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "alice")
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
    rc = kb_cli._cmd_unblock(_unblock_ns(tid))
    assert rc == 1
    err = capsys.readouterr().err
    assert "override" in err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_cli_self_unblock_with_override_succeeds(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "alice")
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
    rc = kb_cli._cmd_unblock(_unblock_ns(tid, override_self=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert f"Unblocked {tid}" in out
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_cli_different_profile_unblocks_without_flag(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "bob")
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="needs_input", author="alice")
    rc = kb_cli._cmd_unblock(_unblock_ns(tid))
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_cli_transient_self_unblock_warns(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "alice")
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="transient", author="alice")
    rc = kb_cli._cmd_unblock(_unblock_ns(tid))
    assert rc == 0
    err = capsys.readouterr().err
    assert "warning" in err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Tool layer: kanban_unblock override_self
# ---------------------------------------------------------------------------


def test_tool_self_unblock_refused_without_override(kanban_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="capability", author="orchestrator")
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_unblock({"task_id": tid}))
    assert out.get("ok") is not True
    assert "override" in out.get("error", "")
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_tool_self_unblock_with_override_succeeds(kanban_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    with kb.connect() as conn:
        tid = _blocked_task(conn, kind="capability", author="orchestrator")
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_unblock({"task_id": tid, "override_self": True}))
    assert out["ok"] is True
    assert out["status"] == "ready"
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"
