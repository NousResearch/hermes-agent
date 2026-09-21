"""PROFILE-GATE tests (G1 core): approval primitive, fail-closed guard,
execute-on-approve resolver.

Covers the locked design:
  - autonomous profile create/delete blocks its task + records a pending row
  - the primitive guard fails CLOSED in a worker context without authorisation
  - approve executes the op once and closes the task; reject never runs it
  - double-resolve is refused (a double-click cannot run the op twice)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod
from hermes_cli import profile_lifecycle_gate as gate


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _new_task(conn) -> str:
    return kb.create_task(
        conn, title="lifecycle op", assignee="denji", initial_status="running"
    )


# --------------------------------------------------------------------------
# Approval primitive
# --------------------------------------------------------------------------

def test_request_blocks_task_and_records(kanban_home):
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = kb.request_profile_lifecycle_approval(
            conn, tid, op="delete", profile="coachsense-sub",
            requested_by="denji", blast_summary="DELETE coachsense-sub; 0 tasks",
        )
        assert aid.startswith("pla-")
        row = kb.get_profile_lifecycle_approval(conn, aid)
        assert row["status"] == "pending"
        assert row["op"] == "delete"
        assert row["token"]
        # task is blocked awaiting approval
        assert kb.get_task(conn, tid).status == "blocked"
        # surfaced to the bridge
        pend = kb.list_pending_profile_lifecycle_approvals(conn, undelivered_only=True)
        assert [p["id"] for p in pend] == [aid]


def test_approve_executes_create_with_no_clone_args(kanban_home, monkeypatch):
    """Regression: a create request with no clone_from stores no args (JSON
    null); the approve path must coerce that to {} so create_profile(**args)
    does not throw. This is the previously-unproven approve path."""
    called = {}

    def _fake_create(name, **kwargs):
        called["name"] = name
        called["kwargs"] = kwargs
        return Path("/tmp") / name

    monkeypatch.setattr(profiles_mod, "create_profile", _fake_create)
    # lifecycle_authorised is a context manager guarding the real op; keep it
    monkeypatch.setattr(profiles_mod, "lifecycle_authorised", lambda token: __import__("contextlib").nullcontext())

    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = gate.submit_lifecycle_request(
            conn, tid, op="create", profile="freshsub", args=None,
            requested_by="denji",
        )
        result = gate.approve(conn, aid, resolved_by="sahil")
        assert result["ok"] is True, result
        assert called["name"] == "freshsub"
        assert called["kwargs"] == {}
        assert kb.get_profile_lifecycle_approval(conn, aid)["status"] == "executed"


def test_request_atomic_no_orphan_when_block_fails(kanban_home):
    """If the task can't be blocked, the approval INSERT rolls back too:
    no approval row without a blocked task."""
    with kb.connect() as conn:
        tid = _new_task(conn)
        # Move it out of a blockable state.
        kb.complete_task(conn, tid, result="done")
        before = len(kb.list_pending_profile_lifecycle_approvals(conn))
        with pytest.raises(RuntimeError):
            kb.request_profile_lifecycle_approval(
                conn, tid, op="create", profile="x",
            )
        after = len(kb.list_pending_profile_lifecycle_approvals(conn))
        assert after == before  # no orphan approval row


def test_request_emits_blocked_telemetry(kanban_home, monkeypatch):
    """The approval request must emit the external kanban.task.blocked event
    (preserved through the atomic refactor) so dashboards see profile-gate
    blocks, not just the task_events row."""
    calls = []
    monkeypatch.setattr(
        kb, "record_event_if_enabled",
        lambda **kw: calls.append(kw),
    )
    with kb.connect() as conn:
        tid = _new_task(conn)
        kb.request_profile_lifecycle_approval(
            conn, tid, op="create", profile="newbot",
        )
    blocked = [c for c in calls if c.get("event_type") == "kanban.task.blocked"]
    assert len(blocked) == 1
    assert blocked[0]["object_id"] == tid


def test_request_rejects_bad_op(kanban_home):
    with kb.connect() as conn:
        tid = _new_task(conn)
        with pytest.raises(ValueError):
            kb.request_profile_lifecycle_approval(
                conn, tid, op="rename", profile="x"
            )


def test_double_resolve_refused(kanban_home):
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = kb.request_profile_lifecycle_approval(
            conn, tid, op="create", profile="newbot"
        )
        kb.resolve_profile_lifecycle_approval(
            conn, aid, decision="approve", resolved_by="sahil"
        )
        with pytest.raises(ValueError):
            kb.resolve_profile_lifecycle_approval(
                conn, aid, decision="reject", resolved_by="sahil"
            )


def test_notified_stamp_idempotent(kanban_home):
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = kb.request_profile_lifecycle_approval(
            conn, tid, op="create", profile="newbot"
        )
        kb.mark_profile_lifecycle_notified(conn, aid)
        assert kb.get_profile_lifecycle_approval(conn, aid)["notified_at"]
        # no longer undelivered
        assert kb.list_pending_profile_lifecycle_approvals(
            conn, undelivered_only=True
        ) == []


# --------------------------------------------------------------------------
# Fail-closed primitive guard
# --------------------------------------------------------------------------

def test_guard_allows_human_context(monkeypatch):
    # No worker marker snapshotted (the test process imported with it unset).
    monkeypatch.setattr(profiles_mod, "_WORKER_MARKER", None)
    # Should not raise.
    profiles_mod._assert_lifecycle_authorised("delete", "anything")


def test_guard_blocks_worker_context(monkeypatch):
    # The marker is snapshotted at import; tests patch the module attribute
    # (not os.environ, which the guard deliberately no longer reads live).
    monkeypatch.setattr(profiles_mod, "_WORKER_MARKER", "t_abc123")
    with pytest.raises(PermissionError):
        profiles_mod._assert_lifecycle_authorised("delete", "coachsense-sub")
    # Authorised context (resolver / test) is allowed.
    with profiles_mod.lifecycle_authorised("tok-xyz"):
        profiles_mod._assert_lifecycle_authorised("delete", "coachsense-sub")
    # Authorisation is scoped: it does not leak past the context manager.
    with pytest.raises(PermissionError):
        profiles_mod._assert_lifecycle_authorised("delete", "coachsense-sub")


def test_lifecycle_authorised_requires_token():
    with pytest.raises(ValueError):
        with profiles_mod.lifecycle_authorised(""):
            pass


def test_delete_profile_blocked_in_worker_context(kanban_home, monkeypatch):
    monkeypatch.setattr(profiles_mod, "_WORKER_MARKER", "t_abc123")
    with pytest.raises(PermissionError):
        profiles_mod.delete_profile("denji", yes=True)


# --------------------------------------------------------------------------
# Resolver (execute-on-approve)
# --------------------------------------------------------------------------

def test_approve_executes_and_closes(kanban_home, monkeypatch):
    calls = []
    monkeypatch.setattr(
        profiles_mod, "create_profile",
        lambda name, **kw: calls.append(("create", name, kw)) or Path("/tmp/x"),
    )
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = gate.submit_lifecycle_request(
            conn, tid, op="create", profile="newbot",
            args={"clone_from": "denji"}, requested_by="denji",
        )
        assert kb.get_task(conn, tid).status == "blocked"
        res = gate.approve(conn, aid, resolved_by="sahil")
        assert res["ok"] is True
        assert calls == [("create", "newbot", {"clone_from": "denji"})]
        assert kb.get_profile_lifecycle_approval(conn, aid)["status"] == "executed"
        assert kb.get_task(conn, tid).status in ("done", "archived")


def test_reject_does_not_execute(kanban_home, monkeypatch):
    calls = []
    monkeypatch.setattr(
        profiles_mod, "delete_profile",
        lambda name, **kw: calls.append(("delete", name, kw)),
    )
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = gate.submit_lifecycle_request(
            conn, tid, op="delete", profile="coachsense-sub", requested_by="denji",
        )
        res = gate.reject(conn, aid, resolved_by="sahil")
        assert res.get("rejected") is True
        assert calls == []
        assert kb.get_profile_lifecycle_approval(conn, aid)["status"] == "rejected"
        assert kb.get_task(conn, tid).status in ("done", "archived")


def test_cli_list_approve_reject(kanban_home, monkeypatch):
    import argparse
    from hermes_cli import kanban as kanban_cli

    created = []
    monkeypatch.setattr(
        profiles_mod, "create_profile",
        lambda name, **kw: created.append(name) or Path("/tmp/x"),
    )
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = gate.submit_lifecycle_request(
            conn, tid, op="create", profile="newbot", requested_by="denji",
        )
    # list shows it
    rc = kanban_cli._cmd_profile_gate(
        argparse.Namespace(pg_action="list", approval_id=None)
    )
    assert rc == 0
    # approve missing id is a usage error
    assert kanban_cli._cmd_profile_gate(
        argparse.Namespace(pg_action="approve", approval_id=None)
    ) == 2
    # approve executes
    rc = kanban_cli._cmd_profile_gate(
        argparse.Namespace(pg_action="approve", approval_id=aid)
    )
    assert rc == 0
    assert created == ["newbot"]
    # re-approve is refused (already resolved)
    assert kanban_cli._cmd_profile_gate(
        argparse.Namespace(pg_action="approve", approval_id=aid)
    ) == 1


def test_approve_op_failure_marks_failed(kanban_home, monkeypatch):
    def _boom(name, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(profiles_mod, "create_profile", _boom)
    with kb.connect() as conn:
        tid = _new_task(conn)
        aid = gate.submit_lifecycle_request(
            conn, tid, op="create", profile="newbot", requested_by="denji",
        )
        res = gate.approve(conn, aid, resolved_by="sahil")
        assert res["ok"] is False
        assert "disk full" in res["error"]
        assert kb.get_profile_lifecycle_approval(conn, aid)["status"] == "failed"
