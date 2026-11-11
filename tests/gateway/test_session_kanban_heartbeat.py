"""C3 (#47 Phase 2): the worker beats BOTH leases so a long turn cannot be taken over.

P0 under test: the dispatcher claims the card and the cross-session ledger with a
900 s TTL. A worker blocked in one long turn emits no tool-layer activity, so
nothing extended either lease; at 900 s the ledger went STALE, a second session
took the card over, and two workers executed the same work item.

Matrix:
  (a) a worker outliving the TTL keeps both leases and is NOT taken over;
  (b) a wrong Kanban claimer / wrong ledger owner (or lock) is refused;
  (c) a heartbeat DB failure never kills the worker.
"""
import json
import os
import time

from cross_session_ownership import (
    canonical_key_from_kanban,
    claim_ownership,
    heartbeat_ownership,
    init_schema,
    inspect_ownership,
)
from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


class _Agent:
    session_id = "s1"

    def __init__(self, results):
        self.results, self.calls = list(results), []
        self._session_db = self

    def get_messages_as_conversation(self, _sid):
        return [{"role": "user", "content": "from-store"}]

    def run_conversation(self, message, conversation_history=None, turn_author=None, task_id=None):
        self.calls.append((message, conversation_history))
        return self.results.pop(0)


def _claimed_context(tmp_path, monkeypatch, *, ttl=900, session_id="s1"):
    """A claimed card + a claimed ledger scope + the ``owner_admitted`` marker.

    Returns ``(path, context, task, scope)``. The scope is derived exactly as
    the admission gate derives it (card row -> canonical key), so the ledger row
    and the worker's later derivation agree.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title="hb", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="owner")
        assert task is not None
        init_schema(conn)
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        scope = canonical_key_from_kanban({c: row[c] for c in row.keys()})
        assert scope
        claimed = claim_ownership(conn, scope, session_id, ttl_seconds=ttl)
        assert claimed["ok"], claimed
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "owner_admitted",
                             {"session_id": session_id, "scope": scope},
                             run_id=task.current_run_id)
        context = {"db": str(path), "task_id": task_id, "run_id": task.current_run_id,
                   "claim_lock": task.claim_lock, "skills": [], "context": "",
                   "goal_mode": False, "goal_text": "", "goal_max_turns": 0}
    return path, context, task, scope


def _expire_all_leases(path, context, scope):
    """Simulate the worker having outlived the TTL with no beat at all."""
    with connect_closing(path) as conn:
        conn.execute("UPDATE tasks SET claim_expires=1 WHERE id=?", (context["task_id"],))
        conn.execute("UPDATE task_runs SET claim_expires=1 WHERE id=?", (context["run_id"],))
        conn.execute("UPDATE task_ownership SET claim_expires=1 WHERE scope_key=?", (scope,))


def _arm_ownership_gate(monkeypatch):
    """C5a staged activation: the ledger fence is OFF unless the flag is set.

    These heartbeat tests exercise the fence itself, so they must arm it
    explicitly — exactly as ``test_session_kanban_ownership._run_task`` and
    ``test_phase2_e2e`` do. Without this the ledger half of the beat is inert
    (``ownership_not_enforced``) and the fence is never under test.
    """
    monkeypatch.setenv("HERMES_KANBAN_OWNERSHIP_ENFORCE", "1")


# --- (a) a worker outliving the TTL keeps both leases and is NOT taken over -------

def test_heartbeat_extends_both_leases_before_expiry(tmp_path, monkeypatch):
    _arm_ownership_gate(monkeypatch)
    path, context, _task, scope = _claimed_context(tmp_path, monkeypatch, ttl=1)
    # The lease is still live but about to lapse (the worker is mid-turn).
    res = session_kanban._heartbeat_claims_once(context)

    assert res["kanban"] is True, res
    assert res["ledger"] is True, res
    with connect_closing(path) as conn:
        ownership = inspect_ownership(conn, scope)
        assert ownership["verdict"] == "OWNED"
        assert ownership["seconds_to_expiry"] > 800
        # The whole point: a second session cannot take over a beating worker.
        assert claim_ownership(conn, scope, "s2")["ok"] is False


def test_heartbeat_does_not_resurrect_an_expired_lease(tmp_path, monkeypatch):
    """Fail-closed: once the ledger lease has lapsed, a late beat must NOT renew it
    (the scope may already have been taken over by another session)."""
    _arm_ownership_gate(monkeypatch)
    path, context, _task, scope = _claimed_context(tmp_path, monkeypatch, ttl=1)
    _expire_all_leases(path, context, scope)
    res = session_kanban._heartbeat_claims_once(context)
    assert res["ledger"] is False, res
    assert res["ledger_reason"] == "lease_expired", res


def test_periodic_heartbeat_thread_keeps_lease_alive(tmp_path, monkeypatch):
    _arm_ownership_gate(monkeypatch)
    path, context, _task, scope = _claimed_context(tmp_path, monkeypatch, ttl=2)
    stop = session_kanban._start_claim_heartbeat(
        context, interval_seconds=0.05, ttl_seconds=2)
    try:
        time.sleep(2.5)  # well past the 2 s TTL, with no tool activity at all
        with connect_closing(path) as conn:
            ownership = inspect_ownership(conn, scope)
            assert ownership["verdict"] == "OWNED", ownership
            assert ownership["expires_at"] >= int(time.time()) + 1, ownership
            assert claim_ownership(conn, scope, "s2")["ok"] is False
    finally:
        stop.set()


def test_without_heartbeat_the_lease_goes_stale_and_is_taken_over(tmp_path, monkeypatch):
    """The failure mode this wiring exists to prevent (control case)."""
    path, context, _task, scope = _claimed_context(tmp_path, monkeypatch, ttl=1)
    _expire_all_leases(path, context, scope)
    with connect_closing(path) as conn:
        assert inspect_ownership(conn, scope)["verdict"] == "STALE"
        assert claim_ownership(conn, scope, "s2")["ok"] is True


# --- (b) a wrong claimer / owner is refused ---------------------------------------

def test_heartbeat_rejects_wrong_kanban_claimer(tmp_path, monkeypatch):
    path, context, _task, _scope = _claimed_context(tmp_path, monkeypatch)
    res = session_kanban._heartbeat_claims_once(dict(context, claim_lock="wrong-lock"))
    assert res["kanban"] is False, res
    with connect_closing(path) as conn:
        row = conn.execute("SELECT claim_expires FROM tasks WHERE id=?", (context["task_id"],)).fetchone()
        # The real claim was not extended by the wrong claimer.
        assert row[0] is not None and row[0] <= int(time.time()) + 900


def test_heartbeat_rejects_wrong_ledger_owner(tmp_path, monkeypatch):
    _arm_ownership_gate(monkeypatch)
    path, context, _task, scope = _claimed_context(tmp_path, monkeypatch)
    # No admission marker -> the helper falls back to the caller's owner, which is wrong.
    with connect_closing(path) as conn:
        conn.execute("DELETE FROM task_events WHERE kind='owner_admitted'")
    res = session_kanban._heartbeat_claims_once(context, owner_session="attacker")
    assert res["ledger"] is False, res
    assert res["ledger_reason"] == "not_owner:s1", res
    # The ledger's own CAS: wrong session AND wrong lock are both refused.
    with connect_closing(path) as conn:
        assert heartbeat_ownership(conn, scope, "attacker")["ok"] is False
        lock = conn.execute("SELECT claim_lock FROM task_ownership WHERE scope_key=?",
                            (scope,)).fetchone()["claim_lock"]
        assert heartbeat_ownership(conn, scope, "s1", expected_lock="wrong")["ok"] is False
        assert heartbeat_ownership(conn, scope, "s1", expected_lock=lock)["ok"] is True


# --- (c) a heartbeat DB failure never kills the worker ---------------------------

def test_heartbeat_never_raises_on_a_broken_db(tmp_path, monkeypatch):
    _path, context, _task, _scope = _claimed_context(tmp_path, monkeypatch)
    bad_db = tmp_path / "bad.db"
    bad_db.write_bytes(b"this is not a sqlite database")
    res = session_kanban._heartbeat_claims_once(dict(context, db=str(bad_db)))
    assert res["error"], res
    assert res["kanban"] is False and res["ledger"] is False


def test_worker_turn_survives_a_heartbeat_db_failure(tmp_path, monkeypatch):
    _path, context, _task, _scope = _claimed_context(tmp_path, monkeypatch)
    monkeypatch.setattr("agent.skill_commands.build_preloaded_skills_prompt",
                        lambda *a, **k: ("", [], []))

    def boom(*a, **k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr("cross_session_ownership.heartbeat_ownership", boom)
    frame = {"text": "work", "policy": {"kanban_json": json.dumps(context)}}
    agent = _Agent([{"completed": True, "final_response": "done"}])

    result = session_kanban.run_worker_turns(agent, frame, [])

    assert result["final_response"] == "done"
    assert len(agent.calls) == 1
