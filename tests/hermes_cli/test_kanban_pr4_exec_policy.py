"""PR4 — execution policy, fallback, auto-chain, and operator_needed gate.

Covers the new exec_policy field on tasks and its effect on dispatch_once:

* ``auto``            — default: dispatcher spawns normally (no change to existing behavior).
* ``fallback_local``  — if the primary assignee is not a spawnable Hermes profile,
                        fall back to ``kanban.fallback_local_profile`` when configured.
* ``operator_needed`` — human gate: dispatcher skips the task, emits a rate-limited
                        ``human_gate`` event, and adds it to ``DispatchResult.operator_needed``.

Also covers the ``operator_needed_gate`` diagnostic rule.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Patch profile_exists so every assignee name is considered spawnable."""
    monkeypatch.setattr(
        "hermes_cli.kanban_db.profile_exists",
        lambda _name: True,
        raising=False,
    )


@pytest.fixture
def no_assignees_spawnable(monkeypatch):
    """Patch profile_exists so every assignee name is NOT spawnable."""
    monkeypatch.setattr(
        "hermes_cli.kanban_db.profile_exists",
        lambda _name: False,
        raising=False,
    )


# ---------------------------------------------------------------------------
# exec_policy field: persistence and validation
# ---------------------------------------------------------------------------

def test_exec_policy_default_is_auto(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="default policy")
        task = kb.get_task(conn, tid)
        assert task.exec_policy == "auto"
    finally:
        conn.close()


def test_exec_policy_persists_fallback_local(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="needs local fallback", exec_policy="fallback_local"
        )
        task = kb.get_task(conn, tid)
        assert task.exec_policy == "fallback_local"
    finally:
        conn.close()


def test_exec_policy_persists_operator_needed(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="human gate required", exec_policy="operator_needed"
        )
        task = kb.get_task(conn, tid)
        assert task.exec_policy == "operator_needed"
    finally:
        conn.close()


def test_exec_policy_rejects_invalid_value(kanban_home):
    conn = kb.connect()
    try:
        with pytest.raises(ValueError, match="exec_policy"):
            kb.create_task(conn, title="bad policy", exec_policy="remote_only")
    finally:
        conn.close()


def test_exec_policy_in_created_event(kanban_home):
    """The 'created' event payload includes exec_policy."""
    import json as _json
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="gate task", exec_policy="operator_needed"
        )
        events = kb.list_events(conn, tid)
        created_ev = next(e for e in events if e.kind == "created")
        payload = _json.loads(created_ev.payload) if isinstance(created_ev.payload, str) else (created_ev.payload or {})
        assert payload.get("exec_policy") == "operator_needed"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migration: legacy DB gains exec_policy column with default 'auto'
# ---------------------------------------------------------------------------

def test_exec_policy_migration_legacy_db(tmp_path, monkeypatch):
    """Opening a DB without exec_policy adds the column; existing rows default to 'auto'."""
    import sqlite3 as _sqlite3
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Build a minimal legacy DB without exec_policy.
    db_path = home / "kanban.db"
    raw = _sqlite3.connect(str(db_path))
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    raw.execute(
        "INSERT INTO tasks (id, title, created_at) VALUES ('t_legacy01', 'old task', 1)"
    )
    raw.commit()
    raw.close()

    # Opening via kb.connect() should run migration without error.
    conn = kb.connect()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "exec_policy" in cols

        # Legacy row reads back as 'auto'.
        row = conn.execute("SELECT exec_policy FROM tasks WHERE id = 't_legacy01'").fetchone()
        assert row["exec_policy"] == "auto"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# dispatch_once: operator_needed tasks are NOT dispatched
# ---------------------------------------------------------------------------

def test_operator_needed_not_dispatched(kanban_home):
    """Tasks with exec_policy='operator_needed' stay in ready; not spawned."""
    spawned_ids: list[str] = []

    def _stub_spawn(task, ws, **_kw):
        spawned_ids.append(task.id)
        return None

    conn = kb.connect()
    try:
        # Patch profile_exists so all profiles are "spawnable" — the gate
        # should fire regardless of profile availability.
        import unittest.mock as _mock
        with _mock.patch("hermes_cli.profiles.profile_exists", lambda _: True):
            tid = kb.create_task(
                conn,
                title="needs human decision",
                assignee="worker",
                exec_policy="operator_needed",
            )
            res = kb.dispatch_once(conn, spawn_fn=_stub_spawn)

        assert tid not in spawned_ids
        assert tid in res.operator_needed
        # Task must still be in ready (not claimed or blocked).
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
    finally:
        conn.close()


def test_operator_needed_emits_human_gate_event(kanban_home):
    """dispatch_once emits a human_gate event for operator_needed tasks."""
    import unittest.mock as _mock

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="gate task",
            assignee="worker",
            exec_policy="operator_needed",
        )
        with _mock.patch("hermes_cli.profiles.profile_exists", lambda _: True):
            kb.dispatch_once(conn, spawn_fn=lambda task, ws, **_: None)

        events = kb.list_events(conn, tid)
        gate_events = [e for e in events if e.kind == "human_gate"]
        assert len(gate_events) == 1
    finally:
        conn.close()


def test_operator_needed_human_gate_event_is_rate_limited(kanban_home):
    """A second dispatch tick within the throttle window emits no duplicate event."""
    import unittest.mock as _mock

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="gate task",
            assignee="worker",
            exec_policy="operator_needed",
        )
        stub = lambda task, ws, **_: None

        with _mock.patch("hermes_cli.profiles.profile_exists", lambda _: True):
            kb.dispatch_once(conn, spawn_fn=stub)
            kb.dispatch_once(conn, spawn_fn=stub)
            kb.dispatch_once(conn, spawn_fn=stub)

        events = kb.list_events(conn, tid)
        gate_events = [e for e in events if e.kind == "human_gate"]
        assert len(gate_events) == 1, "human_gate should be emitted only once per hour"
    finally:
        conn.close()


def test_operator_needed_does_not_suppress_auto_tasks(kanban_home):
    """operator_needed only gates its own task; other tasks in the queue dispatch normally."""
    import unittest.mock as _mock

    conn = kb.connect()
    try:
        gate_id = kb.create_task(
            conn, title="gate task", assignee="worker",
            exec_policy="operator_needed",
        )
        normal_id = kb.create_task(
            conn, title="normal task", assignee="worker",
            exec_policy="auto",
        )
        spawned: list[str] = []

        def _stub(task, ws, **_):
            spawned.append(task.id)
            return None

        with _mock.patch("hermes_cli.profiles.profile_exists", lambda _: True):
            res = kb.dispatch_once(conn, spawn_fn=_stub)

        assert gate_id not in spawned
        assert normal_id in spawned
        assert gate_id in res.operator_needed
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# dispatch_once: fallback_local
# ---------------------------------------------------------------------------

def test_fallback_local_uses_fallback_profile_when_primary_nonspawnable(
    kanban_home, monkeypatch
):
    """fallback_local task spawns with the fallback profile when primary is nonspawnable."""
    import unittest.mock as _mock

    # Primary assignee is NOT spawnable; fallback IS.
    def _profile_exists(name):
        return name == "local-worker"

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="remote task with local fallback",
            assignee="remote-worker",
            exec_policy="fallback_local",
        )

        spawned_as: list[str] = []

        def _stub(task, ws, **_):
            spawned_as.append(task.assignee)
            return None

        with (
            _mock.patch("hermes_cli.profiles.profile_exists", _profile_exists),
            _mock.patch(
                "hermes_cli.kanban_db._get_fallback_local_profile",
                return_value="local-worker",
            ),
        ):
            res = kb.dispatch_once(conn, spawn_fn=_stub)

        # Should have spawned via the fallback, not been nonspawnable-skipped.
        assert tid not in res.skipped_nonspawnable
        assert ("local-worker" in spawned_as) or (len(res.fallback_spawned) > 0)
        if res.fallback_spawned:
            assert res.fallback_spawned[0][0] == tid
            assert res.fallback_spawned[0][1] == "local-worker"
    finally:
        conn.close()


def test_fallback_local_falls_through_to_nonspawnable_when_no_fallback_profile(
    kanban_home,
):
    """fallback_local with no fallback profile configured behaves like auto (nonspawnable)."""
    import unittest.mock as _mock

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="orphaned fallback",
            assignee="remote-worker",
            exec_policy="fallback_local",
        )

        with (
            _mock.patch("hermes_cli.profiles.profile_exists", lambda _: False),
            _mock.patch(
                "hermes_cli.kanban_db._get_fallback_local_profile",
                return_value=None,
            ),
        ):
            res = kb.dispatch_once(conn, spawn_fn=lambda t, ws, **_: None)

        assert tid in res.skipped_nonspawnable
        assert not res.fallback_spawned
    finally:
        conn.close()


def test_fallback_local_auto_tasks_unchanged(kanban_home):
    """exec_policy='auto' tasks are not affected by fallback_local logic."""
    import unittest.mock as _mock

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="normal task",
            assignee="nonspawnable-lane",
            exec_policy="auto",
        )

        with (
            _mock.patch("hermes_cli.profiles.profile_exists", lambda _: False),
            _mock.patch(
                "hermes_cli.kanban_db._get_fallback_local_profile",
                return_value="local-worker",
            ),
        ):
            res = kb.dispatch_once(conn, spawn_fn=lambda t, ws, **_: None)

        # 'auto' policy still goes through the normal nonspawnable path.
        assert tid in res.skipped_nonspawnable
        assert not res.fallback_spawned
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _maybe_emit_human_gate_event: rate-limit helper
# ---------------------------------------------------------------------------

def test_maybe_emit_human_gate_event_emits_first_time(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="gate")
        emitted = kb._maybe_emit_human_gate_event(conn, tid)
        assert emitted is True
        events = kb.list_events(conn, tid)
        assert any(e.kind == "human_gate" for e in events)
    finally:
        conn.close()


def test_maybe_emit_human_gate_event_throttled_on_second_call(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="gate")
        first = kb._maybe_emit_human_gate_event(conn, tid)
        second = kb._maybe_emit_human_gate_event(conn, tid)
        assert first is True
        assert second is False
        events = kb.list_events(conn, tid)
        gate_events = [e for e in events if e.kind == "human_gate"]
        assert len(gate_events) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _get_task_exec_policy: helper
# ---------------------------------------------------------------------------

def test_get_task_exec_policy_unknown_id_returns_default(kanban_home):
    conn = kb.connect()
    try:
        policy = kb._get_task_exec_policy(conn, "t_nonexistent")
        assert policy == kb.DEFAULT_EXEC_POLICY
    finally:
        conn.close()


def test_get_task_exec_policy_reads_stored_value(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="gate", exec_policy="operator_needed"
        )
        assert kb._get_task_exec_policy(conn, tid) == "operator_needed"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Diagnostic rule: operator_needed_gate
# ---------------------------------------------------------------------------

def _make_task(status="ready", exec_policy="auto", task_id="t_test01", **extra):
    """Build a simple task dict for diagnostics tests."""
    return {
        "id": task_id,
        "title": "test task",
        "status": status,
        "exec_policy": exec_policy,
        "assignee": "worker",
        "claim_lock": None,
        "created_at": int(time.time()) - 3600,
        **extra,
    }


def test_diagnostic_operator_needed_gate_fires_when_ready(kanban_home):
    task = _make_task(status="ready", exec_policy="operator_needed")
    diags = kd.compute_task_diagnostics(task, events=[], runs=[], config={})
    kinds = [d.kind for d in diags]
    assert "operator_needed_gate" in kinds


def test_diagnostic_operator_needed_gate_severity_is_warning(kanban_home):
    task = _make_task(status="ready", exec_policy="operator_needed")
    diags = kd.compute_task_diagnostics(task, events=[], runs=[], config={})
    gate = next(d for d in diags if d.kind == "operator_needed_gate")
    assert gate.severity == "warning"


def test_diagnostic_operator_needed_gate_silent_for_auto_policy(kanban_home):
    task = _make_task(status="ready", exec_policy="auto")
    diags = kd.compute_task_diagnostics(task, events=[], runs=[], config={})
    assert not any(d.kind == "operator_needed_gate" for d in diags)


def test_diagnostic_operator_needed_gate_silent_when_not_ready(kanban_home):
    for status in ("todo", "running", "blocked", "done", "archived"):
        task = _make_task(status=status, exec_policy="operator_needed")
        diags = kd.compute_task_diagnostics(task, events=[], runs=[], config={})
        assert not any(d.kind == "operator_needed_gate" for d in diags), (
            f"gate should not fire for status={status!r}"
        )


def test_diagnostic_operator_needed_gate_has_unblock_action(kanban_home):
    task = _make_task(status="ready", exec_policy="operator_needed")
    diags = kd.compute_task_diagnostics(task, events=[], runs=[], config={})
    gate = next(d for d in diags if d.kind == "operator_needed_gate")
    action_kinds = [a.kind for a in gate.actions]
    assert "cli_hint" in action_kinds


def test_diagnostic_operator_needed_gate_has_gate_ts_from_event(kanban_home):
    """first_seen_at is derived from the human_gate event when present."""
    gate_ts = int(time.time()) - 7200
    events = [
        SimpleNamespace(kind="human_gate", payload={"policy": "operator_needed"}, created_at=gate_ts)
    ]
    task = _make_task(status="ready", exec_policy="operator_needed")
    diags = kd.compute_task_diagnostics(task, events=events, runs=[], config={})
    gate = next(d for d in diags if d.kind == "operator_needed_gate")
    assert gate.first_seen_at == gate_ts


# ---------------------------------------------------------------------------
# DispatchResult new fields
# ---------------------------------------------------------------------------

def test_dispatch_result_has_operator_needed_field():
    res = kb.DispatchResult()
    assert hasattr(res, "operator_needed")
    assert isinstance(res.operator_needed, list)


def test_dispatch_result_has_fallback_spawned_field():
    res = kb.DispatchResult()
    assert hasattr(res, "fallback_spawned")
    assert isinstance(res.fallback_spawned, list)


# ---------------------------------------------------------------------------
# VALID_EXEC_POLICIES constant
# ---------------------------------------------------------------------------

def test_valid_exec_policies_contains_expected_values():
    assert kb.VALID_EXEC_POLICIES == {"auto", "fallback_local", "operator_needed"}


def test_default_exec_policy_is_auto():
    assert kb.DEFAULT_EXEC_POLICY == "auto"
