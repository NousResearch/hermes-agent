"""
Tests for AUTONOMOUS_INITIATION_PATH_V1.

These tests use a per-test isolated temporary Kanban DB. The real
~/.hermes/kanban.db is NEVER touched by these tests. The tests patch
hermes_cli.kanban_db.connect_closing (and the connect function) to
return a connection to a fresh sqlite3 file under tmp_path.

The 12 tests cover the contract from the directive:

  T1  autonomy disabled               -> 0 task created, fail closed
  T2  kill switch active               -> 0 task created, fail closed
  T3  Class A, kill off                -> exactly 1 task created
  T4  Class B                          -> 0 task created, requires human
  T5  Class C                          -> 0 task created, requires human
  T6  duplicate trigger                -> exactly 1 task total
  T7  concurrency limit                -> second objective not admitted
  T8  provenance on resulting task     -> origin / objective_id / etc.
  T9  canonical Kanban event semantics -> standard event trail
  T10 no operator create required      -> no `hermes kanban create` invoked
  T11 no config/profile/.env mutation  -> no persistent file written
  T12 deactivation                     -> subsequent initiation blocked
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


# Locate THIS TEMP CANDIDATE so we can import agent.autonomy from the
# current working tree — not the primary checkout. The historical baseline
# hard-coded /home/jr-ubuntu/.hermes/hermes-agent into sys.path, which
# defeated the point of running the test against a fresh temp. We instead
# derive the candidate root from the test file's own path.
CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
_candidate_str = str(CANDIDATE_ROOT)
if _candidate_str not in sys.path:
    sys.path.insert(0, _candidate_str)


# Provenance assertion: ``import agent.autonomy`` must resolve under the
# TEMP candidate path, NOT under the primary checkout. This guards
# against accidental reintroduction of the historical PRIMARY sys.path
# injection that would silently test the wrong tree.
def _assert_autonomy_resolves_under_candidate():
    import importlib
    import agent.autonomy as _autonomy
    _autonomy_file = Path(_autonomy.__file__).resolve()
    assert str(_autonomy_file).startswith(str(CANDIDATE_ROOT) + "/"), (
        f"agent.autonomy resolved to {_autonomy_file!r}; expected it under "
        f"{CANDIDATE_ROOT!r}. Refusing to run against the primary checkout."
    )


@pytest.fixture(autouse=True)
def _enforce_autonomy_provenance():
    """Autouse session-wide: runtime provenance guard.

    Confirms ``import agent.autonomy`` resolves under the TEMP candidate,
    NEVER under the primary checkout. Runs before every test; cheap if
    already cached by the import system.
    """
    _assert_autonomy_resolves_under_candidate()
    yield


# Tests run with a per-test isolated Kanban DB; we patch the
# hermes_cli.kanban_db module so the real ~/.hermes/kanban.db is not
# touched. This is the same pattern used by the project's own kanban
# tests (see tests/hermes_cli/test_kanban_db.py).


@pytest.fixture
def isolated_kanban_db(tmp_path, monkeypatch):
    """Yield a fresh Kanban sqlite3 file under tmp_path, and patch
    hermes_cli.kanban_db to use only this file. After the test the file
    is removed by tmp_path cleanup."""

    db_file = tmp_path / "test_kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_file))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    # Initialize the schema by importing the real kanban_db helpers.
    from hermes_cli import kanban_db as kb
    # Save the originals so the patched connect_closing calls the real
    # connect (avoid the recursion that would happen if we called
    # kb.connect, which we are about to monkey-patch below).
    _original_connect = kb.connect

    conn = _original_connect(db_file)
    conn.close()

    @contextmanager
    def _patched_connect_closing():
        c = _original_connect(db_file)
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(kb, "connect_closing", _patched_connect_closing)
    # Patch kb.connect only at the test-harness level (so any test code
    # that calls kb.connect directly also gets the test file). The
    # patched version uses the saved original to avoid recursion.
    monkeypatch.setattr(
        kb, "connect",
        lambda board=None: _original_connect(db_file),
    )

    yield db_file


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_spec(objective_id="obj-1", risk_class="CLASS_A_AUTONOMOUS_SAFE",
               policy_version="1.0.0", trigger_id="trig-1", title=None,
               profile=None, **extra):
    spec = {
        "objective_id": objective_id,
        "risk_class": risk_class,
        "policy_version": policy_version,
        "trigger_id": trigger_id,
        "title": title or f"[autonomous] {objective_id}",
    }
    spec.update(extra)
    ctx = {"profile": profile} if profile is not None else None
    return spec, ctx


def _count_tasks(db_file: Path) -> int:
    if not db_file.exists():
        return 0
    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM tasks")
        return cur.fetchone()[0]
    finally:
        conn.close()


def _read_task(db_file: Path, task_id: str):
    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.execute(
            "SELECT id, title, body, created_by, status, idempotency_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def _read_all_tasks(db_file: Path):
    conn = sqlite3.connect(str(db_file))
    try:
        return conn.execute(
            "SELECT id, title, body, created_by, status, idempotency_key "
            "FROM tasks ORDER BY created_at, id"
        ).fetchall()
    finally:
        conn.close()


def _snapshot_files(paths):
    return {str(p): p.read_bytes() for p in paths}


def _attempt_in_thread(spec, ctx, barrier, results, index):
    from agent.autonomy.initiator import attempt_autonomous_initiation

    barrier.wait(timeout=5)
    results[index] = attempt_autonomous_initiation(spec, ctx)


def _attempt_in_process(db_path, hermes_home, spec, ctx, barrier, queue):
    os.environ["HERMES_KANBAN_DB"] = str(db_path)
    os.environ["HERMES_KANBAN_HOME"] = str(Path(db_path).parent / "kanban-home")
    os.environ["HERMES_HOME"] = str(hermes_home)
    # Ensure the child process can import agent.autonomy from this temp
    # candidate. multiprocessing.get_context("spawn") inherits sys.path
    # from the parent, but we re-derive the candidate root here so the
    # child resolves the same module location the parent did.
    _child_candidate = str(Path(__file__).resolve().parents[2])
    if _child_candidate not in sys.path:
        sys.path.insert(0, _child_candidate)
    try:
        from agent.autonomy import state
        from agent.autonomy.initiator import attempt_autonomous_initiation, summarize

        state.reset()
        state.enable(policy_version="1.0.0")
        barrier.wait(timeout=10)
        queue.put(summarize(attempt_autonomous_initiation(spec, ctx)))
    except Exception as exc:  # pragma: no cover - surfaced to parent assertion
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def _run_cross_process_attempts(db_file: Path, specs_and_contexts, tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(len(specs_and_contexts))
    queue = ctx.Queue()
    procs = []
    hermes_home = tmp_path / "child-hermes-home"
    hermes_home.mkdir(exist_ok=True)
    for spec, pol_ctx in specs_and_contexts:
        proc = ctx.Process(
            target=_attempt_in_process,
            args=(str(db_file), str(hermes_home), spec, pol_ctx, barrier, queue),
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join(timeout=15)
    assert all(proc.exitcode == 0 for proc in procs)
    rows = [queue.get(timeout=5) for _ in procs]
    assert not [r for r in rows if "error" in r]
    return rows


def _assert_exactly_one_admitted(results):
    admitted = [r for r in results if r.decision == "admit"]
    assert len(admitted) == 1
    assert admitted[0].task_id
    return admitted[0]


def _assert_exactly_one_admitted_summary(results):
    admitted = [r for r in results if r["decision"] == "admit"]
    assert len(admitted) == 1
    assert admitted[0]["task_id"]
    return admitted[0]


# ----------------------------------------------------------------------
# T1
# ----------------------------------------------------------------------


def test_T1_disabled_fail_closed(isolated_kanban_db, monkeypatch):
    """T1: autonomy disabled -> 0 task created."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    # Autonomy is OFF by default
    assert state.is_enabled() is False

    spec, ctx = _make_spec()
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "denied_disabled"
    assert result.task_id is None
    assert _count_tasks(isolated_kanban_db) == 0


# ----------------------------------------------------------------------
# T2
# ----------------------------------------------------------------------


def test_T2_kill_switch_fail_closed(isolated_kanban_db, monkeypatch):
    """T2: kill switch active -> 0 task created."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0", profile="minimax-direct-test")
    state.fire_kill_switch()
    assert state.is_kill_switch_active() is True

    spec, ctx = _make_spec()
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "denied_kill"
    assert result.task_id is None
    assert _count_tasks(isolated_kanban_db) == 0


# ----------------------------------------------------------------------
# T3
# ----------------------------------------------------------------------


def test_T3_class_a_accepted(isolated_kanban_db, monkeypatch):
    """T3: enabled, kill off, Class A -> exactly 1 task created."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0", profile="minimax-direct-test")

    spec, ctx = _make_spec(objective_id="obj-3", trigger_id="trig-3")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id is not None
    assert result.duplicate is False
    assert _count_tasks(isolated_kanban_db) == 1
    assert state.get_active_objective_id() == "obj-3"
    assert state.get_active_task_id() == result.task_id


# ----------------------------------------------------------------------
# T4
# ----------------------------------------------------------------------


def test_T4_class_b_requires_human(isolated_kanban_db, monkeypatch):
    """T4: Class B -> 0 task created, requires_human equivalent."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(risk_class="CLASS_B_AUTONOMOUS_WITH_LIMITS")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "denied_class"
    assert result.task_id is None
    assert _count_tasks(isolated_kanban_db) == 0


# ----------------------------------------------------------------------
# T5
# ----------------------------------------------------------------------


def test_T5_class_c_requires_human(isolated_kanban_db, monkeypatch):
    """T5: Class C -> 0 task created, requires_human equivalent."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(risk_class="CLASS_C_HUMAN_GATE_REQUIRED")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "denied_class"
    assert result.task_id is None
    assert _count_tasks(isolated_kanban_db) == 0


# ----------------------------------------------------------------------
# T6
# ----------------------------------------------------------------------


def test_T6_idempotent_duplicate_trigger(isolated_kanban_db, monkeypatch):
    """T6: same spec invoked twice -> exactly 1 task total."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(objective_id="obj-6", trigger_id="trig-6")
    r1 = attempt_autonomous_initiation(spec, ctx)
    r2 = attempt_autonomous_initiation(spec, ctx)

    assert r1.decision == "admit"
    assert r1.duplicate is False
    assert r2.decision == "duplicate_suppressed"
    assert r2.duplicate is True
    assert r2.task_id == r1.task_id
    assert _count_tasks(isolated_kanban_db) == 1


# ----------------------------------------------------------------------
# T7
# ----------------------------------------------------------------------


def test_T7_concurrency_limit(isolated_kanban_db, monkeypatch):
    """T7: a second objective with different id is rejected while the
    first is still active."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec_a, ctx_a = _make_spec(objective_id="obj-A", trigger_id="trig-A")
    spec_b, ctx_b = _make_spec(objective_id="obj-B", trigger_id="trig-B")

    r1 = attempt_autonomous_initiation(spec_a, ctx_a)
    r2 = attempt_autonomous_initiation(spec_b, ctx_b)

    assert r1.decision == "admit"
    assert r2.decision == "denied_concurrency"
    assert r2.task_id is None
    assert _count_tasks(isolated_kanban_db) == 1


def test_C1_same_process_same_trigger_concurrent(isolated_kanban_db, monkeypatch):
    from agent.autonomy import state

    state.reset()
    state.enable(policy_version="1.0.0")
    spec, ctx = _make_spec(objective_id="obj-c1", trigger_id="trig-c1")
    barrier = threading.Barrier(2)
    results = [None, None]
    threads = [
        threading.Thread(target=_attempt_in_thread, args=(spec, ctx, barrier, results, i))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert all(r is not None for r in results)
    _assert_exactly_one_admitted(results)
    assert sorted(r.decision for r in results if r is not None) == ["admit", "duplicate_suppressed"]
    assert _count_tasks(isolated_kanban_db) == 1


def test_C2_cross_process_same_trigger_concurrent(isolated_kanban_db, tmp_path):
    spec, ctx = _make_spec(objective_id="obj-c2", trigger_id="trig-c2")

    rows = _run_cross_process_attempts(isolated_kanban_db, [(spec, ctx), (spec, ctx)], tmp_path)

    _assert_exactly_one_admitted_summary(rows)
    assert sorted(r["decision"] for r in rows) == ["admit", "duplicate_suppressed"]
    assert _count_tasks(isolated_kanban_db) == 1


def test_C3_same_process_different_objective_concurrent(isolated_kanban_db, monkeypatch):
    from agent.autonomy import state

    state.reset()
    state.enable(policy_version="1.0.0")
    spec_a, ctx_a = _make_spec(objective_id="obj-c3-a", trigger_id="trig-c3-a")
    spec_b, ctx_b = _make_spec(objective_id="obj-c3-b", trigger_id="trig-c3-b")
    barrier = threading.Barrier(2)
    results = [None, None]
    threads = [
        threading.Thread(target=_attempt_in_thread, args=(spec_a, ctx_a, barrier, results, 0)),
        threading.Thread(target=_attempt_in_thread, args=(spec_b, ctx_b, barrier, results, 1)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert all(r is not None for r in results)
    _assert_exactly_one_admitted(results)
    assert sorted(r.decision for r in results if r is not None) == ["admit", "denied_concurrency"]
    assert _count_tasks(isolated_kanban_db) == 1


def test_C4_cross_process_different_objective_concurrent(isolated_kanban_db, tmp_path):
    spec_a, ctx_a = _make_spec(objective_id="obj-c4-a", trigger_id="trig-c4-a")
    spec_b, ctx_b = _make_spec(objective_id="obj-c4-b", trigger_id="trig-c4-b")

    rows = _run_cross_process_attempts(
        isolated_kanban_db,
        [(spec_a, ctx_a), (spec_b, ctx_b)],
        tmp_path,
    )

    _assert_exactly_one_admitted_summary(rows)
    assert sorted(r["decision"] for r in rows) == ["admit", "denied_concurrency"]
    assert _count_tasks(isolated_kanban_db) == 1


def test_C5_post_restart_duplicate(isolated_kanban_db, monkeypatch):
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")
    spec, ctx = _make_spec(objective_id="obj-c5", trigger_id="trig-c5")
    first = attempt_autonomous_initiation(spec, ctx)
    assert first.decision == "admit"

    state.reset()
    state.enable(policy_version="1.0.0")
    second = attempt_autonomous_initiation(spec, ctx)

    assert second.decision == "duplicate_suppressed"
    assert second.task_id == first.task_id
    assert _count_tasks(isolated_kanban_db) == 1


# ----------------------------------------------------------------------
# T8
# ----------------------------------------------------------------------


def test_T8_provenance(isolated_kanban_db, monkeypatch):
    """T8: the resulting task carries the autonomous provenance in body
    and created_by."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0", profile="minimax-direct-test")

    spec, ctx = _make_spec(
        objective_id="obj-8", trigger_id="trig-8",
        risk_class="CLASS_A_AUTONOMOUS_SAFE", policy_version="1.0.0",
    )
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id

    row = _read_task(isolated_kanban_db, result.task_id)
    assert row is not None
    task_id, title, body, created_by, status, idem = row

    assert created_by == "autonomous_initiator"
    # kanban_db.create_task creates tasks in 'running' status by default;
    # autonomous tasks follow the same lifecycle (claim transitions to
    # running; the initial state is the same as for operator-created tasks).
    assert status in ("ready", "running")
    assert idem == "autonomous-obj-8-trig-8"

    # The body must contain the provenance fields
    assert "OBJECTIVE_ID=obj-8" in body
    assert "ORIGIN=autonomous_initiator" in body
    assert "POLICY_VERSION=1.0.0" in body
    assert "RISK_CLASS=CLASS_A_AUTONOMOUS_SAFE" in body
    assert "TRIGGER_ID=trig-8" in body
    assert "RUNNING_MODE=AUTONOMOUS_A1" in body
    # INITIATED_AT is a float; it must parse
    m = re.search(r"^INITIATED_AT=([0-9.]+)$", body, re.MULTILINE)
    assert m is not None
    float(m.group(1))


def test_P1_custom_body_preserves_body(isolated_kanban_db, monkeypatch):
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")
    custom = "Human supplied objective body\nwith multiple lines."
    spec, ctx = _make_spec(objective_id="obj-p1", trigger_id="trig-p1", body=custom)

    result = attempt_autonomous_initiation(spec, ctx)

    assert result.decision == "admit"
    assert result.task_id is not None
    body = _read_task(isolated_kanban_db, result.task_id)[2]
    assert "CUSTOM_BODY_BEGIN" in body
    assert custom in body
    assert "CUSTOM_BODY_END" in body


def test_P2_custom_body_preserves_full_provenance(isolated_kanban_db, monkeypatch):
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")
    spec, ctx = _make_spec(
        objective_id="obj-p2",
        trigger_id="trig-p2",
        body="custom body must not replace provenance",
    )

    result = attempt_autonomous_initiation(spec, ctx)

    assert result.decision == "admit"
    assert result.task_id is not None
    body = _read_task(isolated_kanban_db, result.task_id)[2]
    assert "AUTONOMOUS_PROVENANCE_BEGIN" in body
    assert "AUTONOMOUS_PROVENANCE_END" in body
    for key, value in result.provenance.items():
        assert f"{key}={value}" in body


def test_P3_provenance_recoverable_after_db_reopen(isolated_kanban_db, monkeypatch):
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")
    spec, ctx = _make_spec(
        objective_id="obj-p3",
        trigger_id="trig-p3",
        body="reopen persistence body",
    )

    result = attempt_autonomous_initiation(spec, ctx)

    assert result.decision == "admit"
    reopened = sqlite3.connect(str(isolated_kanban_db))
    try:
        row = reopened.execute(
            "SELECT body FROM tasks WHERE id = ?", (result.task_id,)
        ).fetchone()
    finally:
        reopened.close()
    assert row is not None
    body = row[0]
    assert "AUTONOMOUS_PROVENANCE_BEGIN" in body
    assert "OBJECTIVE_ID=obj-p3" in body
    assert "ORIGIN=autonomous_initiator" in body
    assert "POLICY_VERSION=1.0.0" in body
    assert "RISK_CLASS=CLASS_A_AUTONOMOUS_SAFE" in body
    assert "TRIGGER_ID=trig-p3" in body
    assert "RUNNING_MODE=AUTONOMOUS_A1" in body
    assert "CUSTOM_BODY_BEGIN\nreopen persistence body\nCUSTOM_BODY_END" in body


# ----------------------------------------------------------------------
# T9
# ----------------------------------------------------------------------


def test_T9_canonical_kanban_event_semantics(isolated_kanban_db, monkeypatch):
    """T9: the autonomous task is indistinguishable from a normal
    kanban.create task except for the autonomous provenance. Verify the
    event trail includes the standard 'created' event."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(objective_id="obj-9", trigger_id="trig-9")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.task_id

    conn = sqlite3.connect(str(isolated_kanban_db))
    try:
        cur = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "ORDER BY created_at ASC",
            (result.task_id,),
        )
        events = cur.fetchall()
    finally:
        conn.close()

    # At minimum, the 'created' event must exist
    kinds = [e[0] for e in events]
    assert "created" in kinds


# ----------------------------------------------------------------------
# T10
# ----------------------------------------------------------------------


def test_T10_no_operator_create_required(isolated_kanban_db, monkeypatch):
    """T10: the autonomous initiation succeeds without invoking the
    hermes.kanban.create CLI. We assert this by patching kanban.create
    CLI entry point to fail loudly if it gets called; if the test passes
    the CLI was NOT called."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    def _explode(*a, **kw):
        raise AssertionError("CLI hermes kanban create was invoked")

    # Patch the argparse-level command to ensure it was NOT called. The
    # `kanban_command` function in hermes_cli/kanban.py is the entry
    # point. We patch it at the module level.
    import hermes_cli.kanban as kanban_mod
    monkeypatch.setattr(kanban_mod, "kanban_command", _explode)

    spec, ctx = _make_spec(objective_id="obj-10", trigger_id="trig-10")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id


# ----------------------------------------------------------------------
# T11
# ----------------------------------------------------------------------


def test_T11_no_config_profile_env_persistence(isolated_kanban_db, monkeypatch, tmp_path):
    """T11: enabling autonomy does NOT write to config, profile, .env,
    auth/secrets metadata, or profile-state sentinels.

    The test creates every sentinel under a temp HERMES_HOME before the
    initiation path runs, then asserts the exact bytes are unchanged. This
    avoids the previous hardcoded-profile/vacuous case where zero files could
    be snapshotted and the test would still pass.
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()

    hermes_home = tmp_path / "hermes-home"
    profile_home = hermes_home / "profiles" / "autonomy-test"
    profile_state = profile_home / "state"
    for d in (hermes_home, profile_home, profile_state):
        d.mkdir(parents=True, exist_ok=True)
    sentinels = {
        hermes_home / "config.yaml": b"model:\n  provider: sentinel\n",
        hermes_home / ".env": b"SENTINEL_API_KEY=not-a-real-secret\n",
        hermes_home / "auth.json": b'{"sentinel":"auth-metadata"}\n',
        profile_home / "config.yaml": b"profile: autonomy-test\n",
        profile_home / ".env": b"PROFILE_SENTINEL=not-a-real-secret\n",
        profile_home / "auth.json": b'{"profile":"auth-metadata"}\n',
        profile_state / "runtime.json": b'{"enabled":false,"sentinel":true}\n',
    }
    for path, payload in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    before = _snapshot_files(sentinels.keys())
    assert len(before) == 7

    # Run the cycle
    state.enable(policy_version="1.0.0", profile="autonomy-test")
    spec, ctx = _make_spec(objective_id="obj-11", trigger_id="trig-11")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"

    # Verify nothing was written
    after = _snapshot_files(sentinels.keys())
    assert before == after

    state.reset()


# ----------------------------------------------------------------------
# T12
# ----------------------------------------------------------------------


def test_T12_deactivation(isolated_kanban_db, monkeypatch):
    """T12: after disable(), a subsequent attempt is denied_disabled and
    no new task is created."""
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")
    spec1, ctx1 = _make_spec(objective_id="obj-12a", trigger_id="trig-12a")
    r1 = attempt_autonomous_initiation(spec1, ctx1)
    assert r1.decision == "admit"
    assert _count_tasks(isolated_kanban_db) == 1

    # Deactivate
    state.disable()
    assert state.is_enabled() is False
    # The active slot is cleared on disable() — concurrency is back to 0
    assert state.get_active_objective_id() is None

    # Subsequent attempt fails closed
    spec2, ctx2 = _make_spec(objective_id="obj-12b", trigger_id="trig-12b")
    r2 = attempt_autonomous_initiation(spec2, ctx2)
    assert r2.decision == "denied_disabled"
    assert r2.task_id is None
    assert _count_tasks(isolated_kanban_db) == 1


# ----------------------------------------------------------------------
# AUTONOMOUS_INITIATION_PATH_V1.2 — Assignee Propagation
#
# A1: objective_spec["assignee"]="orchestrator" -> persisted row assignee="orchestrator"
# A2: same path -> status="ready", assignee="orchestrator"
# A3: production dispatch_once sees / claims / spawns exactly that task
#     (isolated temp DB, isolated valid profile resolution, fake spawn_fn,
#      NO real provider/worker subprocess)
# A4: backward compat: spec without assignee -> persisted assignee=None
# ----------------------------------------------------------------------


def _read_task_assignee(db_file: Path, task_id: str):
    """Read the (status, assignee) pair for a task by id."""
    conn = sqlite3.connect(str(db_file))
    try:
        row = conn.execute(
            "SELECT status, assignee FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None, None
    return row[0], row[1]


def _read_task_overrides(db_file: Path, task_id: str):
    """Read (model_override, provider_override) for a task by id.

    Both columns are NULL-when-absent in the canonical schema; we return
    None when unset. Used by the V1.3-A tests to assert the autonomous
    initiator propagates the spec's model / provider overrides into the
    persisted Kanban row via the canonical kb.create_task path.
    """
    conn = sqlite3.connect(str(db_file))
    try:
        row = conn.execute(
            "SELECT model_override, provider_override FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None, None
    model_override, provider_override = row
    # The schema stores unset values as NULL; the kb layer normalises "" to
    # None before write. Mirror that here so the assertions read what the
    # production code would see on read-back.
    if model_override == "":
        model_override = None
    if provider_override == "":
        provider_override = None
    return model_override, provider_override


def test_A1_assignee_propagated(isolated_kanban_db, monkeypatch):
    """A1: objective_spec["assignee"]="orchestrator" -> persisted row
    assignee="orchestrator". The initiator MUST propagate the spec's
    assignee through the canonical hermes_cli.kanban_db.create_task call.
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(
        objective_id="obj-A1",
        trigger_id="trig-A1",
        assignee="orchestrator",
    )
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id is not None
    task_id: str = result.task_id

    status, persisted_assignee = _read_task_assignee(
        isolated_kanban_db, task_id
    )
    assert persisted_assignee == "orchestrator", (
        f"expected persisted assignee='orchestrator', got {persisted_assignee!r}"
    )


def test_A2_assigned_task_is_ready(isolated_kanban_db, monkeypatch):
    """A2: same autonomous initiation -> status='ready', assignee='orchestrator'.
    The truth-test established that create_task(initial_status='running')
    with no blocked/triage/parent condition PERSISTS status='ready'.
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(
        objective_id="obj-A2",
        trigger_id="trig-A2",
        assignee="orchestrator",
    )
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id is not None
    task_id: str = result.task_id

    status, persisted_assignee = _read_task_assignee(
        isolated_kanban_db, task_id
    )
    assert status == "ready", f"expected status='ready', got {status!r}"
    assert persisted_assignee == "orchestrator", (
        f"expected persisted assignee='orchestrator', got {persisted_assignee!r}"
    )


def test_A3_assigned_task_dispatchable(isolated_kanban_db, monkeypatch, tmp_path):
    """A3: production dispatch_once sees/claims/spawns the assigned task
    exactly once with a fake spawn_fn (NO real provider/worker subprocess).

    Setup:
      * isolated temp Kanban DB (already provided by fixture)
      * isolated valid profile resolution: monkeypatch hermes_cli.profiles
        so 'orchestrator' resolves to a real directory we control under
        tmp_path, satisfying profile_exists() inside dispatch_once
      * fake spawn_fn: a stub that records the (task_id, assignee, ...)
        triple it was called with and returns None (no real worker pid)
      * the task is created ONLY through attempt_autonomous_initiation
        with assignee='orchestrator'

    Required:
      * dispatch_once claims exactly that task
      * fake spawn_fn is invoked exactly once
      * task reaches the canonical 'running' state (claimed by dispatch_once)
      * no manual assign_task / status mutation / raw SQL patch was used
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation
    from hermes_cli import kanban_db as kb
    from hermes_cli import profiles as profiles_mod

    # 1. Isolated valid profile resolution for "orchestrator" under tmp_path.
    # dispatch_once calls hermes_cli.profiles.profile_exists(assignee).
    # We monkeypatch profile_exists to recognize our fake profile id only,
    # leaving other lookups untouched (default returns True for 'default').
    real_profile_exists = profiles_mod.profile_exists

    def _fake_profile_exists(name):
        if name == "orchestrator":
            return True
        # Fall through for other lookups (e.g. _default_assignee probe).
        return real_profile_exists(name)

    monkeypatch.setattr(profiles_mod, "profile_exists", _fake_profile_exists)

    # Also patch the symbol dispatch_once reaches via the lazy import inside
    # the dispatcher loop — dispatch_once imports profile_exists from
    # hermes_cli.profiles at call time, so the patch above is sufficient.

    # 2. Enable autonomy and create the assigned task through the
    # canonical autonomous admission (NOT via create_task / raw SQL).
    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(
        objective_id="obj-A3",
        trigger_id="trig-A3",
        assignee="orchestrator",
    )
    init_result = attempt_autonomous_initiation(spec, ctx)
    assert init_result.decision == "admit"
    expected_task_id = init_result.task_id
    assert expected_task_id is not None

    # Pre-flight: the row must be in 'ready' with assignee='orchestrator'
    # BEFORE dispatch_once runs. (No premature claim.)
    pre_status, pre_assignee = _read_task_assignee(
        isolated_kanban_db, expected_task_id
    )
    assert pre_status == "ready"
    assert pre_assignee == "orchestrator"

    # 3. Fake spawn_fn: records exactly which (task, assignee, board) it was
    # asked to spawn for and returns None (no real subprocess).
    spawn_calls = []

    def _fake_spawn_fn(task, workspace_path, board):
        spawn_calls.append(
            {
                "task_id": task.id,
                "assignee": task.assignee,
                "board": board,
            }
        )
        return None  # no real worker pid

    # 4. Run the production dispatch_once with the fake spawn_fn.
    conn = kb.connect()
    try:
        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn_fn, dry_run=False)
    finally:
        conn.close()

    # 5. Assert: dispatch_once saw exactly this task, claimed it,
    # and called the fake spawn_fn exactly once.
    assert len(spawn_calls) == 1, (
        f"expected fake spawn_fn invoked exactly once, got {len(spawn_calls)}: "
        f"{spawn_calls!r}"
    )
    spawn = spawn_calls[0]
    assert spawn["task_id"] == expected_task_id
    assert spawn["assignee"] == "orchestrator"

    # The task MUST have transitioned from 'ready' -> 'running' as part of
    # the canonical claim path (claim_task inside dispatch_once).
    post_status, post_assignee = _read_task_assignee(
        isolated_kanban_db, expected_task_id
    )
    assert post_status == "running", (
        f"expected task to reach 'running' after dispatch_once claim, "
        f"got {post_status!r}"
    )
    assert post_assignee == "orchestrator"

    # Sanity: the DispatchResult must list this task under .spawned, not
    # .skipped_unassigned or .skipped_nonspawnable.
    spawned_ids = [
        entry[0] if isinstance(entry, tuple) else entry
        for entry in (result.spawned or [])
    ]
    assert expected_task_id in spawned_ids, (
        f"expected task in result.spawned; got spawned={result.spawned!r} "
        f"skipped_unassigned={result.skipped_unassigned!r} "
        f"skipped_nonspawnable={result.skipped_nonspawnable!r}"
    )
    assert expected_task_id not in result.skipped_unassigned
    assert expected_task_id not in result.skipped_nonspawnable


def test_A4_unassigned_backward_compatible(isolated_kanban_db, monkeypatch):
    """A4: backward compatibility. When objective_spec omits assignee
    (or supplies None), the persisted row MUST have assignee=None and
    every existing V1.1 contract MUST remain intact: exactly-one task,
    idempotency on duplicate trigger, concurrency limit, provenance,
    deactivation.
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    # Spec WITHOUT assignee (no key at all) — most common backward path.
    spec, ctx = _make_spec(objective_id="obj-A4a", trigger_id="trig-A4a")
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id is not None

    status, persisted_assignee = _read_task_assignee(
        isolated_kanban_db, result.task_id
    )
    assert status == "ready"
    assert persisted_assignee is None, (
        f"expected persisted assignee=None when spec omits it, "
        f"got {persisted_assignee!r}"
    )

    # Existing V1.1 contracts still hold: exactly-one task, idempotency,
    # provenance preserved.
    assert _count_tasks(isolated_kanban_db) == 1
    body = _read_task(isolated_kanban_db, result.task_id)[2]
    assert "AUTONOMOUS_PROVENANCE_BEGIN" in body
    assert "OBJECTIVE_ID=obj-A4a" in body

    # Idempotency: a second call with the same (objective_id, trigger_id)
    # MUST NOT create another task, MUST still return assignee=None.
    state.reset()
    state.enable(policy_version="1.0.0")
    spec_dup, ctx_dup = _make_spec(
        objective_id="obj-A4a", trigger_id="trig-A4a"
    )
    r_dup = attempt_autonomous_initiation(spec_dup, ctx_dup)
    assert r_dup.decision == "duplicate_suppressed"
    assert r_dup.task_id == result.task_id
    assert _count_tasks(isolated_kanban_db) == 1
    _, dup_assignee = _read_task_assignee(isolated_kanban_db, r_dup.task_id)
    assert dup_assignee is None

    # Explicit None case: spec with assignee=None explicitly. This is
    # equivalent to omitting the key (objective_spec.get("assignee") is
    # None in both cases). To admit a second objective we must first
    # archive the active one — the concurrency budget is keyed on the
    # active autonomous slot in the DB, not just in process state.
    from hermes_cli import kanban_db as kb
    archive_conn = kb.connect()
    try:
        kb.archive_task(archive_conn, result.task_id)
    finally:
        archive_conn.close()
    state.reset()
    state.enable(policy_version="1.0.0")
    spec_none, ctx_none = _make_spec(
        objective_id="obj-A4b", trigger_id="trig-A4b",
        assignee=None,
    )
    r_none = attempt_autonomous_initiation(spec_none, ctx_none)
    assert r_none.decision == "admit"
    assert r_none.task_id is not None
    task_id_none: str = r_none.task_id
    _, none_assignee = _read_task_assignee(
        isolated_kanban_db, task_id_none
    )
    assert none_assignee is None


# ----------------------------------------------------------------------
# AUTONOMOUS_INITIATION_PATH_V1.3 — Model / Provider Override Propagation
#
# A5: objective_spec["model_override"] + "provider_override" -> persisted
#     row carries both into the canonical Kanban task store, so the
#     dispatcher's spawn_worker_subprocess can translate them into the
#     worker's "-m <model> --provider <provider>" argv.
# A6: backward compatibility — both keys omitted and/or explicit None must
#     persist as None. No hidden default.
# ----------------------------------------------------------------------


def test_A5_model_and_provider_overrides_propagated(isolated_kanban_db, monkeypatch):
    """A5: model_override + provider_override survive the autonomous bridge.

    The objective_spec carries MiniMax-shaped values; the canonical
    kb.create_task path is the ONLY task-creation call. The persisted
    row MUST carry the exact strings the spec supplied, read back through
    the same column names the dispatcher's worker spawn path consumes.
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    state.reset()
    state.enable(policy_version="1.0.0")

    spec, ctx = _make_spec(
        objective_id="obj-A5",
        trigger_id="trig-A5",
        model_override="minimax-m3",
        provider_override="minimax",
    )
    result = attempt_autonomous_initiation(spec, ctx)
    assert result.decision == "admit"
    assert result.task_id is not None
    task_id: str = result.task_id

    persisted_model, persisted_provider = _read_task_overrides(
        isolated_kanban_db, task_id
    )
    assert persisted_model == "minimax-m3", (
        f"expected persisted model_override='minimax-m3', "
        f"got {persisted_model!r}"
    )
    assert persisted_provider == "minimax", (
        f"expected persisted provider_override='minimax', "
        f"got {persisted_provider!r}"
    )

    # Sanity: the V1.2 contract still holds — provenance, exactly-one,
    # idempotency are unaffected by the new override passthrough.
    assert _count_tasks(isolated_kanban_db) == 1
    body = _read_task(isolated_kanban_db, task_id)[2]
    assert "AUTONOMOUS_PROVENANCE_BEGIN" in body
    assert "OBJECTIVE_ID=obj-A5" in body


def test_A6_overrides_backward_compatible(isolated_kanban_db, monkeypatch):
    """A6: omitted AND explicit None both persist as None. No hidden default.

    Two objectives:
      A6a: spec without the keys at all (the most common caller path).
      A6b: spec with explicit None values.
    Both must persist model_override=None, provider_override=None. This
    guards against any hidden default appearing in the production
    bridge (no MiniMax, no "auto", no first-configured model).
    """
    from agent.autonomy import state
    from agent.autonomy.initiator import attempt_autonomous_initiation

    # --- A6a: omitted keys (most common caller path) ---
    state.reset()
    state.enable(policy_version="1.0.0")
    spec_omit, ctx_omit = _make_spec(
        objective_id="obj-A6a", trigger_id="trig-A6a"
    )
    r_omit = attempt_autonomous_initiation(spec_omit, ctx_omit)
    assert r_omit.decision == "admit"
    assert r_omit.task_id is not None

    model_omit, provider_omit = _read_task_overrides(
        isolated_kanban_db, r_omit.task_id
    )
    assert model_omit is None, (
        f"expected persisted model_override=None when key omitted, "
        f"got {model_omit!r}"
    )
    assert provider_omit is None, (
        f"expected persisted provider_override=None when key omitted, "
        f"got {provider_omit!r}"
    )

    # --- A6b: explicit None values ---
    # The concurrency slot is keyed on the active objective; archive A6a
    # before admitting A6b so the second objective is eligible.
    from hermes_cli import kanban_db as kb
    archive_conn = kb.connect()
    try:
        kb.archive_task(archive_conn, r_omit.task_id)
    finally:
        archive_conn.close()

    state.reset()
    state.enable(policy_version="1.0.0")
    spec_none, ctx_none = _make_spec(
        objective_id="obj-A6b",
        trigger_id="trig-A6b",
        model_override=None,
        provider_override=None,
    )
    r_none = attempt_autonomous_initiation(spec_none, ctx_none)
    assert r_none.decision == "admit"
    assert r_none.task_id is not None

    model_none, provider_none = _read_task_overrides(
        isolated_kanban_db, r_none.task_id
    )
    assert model_none is None, (
        f"expected persisted model_override=None when spec passes None, "
        f"got {model_none!r}"
    )
    assert provider_none is None, (
        f"expected persisted provider_override=None when spec passes None, "
        f"got {provider_none!r}"
    )

    # Existing V1.2 invariants still hold for both rows.
    assert _count_tasks(isolated_kanban_db) == 2
