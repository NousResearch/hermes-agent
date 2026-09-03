"""Tests for Task 23 — durable run-scoped write marker and mutation boundary."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import multiprocessing
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

import htr.execution_lock as _el
from htr import artifacts, contracts, events, io, paths
from htr.execution_lock import (
    ERROR_DURABILITY_FAILED,
    ERROR_OCCUPIED_UNKNOWN,
    LOCKS_DIR_NAME,
    RunExecutionLockBoundaryViolationError,
    RunExecutionLockDurabilityError,
    RunExecutionLockIndeterminateError,
    RunExecutionLockOccupiedError,
    RunExecutionLockPathUnsafeError,
    RunExecutionLockReleaseConflictError,
    acquire_marker_directory_entry_coordination,
    begin_run_write,
    disposition_unlink_marker,
    marker_present_noncreating,
    pin_lock_directory,
    read_marker_metadata_at,
    release_marker_directory_entry_coordination,

    run_mutation_boundary,
    run_write_barrier,
)
from htr.ids import new_attempt_id, new_run_id, new_task_id
from htr.finalization import SealState, evaluate_run_seal
from htr.state import ATTEMPT_RUNNING, TASK_RUNNING, RunFinalizedError, RunSealBlockedError

TASK16_PATH = Path(__file__).with_name("test_run_final_closure.py")


def _load_task16():
    import importlib.util

    spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_snapshot(root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    root_exists = root.exists()
    if not root_exists:
        return {
            "root_exists": False,
            "files": {},
            "file_mtimes": {},
            "dir_mtimes": {},
            "locks_exists": False,
            "locks_files": {},
            "locks_file_mtimes": {},
            "locks_dir_mtimes": {},
            "event_count": None,
        }
    files: dict[str, str] = {}
    file_mtimes: dict[str, int] = {}
    dir_mtimes: dict[str, int] = {}
    locks_root = root / LOCKS_DIR_NAME
    locks_exists = locks_root.exists()
    locks_files: dict[str, str] = {}
    locks_file_mtimes: dict[str, int] = {}
    locks_dir_mtimes: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_file():
            files[rel] = _file_digest(path)
            file_mtimes[rel] = path.stat().st_mtime_ns
        elif path.is_dir():
            dir_mtimes[rel] = path.stat().st_mtime_ns
    dir_mtimes["."] = root.stat().st_mtime_ns
    if locks_exists:
        for path in sorted(locks_root.rglob("*")):
            rel = str(path.relative_to(root))
            if path.is_file():
                locks_files[rel] = _file_digest(path)
                locks_file_mtimes[rel] = path.stat().st_mtime_ns
            elif path.is_dir():
                locks_dir_mtimes[rel] = path.stat().st_mtime_ns
        locks_dir_mtimes[".execution_locks"] = locks_root.stat().st_mtime_ns
    event_count = (
        len(events.read_task_events(run_id, base_dir=root)) if run_id else None
    )
    return {
        "root_exists": root_exists,
        "files": files,
        "file_mtimes": file_mtimes,
        "dir_mtimes": dir_mtimes,
        "locks_exists": locks_exists,
        "locks_files": locks_files,
        "locks_file_mtimes": locks_file_mtimes,
        "locks_dir_mtimes": locks_dir_mtimes,
        "event_count": event_count,
    }


def _assert_full_snapshot_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert after["root_exists"] == before["root_exists"]
    assert after["files"] == before["files"]
    assert after["file_mtimes"] == before["file_mtimes"]
    assert after["dir_mtimes"] == before["dir_mtimes"]
    assert after["locks_exists"] == before["locks_exists"]
    assert after["locks_files"] == before["locks_files"]
    assert after["locks_file_mtimes"] == before["locks_file_mtimes"]
    assert after["locks_dir_mtimes"] == before["locks_dir_mtimes"]
    assert after["event_count"] == before["event_count"]


def _finalize_run(tmp_path: Path) -> tuple[str, Any]:
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(
        run_id,
        chain[3],
        chain[4],
        chain[5],
        chain[6],
        chain[7],
        chain[8],
        chain[9],
        chain[10],
        chain[11],
        chain[12],
    )
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    return run_id, closure


MUTATOR_NAMES = frozenset(
    {
        "create_run_workspace",
        "create_task_workspace",
        "create_attempt_workspace",
        "write_task_card",
        "write_artifact_manifest",
        "add_artifact",
        "apply_task_transition",
        "register_attempt",
        "apply_attempt_transition",
        "submit_attempt_result",
        "submit_manual_verification",
        "complete_task_manually",
        "append_task_event",
        "append_run_event",
        "complete_run_manually",
        "review_run_manually",
        "plan_run_followup",
        "request_run_execution",
        "execute_run_execution_request",
        "verify_run_execution_result",
        "plan_post_verification_followup",
        "request_post_verification_execution",
        "record_post_verification_execution_result",
        "record_post_verification_execution_verification",
        "record_run_final_closure",
    }
)


def test_all_mutators_enter_boundary_via_decorator_or_barrier():
    repo = Path(__file__).resolve().parents[2]
    modules = {
        "io": (repo / "htr" / "io.py").read_text(encoding="utf-8"),
        "contracts": (repo / "htr" / "contracts.py").read_text(encoding="utf-8"),
        "artifacts": (repo / "htr" / "artifacts.py").read_text(encoding="utf-8"),
        "events": (repo / "htr" / "events.py").read_text(encoding="utf-8"),
    }
    found: set[str] = set()
    for source in modules.values():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in MUTATOR_NAMES:
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id == "run_mutation_boundary":
                    found.add(node.name)
                    break
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    if dec.func.id == "run_mutation_boundary":
                        found.add(node.name)
                        break
    manual = {"append_run_event", "record_run_final_closure"}
    assert found | manual == MUTATOR_NAMES


def test_preliminary_finalized_rejection_has_zero_filesystem_change(tmp_path):
    run_id, _ = _finalize_run(tmp_path)
    before = _full_snapshot(tmp_path, run_id=run_id)
    with pytest.raises(RunFinalizedError):
        io.create_run_workspace(run_id, base_dir=tmp_path)
    _assert_full_snapshot_unchanged(before, _full_snapshot(tmp_path, run_id=run_id))


def test_preliminary_untrusted_rejection_has_zero_filesystem_change(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(run_id, tmp_path), record
    )
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    before = _full_snapshot(tmp_path, run_id=run_id)
    with pytest.raises(RunSealBlockedError):
        io.create_run_workspace(run_id, base_dir=tmp_path)
    _assert_full_snapshot_unchanged(before, _full_snapshot(tmp_path, run_id=run_id))


def test_exact_closure_replay_has_zero_filesystem_change(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    first = events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    event_id = [
        e["event_id"]
        for e in events.read_task_events(run_id, base_dir=tmp_path)
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ][0]
    before = _full_snapshot(tmp_path, run_id=run_id)
    replay = events.record_run_final_closure(
        tmp_path,
        run_id,
        record,
        actor="human",
        event_id=event_id,
    )
    assert replay == first
    _assert_full_snapshot_unchanged(before, _full_snapshot(tmp_path, run_id=run_id))


def test_exact_replay_does_not_bootstrap_missing_lock_root(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    lock_root = tmp_path / LOCKS_DIR_NAME
    if lock_root.exists():
        import shutil

        shutil.rmtree(lock_root)
    assert not lock_root.exists()
    event_id = [
        e["event_id"]
        for e in events.read_task_events(run_id, base_dir=tmp_path)
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ][0]
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert not lock_root.exists()


def test_marker_present_blocks_exact_replay(tmp_path):
    run_id, record = _finalize_run(tmp_path)
    event_id = [
        e["event_id"]
        for e in events.read_task_events(run_id, base_dir=tmp_path)
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ][0]
    lock_root = tmp_path / LOCKS_DIR_NAME
    lock_root.mkdir(parents=True, exist_ok=True)
    (lock_root / f"{run_id}.marker").write_text("{}", encoding="utf-8")
    with pytest.raises(RunExecutionLockOccupiedError) as exc_info:
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human", event_id=event_id
        )
    assert exc_info.value.error_code == ERROR_OCCUPIED_UNKNOWN


def test_revalidation_failure_before_write_removes_marker(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    task_id = chain[1][0]
    with patch(
        "htr.finalization.assert_run_mutation_allowed",
        side_effect=RunSealBlockedError(run_id=run_id, reason_codes=("TEST",)),
    ):
        with pytest.raises(RunSealBlockedError):
            io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    assert not marker_present_noncreating(tmp_path, run_id)


def test_successful_mutation_removes_marker(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    task_id = chain[1][0]
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    assert not marker_present_noncreating(tmp_path, run_id)


def test_same_thread_same_key_reentrancy(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    task_id = chain[1][0]
    attempt_id = new_attempt_id()
    events.register_attempt(run_id, task_id, attempt_id, actor="human", base_dir=tmp_path)
    assert not marker_present_noncreating(tmp_path, run_id)


def test_cross_key_nesting_rejected(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_a = chain[0]
    run_b = TASK16._run_with_post_verification_execution_verification(tmp_path)[0]

    errors: list[BaseException] = []

    def outer() -> None:
        with run_write_barrier(run_a, tmp_path):
            try:
                with run_write_barrier(run_b, tmp_path):
                    pass
            except BaseException as exc:
                errors.append(exc)

    outer()
    assert len(errors) == 1
    assert isinstance(errors[0], RunExecutionLockBoundaryViolationError)


def test_other_thread_same_run_rejected(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def holder() -> None:
        with run_write_barrier(run_id, tmp_path):
            started.set()
            release.wait(timeout=5)

    def contender() -> None:
        started.wait(timeout=5)
        try:
            with run_write_barrier(run_id, tmp_path):
                pass
        except BaseException as exc:
            errors.append(exc)
        finally:
            release.set()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(errors) == 1
    assert isinstance(errors[0], RunExecutionLockOccupiedError)


def test_different_runs_do_not_contend(tmp_path):
    chain_a = TASK16._run_with_post_verification_execution_verification(tmp_path)
    chain_b = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_a = chain_a[0]
    run_b = chain_b[0]
    finished: list[str] = []

    def hold(run_id: str) -> None:
        with run_write_barrier(run_id, tmp_path):
            finished.append(run_id)

    t1 = threading.Thread(target=hold, args=(run_a,))
    t2 = threading.Thread(target=hold, args=(run_b,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert set(finished) == {run_a, run_b}


def test_exception_after_run_write_started_preserves_marker(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    with pytest.raises(RuntimeError, match="simulated write failure"):
        with run_write_barrier(run_id, tmp_path) as wb:
            wb.mark_run_write_started()
            raise RuntimeError("simulated write failure")
    assert marker_present_noncreating(tmp_path, run_id)


def test_durability_failure_after_successful_body_never_returns_plain_success(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]

    class _Body:
        entered = False

        def __enter__(self):
            _Body.entered = True
            return self

        def __exit__(self, *args):
            return False

    with patch("htr.execution_lock._release_marker_success") as release:
        release.side_effect = RunExecutionLockDurabilityError(run_id=run_id)
        with pytest.raises(RunExecutionLockDurabilityError) as exc_info:
            with run_write_barrier(run_id, tmp_path) as wb:
                wb.mark_run_write_started()
        assert exc_info.value.error_code == ERROR_DURABILITY_FAILED


def test_internal_append_rejected_outside_closure_context(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    event = events.make_run_event(
        event_type=events.EVENT_TYPE_MANUAL_RUN_REVIEWED,
        run_id=run_id,
        actor="human",
        payload={"run_id": run_id},
    )
    with pytest.raises(RunExecutionLockBoundaryViolationError):
        events._append_run_event_internal(run_id, event, tmp_path)


def _subprocess_contention_worker(
    runs_root: str,
    run_id: str,
    slot: Any,
    barrier: Any,
) -> None:
    base = Path(runs_root)
    barrier.wait()
    try:
        with run_write_barrier(run_id, base):
            slot.put("held")
    except RunExecutionLockOccupiedError:
        slot.put("blocked")


def test_subprocess_same_run_contention(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slot_a = ctx.Queue()
    slot_b = ctx.Queue()
    p1 = ctx.Process(
        target=_subprocess_contention_worker,
        args=(str(tmp_path), run_id, slot_a, barrier),
    )
    p2 = ctx.Process(
        target=_subprocess_contention_worker,
        args=(str(tmp_path), run_id, slot_b, barrier),
    )
    p1.start()
    p2.start()
    p1.join(timeout=15)
    p2.join(timeout=15)
    assert p1.exitcode == 0
    assert p2.exitcode == 0
    results = {slot_a.get(timeout=5), slot_b.get(timeout=5)}
    assert results == {"held", "blocked"}


def test_observe_and_plan_remain_lock_free(tmp_path):
    from htr.action_plan import PlanningIntent, build_action_plan
    from htr.observe import build_run_snapshot

    run_id, _ = _finalize_run(tmp_path)
    before = _full_snapshot(tmp_path, run_id=run_id)
    build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    build_action_plan(
        build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed"),
        PlanningIntent(
            requested_action="review_run_manually",
            htr_runs_root=str(tmp_path),
        ),
    )
    _assert_full_snapshot_unchanged(before, _full_snapshot(tmp_path, run_id=run_id))


# --- Task 23 final hardening ---


@dataclass
class WritePathProbe:
    acquires: int = 0
    revalidations: int = 0
    write_starts: int = 0
    releases: int = 0
    cleanups: int = 0
    max_depth: int = 0


@contextmanager
def _write_path_probe():
    probe = WritePathProbe()
    orig_acquire = _el._acquire_marker
    orig_release = _el._release_marker_success
    orig_cleanup = _el._cleanup_owned_marker
    orig_begin = _el.begin_run_write
    orig_revalidate = _el.RunWriteContext.revalidate_mutation_allowed
    orig_mark = _el.RunWriteContext.mark_run_write_started

    def track_acquire(*args, **kwargs):
        probe.acquires += 1
        return orig_acquire(*args, **kwargs)

    def track_release(entry):
        probe.releases += 1
        return orig_release(entry)

    def track_cleanup(*args, **kwargs):
        probe.cleanups += 1
        return orig_cleanup(*args, **kwargs)

    def track_mark(self):
        probe.write_starts += 1
        return orig_mark(self)

    def track_begin():
        return orig_begin()

    def track_revalidate(self):
        probe.revalidations += 1
        with _el._registry_lock:
            for entry in _el._registry.values():
                if (
                    entry.owner_pid == os.getpid()
                    and entry.owner_thread_id == threading.get_ident()
                ):
                    probe.max_depth = max(probe.max_depth, entry.depth)
        return orig_revalidate(self)

    with patch.object(_el, "_acquire_marker", side_effect=track_acquire), patch.object(
        _el, "_release_marker_success", side_effect=track_release
    ), patch.object(_el, "_cleanup_owned_marker", side_effect=track_cleanup    ), patch.object(
        _el, "begin_run_write", side_effect=track_begin
    ), patch.object(
        _el.RunWriteContext, "revalidate_mutation_allowed", track_revalidate
    ), patch.object(
        _el.RunWriteContext, "mark_run_write_started", track_mark
    ):
        yield probe


def _assert_successful_write_path(probe: WritePathProbe, *, nested: bool) -> None:
    assert probe.acquires == 1, probe
    assert probe.revalidations >= 1, probe
    assert probe.write_starts >= 1, probe
    assert probe.releases == 1, probe
    assert probe.cleanups == 0, probe
    if nested:
        assert probe.revalidations >= 2, probe
    else:
        assert probe.revalidations >= 1, probe


def _runtime_write_case(name: str, tmp_path: Path) -> tuple[Callable[[], Any], bool, str]:
    if name == "create_run_workspace":
        rid = new_run_id()
        return lambda: io.create_run_workspace(rid, base_dir=tmp_path), False, rid
    if name == "create_task_workspace":
        rid = new_run_id()
        tid = new_task_id()
        io.create_run_workspace(rid, base_dir=tmp_path)
        return lambda: io.create_task_workspace(rid, tid, base_dir=tmp_path), False, rid
    if name == "create_attempt_workspace":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_run_workspace(rid, base_dir=tmp_path)
        return (
            lambda: io.create_attempt_workspace(rid, tid, aid, base_dir=tmp_path),
            True,
            rid,
        )
    if name == "write_task_card":
        rid = new_run_id()
        tid = new_task_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        card = contracts.make_task_card(
            run_id=rid, task_id=tid, title="t", instruction="d", created_by="human"
        )
        return lambda: contracts.write_task_card(rid, tid, card, base_dir=tmp_path), False, rid
    if name == "apply_task_transition":
        rid = new_run_id()
        tid = new_task_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        return (
            lambda: events.apply_task_transition(
                rid, tid, TASK_RUNNING, actor="human", base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "register_attempt":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        return (
            lambda: events.register_attempt(rid, tid, aid, actor="human", base_dir=tmp_path),
            True,
            rid,
        )
    if name == "apply_attempt_transition":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        events.apply_task_transition(
            rid, tid, TASK_RUNNING, actor="human", base_dir=tmp_path
        )
        events.register_attempt(rid, tid, aid, actor="human", base_dir=tmp_path)
        return (
            lambda: events.apply_attempt_transition(
                rid, tid, aid, ATTEMPT_RUNNING, actor="human", base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "submit_attempt_result":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        events.apply_task_transition(
            rid, tid, TASK_RUNNING, actor="human", base_dir=tmp_path
        )
        events.register_attempt(rid, tid, aid, actor="human", base_dir=tmp_path)
        events.apply_attempt_transition(
            rid, tid, aid, ATTEMPT_RUNNING, actor="human", base_dir=tmp_path
        )
        result = contracts.make_attempt_result(
            run_id=rid, task_id=tid, attempt_id=aid, produced_by="human", summary="s"
        )
        return (
            lambda: events.submit_attempt_result(
                rid, tid, aid, result, actor="human", base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "submit_manual_verification":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        events.apply_task_transition(
            rid, tid, TASK_RUNNING, actor="human", base_dir=tmp_path
        )
        events.register_attempt(rid, tid, aid, actor="human", base_dir=tmp_path)
        events.apply_attempt_transition(
            rid, tid, aid, ATTEMPT_RUNNING, actor="human", base_dir=tmp_path
        )
        result = contracts.make_attempt_result(
            run_id=rid, task_id=tid, attempt_id=aid, produced_by="human", summary="s"
        )
        events.submit_attempt_result(
            rid, tid, aid, result, actor="human", base_dir=tmp_path
        )
        vr = contracts.make_verification_result(
            run_id=rid, task_id=tid, attempt_id=aid, outcome="passed"
        )
        return (
            lambda: events.submit_manual_verification(
                rid, tid, aid, vr, actor="human", base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "complete_task_manually":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        events.apply_task_transition(
            rid, tid, TASK_RUNNING, actor="human", base_dir=tmp_path
        )
        events.register_attempt(rid, tid, aid, actor="human", base_dir=tmp_path)
        events.apply_attempt_transition(
            rid, tid, aid, ATTEMPT_RUNNING, actor="human", base_dir=tmp_path
        )
        result = contracts.make_attempt_result(
            run_id=rid, task_id=tid, attempt_id=aid, produced_by="human", summary="s"
        )
        events.submit_attempt_result(
            rid, tid, aid, result, actor="human", base_dir=tmp_path
        )
        vr = contracts.make_verification_result(
            run_id=rid, task_id=tid, attempt_id=aid, outcome="passed"
        )
        events.submit_manual_verification(
            rid, tid, aid, vr, actor="human", base_dir=tmp_path
        )
        rec = contracts.make_task_completion_record(
            run_id=rid, task_id=tid, attempt_id=aid, reason="done"
        )
        return (
            lambda: events.complete_task_manually(
                rid, tid, aid, rec, actor="human", base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "write_artifact_manifest":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_attempt_workspace(rid, tid, aid, base_dir=tmp_path)
        manifest = {
            "schema_version": "1",
            "run_id": rid,
            "task_id": tid,
            "attempt_id": aid,
            "artifacts": [],
        }
        return (
            lambda: artifacts.write_artifact_manifest(rid, tid, aid, manifest, base_dir=tmp_path),
            False,
            rid,
        )
    if name == "add_artifact":
        rid = new_run_id()
        tid = new_task_id()
        aid = new_attempt_id()
        io.create_attempt_workspace(rid, tid, aid, base_dir=tmp_path)
        return (
            lambda: artifacts.add_artifact(
                rid, tid, aid, path="output/x.txt", kind="file", base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "append_task_event":
        rid = new_run_id()
        tid = new_task_id()
        io.create_task_workspace(rid, tid, base_dir=tmp_path)
        ev = events.make_event(
            run_id=rid,
            task_id=tid,
            event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
            previous_status="created",
            new_status="running",
            actor="human",
            payload={},
        )
        return lambda: events.append_task_event(rid, ev, base_dir=tmp_path), False, rid
    if name == "append_run_event":
        rid, _, _, _, _ = TASK16._run_with_reviewed_run(tmp_path)
        ev = events.make_run_event(
            event_type=events.EVENT_TYPE_MANUAL_RUN_REVIEWED,
            run_id=rid,
            actor="human",
            payload={"run_id": rid},
        )
        return lambda: events.append_run_event(rid, ev, base_dir=tmp_path), False, rid
    if name == "complete_run_manually":
        rid = new_run_id()
        tid = new_task_id()
        io.create_run_workspace(rid, base_dir=tmp_path)
        TASK16._complete_task(tmp_path, rid, tid)
        rec = contracts.make_run_completion_record(run_id=rid, completed_task_ids=[tid])
        return lambda: events.complete_run_manually(rid, rec, base_dir=tmp_path), True, rid
    if name == "review_run_manually":
        rid = new_run_id()
        tid = new_task_id()
        io.create_run_workspace(rid, base_dir=tmp_path)
        TASK16._complete_task(tmp_path, rid, tid)
        completion = contracts.make_run_completion_record(
            run_id=rid, completed_task_ids=[tid]
        )
        events.complete_run_manually(rid, completion, base_dir=tmp_path)
        review = contracts.make_run_review_record(
            run_id=rid, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
        )
        return (
            lambda: events.review_run_manually(rid, review, base_dir=tmp_path),
            True,
            rid,
        )
    if name == "plan_run_followup":
        rid, _, _, _, review = TASK16._run_with_reviewed_run(tmp_path)
        return (
            lambda: events.plan_run_followup(
                rid, TASK16._plan_record(rid), base_dir=tmp_path
            ),
            True,
            rid,
        )
    if name == "request_run_execution":
        rid, _, _, _, _, plan = TASK16._run_with_planned_run(tmp_path)
        req = contracts.make_run_execution_request_record(
            run_id=rid,
            source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
            execution_items=[TASK16._sample_execution_item()],
        )
        return lambda: events.request_run_execution(rid, req, base_dir=tmp_path), True, rid
    if name == "execute_run_execution_request":
        rid, _, _, _, _, _, req = TASK16._run_with_execution_request(tmp_path)
        return (
            lambda: events.execute_run_execution_request(tmp_path, rid, "human"),
            True,
            rid,
        )
    if name == "verify_run_execution_result":
        rid, _, _, _, _, _, _, result = TASK16._run_with_execution_result(tmp_path)
        vr = TASK16._verification_record(rid, result)
        return (
            lambda: events.verify_run_execution_result(tmp_path, rid, vr, actor="human"),
            True,
            rid,
        )
    if name == "plan_post_verification_followup":
        c = TASK16._run_with_verified_execution(tmp_path)
        rid = c[0]
        plan = TASK16._post_verification_plan_record(rid, c[7], c[8])
        return (
            lambda: events.plan_post_verification_followup(tmp_path, rid, plan, actor="human"),
            True,
            rid,
        )
    if name == "request_post_verification_execution":
        c = TASK16._run_with_post_verification_plan(tmp_path)
        rid = c[0]
        req = TASK16._post_verification_execution_request_record(rid, c[7], c[8], c[9])
        return (
            lambda: events.request_post_verification_execution(
                tmp_path, rid, req, actor="human"
            ),
            True,
            rid,
        )
    if name == "record_post_verification_execution_result":
        c = TASK16._run_with_post_verification_execution_request(tmp_path)
        rid = c[0]
        rec = TASK16._post_verification_execution_result_record(rid, c[7], c[8], c[9], c[10])
        return (
            lambda: events.record_post_verification_execution_result(
                tmp_path, rid, rec, actor="human"
            ),
            True,
            rid,
        )
    if name == "record_post_verification_execution_verification":
        c = TASK16._run_with_post_verification_execution_result(tmp_path)
        rid = c[0]
        rec = TASK16._post_verification_execution_verification_record(
            rid, c[7], c[8], c[9], c[10], c[11]
        )
        return (
            lambda: events.record_post_verification_execution_verification(
                tmp_path, rid, rec, actor="human"
            ),
            True,
            rid,
        )
    if name == "record_run_final_closure":
        c = TASK16._run_with_post_verification_execution_verification(tmp_path)
        rid = c[0]
        closure = TASK16._run_final_closure_record(rid, *c[3:13])
        return (
            lambda: events.record_run_final_closure(tmp_path, rid, closure, actor="human"),
            False,
            rid,
        )
    raise AssertionError(f"unknown mutator {name!r}")


def test_runtime_write_path_matrix_all_twenty_five_mutators(tmp_path):
    verified = 0
    for name in sorted(MUTATOR_NAMES):
        invoke, nested, run_id = _runtime_write_case(name, tmp_path)
        try:
            with _write_path_probe() as probe:
                invoke()
                _assert_successful_write_path(probe, nested=nested)
        except AssertionError as exc:
            raise AssertionError(f"mutator {name!r} failed: {exc}") from exc
        assert not marker_present_noncreating(tmp_path, run_id)
        verified += 1
    assert verified == 25


RUNTIME_WRITE_PATH_VERIFIED = "25/25"


# --- Subprocess workers (module-level for pickling) ---


def _subprocess_o_excl_race_worker(
    runs_root: str, run_id: str, slot: Any, release_gate: Any
) -> None:
    base = Path(runs_root)
    try:
        with run_write_barrier(run_id, base):
            slot.put("winner")
            release_gate.wait(timeout=10)
    except RunExecutionLockOccupiedError:
        slot.put("blocked")


def _subprocess_crash_before_marker_worker(runs_root: str, run_id: str) -> None:
    from unittest.mock import patch

    def _crash(*_args: Any, **_kwargs: Any) -> None:
        os._exit(55)

    with patch.object(_el, "_acquire_marker", side_effect=_crash):
        try:
            with run_write_barrier(run_id, Path(runs_root)):
                pass
        except BaseException:
            pass


def _subprocess_crash_after_marker_before_write_worker(runs_root: str, run_id: str) -> None:
    gen = run_write_barrier(run_id, Path(runs_root))
    wb = gen.__enter__()
    wb.revalidate_mutation_allowed()
    os._exit(44)


def _subprocess_crash_after_run_write_started_worker(runs_root: str, run_id: str) -> None:
    gen = run_write_barrier(run_id, Path(runs_root))
    wb = gen.__enter__()
    wb.mark_run_write_started()
    os._exit(33)


def _subprocess_first_closure_worker(
    runs_root: str,
    run_id: str,
    record_json: str,
    slot: Any,
    start_gate: Any,
) -> None:
    import json as _json
    from unittest.mock import patch

    record = _json.loads(record_json)
    real_write = io.atomic_write_json

    def slow_write(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.4)
        return real_write(*args, **kwargs)

    start_gate.wait(timeout=10)
    try:
        from htr.finalization import SealEvaluation, SealState

        with patch(
            "htr.events.evaluate_run_seal",
            return_value=SealEvaluation(SealState.NOT_FINALIZED, (), run_id),
        ), patch(
            "htr.finalization.evaluate_run_seal",
            return_value=SealEvaluation(SealState.NOT_FINALIZED, (), run_id),
        ), patch.object(io, "atomic_write_json", side_effect=slow_write):
            events.record_run_final_closure(Path(runs_root), run_id, record, actor="human")
        slot.put("written")
    except RunExecutionLockOccupiedError:
        slot.put("blocked")
    except BaseException as exc:
        slot.put(f"error:{type(exc).__name__}")




def _queue_get(queue: Any, *, timeout: float = 10) -> Any:
    import queue as queue_module

    return queue.get(timeout=timeout)


def _queue_has_item(queue: Any, *, timeout: float = 0.5) -> bool:
    import queue as queue_module

    try:
        item = queue.get(timeout=timeout)
        queue.put(item)
        return True
    except queue_module.Empty:
        return False

def _subprocess_exact_replay_worker(
    runs_root: str,
    run_id: str,
    record_json: str,
    event_id: str,
    slot: Any,
) -> None:
    import json as _json

    record = _json.loads(record_json)
    try:
        events.record_run_final_closure(
            Path(runs_root), run_id, record, actor="human", event_id=event_id
        )
        slot.put("replayed")
    except RunExecutionLockOccupiedError:
        slot.put("blocked")
    except BaseException as exc:
        slot.put(f"error:{type(exc).__name__}")


def _subprocess_fork_child_mutate_worker(runs_root: str, run_id: str, slot: Any) -> None:
    base = Path(runs_root)
    tid = new_task_id()
    try:
        io.create_task_workspace(run_id, tid, base_dir=base)
        slot.put("mutated")
    except RunExecutionLockOccupiedError:
        slot.put("blocked")
    except BaseException as exc:
        slot.put(f"error:{type(exc).__name__}")


def _subprocess_fork_child_release_worker(runs_root: str, run_id: str, slot: Any) -> None:
    with _el._registry_lock:
        for entry in _el._registry.values():
            if entry.key[2] == run_id:
                try:
                    _el._release_marker_success(entry)
                    slot.put("released")
                    return
                except BaseException as exc:
                    slot.put(f"error:{type(exc).__name__}")
                    return
    slot.put("no_entry")


def _subprocess_concurrent_bootstrap_worker(
    runs_root: str, run_id: str, slot: Any, release_gate: Any
) -> None:
    try:
        with run_write_barrier(run_id, Path(runs_root)):
            slot.put("ok")
            release_gate.wait(timeout=10)
    except RunExecutionLockOccupiedError:
        slot.put("blocked")
    except BaseException as exc:
        slot.put(f"error:{type(exc).__name__}")


# --- Mandatory subprocess tests ---


def test_subprocess_o_excl_race_exactly_one_winner(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    release_gate = ctx.Event()
    slots = [ctx.Queue() for _ in range(8)]
    procs = [
        ctx.Process(
            target=_subprocess_o_excl_race_worker,
            args=(str(tmp_path), run_id, slots[i], release_gate),
        )
        for i in range(8)
    ]
    for proc in procs:
        proc.start()
    outcomes = [slots[i].get(timeout=5) for i in range(8)]
    release_gate.set()
    for proc in procs:
        proc.join(timeout=10)
        assert proc.exitcode == 0
    assert outcomes.count("winner") == 1
    assert outcomes.count("blocked") == 7


def test_subprocess_crash_before_durable_marker_leaves_no_marker(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_subprocess_crash_before_marker_worker,
        args=(str(tmp_path), run_id),
    )
    proc.start()
    proc.join(timeout=10)
    assert proc.exitcode == 55
    assert not marker_present_noncreating(tmp_path, run_id)


def test_subprocess_crash_after_marker_before_run_write_preserves_marker(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_subprocess_crash_after_marker_before_write_worker,
        args=(str(tmp_path), run_id),
    )
    proc.start()
    proc.join(timeout=10)
    assert proc.exitcode == 44
    assert marker_present_noncreating(tmp_path, run_id)
    assert not (tmp_path / run_id).exists()


def test_subprocess_crash_after_run_write_started_marker_remains_retry_unsafe(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_subprocess_crash_after_run_write_started_worker,
        args=(str(tmp_path), run_id),
    )
    proc.start()
    proc.join(timeout=10)
    assert proc.exitcode == 33
    assert marker_present_noncreating(tmp_path, run_id)
    with pytest.raises(RunExecutionLockOccupiedError) as exc_info:
        with run_write_barrier(run_id, tmp_path):
            pass
    assert exc_info.value.error_code == ERROR_OCCUPIED_UNKNOWN
    with pytest.raises(RunExecutionLockOccupiedError):
        io.create_run_workspace(run_id, base_dir=tmp_path)


def test_subprocess_first_closure_versus_exact_replay_interleaving(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    import json as _json

    record_json = _json.dumps(record)
    ctx = multiprocessing.get_context("spawn")
    slot_a = ctx.Queue()
    slot_b = ctx.Queue()
    start_gate = ctx.Barrier(2)
    p_a = ctx.Process(
        target=_subprocess_first_closure_worker,
        args=(str(tmp_path), run_id, record_json, slot_a, start_gate),
    )
    p_b = ctx.Process(
        target=_subprocess_first_closure_worker,
        args=(str(tmp_path), run_id, record_json, slot_b, start_gate),
    )
    p_a.start()
    p_b.start()
    p_a.join(timeout=15)
    p_b.join(timeout=15)
    outcomes = {slot_a.get(timeout=2), slot_b.get(timeout=2)}
    assert outcomes == {"written", "blocked"}
    event_id = [
        e["event_id"]
        for e in events.read_task_events(run_id, base_dir=tmp_path)
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ][0]
    lock_root = tmp_path / LOCKS_DIR_NAME
    (lock_root / f"{run_id}.marker").write_text("{}", encoding="utf-8")
    with pytest.raises(RunExecutionLockOccupiedError):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human", event_id=event_id
        )
    (lock_root / f"{run_id}.marker").unlink()
    before = _full_snapshot(tmp_path, run_id=run_id)
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    _assert_full_snapshot_unchanged(before, _full_snapshot(tmp_path, run_id=run_id))


def _subprocess_spawned_child_ownership_worker(
    runs_root: str,
    run_id: str,
    mutate_slot: Any,
    release_slot: Any,
    parent_ready: Any,
) -> None:
    parent_ready.wait(timeout=10)
    _subprocess_fork_child_mutate_worker(runs_root, run_id, mutate_slot)
    _subprocess_fork_child_release_worker(runs_root, run_id, release_slot)


def test_spawned_child_cannot_mutate_or_release_parent_ownership(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    parent_ready = ctx.Event()
    mutate_slot = ctx.Queue()
    release_slot = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_spawned_child_ownership_worker,
        args=(str(tmp_path), run_id, mutate_slot, release_slot, parent_ready),
    )
    with run_write_barrier(run_id, tmp_path):
        proc.start()
        parent_ready.set()
        proc.join(timeout=15)
        assert proc.exitcode == 0
        assert mutate_slot.get(timeout=5) == "blocked"
        release_result = release_slot.get(timeout=5)
        assert release_result == "no_entry" or str(release_result).startswith("error:")
    assert not marker_present_noncreating(tmp_path, run_id)


# --- Mandatory path/release tests ---


def test_different_project_roots_same_run_id_remain_isolated(tmp_path):
    run_id = new_run_id()
    root_a = tmp_path / "project_a"
    root_b = tmp_path / "project_b"
    root_a.mkdir()
    root_b.mkdir()
    finished: list[str] = []

    def hold(root: Path, label: str) -> None:
        with run_write_barrier(run_id, root):
            finished.append(label)
            time.sleep(0.2)

    t1 = threading.Thread(target=hold, args=(root_a, "a"))
    t2 = threading.Thread(target=hold, args=(root_b, "b"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert set(finished) == {"a", "b"}


def test_supported_path_aliases_share_physical_marker(tmp_path):
    run_id = new_run_id()
    alias = Path(os.path.join(str(tmp_path), ".", "runs-root"))
    alias.mkdir(parents=True)
    canonical = alias.resolve()
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold_alias() -> None:
        with run_write_barrier(run_id, alias):
            started.set()
            release.wait(timeout=5)

    def try_canonical() -> None:
        started.wait(timeout=5)
        try:
            with run_write_barrier(run_id, canonical):
                pass
        except BaseException as exc:
            errors.append(exc)
        finally:
            release.set()

    t1 = threading.Thread(target=hold_alias)
    t2 = threading.Thread(target=try_canonical)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert len(errors) == 1
    assert isinstance(errors[0], RunExecutionLockOccupiedError)


def test_symlink_base_dir_fails_closed(tmp_path):
    run_id = new_run_id()
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    symlink = anchor / "link"
    symlink.symlink_to(target, target_is_directory=True)
    unsafe_root = symlink / "nested" / "runs"
    with pytest.raises(
        (
            RunExecutionLockPathUnsafeError,
            RunExecutionLockIndeterminateError,
            OSError,
        )
    ):
        with run_write_barrier(run_id, unsafe_root):
            pass


def test_multi_level_missing_runs_root_bootstrap(tmp_path):
    run_id = new_run_id()
    deep = tmp_path / "a" / "b" / "c" / "runs"
    assert not deep.exists()
    io.create_run_workspace(run_id, base_dir=deep)
    assert (deep / LOCKS_DIR_NAME).is_dir()
    assert not marker_present_noncreating(deep, run_id)


def test_concurrent_bootstrap_succeeds(tmp_path):
    run_id = new_run_id()
    deep = tmp_path / "nested" / "deep" / "runs"
    ctx = multiprocessing.get_context("spawn")
    release_gate = ctx.Event()
    slots = [ctx.Queue() for _ in range(6)]
    procs = [
        ctx.Process(
            target=_subprocess_concurrent_bootstrap_worker,
            args=(str(deep), run_id, slots[i], release_gate),
        )
        for i in range(6)
    ]
    for proc in procs:
        proc.start()
    outcomes = [slots[i].get(timeout=5) for i in range(6)]
    assert outcomes.count("ok") == 1
    assert outcomes.count("blocked") == 5
    release_gate.set()
    for proc in procs:
        proc.join(timeout=10)
        assert proc.exitcode == 0


def test_missing_marker_release_conflict(tmp_path):
    run_id = new_run_id()
    with pytest.raises((RunExecutionLockReleaseConflictError, RunExecutionLockDurabilityError)):
        with run_write_barrier(run_id, tmp_path) as wb:
            wb.mark_run_write_started()
            entry = _el._require_entry(wb.key, wb.token)
            os.unlink(entry.marker_name, dir_fd=entry.lock_root_fd)


def test_replaced_marker_release_conflict(tmp_path):
    run_id = new_run_id()
    with pytest.raises((RunExecutionLockReleaseConflictError, RunExecutionLockDurabilityError)):
        with run_write_barrier(run_id, tmp_path) as wb:
            wb.mark_run_write_started()
            entry = _el._require_entry(wb.key, wb.token)
            os.unlink(entry.marker_name, dir_fd=entry.lock_root_fd)
            replacement = os.open(
                entry.marker_name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | _el._O_NOFOLLOW | _el._O_CLOEXEC,
                _el.MARKER_MODE,
                dir_fd=entry.lock_root_fd,
            )
            os.close(replacement)


def test_symlink_marker_release_conflict(tmp_path):
    run_id = new_run_id()
    with pytest.raises((RunExecutionLockReleaseConflictError, RunExecutionLockDurabilityError)):
        with run_write_barrier(run_id, tmp_path) as wb:
            wb.mark_run_write_started()
            entry = _el._require_entry(wb.key, wb.token)
            lock_root_path = tmp_path / LOCKS_DIR_NAME
            os.unlink(entry.marker_name, dir_fd=entry.lock_root_fd)
            (lock_root_path / entry.marker_name).symlink_to("/tmp/nonexistent-marker")


def test_pid_thread_token_mismatch_cannot_release(tmp_path):
    run_id = new_run_id()
    with run_write_barrier(run_id, tmp_path) as wb:
        entry = _el._require_entry(wb.key, wb.token)
        entry.owner_pid = entry.owner_pid + 1
        with pytest.raises(RunExecutionLockBoundaryViolationError):
            _el._release_marker_success(entry)
        entry.owner_pid = os.getpid()
        entry.owner_thread_id = entry.owner_thread_id + 999_999
        with pytest.raises(RunExecutionLockBoundaryViolationError):
            _el._release_marker_success(entry)
        entry.owner_thread_id = threading.get_ident()
        with pytest.raises(RunExecutionLockBoundaryViolationError):
            _el._require_entry(wb.key, "wrong-token")


def test_cleanup_fsync_failure_never_returns_ordinary_success(tmp_path):
    run_id = new_run_id()
    with patch.object(_el, "_fsync_dir_fd", side_effect=RunExecutionLockIndeterminateError("fsync")):
        with pytest.raises(RunExecutionLockIndeterminateError):
            with run_write_barrier(run_id, tmp_path):
                pass
    assert not marker_present_noncreating(tmp_path, run_id)


def test_directory_fsync_failure_on_release_never_returns_plain_success(tmp_path):
    run_id = new_run_id()
    with pytest.raises(RunExecutionLockDurabilityError):
        with run_write_barrier(run_id, tmp_path) as wb:
            wb.mark_run_write_started()
            entry = _el._require_entry(wb.key, wb.token)
            with patch.object(_el, "_fsync_dir_fd", side_effect=OSError("dir fsync")):
                _el._release_marker_success(entry)


def test_existing_marker_is_always_occupied_unknown(tmp_path):
    run_id = new_run_id()
    lock_root = tmp_path / LOCKS_DIR_NAME
    lock_root.mkdir(parents=True)
    (lock_root / f"{run_id}.marker").write_text("{}", encoding="utf-8")
    with pytest.raises(RunExecutionLockOccupiedError) as exc_info:
        with run_write_barrier(run_id, tmp_path):
            pass
    assert exc_info.value.error_code == ERROR_OCCUPIED_UNKNOWN


def test_no_force_unlock_skip_env_bypass_or_automatic_takeover():
    source = Path(__file__).resolve().parents[2] / "htr" / "execution_lock.py"
    text = source.read_text(encoding="utf-8")
    forbidden = (
        "force_unlock",
        "FORCE_UNLOCK",
        "skip_lock",
        "SKIP_LOCK",
        "HERMES_EXEC_LOCK",
        "automatic_takeover",
    )
    for token in forbidden:
        assert token not in text
    assert "os.environ" not in text


def test_marker_directory_coordination_flock_acquire_release(tmp_path):
    """Task 26C: exclusive flock on pinned .execution_locks fd acquires and releases cleanly."""
    import os

    from htr.execution_lock import (
        acquire_marker_directory_entry_coordination,
        lock_directory_identity,
        pin_lock_directory,
        release_marker_directory_entry_coordination,
    )

    runs_fd, lock_fd = pin_lock_directory(tmp_path)
    try:
        before = lock_directory_identity(lock_fd)
        acquire_marker_directory_entry_coordination(lock_fd)
        release_marker_directory_entry_coordination(lock_fd)
        after = lock_directory_identity(lock_fd)
        assert before == after
    finally:
        os.close(lock_fd)
        os.close(runs_fd)


def _prepare_disposition_execute_chain(tmp_path: Path) -> tuple[str, str, str]:
    from htr.action_plan import PlanningIntent
    from htr.approval_control import issue_approval
    from htr.ids import (
        generate_marker_disposition_approval_id,
        generate_marker_disposition_claim_id,
        generate_marker_disposition_id,
        generate_reconciliation_case_id,
        new_event_id,
        new_run_id,
        new_task_id,
    )
    from htr.invoke_run_completion import invoke_approved_run_completion
    from htr.marker_disposition import (
        claim_marker_disposition_approval,
        create_marker_disposition_request,
        issue_marker_disposition_approval,
    )
    from htr.reconciliation_cases import (
        ReconciliationDecisionClass,
        ReconciliationNextProtocol,
        ReconciliationScopeReason,
        open_reconciliation_case,
        record_reconciliation_decision,
        record_reconciliation_observation,
    )
    from htr.reconciliation_inspection import PILOT_BOUND_API
    from htr.finalization import SealEvaluation, SealState
    from datetime import datetime, timedelta, timezone
    from unittest.mock import patch

    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[task_id]
    )
    intent = PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs={"record": completion, "actor": "human", "event_id": new_event_id()},
        htr_runs_root=str(tmp_path),
    )
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=expires,
        base_dir=tmp_path,
    )
    invoke_approved_run_completion(
        issue["approval_id"], claim_id="claim-el-coord", base_dir=tmp_path
    )
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    (locks_root / f"{run_id}.marker").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "acquisition_id": str(uuid.uuid4()),
                "pid": os.getpid(),
                "hostname": "test-host",
                "run_id": run_id,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    obs, _ = record_reconciliation_observation(
        case_id, base_dir=tmp_path, observed_by="operator"
    )
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.marker_disposition_review,
    )
    disposition_id = generate_marker_disposition_id()
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=SealEvaluation(SealState.FINALIZED_VALID, (), run_id),
    ):
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        base_dir=tmp_path,
    )
    claim_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    return disposition_id, run_id, generate_marker_disposition_claim_id()


def _subprocess_hold_run_write_barrier_worker(
    runs_root: str,
    run_id: str,
    slot: Any,
    barrier: Any,
    release_gate: Any,
) -> None:
    from pathlib import Path

    from htr.execution_lock import (
        acquire_marker_directory_entry_coordination,
        pin_lock_directory,
        release_marker_directory_entry_coordination,
    )

    runs_fd, lock_fd = pin_lock_directory(Path(runs_root))
    try:
        acquire_marker_directory_entry_coordination(lock_fd)
        barrier.wait()
        slot.put("held")
        release_gate.wait(timeout=10)
    finally:
        release_marker_directory_entry_coordination(lock_fd)
        os.close(lock_fd)
        os.close(runs_fd)


def _subprocess_disposition_execute_worker(
    runs_root: str,
    disposition_id: str,
    attempt_id: str,
    slot: Any,
    barrier: Any,
) -> None:
    from pathlib import Path

    from htr.finalization import SealEvaluation, SealState
    from htr.marker_disposition import execute_approved_marker_disposition
    from unittest.mock import patch

    request = json.loads(
        (
            Path(runs_root)
            / ".control"
            / "marker_dispositions"
            / disposition_id
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    run_id = request["marker_run_id"]
    barrier.wait()
    try:
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=SealEvaluation(SealState.FINALIZED_VALID, (), run_id),
        ):
            result = execute_approved_marker_disposition(
                disposition_id,
                attempt_id,
                executor="executor",
                base_dir=Path(runs_root),
            )
        slot.put(("ok", result.outcome_class))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def test_subprocess_disposition_execute_waits_for_run_write_barrier(tmp_path):
    disposition_id, run_id, attempt_id = _prepare_disposition_execute_chain(tmp_path)
    from htr.ids import generate_marker_disposition_attempt_id

    attempt_id = generate_marker_disposition_attempt_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    release_gate = ctx.Event()
    hold_slot = ctx.Queue()
    exec_slot = ctx.Queue()
    holder = ctx.Process(
        target=_subprocess_hold_run_write_barrier_worker,
        args=(str(tmp_path), run_id, hold_slot, barrier, release_gate),
    )
    executor = ctx.Process(
        target=_subprocess_disposition_execute_worker,
        args=(str(tmp_path), disposition_id, attempt_id, exec_slot, barrier),
    )
    holder.start()
    executor.start()
    assert _queue_get(hold_slot) == "held"
    assert not _queue_has_item(exec_slot)
    release_gate.set()
    holder.join(timeout=15)
    executor.join(timeout=15)
    assert holder.exitcode == 0
    assert executor.exitcode == 0
    outcome = _queue_get(exec_slot)
    assert outcome[0] == "ok"
    assert outcome[1] == "disposed_verified"
    assert not marker_present_noncreating(tmp_path, run_id)


def _subprocess_hold_disposition_flock_worker(
    runs_root: str,
    slot: Any,
    start_gate: Any,
    release_gate: Any,
) -> None:
    from pathlib import Path

    from htr.execution_lock import (
        acquire_marker_directory_entry_coordination,
        pin_lock_directory,
        release_marker_directory_entry_coordination,
    )

    runs_fd, lock_fd = pin_lock_directory(Path(runs_root))
    try:
        acquire_marker_directory_entry_coordination(lock_fd)
        slot.put("flock_held")
        start_gate.set()
        release_gate.wait(timeout=10)
        release_marker_directory_entry_coordination(lock_fd)
    finally:
        os.close(lock_fd)
        os.close(runs_fd)


def _subprocess_run_write_barrier_after_flock_worker(
    runs_root: str,
    run_id: str,
    slot: Any,
    start_gate: Any,
) -> None:
    from pathlib import Path

    from htr.execution_lock import run_write_barrier

    start_gate.wait(timeout=10)
    try:
        with run_write_barrier(run_id, Path(runs_root)):
            slot.put("acquired")
    except Exception as exc:
        slot.put(("blocked", type(exc).__name__))


def test_subprocess_run_write_barrier_blocks_while_disposition_flock_held(tmp_path):
    run_id = new_run_id()
    ctx = multiprocessing.get_context("spawn")
    start_gate = ctx.Event()
    release_gate = ctx.Event()
    flock_slot = ctx.Queue()
    acquire_slot = ctx.Queue()
    flock_proc = ctx.Process(
        target=_subprocess_hold_disposition_flock_worker,
        args=(str(tmp_path), flock_slot, start_gate, release_gate),
    )
    acquire_proc = ctx.Process(
        target=_subprocess_run_write_barrier_after_flock_worker,
        args=(str(tmp_path), run_id, acquire_slot, start_gate),
    )
    flock_proc.start()
    acquire_proc.start()
    assert flock_slot.get(timeout=10) == "flock_held"
    start_gate.wait(timeout=5)
    assert not _queue_has_item(acquire_slot)
    release_gate.set()
    flock_proc.join(timeout=15)
    result = acquire_slot.get(timeout=10)
    acquire_proc.join(timeout=15)
    assert flock_proc.exitcode == 0
    assert acquire_proc.exitcode == 0
    assert result == "acquired"


# --- Task 26C spawn+Barrier coordination (zero sleep/retry) ---


def _subprocess_concurrent_disposition_execute_worker(
    runs_root: str,
    disposition_id: str,
    attempt_id: str,
    slot: Any,
    barrier: Any,
) -> None:
    from pathlib import Path

    from htr.finalization import SealEvaluation, SealState
    from htr.marker_disposition import execute_approved_marker_disposition
    from unittest.mock import patch

    request = json.loads(
        (
            Path(runs_root)
            / ".control"
            / "marker_dispositions"
            / disposition_id
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    run_id = request["marker_run_id"]
    barrier.wait()
    try:
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=SealEvaluation(SealState.FINALIZED_VALID, (), run_id),
        ):
            result = execute_approved_marker_disposition(
                disposition_id,
                attempt_id,
                executor="executor",
                base_dir=Path(runs_root),
            )
        slot.put(("ok", result.outcome_class, result.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def test_subprocess_concurrent_identical_disposition_execution(tmp_path):
    from htr.ids import generate_marker_disposition_attempt_id

    disposition_id, run_id, _claim = _prepare_disposition_execute_chain(tmp_path)
    attempt_id = generate_marker_disposition_attempt_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_concurrent_disposition_execute_worker,
            args=(str(tmp_path), disposition_id, attempt_id, slots[i], barrier),
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [_queue_get(slots[i]) for i in range(2)]
    ok = [r for r in results if r[0] == "ok"]
    assert len(ok) == 2
    assert sum(1 for r in ok if r[2] is True) == 1
    assert sum(1 for r in ok if r[2] is False) == 1
    assert not marker_present_noncreating(tmp_path, run_id)


def test_subprocess_conflicting_concurrent_disposition_execution(tmp_path):
    from htr.ids import generate_marker_disposition_attempt_id

    disposition_id, run_id, _claim = _prepare_disposition_execute_chain(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_concurrent_disposition_execute_worker,
            args=(
                str(tmp_path),
                disposition_id,
                generate_marker_disposition_attempt_id(),
                slots[i],
                barrier,
            ),
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [_queue_get(slots[i]) for i in range(2)]
    classes = [r[1] for r in results if r[0] == "ok"]
    assert len(classes) == 2
    assert "disposed_verified" in classes
    assert "execution_ambiguous" in classes
    assert not marker_present_noncreating(tmp_path, run_id)


def _subprocess_crash_holding_flock_worker(
    runs_root: str,
    slot: Any,
    start_gate: Any,
) -> None:
    from pathlib import Path

    from htr.execution_lock import (
        acquire_marker_directory_entry_coordination,
        pin_lock_directory,
    )

    runs_fd, lock_fd = pin_lock_directory(Path(runs_root))
    try:
        acquire_marker_directory_entry_coordination(lock_fd)
        slot.put("flock_held")
        start_gate.wait(timeout=10)
        os._exit(1)
    finally:
        os.close(lock_fd)
        os.close(runs_fd)


def test_subprocess_crash_holding_flock_releases_on_next_acquire(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    ctx = multiprocessing.get_context("spawn")
    start_gate = ctx.Event()
    slot = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_crash_holding_flock_worker,
        args=(str(tmp_path), slot, start_gate),
    )
    proc.start()
    assert slot.get(timeout=10) == "flock_held"
    start_gate.set()
    proc.join(timeout=10)
    assert proc.exitcode == 1
    with run_write_barrier(run_id, base_dir=tmp_path):
        pass
    assert not marker_present_noncreating(tmp_path, run_id)




def test_thread_recursive_coordination_acquire_rejected(tmp_path):
    runs_fd, lock_fd = pin_lock_directory(tmp_path)
    try:
        acquire_marker_directory_entry_coordination(lock_fd)
        with pytest.raises(RunExecutionLockBoundaryViolationError, match="recursive"):
            acquire_marker_directory_entry_coordination(lock_fd)
        release_marker_directory_entry_coordination(lock_fd)
    finally:
        os.close(lock_fd)
        os.close(runs_fd)


def test_disposition_unlink_requires_coordination_held(tmp_path):
    run_id = new_run_id()
    with run_write_barrier(run_id, tmp_path):
        entry = _el._thread_active_entry()
        assert entry is not None
        metadata, identity = read_marker_metadata_at(entry.lock_root_fd, run_id)
        with pytest.raises(RunExecutionLockBoundaryViolationError, match="coordination required"):
            disposition_unlink_marker(
                entry.lock_root_fd,
                run_id,
                expected_identity=identity,
                expected_acquisition_id=metadata["acquisition_id"],
            )


def _subprocess_disposition_exception_releases_flock_worker(
    runs_root: str,
    slot: Any,
    start_gate: Any,
) -> None:
    from pathlib import Path

    from htr.execution_lock import (
        acquire_marker_directory_entry_coordination,
        pin_lock_directory,
        release_marker_directory_entry_coordination,
    )

    runs_fd, lock_fd = pin_lock_directory(Path(runs_root))
    try:
        acquire_marker_directory_entry_coordination(lock_fd)
        slot.put("flock_held")
        start_gate.set()
        raise RuntimeError("simulated disposition failure")
    except RuntimeError:
        release_marker_directory_entry_coordination(lock_fd)
        slot.put("released")
    finally:
        os.close(lock_fd)
        os.close(runs_fd)


def test_subprocess_disposition_exception_releases_flock(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    ctx = multiprocessing.get_context("spawn")
    start_gate = ctx.Event()
    slot = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_disposition_exception_releases_flock_worker,
        args=(str(tmp_path), slot, start_gate),
    )
    proc.start()
    assert slot.get(timeout=10) == "flock_held"
    start_gate.wait(timeout=5)
    proc.join(timeout=10)
    assert proc.exitcode == 0
    assert slot.get(timeout=5) == "released"
    with run_write_barrier(run_id, base_dir=tmp_path):
        pass
    assert not marker_present_noncreating(tmp_path, run_id)


def test_owned_release_uses_nonrecursive_coordination(tmp_path):
    run_id = new_run_id()
    with run_write_barrier(run_id, tmp_path):
        entry = _el._thread_active_entry()
        assert entry is not None
        acquire_marker_directory_entry_coordination(entry.lock_root_fd)
        try:
            with pytest.raises(RunExecutionLockBoundaryViolationError, match="recursive"):
                acquire_marker_directory_entry_coordination(entry.lock_root_fd)
        finally:
            release_marker_directory_entry_coordination(entry.lock_root_fd)


def test_disposition_flock_reacquire_after_execute(tmp_path):
    """Directory flock is released after disposition execute — re-acquire succeeds."""
    disposition_id, run_id, _claim = _prepare_disposition_execute_chain(tmp_path)
    from htr.ids import generate_marker_disposition_attempt_id
    from htr.execution_lock import (
        acquire_marker_directory_entry_coordination,
        pin_lock_directory,
        release_marker_directory_entry_coordination,
    )
    from htr.finalization import SealEvaluation, SealState
    from htr.marker_disposition import execute_approved_marker_disposition

    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=SealEvaluation(SealState.FINALIZED_VALID, (), run_id),
    ):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == "disposed_verified"
    runs_fd, lock_fd = pin_lock_directory(tmp_path)
    try:
        acquire_marker_directory_entry_coordination(lock_fd)
        release_marker_directory_entry_coordination(lock_fd)
    finally:
        os.close(lock_fd)
        os.close(runs_fd)


def test_disposition_marker_replacement_before_unlink_fails_closed(tmp_path):
    disposition_id, run_id, _claim = _prepare_disposition_execute_chain(tmp_path)
    from htr.ids import generate_marker_disposition_attempt_id
    from htr.finalization import SealEvaluation, SealState
    from htr.marker_disposition import execute_approved_marker_disposition

    marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["acquisition_id"] = str(uuid.uuid4())
    marker_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=SealEvaluation(SealState.FINALIZED_VALID, (), run_id),
    ):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == "marker_changed"
    assert marker_present_noncreating(tmp_path, run_id)
