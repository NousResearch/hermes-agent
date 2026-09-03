"""Tests for Task 22 — immutable finalized-run enforcement."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from htr import artifacts, contracts, events, io, paths
from htr.finalization import (
    SealState,
    assert_run_mutation_allowed,
    evaluate_run_seal,
)
from htr.ids import new_attempt_id, new_event_id
from htr.state import (
    ERROR_CODE_RUN_FINALIZED,
    ERROR_CODE_RUN_SEAL_BLOCKED,
    EventConflict,
    InvalidTransition,
    RunFinalizedError,
    RunSealBlockedError,
)

TASK16_PATH = Path(__file__).with_name("test_run_final_closure.py")


def _load_task16():
    spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16()


def _run_root(run_id: str, base_dir: Path) -> Path:
    return paths.run_root(run_id, base_dir)


def _tree_digest(run_root: Path) -> dict[str, str]:
    if not run_root.exists():
        return {}
    return {
        str(p.relative_to(run_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run_root.rglob("*"))
        if p.is_file()
    }


def _file_mtimes(run_root: Path) -> dict[str, int]:
    if not run_root.exists():
        return {}
    return {
        str(p.relative_to(run_root)): p.stat().st_mtime_ns
        for p in sorted(run_root.rglob("*"))
        if p.is_file()
    }


def _full_project_snapshot(base_dir: Path, *, run_id: str | None = None) -> dict[str, Any]:
    files: dict[str, str] = {}
    file_mtimes: dict[str, int] = {}
    dir_mtimes: dict[str, int] = {}
    root_exists = base_dir.exists()
    locks_root = base_dir / ".execution_locks"
    locks_exists = locks_root.exists()
    locks_files: dict[str, str] = {}
    locks_file_mtimes: dict[str, int] = {}
    locks_dir_mtimes: dict[str, int] = {}
    if root_exists:
        for path in sorted(base_dir.rglob("*")):
            rel = str(path.relative_to(base_dir))
            if path.is_file():
                files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
                file_mtimes[rel] = path.stat().st_mtime_ns
            elif path.is_dir():
                dir_mtimes[rel] = path.stat().st_mtime_ns
        dir_mtimes["."] = base_dir.stat().st_mtime_ns
    if locks_exists:
        for path in sorted(locks_root.rglob("*")):
            rel = str(path.relative_to(base_dir))
            if path.is_file():
                locks_files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
                locks_file_mtimes[rel] = path.stat().st_mtime_ns
            elif path.is_dir():
                locks_dir_mtimes[rel] = path.stat().st_mtime_ns
        locks_dir_mtimes[".execution_locks"] = locks_root.stat().st_mtime_ns
    event_count = (
        len(events.read_task_events(run_id, base_dir=base_dir)) if run_id else None
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


def _assert_full_project_snapshot_unchanged(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    assert after["root_exists"] == before["root_exists"]
    assert after["files"] == before["files"]
    assert after["file_mtimes"] == before["file_mtimes"]
    assert after["dir_mtimes"] == before["dir_mtimes"]
    assert after["locks_exists"] == before["locks_exists"]
    assert after["locks_files"] == before["locks_files"]
    assert after["locks_file_mtimes"] == before["locks_file_mtimes"]
    assert after["locks_dir_mtimes"] == before["locks_dir_mtimes"]
    assert after["event_count"] == before["event_count"]


@dataclass(frozen=True)
class RunSnapshot:
    digest: dict[str, str]
    mtimes: dict[str, int]
    event_count: int


def _snapshot_run(run_id: str, base_dir: Path) -> RunSnapshot:
    run_root = _run_root(run_id, base_dir)
    return RunSnapshot(
        digest=_tree_digest(run_root),
        mtimes=_file_mtimes(run_root),
        event_count=len(events.read_task_events(run_id, base_dir=base_dir)),
    )


def _assert_snapshot_unchanged(before: RunSnapshot, after: RunSnapshot) -> None:
    assert after.digest == before.digest
    assert after.mtimes == before.mtimes
    assert after.event_count == before.event_count


def _finalize_run(tmp_path: Path):
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
    return run_id, chain[1], chain[2], closure


def _invoke_with_base_dir(
    base_dir: Path,
    caller: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if "base_dir" in inspect.signature(caller).parameters:
        kwargs.setdefault("base_dir", base_dir)
    return caller(*args, **kwargs)


def _assert_blocked_on_finalized(
    run_id: str,
    base_dir: Path,
    caller: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    before = _snapshot_run(run_id, base_dir)
    with pytest.raises(RunFinalizedError) as exc_info:
        _invoke_with_base_dir(base_dir, caller, *args, **kwargs)
    assert exc_info.value.error_code == ERROR_CODE_RUN_FINALIZED
    _assert_snapshot_unchanged(before, _snapshot_run(run_id, base_dir))


def _assert_blocked_on_untrusted(
    run_id: str,
    base_dir: Path,
    caller: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    before = _snapshot_run(run_id, base_dir)
    with pytest.raises(RunSealBlockedError) as exc_info:
        _invoke_with_base_dir(base_dir, caller, *args, **kwargs)
    assert exc_info.value.error_code == ERROR_CODE_RUN_SEAL_BLOCKED
    _assert_snapshot_unchanged(before, _snapshot_run(run_id, base_dir))


# --- Workspace / task / attempt / artifact / event mutators (14 callables) ---

MUTATION_CASES = [
    pytest.param(
        lambda run_id, base, task_id, attempt_id: io.create_run_workspace(
            run_id, base_dir=base
        ),
        id="create_run_workspace",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: io.create_task_workspace(
            run_id, task_id, base_dir=base
        ),
        id="create_task_workspace",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: io.create_attempt_workspace(
            run_id, task_id, attempt_id, base_dir=base
        ),
        id="create_attempt_workspace",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: contracts.write_task_card(
            run_id,
            task_id,
            contracts.make_task_card(
                run_id=run_id,
                task_id=task_id,
                title="t",
                instruction="do",
                created_by="human",
            ),
            base_dir=base,
        ),
        id="write_task_card",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.apply_task_transition(
            run_id,
            task_id,
            "running",
            actor="human",
            base_dir=base,
        ),
        id="apply_task_transition",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.register_attempt(
            run_id,
            task_id,
            attempt_id,
            actor="human",
            base_dir=base,
        ),
        id="register_attempt",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.apply_attempt_transition(
            run_id,
            task_id,
            attempt_id,
            "running",
            actor="human",
            base_dir=base,
        ),
        id="apply_attempt_transition",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.submit_attempt_result(
            run_id,
            task_id,
            attempt_id,
            contracts.make_attempt_result(
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                produced_by="human",
                summary="done",
            ),
            actor="human",
            base_dir=base,
        ),
        id="submit_attempt_result",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.submit_manual_verification(
            run_id,
            task_id,
            attempt_id,
            contracts.make_verification_result(
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                outcome="passed",
            ),
            actor="human",
            base_dir=base,
        ),
        id="submit_manual_verification",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.complete_task_manually(
            run_id,
            task_id,
            attempt_id,
            contracts.make_task_completion_record(
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                reason="done",
            ),
            actor="human",
            base_dir=base,
        ),
        id="complete_task_manually",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.append_task_event(
            run_id,
            events.make_event(
                run_id=run_id,
                task_id=task_id,
                event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
                previous_status="running",
                new_status="running",
                actor="human",
                payload={},
            ),
            base_dir=base,
        ),
        id="append_task_event",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: events.append_run_event(
            run_id,
            events.make_run_event(
                event_type=events.EVENT_TYPE_MANUAL_RUN_REVIEWED,
                run_id=run_id,
                actor="human",
                payload={"run_id": run_id},
            ),
            base_dir=base,
        ),
        id="append_run_event",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: artifacts.write_artifact_manifest(
            run_id,
            task_id,
            attempt_id,
            {
                "schema_version": "1",
                "run_id": run_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "artifacts": [],
            },
            base_dir=base,
        ),
        id="write_artifact_manifest",
    ),
    pytest.param(
        lambda run_id, base, task_id, attempt_id: artifacts.add_artifact(
            run_id,
            task_id,
            attempt_id,
            path="output/x.txt",
            kind="file",
            base_dir=base,
        ),
        id="add_artifact",
    ),
]


@pytest.mark.parametrize("caller", MUTATION_CASES)
def test_sealed_run_blocks_mutation(tmp_path, caller):
    run_id, task_ids, attempt_ids, _ = _finalize_run(tmp_path)
    task_id = task_ids[0]
    attempt_id = attempt_ids[0]
    _assert_blocked_on_finalized(
        run_id,
        tmp_path,
        caller,
        run_id,
        tmp_path,
        task_id,
        attempt_id,
    )


# --- Phase 1 run-chain pre-closure APIs (10 callables) ---

RUN_CHAIN_PRE_CLOSURE_CASES = [
    pytest.param(
        lambda run_id, base: events.complete_run_manually(
            run_id, {"run_id": run_id}, base_dir=base
        ),
        id="complete_run_manually",
    ),
    pytest.param(
        lambda run_id, base: events.review_run_manually(
            run_id, {"run_id": run_id}, base_dir=base
        ),
        id="review_run_manually",
    ),
    pytest.param(
        lambda run_id, base: events.plan_run_followup(
            run_id, {"run_id": run_id}, base_dir=base
        ),
        id="plan_run_followup",
    ),
    pytest.param(
        lambda run_id, base: events.request_run_execution(
            run_id, {"run_id": run_id}, base_dir=base
        ),
        id="request_run_execution",
    ),
    pytest.param(
        lambda run_id, base: events.execute_run_execution_request(
            base, run_id, "executor"
        ),
        id="execute_run_execution_request",
    ),
    pytest.param(
        lambda run_id, base: events.verify_run_execution_result(
            base, run_id, {"run_id": run_id}, "human"
        ),
        id="verify_run_execution_result",
    ),
    pytest.param(
        lambda run_id, base: events.plan_post_verification_followup(
            base, run_id, {"run_id": run_id}, "human"
        ),
        id="plan_post_verification_followup",
    ),
    pytest.param(
        lambda run_id, base: events.request_post_verification_execution(
            base, run_id, {"run_id": run_id}, "human"
        ),
        id="request_post_verification_execution",
    ),
    pytest.param(
        lambda run_id, base: events.record_post_verification_execution_result(
            base, run_id, {"run_id": run_id}, "human"
        ),
        id="record_post_verification_execution_result",
    ),
    pytest.param(
        lambda run_id, base: events.record_post_verification_execution_verification(
            base, run_id, {"run_id": run_id}, "human"
        ),
        id="record_post_verification_execution_verification",
    ),
]


@pytest.mark.parametrize("caller", RUN_CHAIN_PRE_CLOSURE_CASES)
def test_sealed_run_blocks_run_chain_mutations(tmp_path, caller):
    run_id, _, _, _ = _finalize_run(tmp_path)
    _assert_blocked_on_finalized(run_id, tmp_path, caller, run_id, tmp_path)


# --- record_run_final_closure (11th run-chain API) ---

def test_first_closure_succeeds_and_orders_json_before_event(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(
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
    ops: list[str] = []
    real_append = events._append_run_event_internal

    def track_write(path, data):
        if str(path).endswith("run_final_closure_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events._append_run_event_internal", side_effect=track_append
    ):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert ops == ["write_record", "append_event"]
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID


def test_closure_event_appended_exactly_once(tmp_path):
    run_id, _, _, _ = _finalize_run(tmp_path)
    events_log = events.read_task_events(run_id, base_dir=tmp_path)
    closure_events = [
        e
        for e in events_log
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ]
    assert len(closure_events) == 1


def test_exact_closure_replay_is_zero_write(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(
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
    event_id = new_event_id()
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    before = _snapshot_run(run_id, tmp_path)
    returned = events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert returned == record
    _assert_snapshot_unchanged(before, _snapshot_run(run_id, tmp_path))


def test_closure_replay_conflicting_event_id_raises_event_conflict(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    event_id = new_event_id()
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    tampered = dict(record)
    tampered["closer"] = "other_actor"
    before = _snapshot_run(run_id, tmp_path)
    with pytest.raises(EventConflict):
        events.record_run_final_closure(
            tmp_path, run_id, tampered, actor="human", event_id=event_id
        )
    _assert_snapshot_unchanged(before, _snapshot_run(run_id, tmp_path))


def test_closure_replay_missing_event_id_raises_finalized_error(tmp_path):
    run_id, _, _, record = _finalize_run(tmp_path)
    _assert_blocked_on_finalized(
        run_id,
        tmp_path,
        events.record_run_final_closure,
        tmp_path,
        run_id,
        record,
        actor="human",
    )


def test_closure_replay_different_event_id_raises_finalized_error(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=new_event_id()
    )
    _assert_blocked_on_finalized(
        run_id,
        tmp_path,
        events.record_run_final_closure,
        tmp_path,
        run_id,
        record,
        actor="human",
        event_id=new_event_id(),
    )


# --- Guard-order verification ---

def test_finalized_guard_runs_before_idempotent_task_replay(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    task_id = chain[1][0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=new_event_id()
    )
    prior_events = events.read_task_events(run_id, base_dir=tmp_path)
    replay_event = next(
        e
        for e in prior_events
        if e.get("event_type") == events.EVENT_TYPE_TASK_STATUS_CHANGED
        and e.get("task_id") == task_id
    )
    _assert_blocked_on_finalized(
        run_id,
        tmp_path,
        events.apply_task_transition,
        run_id,
        task_id,
        replay_event["new_status"],
        actor=replay_event["actor"],
        event_id=replay_event["event_id"],
    )


def test_finalized_guard_runs_before_idempotent_run_chain_replay(tmp_path):
    run_id, task_ids, _, _ = _finalize_run(tmp_path)
    prior_events = events.read_task_events(run_id, base_dir=tmp_path)
    replay_event = next(
        e
        for e in prior_events
        if e.get("event_type") == events.EVENT_TYPE_MANUAL_RUN_COMPLETED
    )
    completion = contracts.make_run_completion_record(
        run_id=run_id,
        completed_task_ids=[task_ids[0]],
        reason="done",
    )
    _assert_blocked_on_finalized(
        run_id,
        tmp_path,
        events.complete_run_manually,
        run_id,
        completion,
        actor=replay_event["actor"],
        event_id=replay_event["event_id"],
    )


def test_workspace_guard_runs_before_ensure_dir(tmp_path):
    run_id, task_ids, _, _ = _finalize_run(tmp_path)
    with patch("htr.io.ensure_dir") as ensure_dir:
        with pytest.raises(RunFinalizedError):
            io.create_task_workspace(run_id, task_ids[0], base_dir=tmp_path)
        ensure_dir.assert_not_called()


def test_artifact_guard_runs_before_manifest_write(tmp_path):
    run_id, task_ids, attempt_ids, _ = _finalize_run(tmp_path)
    with patch("htr.artifacts.atomic_write_json") as write_json:
        with pytest.raises(RunFinalizedError):
            artifacts.write_artifact_manifest(
                run_id,
                task_ids[0],
                attempt_ids[0],
                {
                    "schema_version": "1",
                    "run_id": run_id,
                    "task_id": task_ids[0],
                    "attempt_id": attempt_ids[0],
                    "artifacts": [],
                },
                base_dir=tmp_path,
            )
        write_json.assert_not_called()


def test_public_append_guard_runs_before_jsonl_append(tmp_path):
    run_id, task_ids, _, _ = _finalize_run(tmp_path)
    event = events.make_event(
        run_id=run_id,
        task_id=task_ids[0],
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        previous_status="running",
        new_status="running",
        actor="human",
        payload={},
    )
    with patch("htr.events.append_jsonl") as append_jsonl:
        with pytest.raises(RunFinalizedError):
            events.append_task_event(run_id, event, base_dir=tmp_path)
        append_jsonl.assert_not_called()


def test_public_append_rejects_closure_event_before_finalization(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    event = events.make_run_event(
        event_type=events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED,
        run_id=run_id,
        actor="human",
        payload={"run_id": run_id, "closer": record["closer"]},
    )
    with pytest.raises(InvalidTransition):
        events.append_run_event(run_id, event, base_dir=tmp_path)


def test_json_only_partial_closure_cannot_complete_via_public_append(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(run_id, tmp_path), record
    )
    event = events.make_run_event(
        event_type=events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED,
        run_id=run_id,
        actor="human",
        payload={"run_id": run_id, "closer": record["closer"]},
    )
    before = _snapshot_run(run_id, tmp_path)
    with pytest.raises(InvalidTransition):
        events.append_run_event(run_id, event, base_dir=tmp_path)
    _assert_snapshot_unchanged(before, _snapshot_run(run_id, tmp_path))


# --- Untrusted-state matrix ---

def test_untrusted_malformed_closure_json_blocks_mutation(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure_path = contracts.run_final_closure_record_json_path(run_id, tmp_path)
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_text("{not valid json", encoding="utf-8")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id,
        tmp_path,
        events.apply_task_transition,
        run_id,
        chain[1][0],
        "running",
        actor="human",
    )


def test_untrusted_schema_invalid_closure_blocks_mutation(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(run_id, tmp_path),
        {"run_id": run_id},
    )
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id,
        tmp_path,
        events.apply_task_transition,
        run_id,
        chain[1][0],
        "running",
        actor="human",
    )


def test_untrusted_fingerprint_mismatch_blocks_mutation(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    record = dict(record)
    record["source_run_completion_fingerprint"] = "deadbeef"
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(run_id, tmp_path), record
    )
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id, tmp_path, assert_run_mutation_allowed, run_id
    )


def test_untrusted_json_without_event_blocks_mutation(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(run_id, tmp_path), record
    )
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id,
        tmp_path,
        events.record_run_final_closure,
        tmp_path,
        run_id,
        record,
        actor="human",
        event_id=new_event_id(),
    )


def test_untrusted_event_without_json_blocks_mutation(tmp_path):
    run_id, _, _, _ = _finalize_run(tmp_path)
    closure_path = contracts.run_final_closure_record_json_path(run_id, tmp_path)
    closure_path.unlink()
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id, tmp_path, assert_run_mutation_allowed, run_id
    )


def test_untrusted_duplicate_closure_events_block_mutation(tmp_path):
    run_id, _, _, _ = _finalize_run(tmp_path)
    events_path = paths.task_events_path(run_id, tmp_path)
    existing = events.read_task_events(run_id, base_dir=tmp_path)
    duplicate = [
        e
        for e in existing
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ][0]
    duplicate = dict(duplicate)
    duplicate["event_id"] = new_event_id()
    io.append_jsonl(events_path, duplicate)
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id, tmp_path, assert_run_mutation_allowed, run_id
    )


def test_untrusted_conflicting_closure_event_blocks_mutation(tmp_path):
    import json

    run_id, _, _, record = _finalize_run(tmp_path)
    events_path = paths.task_events_path(run_id, tmp_path)
    all_events = events.read_task_events(run_id, base_dir=tmp_path)
    closure_event = next(
        e
        for e in all_events
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    )
    tampered = dict(closure_event)
    payload = dict(tampered["payload"])
    payload["closer"] = "tampered_actor"
    tampered["payload"] = payload
    rewritten = [
        e for e in all_events if e["event_id"] != closure_event["event_id"]
    ] + [tampered]
    events_path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in rewritten) + "\n",
        encoding="utf-8",
    )
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id,
        tmp_path,
        events.record_run_final_closure,
        tmp_path,
        run_id,
        record,
        actor="human",
        event_id=closure_event["event_id"],
    )


def test_untrusted_correspondence_failure_blocks_mutation(tmp_path):
    run_id, _, _, record = _finalize_run(tmp_path)
    closure_path = contracts.run_final_closure_record_json_path(run_id, tmp_path)
    tampered = dict(record)
    tampered["closure_items"] = []
    io.atomic_write_json(closure_path, tampered)
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    _assert_blocked_on_untrusted(
        run_id, tmp_path, assert_run_mutation_allowed, run_id
    )


def test_untrusted_symlink_escape_blocks_mutation(tmp_path):
    run_id, _, _, _ = _finalize_run(tmp_path)
    run_root = _run_root(run_id, tmp_path)
    outside = tmp_path.parent / f"outside_{run_id}"
    run_root.rename(outside)
    run_root.symlink_to(outside)
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.INDETERMINATE
    _assert_blocked_on_untrusted(
        run_id, tmp_path, assert_run_mutation_allowed, run_id
    )


def test_untrusted_unreadable_events_is_indeterminate(tmp_path):
    run_id, _, _, _ = _finalize_run(tmp_path)
    run_root = _run_root(run_id, tmp_path)
    events_path = paths.task_events_path(run_id, tmp_path)
    before_digest = _tree_digest(run_root)
    before_mtimes = _file_mtimes(run_root)
    events_path.write_text("{bad json\n", encoding="utf-8")
    after_corrupt_digest = _tree_digest(run_root)
    after_corrupt_mtimes = _file_mtimes(run_root)
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.INDETERMINATE
    with pytest.raises(RunSealBlockedError):
        assert_run_mutation_allowed(run_id, tmp_path)
    assert _tree_digest(run_root) == after_corrupt_digest
    assert _file_mtimes(run_root) == after_corrupt_mtimes


# --- Internal append boundary + import audit ---

def test_import_smoke_no_cycles():
    importlib.reload(importlib.import_module("htr.finalization"))
    importlib.reload(importlib.import_module("htr.events"))


def test_import_smoke_all_htr_modules():
    for name in (
        "htr",
        "htr.finalization",
        "htr.events",
        "htr.contracts",
        "htr.io",
        "htr.artifacts",
        "htr.observe",
        "htr.action_plan",
    ):
        importlib.import_module(name)


def test_internal_append_call_sites_are_narrow():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "htr" / "events.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed = {"record_run_final_closure"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "_append_run_event_internal":
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef) and node in ast.walk(fn):
                    found.add(fn.name)
                    break
    assert found
    assert found <= allowed


def test_no_bypass_parameters_on_internal_append():
    sig = inspect.signature(events._append_run_event_internal)
    assert list(sig.parameters) == ["run_id", "event", "base_dir"]


def test_internal_append_not_exported_from_htr_init():
    import htr

    assert "_append_run_event_internal" not in dir(htr)
    assert "_append_run_event_internal" not in htr.__all__


def test_finalization_does_not_import_events():
    import htr.finalization as finalization

    assert "htr.events" not in finalization.__dict__
    source = Path(finalization.__file__).read_text(encoding="utf-8")
    assert "from htr.events" not in source
    assert "import htr.events" not in source


def test_finalized_error_codes():
    err = RunFinalizedError(run_id="run_x")
    assert err.error_code == ERROR_CODE_RUN_FINALIZED
    blocked = RunSealBlockedError(
        run_id="run_x", reason_codes=("CLOSURE_PENDING_EVENT",)
    )
    assert blocked.error_code == ERROR_CODE_RUN_SEAL_BLOCKED


def test_observe_and_plan_still_read_only_on_sealed_run(tmp_path):
    from htr.action_plan import (
        STATE_BLOCKED_FINALIZED,
        PlanningIntent,
        build_action_plan,
    )
    from htr.observe import build_run_snapshot

    run_id, _, _, _ = _finalize_run(tmp_path)
    snapshot = build_run_snapshot(run_id, base_dir=tmp_path, observed_at="fixed")
    assert snapshot["phase1_chain"]["terminal_reached"] is True
    plan = build_action_plan(
        snapshot,
        PlanningIntent(
            requested_action="review_run_manually",
            htr_runs_root=str(tmp_path),
        ),
    )
    assert plan["plan_state"] == STATE_BLOCKED_FINALIZED


def test_guard_uses_new_attempt_id_for_register_attempt_on_sealed_run(tmp_path):
    run_id, task_ids, _, _ = _finalize_run(tmp_path)
    fresh_attempt = new_attempt_id()
    _assert_blocked_on_finalized(
        run_id,
        tmp_path,
        events.register_attempt,
        run_id,
        task_ids[0],
        fresh_attempt,
        actor="human",
    )


def test_finalized_preliminary_rejection_has_literal_project_zero_write(tmp_path):
    run_id, _, _, _ = _finalize_run(tmp_path)
    before = _full_project_snapshot(tmp_path, run_id=run_id)
    with pytest.raises(RunFinalizedError):
        io.create_run_workspace(run_id, base_dir=tmp_path)
    _assert_full_project_snapshot_unchanged(
        before, _full_project_snapshot(tmp_path, run_id=run_id)
    )


def test_untrusted_preliminary_rejection_has_literal_project_zero_write(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(run_id, tmp_path), record
    )
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    before = _full_project_snapshot(tmp_path, run_id=run_id)
    with pytest.raises(RunSealBlockedError):
        io.create_run_workspace(run_id, base_dir=tmp_path)
    _assert_full_project_snapshot_unchanged(
        before, _full_project_snapshot(tmp_path, run_id=run_id)
    )


def test_exact_closure_replay_has_literal_project_zero_write(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    record = TASK16._run_final_closure_record(run_id, *chain[3:13])
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    event_id = [
        e["event_id"]
        for e in events.read_task_events(run_id, base_dir=tmp_path)
        if e.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ][0]
    before = _full_project_snapshot(tmp_path, run_id=run_id)
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    _assert_full_project_snapshot_unchanged(
        before, _full_project_snapshot(tmp_path, run_id=run_id)
    )
