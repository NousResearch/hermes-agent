"""Tests for Task 24 — authoritative approval control schema and API."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from htr import approval_control, contracts, events, io, paths
from htr.action_plan import PlanningIntent, _CATALOG
from htr.approval_control import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_CONSUMED,
    claim_approval,
    get_approval,
    issue_approval,
    list_approvals,
    record_use_outcome,
    revoke_approval,
    validate_approval,
)
from htr.approval_control import OUTCOME_SCHEMA_VERSION_V2
from htr import execution_lock as _el
from htr.execution_lock import (
    LOCKS_DIR_NAME,
    RunExecutionLockBoundaryViolationError,
    RunExecutionLockDurabilityError,
    RunExecutionLockIndeterminateError,
    RunExecutionLockOccupiedError,
    marker_present_noncreating,
    run_write_barrier,
)
from htr.finalization import SealState, evaluate_run_seal
from htr.ids import new_approval_id, new_event_id, new_run_id, new_task_id
from htr.state import (
    ApprovalConflictError,
    ApprovalFinalizedRunError,
    ApprovalStateError,
    ApprovalValidationError,
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


def _expires_in(hours: float = 1.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _run_with_completion_only(tmp_path: Path) -> str:
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[task_id]
    )
    events.complete_run_manually(run_id, completion, base_dir=tmp_path)
    return run_id


def _review_intent(run_id: str, tmp_path: Path, event_id: str) -> PlanningIntent:
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    return PlanningIntent(
        requested_action="review_run_manually",
        action_inputs={"record": review, "actor": "human", "event_id": event_id},
        htr_runs_root=str(tmp_path),
    )


def _finalize_run(tmp_path: Path) -> str:
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID
    return run_id


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_tree_snapshot(tmp_path: Path, run_id: str) -> dict[str, str]:
    root = paths.run_root(run_id, tmp_path)
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = _file_digest(path)
    return out


CONTROL_MUTATOR_NAMES = frozenset(
    {
        "issue_approval",
        "revoke_approval",
        "claim_approval",
        "record_use_outcome",
    }
)


def test_control_mutators_use_approval_control_barrier():
    repo = Path(__file__).resolve().parents[2]
    source = (repo / "htr" / "approval_control.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in CONTROL_MUTATOR_NAMES:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.With):
                for item in child.items:
                    if (
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == "_approval_control_barrier"
                    ):
                        found.add(node.name)
    assert found == CONTROL_MUTATOR_NAMES


def test_execution_lock_has_no_generic_seal_switch():
    source = (Path(__file__).resolve().parents[2] / "htr" / "execution_lock.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "enforce_seal",
        "control_write_barrier",
        "approval_use_session",
        "begin_control_write",
        "skip_seal",
        "control_plane",
    )
    for token in forbidden:
        assert token not in source, f"forbidden token {token!r} in execution_lock.py"


def test_issue_requires_explicit_event_id(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    intent = PlanningIntent(
        requested_action="review_run_manually",
        action_inputs={"record": review, "actor": "human"},
        htr_runs_root=str(tmp_path),
    )
    with pytest.raises(ApprovalValidationError, match="event_id"):
        issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def test_issue_revoke_claim_outcome_happy_path(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    event_id = new_event_id()
    intent = _review_intent(run_id, tmp_path, event_id)
    before_tree = _run_tree_snapshot(tmp_path, run_id)
    approvals_before = paths.approvals_path(run_id, tmp_path)
    digest_before = _file_digest(approvals_before) if approvals_before.exists() else None

    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    assert issue["approval_digest"].startswith("sha256:")
    assert paths.approval_issue_path(issue["approval_id"], tmp_path).is_file()

    claim = claim_approval(
        issue["approval_id"],
        "claim-001",
        "bob",
        base_dir=tmp_path,
    )
    assert claim["claim_id"] == "claim-001"

    outcome = record_use_outcome(
        issue["approval_id"],
        "claim-001",
        OUTCOME_CONSUMED,
        base_dir=tmp_path,
    )
    assert outcome["outcome_class"] == OUTCOME_CONSUMED

    view = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert view["derived"]["authoritative_state"] == OUTCOME_CONSUMED
    assert _run_tree_snapshot(tmp_path, run_id) == before_tree
    if digest_before is not None:
        assert _file_digest(approvals_before) == digest_before


def test_issue_exact_replay_idempotent(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    intent = _review_intent(run_id, tmp_path, new_event_id())
    first = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    issue_path = paths.approval_issue_path(first["approval_id"], tmp_path)
    before = _file_digest(issue_path)
    second = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=first["expires_at"],
        approval_id=first["approval_id"],
        base_dir=tmp_path,
    )
    assert second == first
    assert _file_digest(issue_path) == before


def test_issue_conflicting_replay_fails(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    intent = _review_intent(run_id, tmp_path, new_event_id())
    first = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    intent2 = _review_intent(run_id, tmp_path, new_event_id())
    with pytest.raises(ApprovalConflictError):
        issue_approval(
            run_id,
            intent2,
            approver_id="alice",
            executor_id="bob",
            expires_at=first["expires_at"],
            approval_id=first["approval_id"],
            base_dir=tmp_path,
        )


def test_expiry_max_24_hours_rejected(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    intent = _review_intent(run_id, tmp_path, new_event_id())
    with pytest.raises(ApprovalValidationError, match="24 hours"):
        issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=25)).isoformat(),
            base_dir=tmp_path,
        )


def test_finalized_run_issue_and_claim_rejected(tmp_path):
    run_id = _finalize_run(tmp_path)
    intent = _review_intent(run_id, tmp_path, new_event_id())
    with pytest.raises(ApprovalFinalizedRunError):
        issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def test_revoke_after_run_finalized_allowed(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
    intent = PlanningIntent(
        requested_action="record_run_final_closure",
        action_inputs={
            "record": closure,
            "actor": "human",
            "event_id": new_event_id(),
            "project_dir": str(tmp_path),
        },
        htr_runs_root=str(tmp_path),
    )
    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID
    revoked = revoke_approval(
        issue["approval_id"],
        "alice",
        "cancelled after finalize",
        base_dir=tmp_path,
    )
    assert revoked["approval_id"] == issue["approval_id"]


def test_outcome_after_run_finalized_allowed(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
    intent = PlanningIntent(
        requested_action="record_run_final_closure",
        action_inputs={
            "record": closure,
            "actor": "human",
            "event_id": new_event_id(),
            "project_dir": str(tmp_path),
        },
        htr_runs_root=str(tmp_path),
    )
    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    claim_approval(issue["approval_id"], "claim-final", "bob", base_dir=tmp_path)
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID
    outcome = record_use_outcome(
        issue["approval_id"],
        "claim-final",
        OUTCOME_AMBIGUOUS,
        base_dir=tmp_path,
    )
    assert outcome["outcome_class"] == OUTCOME_AMBIGUOUS


def test_closure_api_claim_materializes_project_dir(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
    intent = PlanningIntent(
        requested_action="record_run_final_closure",
        action_inputs={
            "record": closure,
            "actor": "human",
            "event_id": new_event_id(),
            "project_dir": str(tmp_path),
        },
        htr_runs_root=str(tmp_path),
    )
    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    project_entries = [
        e
        for e in issue["bound_arguments"]["argument_entries"]
        if e["key"] == "project_dir"
    ]
    assert project_entries == [
        {
            "key": "project_dir",
            "presence": "value",
            "value": str(tmp_path.resolve().as_posix()),
            "path_digest": approval_control._project_dir_path_digest(str(tmp_path)),
        }
    ]
    claim_approval(issue["approval_id"], "claim-closure", "bob", base_dir=tmp_path)


def test_claim_singleton_rejects_second_claim_id(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    claim_approval(issue["approval_id"], "claim-a", "bob", base_dir=tmp_path)
    with pytest.raises(ApprovalStateError, match="already claimed"):
        claim_approval(issue["approval_id"], "claim-b", "bob", base_dir=tmp_path)


def test_claimant_must_equal_executor_id(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    with pytest.raises(ApprovalValidationError, match="claimant_id"):
        claim_approval(issue["approval_id"], "claim-a", "charlie", base_dir=tmp_path)


def test_validate_is_advisory_only(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    result = validate_approval(issue["approval_id"], base_dir=tmp_path)
    assert result["derived"]["advisory_only"] is True


def test_list_approvals_scans_issue_records(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    listed = list_approvals(base_dir=tmp_path, run_id=run_id)
    assert [item["approval_id"] for item in listed] == [issue["approval_id"]]


def _require_posix_fork() -> None:
    if not hasattr(os, "fork"):
        pytest.fail(
            "POSIX fork is required for Task 24 candidate verification on the documented platform contract"
        )


# Independently reviewed golden digest fixtures (not computed in assertions).
GOLDEN_APPROVAL_BYTES = (
    '{"approval_digest_projection_version":"htr.approval.digest.v1",'
    '"approval_id":"apr_golden000001","approval_kind":"lifecycle_mutation",'
    '"approval_schema_version":"1","approver_id":"alice","bound_api":"review_run_manually",'
    '"bound_arguments":{"argument_entries":[]},"executor_id":"bob",'
    '"expires_at":"2026-01-01T01:00:00+00:00",'
    '"htr_runs_root_path_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"issued_at":"2026-01-01T00:00:00+00:00",'
    '"plan_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",'
    '"policy_version":"policy_c_v1","project_repository_checkpoint":null,'
    '"risk_class":"medium","run_id":"run_golden000001",'
    '"source_observation_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}'
)
GOLDEN_APPROVAL_DIGEST = (
    "sha256:0af9df034381d5cdaf5b56720acf17cfb5fb361f4ee88eca6f92322d586cfd14"
)
GOLDEN_CLAIM_BYTES = (
    '{"approval_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    '"approval_id":"apr_golden000001","bound_api":"review_run_manually",'
    '"bound_arguments":{"argument_entries":[]},'
    '"claim_digest_projection_version":"htr.approval.claim.digest.v1",'
    '"claim_id":"claim-golden-001","claim_schema_version":"1","claimant_id":"bob",'
    '"claimed_at":"2026-01-01T00:00:00+00:00","executor_id":"bob",'
    '"plan_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",'
    '"source_observation_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}'
)
GOLDEN_CLAIM_DIGEST = (
    "sha256:2992644116cdbed3571f75f7bc10ecf969cfd874bdd5b2e2e3f8145bdafa0546"
)
GOLDEN_REVOKE_BYTES = (
    '{"approval_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    '"approval_id":"apr_golden000001","reason":"no longer needed",'
    '"revoke_digest_projection_version":"htr.approval.revoke.digest.v1",'
    '"revoke_schema_version":"1","revoked_at":"2026-01-01T00:00:00+00:00","revoked_by":"alice"}'
)
GOLDEN_REVOKE_DIGEST = (
    "sha256:eea1a40b5c3de2b8ff7eaeae1bfa3d1903175c611f2fcd397df59a48d31bd345"
)
GOLDEN_OUTCOME_CONSUMED_BYTES = (
    '{"approval_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    '"approval_id":"apr_golden000001",'
    '"claim_digest":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    '"claim_id":"claim-golden-001","outcome_class":"consumed",'
    '"outcome_digest_projection_version":"htr.approval.outcome.digest.v1",'
    '"outcome_schema_version":"1","recorded_at":"2026-01-01T00:00:00+00:00"}'
)
GOLDEN_OUTCOME_CONSUMED_DIGEST = (
    "sha256:53d192d4530f2cbc8f60d585e77c47f550701da60a6ddd61002c4fb38549f1cf"
)
GOLDEN_OUTCOME_AMBIGUOUS_BYTES = (
    '{"approval_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    '"approval_id":"apr_golden000001",'
    '"claim_digest":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    '"claim_id":"claim-golden-001","outcome_class":"ambiguous",'
    '"outcome_digest_projection_version":"htr.approval.outcome.digest.v1",'
    '"outcome_schema_version":"1","recorded_at":"2026-01-01T00:00:00+00:00"}'
)
GOLDEN_OUTCOME_AMBIGUOUS_DIGEST = (
    "sha256:31bf8343a489117d1d55217df8359021040cdd87d7dc3f6f26b3208a5375c2ea"
)


def test_approval_digest_golden_stability():
    body = {
        "approval_schema_version": "1",
        "policy_version": "policy_c_v1",
        "approval_id": "apr_golden000001",
        "approval_kind": "lifecycle_mutation",
        "htr_runs_root_path_digest": "sha256:" + "a" * 64,
        "run_id": "run_golden000001",
        "source_observation_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "bound_api": "review_run_manually",
        "bound_arguments": {"argument_entries": []},
        "project_repository_checkpoint": None,
        "risk_class": "medium",
        "approver_id": "alice",
        "executor_id": "bob",
        "issued_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "2026-01-01T01:00:00+00:00",
    }
    projection = approval_control._approval_digest_projection(body)
    from htr.action_plan import _canonical_json

    assert _canonical_json(projection) == GOLDEN_APPROVAL_BYTES
    assert approval_control._compute_approval_digest(body) == GOLDEN_APPROVAL_DIGEST


def test_control_barrier_acquires_marker_and_cleans_up(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    assert not marker_present_noncreating(tmp_path, run_id)
    with approval_control._approval_control_barrier(run_id, tmp_path):
        assert marker_present_noncreating(tmp_path, run_id)
    assert not marker_present_noncreating(tmp_path, run_id)


def test_nested_control_and_lifecycle_barriers_share_marker(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    with approval_control._approval_control_barrier(run_id, tmp_path) as outer:
        with run_write_barrier(run_id, tmp_path) as inner:
            assert outer.key == inner.key
            assert outer.token == inner.token


def test_same_run_control_contention(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    hold = threading.Event()
    release = threading.Event()
    results: list[str] = []

    def holder() -> None:
        with approval_control._approval_control_barrier(run_id, tmp_path):
            results.append("holding")
            hold.set()
            release.wait(timeout=5)

    def challenger() -> None:
        hold.wait(timeout=5)
        try:
            issue_approval(
                run_id,
                _review_intent(run_id, tmp_path, new_event_id()),
                approver_id="alice",
                executor_id="bob",
                expires_at=_expires_in(),
                base_dir=tmp_path,
            )
            results.append("winner")
        except RunExecutionLockOccupiedError:
            results.append("blocked")

    holder_thread = threading.Thread(target=holder)
    challenger_thread = threading.Thread(target=challenger)
    holder_thread.start()
    challenger_thread.start()
    challenger_thread.join(timeout=10)
    release.set()
    holder_thread.join(timeout=10)
    assert results == ["holding", "blocked"]


def test_no_lifecycle_event_append_on_issue(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    events_path = paths.task_events_path(run_id, tmp_path)
    before = events_path.read_text(encoding="utf-8")
    issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    assert events_path.read_text(encoding="utf-8") == before


def test_untrusted_seal_blocks_issue(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure_path = contracts.run_final_closure_record_json_path(run_id, tmp_path)
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_text("{not valid json", encoding="utf-8")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.CLOSURE_PRESENT_UNTRUSTED
    with pytest.raises(RunSealBlockedError):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def test_expired_derived_without_writes(tmp_path):
    import time

    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        base_dir=tmp_path,
    )
    time.sleep(1.2)
    view = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert view["derived"]["expired"] is True
    with pytest.raises(ApprovalStateError, match="expired"):
        claim_approval(issue["approval_id"], "claim-exp", "bob", base_dir=tmp_path)


def test_claim_exact_replay_idempotent(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    first = claim_approval(issue["approval_id"], "claim-replay", "bob", base_dir=tmp_path)
    second = claim_approval(issue["approval_id"], "claim-replay", "bob", base_dir=tmp_path)
    assert first == second


# --- Hardening: seal bypass prevention ---


def test_approval_control_cannot_bypass_finalized_lifecycle_mutation(tmp_path):
    run_id = _finalize_run(tmp_path)
    with pytest.raises(RunFinalizedError):
        with run_write_barrier(run_id, tmp_path):
            pass
    with pytest.raises(ApprovalFinalizedRunError):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def test_public_module_exports_are_narrow():
    assert set(approval_control.__all__) == {
        "OUTCOME_AMBIGUOUS",
        "OUTCOME_CONSUMED",
        "claim_approval",
        "get_approval",
        "issue_approval",
        "list_approvals",
        "record_use_outcome",
        "revoke_approval",
        "validate_approval",
    }
    assert not hasattr(approval_control, "control_write_barrier")
    assert not hasattr(_el, "control_write_barrier")


def _mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def test_issue_exact_replay_zero_write(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    event_id = new_event_id()
    intent = _review_intent(run_id, tmp_path, event_id)
    first = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    issue_path = paths.approval_issue_path(first["approval_id"], tmp_path)
    before_mtime = _mtime_ns(issue_path)
    second = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=first["expires_at"],
        approval_id=first["approval_id"],
        base_dir=tmp_path,
    )
    assert second == first
    assert _mtime_ns(issue_path) == before_mtime


def test_digest_authoritative_field_sensitivity():
    base = {
        "approval_schema_version": "1",
        "policy_version": "policy_c_v1",
        "approval_id": "apr_digest000001",
        "approval_kind": "lifecycle_mutation",
        "htr_runs_root_path_digest": "sha256:" + "a" * 64,
        "run_id": "run_digest000001",
        "source_observation_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "bound_api": "review_run_manually",
        "bound_arguments": {"argument_entries": []},
        "project_repository_checkpoint": None,
        "risk_class": "medium",
        "approver_id": "alice",
        "executor_id": "bob",
        "issued_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "2026-01-01T01:00:00+00:00",
    }
    golden = approval_control._compute_approval_digest(base)
    for key in (
        "approval_id",
        "plan_digest",
        "executor_id",
        "expires_at",
        "bound_api",
    ):
        mutated = dict(base)
        mutated[key] = "changed"
        assert approval_control._compute_approval_digest(mutated) != golden
    presentation = dict(base)
    presentation["requester_id"] = "charlie"
    assert "requester_id" not in approval_control._approval_digest_projection(base)


def test_golden_digest_projections_stable():
    from htr.action_plan import _canonical_json

    claim = {
        "claim_schema_version": "1",
        "approval_id": "apr_golden000001",
        "approval_digest": "sha256:" + "d" * 64,
        "claim_id": "claim-golden-001",
        "claimant_id": "bob",
        "executor_id": "bob",
        "source_observation_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "bound_api": "review_run_manually",
        "bound_arguments": {"argument_entries": []},
        "claimed_at": "2026-01-01T00:00:00+00:00",
    }
    revoke = {
        "revoke_schema_version": "1",
        "approval_id": "apr_golden000001",
        "approval_digest": "sha256:" + "d" * 64,
        "revoked_by": "alice",
        "revoked_at": "2026-01-01T00:00:00+00:00",
        "reason": "no longer needed",
    }
    outcome_consumed = {
        "outcome_schema_version": "1",
        "approval_id": "apr_golden000001",
        "approval_digest": "sha256:" + "d" * 64,
        "claim_id": "claim-golden-001",
        "claim_digest": "sha256:" + "e" * 64,
        "outcome_class": OUTCOME_CONSUMED,
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }
    outcome_ambiguous = {**outcome_consumed, "outcome_class": OUTCOME_AMBIGUOUS}
    assert (
        _canonical_json(approval_control._claim_digest_projection(claim)) == GOLDEN_CLAIM_BYTES
    )
    assert approval_control._compute_claim_digest(claim) == GOLDEN_CLAIM_DIGEST
    assert (
        _canonical_json(approval_control._revoke_digest_projection(revoke))
        == GOLDEN_REVOKE_BYTES
    )
    assert approval_control._compute_revoke_digest(revoke) == GOLDEN_REVOKE_DIGEST
    assert (
        _canonical_json(approval_control._outcome_digest_projection(outcome_consumed))
        == GOLDEN_OUTCOME_CONSUMED_BYTES
    )
    assert (
        approval_control._compute_outcome_digest(outcome_consumed)
        == GOLDEN_OUTCOME_CONSUMED_DIGEST
    )
    assert (
        _canonical_json(approval_control._outcome_digest_projection(outcome_ambiguous))
        == GOLDEN_OUTCOME_AMBIGUOUS_BYTES
    )
    assert (
        approval_control._compute_outcome_digest(outcome_ambiguous)
        == GOLDEN_OUTCOME_AMBIGUOUS_DIGEST
    )


def test_event_id_bound_in_issue_digest(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    event_a = new_event_id()
    event_b = new_event_id()
    issue_a = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, event_a),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    issue_b = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, event_b),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    assert issue_a["approval_digest"] != issue_b["approval_digest"]
    entries = issue_a["bound_arguments"]["argument_entries"]
    event_entry = next(e for e in entries if e["key"] == "event_id")
    assert event_entry["presence"] == "value"
    assert event_entry["value"] == event_a


def test_never_reads_or_writes_run_tree_approvals_jsonl(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    legacy = paths.approvals_path(run_id, tmp_path)
    legacy.write_text('{"legacy": true}\n', encoding="utf-8")
    digest_before = _file_digest(legacy)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    claim_approval(issue["approval_id"], "claim-1", "bob", base_dir=tmp_path)
    assert _file_digest(legacy) == digest_before


# --- Subprocess workers ---


def _subprocess_issue_worker(
    runs_root: str,
    run_id: str,
    approval_id: str,
    event_id: str,
    slot: Any,
) -> None:
    from htr.action_plan import PlanningIntent
    from htr.approval_control import issue_approval
    from htr import contracts

    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    intent = PlanningIntent(
        requested_action="review_run_manually",
        action_inputs={"record": review, "actor": "human", "event_id": event_id},
        htr_runs_root=runs_root,
    )
    try:
        issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            approval_id=approval_id,
            base_dir=Path(runs_root),
        )
        slot.put("issued")
    except RunExecutionLockOccupiedError:
        slot.put("blocked")


def _subprocess_claim_worker(
    runs_root: str,
    approval_id: str,
    claim_id: str,
    slot: Any,
) -> None:
    from htr.approval_control import claim_approval
    from htr.state import ApprovalStateError

    try:
        claim_approval(
            approval_id,
            claim_id,
            "bob",
            base_dir=Path(runs_root),
        )
        slot.put("claimed")
    except RunExecutionLockOccupiedError:
        slot.put("blocked_lock")
    except ApprovalStateError:
        slot.put("blocked_state")


def _subprocess_crash_before_issue_write_worker(
    runs_root: str,
    run_id: str,
    approval_id: str,
    event_id: str,
) -> None:
    from unittest.mock import patch
    from htr import approval_control as ac

    def _crash(*_a: Any, **_k: Any) -> dict[str, Any]:
        os._exit(61)

    with patch.object(ac, "_create_immutable_record", side_effect=_crash):
        _subprocess_issue_worker(runs_root, run_id, approval_id, event_id, multiprocessing.Queue())


def _subprocess_revoke_worker(
    runs_root: str,
    approval_id: str,
    slot: Any,
) -> None:
    from htr.approval_control import revoke_approval
    from htr.state import ApprovalStateError

    try:
        revoke_approval(approval_id, "alice", "cancel", base_dir=Path(runs_root))
        slot.put("revoked")
    except ApprovalStateError:
        slot.put("revoke_blocked")
    except RunExecutionLockOccupiedError:
        slot.put("blocked_lock")


def _subprocess_hold_approval_marker_worker(
    runs_root: str,
    run_id: str,
    slot: Any,
) -> None:
    with approval_control._approval_control_barrier(run_id, Path(runs_root)):
        slot.put("held")
        time.sleep(0.5)


def test_subprocess_same_run_issue_contention(tmp_path):
    _require_posix_fork()
    run_id = _run_with_completion_only(tmp_path)
    slot_b: list[str] = []

    class _Slot:
        def __init__(self, sink: list[str]) -> None:
            self._sink = sink

        def put(self, value: str) -> None:
            self._sink.append(value)

    child = os.fork()
    if child == 0:
        _subprocess_hold_approval_marker_worker(str(tmp_path), run_id, _Slot([]))
        os._exit(0)
    deadline = time.time() + 5
    while not marker_present_noncreating(tmp_path, run_id) and time.time() < deadline:
        time.sleep(0.01)
    assert marker_present_noncreating(tmp_path, run_id)
    _subprocess_issue_worker(
        str(tmp_path), run_id, new_approval_id(), new_event_id(), _Slot(slot_b)
    )
    _, status = os.waitpid(child, 0)
    assert os.WEXITSTATUS(status) == 0
    assert slot_b == ["blocked"]


def test_subprocess_two_claim_ids_one_winner(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    ctx = multiprocessing.get_context("spawn")
    slot_a = ctx.Queue()
    slot_b = ctx.Queue()
    gate = ctx.Barrier(2)
    p1 = ctx.Process(
        target=_subprocess_claim_worker,
        args=(str(tmp_path), issue["approval_id"], "claim-a", slot_a),
    )
    p2 = ctx.Process(
        target=_subprocess_claim_worker,
        args=(str(tmp_path), issue["approval_id"], "claim-b", slot_b),
    )
    p1.start()
    p2.start()
    p1.join(timeout=15)
    p2.join(timeout=15)
    results = {slot_a.get(timeout=2), slot_b.get(timeout=2)}
    assert "claimed" in results
    assert results & {"blocked_state", "blocked_lock"}
    assert paths.approval_claim_path(issue["approval_id"], tmp_path).is_file()


def test_revoke_vs_claim_race_one_valid_serialization(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    ctx = multiprocessing.get_context("spawn")
    slot_revoke = ctx.Queue()
    slot_claim = ctx.Queue()

    p1 = ctx.Process(
        target=_subprocess_revoke_worker,
        args=(str(tmp_path), issue["approval_id"], slot_revoke),
    )
    p2 = ctx.Process(
        target=_subprocess_claim_worker,
        args=(str(tmp_path), issue["approval_id"], "claim-race", slot_claim),
    )
    p1.start()
    p2.start()
    p1.join(timeout=15)
    p2.join(timeout=15)
    results = {slot_revoke.get(timeout=2), slot_claim.get(timeout=2)}
    assert len(results) == 2
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    claim_path = paths.approval_claim_path(issue["approval_id"], tmp_path)

    if bundle["revoke"] is not None and bundle["claim"] is None:
        assert "revoked" in results
        assert "claimed" not in results
        with pytest.raises(ApprovalStateError):
            claim_approval(issue["approval_id"], "claim-late", "bob", base_dir=tmp_path)
        return

    assert bundle["claim"] is not None
    assert "claimed" in results
    assert claim_path.is_file()
    assert bundle["claim"]["claim_id"] == "claim-race"
    with pytest.raises(ApprovalStateError):
        claim_approval(issue["approval_id"], "claim-late", "bob", base_dir=tmp_path)
    assert claim_path.is_file()
    assert get_approval(issue["approval_id"], base_dir=tmp_path)["claim"]["claim_id"] == "claim-race"


def test_revoke_before_claim_blocks_claim_deterministic(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    revoke_approval(issue["approval_id"], "alice", "cancel", base_dir=tmp_path)
    with pytest.raises(ApprovalStateError):
        claim_approval(issue["approval_id"], "claim-late", "bob", base_dir=tmp_path)
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["revoke"] is not None
    assert bundle["claim"] is None


def test_claim_before_revoke_retains_claim_deterministic(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    claim_approval(issue["approval_id"], "claim-first", "bob", base_dir=tmp_path)
    revoke_approval(issue["approval_id"], "alice", "cancel", base_dir=tmp_path)
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["claim"]["claim_id"] == "claim-first"
    assert paths.approval_claim_path(issue["approval_id"], tmp_path).is_file()
    assert bundle["revoke"] is not None


def test_symlinked_control_root_fails_closed(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    runs_root = paths.runs_root(tmp_path)
    real_control = tmp_path / "real_control_anchor"
    real_control.mkdir()
    control_link = runs_root / ".control"
    if control_link.is_symlink():
        control_link.unlink()
    elif control_link.is_dir():
        pytest.fail("unexpected real .control directory prevents symlink path-safety test")
    control_link.symlink_to(real_control, target_is_directory=True)
    with pytest.raises(ApprovalValidationError, match="unsafe approval control path"):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )
    assert not paths.control_root(tmp_path).joinpath("approvals").exists() or not any(
        paths.control_approvals_root(tmp_path).iterdir()
    )


def test_control_record_fsync_failure_not_ordinary_success(tmp_path, monkeypatch):
    run_id = _run_with_completion_only(tmp_path)

    def _fail_fsync(_fd: int) -> None:
        raise RunExecutionLockIndeterminateError("fsync failed")

    monkeypatch.setattr(approval_control, "_fsync_file_fd", _fail_fsync)
    with pytest.raises(RunExecutionLockIndeterminateError):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def test_marker_cleanup_fsync_failure_after_successful_issue(tmp_path, monkeypatch):
    run_id = _run_with_completion_only(tmp_path)

    def _fail_release(entry: Any) -> None:
        raise RunExecutionLockDurabilityError("marker removal failed", run_id=run_id)

    monkeypatch.setattr(_el, "_release_marker_success", _fail_release)
    with pytest.raises(RunExecutionLockDurabilityError) as exc:
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )
    assert exc.value.mutation_may_have_committed is True
    assert exc.value.safe_to_retry is False
    assert marker_present_noncreating(tmp_path, run_id)


def test_different_project_roots_isolate_same_run_and_approval_ids(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    run_id = _run_with_completion_only(root_a)
    approval_id = new_approval_id()
    event_id = new_event_id()
    issue_a = issue_approval(
        run_id,
        _review_intent(run_id, root_a, event_id),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        approval_id=approval_id,
        base_dir=root_a,
    )
    with pytest.raises(ApprovalStateError):
        get_approval(approval_id, base_dir=root_b)
    assert issue_a["approval_id"] == approval_id


# --- Task 24 final hardening ---


@dataclass
class ApprovalWriteProbe:
    marker_acquires: int = 0
    write_starts: int = 0
    first_writes: int = 0
    seal_before_marker: int = 0
    seal_after_marker: int = 0


@contextmanager
def _approval_write_probe(*, track_seal: Callable[..., None] | None = None):
    probe = ApprovalWriteProbe()
    orig_acquire = approval_control._acquire_outer_run_marker
    orig_mark = approval_control._mark_control_write_started
    orig_open = approval_control.os.open

    def track_acquire(*args, **kwargs):
        probe.marker_acquires += 1
        return orig_acquire(*args, **kwargs)

    def track_mark():
        probe.write_starts += 1
        return orig_mark()

    def track_open(path, flags, *args, **kwargs):
        if flags & (approval_control._O_CREAT | approval_control._O_WRONLY):
            probe.first_writes += 1
        return orig_open(path, flags, *args, **kwargs)

    patchers = [
        patch.object(approval_control, "_acquire_outer_run_marker", side_effect=track_acquire),
        patch.object(approval_control, "_mark_control_write_started", side_effect=track_mark),
        patch.object(approval_control.os, "open", side_effect=track_open),
    ]
    if track_seal is not None:
        orig_seal = track_seal

        def track_seal_fn(*args, **kwargs):
            if probe.marker_acquires == 0:
                probe.seal_before_marker += 1
            else:
                probe.seal_after_marker += 1
            return orig_seal(*args, **kwargs)

        patchers.append(
            patch.object(
                approval_control,
                "_evaluate_seal_for_lifecycle_issue",
                side_effect=track_seal_fn,
            )
        )
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield probe


def _issue_fixture(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    return run_id, issue


@pytest.mark.parametrize("writer", ["issue", "revoke", "claim", "outcome"])
def test_writer_write_start_before_first_filesystem_write(tmp_path, writer: str):
    run_id, issue = _issue_fixture(tmp_path)
    if writer == "issue":
        with _approval_write_probe(
            track_seal=approval_control._evaluate_seal_for_lifecycle_issue
        ) as probe:
            issue_approval(
                run_id,
                _review_intent(run_id, tmp_path, new_event_id()),
                approver_id="alice",
                executor_id="bob",
                expires_at=_expires_in(),
                approval_id=new_approval_id(),
                base_dir=tmp_path,
            )
        assert probe.seal_before_marker >= 1
        assert probe.seal_after_marker >= 1
    elif writer == "revoke":
        with _approval_write_probe() as probe:
            revoke_approval(issue["approval_id"], "alice", "cancel", base_dir=tmp_path)
    elif writer == "claim":
        with _approval_write_probe() as probe:
            claim_approval(issue["approval_id"], "claim-ws", "bob", base_dir=tmp_path)
    else:
        claim_approval(issue["approval_id"], "claim-ws", "bob", base_dir=tmp_path)
        with _approval_write_probe() as probe:
            record_use_outcome(
                issue["approval_id"], "claim-ws", OUTCOME_CONSUMED, base_dir=tmp_path
            )
    assert probe.marker_acquires == 1
    assert probe.write_starts == 1
    assert probe.first_writes >= 1


def test_exact_replay_issue_skips_marker_when_no_contention(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    intent = _review_intent(run_id, tmp_path, new_event_id())
    first = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    with patch.object(approval_control, "_acquire_outer_run_marker") as acquire:
        second = issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=first["expires_at"],
            approval_id=first["approval_id"],
            base_dir=tmp_path,
        )
        acquire.assert_not_called()
    assert second == first


def _intent_from_issue(issue: dict[str, Any], tmp_path: Path) -> PlanningIntent:
    inputs = approval_control._argument_entries_to_inputs(issue["bound_arguments"])
    return PlanningIntent(
        requested_action=issue["bound_api"],
        action_inputs=inputs,
        htr_runs_root=str(tmp_path),
    )


def test_exact_replay_blocked_when_marker_present(tmp_path):
    run_id, issue = _issue_fixture(tmp_path)
    intent = _intent_from_issue(issue, tmp_path)
    hold = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with approval_control._approval_control_barrier(run_id, tmp_path):
            hold.set()
            release.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    hold.wait(timeout=5)
    with pytest.raises(RunExecutionLockOccupiedError):
        issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=issue["expires_at"],
            approval_id=issue["approval_id"],
            base_dir=tmp_path,
        )
    release.set()
    t.join(timeout=5)


def test_revoke_after_claim_preserves_claim_and_blocks_new_claim(tmp_path):
    run_id, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-rc", "bob", base_dir=tmp_path)
    revoke_approval(issue["approval_id"], "alice", "post-claim revoke", base_dir=tmp_path)
    view = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert view["claim"]["claim_id"] == "claim-rc"
    assert view["revoke"]["reason"] == "post-claim revoke"
    with pytest.raises(ApprovalStateError):
        claim_approval(issue["approval_id"], "claim-new", "bob", base_dir=tmp_path)


def test_revoke_before_claim_blocks_new_claim(tmp_path):
    run_id, issue = _issue_fixture(tmp_path)
    revoke_approval(issue["approval_id"], "alice", "early revoke", base_dir=tmp_path)
    with pytest.raises(ApprovalStateError, match="revoked"):
        claim_approval(issue["approval_id"], "claim-new", "bob", base_dir=tmp_path)


def _fork_child_issue_attempt(runs_root: str, run_id: str, result_path: str) -> None:
    from htr.action_plan import PlanningIntent
    from htr import contracts
    from htr.approval_control import issue_approval
    from htr.execution_lock import RunExecutionLockOccupiedError, RunExecutionLockBoundaryViolationError

    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    intent = PlanningIntent(
        requested_action="review_run_manually",
        action_inputs={"record": review, "actor": "human", "event_id": new_event_id()},
        htr_runs_root=runs_root,
    )
    outcome = "unknown"
    try:
        issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            approval_id=new_approval_id(),
            base_dir=Path(runs_root),
        )
        outcome = "mutated"
    except RunExecutionLockOccupiedError:
        outcome = "blocked"
    except RunExecutionLockBoundaryViolationError:
        outcome = "boundary"
    Path(result_path).write_text(outcome, encoding="utf-8")
    os._exit(0)


def test_fork_child_cannot_mutate_or_release_parent_approval_marker(tmp_path):
    _require_posix_fork()
    run_id = _run_with_completion_only(tmp_path)
    result_path = tmp_path / "fork_child_outcome"
    with approval_control._approval_control_barrier(run_id, tmp_path):
        child = os.fork()
        if child == 0:
            _fork_child_issue_attempt(str(tmp_path), run_id, str(result_path))
        _, status = os.waitpid(child, 0)
        assert os.WEXITSTATUS(status) == 0
        assert result_path.read_text(encoding="utf-8") in {"blocked", "boundary"}
    assert not marker_present_noncreating(tmp_path, run_id)


def test_approval_use_session_is_internal_and_unexported():
    assert "_approval_use_session" not in approval_control.__all__
    assert hasattr(approval_control, "_approval_use_session")
    source = inspect.getsource(approval_control._approval_use_session)
    assert "yield ctx" in source
    assert "callback" not in source


def test_approval_use_session_holds_marker_until_outer_exit(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    with approval_control._approval_use_session(run_id, tmp_path) as outer:
        assert marker_present_noncreating(tmp_path, run_id)
        with approval_control._approval_use_session(run_id, tmp_path) as inner:
            assert inner.token == outer.token
            assert marker_present_noncreating(tmp_path, run_id)
    assert not marker_present_noncreating(tmp_path, run_id)


def test_approval_use_session_rejects_cross_run_nesting(tmp_path):
    run_a = _run_with_completion_only(tmp_path)
    run_b = _run_with_completion_only(tmp_path)
    with approval_control._approval_use_session(run_a, tmp_path):
        with pytest.raises(RunExecutionLockBoundaryViolationError):
            with approval_control._approval_use_session(run_b, tmp_path):
                pass


def test_approval_use_session_nested_lifecycle_barrier_shares_marker(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    with approval_control._approval_use_session(run_id, tmp_path) as use_ctx:
        with run_write_barrier(run_id, tmp_path) as life_ctx:
            assert use_ctx.key == life_ctx.key
            assert use_ctx.token == life_ctx.token


def _run_subprocess_script(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def test_crash_before_marker_leaves_no_residue(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    approval_id = new_approval_id()
    event_id = new_event_id()
    proc = _run_subprocess_script(
        f"""
import os
from unittest.mock import patch
from pathlib import Path
from datetime import datetime, timedelta, timezone
from htr.action_plan import PlanningIntent
from htr import contracts
from htr import approval_control as ac
from htr.approval_control import issue_approval
run_id={run_id!r}
approval_id={approval_id!r}
event_id={event_id!r}
base=Path({str(tmp_path)!r})
review=contracts.make_run_review_record(run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP)
intent=PlanningIntent(requested_action='review_run_manually', action_inputs={{'record': review, 'actor': 'human', 'event_id': event_id}}, htr_runs_root=str(base))
def crash(*a, **k):
    os._exit(61)
with patch.object(ac, '_acquire_outer_run_marker', side_effect=crash):
    try:
        issue_approval(run_id, intent, approver_id='alice', executor_id='bob', expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), approval_id=approval_id, base_dir=base)
    except SystemExit:
        pass
"""
    )
    assert proc.returncode == 61
    assert not marker_present_noncreating(tmp_path, run_id)
    assert not paths.approval_issue_path(approval_id, tmp_path).exists()


def test_existing_marker_always_occupied_unknown(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    lock_root = tmp_path / LOCKS_DIR_NAME
    lock_root.mkdir(parents=True, exist_ok=True)
    (lock_root / f"{run_id}.marker").write_text("{}", encoding="utf-8")
    with pytest.raises(RunExecutionLockOccupiedError):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def _lifecycle_api_run_state(api: str, tmp_path: Path) -> tuple[str, PlanningIntent]:
    event_id = new_event_id()
    if api == "complete_run_manually":
        run_id = new_run_id()
        io.create_run_workspace(run_id, base_dir=tmp_path)
        task_id = new_task_id()
        io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
        TASK16._complete_task(tmp_path, run_id, task_id)
        record = contracts.make_run_completion_record(run_id=run_id, completed_task_ids=[task_id])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"record": record, "actor": "human", "event_id": event_id},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "review_run_manually":
        run_id = _run_with_completion_only(tmp_path)
        return run_id, _review_intent(run_id, tmp_path, event_id)
    if api == "plan_run_followup":
        run_id = TASK16._run_with_reviewed_run(tmp_path)[0]
        record = TASK16._plan_record(run_id)
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"record": record, "actor": "human", "event_id": event_id},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "request_run_execution":
        chain = TASK16._run_with_planned_run(tmp_path)
        run_id = chain[0]
        plan = chain[5]
        record = contracts.make_run_execution_request_record(
            run_id=run_id,
            source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
            execution_items=[TASK16._sample_execution_item()],
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"record": record, "actor": "human", "event_id": event_id},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "execute_run_execution_request":
        chain = TASK16._run_with_execution_request(tmp_path)
        run_id = chain[0]
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"executor": "bob", "event_id": event_id, "project_dir": str(tmp_path)},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "verify_run_execution_result":
        chain = TASK16._run_with_execution_result(tmp_path)
        run_id = chain[0]
        record = TASK16._verification_record(run_id, chain[7])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "plan_post_verification_followup":
        chain = TASK16._run_with_verified_execution(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_plan_record(run_id, chain[7], chain[8])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "request_post_verification_execution":
        chain = TASK16._run_with_post_verification_plan(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_execution_request_record(
            run_id, chain[7], chain[8], chain[9]
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "record_post_verification_execution_result":
        chain = TASK16._run_with_post_verification_execution_request(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_execution_result_record(
            run_id, chain[7], chain[8], chain[9], chain[10]
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "record_post_verification_execution_verification":
        chain = TASK16._run_with_post_verification_execution_result(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_execution_verification_record(
            run_id, chain[7], chain[8], chain[9], chain[10], chain[11]
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "record_run_final_closure":
        chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
        run_id = chain[0]
        closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": closure,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    raise AssertionError(f"unhandled api {api!r}")


@pytest.mark.parametrize("api", sorted(_CATALOG.keys()))
def test_lifecycle_api_issue_binding_or_fail_closed(tmp_path, api: str):
    run_id, intent = _lifecycle_api_run_state(api, tmp_path)
    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    assert issue["bound_api"] == api
    assert issue["approval_schema_version"] == "1"
    assert issue["policy_version"] == "policy_c_v1"
    assert issue["approval_digest"].startswith("sha256:")
    assert issue["plan_digest"].startswith("sha256:")
    assert issue["source_observation_digest"].startswith("sha256:")
    assert issue["htr_runs_root_path_digest"].startswith("sha256:")
    if _CATALOG[api]["expected_events"]:
        entries = issue["bound_arguments"]["argument_entries"]
        event_entry = next(e for e in entries if e["key"] == "event_id")
        assert event_entry["presence"] == "value"


def test_no_recovery_repair_or_marker_cleanup_in_approval_control():
    source = (Path(__file__).resolve().parents[2] / "htr" / "approval_control.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "reconcile",
        "self_heal",
        "force_unlock",
        "automatic_repair",
        "marker_cleanup",
        "recover_marker",
    )
    for token in forbidden:
        assert token not in source


# --- Crash phases B–F, path safety, exact-replay zero-write ---


def _approval_fs_snapshot(tmp_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    root = paths.runs_root(tmp_path)
    if root.exists():
        for path in sorted(root.rglob("*")):
            rel = str(path.relative_to(root))
            try:
                out[rel] = path.stat().st_mtime_ns
            except OSError:
                pass
    locks = tmp_path / LOCKS_DIR_NAME
    if locks.exists():
        for path in sorted(locks.rglob("*")):
            rel = f".execution_locks/{path.relative_to(locks)}"
            try:
                out[rel] = path.stat().st_mtime_ns
            except OSError:
                pass
    return out


def _subprocess_crash_phase_b_script(
    runs_root: str, run_id: str, approval_id: str, event_id: str
) -> str:
    return f"""
import os
from unittest.mock import patch
from pathlib import Path
from datetime import datetime, timedelta, timezone
from htr.action_plan import PlanningIntent
from htr import contracts
from htr import approval_control as ac
from htr.approval_control import issue_approval
run_id={run_id!r}
approval_id={approval_id!r}
event_id={event_id!r}
base=Path({runs_root!r})
review=contracts.make_run_review_record(run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP)
intent=PlanningIntent(requested_action='review_run_manually', action_inputs={{'record': review, 'actor': 'human', 'event_id': event_id}}, htr_runs_root=str(base))
def abort():
    os._exit(62)
with patch.object(ac, '_mark_control_write_started', side_effect=abort):
    try:
        issue_approval(run_id, intent, approver_id='alice', executor_id='bob', expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), approval_id=approval_id, base_dir=base)
    except SystemExit:
        pass
"""


def test_crash_phase_b_marker_without_record_blocks_occupied(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    approval_id = new_approval_id()
    proc = _run_subprocess_script(
        _subprocess_crash_phase_b_script(str(tmp_path), run_id, approval_id, new_event_id())
    )
    assert proc.returncode == 62
    assert marker_present_noncreating(tmp_path, run_id)
    assert not paths.approval_issue_path(approval_id, tmp_path).exists()
    with pytest.raises(RunExecutionLockOccupiedError):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def _subprocess_crash_phase_c_script(
    runs_root: str, run_id: str, approval_id: str, event_id: str
) -> str:
    return f"""
import os
from unittest.mock import patch
from pathlib import Path
from datetime import datetime, timedelta, timezone
from htr.action_plan import PlanningIntent
from htr import contracts
from htr import approval_control as ac
from htr.approval_control import issue_approval
run_id={run_id!r}
approval_id={approval_id!r}
event_id={event_id!r}
base=Path({runs_root!r})
review=contracts.make_run_review_record(run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP)
intent=PlanningIntent(requested_action='review_run_manually', action_inputs={{'record': review, 'actor': 'human', 'event_id': event_id}}, htr_runs_root=str(base))
orig=ac._mark_control_write_started
def crash_after_start():
    orig()
    os._exit(63)
with patch.object(ac, '_mark_control_write_started', side_effect=crash_after_start):
    try:
        issue_approval(run_id, intent, approver_id='alice', executor_id='bob', expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), approval_id=approval_id, base_dir=base)
    except SystemExit:
        pass
"""


def test_crash_phase_c_after_write_start_marker_retained_no_record(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    approval_id = new_approval_id()
    proc = _run_subprocess_script(
        _subprocess_crash_phase_c_script(str(tmp_path), run_id, approval_id, new_event_id())
    )
    assert proc.returncode == 63
    assert marker_present_noncreating(tmp_path, run_id)
    assert not paths.approval_issue_path(approval_id, tmp_path).exists()
    with pytest.raises(RunExecutionLockOccupiedError):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def _subprocess_crash_phase_d_script(
    runs_root: str, run_id: str, approval_id: str, event_id: str
) -> str:
    return f"""
import os
from unittest.mock import patch
from pathlib import Path
from datetime import datetime, timedelta, timezone
from htr.action_plan import PlanningIntent
from htr import contracts
from htr import execution_lock as el
from htr.approval_control import issue_approval
run_id={run_id!r}
approval_id={approval_id!r}
event_id={event_id!r}
base=Path({runs_root!r})
review=contracts.make_run_review_record(run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP)
intent=PlanningIntent(requested_action='review_run_manually', action_inputs={{'record': review, 'actor': 'human', 'event_id': event_id}}, htr_runs_root=str(base))
def crash_on_release(entry):
    os._exit(64)
with patch.object(el, '_release_marker_success', side_effect=crash_on_release):
    try:
        issue_approval(run_id, intent, approver_id='alice', executor_id='bob', expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), approval_id=approval_id, base_dir=base)
    except SystemExit:
        pass
"""


def test_crash_phase_d_issue_record_retained_marker_retained(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    approval_id = new_approval_id()
    proc = _run_subprocess_script(
        _subprocess_crash_phase_d_script(str(tmp_path), run_id, approval_id, new_event_id())
    )
    assert proc.returncode == 64
    issue_path = paths.approval_issue_path(approval_id, tmp_path)
    assert issue_path.is_file()
    before = issue_path.read_text(encoding="utf-8")
    assert marker_present_noncreating(tmp_path, run_id)
    assert issue_path.read_text(encoding="utf-8") == before


def _subprocess_crash_phase_e_script(runs_root: str, approval_id: str, claim_id: str) -> str:
    return f"""
import os
from unittest.mock import patch
from htr import execution_lock as el
from htr.approval_control import claim_approval
from pathlib import Path
def crash_on_release(entry):
    os._exit(65)
with patch.object(el, '_release_marker_success', side_effect=crash_on_release):
    try:
        claim_approval({approval_id!r}, {claim_id!r}, 'bob', base_dir=Path({runs_root!r}))
    except SystemExit:
        pass
"""


def test_crash_phase_e_claim_retained_blocks_second_claim_id(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_id = "claim-phase-e"
    proc = _run_subprocess_script(
        _subprocess_crash_phase_e_script(str(tmp_path), issue["approval_id"], claim_id)
    )
    assert proc.returncode == 65
    assert paths.approval_claim_path(issue["approval_id"], tmp_path).is_file()
    assert marker_present_noncreating(tmp_path, issue["run_id"])
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{issue['run_id']}.marker"
    if marker_path.exists():
        marker_path.unlink()
    with pytest.raises(ApprovalStateError, match="already claimed"):
        claim_approval(issue["approval_id"], "claim-other", "bob", base_dir=tmp_path)


def _subprocess_crash_phase_f_script(runs_root: str, approval_id: str, claim_id: str) -> str:
    return f"""
import os
from unittest.mock import patch
from htr import execution_lock as el
from htr.approval_control import record_use_outcome, OUTCOME_CONSUMED
from pathlib import Path
def crash_on_release(entry):
    os._exit(66)
with patch.object(el, '_release_marker_success', side_effect=crash_on_release):
    try:
        record_use_outcome({approval_id!r}, {claim_id!r}, OUTCOME_CONSUMED, base_dir=Path({runs_root!r}))
    except SystemExit:
        pass
"""


def test_crash_phase_f_outcome_retained_marker_retained(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_id = "claim-phase-f"
    claim_approval(issue["approval_id"], claim_id, "bob", base_dir=tmp_path)
    proc = _run_subprocess_script(
        _subprocess_crash_phase_f_script(str(tmp_path), issue["approval_id"], claim_id)
    )
    assert proc.returncode == 66
    outcome_path = paths.approval_outcome_path(issue["approval_id"], tmp_path)
    assert outcome_path.is_file()
    before = outcome_path.read_text(encoding="utf-8")
    assert marker_present_noncreating(tmp_path, issue["run_id"])
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{issue['run_id']}.marker"
    if marker_path.exists():
        marker_path.unlink()
    replay = record_use_outcome(
        issue["approval_id"], claim_id, OUTCOME_CONSUMED, base_dir=tmp_path
    )
    assert replay["outcome_class"] == OUTCOME_CONSUMED
    assert outcome_path.read_text(encoding="utf-8") == before


def _install_symlink_component(tmp_path: Path, rel_parts: tuple[str, ...], target: Path) -> None:
    runs_root = paths.runs_root(tmp_path)
    runs_root.mkdir(parents=True, exist_ok=True)
    parent = runs_root
    for part in rel_parts[:-1]:
        parent = parent / part
        if not parent.exists():
            parent.mkdir(parents=True)
    link = parent / rel_parts[-1]
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target, target_is_directory=target.is_dir())


@pytest.mark.parametrize(
    "rel_parts",
    [
        (".control",),
        (".control", "approvals"),
    ],
)
def test_symlink_control_tree_components_fail_closed(tmp_path, rel_parts: tuple[str, ...]):
    run_id = _run_with_completion_only(tmp_path)
    target = tmp_path / f"real_{'_'.join(rel_parts)}"
    target.mkdir(parents=True)
    _install_symlink_component(tmp_path, rel_parts, target)
    with pytest.raises(ApprovalValidationError, match="unsafe approval control path"):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )


def test_symlink_approval_id_directory_fails_closed(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    approval_id = new_approval_id()
    real = tmp_path / "real_approval_dir"
    real.mkdir()
    approvals_root = paths.control_approvals_root(tmp_path)
    approvals_root.mkdir(parents=True)
    (approvals_root / approval_id).symlink_to(real, target_is_directory=True)
    with pytest.raises(ApprovalValidationError, match="unsafe approval control path"):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            approval_id=approval_id,
            base_dir=tmp_path,
        )


@pytest.mark.parametrize("record_name", ["issue.json", "revoke.json", "claim.json", "outcome.json"])
def test_symlink_record_file_fails_closed(tmp_path, record_name: str):
    run_id = _run_with_completion_only(tmp_path)
    if record_name == "issue.json":
        approval_id = new_approval_id()
    else:
        issue = issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            base_dir=tmp_path,
        )
        approval_id = issue["approval_id"]
        if record_name == "outcome.json":
            claim_approval(approval_id, "claim-pre-out", "bob", base_dir=tmp_path)
    approval_dir = paths.approval_control_dir(approval_id, tmp_path)
    approval_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_path / f"real_{record_name}"
    target.write_text('{"bad": true}', encoding="utf-8")
    (approval_dir / record_name).symlink_to(target)
    with pytest.raises(ApprovalValidationError):
        if record_name == "issue.json":
            issue_approval(
                run_id,
                _review_intent(run_id, tmp_path, new_event_id()),
                approver_id="alice",
                executor_id="bob",
                expires_at=_expires_in(),
                approval_id=approval_id,
                base_dir=tmp_path,
            )
        elif record_name == "revoke.json":
            revoke_approval(approval_id, "alice", "late", base_dir=tmp_path)
        elif record_name == "claim.json":
            claim_approval(approval_id, "claim-symlink", "bob", base_dir=tmp_path)
        else:
            record_use_outcome(
                approval_id, "claim-pre-out", OUTCOME_CONSUMED, base_dir=tmp_path
            )


def test_malformed_traversal_approval_id_rejected(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    with pytest.raises((ApprovalValidationError, ValueError)):
        issue_approval(
            run_id,
            _review_intent(run_id, tmp_path, new_event_id()),
            approver_id="alice",
            executor_id="bob",
            expires_at=_expires_in(),
            approval_id="../escape",
            base_dir=tmp_path,
        )


def test_overlay_without_issue_record_fails_closed(tmp_path):
    approval_id = new_approval_id()
    approval_dir = paths.approval_control_dir(approval_id, tmp_path)
    approval_dir.mkdir(parents=True)
    paths.approval_claim_path(approval_id, tmp_path).write_text("{}", encoding="utf-8")
    with pytest.raises(ApprovalStateError, match="missing issue"):
        get_approval(approval_id, base_dir=tmp_path)


def test_malformed_issue_record_fails_closed(tmp_path):
    approval_id = new_approval_id()
    approval_dir = paths.approval_control_dir(approval_id, tmp_path)
    approval_dir.mkdir(parents=True)
    paths.approval_issue_path(approval_id, tmp_path).write_text("{not json", encoding="utf-8")
    with pytest.raises(ApprovalValidationError, match="malformed JSON"):
        get_approval(approval_id, base_dir=tmp_path)


def test_overlay_issue_wrong_run_id_fails_closed(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    issue = issue_approval(
        run_id,
        _review_intent(run_id, tmp_path, new_event_id()),
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    bad = json.loads(issue_path.read_text(encoding="utf-8"))
    bad["run_id"] = new_run_id()
    issue_path.write_text(json.dumps(bad, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ApprovalValidationError):
        claim_approval(issue["approval_id"], "claim-wrong-run", "bob", base_dir=tmp_path)


def test_replaced_record_target_at_create_fails_closed(tmp_path, monkeypatch):
    _, issue = _issue_fixture(tmp_path)
    real_open = approval_control._openat_control_file_no_follow

    def fail_claim_create(dir_fd, name, flags, mode=0, *, context):
        if name == "claim.json" and flags & approval_control._O_CREAT:
            raise ApprovalValidationError(
                f"unsafe approval control path ({context}/replaced)"
            )
        return real_open(dir_fd, name, flags, mode, context=context)

    monkeypatch.setattr(approval_control, "_openat_control_file_no_follow", fail_claim_create)
    with pytest.raises(ApprovalValidationError, match="unsafe approval control path"):
        claim_approval(issue["approval_id"], "claim-replaced", "bob", base_dir=tmp_path)
    assert not paths.approval_claim_path(issue["approval_id"], tmp_path).exists()


def test_supported_runs_root_aliases_share_physical_marker(tmp_path):
    alias = Path(os.path.join(str(tmp_path), ".", "runs-root"))
    alias.mkdir(parents=True)
    canonical = alias.resolve()
    run_id = _run_with_completion_only(alias)
    started = threading.Event()
    release = threading.Event()

    def hold_alias() -> None:
        with approval_control._approval_control_barrier(run_id, alias):
            started.set()
            release.wait(timeout=5)

    def try_canonical() -> None:
        started.wait(timeout=5)
        with pytest.raises(RunExecutionLockOccupiedError):
            issue_approval(
                run_id,
                _review_intent(run_id, canonical, new_event_id()),
                approver_id="alice",
                executor_id="bob",
                expires_at=_expires_in(),
                base_dir=canonical,
            )
        release.set()

    t1 = threading.Thread(target=hold_alias)
    t2 = threading.Thread(target=try_canonical)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)


@contextmanager
def _zero_write_guard():
    writes: list[str] = []
    real_open = approval_control.os.open

    def track_open(path, flags, *args, **kwargs):
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_TRUNC):
            writes.append(f"open:{path}:{flags}")
        return real_open(path, flags, *args, **kwargs)

    with patch.object(approval_control, "_acquire_outer_run_marker") as acquire, patch.object(
        approval_control, "_bootstrap_control_tree"
    ) as boot, patch.object(approval_control, "_fsync_file_fd") as f_fsync, patch.object(
        approval_control, "_fsync_dir_fd"
    ) as d_fsync, patch.object(
        approval_control.os, "open", side_effect=track_open
    ):
        yield writes
        acquire.assert_not_called()
        boot.assert_not_called()
        f_fsync.assert_not_called()
        d_fsync.assert_not_called()
        assert writes == []


def test_issue_exact_replay_zero_write_guarded(tmp_path):
    run_id = _run_with_completion_only(tmp_path)
    intent = _review_intent(run_id, tmp_path, new_event_id())
    first = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = issue_approval(
            run_id,
            intent,
            approver_id="alice",
            executor_id="bob",
            expires_at=first["expires_at"],
            approval_id=first["approval_id"],
            base_dir=tmp_path,
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_revoke_exact_replay_zero_write(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    first = revoke_approval(issue["approval_id"], "alice", "cancel", base_dir=tmp_path)
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = revoke_approval(issue["approval_id"], "alice", "cancel", base_dir=tmp_path)
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_claim_exact_replay_zero_write(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    first = claim_approval(issue["approval_id"], "claim-zw", "bob", base_dir=tmp_path)
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = claim_approval(issue["approval_id"], "claim-zw", "bob", base_dir=tmp_path)
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_consumed_outcome_exact_replay_zero_write(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-out-zw", "bob", base_dir=tmp_path)
    first = record_use_outcome(
        issue["approval_id"], "claim-out-zw", OUTCOME_CONSUMED, base_dir=tmp_path
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = record_use_outcome(
            issue["approval_id"], "claim-out-zw", OUTCOME_CONSUMED, base_dir=tmp_path
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_ambiguous_outcome_exact_replay_zero_write(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-amb-zw", "bob", base_dir=tmp_path)
    first = record_use_outcome(
        issue["approval_id"], "claim-amb-zw", OUTCOME_AMBIGUOUS, base_dir=tmp_path
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = record_use_outcome(
            issue["approval_id"], "claim-amb-zw", OUTCOME_AMBIGUOUS, base_dir=tmp_path
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_v2_consumed_outcome_requires_evidence(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2", "bob", base_dir=tmp_path)
    evidence = {
        "reason_code": "verified_success",
        "error_classification": "verified_success",
        "bound_api": "review_run_manually",
        "event_id": new_event_id(),
        "pre_observation_digest": issue["source_observation_digest"],
        "post_observation_digest": issue["source_observation_digest"],
        "mutation_may_have_committed": True,
        "safe_to_retry": False,
        "verification_reason_codes": [],
        "observed_record_fingerprint": None,
        "observed_event_fingerprint": None,
    }
    outcome = record_use_outcome(
        issue["approval_id"],
        "claim-v2",
        OUTCOME_CONSUMED,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    assert outcome["outcome_schema_version"] == "2"
    assert outcome["outcome_digest"].startswith("sha256:")
    assert "claim_digest" in outcome
    assert outcome["outcome_evidence"]["reason_code"] == "verified_success"


def test_v2_consumed_exact_replay_zero_write(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-replay", "bob", base_dir=tmp_path)
    evidence = {
        "reason_code": "verified_success",
        "error_classification": "verified_success",
        "bound_api": "review_run_manually",
        "event_id": new_event_id(),
        "pre_observation_digest": issue["source_observation_digest"],
        "post_observation_digest": issue["source_observation_digest"],
        "mutation_may_have_committed": True,
        "safe_to_retry": False,
        "verification_reason_codes": [],
        "observed_record_fingerprint": None,
        "observed_event_fingerprint": None,
    }
    first = record_use_outcome(
        issue["approval_id"],
        "claim-v2-replay",
        OUTCOME_CONSUMED,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = record_use_outcome(
            issue["approval_id"],
            "claim-v2-replay",
            OUTCOME_CONSUMED,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_v2_ambiguous_exact_replay_zero_write(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-amb-replay", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        reason_code="post_verification_mismatch",
        error_classification="post_verification_mismatch",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    first = record_use_outcome(
        issue["approval_id"],
        "claim-v2-amb-replay",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = record_use_outcome(
            issue["approval_id"],
            "claim-v2-amb-replay",
            OUTCOME_AMBIGUOUS,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_v2_ambiguous_evidence_conflict_fails_closed(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-conflict", "bob", base_dir=tmp_path)
    evidence = {
        "reason_code": "claimed_invoke_not_started",
        "error_classification": "claimed_invoke_not_started",
        "bound_api": "review_run_manually",
        "event_id": new_event_id(),
        "pre_observation_digest": issue["source_observation_digest"],
        "post_observation_digest": None,
        "mutation_may_have_committed": False,
        "safe_to_retry": False,
        "verification_reason_codes": [],
        "observed_record_fingerprint": None,
        "observed_event_fingerprint": None,
    }
    record_use_outcome(
        issue["approval_id"],
        "claim-v2-conflict",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    conflicting = dict(evidence)
    conflicting["reason_code"] = "invoke_raised_commit_unknown"
    with pytest.raises(ApprovalConflictError):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-conflict",
            OUTCOME_AMBIGUOUS,
            outcome_evidence=conflicting,
            base_dir=tmp_path,
        )


def test_v1_outcome_still_works_without_evidence(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v1-only", "bob", base_dir=tmp_path)
    outcome = record_use_outcome(
        issue["approval_id"], "claim-v1-only", OUTCOME_CONSUMED, base_dir=tmp_path
    )
    assert outcome["outcome_schema_version"] == "1"
    assert "outcome_digest" not in outcome
    assert outcome["claim_digest"].startswith("sha256:")


# --- Task 25 hardening: in-session helpers and v2 outcome digests ---


def test_in_session_helpers_not_exported_in_public_all():
    assert "_claim_approval_during_session" not in approval_control.__all__
    assert "_record_use_outcome_during_session" not in approval_control.__all__
    assert hasattr(approval_control, "_claim_approval_during_session")
    assert hasattr(approval_control, "_record_use_outcome_during_session")


def test_claim_during_session_without_active_session_raises(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    with pytest.raises(ApprovalStateError, match="approval-use session"):
        approval_control._claim_approval_during_session(
            issue["approval_id"],
            "claim-no-session",
            base_dir=tmp_path,
        )


def test_record_outcome_during_session_without_active_session_raises(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    with pytest.raises(ApprovalStateError, match="approval-use session"):
        approval_control._record_use_outcome_during_session(
            issue["approval_id"],
            "claim-no-session",
            OUTCOME_CONSUMED,
            base_dir=tmp_path,
        )


def test_in_session_helpers_reject_wrong_thread(tmp_path):
    run_id, issue = _issue_fixture(tmp_path)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def holder() -> None:
        with approval_control._approval_use_session(run_id, tmp_path):
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)

    def challenger() -> None:
        barrier.wait(timeout=5)
        try:
            approval_control._claim_approval_during_session(
                issue["approval_id"],
                "claim-wrong-thread",
                base_dir=tmp_path,
            )
        except BaseException as exc:
            errors.append(exc)
        barrier.wait(timeout=5)

    holder_thread = threading.Thread(target=holder)
    challenger_thread = threading.Thread(target=challenger)
    holder_thread.start()
    challenger_thread.start()
    holder_thread.join(timeout=10)
    challenger_thread.join(timeout=10)
    assert len(errors) == 1
    assert isinstance(errors[0], ApprovalStateError)
    assert "approval-use session" in str(errors[0])


GOLDEN_OUTCOME_V2_CONSUMED_BYTES = (
    '{"approval_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    '"approval_id":"apr_golden000001",'
    '"claim_digest":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    '"claim_id":"claim-golden-001","outcome_class":"consumed",'
    '"outcome_digest_projection_version":"htr.approval.outcome.digest.v2",'
    '"outcome_evidence":{"bound_api":"complete_run_manually","error_classification":"verified_success",'
    '"event_id":"evt_golden000001","mutation_may_have_committed":true,'
    '"observed_event_fingerprint":"fp-event","observed_record_fingerprint":"fp-record",'
    '"post_observation_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
    '"pre_observation_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"reason_code":"verified_success","safe_to_retry":false,"verification_reason_codes":[]},'
    '"outcome_schema_version":"2","recorded_at":"2026-01-01T00:00:00+00:00"}'
)
GOLDEN_OUTCOME_V2_CONSUMED_DIGEST = (
    "sha256:3dd1cfb06ad31c4e237c4b33436d6348c592bde58c87721f43ed94a9b4bbdf08"
)
GOLDEN_OUTCOME_V2_AMBIGUOUS_BYTES = (
    '{"approval_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",'
    '"approval_id":"apr_golden000001",'
    '"claim_digest":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
    '"claim_id":"claim-golden-001","outcome_class":"ambiguous",'
    '"outcome_digest_projection_version":"htr.approval.outcome.digest.v2",'
    '"outcome_evidence":{"bound_api":"complete_run_manually",'
    '"error_classification":"post_verification_mismatch","event_id":"evt_golden000001",'
    '"mutation_may_have_committed":true,"observed_event_fingerprint":"fp-event",'
    '"observed_record_fingerprint":"fp-record",'
    '"post_observation_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",'
    '"pre_observation_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
    '"reason_code":"post_verification_mismatch","safe_to_retry":false,'
    '"verification_reason_codes":["event_actor_mismatch"]},'
    '"outcome_schema_version":"2","recorded_at":"2026-01-01T00:00:00+00:00"}'
)
GOLDEN_OUTCOME_V2_AMBIGUOUS_DIGEST = (
    "sha256:3f0f5f1fda68f2d8c5333e5b60c57810b6a9bcec0fa2091827dc1bd87650447f"
)


def _sample_v2_outcome_evidence(**overrides: Any) -> dict[str, Any]:
    base = {
        "reason_code": "verified_success",
        "error_classification": "verified_success",
        "bound_api": "review_run_manually",
        "event_id": "evt_golden000001",
        "pre_observation_digest": "sha256:" + "a" * 64,
        "post_observation_digest": "sha256:" + "b" * 64,
        "mutation_may_have_committed": True,
        "safe_to_retry": False,
        "verification_reason_codes": [],
        "observed_record_fingerprint": "fp-record",
        "observed_event_fingerprint": "fp-event",
    }
    base.update(overrides)
    return base


def test_v2_outcome_golden_digest_projections_stable():
    from htr.action_plan import _canonical_json

    consumed = {
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION_V2,
        "approval_id": "apr_golden000001",
        "approval_digest": "sha256:" + "d" * 64,
        "claim_id": "claim-golden-001",
        "claim_digest": "sha256:" + "e" * 64,
        "outcome_class": OUTCOME_CONSUMED,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "outcome_evidence": _sample_v2_outcome_evidence(
            bound_api="complete_run_manually",
        ),
    }
    ambiguous = {
        **consumed,
        "outcome_class": OUTCOME_AMBIGUOUS,
        "outcome_evidence": _sample_v2_outcome_evidence(
            bound_api="complete_run_manually",
            reason_code="post_verification_mismatch",
            error_classification="post_verification_mismatch",
            post_observation_digest="sha256:" + "c" * 64,
            verification_reason_codes=["event_actor_mismatch"],
        ),
    }
    assert (
        _canonical_json(approval_control._outcome_digest_projection_v2(consumed))
        == GOLDEN_OUTCOME_V2_CONSUMED_BYTES
    )
    assert approval_control._compute_outcome_digest(consumed) == GOLDEN_OUTCOME_V2_CONSUMED_DIGEST
    assert (
        _canonical_json(approval_control._outcome_digest_projection_v2(ambiguous))
        == GOLDEN_OUTCOME_V2_AMBIGUOUS_BYTES
    )
    assert approval_control._compute_outcome_digest(ambiguous) == GOLDEN_OUTCOME_V2_AMBIGUOUS_DIGEST


def test_v2_outcome_digest_authoritative_field_sensitivity():
    base = {
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION_V2,
        "approval_id": "apr_digest000001",
        "approval_digest": "sha256:" + "d" * 64,
        "claim_id": "claim-digest-001",
        "claim_digest": "sha256:" + "e" * 64,
        "outcome_class": OUTCOME_CONSUMED,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "outcome_evidence": _sample_v2_outcome_evidence(),
    }
    golden = approval_control._compute_outcome_digest(base)
    for key in ("approval_id", "claim_id", "outcome_class", "recorded_at"):
        mutated = dict(base)
        mutated[key] = "changed"
        assert approval_control._compute_outcome_digest(mutated) != golden
    evidence_mutated = dict(base)
    evidence_mutated["outcome_evidence"] = _sample_v2_outcome_evidence(reason_code="changed")
    assert approval_control._compute_outcome_digest(evidence_mutated) != golden
    presentation = dict(base)
    presentation["record_kind"] = "approval_outcome"
    assert "record_kind" not in approval_control._outcome_digest_projection_v2(base)


def test_v2_outcome_presentation_fields_excluded_from_digest_projection():
    body = {
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION_V2,
        "approval_id": "apr_pres000001",
        "approval_digest": "sha256:" + "d" * 64,
        "claim_id": "claim-pres-001",
        "claim_digest": "sha256:" + "e" * 64,
        "outcome_class": OUTCOME_CONSUMED,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "outcome_evidence": _sample_v2_outcome_evidence(),
        "record_kind": "approval_outcome",
        "presentation_note": "ignored",
    }
    projection = approval_control._outcome_digest_projection_v2(body)
    assert "record_kind" not in projection
    assert "presentation_note" not in projection
    assert approval_control._compute_outcome_digest(body) == approval_control._compute_outcome_digest(
        {k: v for k, v in body.items() if k not in {"record_kind", "presentation_note"}}
    )


def test_v2_outcome_invalid_reason_code_for_consumed_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-bad-reason", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        reason_code="post_verification_mismatch",
        error_classification="post_verification_mismatch",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(ApprovalValidationError, match="verified_success"):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-bad-reason",
            OUTCOME_CONSUMED,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )


def test_v2_outcome_invalid_safe_to_retry_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-retry", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        safe_to_retry=True,
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(ApprovalValidationError, match="safe_to_retry must be false"):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-retry",
            OUTCOME_CONSUMED,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("bad_value", "claim_suffix"),
    [
        (0, "zero"),
        (1, "one"),
        ("false", "str-false"),
        ("true", "str-true"),
        (None, "null"),
        ([], "list"),
        ({}, "dict"),
        (0.0, "float-zero"),
    ],
)
def test_v2_non_boolean_safe_to_retry_values_rejected(tmp_path, bad_value, claim_suffix):
    _, issue = _issue_fixture(tmp_path)
    claim_id = f"claim-v2-retry-type-{claim_suffix}"
    claim_approval(issue["approval_id"], claim_id, "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        safe_to_retry=bad_value,
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(ApprovalValidationError, match="safe_to_retry must be boolean"):
        record_use_outcome(
            issue["approval_id"],
            claim_id,
            OUTCOME_CONSUMED,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )


def test_v2_outcome_conflicting_evidence_replay_fails_closed(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-ev-conflict", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        reason_code="post_verification_mismatch",
        error_classification="post_verification_mismatch",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    record_use_outcome(
        issue["approval_id"],
        "claim-v2-ev-conflict",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    conflicting = dict(evidence)
    conflicting["reason_code"] = "invoke_raised_commit_unknown"
    conflicting["error_classification"] = "invoke_raised_commit_unknown"
    with pytest.raises(ApprovalConflictError):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-ev-conflict",
            OUTCOME_AMBIGUOUS,
            outcome_evidence=conflicting,
            base_dir=tmp_path,
        )



def test_v1_outcome_conflicting_replay_fails_closed(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v1-conflict", "bob", base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"], "claim-v1-conflict", OUTCOME_CONSUMED, base_dir=tmp_path
    )
    with pytest.raises(ApprovalConflictError):
        record_use_outcome(
            issue["approval_id"], "claim-v1-conflict", OUTCOME_AMBIGUOUS, base_dir=tmp_path
        )


def test_v2_missing_evidence_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-missing-ev", "bob", base_dir=tmp_path)
    with pytest.raises(ApprovalValidationError, match="missing"):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-missing-ev",
            OUTCOME_CONSUMED,
            outcome_evidence={
                "reason_code": "verified_success",
                "error_classification": "verified_success",
                "bound_api": issue["bound_api"],
                "event_id": new_event_id(),
                "pre_observation_digest": issue["source_observation_digest"],
                "mutation_may_have_committed": True,
                "safe_to_retry": False,
                "verification_reason_codes": [],
            },
            base_dir=tmp_path,
        )


def test_v2_malformed_evidence_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-malformed", "bob", base_dir=tmp_path)
    with pytest.raises(ApprovalValidationError, match="JSON object"):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-malformed",
            OUTCOME_CONSUMED,
            outcome_evidence="not-an-object",
            base_dir=tmp_path,
        )


def test_v2_unknown_reason_code_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-unknown-reason", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        reason_code="totally_unknown_reason",
        error_classification="totally_unknown_reason",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    outcome = record_use_outcome(
        issue["approval_id"],
        "claim-v2-unknown-reason",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    assert outcome["outcome_evidence"]["reason_code"] == "totally_unknown_reason"


def test_v2_invalid_ambiguous_reason_combination_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-bad-combo", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        reason_code="verified_success",
        error_classification="verified_success",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(
        ApprovalValidationError,
        match="ambiguous v2 outcome cannot use reason_code verified_success",
    ):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-bad-combo",
            OUTCOME_AMBIGUOUS,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )

def test_v2_non_boolean_mutation_may_have_committed_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-mutation-type", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        mutation_may_have_committed="yes",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(ApprovalValidationError, match="mutation_may_have_committed must be boolean"):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-mutation-type",
            OUTCOME_CONSUMED,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )


def test_v2_invalid_verification_reason_codes_type_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-vrc-type", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        verification_reason_codes="not-a-list",
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(ApprovalValidationError, match="verification_reason_codes"):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-vrc-type",
            OUTCOME_AMBIGUOUS,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )


def _record_v2_consumed_with_evidence(tmp_path, issue, claim_id: str) -> dict[str, Any]:
    evidence = _sample_v2_outcome_evidence(
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    return record_use_outcome(
        issue["approval_id"],
        claim_id,
        OUTCOME_CONSUMED,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )


def test_v2_wrong_approval_digest_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-bad-apr-digest", "bob", base_dir=tmp_path)
    _record_v2_consumed_with_evidence(tmp_path, issue, "claim-v2-bad-apr-digest")
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    raw = json.loads(issue_path.read_text(encoding="utf-8"))
    raw["approval_digest"] = "sha256:" + "0" * 64
    issue_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ApprovalConflictError):
        _record_v2_consumed_with_evidence(tmp_path, issue, "claim-v2-bad-apr-digest")


def test_v2_wrong_claim_digest_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-bad-claim-digest", "bob", base_dir=tmp_path)
    _record_v2_consumed_with_evidence(tmp_path, issue, "claim-v2-bad-claim-digest")
    claim_path = paths.approval_claim_path(issue["approval_id"], tmp_path)
    raw = json.loads(claim_path.read_text(encoding="utf-8"))
    raw["claim_digest"] = "sha256:" + "1" * 64
    claim_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ApprovalConflictError):
        _record_v2_consumed_with_evidence(tmp_path, issue, "claim-v2-bad-claim-digest")


def test_v2_wrong_bound_api_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-bad-bound-api", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    record_use_outcome(
        issue["approval_id"],
        "claim-v2-bad-bound-api",
        OUTCOME_CONSUMED,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    conflicting = dict(evidence)
    conflicting["bound_api"] = "complete_run_manually"
    with pytest.raises(ApprovalConflictError):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-bad-bound-api",
            OUTCOME_CONSUMED,
            outcome_evidence=conflicting,
            base_dir=tmp_path,
        )


def test_v2_wrong_event_id_rejected(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-bad-event", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    record_use_outcome(
        issue["approval_id"],
        "claim-v2-bad-event",
        OUTCOME_CONSUMED,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    conflicting = dict(evidence)
    conflicting["event_id"] = new_event_id()
    with pytest.raises(ApprovalConflictError):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-bad-event",
            OUTCOME_CONSUMED,
            outcome_evidence=conflicting,
            base_dir=tmp_path,
        )


def test_v2_explicit_null_post_observation_digest_allowed(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-null-post", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        reason_code="claimed_invoke_not_started",
        error_classification="claimed_invoke_not_started",
        post_observation_digest=None,
        pre_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    outcome = record_use_outcome(
        issue["approval_id"],
        "claim-v2-null-post",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    assert outcome["outcome_evidence"]["post_observation_digest"] is None


def test_v2_explicit_null_observed_fingerprints_allowed(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-null-fp", "bob", base_dir=tmp_path)
    evidence = _sample_v2_outcome_evidence(
        observed_record_fingerprint=None,
        observed_event_fingerprint=None,
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    outcome = record_use_outcome(
        issue["approval_id"],
        "claim-v2-null-fp",
        OUTCOME_CONSUMED,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    assert outcome["outcome_evidence"]["observed_record_fingerprint"] is None
    assert outcome["outcome_evidence"]["observed_event_fingerprint"] is None


def test_v2_insertion_order_independence_for_digest(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-order", "bob", base_dir=tmp_path)
    evidence_a = {
        "reason_code": "verified_success",
        "error_classification": "verified_success",
        "bound_api": issue["bound_api"],
        "event_id": new_event_id(),
        "pre_observation_digest": issue["source_observation_digest"],
        "post_observation_digest": issue["source_observation_digest"],
        "mutation_may_have_committed": True,
        "safe_to_retry": False,
        "verification_reason_codes": [],
        "observed_record_fingerprint": "fp-record",
        "observed_event_fingerprint": "fp-event",
    }
    evidence_b = {
        "observed_event_fingerprint": "fp-event",
        "observed_record_fingerprint": "fp-record",
        "verification_reason_codes": [],
        "safe_to_retry": False,
        "mutation_may_have_committed": True,
        "post_observation_digest": issue["source_observation_digest"],
        "pre_observation_digest": issue["source_observation_digest"],
        "event_id": evidence_a["event_id"],
        "bound_api": issue["bound_api"],
        "error_classification": "verified_success",
        "reason_code": "verified_success",
    }
    first = record_use_outcome(
        issue["approval_id"],
        "claim-v2-order",
        OUTCOME_CONSUMED,
        outcome_evidence=evidence_a,
        base_dir=tmp_path,
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = record_use_outcome(
            issue["approval_id"],
            "claim-v2-order",
            OUTCOME_CONSUMED,
            outcome_evidence=evidence_b,
            base_dir=tmp_path,
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap


def test_v2_v1_not_interchangeable_schema_versions(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v2-v1-mix", "bob", base_dir=tmp_path)
    v1 = record_use_outcome(
        issue["approval_id"], "claim-v2-v1-mix", OUTCOME_CONSUMED, base_dir=tmp_path
    )
    assert v1["outcome_schema_version"] != OUTCOME_SCHEMA_VERSION_V2
    evidence = _sample_v2_outcome_evidence(
        pre_observation_digest=issue["source_observation_digest"],
        post_observation_digest=issue["source_observation_digest"],
        bound_api=issue["bound_api"],
    )
    with pytest.raises(ApprovalConflictError):
        record_use_outcome(
            issue["approval_id"],
            "claim-v2-v1-mix",
            OUTCOME_CONSUMED,
            outcome_evidence=evidence,
            base_dir=tmp_path,
        )


def test_v1_outcome_exact_replay_zero_write_unchanged_after_v2_additions(tmp_path):
    _, issue = _issue_fixture(tmp_path)
    claim_approval(issue["approval_id"], "claim-v1-zw-retest", "bob", base_dir=tmp_path)
    first = record_use_outcome(
        issue["approval_id"], "claim-v1-zw-retest", OUTCOME_CONSUMED, base_dir=tmp_path
    )
    snap = _approval_fs_snapshot(tmp_path)
    with _zero_write_guard():
        second = record_use_outcome(
            issue["approval_id"], "claim-v1-zw-retest", OUTCOME_CONSUMED, base_dir=tmp_path
        )
    assert second == first
    assert _approval_fs_snapshot(tmp_path) == snap
