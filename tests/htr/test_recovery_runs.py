"""Tests for Task 27 — approved recovery run creation (Path R1)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import htr.recovery_runs as recovery_runs_mod

from htr import contracts, io, paths
from htr.action_plan import PlanningIntent
from htr.approval_control import issue_approval
from htr.contracts import run_completion_record_json_path
from htr.execution_lock import LOCKS_DIR_NAME
from htr.finalization import SealEvaluation, SealState
from htr.ids import generate_recovery_request_id, new_run_id, new_task_id
from htr.invoke_run_completion import invoke_approved_run_completion
from htr.recovery_runs import (
    RecoveryRunConflictError,
    RecoveryRunOutcomeClass,
    RecoveryRunValidationError,
    RecoveryScope,
    claim_recovery_run_approval,
    create_recovery_run_request,
    execute_approved_successor_run_creation,
    generate_recovery_approval_id,
    generate_recovery_attempt_id,
    generate_recovery_claim_id,
    generate_successor_run_id,
    issue_recovery_run_approval,
    load_recovery_run_bundle,
    reconcile_recovery_run_creation,
    revoke_recovery_run_approval,
)
from htr.reconciliation_cases import (
    ReconciliationDecisionClass,
    ReconciliationNextProtocol,
    ReconciliationScopeReason,
    _compute_decision_digest,
    _decision_revalidation_digest_projection,
    generate_reconciliation_case_id,
    load_reconciliation_case,
    open_reconciliation_case,
    record_reconciliation_decision,
    record_reconciliation_observation,
)
from htr.action_plan import _sha256_digest
from htr.state import RecoveryRunDurabilityError, RecoveryRunStateError

TASK16_PATH = Path(__file__).with_name("test_run_final_closure.py")


def _load_task16():
    spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16()


def _seal_finalized(run_id: str) -> SealEvaluation:
    return SealEvaluation(SealState.FINALIZED_VALID, (), run_id)


def _expires_in_minutes(minutes: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _expires_in_hours(hours: float = 1.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _control_snapshot(root: Path) -> dict[str, str]:
    control = root / ".control"
    if not control.exists():
        return {}
    return {
        str(p.relative_to(control)): _file_digest(p)
        for p in sorted(control.rglob("*"))
        if p.is_file()
    }


def _run_ready_for_completion(tmp_path: Path) -> tuple[str, str, dict]:
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(run_id=run_id, completed_task_ids=[task_id])
    return run_id, task_id, completion


def _issue_completion_approval(tmp_path: Path, run_id: str, completion: dict) -> dict:
    from htr.ids import new_event_id
    from htr.invoke_run_completion import PILOT_BOUND_API

    event_id = new_event_id()
    intent = PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs={"record": completion, "actor": "human", "event_id": event_id},
        htr_runs_root=str(tmp_path),
    )
    return issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in_hours(),
        base_dir=tmp_path,
    )


def _open_case(tmp_path: Path, issue: dict) -> str:
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    return case_id


def _consumed_contradiction_case(tmp_path: Path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-r1", base_dir=tmp_path)
    wrong = dict(completion)
    wrong["metadata"] = {"tampered": True}
    io.atomic_write_json(run_completion_record_json_path(run_id, tmp_path), wrong)
    case_id = _open_case(tmp_path, issue)
    obs, _ = record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
    return case_id, obs, run_id, issue


def _path_r1_ready(tmp_path: Path) -> tuple[str, str, str, str]:
    case_id, obs, source_run_id, _issue = _consumed_contradiction_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.recovery_run_review,
    )
    recovery_request_id = generate_recovery_request_id()
    successor_run_id = generate_successor_run_id()
    return case_id, source_run_id, recovery_request_id, successor_run_id


def _path_r1_request(tmp_path: Path, *, scope: RecoveryScope = RecoveryScope.diagnostic_only):
    case_id, source_run_id, recovery_request_id, successor_run_id = _path_r1_ready(tmp_path)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        create_recovery_run_request(
            recovery_request_id,
            case_id,
            recovery_scope=scope,
            recovery_reason="lifecycle evidence conflict review",
            successor_run_id=successor_run_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
    return recovery_request_id, successor_run_id, case_id, source_run_id


def _full_chain(tmp_path: Path) -> tuple[str, str, str, str]:
    recovery_request_id, successor_run_id, case_id, source_run_id = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_recovery_run_approval(
        recovery_request_id,
        generate_recovery_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    return recovery_request_id, successor_run_id, case_id, source_run_id


def test_generate_ids_are_zero_write(tmp_path):
    before = _control_snapshot(tmp_path)
    ids = (
        generate_recovery_request_id(),
        generate_recovery_approval_id(),
        generate_recovery_claim_id(),
        generate_recovery_attempt_id(),
        generate_successor_run_id(),
    )
    after = _control_snapshot(tmp_path)
    assert before == after
    assert all(isinstance(x, str) and x for x in ids)


def test_path_r1_happy_path_creates_successor(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.successor_created_verified.value
    assert paths.run_root(successor_run_id, tmp_path).is_dir()
    assert paths.recovery_origin_path(successor_run_id, tmp_path).is_file()
    manifest = io.read_json(paths.run_manifest_path(successor_run_id, tmp_path))
    assert manifest["status"] == "created"
    marker = tmp_path / LOCKS_DIR_NAME / f"{successor_run_id}.marker"
    assert not marker.exists()
    outcome = io.read_json(paths.recovery_run_outcome_path(recovery_request_id, tmp_path))
    for field in (
        "source_run_mutation_allowed",
        "retry_allowed",
        "repair_allowed",
        "invoke_allowed",
        "automatic_execution_allowed",
        "outcome_rewrite_allowed",
    ):
        assert outcome[field] is False


def test_request_rejects_marker_disposition_path(tmp_path):
    from htr.ids import new_event_id
    from htr.invoke_run_completion import PILOT_BOUND_API

    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-marker", base_dir=tmp_path)
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    (locks_root / f"{run_id}.marker").write_text(
        json.dumps({"schema_version": "1", "acquisition_id": "acq", "run_id": run_id}),
        encoding="utf-8",
    )
    case_id = _open_case(tmp_path, issue)
    obs, _ = record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.marker_disposition_review,
    )
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        with pytest.raises(RecoveryRunValidationError, match="not eligible"):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


def test_request_rejects_forward_fix_scope(tmp_path):
    case_id, obs, source_run_id, _ = _consumed_contradiction_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.recovery_run_review,
    )
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        with pytest.raises(RecoveryRunValidationError, match="forward_fix"):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.forward_fix,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


def test_request_rejects_non_finalized_seal(tmp_path):
    case_id, obs, source_run_id, _ = _consumed_contradiction_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.recovery_run_review,
    )
    not_final = SealEvaluation(SealState.NOT_FINALIZED, ("missing_closure",), source_run_id)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=not_final):
        with pytest.raises(RecoveryRunValidationError, match="finalized_valid"):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


def test_execute_rejects_expired_approval(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _path_r1_request(tmp_path)
    issued_at = datetime.now(timezone.utc) - timedelta(minutes=20)
    expires_at = (issued_at + timedelta(minutes=10)).isoformat()
    claim_at = issued_at + timedelta(minutes=5)
    with patch("htr.recovery_runs._utc_now", return_value=issued_at), patch(
        "htr.recovery_runs._utc_now_iso", return_value=issued_at.isoformat()
    ):
        issue_recovery_run_approval(
            recovery_request_id,
            generate_recovery_approval_id(),
            issued_by="approver",
            expires_at=expires_at,
            base_dir=tmp_path,
        )
    with patch("htr.recovery_runs._utc_now", return_value=claim_at):
        claim_recovery_run_approval(
            recovery_request_id,
            generate_recovery_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.approval_invalid.value


def test_revoke_before_claim_blocks_claim(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    revoke_recovery_run_approval(
        recovery_request_id,
        revoked_by="approver",
        reason="changed mind",
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunValidationError, match="revoked"):
        claim_recovery_run_approval(
            recovery_request_id,
            generate_recovery_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )


def test_execute_exact_replay_skips_reexecution(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_id = generate_recovery_attempt_id()
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        first = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
        second = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert first.outcome_class == RecoveryRunOutcomeClass.successor_created_verified.value
    assert second.exact_replay is True
    assert second.outcome_digest == first.outcome_digest


def test_successor_id_conflict_when_preexisting_run(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    io.create_run_workspace(successor_run_id, base_dir=tmp_path)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.successor_id_conflict.value


def test_reconcile_is_zero_write(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _full_chain(tmp_path)
    before = _control_snapshot(tmp_path)
    reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    after = _control_snapshot(tmp_path)
    assert before == after


def test_load_bundle_round_trip(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    bundle = load_recovery_run_bundle(recovery_request_id, base_dir=tmp_path)
    assert bundle.recovery_request_id == recovery_request_id
    assert bundle.request_record is not None
    assert bundle.request_record.recovery_scope == RecoveryScope.diagnostic_only.value


def test_claim_conflicting_replay_fails_closed(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_recovery_run_approval(
        recovery_request_id,
        generate_recovery_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunConflictError):
        claim_recovery_run_approval(
            recovery_request_id,
            generate_recovery_claim_id(),
            claimant="other-executor",
            base_dir=tmp_path,
        )


def test_thread_simultaneous_claim_one_creator_one_replay(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_id = generate_recovery_claim_id()
    barrier = threading.Barrier(3)
    slots: list[tuple[str, ...] | Exception] = [()] * 3

    def _worker(index: int) -> None:
        try:
            barrier.wait()
            _rec, meta = claim_recovery_run_approval(
                recovery_request_id,
                claim_id,
                claimant="executor",
                base_dir=tmp_path,
            )
            slots[index] = ("ok", meta.exact_replay)
        except Exception as exc:
            slots[index] = exc

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    ok = [s for s in slots if isinstance(s, tuple) and s[0] == "ok"]
    assert len(ok) == 3
    assert sum(1 for r in ok if r[1] is False) == 1
    assert sum(1 for r in ok if r[1] is True) == 2


def test_attempt_record_exists_before_successor_reservation(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_seen: list[bool] = []
    real_reserve = io.reserve_run_root_exclusive

    def _assert_attempt_before_reserve(*args, **kwargs):
        attempt_seen.append(
            paths.recovery_run_attempt_path(recovery_request_id, tmp_path).is_file()
        )
        return real_reserve(*args, **kwargs)

    with patch(
        "htr.recovery_runs.reserve_run_root_exclusive",
        side_effect=_assert_attempt_before_reserve,
    ), patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert attempt_seen == [True]
    assert paths.run_root(successor_run_id, tmp_path).is_dir()


def test_source_control_records_unchanged_after_creation(tmp_path):
    recovery_request_id, successor_run_id, case_id, source_run_id = _full_chain(tmp_path)

    def _source_snapshot() -> dict[str, str]:
        snap: dict[str, str] = {}
        for rel in (
            paths.reconciliation_open_path(case_id, tmp_path),
            paths.reconciliation_observation_path(case_id, tmp_path),
            paths.reconciliation_decision_path(case_id, tmp_path),
        ):
            if rel.is_file():
                snap[str(rel.relative_to(tmp_path))] = _file_digest(rel)
        return snap

    before = _source_snapshot()
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=_seal_finalized(source_run_id)):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    after = _source_snapshot()
    assert before == after
    assert paths.recovery_origin_path(successor_run_id, tmp_path).is_file()


def test_revoke_after_claim_conflicts(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _full_chain(tmp_path)
    with pytest.raises(RecoveryRunConflictError, match="revoke after claim"):
        revoke_recovery_run_approval(
            recovery_request_id,
            revoked_by="approver",
            reason="too late",
            base_dir=tmp_path,
        )


# --- Task 27 comprehensive hardening ---

_PERMITTED_SCOPES = (
    RecoveryScope.diagnostic_only,
    RecoveryScope.verification_only,
    RecoveryScope.artifact_reconstruction_review,
    RecoveryScope.controlled_follow_up,
)

_NON_PERMISSION_FIELDS = (
    "source_run_mutation_allowed",
    "retry_allowed",
    "repair_allowed",
    "invoke_allowed",
    "automatic_execution_allowed",
    "outcome_rewrite_allowed",
)

_INVALID_BOOL_VALUES = (
    pytest.param(True, id="true"),
    pytest.param(0, id="zero"),
    pytest.param("false", id="string-false"),
    pytest.param(None, id="null"),
    pytest.param(1, id="one"),
    pytest.param([], id="empty-list"),
)

_DURABILITY_STAGES = (
    "record_fsync",
    "recovery_case_dir_fsync",
    "control_dir_fsync",
    "parent_dir_fsync",
    "record_write",
)

_RECONCILE_CLASSIFICATIONS = (
    "missing_request",
    "incomplete",
    "claim_without_attempt",
    "valid_durable_outcome",
    "malformed_outcome",
    "attempt_with_verified_successor",
    "attempt_without_successor",
)

_HARDENING_THREAD_COUNT = 8


def _require_posix_fork() -> None:
    if not hasattr(os, "fork"):
        pytest.fail("POSIX fork required for subprocess race tests")


def _path_r1_case_decided(tmp_path: Path) -> tuple[str, str]:
    case_id, obs, source_run_id, _issue = _consumed_contradiction_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.recovery_run_review,
    )
    return case_id, source_run_id


def _tamper_decision_raw(
    case_id: str,
    base_dir: Path,
    mutator: Any,
) -> None:
    decision_path = paths.reconciliation_decision_path(case_id, base_dir)
    raw = json.loads(decision_path.read_text(encoding="utf-8"))
    mutator(raw)
    raw["decision_digest"] = _compute_decision_digest(raw)
    decision_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _tamper_decision_projection(
    case_id: str,
    base_dir: Path,
    **projection_overrides: Any,
) -> None:
    def _mutator(raw: dict[str, Any]) -> None:
        revalidation = dict(raw.get("decision_time_revalidation") or {})
        projection = dict(revalidation.get("inspection_semantic_projection") or {})
        projection.update(projection_overrides)
        revalidation["inspection_semantic_projection"] = projection
        revalidation["decision_revalidation_record_digest"] = _sha256_digest(
            _decision_revalidation_digest_projection(revalidation)
        )
        raw["decision_time_revalidation"] = revalidation
        drift = dict(raw.get("observation_decision_drift") or {})
        drift["inspection_semantic_digest_at_decision"] = projection.get(
            "current_observation_semantic_digest",
            drift.get("inspection_semantic_digest_at_decision"),
        )
        raw["observation_decision_drift"] = drift

    _tamper_decision_raw(case_id, base_dir, _mutator)


def _tamper_digest(path: Path, digest_field: str, *, mode: str) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if mode == "wrong_digest":
        raw[digest_field] = "sha256:" + "c" * 64
    elif mode == "missing_digest":
        raw.pop(digest_field, None)
    else:
        raise ValueError(mode)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _inject_recovery_durability_failure(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    if stage == "record_write":
        real_write = recovery_runs_mod._write_all

        def _fail_write(*args: Any, **kwargs: Any) -> None:
            raise RecoveryRunDurabilityError(
                "injected record_write",
                recovery_request_id=kwargs.get("recovery_request_id", "unknown"),
                record_name=kwargs.get("record_name", "unknown"),
                durability_stage="record_write",
                record_may_have_committed=False,
                successor_may_have_been_created=False,
                exact_replay_status="no",
            )

        monkeypatch.setattr(recovery_runs_mod, "_write_all", _fail_write)
        return

    if stage == "record_fsync":
        def _fail_file(
            _fd: int,
            *,
            recovery_request_id: str,
            record_name: str,
        ) -> None:
            raise RecoveryRunDurabilityError(
                "injected record_fsync",
                recovery_request_id=recovery_request_id,
                record_name=record_name,
                durability_stage="record_fsync",
                record_may_have_committed=True,
                successor_may_have_been_created=False,
                exact_replay_status="indeterminate",
            )

        monkeypatch.setattr(recovery_runs_mod, "_fsync_file_fd", _fail_file)
        return

    def _make_dir_fail(expected_stage: str):
        real = recovery_runs_mod._fsync_dir_fd

        def _fail_dir(
            dir_fd: int,
            *,
            recovery_request_id: str,
            record_name: str,
            stage: str,
        ) -> None:
            if stage == expected_stage:
                raise RecoveryRunDurabilityError(
                    f"injected {expected_stage}",
                    recovery_request_id=recovery_request_id,
                    record_name=record_name,
                    durability_stage=expected_stage,
                    record_may_have_committed=True,
                    successor_may_have_been_created=False,
                    exact_replay_status="indeterminate",
                )
            real(
                dir_fd,
                recovery_request_id=recovery_request_id,
                record_name=record_name,
                stage=stage,
            )

        return _fail_dir

    monkeypatch.setattr(recovery_runs_mod, "_fsync_dir_fd", _make_dir_fail(stage))


def _setup_reconcile_classification(
    tmp_path: Path, classification: str
) -> tuple[str, str]:
    recovery_request_id = generate_recovery_request_id()
    if classification == "missing_request":
        return recovery_request_id, classification
    recovery_request_id, successor_run_id, _case_id, source_run_id = _path_r1_request(tmp_path)
    if classification == "incomplete":
        return recovery_request_id, classification
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    if classification == "claim_without_attempt":
        claim_recovery_run_approval(
            recovery_request_id,
            generate_recovery_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )
        return recovery_request_id, classification
    claim_recovery_run_approval(
        recovery_request_id,
        generate_recovery_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    attempt_id = generate_recovery_attempt_id()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        if classification == "valid_durable_outcome":
            execute_approved_successor_run_creation(
                recovery_request_id,
                attempt_id,
                executor="executor",
                base_dir=tmp_path,
            )
            return recovery_request_id, classification
        if classification == "malformed_outcome":
            execute_approved_successor_run_creation(
                recovery_request_id,
                attempt_id,
                executor="executor",
                base_dir=tmp_path,
            )
            _tamper_digest(
                paths.recovery_run_outcome_path(recovery_request_id, tmp_path),
                "outcome_digest",
                mode="wrong_digest",
            )
            return recovery_request_id, classification
        execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
        paths.recovery_run_outcome_path(recovery_request_id, tmp_path).unlink()
        if classification == "attempt_with_verified_successor":
            return recovery_request_id, classification
        if classification == "attempt_without_successor":
            import shutil

            shutil.rmtree(paths.run_root(successor_run_id, tmp_path))
            return recovery_request_id, classification
    raise ValueError(classification)


def _subprocess_recovery_claim_worker(
    runs_root: str,
    recovery_request_id: str,
    claim_id: str,
    slot: Any,
    *,
    barrier: Any | None = None,
    gate: Any | None = None,
) -> None:
    try:
        if gate is not None:
            gate.wait(timeout=30)
        if barrier is not None:
            barrier.wait(timeout=30)
        _rec, meta = claim_recovery_run_approval(
            recovery_request_id,
            claim_id,
            claimant="executor",
            base_dir=Path(runs_root),
        )
        slot.put(("claimed", meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _subprocess_recovery_revoke_worker(
    runs_root: str,
    recovery_request_id: str,
    slot: Any,
    *,
    barrier: Any | None = None,
    gate: Any | None = None,
    done: Any | None = None,
) -> None:
    try:
        if gate is not None:
            gate.wait(timeout=30)
        if barrier is not None:
            barrier.wait(timeout=30)
        _rec, meta = revoke_recovery_run_approval(
            recovery_request_id,
            revoked_by="approver",
            reason="race",
            base_dir=Path(runs_root),
        )
        if done is not None:
            done.set()
        slot.put(("revoked", meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _subprocess_crash_holding_recovery_case_flock_worker(
    runs_root: str,
    recovery_request_id: str,
    slot: Any,
    release_gate: Any,
) -> None:
    import fcntl

    from htr.recovery_runs import _open_control_dir_no_follow

    case_dir = paths.recovery_run_control_dir(recovery_request_id, Path(runs_root))
    case_fd = _open_control_dir_no_follow(case_dir, context=f"recovery_runs/{recovery_request_id}")
    try:
        fcntl.flock(case_fd, fcntl.LOCK_EX)
        slot.put("flock_held")
        release_gate.wait(timeout=10)
        os._exit(1)
    finally:
        os.close(case_fd)


def _subprocess_claim_then_signal_worker(
    runs_root: str,
    recovery_request_id: str,
    claim_id: str,
    slot: Any,
    claim_done: Any,
) -> None:
    try:
        _rec, meta = claim_recovery_run_approval(
            recovery_request_id,
            claim_id,
            claimant="executor",
            base_dir=Path(runs_root),
        )
        claim_done.set()
        slot.put(("claimed", meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _issued_recovery_request(tmp_path: Path) -> str:
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    return recovery_request_id


def _recovery_case_entries(recovery_request_id: str, tmp_path: Path) -> frozenset[str]:
    case_dir = paths.recovery_run_control_dir(recovery_request_id, tmp_path)
    return frozenset(p.name for p in case_dir.iterdir() if p.is_file())


def test_subprocess_revoke_wins_claim_rejected(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    claim_id = generate_recovery_claim_id()
    ctx = multiprocessing.get_context("spawn")
    revoke_done = ctx.Event()
    claim_q: Any = ctx.Queue()
    revoke_q: Any = ctx.Queue()
    revoke_proc = ctx.Process(
        target=_subprocess_recovery_revoke_worker,
        args=(str(tmp_path), recovery_request_id, revoke_q),
        kwargs={"done": revoke_done},
    )
    claim_proc = ctx.Process(
        target=_subprocess_recovery_claim_worker,
        args=(str(tmp_path), recovery_request_id, claim_id, claim_q),
        kwargs={"gate": revoke_done},
    )
    revoke_proc.start()
    claim_proc.start()
    revoke_proc.join(timeout=30)
    claim_proc.join(timeout=30)
    assert revoke_proc.exitcode == 0
    assert claim_proc.exitcode == 0
    assert revoke_q.get(timeout=5) == ("revoked", False)
    claim_result = claim_q.get(timeout=5)
    assert claim_result[0] == "err"
    assert claim_result[1] == "RecoveryRunValidationError"
    assert paths.recovery_run_revoke_path(recovery_request_id, tmp_path).is_file()
    assert not paths.recovery_run_claim_path(recovery_request_id, tmp_path).exists()
    assert _recovery_case_entries(recovery_request_id, tmp_path) == frozenset(
        {"request.json", "issue.json", "revoke.json"}
    )


def test_subprocess_claim_wins_revoke_rejected(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    claim_id = generate_recovery_claim_id()
    ctx = multiprocessing.get_context("spawn")
    claim_done = ctx.Event()
    claim_q: Any = ctx.Queue()
    revoke_q: Any = ctx.Queue()
    claim_proc = ctx.Process(
        target=_subprocess_claim_then_signal_worker,
        args=(str(tmp_path), recovery_request_id, claim_id, claim_q, claim_done),
    )
    revoke_proc = ctx.Process(
        target=_subprocess_recovery_revoke_worker,
        args=(str(tmp_path), recovery_request_id, revoke_q),
        kwargs={"gate": claim_done},
    )
    claim_proc.start()
    revoke_proc.start()
    claim_proc.join(timeout=30)
    revoke_proc.join(timeout=30)
    assert claim_proc.exitcode == 0
    assert revoke_proc.exitcode == 0
    assert claim_q.get(timeout=5)[0] == "claimed"
    revoke_result = revoke_q.get(timeout=5)
    assert revoke_result[0] == "err"
    assert revoke_result[1] == "RecoveryRunConflictError"
    assert paths.recovery_run_claim_path(recovery_request_id, tmp_path).is_file()
    assert not paths.recovery_run_revoke_path(recovery_request_id, tmp_path).exists()
    assert _recovery_case_entries(recovery_request_id, tmp_path) == frozenset(
        {"request.json", "issue.json", "claim.json"}
    )


def test_subprocess_simultaneous_revoke_claim_exactly_one_wins(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    claim_id = generate_recovery_claim_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    claim_q: Any = ctx.Queue()
    revoke_q: Any = ctx.Queue()
    claim_proc = ctx.Process(
        target=_subprocess_recovery_claim_worker,
        args=(str(tmp_path), recovery_request_id, claim_id, claim_q),
        kwargs={"barrier": barrier},
    )
    revoke_proc = ctx.Process(
        target=_subprocess_recovery_revoke_worker,
        args=(str(tmp_path), recovery_request_id, revoke_q),
        kwargs={"barrier": barrier},
    )
    claim_proc.start()
    revoke_proc.start()
    claim_proc.join(timeout=30)
    revoke_proc.join(timeout=30)
    assert claim_proc.exitcode == 0
    assert revoke_proc.exitcode == 0
    claim_result = claim_q.get(timeout=5)
    revoke_result = revoke_q.get(timeout=5)
    winners = [r for r in (claim_result, revoke_result) if r[0] in ("claimed", "revoked")]
    losers = [r for r in (claim_result, revoke_result) if r[0] == "err"]
    assert len(winners) == 1
    assert len(losers) == 1
    if winners[0][0] == "revoked":
        assert losers[0][1] == "RecoveryRunValidationError"
        assert _recovery_case_entries(recovery_request_id, tmp_path) == frozenset(
            {"request.json", "issue.json", "revoke.json"}
        )
    else:
        assert losers[0][1] == "RecoveryRunConflictError"
        assert _recovery_case_entries(recovery_request_id, tmp_path) == frozenset(
            {"request.json", "issue.json", "claim.json"}
        )


def test_subprocess_simultaneous_identical_revoke_replay(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_recovery_revoke_worker,
            args=(str(tmp_path), recovery_request_id, slots[i]),
            kwargs={"barrier": barrier},
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=5) for i in range(2)]
    assert all(r[0] == "revoked" for r in results)
    assert sum(1 for r in results if r[1] is False) == 1
    assert sum(1 for r in results if r[1] is True) == 1


def test_subprocess_simultaneous_identical_claim_replay(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    claim_id = generate_recovery_claim_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_recovery_claim_worker,
            args=(str(tmp_path), recovery_request_id, claim_id, slots[i]),
            kwargs={"barrier": barrier},
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=5) for i in range(2)]
    assert all(r[0] == "claimed" for r in results)
    assert sum(1 for r in results if r[1] is False) == 1
    assert sum(1 for r in results if r[1] is True) == 1


def test_subprocess_conflicting_claim_fails_closed(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    claim_ids = [generate_recovery_claim_id(), generate_recovery_claim_id()]
    procs = [
        ctx.Process(
            target=_subprocess_recovery_claim_worker,
            args=(str(tmp_path), recovery_request_id, claim_ids[i], slots[i]),
            kwargs={"barrier": barrier},
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=5) for i in range(2)]
    ok = [r for r in results if r[0] == "claimed"]
    err = [r for r in results if r[0] == "err"]
    assert len(ok) == 1
    assert len(err) == 1
    assert err[0][1] == "RecoveryRunConflictError"


def test_revoke_conflicting_actor_replay_fails_closed(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    revoke_recovery_run_approval(
        recovery_request_id,
        revoked_by="approver",
        reason="first",
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunConflictError):
        revoke_recovery_run_approval(
            recovery_request_id,
            revoked_by="other-approver",
            reason="first",
            base_dir=tmp_path,
        )


def test_claim_conflicting_actor_replay_fails_closed(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    claim_id = generate_recovery_claim_id()
    claim_recovery_run_approval(
        recovery_request_id,
        claim_id,
        claimant="executor",
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunConflictError):
        claim_recovery_run_approval(
            recovery_request_id,
            claim_id,
            claimant="other-executor",
            base_dir=tmp_path,
        )


@pytest.mark.parametrize("filename", ("revoke.json", "claim.json"))
def test_partial_revoke_or_claim_record_fails_closed(tmp_path, filename):
    recovery_request_id = _issued_recovery_request(tmp_path)
    target = paths.recovery_run_control_dir(recovery_request_id, tmp_path) / filename
    target.write_text('{"partial": true', encoding="utf-8")
    if filename == "revoke.json":
        with pytest.raises(RecoveryRunValidationError):
            revoke_recovery_run_approval(
                recovery_request_id,
                revoked_by="approver",
                reason="race",
                base_dir=tmp_path,
            )
    else:
        with pytest.raises(RecoveryRunValidationError):
            claim_recovery_run_approval(
                recovery_request_id,
                generate_recovery_claim_id(),
                claimant="executor",
                base_dir=tmp_path,
            )


def test_subprocess_crash_holding_case_flock_releases_lock(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    release_gate = ctx.Event()
    slot: Any = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_crash_holding_recovery_case_flock_worker,
        args=(str(tmp_path), recovery_request_id, slot, release_gate),
    )
    proc.start()
    assert slot.get(timeout=10) == "flock_held"
    release_gate.set()
    proc.join(timeout=10)
    assert proc.exitcode == 1
    revoke_recovery_run_approval(
        recovery_request_id,
        revoked_by="approver",
        reason="after-crash",
        base_dir=tmp_path,
    )
    assert paths.recovery_run_revoke_path(recovery_request_id, tmp_path).is_file()


def test_thread_revoke_claim_coordination_no_deadlock(tmp_path):
    recovery_request_id = _issued_recovery_request(tmp_path)
    claim_id = generate_recovery_claim_id()
    barrier = threading.Barrier(2)
    slots: list[tuple[str, ...]] = []

    def _revoke() -> None:
        barrier.wait(timeout=30)
        try:
            revoke_recovery_run_approval(
                recovery_request_id,
                revoked_by="approver",
                reason="thread",
                base_dir=tmp_path,
            )
            slots.append(("revoked",))
        except Exception as exc:
            slots.append(("err", type(exc).__name__))

    def _claim() -> None:
        barrier.wait(timeout=30)
        try:
            _rec, meta = claim_recovery_run_approval(
                recovery_request_id,
                claim_id,
                claimant="executor",
                base_dir=tmp_path,
            )
            slots.append(("claimed", str(meta.exact_replay)))
        except Exception as exc:
            slots.append(("err", type(exc).__name__))

    threads = [threading.Thread(target=_revoke), threading.Thread(target=_claim)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert len(slots) == 2
    assert sum(1 for s in slots if s[0] in ("revoked", "claimed")) == 1


def test_hardening_path_r1_integration_all_eligibility_fields(tmp_path):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)
    recovery_request_id = generate_recovery_request_id()
    successor_run_id = generate_successor_run_id()
    scope = RecoveryScope.controlled_follow_up
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        record, meta = create_recovery_run_request(
            recovery_request_id,
            case_id,
            recovery_scope=scope,
            recovery_reason="lifecycle evidence conflict review",
            successor_run_id=successor_run_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
    bundle = load_reconciliation_case(case_id, base_dir=tmp_path)
    raw = io.read_json(paths.recovery_run_request_path(recovery_request_id, tmp_path))
    assert meta.exact_replay is False
    assert record.recovery_scope == scope.value
    assert raw["decision_class"] == ReconciliationDecisionClass.evidence_conflict_confirmed.value
    assert raw["recommended_next_protocol"] == ReconciliationNextProtocol.recovery_run_review.value
    assert raw["recovery_of_run_id"] == source_run_id
    assert raw["successor_run_id"] == successor_run_id
    assert raw["case_open_digest"] == bundle.open_record.case_open_digest
    assert raw["observation_digest"] == bundle.observation_record.observation_digest
    assert raw["decision_digest"] == bundle.decision_record.decision_digest
    assert raw["finalized_valid_seal_evidence"]["seal_state"] == SealState.FINALIZED_VALID.value
    assert isinstance(raw["task25_approval_id"], str)
    assert isinstance(raw["task25_consumed_outcome_digest"], str)
    assert isinstance(raw["task26a_observation_inspection_digest"], str)
    assert isinstance(raw["task26b_decision_revalidation_inspection_digest"], str)
    assert isinstance(raw["htr_runs_root_path_digest"], str)
    assert isinstance(raw["htr_project_dir_path_digest"], str)


@pytest.mark.parametrize("scope", _PERMITTED_SCOPES)
def test_hardening_permitted_scopes_succeed_at_request(tmp_path, scope):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        record, _meta = create_recovery_run_request(
            generate_recovery_request_id(),
            case_id,
            recovery_scope=scope,
            recovery_reason="scope test",
            successor_run_id=generate_successor_run_id(),
            requested_by="operator",
            base_dir=tmp_path,
        )
    assert record.recovery_scope == scope.value


def test_hardening_request_rejects_arbitrary_scope_string(tmp_path):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        with pytest.raises(RecoveryRunValidationError, match="not permitted"):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope="totally_bogus_scope",
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


def test_hardening_request_scope_replay_with_different_scope_conflicts(tmp_path):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)
    recovery_request_id = generate_recovery_request_id()
    successor_run_id = generate_successor_run_id()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        create_recovery_run_request(
            recovery_request_id,
            case_id,
            recovery_scope=RecoveryScope.diagnostic_only,
            recovery_reason="first",
            successor_run_id=successor_run_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
        with pytest.raises(RecoveryRunConflictError, match="conflicting semantics"):
            create_recovery_run_request(
                recovery_request_id,
                case_id,
                recovery_scope=RecoveryScope.verification_only,
                recovery_reason="first",
                successor_run_id=successor_run_id,
                requested_by="operator",
                base_dir=tmp_path,
            )


@pytest.mark.parametrize(
    "seal_state,match",
    [
        (SealState.NOT_FINALIZED, "finalized_valid"),
        (SealState.CLOSURE_PRESENT_UNTRUSTED, "finalized_valid"),
        (SealState.INDETERMINATE, "finalized_valid"),
    ],
)
def test_hardening_request_rejects_non_finalized_seal_states(tmp_path, seal_state, match):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)
    evaluation = SealEvaluation(seal_state, ("test_reason",), source_run_id)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=evaluation):
        with pytest.raises(RecoveryRunValidationError, match=match):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


@pytest.mark.parametrize(
    "projection_overrides,match",
    [
        ({"overall_classification": "partial_lifecycle_commit"}, "partial_lifecycle_commit"),
        ({"overall_classification": "integrity_blocked", "integrity_state": "seal_blocked"}, "integrity_blocked"),
        ({"approval_control_state": "issue_only", "outcome_class": "issued"}, "consumed_outcome"),
        ({"overall_classification": "no_reconciliation_needed"}, "overall classification mismatch"),
    ],
)
def test_hardening_request_rejects_ineligible_inspection_projection(
    tmp_path, projection_overrides, match
):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)
    _tamper_decision_projection(case_id, tmp_path, **projection_overrides)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        with pytest.raises(RecoveryRunValidationError, match=match):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "decision_class",
            ReconciliationDecisionClass.indeterminate_insufficient_evidence.value,
            "decision class not eligible",
        ),
        (
            "recommended_next_protocol",
            ReconciliationNextProtocol.marker_disposition_review.value,
            "recovery_run_review",
        ),
        (
            "decision_class",
            ReconciliationDecisionClass.case_closed_deferred_to_protocol.value,
            "decision class not eligible",
        ),
    ],
)
def test_hardening_request_rejects_wrong_decision_class_or_protocol(
    tmp_path, field, value, match
):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)

    def _mutator(raw: dict[str, Any]) -> None:
        raw[field] = value

    _tamper_decision_raw(case_id, tmp_path, _mutator)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        with pytest.raises(RecoveryRunValidationError, match=match):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


def test_hardening_request_rejects_decision_time_drift(tmp_path):
    case_id, source_run_id = _path_r1_case_decided(tmp_path)

    def _mutator(raw: dict[str, Any]) -> None:
        drift = dict(raw.get("observation_decision_drift") or {})
        drift["drift_detected"] = True
        raw["observation_decision_drift"] = drift

    _tamper_decision_raw(case_id, tmp_path, _mutator)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        with pytest.raises(RecoveryRunValidationError, match="drift"):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="test",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=tmp_path,
            )


def test_hardening_request_rejects_cross_project_case(tmp_path):
    root_a = tmp_path / "project_a"
    root_b = tmp_path / "project_b"
    case_id, source_run_id = _path_r1_case_decided(root_a)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        with pytest.raises(Exception):
            create_recovery_run_request(
                generate_recovery_request_id(),
                case_id,
                recovery_scope=RecoveryScope.diagnostic_only,
                recovery_reason="cross project",
                successor_run_id=generate_successor_run_id(),
                requested_by="operator",
                base_dir=root_b,
            )


def test_hardening_issue_exact_replay(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    approval_id = generate_recovery_approval_id()
    expires_at = _expires_in_minutes(10)
    issue_recovery_run_approval(
        recovery_request_id,
        approval_id,
        issued_by="approver",
        expires_at=expires_at,
        base_dir=tmp_path,
    )
    _rec, meta = issue_recovery_run_approval(
        recovery_request_id,
        approval_id,
        issued_by="approver",
        expires_at=expires_at,
        base_dir=tmp_path,
    )
    assert meta.exact_replay is True


def test_hardening_issue_conflicting_replay_fails_closed(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunConflictError):
        issue_recovery_run_approval(
            recovery_request_id,
            generate_recovery_approval_id(),
            issued_by="other-approver",
            expires_at=_expires_in_minutes(10),
            base_dir=tmp_path,
        )


def test_hardening_issue_rejects_expiry_equal_to_issued_at(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issued_at = datetime.now(timezone.utc)
    with patch("htr.recovery_runs._utc_now", return_value=issued_at), patch(
        "htr.recovery_runs._utc_now_iso", return_value=issued_at.isoformat()
    ):
        with pytest.raises(RecoveryRunValidationError, match="expires_at"):
            issue_recovery_run_approval(
                recovery_request_id,
                generate_recovery_approval_id(),
                issued_by="approver",
                expires_at=issued_at.isoformat(),
                base_dir=tmp_path,
            )


def test_hardening_issue_rejects_expiry_beyond_fifteen_minutes(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issued_at = datetime.now(timezone.utc)
    expires_at = (issued_at + timedelta(minutes=16)).isoformat()
    with patch("htr.recovery_runs._utc_now", return_value=issued_at), patch(
        "htr.recovery_runs._utc_now_iso", return_value=issued_at.isoformat()
    ):
        with pytest.raises(RecoveryRunValidationError, match="lifetime exceeds"):
            issue_recovery_run_approval(
                recovery_request_id,
                generate_recovery_approval_id(),
                issued_by="approver",
                expires_at=expires_at,
                base_dir=tmp_path,
            )


def test_hardening_claim_before_expiry_succeeds(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issued_at = datetime.now(timezone.utc)
    expires_at = (issued_at + timedelta(minutes=10)).isoformat()
    claim_at = issued_at + timedelta(minutes=5)
    with patch("htr.recovery_runs._utc_now", return_value=issued_at), patch(
        "htr.recovery_runs._utc_now_iso", return_value=issued_at.isoformat()
    ):
        issue_recovery_run_approval(
            recovery_request_id,
            generate_recovery_approval_id(),
            issued_by="approver",
            expires_at=expires_at,
            base_dir=tmp_path,
        )
    with patch("htr.recovery_runs._utc_now", return_value=claim_at):
        claim, meta = claim_recovery_run_approval(
            recovery_request_id,
            generate_recovery_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )
    assert claim.claimant == "executor"
    assert meta.exact_replay is False


def test_hardening_claim_at_or_after_expiry_rejected(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issued_at = datetime.now(timezone.utc)
    expires_at = (issued_at + timedelta(minutes=10)).isoformat()
    with patch("htr.recovery_runs._utc_now", return_value=issued_at), patch(
        "htr.recovery_runs._utc_now_iso", return_value=issued_at.isoformat()
    ):
        issue_recovery_run_approval(
            recovery_request_id,
            generate_recovery_approval_id(),
            issued_by="approver",
            expires_at=expires_at,
            base_dir=tmp_path,
        )
    for claim_at in (issued_at + timedelta(minutes=10), issued_at + timedelta(minutes=11)):
        with patch("htr.recovery_runs._utc_now", return_value=claim_at):
            with pytest.raises(RecoveryRunValidationError, match="expired"):
                claim_recovery_run_approval(
                    recovery_request_id,
                    generate_recovery_claim_id(),
                    claimant="executor",
                    base_dir=tmp_path,
                )


def test_hardening_execute_at_or_after_expiry_rejected(tmp_path):
    recovery_request_id, _successor, _case_id, source_run_id = _path_r1_request(tmp_path)
    issued_at = datetime.now(timezone.utc)
    expires_at = (issued_at + timedelta(minutes=10)).isoformat()
    claim_at = issued_at + timedelta(minutes=5)
    execute_at = issued_at + timedelta(minutes=10)
    with patch("htr.recovery_runs._utc_now", return_value=issued_at), patch(
        "htr.recovery_runs._utc_now_iso", return_value=issued_at.isoformat()
    ):
        issue_recovery_run_approval(
            recovery_request_id,
            generate_recovery_approval_id(),
            issued_by="approver",
            expires_at=expires_at,
            base_dir=tmp_path,
        )
    with patch("htr.recovery_runs._utc_now", return_value=claim_at):
        claim_recovery_run_approval(
            recovery_request_id,
            generate_recovery_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )
    with patch("htr.recovery_runs._utc_now", return_value=execute_at), patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.approval_invalid.value


def test_hardening_claim_actor_mismatch_rejected(tmp_path):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_id = generate_recovery_claim_id()
    claim_recovery_run_approval(
        recovery_request_id,
        claim_id,
        claimant="executor",
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunConflictError):
        claim_recovery_run_approval(
            recovery_request_id,
            claim_id,
            claimant="other-executor",
            base_dir=tmp_path,
        )


def test_hardening_invalid_ids_rejected(tmp_path):
    with pytest.raises(RecoveryRunValidationError, match="invalid"):
        create_recovery_run_request(
            "not-a-valid-id",
            generate_reconciliation_case_id(),
            recovery_scope=RecoveryScope.diagnostic_only,
            recovery_reason="test",
            successor_run_id=generate_successor_run_id(),
            requested_by="operator",
            base_dir=tmp_path,
        )
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    issue_recovery_run_approval(
        recovery_request_id,
        generate_recovery_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    with pytest.raises(RecoveryRunValidationError, match="invalid"):
        claim_recovery_run_approval(
            recovery_request_id,
            "bad-claim-id",
            claimant="executor",
            base_dir=tmp_path,
        )
    with pytest.raises(RecoveryRunValidationError, match="invalid"):
        execute_approved_successor_run_creation(
            recovery_request_id,
            "bad-attempt-id",
            executor="executor",
            base_dir=tmp_path,
        )


def test_hardening_thread_concurrent_execute_one_creator_exact_replays(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_id = generate_recovery_attempt_id()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        first = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert first.outcome_class == RecoveryRunOutcomeClass.successor_created_verified.value

    barrier = threading.Barrier(_HARDENING_THREAD_COUNT)
    slots: list[tuple[str, bool] | Exception] = [()] * _HARDENING_THREAD_COUNT

    def _worker(index: int) -> None:
        try:
            barrier.wait()
            with patch(
                "htr.recovery_runs.evaluate_run_seal",
                return_value=_seal_finalized(source_run_id),
            ):
                result = execute_approved_successor_run_creation(
                    recovery_request_id,
                    attempt_id,
                    executor="executor",
                    base_dir=tmp_path,
                )
            slots[index] = (result.outcome_class, result.exact_replay)
        except Exception as exc:
            slots[index] = exc

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(_HARDENING_THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    ok = [s for s in slots if isinstance(s, tuple)]
    assert len(ok) == _HARDENING_THREAD_COUNT
    assert all(replay is True for _oc, replay in ok)
    assert all(
        oc
        in (
            RecoveryRunOutcomeClass.successor_created_verified.value,
            RecoveryRunOutcomeClass.successor_already_exists_verified.value,
        )
        for oc, _replay in ok
    )
    assert paths.run_root(successor_run_id, tmp_path).is_dir()


def test_hardening_successor_absent_before_attempt_record(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    successor_seen: list[bool] = []
    real_reserve = io.reserve_run_root_exclusive

    def _assert_successor_absent_before_reserve(*args: Any, **kwargs: Any):
        successor_seen.append(paths.run_root(successor_run_id, tmp_path).exists())
        return real_reserve(*args, **kwargs)

    with patch(
        "htr.recovery_runs.reserve_run_root_exclusive",
        side_effect=_assert_successor_absent_before_reserve,
    ), patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert successor_seen == [False]
    assert paths.recovery_run_attempt_path(recovery_request_id, tmp_path).is_file()


@pytest.mark.parametrize(
    "field",
    [
        "recovery_of_run_id",
        "recovery_request_digest",
        "task25_consumed_outcome_digest",
        "recovery_scope",
    ],
)
def test_hardening_recovery_origin_tampering_breaks_verification(tmp_path, field):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    origin_path = paths.recovery_origin_path(successor_run_id, tmp_path)
    raw = json.loads(origin_path.read_text(encoding="utf-8"))
    raw[field] = "tampered-value"
    origin_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    assert recovery_runs_mod._verify_successor_initial_state(successor_run_id, tmp_path) is True
    bundle = load_recovery_run_bundle(recovery_request_id, base_dir=tmp_path)
    request = io.read_json(paths.recovery_run_request_path(recovery_request_id, tmp_path))
    issue = io.read_json(paths.recovery_run_issue_path(recovery_request_id, tmp_path))
    claim = io.read_json(paths.recovery_run_claim_path(recovery_request_id, tmp_path))
    attempt = io.read_json(paths.recovery_run_attempt_path(recovery_request_id, tmp_path))
    assert recovery_runs_mod._successor_exists_verified(request, issue, claim, attempt, tmp_path) is False


def test_hardening_initial_successor_state_exact_tree(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    root = paths.run_root(successor_run_id, tmp_path)
    top_level = {p.name for p in root.iterdir()}
    assert top_level == {
        "recovery_origin.json",
        "run_manifest.json",
        "task_events.jsonl",
        "approvals.jsonl",
        "reports",
        "tasks",
    }
    marker = tmp_path / LOCKS_DIR_NAME / f"{successor_run_id}.marker"
    assert not marker.exists()
    assert io.read_json(paths.run_manifest_path(successor_run_id, tmp_path))["status"] == "created"
    assert io.read_json(paths.recovery_origin_path(successor_run_id, tmp_path))["successor_run_id"] == successor_run_id
    assert io.read_jsonl(paths.task_events_path(successor_run_id, tmp_path)) == []
    assert io.read_jsonl(paths.approvals_path(successor_run_id, tmp_path)) == []


@pytest.mark.parametrize(
    "setup,expected_outcome",
    [
        ("none", RecoveryRunOutcomeClass.successor_created_verified.value),
        ("preexisting_unrelated", RecoveryRunOutcomeClass.successor_id_conflict.value),
        ("verified_existing", RecoveryRunOutcomeClass.successor_already_exists_verified.value),
        ("partial_bootstrap", RecoveryRunOutcomeClass.creation_partial.value),
        ("bootstrap_failure", RecoveryRunOutcomeClass.creation_failed.value),
    ],
)
def test_hardening_successor_state_matrix(tmp_path, setup, expected_outcome):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_id = generate_recovery_attempt_id()
    patches: list[Any] = [
        patch(
            "htr.recovery_runs.evaluate_run_seal",
            return_value=_seal_finalized(source_run_id),
        )
    ]
    if setup == "preexisting_unrelated":
        io.create_run_workspace(successor_run_id, base_dir=tmp_path)
    elif setup == "verified_existing":
        first_attempt = generate_recovery_attempt_id()
        with patch(
            "htr.recovery_runs.evaluate_run_seal",
            return_value=_seal_finalized(source_run_id),
        ):
            execute_approved_successor_run_creation(
                recovery_request_id,
                first_attempt,
                executor="executor",
                base_dir=tmp_path,
            )
        paths.recovery_run_outcome_path(recovery_request_id, tmp_path).unlink()
        attempt_id = first_attempt
        expected_outcome = RecoveryRunOutcomeClass.successor_already_exists_verified.value
        patches = [
            patch(
                "htr.recovery_runs.evaluate_run_seal",
                return_value=_seal_finalized(source_run_id),
            )
        ]
    elif setup == "partial_bootstrap":
        patches.append(
            patch(
                "htr.recovery_runs._verify_successor_initial_state",
                return_value=False,
            )
        )
    elif setup == "bootstrap_failure":
        patches.append(
            patch(
                "htr.recovery_runs.bootstrap_reserved_run_workspace",
                side_effect=RuntimeError("bootstrap failed"),
            )
        )
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == expected_outcome


def test_hardening_all_ten_outcome_classes_are_distinct_enum_members():
    values = {member.value for member in RecoveryRunOutcomeClass}
    assert len(values) == 10


@pytest.mark.parametrize("outcome_class", list(RecoveryRunOutcomeClass))
def test_hardening_outcome_authority_booleans_must_be_false(outcome_class):
    body = recovery_runs_mod._outcome_body(
        recovery_request_id=generate_recovery_request_id(),
        outcome_class=outcome_class,
        request_digest="sha256:" + "a" * 64,
        issue_digest="sha256:" + "b" * 64,
        claim_digest="sha256:" + "c" * 64,
        attempt_digest="sha256:" + "d" * 64,
        successor_run_id=generate_successor_run_id(),
    )
    for field in _NON_PERMISSION_FIELDS:
        assert body[field] is False


@pytest.mark.parametrize("field", _NON_PERMISSION_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_BOOL_VALUES)
def test_hardening_reconcile_rejects_non_permission_boolean(tmp_path, field, bad_value):
    recovery_request_id, _expected = _setup_reconcile_classification(
        tmp_path, "valid_durable_outcome"
    )
    outcome_path = paths.recovery_run_outcome_path(recovery_request_id, tmp_path)
    raw = json.loads(outcome_path.read_text(encoding="utf-8"))
    raw[field] = bad_value
    outcome_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    result = reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    assert result.classification == "malformed_outcome"


@pytest.mark.parametrize("field", _NON_PERMISSION_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_BOOL_VALUES)
def test_hardening_load_bundle_rejects_non_permission_boolean(tmp_path, field, bad_value):
    recovery_request_id, _expected = _setup_reconcile_classification(
        tmp_path, "valid_durable_outcome"
    )
    outcome_path = paths.recovery_run_outcome_path(recovery_request_id, tmp_path)
    raw = json.loads(outcome_path.read_text(encoding="utf-8"))
    raw[field] = bad_value
    outcome_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RecoveryRunValidationError, match=field):
        load_recovery_run_bundle(recovery_request_id, base_dir=tmp_path)


@pytest.mark.parametrize("stage", _DURABILITY_STAGES)
def test_hardening_issue_durability_error_envelope(tmp_path, monkeypatch, stage):
    recovery_request_id, _successor, _case_id, _source = _path_r1_request(tmp_path)
    _inject_recovery_durability_failure(monkeypatch, stage)
    with pytest.raises(RecoveryRunDurabilityError) as excinfo:
        issue_recovery_run_approval(
            recovery_request_id,
            generate_recovery_approval_id(),
            issued_by="approver",
            expires_at=_expires_in_minutes(10),
            base_dir=tmp_path,
        )
    err = excinfo.value
    assert err.recovery_request_id == recovery_request_id
    assert err.record_name == "issue.json"
    assert err.durability_stage == stage if stage != "record_write" else "record_write"
    assert isinstance(err.record_may_have_committed, bool)
    assert err.successor_may_have_been_created is False


@pytest.mark.parametrize("classification", _RECONCILE_CLASSIFICATIONS)
def test_hardening_reconcile_classification_zero_write(tmp_path, classification):
    recovery_request_id, expected = _setup_reconcile_classification(tmp_path, classification)
    before = _control_snapshot(tmp_path)
    result = reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    assert before == _control_snapshot(tmp_path)
    assert result.classification == expected


def test_hardening_load_bundle_zero_write_when_outcome_present(tmp_path):
    recovery_request_id, _expected = _setup_reconcile_classification(
        tmp_path, "valid_durable_outcome"
    )
    before = _control_snapshot(tmp_path)
    bundle = load_recovery_run_bundle(recovery_request_id, base_dir=tmp_path)
    assert before == _control_snapshot(tmp_path)
    assert bundle.outcome_record is not None


def test_hardening_source_immutability_before_and_after_execute(tmp_path):
    recovery_request_id, successor_run_id, case_id, source_run_id = _full_chain(tmp_path)

    def _source_snapshot() -> dict[str, str]:
        snap: dict[str, str] = {}
        for rel in (
            paths.reconciliation_open_path(case_id, tmp_path),
            paths.reconciliation_observation_path(case_id, tmp_path),
            paths.reconciliation_decision_path(case_id, tmp_path),
        ):
            if rel.is_file():
                snap[str(rel.relative_to(tmp_path))] = _file_digest(rel)
        return snap

    before = _source_snapshot()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    after = _source_snapshot()
    assert before == after
    assert before
    assert paths.recovery_origin_path(successor_run_id, tmp_path).is_file()


def test_hardening_execute_source_evidence_drift_outcome(tmp_path):
    recovery_request_id, successor_run_id, case_id, source_run_id = _full_chain(tmp_path)
    decision_path = paths.reconciliation_decision_path(case_id, tmp_path)
    raw = json.loads(decision_path.read_text(encoding="utf-8"))
    raw["decision_class"] = ReconciliationDecisionClass.case_closed_no_action_required.value
    raw["decision_digest"] = _compute_decision_digest(raw)
    decision_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.source_evidence_drifted.value
    assert not paths.run_root(successor_run_id, tmp_path).exists()


def test_hardening_execution_ambiguous_conflicting_attempt(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    first_attempt = generate_recovery_attempt_id()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        execute_approved_successor_run_creation(
            recovery_request_id,
            first_attempt,
            executor="executor",
            base_dir=tmp_path,
        )
    paths.recovery_run_outcome_path(recovery_request_id, tmp_path).unlink()
    second_attempt = generate_recovery_attempt_id()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            second_attempt,
            executor="other-executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.execution_ambiguous.value


def test_hardening_public_api_excludes_private_helpers():
    import htr

    assert "reserve_run_root_exclusive" not in htr.__all__
    assert set(recovery_runs_mod.__all__) == {
        "RecoveryScope",
        "RecoveryRunOutcomeClass",
        "generate_recovery_request_id",
        "generate_recovery_approval_id",
        "generate_recovery_claim_id",
        "generate_recovery_attempt_id",
        "generate_successor_run_id",
        "create_recovery_run_request",
        "issue_recovery_run_approval",
        "revoke_recovery_run_approval",
        "claim_recovery_run_approval",
        "execute_approved_successor_run_creation",
        "load_recovery_run_bundle",
        "reconcile_recovery_run_creation",
    }


# --- Task 27 checkpoint: successor durability-stage fault injection ---

_EXECUTE_DURABILITY_STAGES = (
    "attempt_record_write",
    "attempt_record_fsync",
    "attempt_recovery_case_dir_fsync",
    "successor_root_reservation",
    "runs_root_fsync_after_reservation",
    "recovery_origin_write",
    "recovery_origin_record_fsync",
    "successor_root_fsync_after_origin",
    "recovery_origin_runs_root_fsync",
    "run_manifest_bootstrap",
    "task_events_bootstrap",
    "approvals_bootstrap",
    "reports_dir_bootstrap",
    "tasks_dir_bootstrap",
    "bootstrap_final_successor_root_fsync",
    "bootstrap_final_runs_root_fsync",
    "outcome_record_write",
    "outcome_record_fsync",
    "outcome_recovery_case_dir_fsync",
)


def _execute_durability_error(
    *,
    recovery_request_id: str,
    successor_run_id: str,
    record_name: str,
    durability_stage: str,
    record_may_have_committed: bool,
    successor_may_have_been_created: bool,
    exact_replay_status: str = "indeterminate",
) -> RecoveryRunDurabilityError:
    return RecoveryRunDurabilityError(
        f"injected {durability_stage}",
        recovery_request_id=recovery_request_id,
        successor_run_id=successor_run_id,
        record_name=record_name,  # type: ignore[arg-type]
        durability_stage=durability_stage,
        record_may_have_committed=record_may_have_committed,
        successor_may_have_been_created=successor_may_have_been_created,
        exact_replay_status=exact_replay_status,
    )


def _inject_execute_durability_stage(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    *,
    recovery_request_id: str,
    successor_run_id: str,
) -> None:
    def _err(
        record_name: str,
        durability_stage: str,
        *,
        record_may_have_committed: bool,
        successor_may_have_been_created: bool,
        exact_replay_status: str = "indeterminate",
    ) -> RecoveryRunDurabilityError:
        return _execute_durability_error(
            recovery_request_id=recovery_request_id,
            successor_run_id=successor_run_id,
            record_name=record_name,
            durability_stage=durability_stage,
            record_may_have_committed=record_may_have_committed,
            successor_may_have_been_created=successor_may_have_been_created,
            exact_replay_status=exact_replay_status,
        )

    if stage == "attempt_record_write":
        real = recovery_runs_mod._write_all

        def _fail(fd: int, payload: bytes, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "attempt.json":
                raise _err(
                    "attempt.json",
                    "record_write",
                    record_may_have_committed=False,
                    successor_may_have_been_created=False,
                    exact_replay_status="no",
                )
            real(fd, payload, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_write_all", _fail)
        return

    if stage == "attempt_record_fsync":
        real = recovery_runs_mod._fsync_file_fd

        def _fail(fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "attempt.json":
                raise _err(
                    "attempt.json",
                    "record_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=False,
                )
            real(fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_file_fd", _fail)
        return

    if stage == "attempt_recovery_case_dir_fsync":
        real = recovery_runs_mod._fsync_dir_fd

        def _fail(dir_fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "attempt.json" and kwargs.get("stage") == "recovery_case_dir_fsync":
                raise _err(
                    "attempt.json",
                    "recovery_case_dir_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=False,
                )
            real(dir_fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_dir_fd", _fail)
        return

    if stage == "successor_root_reservation":
        def _fail(*_args: Any, **_kwargs: Any) -> None:
            raise _err(
                "recovery_origin.json",
                "successor_root_reservation",
                record_may_have_committed=False,
                successor_may_have_been_created=False,
                exact_replay_status="no",
            )

        monkeypatch.setattr(recovery_runs_mod, "reserve_run_root_exclusive", _fail)
        return

    if stage == "runs_root_fsync_after_reservation":
        real = recovery_runs_mod.reserve_run_root_exclusive

        def _fail(run_id: str, base_dir: Path | None = None) -> Any:
            reservation = real(run_id, base_dir)
            release = recovery_runs_mod.release_run_root_reservation
            release(reservation)
            raise _err(
                "recovery_origin.json",
                "runs_root_fsync",
                record_may_have_committed=False,
                successor_may_have_been_created=True,
                exact_replay_status="no",
            )

        monkeypatch.setattr(recovery_runs_mod, "reserve_run_root_exclusive", _fail)
        return

    if stage == "recovery_origin_write":
        real = recovery_runs_mod._write_all

        def _fail(fd: int, payload: bytes, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "recovery_origin.json":
                raise _err(
                    "recovery_origin.json",
                    "recovery_origin_write",
                    record_may_have_committed=False,
                    successor_may_have_been_created=True,
                    exact_replay_status="no",
                )
            real(fd, payload, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_write_all", _fail)
        return

    if stage == "recovery_origin_record_fsync":
        real = recovery_runs_mod._fsync_file_fd

        def _fail(fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "recovery_origin.json":
                raise _err(
                    "recovery_origin.json",
                    "record_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=True,
                )
            real(fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_file_fd", _fail)
        return

    if stage == "successor_root_fsync_after_origin":
        real = recovery_runs_mod._fsync_dir_fd

        def _fail(dir_fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "recovery_origin.json" and kwargs.get("stage") == "successor_root_fsync":
                raise _err(
                    "recovery_origin.json",
                    "successor_root_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=True,
                )
            real(dir_fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_dir_fd", _fail)
        return

    if stage == "recovery_origin_runs_root_fsync":
        real = recovery_runs_mod._fsync_dir_fd

        def _fail(dir_fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "recovery_origin.json" and kwargs.get("stage") == "runs_root_fsync":
                raise _err(
                    "recovery_origin.json",
                    "runs_root_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=True,
                )
            real(dir_fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_dir_fd", _fail)
        return

    if stage == "run_manifest_bootstrap":
        def _fail(*_args: Any, **_kwargs: Any) -> None:
            raise _err(
                "run_manifest.json",
                "bootstrap_write",
                record_may_have_committed=False,
                successor_may_have_been_created=True,
                exact_replay_status="no",
            )

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "task_events_bootstrap":
        real = recovery_runs_mod.bootstrap_reserved_run_workspace
        real_touch = io._touch_jsonl_exclusive_at

        def _fail(run_id: str, base_dir: Path | None = None, *, reservation: Any) -> Path:
            def _touch_wrapper(dir_fd: int, name: str) -> None:
                if name == "task_events.jsonl":
                    raise _err(
                        "task_events.jsonl",
                        "bootstrap_write",
                        record_may_have_committed=False,
                        successor_may_have_been_created=True,
                        exact_replay_status="no",
                    )
                real_touch(dir_fd, name)

            monkeypatch.setattr(io, "_touch_jsonl_exclusive_at", _touch_wrapper)
            return real(run_id, base_dir, reservation=reservation)

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "approvals_bootstrap":
        real = recovery_runs_mod.bootstrap_reserved_run_workspace
        real_touch = io._touch_jsonl_exclusive_at

        def _fail(run_id: str, base_dir: Path | None = None, *, reservation: Any) -> Path:
            def _touch_wrapper(dir_fd: int, touch_name: str) -> None:
                if touch_name == "approvals.jsonl":
                    raise _err(
                        "approvals.jsonl",
                        "bootstrap_write",
                        record_may_have_committed=False,
                        successor_may_have_been_created=True,
                        exact_replay_status="no",
                    )
                real_touch(dir_fd, touch_name)

            monkeypatch.setattr(io, "_touch_jsonl_exclusive_at", _touch_wrapper)
            return real(run_id, base_dir, reservation=reservation)

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "reports_dir_bootstrap":
        real = recovery_runs_mod.bootstrap_reserved_run_workspace
        real_mkdir = io._mkdirat_name

        def _fail(run_id: str, base_dir: Path | None = None, *, reservation: Any) -> Path:
            def _mkdir(dir_fd: int, name: str, mode: int) -> bool:
                if name == "reports":
                    raise _err(
                        "run_manifest.json",
                        "bootstrap_write",
                        record_may_have_committed=False,
                        successor_may_have_been_created=True,
                        exact_replay_status="no",
                    )
                return real_mkdir(dir_fd, name, mode)

            monkeypatch.setattr(io, "_mkdirat_name", _mkdir)
            return real(run_id, base_dir, reservation=reservation)

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "tasks_dir_bootstrap":
        real = recovery_runs_mod.bootstrap_reserved_run_workspace
        real_mkdir = io._mkdirat_name

        def _fail(run_id: str, base_dir: Path | None = None, *, reservation: Any) -> Path:
            def _mkdir(dir_fd: int, name: str, mode: int) -> bool:
                if name == "tasks":
                    raise _err(
                        "run_manifest.json",
                        "bootstrap_write",
                        record_may_have_committed=False,
                        successor_may_have_been_created=True,
                        exact_replay_status="no",
                    )
                return real_mkdir(dir_fd, name, mode)

            monkeypatch.setattr(io, "_mkdirat_name", _mkdir)
            return real(run_id, base_dir, reservation=reservation)

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "bootstrap_final_successor_root_fsync":
        real = recovery_runs_mod.bootstrap_reserved_run_workspace
        real_fsync = io._fsync_dir_fd
        calls: list[int] = []

        def _fail(run_id: str, base_dir: Path | None = None, *, reservation: Any) -> Path:
            def _fsync(fd: int) -> None:
                calls.append(fd)
                if len(calls) == 1:
                    raise _err(
                        "run_manifest.json",
                        "successor_root_fsync",
                        record_may_have_committed=True,
                        successor_may_have_been_created=True,
                    )
                real_fsync(fd)

            monkeypatch.setattr(io, "_fsync_dir_fd", _fsync)
            return real(run_id, base_dir, reservation=reservation)

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "bootstrap_final_runs_root_fsync":
        real = recovery_runs_mod.bootstrap_reserved_run_workspace
        real_fsync = io._fsync_dir_fd
        calls: list[int] = []

        def _fail(run_id: str, base_dir: Path | None = None, *, reservation: Any) -> Path:
            def _fsync(fd: int) -> None:
                calls.append(fd)
                if len(calls) >= 2:
                    raise _err(
                        "run_manifest.json",
                        "runs_root_fsync",
                        record_may_have_committed=True,
                        successor_may_have_been_created=True,
                    )
                real_fsync(fd)

            monkeypatch.setattr(io, "_fsync_dir_fd", _fsync)
            return real(run_id, base_dir, reservation=reservation)

        monkeypatch.setattr(recovery_runs_mod, "bootstrap_reserved_run_workspace", _fail)
        return

    if stage == "outcome_record_write":
        real = recovery_runs_mod._write_all

        def _fail(fd: int, payload: bytes, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "outcome.json":
                raise _err(
                    "outcome.json",
                    "record_write",
                    record_may_have_committed=False,
                    successor_may_have_been_created=True,
                    exact_replay_status="no",
                )
            real(fd, payload, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_write_all", _fail)
        return

    if stage == "outcome_record_fsync":
        real = recovery_runs_mod._fsync_file_fd

        def _fail(fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "outcome.json":
                raise _err(
                    "outcome.json",
                    "record_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=True,
                )
            real(fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_file_fd", _fail)
        return

    if stage == "outcome_recovery_case_dir_fsync":
        real = recovery_runs_mod._fsync_dir_fd

        def _fail(dir_fd: int, **kwargs: Any) -> None:
            if kwargs.get("record_name") == "outcome.json" and kwargs.get("stage") == "recovery_case_dir_fsync":
                raise _err(
                    "outcome.json",
                    "recovery_case_dir_fsync",
                    record_may_have_committed=True,
                    successor_may_have_been_created=True,
                )
            real(dir_fd, **kwargs)

        monkeypatch.setattr(recovery_runs_mod, "_fsync_dir_fd", _fail)
        return

    raise ValueError(f"unknown execute durability stage: {stage}")


def _execute_stage_expectations(stage: str) -> dict[str, Any]:
    no_successor = {
        "attempt_present": True,
        "outcome_present": False,
        "successor_root_present": False,
        "successor_complete": False,
        "reconcile_classification": "attempt_without_successor",
    }
    unverified = {
        "attempt_present": True,
        "outcome_present": False,
        "successor_root_present": True,
        "successor_complete": False,
        "reconcile_classification": "attempt_with_unverified_successor",
    }
    verified_no_outcome = {
        "attempt_present": True,
        "outcome_present": False,
        "successor_root_present": True,
        "successor_complete": True,
        "reconcile_classification": "attempt_with_verified_successor",
    }
    if stage == "attempt_record_write":
        return {**no_successor, "attempt_present": False, "reconcile_classification": "claim_without_attempt"}
    if stage in (
        "attempt_record_fsync",
        "attempt_recovery_case_dir_fsync",
    ):
        return {**no_successor, "attempt_present": False, "reconcile_classification": "claim_without_attempt"}
    if stage == "successor_root_reservation":
        return no_successor
    if stage == "runs_root_fsync_after_reservation":
        return unverified
    if stage.startswith("recovery_origin_") or stage in (
        "successor_root_fsync_after_origin",
        "run_manifest_bootstrap",
        "task_events_bootstrap",
        "approvals_bootstrap",
        "reports_dir_bootstrap",
        "tasks_dir_bootstrap",
    ):
        return unverified
    if stage.startswith("bootstrap_final") or stage.startswith("outcome_"):
        return verified_no_outcome
    raise ValueError(stage)


def _successor_is_complete(successor_run_id: str, base_dir: Path) -> bool:
    root = paths.run_root(successor_run_id, base_dir)
    if not root.is_dir():
        return False
    required = (
        "recovery_origin.json",
        "run_manifest.json",
        "task_events.jsonl",
        "approvals.jsonl",
        "reports",
        "tasks",
    )
    return all((root / name).exists() for name in required)


def _run_recovery_subprocess_script(script: str) -> subprocess.CompletedProcess[str]:
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


def _execute_subprocess_body(
    tmp_path: Path,
    recovery_request_id: str,
    attempt_id: str,
    source_run_id: str,
    *,
    body_lines: list[str],
) -> subprocess.CompletedProcess[str]:
    lines = [
        "import os",
        "from unittest.mock import patch",
        "from pathlib import Path",
        "import htr.recovery_runs as rr",
        "from htr.finalization import SealEvaluation, SealState",
        "from htr.recovery_runs import execute_approved_successor_run_creation",
        f"base = Path({str(tmp_path)!r})",
        f"recovery_request_id = {recovery_request_id!r}",
        f"attempt_id = {attempt_id!r}",
        f"source_run_id = {source_run_id!r}",
        "seal = SealEvaluation(SealState.FINALIZED_VALID, (), source_run_id)",
        *body_lines,
    ]
    return _run_recovery_subprocess_script("\n".join(lines))


@pytest.mark.parametrize("stage", _EXECUTE_DURABILITY_STAGES)
def test_execute_durability_stage_fault_injection(tmp_path, monkeypatch, stage):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _inject_execute_durability_stage(
        monkeypatch,
        stage,
        recovery_request_id=recovery_request_id,
        successor_run_id=successor_run_id,
    )
    before = _control_snapshot(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        with pytest.raises(RecoveryRunDurabilityError) as excinfo:
            execute_approved_successor_run_creation(
                recovery_request_id,
                generate_recovery_attempt_id(),
                executor="executor",
                base_dir=tmp_path,
            )
    err = excinfo.value
    assert err.recovery_request_id == recovery_request_id
    assert err.successor_run_id == successor_run_id
    assert isinstance(err.record_may_have_committed, bool)
    assert isinstance(err.successor_may_have_been_created, bool)
    assert err.exact_replay_status in ("no", "indeterminate")
    assert err.durability_stage  # type: ignore[attr-defined]
    post_failure = _control_snapshot(tmp_path)
    expected = _execute_stage_expectations(stage)
    assert (
        paths.recovery_run_attempt_path(recovery_request_id, tmp_path).is_file()
        == expected["attempt_present"]
    )
    assert (
        paths.recovery_run_outcome_path(recovery_request_id, tmp_path).exists()
        == expected["outcome_present"]
    )
    assert paths.run_root(successor_run_id, tmp_path).exists() == expected["successor_root_present"]
    assert _successor_is_complete(successor_run_id, tmp_path) == expected["successor_complete"]
    reconcile = reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    assert reconcile.classification == expected["reconcile_classification"]
    assert post_failure == _control_snapshot(tmp_path)


def test_execute_initial_state_verification_failure_is_creation_partial_not_durability(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_id = generate_recovery_attempt_id()
    before = _control_snapshot(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ), patch(
        "htr.recovery_runs._verify_successor_initial_state",
        return_value=False,
    ):
        result = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == RecoveryRunOutcomeClass.creation_partial.value
    assert result.exact_replay is False
    assert paths.recovery_run_attempt_path(recovery_request_id, tmp_path).is_file()
    assert paths.recovery_run_outcome_path(recovery_request_id, tmp_path).is_file()
    assert paths.run_root(successor_run_id, tmp_path).exists()
    reconcile = reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    assert reconcile.classification == "valid_durable_outcome"
    assert before != _control_snapshot(tmp_path)
    replay_before = _control_snapshot(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        replay = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert replay.exact_replay is True
    assert replay.outcome_class == RecoveryRunOutcomeClass.creation_partial.value
    assert replay_before == _control_snapshot(tmp_path)


@pytest.mark.parametrize(
    "label,exit_code,body_lines,expect",
    [
        (
            "after_durable_attempt_before_reservation",
            71,
            [
                "real_create = rr._create_immutable_record",
                "def crash_after_attempt(**kwargs):",
                "    persisted, exact = real_create(**kwargs)",
                "    if kwargs.get('filename') == 'attempt.json' and not exact:",
                "        os._exit(71)",
                "    return persisted, exact",
                "with patch.object(rr, '_create_immutable_record', side_effect=crash_after_attempt), patch.object(rr, 'evaluate_run_seal', return_value=seal):",
                "    execute_approved_successor_run_creation(recovery_request_id, attempt_id, executor='executor', base_dir=base)",
            ],
            {"attempt": True, "outcome": False, "successor": False, "same_attempt_resumes": False},
        ),
        (
            "after_root_reservation",
            72,
            [
                "real_reserve = rr.reserve_run_root_exclusive",
                "def crash_after_reserve(*args, **kwargs):",
                "    real_reserve(*args, **kwargs)",
                "    os._exit(72)",
                "with patch.object(rr, 'reserve_run_root_exclusive', side_effect=crash_after_reserve), patch.object(rr, 'evaluate_run_seal', return_value=seal):",
                "    execute_approved_successor_run_creation(recovery_request_id, attempt_id, executor='executor', base_dir=base)",
            ],
            {"attempt": True, "outcome": False, "successor": True, "same_attempt_resumes": False, "conflict_on_resume": True},
        ),
        (
            "after_recovery_origin_write",
            73,
            [
                "real_origin = rr._write_recovery_origin_exclusive",
                "def crash_after_origin(reservation, body):",
                "    real_origin(reservation, body)",
                "    os._exit(73)",
                "with patch.object(rr, '_write_recovery_origin_exclusive', side_effect=crash_after_origin), patch.object(rr, 'evaluate_run_seal', return_value=seal):",
                "    execute_approved_successor_run_creation(recovery_request_id, attempt_id, executor='executor', base_dir=base)",
            ],
            {"attempt": True, "outcome": False, "successor": True, "same_attempt_resumes": False, "conflict_on_resume": True},
        ),
        (
            "during_partial_bootstrap",
            74,
            [
                "real_boot = rr.bootstrap_reserved_run_workspace",
                "def crash_mid_bootstrap(*args, **kwargs):",
                "    os._exit(74)",
                "with patch.object(rr, 'bootstrap_reserved_run_workspace', side_effect=crash_mid_bootstrap), patch.object(rr, 'evaluate_run_seal', return_value=seal):",
                "    execute_approved_successor_run_creation(recovery_request_id, attempt_id, executor='executor', base_dir=base)",
            ],
            {"attempt": True, "outcome": False, "successor": True, "same_attempt_resumes": False, "conflict_on_resume": True},
        ),
        (
            "after_complete_bootstrap_before_outcome",
            75,
            [
                "real_verify = rr._verify_successor_initial_state",
                "def crash_after_verify(successor_run_id, base_dir):",
                "    assert real_verify(successor_run_id, base_dir)",
                "    os._exit(75)",
                "with patch.object(rr, '_verify_successor_initial_state', side_effect=crash_after_verify), patch.object(rr, 'evaluate_run_seal', return_value=seal):",
                "    execute_approved_successor_run_creation(recovery_request_id, attempt_id, executor='executor', base_dir=base)",
            ],
            {"attempt": True, "outcome": False, "successor": True, "same_attempt_resumes": True},
        ),
    ],
)
def test_subprocess_crash_state_never_blindly_resumes_or_adopts_partial(
    tmp_path, label, exit_code, body_lines, expect
):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_id = generate_recovery_attempt_id()
    proc = _execute_subprocess_body(
        tmp_path,
        recovery_request_id,
        attempt_id,
        source_run_id,
        body_lines=body_lines,
    )
    assert proc.returncode == exit_code, proc.stderr
    assert (
        paths.recovery_run_attempt_path(recovery_request_id, tmp_path).is_file()
        == expect["attempt"]
    )
    assert (
        paths.recovery_run_outcome_path(recovery_request_id, tmp_path).exists()
        == expect["outcome"]
    )
    assert paths.run_root(successor_run_id, tmp_path).exists() == expect["successor"]
    before = _control_snapshot(tmp_path)
    reconcile = reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    assert reconcile.classification in (
        "attempt_without_successor",
        "attempt_with_verified_successor",
        "attempt_with_unverified_successor",
    )
    assert before == _control_snapshot(tmp_path)
    other_attempt = generate_recovery_attempt_id()
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        ambiguous = execute_approved_successor_run_creation(
            recovery_request_id,
            other_attempt,
            executor="other-executor",
            base_dir=tmp_path,
        )
    assert ambiguous.outcome_class == RecoveryRunOutcomeClass.execution_ambiguous.value
    assert ambiguous.exact_replay is False
    if expect["same_attempt_resumes"]:
        with patch(
            "htr.recovery_runs.evaluate_run_seal",
            return_value=_seal_finalized(source_run_id),
        ):
            replay = execute_approved_successor_run_creation(
                recovery_request_id,
                attempt_id,
                executor="executor",
                base_dir=tmp_path,
            )
        assert replay.exact_replay is True
        assert replay.outcome_class in (
            RecoveryRunOutcomeClass.successor_created_verified.value,
            RecoveryRunOutcomeClass.successor_already_exists_verified.value,
        )
    elif expect.get("conflict_on_resume"):
        with patch(
            "htr.recovery_runs.evaluate_run_seal",
            return_value=_seal_finalized(source_run_id),
        ):
            conflict = execute_approved_successor_run_creation(
                recovery_request_id,
                attempt_id,
                executor="executor",
                base_dir=tmp_path,
            )
        assert conflict.outcome_class == RecoveryRunOutcomeClass.successor_id_conflict.value
        assert conflict.exact_replay is False
    else:
        with patch(
            "htr.recovery_runs.evaluate_run_seal",
            return_value=_seal_finalized(source_run_id),
        ):
            with pytest.raises(RecoveryRunStateError):
                execute_approved_successor_run_creation(
                    recovery_request_id,
                    attempt_id,
                    executor="executor",
                    base_dir=tmp_path,
                )


def test_subprocess_crash_response_lost_after_durable_outcome_replays_exactly(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    attempt_id = generate_recovery_attempt_id()
    proc = _execute_subprocess_body(
        tmp_path,
        recovery_request_id,
        attempt_id,
        source_run_id,
        body_lines=[
            "real_persist = rr._persist_outcome_private",
            "def crash_after_outcome(**kwargs):",
            "    result = real_persist(**kwargs)",
            "    os._exit(76)",
            "    return result",
            "with patch.object(rr, '_persist_outcome_private', side_effect=crash_after_outcome), patch.object(rr, 'evaluate_run_seal', return_value=seal):",
            "    execute_approved_successor_run_creation(recovery_request_id, attempt_id, executor='executor', base_dir=base)",
        ],
    )
    assert proc.returncode == 76, proc.stderr
    assert paths.recovery_run_outcome_path(recovery_request_id, tmp_path).is_file()
    before = _control_snapshot(tmp_path)
    with patch(
        "htr.recovery_runs.evaluate_run_seal",
        return_value=_seal_finalized(source_run_id),
    ):
        replay = execute_approved_successor_run_creation(
            recovery_request_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert replay.exact_replay is True
    assert replay.outcome_class == RecoveryRunOutcomeClass.successor_created_verified.value
    assert before == _control_snapshot(tmp_path)
    reconcile = reconcile_recovery_run_creation(recovery_request_id, base_dir=tmp_path)
    assert reconcile.classification == "valid_durable_outcome"
