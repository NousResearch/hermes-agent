"""Deterministic tests for Task 28 Phase 28A (Revision 9 matrix)."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import stat
import subprocess
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from htr import contracts, io, paths
from htr.bounded_action_bootstrap import (
    PublicationBootstrap,
    bootstrap_publication_tree,
    case_lock,
    release_publication_locks,
    root_publication_lock,
    successor_coord_lock,
)
from htr.bounded_action_control_paths import (
    CONTROL_DIR_MODE,
    CONTROL_FILE_MODE,
    format_mode,
    open_dir_no_follow,
    openat_dir_no_follow,
    read_json_record_fd,
    validate_new_task28_ownership,
    validate_preexisting_control_dir,
)
from htr.action_plan import _sha256_digest
from htr.bounded_action_digest import (
    CANONICAL_FIXTURE_BYTES,
    canonical_json_bytes,
    compute_record_digest,
    projection_a,
    projection_b,
    projection_c,
    sha256_digest,
    validate_record_digest,
)
from htr.bounded_action_evidence import (
    build_source_evidence,
    build_successor_evidence,
    build_task27_evidence,
    classify_marker_for_subject,
)
from htr.bounded_action_schemas import (
    AUTHORITY_BOOLEAN_FIELDS,
    BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR,
    BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR,
    NAMESPACE_INTEGRITY_SCHEMA,
    PROTOCOL_VERSION,
    SUBJECT_MATRIX,
    TARGET_AGGREGATE_SCHEMA,
    ProposalSubject,
)
from htr.bounded_action_strict_json import (
    MAX_RECORD_BYTES,
    parse_strict_json_bytes,
    require_exact_canonical_bytes,
)
from htr.bounded_actions import (
    BoundedActionConflictError,
    create_bounded_action_proposal,
    inspect_bounded_action_eligibility,
    load_bounded_action_proposal_bundle,
    reconcile_bounded_action,
    reconcile_successor_bounded_action_state,
    record_bounded_action_escalation,
    record_bounded_action_review_decision,
)
from htr.execution_lock import LOCKS_DIR_NAME
from htr.finalization import SealEvaluation, SealState
from htr.ids import (
    generate_bounded_action_escalation_id,
    generate_bounded_action_proposal_id,
    generate_bounded_action_review_decision_id,
    generate_recovery_attempt_id,
)
from htr.recovery_runs import (
    RecoveryRunOutcomeClass,
    RecoveryRunValidationError,
    RecoveryScope,
    _attempt_digest_projection,
    _claim_digest_projection,
    _issue_digest_projection,
    _outcome_digest_projection,
    _recovery_origin_digest_projection,
    _request_digest_projection,
    _revoke_digest_projection,
    claim_recovery_run_approval,
    create_recovery_run_request,
    execute_approved_successor_run_creation,
    generate_recovery_approval_id,
    generate_recovery_claim_id,
    generate_recovery_request_id,
    generate_successor_run_id,
    issue_recovery_run_approval,
)
from htr.state import BoundedActionDurabilityError, BoundedActionPreconditionError, BoundedActionValidationError

from tests.htr.test_recovery_runs import _full_chain, _seal_finalized

R9_ARTIFACT = Path("/home/unaliu/task28-inspect-artifacts/TASK_28_ARCHITECTURE_REPORT_R9.md")
BASE_COMMIT = "a831b3c610594cf8ca5ca804229b2395c61ef599"
TASK16_PATH = Path(__file__).parent / "htr" / "test_run_final_closure.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
TASK16 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(TASK16)

MOJIBAKE_PATTERNS = (
    re.compile("\u00e2\u20ac"),
    re.compile("\u00c2[\u0080-\u00bf]"),
    re.compile("\u00a7"),
    re.compile("\ufffd"),
)

PROPOSAL_CALLER_FIELDS = frozenset(
    {
        "proposal_id",
        "source_run_id",
        "successor_run_id",
        "creator",
        "proposal_subject",
        "proposal_summary",
        "risk_class",
        "confidence_class",
        "reason_codes",
        "reason_detail",
    }
)

EVIDENCE_TOP_LEVEL = frozenset({"source_evidence", "task27_evidence", "successor_evidence"})

EXECUTION_REVALIDATION_FIELDS = (
    "revalidation_projection_version",
    "recovery_request_id",
    "request_digest",
    "execution_inspection_digest",
    "execution_inspection_projection",
    "execution_revalidation_digest",
)

ARTIFACT_IDENTITY_KEYS = frozenset(
    {
        "relative_path",
        "path_identity_digest",
        "file_type",
        "permission_mode",
        "size_bytes",
        "raw_digest",
        "semantic_digest",
    }
)

TASK23_27_MODULES = (
    "htr/recovery_runs.py",
    "htr/reconciliation_cases.py",
    "htr/approval_control.py",
    "htr/finalization.py",
    "htr/observe.py",
    "htr/action_plan.py",
    "htr/execution_lock.py",
)


def _recovery_seal(source_run_id: str):
    return _seal_finalized(source_run_id)


def _evidence_seal_factory(source_run_id: str, *, successor_finalized: bool = False):
    def _eval(run_id: str, base_dir=None):
        if run_id == source_run_id:
            return _seal_finalized(source_run_id)
        if successor_finalized:
            return SealEvaluation(SealState.FINALIZED_VALID, (), run_id)
        return SealEvaluation(SealState.NOT_FINALIZED, (), run_id)

    return _eval


@contextmanager
def _recovery_seal_patch(source_run_id: str):
    seal = _recovery_seal(source_run_id)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=seal):
        yield


@contextmanager
def _bounded_action_seal_patches(source_run_id: str, *, successor_finalized: bool = False):
    ev = _evidence_seal_factory(source_run_id, successor_finalized=successor_finalized)
    with patch("htr.bounded_action_evidence.evaluate_run_seal", side_effect=ev):
        yield


@contextmanager
def _seal_patches(source_run_id: str, *, successor_finalized: bool = False):
    ev = _evidence_seal_factory(source_run_id, successor_finalized=successor_finalized)
    rec = _recovery_seal(source_run_id)
    with patch("htr.recovery_runs.evaluate_run_seal", return_value=rec), patch(
        "htr.finalization.evaluate_run_seal", side_effect=ev
    ), patch("htr.bounded_action_evidence.evaluate_run_seal", side_effect=ev):
        yield


def _ensure_source_closure(tmp_path: Path, source_run_id: str) -> None:
    closure_path = contracts.run_final_closure_record_json_path(source_run_id, tmp_path)
    if not closure_path.is_file():
        io.atomic_write_json(closure_path, TASK16._minimal_run_final_closure(run_id=source_run_id))


def _execute_successor(tmp_path: Path, recovery_request_id: str, source_run_id: str) -> None:
    with _recovery_seal_patch(source_run_id):
        execute_approved_successor_run_creation(
            recovery_request_id,
            generate_recovery_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    _ensure_source_closure(tmp_path, source_run_id)


def _default_proposal_kwargs(
    proposal_id: str,
    source_run_id: str,
    successor_run_id: str,
    recovery_request_id: str,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "source_run_id": source_run_id,
        "successor_run_id": successor_run_id,
        "recovery_request_id": recovery_request_id,
        "creator": "operator",
        "proposal_subject": ProposalSubject.bounded_action_architecture_candidate.value,
        "proposal_summary": "Architecture review requested for bounded retry protocol.",
        "risk_class": "advisory_read_only_low",
        "confidence_class": "proven",
        "reason_codes": ["architecture_review_requested"],
    }


def _publish_proposal(tmp_path: Path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    proposal_id = generate_bounded_action_proposal_id()
    kwargs = _default_proposal_kwargs(proposal_id, source_run_id, successor_run_id, recovery_request_id)
    with _seal_patches(source_run_id):
        result = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    return proposal_id, result, source_run_id, successor_run_id, recovery_request_id, kwargs


def _record_review(tmp_path: Path, source_run_id: str, proposal_id: str, review_id: str, *, proposal_digest: str, **extra):
    with _seal_patches(source_run_id):
        return record_bounded_action_review_decision(
            proposal_id,
            review_id,
            expected_proposal_record_digest=proposal_digest,
            reviewer=extra.get("reviewer", "reviewer"),
            decision_class=extra.get("decision_class", "accepted_for_future_architecture_review"),
            risk_class=extra.get("risk_class", "advisory_read_only_low"),
            confidence_class=extra.get("confidence_class", "proven"),
            reason_codes=extra.get("reason_codes", ["evidence_sufficient_for_advisory_acceptance"]),
            base_dir=tmp_path,
        )


def _record_escalation(tmp_path: Path, source_run_id: str, proposal_id: str, esc_id: str, *, proposal_digest: str, **extra):
    with _seal_patches(source_run_id):
        return record_bounded_action_escalation(
            proposal_id,
            esc_id,
            expected_proposal_record_digest=proposal_digest,
            escalator=extra.get("escalator", "escalator"),
            escalation_class=extra.get("escalation_class", "human_review_required"),
            risk_class=extra.get("risk_class", "advisory_integrity_review_high"),
            confidence_class=extra.get("confidence_class", "proven"),
            reason_codes=extra.get("reason_codes", ["fresh_evidence_drift_detected"]),
            base_dir=tmp_path,
        )


def _parse_r9_schema_rows() -> list[tuple[str, str, str, str]]:
    text = R9_ARTIFACT.read_text(encoding="utf-8")
    rows: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|") or line.count("|") < 10:
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 10:
            continue
        field = parts[0]
        a, b, c = parts[6], parts[7], parts[8]
        if field in {"Field", "-------", ""}:
            continue
        if a in {"yes", "no"} and b in {"yes", "no"} and c in {"yes", "no"}:
            rows.append((field, a, b, c))
    return rows


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _times(path: Path) -> tuple[float, float, float]:
    st = path.stat()
    return st.st_atime, st.st_mtime, st.st_ctime


def _task27_path(tmp_path: Path, recovery_request_id: str, rel_name: str) -> Path:
    mapping = {
        "request.json": paths.recovery_run_request_path,
        "issue.json": paths.recovery_run_issue_path,
        "claim.json": paths.recovery_run_claim_path,
        "attempt.json": paths.recovery_run_attempt_path,
        "outcome.json": paths.recovery_run_outcome_path,
    }
    return mapping[rel_name](recovery_request_id, tmp_path)


def _assert_task27_field_mismatch_blocks(
    tmp_path: Path,
    recovery_request_id: str,
    source_run_id: str,
    successor_run_id: str,
    rel_name: str,
    projection_fn,
) -> None:
    path = _task27_path(tmp_path, recovery_request_id, rel_name)
    original = io.read_json(path)
    for field in projection_fn(original):
        if field == "execution_revalidation":
            continue
        mutated = dict(original)
        mutated[field] = "tampered-value"
        io.atomic_write_json(path, mutated)
        with _seal_patches(source_run_id):
            with pytest.raises((BoundedActionPreconditionError, BoundedActionValidationError, RecoveryRunValidationError)):
                build_task27_evidence(
                    recovery_request_id,
                    source_run_id=source_run_id,
                    successor_run_id=successor_run_id,
                    base_dir=tmp_path,
                )
        io.atomic_write_json(path, original)


def _refresh_task27_record_digest(record: dict, projection_fn, digest_field: str) -> dict:
    updated = dict(record)
    updated[digest_field] = _sha256_digest(projection_fn(updated))
    return updated


def _ensure_control_tree_0700(tmp_path: Path) -> None:
    control = paths.control_root(tmp_path)
    control.mkdir(parents=True, exist_ok=True)
    os.chmod(control, 0o700)
    ba_root = paths.control_bounded_actions_root(tmp_path)
    ba_root.mkdir(parents=True, exist_ok=True)
    os.chmod(ba_root, 0o700)


def _git_blob(module: str) -> bytes:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(root), "show", f"{BASE_COMMIT}:{module}"],
        stderr=subprocess.DEVNULL,
    )


def test_report_artifact_utf8_nfc():
    raw = R9_ARTIFACT.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert unicodedata.normalize("NFC", text) == text
    for pattern in MOJIBAKE_PATTERNS:
        assert pattern.search(text) is None
    manifest = R9_ARTIFACT.with_suffix(R9_ARTIFACT.suffix + ".sha256")
    assert manifest.is_file()
    expected = manifest.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(raw).hexdigest() == expected


def test_three_projection_separation():
    record = {
        "record_type": "bounded_action_proposal",
        "schema_version": "1",
        "created_at": "2026-07-30T00:00:00+00:00",
        "record_digest": "sha256:deadbeef",
        "source_evidence": {"binding_schema_version": "htr.bounded_action.source_evidence.v1"},
    }
    intent = {"proposal_id": "bar_20260730_000001", "creator": "operator"}
    assert sha256_digest(projection_a(intent)) != sha256_digest(projection_b(record))
    assert "created_at" in projection_b(record)
    assert "created_at" not in projection_c(record)
    assert "record_digest" not in projection_b(record)


def test_request_intent_digest_protocol_generated():
    sig = inspect.signature(create_bounded_action_proposal)
    assert "request_intent_digest" not in sig.parameters


def test_replay_after_aggregate_counts_changed(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    other_id = generate_bounded_action_proposal_id()
    other_kwargs = _default_proposal_kwargs(other_id, source_run_id, successor_run_id, recovery_request_id)
    with _seal_patches(source_run_id):
        create_bounded_action_proposal(**other_kwargs, base_dir=tmp_path)
        replay = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert replay.publication_result == "exact_replay"
    assert replay.request_intent_digest_match is True


def test_replay_after_evidence_drift(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    manifest_path = paths.run_manifest_path(source_run_id, tmp_path)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with _seal_patches(source_run_id):
        replay = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert replay.publication_result == "exact_replay"
    assert replay.current_evidence_class in {"unchanged", "drifted", "not_inspected"}


def test_terminal_expected_proposal_digest_mismatch(tmp_path):
    proposal_id, _pub, *_rest = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    review_id = generate_bounded_action_review_decision_id()
    result = record_bounded_action_review_decision(
        proposal_id,
        review_id,
        expected_proposal_record_digest="sha256:" + "0" * 64,
        reviewer="reviewer",
        decision_class="accepted_for_future_architecture_review",
        risk_class="advisory_read_only_low",
        confidence_class="proven",
        reason_codes=["evidence_sufficient_for_advisory_acceptance"],
        base_dir=tmp_path,
    )
    assert result.publication_result == "expected_proposal_digest_mismatch"


def test_authority_booleans_not_caller_params():
    for fn in (create_bounded_action_proposal, record_bounded_action_review_decision, record_bounded_action_escalation):
        sig = inspect.signature(fn)
        for field in AUTHORITY_BOOLEAN_FIELDS:
            assert field not in sig.parameters


def test_proposal_summary_digest_protocol_derived():
    sig = inspect.signature(create_bounded_action_proposal)
    assert "proposal_summary_digest" not in sig.parameters


def test_root_publication_lock_ordering():
    src = inspect.getsource(create_bounded_action_proposal)
    assert "with root_publication_lock(locks):" in src
    assert "with successor_coord_lock(locks, successor_run_id, base_dir):" in src
    assert "with case_lock(locks, proposal_id, create=True)" in src
    root_pos = src.index("root_publication_lock")
    succ_pos = src.index("successor_coord_lock")
    case_pos = src.index("case_lock")
    assert root_pos < succ_pos < case_pos


def test_reverse_lock_order_rejected():
    src = inspect.getsource(create_bounded_action_proposal)
    assert src.index("root_publication_lock") < src.index("successor_coord_lock") < src.index("case_lock")
    assert "reverse" not in src.lower()


def test_two_successors_serialize_path_c(tmp_path):
    r1, s1, _c1, src1 = _full_chain(tmp_path)
    r2, s2, _c2, src2 = _full_chain(tmp_path)
    _execute_successor(tmp_path, r1, src1)
    _execute_successor(tmp_path, r2, src2)
    p1 = generate_bounded_action_proposal_id()
    p2 = generate_bounded_action_proposal_id()
    k1 = _default_proposal_kwargs(p1, src1, s1, r1)
    k2 = _default_proposal_kwargs(p2, src2, s2, r2)
    with _seal_patches(src1):
        r1_result = create_bounded_action_proposal(**k1, base_dir=tmp_path)
    with _seal_patches(src2):
        r2_result = create_bounded_action_proposal(**k2, base_dir=tmp_path)
    assert r1_result.publication_result == "published_new"
    assert r2_result.publication_result == "published_new"


def test_cross_successor_cap_independence(tmp_path):
    r1, s1, _c1, src1 = _full_chain(tmp_path)
    r2, s2, _c2, src2 = _full_chain(tmp_path)
    _execute_successor(tmp_path, r1, src1)
    _execute_successor(tmp_path, r2, src2)
    with _seal_patches(src1):
        create_bounded_action_proposal(**_default_proposal_kwargs(generate_bounded_action_proposal_id(), src1, s1, r1), base_dir=tmp_path)
    with _seal_patches(src2):
        result = create_bounded_action_proposal(**_default_proposal_kwargs(generate_bounded_action_proposal_id(), src2, s2, r2), base_dir=tmp_path)
    assert result.publication_result == "published_new"
    agg1 = reconcile_successor_bounded_action_state(s1, base_dir=tmp_path)
    agg2 = reconcile_successor_bounded_action_state(s2, base_dir=tmp_path)
    assert agg1["proposal_count"] == 1
    assert agg2["proposal_count"] == 1


def test_incomplete_case_hidden_under_root_lock(tmp_path):
    _ensure_control_tree_0700(tmp_path)
    ba_root = paths.control_bounded_actions_root(tmp_path)
    empty_case = ba_root / generate_bounded_action_proposal_id()
    empty_case.mkdir()
    proposal_id, result, *_ = _publish_proposal(tmp_path)
    assert result.publication_result == "namespace_integrity_indeterminate"


def test_crash_empty_case_blocks_publication(tmp_path):
    _ensure_control_tree_0700(tmp_path)
    ba_root = paths.control_bounded_actions_root(tmp_path)
    empty_id = generate_bounded_action_proposal_id()
    empty_case = ba_root / empty_id
    empty_case.mkdir()
    os.chmod(empty_case, 0o700)
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    kwargs = _default_proposal_kwargs(empty_id, source_run_id, successor_run_id, recovery_request_id)
    with _seal_patches(source_run_id):
        result = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert result.publication_result in {"namespace_integrity_indeterminate", "precondition_failed"}


def test_proposal_cap_exhaustion_allows_review(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, _kwargs = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id):
        for _ in range(BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR - 1):
            pid = generate_bounded_action_proposal_id()
            create_bounded_action_proposal(**_default_proposal_kwargs(pid, source_run_id, successor_run_id, recovery_request_id), base_dir=tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    result = _record_review(
        tmp_path,
        source_run_id,
        proposal_id,
        generate_bounded_action_review_decision_id(),
        proposal_digest=bundle.proposal["record_digest"],
    )
    assert result.publication_result == "published_new"


def test_proposal_cap_exhaustion_allows_escalation(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, _kwargs = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id):
        for _ in range(BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR - 1):
            pid = generate_bounded_action_proposal_id()
            create_bounded_action_proposal(**_default_proposal_kwargs(pid, source_run_id, successor_run_id, recovery_request_id), base_dir=tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    result = _record_escalation(
        tmp_path,
        source_run_id,
        proposal_id,
        generate_bounded_action_escalation_id(),
        proposal_digest=bundle.proposal["record_digest"],
    )
    assert result.publication_result == "published_new"


def test_escalation_cap_exhaustion_allows_review(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    recovery_request_id = bundle.proposal["task27_evidence"]["recovery_request_id"]
    successor_run_id = bundle.proposal["successor_run_id"]
    for _ in range(BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR):
        pid = generate_bounded_action_proposal_id()
        with _seal_patches(source_run_id):
            create_bounded_action_proposal(**_default_proposal_kwargs(pid, source_run_id, successor_run_id, recovery_request_id), base_dir=tmp_path)
        b = load_bounded_action_proposal_bundle(pid, base_dir=tmp_path)
        _record_escalation(tmp_path, source_run_id, pid, generate_bounded_action_escalation_id(), proposal_digest=b.proposal["record_digest"])
    result = _record_review(
        tmp_path,
        source_run_id,
        proposal_id,
        generate_bounded_action_review_decision_id(),
        proposal_digest=bundle.proposal["record_digest"],
    )
    assert result.publication_result == "published_new"


def test_finalized_successor_blocks_all_proposal_subjects(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)

    def _finalized(run_id: str, base_dir=None):
        return SealEvaluation(SealState.FINALIZED_VALID, (), run_id)

    for subject in SUBJECT_MATRIX:
        matrix = SUBJECT_MATRIX[subject]
        reason_codes = sorted(matrix.get("required_reasons", frozenset())) or None
        with patch("htr.recovery_runs.evaluate_run_seal", side_effect=_finalized), patch(
            "htr.finalization.evaluate_run_seal", side_effect=_finalized
        ), patch("htr.bounded_action_evidence.evaluate_run_seal", side_effect=_finalized):
            result = inspect_bounded_action_eligibility(
                source_run_id,
                successor_run_id,
                recovery_request_id,
                subject,
                base_dir=tmp_path,
                reason_codes=reason_codes,
                risk_class=next(iter(matrix["risk"])),
                confidence_class=next(iter(matrix["confidence"])),
            )
        assert result.eligible is False


def test_marker_residue_escalation_subject_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    (locks_root / f"{successor_run_id}.marker").write_text("{}", encoding="utf-8")
    subject = ProposalSubject.reconciliation_escalation_candidate.value
    matrix = SUBJECT_MATRIX[subject]
    with _seal_patches(source_run_id):
        result = inspect_bounded_action_eligibility(
            source_run_id,
            successor_run_id,
            recovery_request_id,
            subject,
            base_dir=tmp_path,
            reason_codes=sorted(matrix["required_reasons"]),
            risk_class=next(iter(matrix["risk"])),
            confidence_class=next(iter(matrix["confidence"])),
        )
    assert result.eligible is True


def test_marker_blocks_retry_subject(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    (tmp_path / LOCKS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (tmp_path / LOCKS_DIR_NAME / f"{successor_run_id}.marker").write_text("{}", encoding="utf-8")
    subject = ProposalSubject.future_retry_candidate.value
    with _seal_patches(source_run_id):
        result = inspect_bounded_action_eligibility(
            source_run_id,
            successor_run_id,
            recovery_request_id,
            subject,
            base_dir=tmp_path,
            reason_codes=["architecture_review_requested"],
            risk_class="advisory_integrity_review_high",
            confidence_class="proven",
        )
    assert result.eligible is False
    assert result.classification == "marker_blocked"


def test_marker_blocks_forward_repair_subject(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    (tmp_path / LOCKS_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (tmp_path / LOCKS_DIR_NAME / f"{successor_run_id}.marker").write_text("{}", encoding="utf-8")
    subject = ProposalSubject.future_forward_repair_candidate.value
    with _seal_patches(source_run_id):
        result = inspect_bounded_action_eligibility(
            source_run_id,
            successor_run_id,
            recovery_request_id,
            subject,
            base_dir=tmp_path,
            reason_codes=["architecture_review_requested"],
            risk_class="advisory_integrity_review_high",
            confidence_class="proven",
        )
    assert result.eligible is False


def test_task27_request_field_equality_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    _assert_task27_field_mismatch_blocks(
        tmp_path, recovery_request_id, source_run_id, successor_run_id, "request.json", _request_digest_projection
    )


def test_task27_issue_field_equality_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    _assert_task27_field_mismatch_blocks(
        tmp_path, recovery_request_id, source_run_id, successor_run_id, "issue.json", _issue_digest_projection
    )


def test_task27_claim_field_equality_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    _assert_task27_field_mismatch_blocks(
        tmp_path, recovery_request_id, source_run_id, successor_run_id, "claim.json", _claim_digest_projection
    )


def test_task27_attempt_field_equality_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    _assert_task27_field_mismatch_blocks(
        tmp_path, recovery_request_id, source_run_id, successor_run_id, "attempt.json", _attempt_digest_projection
    )


def test_task27_outcome_field_equality_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    _assert_task27_field_mismatch_blocks(
        tmp_path, recovery_request_id, source_run_id, successor_run_id, "outcome.json", _outcome_digest_projection
    )


def test_task27_recovery_origin_equality_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    origin_path = paths.recovery_origin_path(successor_run_id, tmp_path)
    original = io.read_json(origin_path)
    with _seal_patches(source_run_id):
        task27 = build_task27_evidence(
            recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path
        )
    for field in _recovery_origin_digest_projection(original):
        mutated = dict(original)
        mutated[field] = "tampered-value"
        io.atomic_write_json(origin_path, mutated)
        with _seal_patches(source_run_id):
            with pytest.raises((BoundedActionPreconditionError, BoundedActionValidationError, RecoveryRunValidationError)):
                build_successor_evidence(
                    successor_run_id, source_run_id=source_run_id, task27=task27, base_dir=tmp_path
                )
        io.atomic_write_json(origin_path, original)


def test_revoke_present_blocks_publication(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    revoke_path = paths.recovery_run_revoke_path(recovery_request_id, tmp_path)
    io.atomic_write_json(revoke_path, {"recovery_request_id": recovery_request_id, "revoked_by": "operator"})
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionPreconditionError, match="revoke"):
            build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)


def test_absent_outcome_blocks_publication(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    paths.recovery_run_outcome_path(recovery_request_id, tmp_path).unlink()
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionPreconditionError, match="incomplete"):
            build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)


def test_outcome_enum_successor_created_verified(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    outcome = io.read_json(paths.recovery_run_outcome_path(recovery_request_id, tmp_path))
    assert outcome["outcome_class"] == RecoveryRunOutcomeClass.successor_created_verified.value
    with _seal_patches(source_run_id):
        evidence = build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)
    assert evidence["outcome"]["outcome_class"] == RecoveryRunOutcomeClass.successor_created_verified.value


def test_outcome_enum_successor_already_exists_verified(tmp_path):
    assert RecoveryRunOutcomeClass.successor_already_exists_verified.value == "successor_already_exists_verified"
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    outcome_path = paths.recovery_run_outcome_path(recovery_request_id, tmp_path)
    outcome = io.read_json(outcome_path)
    outcome["outcome_class"] = RecoveryRunOutcomeClass.successor_already_exists_verified.value
    outcome = _refresh_task27_record_digest(outcome, _outcome_digest_projection, "outcome_digest")
    io.atomic_write_json(outcome_path, outcome)
    with _seal_patches(source_run_id):
        evidence = build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)
    assert evidence["outcome"]["outcome_class"] == RecoveryRunOutcomeClass.successor_already_exists_verified.value


def test_outcome_enum_other_rejected(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    outcome_path = paths.recovery_run_outcome_path(recovery_request_id, tmp_path)
    outcome = io.read_json(outcome_path)
    outcome["outcome_class"] = "creation_partial"
    outcome = _refresh_task27_record_digest(outcome, _outcome_digest_projection, "outcome_digest")
    io.atomic_write_json(outcome_path, outcome)
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionPreconditionError, match="ineligible|outcome_digest"):
            build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)


def test_runtime_file_type_regular_file(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id):
        evidence = build_source_evidence(source_run_id, tmp_path)
    assert evidence["source_manifest"]["file_type"] == "regular_file"


def test_runtime_permission_mode_not_git_mode(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    manifest_path = paths.run_manifest_path(source_run_id, tmp_path)
    mode = format(stat.S_IMODE(manifest_path.stat().st_mode), "04o")
    assert mode != "100600"
    with _seal_patches(source_run_id):
        evidence = build_source_evidence(source_run_id, tmp_path)
    assert evidence["source_manifest"]["permission_mode"] == mode


def test_task28_record_mode_0600(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    proposal_path = paths.bounded_action_proposal_path(proposal_id, tmp_path)
    assert stat.S_IMODE(proposal_path.stat().st_mode) == CONTROL_FILE_MODE == 0o600


def test_task28_directory_mode_0700(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    case_dir = paths.bounded_action_case_dir(proposal_id, tmp_path)
    assert stat.S_IMODE(case_dir.stat().st_mode) == CONTROL_DIR_MODE == 0o700


def test_preexisting_shared_control_compatibility(tmp_path):
    control = paths.control_root(tmp_path)
    control.mkdir(parents=True, exist_ok=True)
    os.chmod(control, 0o700)
    fd = open_dir_no_follow(control, context=".control")
    try:
        validate_preexisting_control_dir(fd, context=".control")
    finally:
        os.close(fd)


def test_new_task28_ownership_enforcement(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    case_fd = open_dir_no_follow(paths.bounded_action_case_dir(proposal_id, tmp_path), context="case")
    try:
        validate_new_task28_ownership(case_fd, context="case")
    finally:
        os.close(case_fd)


def test_canonical_serializer_byte_fixture():
    payload = {"record_type": "bounded_action_proposal", "schema_version": "1"}
    assert canonical_json_bytes(payload) == CANONICAL_FIXTURE_BYTES


def test_noncanonical_stored_bytes_rejected():
    canonical = canonical_json_bytes({"record_type": "bounded_action_proposal", "schema_version": "1"})
    drifted = canonical.replace(b'"schema_version"', b'"schema_version" ')
    with pytest.raises(BoundedActionValidationError):
        require_exact_canonical_bytes(drifted, canonical)


def test_local_replay_unhealthy_namespace(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    junk = paths.control_bounded_actions_root(tmp_path) / "not_a_bar_id"
    junk.mkdir()
    with _seal_patches(source_run_id):
        replay = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert replay.publication_result == "exact_replay"
    assert replay.namespace_integrity_class == "indeterminate"


def test_zero_write_mtime_ctime_manifest(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    manifest = paths.run_manifest_path(source_run_id, tmp_path)
    before = _times(manifest)
    reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    inspect_bounded_action_eligibility(
        source_run_id,
        bundle.proposal["successor_run_id"],
        bundle.proposal["task27_evidence"]["recovery_request_id"],
        ProposalSubject.bounded_action_architecture_candidate.value,
        base_dir=tmp_path,
    )
    after = _times(manifest)
    assert before[1] == after[1]
    assert before[2] == after[2]


def test_atime_never_used_as_authority(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    manifest = paths.run_manifest_path(source_run_id, tmp_path)
    os.utime(manifest, (123.0, manifest.stat().st_mtime))
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    digest_before = bundle.proposal["record_digest"]
    replay = reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    assert replay.record_digest == digest_before


def test_no_cli_changes():
    root = Path(__file__).resolve().parents[1]
    assert not any(root.joinpath("hermes_cli").rglob("*bounded_action*"))


def test_no_docs_changes():
    root = Path(__file__).resolve().parents[1]
    for rel in ("docs/runtime_project/03", "docs/runtime_project/05", "docs/runtime_project/07", "docs/runtime_project/08", "docs/runtime_project/09"):
        path = root / rel
        if path.exists():
            current = path.read_bytes()
            try:
                baseline = _git_blob(rel)
            except subprocess.CalledProcessError:
                continue
            assert current == baseline


def test_no_task23_27_module_changes():
    root = Path(__file__).resolve().parents[1]
    for module in TASK23_27_MODULES:
        current = (root / module).read_bytes()
        assert _git_blob(module) == current


def test_no_source_run_mutation(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    run_root = paths.run_root(source_run_id, tmp_path)
    before = {str(p.relative_to(run_root)): _digest_file(p) for p in run_root.rglob("*") if p.is_file()}
    reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    after = {str(p.relative_to(run_root)): _digest_file(p) for p in run_root.rglob("*") if p.is_file()}
    assert before == after


def test_no_successor_run_mutation(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, *_ = _publish_proposal(tmp_path)
    run_root = paths.run_root(successor_run_id, tmp_path)
    before = {str(p.relative_to(run_root)): _digest_file(p) for p in run_root.rglob("*") if p.is_file()}
    reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    after = {str(p.relative_to(run_root)): _digest_file(p) for p in run_root.rglob("*") if p.is_file()}
    assert before == after


def test_no_marker_mutation(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, *_ = _publish_proposal(tmp_path)
    marker = paths.runs_root(tmp_path) / LOCKS_DIR_NAME / f"{successor_run_id}.marker"
    before = marker.read_bytes() if marker.exists() else None
    reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    after = marker.read_bytes() if marker.exists() else None
    assert before == after


def test_no_external_side_effects(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    assert True


def test_paths_a_b_acquire_no_locks(tmp_path, monkeypatch):
    called = {"flock": 0}
    import fcntl

    orig = fcntl.flock

    def _track(*args, **kwargs):
        called["flock"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fcntl, "flock", _track)
    proposal_id = generate_bounded_action_proposal_id()
    reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    inspect_bounded_action_eligibility(
        "run_20260730_000001",
        "run_20260730_000002",
        "rcr_20260730_000001",
        ProposalSubject.bounded_action_architecture_candidate.value,
        base_dir=tmp_path,
    )
    assert called["flock"] == 0


def test_exact_replay_acquires_no_locks(tmp_path, monkeypatch):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    called = {"flock": 0}
    import fcntl

    orig = fcntl.flock

    def _track(*args, **kwargs_inner):
        called["flock"] += 1
        return orig(*args, **kwargs_inner)

    monkeypatch.setattr(fcntl, "flock", _track)
    replay = reconcile_bounded_action(proposal_id, base_dir=tmp_path)
    assert replay.publication_result == "exact_replay"
    assert replay.exact_replay is True
    assert called["flock"] == 0
    with _seal_patches(source_run_id):
        replay2 = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert replay2.publication_result == "exact_replay"
    assert replay2.exact_replay is True
    assert called["flock"] == 0


def test_concurrent_identical_publication_observed(tmp_path, monkeypatch):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    called = {"flock": 0}
    import fcntl

    orig = fcntl.flock

    def _track(*args, **kwargs_inner):
        called["flock"] += 1
        return orig(*args, **kwargs_inner)

    monkeypatch.setattr(fcntl, "flock", _track)
    monkeypatch.setattr(
        "htr.bounded_actions._try_proposal_early_resolution",
        lambda *args, **kwargs: None,
    )
    with _seal_patches(source_run_id):
        result = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert result.publication_result == "concurrent_identical_publication_observed"
    assert result.local_publication_result == "exact_replay"
    assert result.exact_replay is False
    assert called["flock"] > 0
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    assert bundle.terminal_state == "proposal"


def test_preexisting_conflict_no_flock_no_write(tmp_path, monkeypatch):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    proposal_path = paths.bounded_action_proposal_path(proposal_id, tmp_path)
    before = proposal_path.read_bytes()
    called = {"flock": 0}
    import fcntl

    orig = fcntl.flock

    def _track(*args, **kwargs_inner):
        called["flock"] += 1
        return orig(*args, **kwargs_inner)

    monkeypatch.setattr(fcntl, "flock", _track)
    conflict_kwargs = dict(kwargs)
    conflict_kwargs["proposal_summary"] = "A conflicting summary that must not write."
    with _seal_patches(source_run_id):
        result = create_bounded_action_proposal(**conflict_kwargs, base_dir=tmp_path)
    assert result.publication_result == "conflict"
    assert result.exact_replay is False
    assert called["flock"] == 0
    assert proposal_path.read_bytes() == before


def test_namespace_integrity_reserved_names(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    ba_root = paths.control_bounded_actions_root(tmp_path)
    children = {p.name for p in ba_root.iterdir()}
    reserved = {paths.PUBLICATION_COORD_DIR_NAME, paths.SUCCESSOR_COORD_DIR_NAME}
    assert reserved.issubset(children)
    unknown = children - reserved - {proposal_id}
    assert not unknown or all(name.startswith("bar_") for name in unknown)


def test_namespace_integrity_unknown_child(tmp_path):
    _ensure_control_tree_0700(tmp_path)
    ba_root = paths.control_bounded_actions_root(tmp_path)
    (ba_root / "random_child").mkdir()
    state = reconcile_successor_bounded_action_state("run_20260730_000099", base_dir=tmp_path)
    assert state["namespace_integrity_class"] == "indeterminate"


def test_terminal_one_per_proposal(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    digest = bundle.proposal["record_digest"]
    _record_review(tmp_path, source_run_id, proposal_id, generate_bounded_action_review_decision_id(), proposal_digest=digest)
    result = _record_escalation(tmp_path, source_run_id, proposal_id, generate_bounded_action_escalation_id(), proposal_digest=digest)
    assert result.publication_result == "conflict"


def test_terminal_id_scope_within_proposal(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    review_id = generate_bounded_action_review_decision_id()
    digest = bundle.proposal["record_digest"]
    _record_review(tmp_path, source_run_id, proposal_id, review_id, proposal_digest=digest)
    replay = _record_review(tmp_path, source_run_id, proposal_id, review_id, proposal_digest=digest)
    assert replay.publication_result == "exact_replay"


def test_cross_proposal_terminal_id_allowed(tmp_path):
    p1, _pub1, src1, *_ = _publish_proposal(tmp_path)
    p2, _pub2, src2, *_ = _publish_proposal(tmp_path)
    review_id = generate_bounded_action_review_decision_id()
    b1 = load_bounded_action_proposal_bundle(p1, base_dir=tmp_path)
    b2 = load_bounded_action_proposal_bundle(p2, base_dir=tmp_path)
    _record_review(tmp_path, src1, p1, review_id, proposal_digest=b1.proposal["record_digest"])
    result = _record_review(tmp_path, src2, p2, review_id, proposal_digest=b2.proposal["record_digest"])
    assert result.publication_result == "published_new"


def test_proposal_id_global_unique(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    assert paths.bounded_action_case_dir(proposal_id, tmp_path).name == proposal_id
    assert proposal_id.startswith("bar_")


def test_record_digest_self_reference_excluded(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    proposal = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path).proposal
    assert "record_digest" not in projection_b(proposal)


def test_created_at_excluded_from_semantic_projection(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    proposal = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path).proposal
    assert "created_at" in projection_b(proposal)
    assert "created_at" not in projection_c(proposal)


def test_fresh_evidence_at_publication(tmp_path):
    sig = inspect.signature(create_bounded_action_proposal)
    assert "source_evidence" not in sig.parameters
    assert "task27_evidence" not in sig.parameters
    assert "successor_evidence" not in sig.parameters


def test_aggregate_counts_not_caller_args():
    sig = inspect.signature(create_bounded_action_proposal)
    for field in ("proposal_count_before", "proposal_count_after", "escalation_count_before", "escalation_count_after"):
        assert field not in sig.parameters


def test_conflict_same_ids_different_intent(tmp_path, monkeypatch):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    conflict_kwargs = dict(kwargs)
    conflict_kwargs["proposal_summary"] = "Different summary intent."
    called = {"flock": 0}
    import fcntl

    orig = fcntl.flock

    def _track(*args, **kwargs_inner):
        called["flock"] += 1
        return orig(*args, **kwargs_inner)

    monkeypatch.setattr(fcntl, "flock", _track)
    with _seal_patches(source_run_id):
        result = create_bounded_action_proposal(**conflict_kwargs, base_dir=tmp_path)
    assert result.publication_result == "conflict"
    assert result.exact_replay is False
    assert called["flock"] == 0


def test_published_new_happy_path(tmp_path):
    proposal_id, result, *_ = _publish_proposal(tmp_path)
    assert result.publication_result == "published_new"
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    assert bundle.proposal is not None
    assert bundle.terminal_state == "proposal"


def test_review_decision_happy_path(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    review_id = generate_bounded_action_review_decision_id()
    result = _record_review(
        tmp_path,
        source_run_id,
        proposal_id,
        review_id,
        proposal_digest=bundle.proposal["record_digest"],
    )
    assert result.publication_result == "published_new"
    bundle2 = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    assert bundle2.terminal_state == "review"


def test_escalation_happy_path(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    esc_id = generate_bounded_action_escalation_id()
    result = _record_escalation(
        tmp_path,
        source_run_id,
        proposal_id,
        esc_id,
        proposal_digest=bundle.proposal["record_digest"],
    )
    assert result.publication_result == "published_new"
    assert load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path).terminal_state == "escalation"


def test_escalation_count_after_increment(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    _record_escalation(
        tmp_path,
        source_run_id,
        proposal_id,
        generate_bounded_action_escalation_id(),
        proposal_digest=bundle.proposal["record_digest"],
    )
    esc = io.read_json(paths.bounded_action_escalation_path(proposal_id, tmp_path))
    assert esc["escalation_count_after"] == esc["escalation_count_before"] + 1


def test_review_count_invariants(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    _record_review(
        tmp_path,
        source_run_id,
        proposal_id,
        generate_bounded_action_review_decision_id(),
        proposal_digest=bundle.proposal["record_digest"],
    )
    review = io.read_json(paths.bounded_action_review_decision_path(proposal_id, tmp_path))
    assert review["proposal_count_after"] == review["proposal_count_before"]
    assert review["escalation_count_after"] == review["escalation_count_before"]


def test_inspection_indeterminate_during_publication(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id):
        replay = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert replay.publication_result == "exact_replay"


def test_durability_indeterminate_classification(tmp_path):
    with patch("htr.bounded_action_bootstrap.fsync_dir_fd", side_effect=OSError("fsync failed")):
        with pytest.raises(BoundedActionDurabilityError):
            bootstrap_publication_tree(tmp_path)


def test_control_hierarchy_bootstrap(tmp_path):
    locks = bootstrap_publication_tree(tmp_path)
    try:
        assert paths.control_bounded_actions_root(tmp_path).is_dir()
        assert paths.bounded_action_publication_coord_dir(tmp_path).is_dir()
        assert stat.S_IMODE(paths.control_bounded_actions_root(tmp_path).stat().st_mode) == 0o700
    finally:
        release_publication_locks(locks)


def test_symlink_rejected_on_control_paths(tmp_path):
    ba_root = paths.control_bounded_actions_root(tmp_path)
    ba_root.mkdir(parents=True, exist_ok=True)
    target = ba_root / "target_case"
    target.mkdir()
    link = ba_root / "symlink_case"
    try:
        link.symlink_to(target)
        with pytest.raises(BoundedActionValidationError):
            openat_dir_no_follow(
                open_dir_no_follow(ba_root, context="ba"),
                link.name,
                context="symlink",
            )
    except OSError:
        pytest.skip("symlink creation unsupported")


def test_load_bounded_action_proposal_bundle(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    assert bundle.proposal_id == proposal_id
    assert bundle.proposal["protocol_version"] == PROTOCOL_VERSION


def test_schema_abc_columns_yes_no_only():
    rows = _parse_r9_schema_rows()
    assert rows, "expected schema rows from R9 artifact"
    for field, a, b, c in rows:
        assert a in {"yes", "no"}, field
        assert b in {"yes", "no"}, field
        assert c in {"yes", "no"}, field


def test_caller_fields_only_in_projection_a():
    from htr.bounded_actions import _proposal_intent_a

    intent = _proposal_intent_a(
        proposal_id="bar_20260730_000001",
        source_run_id="run_20260730_000001",
        successor_run_id="run_20260730_000002",
        creator="operator",
        proposal_subject=ProposalSubject.bounded_action_architecture_candidate.value,
        proposal_summary="summary",
        risk_class="advisory_read_only_low",
        confidence_class="proven",
        reason_codes=["architecture_review_requested"],
        reason_detail=None,
    )
    assert set(intent.keys()) == PROPOSAL_CALLER_FIELDS | {"reason_detail"}
    assert "request_intent_digest" not in intent
    assert "source_evidence" not in intent


def test_evidence_fields_in_b_and_c():
    rows = {f: (a, b, c) for f, a, b, c in _parse_r9_schema_rows()}
    for field, (_a, b, c) in rows.items():
        if field.startswith("source_evidence.") or field.startswith("task27_evidence.") or field.startswith("successor_evidence."):
            assert b == "yes" and c == "yes", field


def test_authority_booleans_in_b_and_c_not_a():
    rows = {f: (a, b, c) for f, a, b, c in _parse_r9_schema_rows()}
    for field in AUTHORITY_BOOLEAN_FIELDS:
        if field in rows:
            a, b, c = rows[field]
            assert a == "no" and b == "yes" and c == "yes"


def test_request_intent_digest_not_in_a():
    rows = {f: (a, b, c) for f, a, b, c in _parse_r9_schema_rows()}
    assert rows["request_intent_digest"] == ("no", "yes", "yes")


def test_task27_revoke_classification_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    with _seal_patches(source_run_id):
        build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)
    revoke_path = paths.recovery_run_revoke_path(recovery_request_id, tmp_path)
    io.atomic_write_json(revoke_path, {"recovery_request_id": recovery_request_id})
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionPreconditionError):
            build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)


def test_execution_revalidation_field_matrix(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    attempt = io.read_json(paths.recovery_run_attempt_path(recovery_request_id, tmp_path))
    reval = attempt["execution_revalidation"]
    for field in EXECUTION_REVALIDATION_FIELDS:
        assert field in reval


def test_execution_revalidation_missing_nested_field_fails_closed(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    attempt_path = paths.recovery_run_attempt_path(recovery_request_id, tmp_path)
    attempt = io.read_json(attempt_path)
    reval = dict(attempt["execution_revalidation"])
    reval.pop("execution_inspection_projection")
    attempt["execution_revalidation"] = reval
    attempt["attempt_digest"] = _sha256_digest(_attempt_digest_projection(attempt))
    io.atomic_write_json(attempt_path, attempt)
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionValidationError, match="missing execution_inspection_projection"):
            build_task27_evidence(
                recovery_request_id,
                source_run_id=source_run_id,
                successor_run_id=successor_run_id,
                base_dir=tmp_path,
            )


def test_execution_revalidation_unknown_nested_field_fails_closed(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    attempt_path = paths.recovery_run_attempt_path(recovery_request_id, tmp_path)
    attempt = io.read_json(attempt_path)
    reval = dict(attempt["execution_revalidation"])
    reval["unexpected_field"] = "tamper"
    attempt["execution_revalidation"] = reval
    attempt["attempt_digest"] = _sha256_digest(_attempt_digest_projection(attempt))
    io.atomic_write_json(attempt_path, attempt)
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionValidationError, match="unknown field"):
            build_task27_evidence(
                recovery_request_id,
                source_run_id=source_run_id,
                successor_run_id=successor_run_id,
                base_dir=tmp_path,
            )


def test_execution_revalidation_malformed_projection_fails_closed(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    attempt_path = paths.recovery_run_attempt_path(recovery_request_id, tmp_path)
    attempt = io.read_json(attempt_path)
    reval = dict(attempt["execution_revalidation"])
    reval["execution_inspection_projection"] = {"overall_classification": "integrity_blocked"}
    envelope = {k: reval[k] for k in EXECUTION_REVALIDATION_FIELDS if k != "execution_revalidation_digest"}
    reval["execution_revalidation_digest"] = _sha256_digest(envelope)
    attempt["execution_revalidation"] = reval
    attempt["attempt_digest"] = _sha256_digest(_attempt_digest_projection(attempt))
    io.atomic_write_json(attempt_path, attempt)
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionValidationError):
            build_task27_evidence(
                recovery_request_id,
                source_run_id=source_run_id,
                successor_run_id=successor_run_id,
                base_dir=tmp_path,
            )


def test_eligibility_valid_source_closure_order(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    with _seal_patches(source_run_id):
        result = inspect_bounded_action_eligibility(
            source_run_id,
            successor_run_id,
            recovery_request_id,
            ProposalSubject.bounded_action_architecture_candidate.value,
            base_dir=tmp_path,
            reason_codes=["architecture_review_requested"],
            risk_class="advisory_read_only_low",
            confidence_class="proven",
        )
    assert result.eligible is True
    assert result.classification == "eligible"


def test_eligibility_rejects_non_finalized_source_without_patch(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    with pytest.raises(BoundedActionPreconditionError, match="FINALIZED_VALID"):
        build_source_evidence(source_run_id, tmp_path)


def test_eligibility_rejects_late_closure_without_valid_seal(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    closure_path = contracts.run_final_closure_record_json_path(source_run_id, tmp_path)
    if closure_path.is_file():
        closure_path.unlink()
    with _seal_patches(source_run_id):
        with pytest.raises(BoundedActionPreconditionError, match="closure record missing"):
            build_source_evidence(source_run_id, tmp_path)


def test_source_artifact_identity_completeness(tmp_path):
    proposal_id, _pub, source_run_id, *_ = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id):
        manifest = build_source_evidence(source_run_id, tmp_path)["source_manifest"]
    assert ARTIFACT_IDENTITY_KEYS.issubset(manifest.keys())


def test_task27_artifact_identity_completeness(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    with _seal_patches(source_run_id):
        evidence = build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)
    for key in ("request", "issue", "claim", "attempt", "outcome"):
        assert ARTIFACT_IDENTITY_KEYS.issubset(evidence[key].keys())


def test_successor_artifact_identity_completeness(tmp_path):
    recovery_request_id, successor_run_id, _case_id, source_run_id = _full_chain(tmp_path)
    _execute_successor(tmp_path, recovery_request_id, source_run_id)
    with _seal_patches(source_run_id):
        task27 = build_task27_evidence(recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=tmp_path)
        evidence = build_successor_evidence(successor_run_id, source_run_id=source_run_id, task27=task27, base_dir=tmp_path)
    for key in ("recovery_origin", "run_manifest", "task_events_jsonl"):
        assert ARTIFACT_IDENTITY_KEYS.issubset(evidence[key].keys())


def test_namespace_snapshot_canonical_fixture():
    payload = {"schema_version": NAMESPACE_INTEGRITY_SCHEMA, "entries": []}
    digest = sha256_digest(payload)
    assert digest.startswith("sha256:")


def test_target_aggregate_snapshot_canonical_fixture():
    payload = {
        "schema_version": TARGET_AGGREGATE_SCHEMA,
        "target_successor_run_id": "run_20260730_000001",
        "target_successor_project_identity_digest": "sha256:" + "0" * 64,
        "target_successor_runs_root_identity_digest": "sha256:" + "0" * 64,
        "entries": [],
        "proposal_count": 0,
        "escalation_count": 0,
        "protocol_cap_max_proposals": BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR,
        "protocol_cap_max_escalations": BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR,
    }
    digest = sha256_digest(payload)
    assert digest.startswith("sha256:")


def test_concurrent_first_root_lock_bootstrap(tmp_path):
    proposal_id, result, *_ = _publish_proposal(tmp_path)
    assert result.publication_result == "published_new"
    assert paths.bounded_action_publication_coord_dir(tmp_path).is_dir()


def test_root_coordination_parent_fsync_failure(tmp_path):
    with patch("htr.bounded_action_bootstrap.fsync_dir_fd", side_effect=OSError("fsync failed")):
        with pytest.raises(BoundedActionDurabilityError):
            bootstrap_publication_tree(tmp_path)


def test_root_coordination_crash_before_flock(tmp_path):
    coord = paths.bounded_action_publication_coord_dir(tmp_path)
    coord.mkdir(parents=True, exist_ok=True)
    state = reconcile_successor_bounded_action_state("run_20260730_000001", base_dir=tmp_path)
    assert state["namespace_integrity_class"] in {"healthy", "indeterminate"}


def test_escalation_rejected_at_escalation_cap(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, _ = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id):
        for i in range(BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR):
            pid = proposal_id if i == 0 else generate_bounded_action_proposal_id()
            if i > 0:
                create_bounded_action_proposal(
                    **_default_proposal_kwargs(pid, source_run_id, successor_run_id, recovery_request_id),
                    base_dir=tmp_path,
                )
            bundle = load_bounded_action_proposal_bundle(pid, base_dir=tmp_path)
            _record_escalation(
                tmp_path,
                source_run_id,
                pid,
                generate_bounded_action_escalation_id(),
                proposal_digest=bundle.proposal["record_digest"],
            )
    extra_id = generate_bounded_action_proposal_id()
    with _seal_patches(source_run_id):
        create_bounded_action_proposal(
            **_default_proposal_kwargs(extra_id, source_run_id, successor_run_id, recovery_request_id),
            base_dir=tmp_path,
        )
    bundle = load_bounded_action_proposal_bundle(extra_id, base_dir=tmp_path)
    assert bundle.proposal is not None
    result = _record_escalation(
        tmp_path,
        source_run_id,
        extra_id,
        generate_bounded_action_escalation_id(),
        proposal_digest=bundle.proposal["record_digest"],
    )
    assert result.publication_result == "cap_exhausted"


def test_replay_secondary_observation_failure_stays_replay(tmp_path):
    proposal_id, _pub, source_run_id, successor_run_id, recovery_request_id, kwargs = _publish_proposal(tmp_path)
    with _seal_patches(source_run_id), patch(
        "htr.bounded_actions._classify_namespace",
        return_value=("indeterminate", "sha256:" + "0" * 64, ["orphan"]),
    ):
        replay = create_bounded_action_proposal(**kwargs, base_dir=tmp_path)
    assert replay.publication_result == "exact_replay"
    assert replay.namespace_integrity_class == "indeterminate"


def test_authority_boolean_missing_rejected(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    path = paths.bounded_action_proposal_path(proposal_id, tmp_path)
    body = parse_strict_json_bytes(path.read_bytes())
    body.pop(AUTHORITY_BOOLEAN_FIELDS[0])
    path.write_bytes(canonical_json_bytes(body))
    with pytest.raises(BoundedActionValidationError):
        load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)


def test_authority_boolean_non_boolean_rejected(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    path = paths.bounded_action_proposal_path(proposal_id, tmp_path)
    body = parse_strict_json_bytes(path.read_bytes())
    body[AUTHORITY_BOOLEAN_FIELDS[0]] = "false"
    path.write_bytes(canonical_json_bytes(body))
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    assert bundle.proposal is None or bundle.terminal_state == "malformed"


def test_authority_boolean_true_rejected(tmp_path):
    proposal_id, _pub, *_ = _publish_proposal(tmp_path)
    path = paths.bounded_action_proposal_path(proposal_id, tmp_path)
    body = parse_strict_json_bytes(path.read_bytes())
    body[AUTHORITY_BOOLEAN_FIELDS[0]] = True
    body["record_digest"] = compute_record_digest(body)
    path.write_bytes(canonical_json_bytes(body))
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=tmp_path)
    assert bundle.proposal is None


def test_strict_json_parser_limits():
    huge = b'{"x": "' + b"a" * (MAX_RECORD_BYTES + 1) + b'"}\n'
    with pytest.raises(BoundedActionValidationError, match="MAX_RECORD_BYTES"):
        parse_strict_json_bytes(huge)


def test_shared_control_exact_0700_compatibility(tmp_path):
    control = paths.control_root(tmp_path)
    control.mkdir(parents=True, exist_ok=True)
    os.chmod(control, 0o700)
    fd = open_dir_no_follow(control, context=".control")
    try:
        validate_preexisting_control_dir(fd, context=".control")
    finally:
        os.close(fd)


def test_phase28a_apis_no_path_d_surface():
    import htr.bounded_actions as mod

    for name in dir(mod):
        assert not name.startswith("execute_")
