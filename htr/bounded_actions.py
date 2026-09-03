"""Task 28 Phase 28A — bounded action advisory publication (Path A/B/C)."""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from htr import paths
from htr.approval_control import _project_dir_path_digest, _runs_root_path_digest
from htr.bounded_action_bootstrap import (
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
    fsync_dir_fd,
    fsync_file_fd,
    fstat_identity,
    is_directory_mode,
    list_dir_names_sorted,
    open_dir_no_follow,
    openat_dir_no_follow,
    openat_file_no_follow,
    read_json_record_fd,
    stat_entry_identity,
    stat_entry_mode,
    validate_dir_mode_0700,
    validate_file_mode_0600_link_count,
    validate_preexisting_control_dir,
    write_all_fd,
)
from htr.bounded_action_digest import (
    canonical_json_bytes,
    compute_record_digest,
    projection_b,
    sha256_digest,
    validate_record_digest,
)
from htr.bounded_action_evidence import (
    build_source_evidence,
    build_successor_evidence,
    build_task27_evidence,
    classify_marker_for_subject,
    evidence_drift,
)
from htr.bounded_action_schemas import (
    AUTHORITY_BOOLEAN_FIELDS,
    BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR,
    BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR,
    NAMESPACE_INTEGRITY_SCHEMA,
    PROTOCOL_VERSION,
    RECORD_TYPE_ESCALATION,
    RECORD_TYPE_PROPOSAL,
    RECORD_TYPE_REVIEW,
    SCHEMA_VERSION,
    SUBJECT_MATRIX,
    TARGET_AGGREGATE_SCHEMA,
    authority_booleans_false,
)
from htr.bounded_action_strict_json import (
    MAX_PROPOSAL_SUMMARY_CODEPOINTS,
    MAX_REASON_CODES,
    MAX_REASON_DETAIL_CODEPOINTS,
    parse_strict_json_bytes,
)
from htr.ids import require_id, validate_id
from htr.state import (
    BoundedActionConflictError,
    BoundedActionDurabilityError,
    BoundedActionPreconditionError,
    BoundedActionValidationError,
)

_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

_BAR_RE = re.compile(r"^bar_\d{8}_[a-f0-9]{6}$")
_RESERVED = frozenset({paths.PUBLICATION_COORD_DIR_NAME, paths.SUCCESSOR_COORD_DIR_NAME})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_actor(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BoundedActionValidationError(f"{field} must be non-empty")
    actor = value.strip()
    if len(actor) > 256:
        raise BoundedActionValidationError(f"{field} too long")
    return actor


def _validate_summary(value: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise BoundedActionValidationError("proposal_summary must be NFC")
    if len(value) > MAX_PROPOSAL_SUMMARY_CODEPOINTS:
        raise BoundedActionValidationError("proposal_summary too long")
    return value


def _validate_reason_codes(codes: list[str], *, allowed: frozenset[str]) -> list[str]:
    if not isinstance(codes, list) or len(codes) > MAX_REASON_CODES:
        raise BoundedActionValidationError("invalid reason_codes")
    normalized = sorted(set(codes))
    for code in normalized:
        if code not in allowed:
            raise BoundedActionValidationError(f"reason code not allowed: {code}")
    return normalized


def _proposal_intent_a(**kwargs: Any) -> dict[str, Any]:
    return {
        "proposal_id": kwargs["proposal_id"],
        "source_run_id": kwargs["source_run_id"],
        "successor_run_id": kwargs["successor_run_id"],
        "creator": kwargs["creator"],
        "proposal_subject": kwargs["proposal_subject"],
        "proposal_summary": kwargs["proposal_summary"],
        "risk_class": kwargs["risk_class"],
        "confidence_class": kwargs["confidence_class"],
        "reason_codes": kwargs["reason_codes"],
        "reason_detail": kwargs.get("reason_detail"),
    }


def _review_intent_a(**kwargs: Any) -> dict[str, Any]:
    return {
        "proposal_id": kwargs["proposal_id"],
        "review_decision_id": kwargs["review_decision_id"],
        "expected_proposal_record_digest": kwargs["expected_proposal_record_digest"],
        "reviewer": kwargs["reviewer"],
        "decision_class": kwargs["decision_class"],
        "risk_class": kwargs["risk_class"],
        "confidence_class": kwargs["confidence_class"],
        "reason_codes": kwargs["reason_codes"],
        "reason_detail": kwargs.get("reason_detail"),
    }


def _escalation_intent_a(**kwargs: Any) -> dict[str, Any]:
    return {
        "proposal_id": kwargs["proposal_id"],
        "escalation_id": kwargs["escalation_id"],
        "expected_proposal_record_digest": kwargs["expected_proposal_record_digest"],
        "escalator": kwargs["escalator"],
        "escalation_class": kwargs["escalation_class"],
        "risk_class": kwargs["risk_class"],
        "confidence_class": kwargs["confidence_class"],
        "reason_codes": kwargs["reason_codes"],
        "reason_detail": kwargs.get("reason_detail"),
    }


def _validate_authority_booleans(record: dict[str, Any]) -> None:
    for field in AUTHORITY_BOOLEAN_FIELDS:
        if field not in record:
            raise BoundedActionValidationError(f"missing required field: {field}")
        if record[field] is not False:
            raise BoundedActionValidationError(f"{field} must be literal false")


@dataclass(frozen=True)
class BoundedActionPublicationResult:
    publication_result: str
    local_publication_result: str
    request_intent_digest_match: bool
    current_evidence_class: str
    namespace_integrity_class: str
    target_aggregate_class: str
    new_publication_allowed: bool
    record_digest: str | None = None
    exact_replay: bool = False


@dataclass(frozen=True)
class BoundedActionEligibilityResult:
    eligible: bool
    classification: str
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundedActionProposalBundle:
    proposal_id: str
    proposal: dict[str, Any] | None
    review_decision: dict[str, Any] | None
    escalation: dict[str, Any] | None
    terminal_state: Literal["absent", "proposal", "review", "escalation", "conflicting", "malformed"]


def _load_record_if_present(case_fd: int, name: str) -> tuple[dict[str, Any] | None, bytes | None]:
    try:
        fd = openat_file_no_follow(
            case_fd, name, _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC, context=name
        )
    except BoundedActionValidationError:
        return None, None
    try:
        validate_file_mode_0600_link_count(fd, context=name)
        record, raw = read_json_record_fd(fd)
        validate_record_digest(record)
        _validate_authority_booleans(record)
        return record, raw
    except BoundedActionValidationError:
        return None, None
    finally:
        os.close(fd)


def _try_open_bounded_actions_fd(base_dir: Path | None) -> int | None:
    runs_root = paths.runs_root(base_dir)
    if not runs_root.is_dir():
        return None
    runs_fd = open_dir_no_follow(runs_root, context="runs_root")
    try:
        try:
            control_fd = openat_dir_no_follow(runs_fd, ".control", context=".control")
        except BoundedActionValidationError:
            return None
        try:
            control_st = os.fstat(control_fd)
            if not stat.S_ISDIR(control_st.st_mode):
                return None
            if stat.S_IMODE(control_st.st_mode) != CONTROL_DIR_MODE:
                return None
            try:
                return openat_dir_no_follow(
                    control_fd,
                    paths.BOUNDED_ACTIONS_DIR_NAME,
                    context="bounded_actions",
                )
            except BoundedActionValidationError:
                return None
        finally:
            os.close(control_fd)
    finally:
        os.close(runs_fd)


def _inspect_namespace_child(ba_fd: int, name: str) -> tuple[dict[str, Any], bool]:
    if name in _RESERVED:
        if name == paths.PUBLICATION_COORD_DIR_NAME:
            classification = "reserved_publication_coord"
        else:
            classification = "reserved_successor_coord"
        return {"child_name": name, "classification": classification}, False
    if not _BAR_RE.match(name):
        return {"child_name": name, "classification": "unknown"}, True
    try:
        pre_identity = stat_entry_identity(ba_fd, name)
        pre_mode = stat_entry_mode(ba_fd, name)
    except OSError:
        return {"child_name": name, "classification": "unsafe"}, True
    if not is_directory_mode(pre_mode):
        return {"child_name": name, "classification": "wrong_type"}, True
    if stat.S_IMODE(pre_mode) != CONTROL_DIR_MODE:
        return {"child_name": name, "classification": "unsafe"}, True
    case_fd = openat_dir_no_follow(ba_fd, name, context=name)
    try:
        if fstat_identity(case_fd) != pre_identity:
            return {"child_name": name, "classification": "unsafe"}, True
        validate_dir_mode_0700(case_fd, context=name, require_ownership=True)
        child_names = list_dir_names_sorted(case_fd)
        if not child_names:
            return {"child_name": name, "classification": "malformed"}, True
        proposal_record, _ = _load_record_if_present(case_fd, "proposal.json")
        if proposal_record is None:
            return {"child_name": name, "classification": "malformed"}, True
        review_present = "review_decision.json" in child_names
        escalation_present = "escalation.json" in child_names
        if review_present and escalation_present:
            terminal = "conflicting"
        elif review_present:
            terminal = "review"
        elif escalation_present:
            terminal = "escalation"
        else:
            terminal = "none"
        if fstat_identity(case_fd) != pre_identity:
            return {"child_name": name, "classification": "unsafe"}, True
        entry: dict[str, Any] = {
            "child_name": name,
            "classification": "valid_case",
            "file_type": "directory",
            "permission_mode": format_mode(pre_mode),
            "proposal_id": name,
            "proposal_record_digest": proposal_record.get("record_digest"),
            "terminal_presence": terminal,
        }
        return entry, False
    finally:
        os.close(case_fd)


def _classify_namespace(base_dir: Path | None) -> tuple[str, str, list[str]]:
    runs_root = paths.runs_root(base_dir)
    if not runs_root.is_dir():
        digest = sha256_digest({"schema_version": NAMESPACE_INTEGRITY_SCHEMA, "entries": []})
        return "healthy", digest, []
    runs_fd = open_dir_no_follow(runs_root, context="runs_root")
    try:
        try:
            control_fd = openat_dir_no_follow(runs_fd, ".control", context=".control")
        except BoundedActionValidationError:
            digest = sha256_digest({"schema_version": NAMESPACE_INTEGRITY_SCHEMA, "entries": []})
            return "healthy", digest, []
        try:
            control_st = os.fstat(control_fd)
            if not stat.S_ISDIR(control_st.st_mode) or stat.S_IMODE(control_st.st_mode) != CONTROL_DIR_MODE:
                digest = sha256_digest({"schema_version": NAMESPACE_INTEGRITY_SCHEMA, "entries": []})
                return "indeterminate", digest, ["__control__"]
            try:
                ba_fd = openat_dir_no_follow(
                    control_fd,
                    paths.BOUNDED_ACTIONS_DIR_NAME,
                    context="bounded_actions",
                )
            except BoundedActionValidationError:
                digest = sha256_digest({"schema_version": NAMESPACE_INTEGRITY_SCHEMA, "entries": []})
                return "healthy", digest, []
        finally:
            os.close(control_fd)
    finally:
        os.close(runs_fd)
    try:
        validate_dir_mode_0700(ba_fd, context="bounded_actions", require_ownership=False)
        entries: list[dict[str, Any]] = []
        indeterminate: list[str] = []
        for name in list_dir_names_sorted(ba_fd):
            entry, is_indeterminate = _inspect_namespace_child(ba_fd, name)
            entries.append(entry)
            if is_indeterminate:
                indeterminate.append(name)
        klass = "indeterminate" if indeterminate else "healthy"
        digest = sha256_digest({"schema_version": NAMESPACE_INTEGRITY_SCHEMA, "entries": entries})
        return klass, digest, indeterminate
    finally:
        os.close(ba_fd)


def _scan_case_proposal_entry(case_fd: int, proposal_id: str) -> dict[str, Any] | None:
    proposal_record, _ = _load_record_if_present(case_fd, "proposal.json")
    if proposal_record is None:
        return None
    child_names = list_dir_names_sorted(case_fd)
    terminal = "none"
    esc = 0
    if "review_decision.json" in child_names:
        terminal = "review"
    elif "escalation.json" in child_names:
        terminal = "escalation"
        esc = 1
    return {
        "proposal_id": proposal_id,
        "proposal_record_digest": proposal_record.get("record_digest"),
        "terminal_type": terminal,
        "escalation_contribution": esc,
        "successor_run_id": proposal_record.get("successor_run_id"),
    }


def _scan_target_aggregate(successor_run_id: str, base_dir: Path | None) -> tuple[int, int, str, str]:
    proposal_count = 0
    escalation_count = 0
    entries: list[dict[str, Any]] = []
    ba_fd = _try_open_bounded_actions_fd(base_dir)
    if ba_fd is not None:
        try:
            for name in list_dir_names_sorted(ba_fd):
                if not _BAR_RE.match(name):
                    continue
                try:
                    pre_identity = stat_entry_identity(ba_fd, name)
                    pre_mode = stat_entry_mode(ba_fd, name)
                except OSError:
                    continue
                if not is_directory_mode(pre_mode):
                    continue
                case_fd = openat_dir_no_follow(ba_fd, name, context=name)
                try:
                    if fstat_identity(case_fd) != pre_identity:
                        continue
                    scanned = _scan_case_proposal_entry(case_fd, name)
                    if scanned is None or scanned["successor_run_id"] != successor_run_id:
                        continue
                    proposal_count += 1
                    escalation_count += scanned["escalation_contribution"]
                    entries.append(
                        {
                            "proposal_id": scanned["proposal_id"],
                            "proposal_record_digest": scanned["proposal_record_digest"],
                            "terminal_type": scanned["terminal_type"],
                            "escalation_contribution": scanned["escalation_contribution"],
                        }
                    )
                finally:
                    os.close(case_fd)
        finally:
            os.close(ba_fd)
    proj = {
        "schema_version": TARGET_AGGREGATE_SCHEMA,
        "target_successor_run_id": successor_run_id,
        "target_successor_project_identity_digest": _project_dir_path_digest(base_dir),
        "target_successor_runs_root_identity_digest": _runs_root_path_digest(base_dir),
        "entries": sorted(entries, key=lambda e: e["proposal_id"]),
        "proposal_count": proposal_count,
        "escalation_count": escalation_count,
        "protocol_cap_max_proposals": BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR,
        "protocol_cap_max_escalations": BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR,
    }
    digest = sha256_digest(proj)
    if proposal_count > BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR:
        return proposal_count, escalation_count, "cap_exhausted", digest
    if escalation_count > BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR:
        return proposal_count, escalation_count, "cap_exhausted", digest
    return proposal_count, escalation_count, "healthy", digest


def _load_validated_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = parse_strict_json_bytes(path.read_bytes())
    try:
        _validate_authority_booleans(record)
    except BoundedActionValidationError as exc:
        if str(exc).startswith("missing required field:"):
            raise
        return None
    validate_record_digest(record)
    return record


def load_bounded_action_proposal_bundle(
    proposal_id: str,
    *,
    base_dir: Path | None = None,
) -> BoundedActionProposalBundle:
    require_id(proposal_id, "bounded_action_proposal")
    case_path = paths.bounded_action_case_dir(proposal_id, base_dir)
    if not case_path.is_dir():
        return BoundedActionProposalBundle(proposal_id, None, None, None, "absent")
    proposal = _load_validated_record(paths.bounded_action_proposal_path(proposal_id, base_dir))
    review = _load_validated_record(paths.bounded_action_review_decision_path(proposal_id, base_dir))
    escalation = _load_validated_record(paths.bounded_action_escalation_path(proposal_id, base_dir))
    if review and escalation:
        state: Literal["absent", "proposal", "review", "escalation", "conflicting", "malformed"] = "conflicting"
    elif review:
        state = "review"
    elif escalation:
        state = "escalation"
    elif proposal:
        state = "proposal"
    else:
        state = "malformed"
    return BoundedActionProposalBundle(proposal_id, proposal, review, escalation, state)


def reconcile_bounded_action(proposal_id: str, *, base_dir: Path | None = None) -> BoundedActionPublicationResult:
    require_id(proposal_id, "bounded_action_proposal")
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=base_dir)
    ns_class, _, _ = _classify_namespace(base_dir)
    if bundle.proposal is None:
        return BoundedActionPublicationResult(
            publication_result="absent",
            local_publication_result="absent",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class=ns_class,
            target_aggregate_class="not_inspected",
            new_publication_allowed=ns_class == "healthy",
        )
    return BoundedActionPublicationResult(
        publication_result="exact_replay" if bundle.terminal_state == "proposal" else bundle.terminal_state,
        local_publication_result="exact_replay",
        request_intent_digest_match=True,
        current_evidence_class="not_inspected",
        namespace_integrity_class=ns_class,
        target_aggregate_class="not_inspected",
        new_publication_allowed=ns_class == "healthy",
        record_digest=bundle.proposal.get("record_digest"),
        exact_replay=bundle.terminal_state == "proposal",
    )


def reconcile_successor_bounded_action_state(
    successor_run_id: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    require_id(successor_run_id, "run")
    pc, ec, agg_class, digest = _scan_target_aggregate(successor_run_id, base_dir)
    ns_class, ns_digest, indet = _classify_namespace(base_dir)
    return {
        "successor_run_id": successor_run_id,
        "proposal_count": pc,
        "escalation_count": ec,
        "target_aggregate_class": agg_class,
        "target_aggregate_snapshot_digest": digest,
        "namespace_integrity_class": ns_class,
        "namespace_integrity_snapshot_digest": ns_digest,
        "indeterminate_case_ids": indet,
    }


def inspect_bounded_action_eligibility(
    source_run_id: str,
    successor_run_id: str,
    recovery_request_id: str,
    proposal_subject: str,
    *,
    base_dir: Path | None = None,
    reason_codes: list[str] | None = None,
    risk_class: str | None = None,
    confidence_class: str | None = None,
) -> BoundedActionEligibilityResult:
    require_id(source_run_id, "run")
    require_id(successor_run_id, "run")
    require_id(recovery_request_id, "recovery_run_request")
    matrix = SUBJECT_MATRIX.get(proposal_subject)
    if matrix is None:
        return BoundedActionEligibilityResult(False, "unknown_subject")
    if not classify_marker_for_subject(proposal_subject, successor_run_id, base_dir):
        return BoundedActionEligibilityResult(False, "marker_blocked")
    allow_marker = not matrix.get("marker_required_absent", True)
    try:
        build_task27_evidence(
            recovery_request_id,
            source_run_id=source_run_id,
            successor_run_id=successor_run_id,
            base_dir=base_dir,
        )
        build_source_evidence(source_run_id, base_dir)
        task27 = build_task27_evidence(
            recovery_request_id,
            source_run_id=source_run_id,
            successor_run_id=successor_run_id,
            base_dir=base_dir,
        )
        build_successor_evidence(
            successor_run_id,
            source_run_id=source_run_id,
            task27=task27,
            base_dir=base_dir,
            allow_marker=allow_marker,
        )
    except (BoundedActionPreconditionError, BoundedActionValidationError) as exc:
        return BoundedActionEligibilityResult(False, "task27_ineligible", (str(exc),))
    if reason_codes is not None:
        required = matrix.get("required_reasons", frozenset())
        forbidden = matrix.get("forbidden_reasons", frozenset())
        if not required.issubset(set(reason_codes)):
            return BoundedActionEligibilityResult(False, "reason_codes_ineligible")
        if forbidden & set(reason_codes):
            return BoundedActionEligibilityResult(False, "reason_codes_forbidden")
    if risk_class is not None and risk_class not in matrix.get("risk", frozenset()):
        return BoundedActionEligibilityResult(False, "risk_ineligible")
    if confidence_class is not None and confidence_class not in matrix.get("confidence", frozenset()):
        return BoundedActionEligibilityResult(False, "confidence_ineligible")
    return BoundedActionEligibilityResult(True, "eligible")


def _write_immutable(case_fd: int, filename: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = canonical_json_bytes(body)
    fd = openat_file_no_follow(
        case_fd,
        filename,
        _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC,
        CONTROL_FILE_MODE,
        context=filename,
    )
    try:
        write_all_fd(fd, payload)
        fsync_file_fd(fd)
    finally:
        os.close(fd)
    fsync_dir_fd(case_fd)
    return parse_strict_json_bytes(payload)


def _publication_observations(
    successor_run_id: str,
    *,
    base_dir: Path | None,
) -> tuple[str, str, str]:
    ns_class, _, _ = _classify_namespace(base_dir)
    _, _, agg_class, _ = _scan_target_aggregate(successor_run_id, base_dir)
    return ns_class, agg_class, ns_class


def _literal_exact_replay_result(
    existing: dict[str, Any],
    *,
    successor_run_id: str,
    base_dir: Path | None,
) -> BoundedActionPublicationResult:
    ns_class, agg_class, _ = _publication_observations(successor_run_id, base_dir=base_dir)
    return BoundedActionPublicationResult(
        publication_result="exact_replay",
        local_publication_result="exact_replay",
        request_intent_digest_match=True,
        current_evidence_class="unchanged",
        namespace_integrity_class=ns_class,
        target_aggregate_class=agg_class,
        new_publication_allowed=ns_class == "healthy",
        record_digest=existing.get("record_digest"),
        exact_replay=True,
    )


def _locked_identical_publication_result(
    existing: dict[str, Any],
    *,
    ns_class: str,
    agg_class: str,
) -> BoundedActionPublicationResult:
    return BoundedActionPublicationResult(
        publication_result="concurrent_identical_publication_observed",
        local_publication_result="exact_replay",
        request_intent_digest_match=True,
        current_evidence_class="unchanged",
        namespace_integrity_class=ns_class,
        target_aggregate_class=agg_class,
        new_publication_allowed=ns_class == "healthy",
        record_digest=existing.get("record_digest"),
        exact_replay=False,
    )


def _try_proposal_early_resolution(
    proposal_id: str,
    intent_a: dict[str, Any],
    successor_run_id: str,
    *,
    base_dir: Path | None,
) -> BoundedActionPublicationResult | None:
    """Literal zero-lock resolution before bootstrap or flock acquisition."""
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=base_dir)
    if bundle.proposal is None:
        return None
    validate_record_digest(bundle.proposal)
    intent_digest = sha256_digest(intent_a)
    stored_intent = bundle.proposal.get("request_intent_digest")
    ns_class, agg_class, _ = _publication_observations(successor_run_id, base_dir=base_dir)
    if intent_digest != stored_intent:
        return BoundedActionPublicationResult(
            publication_result="conflict",
            local_publication_result="conflict",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class=ns_class,
            target_aggregate_class=agg_class,
            new_publication_allowed=False,
            exact_replay=False,
        )
    return _literal_exact_replay_result(
        bundle.proposal,
        successor_run_id=successor_run_id,
        base_dir=base_dir,
    )


def _check_replay(existing: dict[str, Any], intent_a: dict[str, Any]) -> bool:
    stored_intent = existing.get("request_intent_digest")
    computed = sha256_digest(intent_a)
    return stored_intent == computed and validate_record_digest(existing) or False


def create_bounded_action_proposal(
    proposal_id: str,
    source_run_id: str,
    successor_run_id: str,
    recovery_request_id: str,
    creator: str,
    proposal_subject: str,
    proposal_summary: str,
    risk_class: str,
    confidence_class: str,
    reason_codes: list[str],
    *,
    reason_detail: str | None = None,
    base_dir: Path | None = None,
) -> BoundedActionPublicationResult:
    require_id(proposal_id, "bounded_action_proposal")
    if proposal_id != paths.bounded_action_case_dir(proposal_id, base_dir).name:
        raise BoundedActionValidationError("proposal_id mismatch")
    creator = _validate_actor(creator, field="creator")
    summary = _validate_summary(proposal_summary)
    codes = _validate_reason_codes(
        reason_codes,
        allowed=frozenset(reason_codes) | frozenset(SUBJECT_MATRIX.get(proposal_subject, {}).get("required_reasons", frozenset())),
    )
    eligibility = inspect_bounded_action_eligibility(
        source_run_id, successor_run_id, recovery_request_id, proposal_subject, base_dir=base_dir,
        reason_codes=codes, risk_class=risk_class, confidence_class=confidence_class,
    )
    if not eligibility.eligible:
        return BoundedActionPublicationResult(
            publication_result="precondition_failed",
            local_publication_result="absent",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class="not_inspected",
            target_aggregate_class="not_inspected",
            new_publication_allowed=False,
        )
    intent_a = _proposal_intent_a(
        proposal_id=proposal_id,
        source_run_id=source_run_id,
        successor_run_id=successor_run_id,
        creator=creator,
        proposal_subject=proposal_subject,
        proposal_summary=summary,
        risk_class=risk_class,
        confidence_class=confidence_class,
        reason_codes=codes,
        reason_detail=reason_detail,
    )
    replay = _try_proposal_early_resolution(proposal_id, intent_a, successor_run_id, base_dir=base_dir)
    if replay is not None:
        return replay
    ns_class, ns_digest, _ = _classify_namespace(base_dir)
    if ns_class != "healthy":
        return BoundedActionPublicationResult(
            publication_result="namespace_integrity_indeterminate",
            local_publication_result="absent",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class=ns_class,
            target_aggregate_class="not_inspected",
            new_publication_allowed=False,
        )
    pc, ec, agg_class, agg_digest = _scan_target_aggregate(successor_run_id, base_dir)
    if agg_class == "cap_exhausted" and pc >= BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR:
        ns_class, _, _ = _classify_namespace(base_dir)
        return BoundedActionPublicationResult(
            publication_result="cap_exhausted",
            local_publication_result="absent",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class=ns_class,
            target_aggregate_class=agg_class,
            new_publication_allowed=False,
        )
    source_ev = build_source_evidence(source_run_id, base_dir)
    task27 = build_task27_evidence(
        recovery_request_id, source_run_id=source_run_id, successor_run_id=successor_run_id, base_dir=base_dir
    )
    successor_ev = build_successor_evidence(
        successor_run_id, source_run_id=source_run_id, task27=task27, base_dir=base_dir
    )
    locks = bootstrap_publication_tree(base_dir)
    try:
        with root_publication_lock(locks):
            ns_class, ns_digest, _ = _classify_namespace(base_dir)
            with successor_coord_lock(locks, successor_run_id, base_dir):
                pc, ec, agg_class, agg_digest = _scan_target_aggregate(successor_run_id, base_dir)
                if pc >= BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR:
                    return BoundedActionPublicationResult(
                        publication_result="cap_exhausted",
                        local_publication_result="absent",
                        request_intent_digest_match=False,
                        current_evidence_class="not_inspected",
                        namespace_integrity_class=ns_class,
                        target_aggregate_class=agg_class,
                        new_publication_allowed=False,
                    )
                if ns_class != "healthy":
                    return BoundedActionPublicationResult(
                        publication_result="namespace_integrity_indeterminate",
                        local_publication_result="absent",
                        request_intent_digest_match=False,
                        current_evidence_class="not_inspected",
                        namespace_integrity_class=ns_class,
                        target_aggregate_class="not_inspected",
                        new_publication_allowed=False,
                    )
                with case_lock(locks, proposal_id, create=True) as case_fd:
                    existing, _ = _load_record_if_present(case_fd, "proposal.json")
                    if existing is not None:
                        if sha256_digest(intent_a) == existing.get("request_intent_digest"):
                            validate_record_digest(existing)
                            return _locked_identical_publication_result(
                                existing,
                                ns_class=ns_class,
                                agg_class=agg_class,
                            )
                        raise BoundedActionConflictError("proposal intent conflict")
                    body: dict[str, Any] = {
                        "record_type": RECORD_TYPE_PROPOSAL,
                        "schema_version": SCHEMA_VERSION,
                        "protocol_version": PROTOCOL_VERSION,
                        "proposal_id": proposal_id,
                        "source_run_id": source_run_id,
                        "successor_run_id": successor_run_id,
                        "source_project_identity_digest": _project_dir_path_digest(base_dir),
                        "source_runs_root_identity_digest": _runs_root_path_digest(base_dir),
                        "successor_project_identity_digest": _project_dir_path_digest(base_dir),
                        "successor_runs_root_identity_digest": _runs_root_path_digest(base_dir),
                        "creator": creator,
                        "created_at": _utc_now_iso(),
                        "predecessor_record_type": None,
                        "predecessor_raw_digest": None,
                        "predecessor_semantic_digest": None,
                        "source_evidence": source_ev,
                        "task27_evidence": task27,
                        "successor_evidence": successor_ev,
                        "proposal_subject": proposal_subject,
                        "proposal_summary": summary,
                        "proposal_summary_digest": sha256_digest({"summary": summary}),
                        "risk_class": risk_class,
                        "confidence_class": confidence_class,
                        "reason_codes": codes,
                        "reason_detail": reason_detail,
                        "request_intent_digest": sha256_digest(intent_a),
                        "namespace_integrity_snapshot_digest": ns_digest,
                        "target_aggregate_snapshot_digest": agg_digest,
                        "proposal_count_before": pc,
                        "proposal_count_after": pc + 1,
                        "escalation_count_before": ec,
                        "escalation_count_after": ec,
                        "protocol_cap_max_proposals": BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR,
                        "protocol_cap_max_escalations": BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR,
                        **authority_booleans_false(),
                    }
                    body["record_digest"] = compute_record_digest(body)
                    persisted = _write_immutable(case_fd, "proposal.json", body)
                    return BoundedActionPublicationResult(
                        publication_result="published_new",
                        local_publication_result="published_new",
                        request_intent_digest_match=True,
                        current_evidence_class="unchanged",
                        namespace_integrity_class=ns_class,
                        target_aggregate_class=agg_class,
                        new_publication_allowed=True,
                        record_digest=persisted.get("record_digest"),
                    )
    finally:
        release_publication_locks(locks)


def record_bounded_action_review_decision(
    proposal_id: str,
    review_decision_id: str,
    expected_proposal_record_digest: str,
    reviewer: str,
    decision_class: str,
    risk_class: str,
    confidence_class: str,
    reason_codes: list[str],
    *,
    reason_detail: str | None = None,
    base_dir: Path | None = None,
) -> BoundedActionPublicationResult:
    return _record_terminal(
        proposal_id=proposal_id,
        terminal_id=review_decision_id,
        terminal_kind="review",
        expected_proposal_record_digest=expected_proposal_record_digest,
        actor=reviewer,
        actor_field="reviewer",
        class_field="decision_class",
        class_value=decision_class,
        risk_class=risk_class,
        confidence_class=confidence_class,
        reason_codes=reason_codes,
        reason_detail=reason_detail,
        base_dir=base_dir,
    )


def record_bounded_action_escalation(
    proposal_id: str,
    escalation_id: str,
    expected_proposal_record_digest: str,
    escalator: str,
    escalation_class: str,
    risk_class: str,
    confidence_class: str,
    reason_codes: list[str],
    *,
    reason_detail: str | None = None,
    base_dir: Path | None = None,
) -> BoundedActionPublicationResult:
    return _record_terminal(
        proposal_id=proposal_id,
        terminal_id=escalation_id,
        terminal_kind="escalation",
        expected_proposal_record_digest=expected_proposal_record_digest,
        actor=escalator,
        actor_field="escalator",
        class_field="escalation_class",
        class_value=escalation_class,
        risk_class=risk_class,
        confidence_class=confidence_class,
        reason_codes=reason_codes,
        reason_detail=reason_detail,
        base_dir=base_dir,
    )


def _try_terminal_early_resolution(
    proposal_id: str,
    terminal_kind: str,
    intent_a: dict[str, Any],
    expected_proposal_record_digest: str,
    *,
    base_dir: Path | None,
) -> BoundedActionPublicationResult | None:
    bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=base_dir)
    if bundle.proposal is None:
        return None
    successor_run_id = bundle.proposal["successor_run_id"]
    ns_class, agg_class, _ = _publication_observations(successor_run_id, base_dir=base_dir)
    if bundle.proposal.get("record_digest") != expected_proposal_record_digest:
        return BoundedActionPublicationResult(
            publication_result="expected_proposal_digest_mismatch",
            local_publication_result="conflict",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class=ns_class,
            target_aggregate_class=agg_class,
            new_publication_allowed=False,
            exact_replay=False,
        )
    existing = bundle.review_decision if terminal_kind == "review" else bundle.escalation
    if existing is None:
        return None
    validate_record_digest(existing)
    intent_digest = sha256_digest(intent_a)
    if intent_digest != existing.get("request_intent_digest"):
        return BoundedActionPublicationResult(
            publication_result="conflict",
            local_publication_result="conflict",
            request_intent_digest_match=False,
            current_evidence_class="not_inspected",
            namespace_integrity_class=ns_class,
            target_aggregate_class=agg_class,
            new_publication_allowed=False,
            exact_replay=False,
        )
    return _literal_exact_replay_result(
        existing,
        successor_run_id=successor_run_id,
        base_dir=base_dir,
    )


def _record_terminal(
    *,
    proposal_id: str,
    terminal_id: str,
    terminal_kind: str,
    expected_proposal_record_digest: str,
    actor: str,
    actor_field: str,
    class_field: str,
    class_value: str,
    risk_class: str,
    confidence_class: str,
    reason_codes: list[str],
    reason_detail: str | None,
    base_dir: Path | None,
) -> BoundedActionPublicationResult:
    require_id(proposal_id, "bounded_action_proposal")
    id_kind = "bounded_action_review_decision" if terminal_kind == "review" else "bounded_action_escalation"
    require_id(terminal_id, id_kind)
    actor = _validate_actor(actor, field=actor_field)
    filename = "review_decision.json" if terminal_kind == "review" else "escalation.json"
    record_type = RECORD_TYPE_REVIEW if terminal_kind == "review" else RECORD_TYPE_ESCALATION
    intent_kwargs = {
        "proposal_id": proposal_id,
        "expected_proposal_record_digest": expected_proposal_record_digest,
        "risk_class": risk_class,
        "confidence_class": confidence_class,
        "reason_codes": reason_codes,
        "reason_detail": reason_detail,
    }
    if terminal_kind == "review":
        intent_a = _review_intent_a(review_decision_id=terminal_id, reviewer=actor, decision_class=class_value, **intent_kwargs)
    else:
        intent_a = _escalation_intent_a(escalation_id=terminal_id, escalator=actor, escalation_class=class_value, **intent_kwargs)
    replay = _try_terminal_early_resolution(
        proposal_id,
        terminal_kind,
        intent_a,
        expected_proposal_record_digest,
        base_dir=base_dir,
    )
    if replay is not None:
        return replay
    locks = bootstrap_publication_tree(base_dir)
    try:
        with root_publication_lock(locks):
            ns_class, ns_digest, _ = _classify_namespace(base_dir)
            if ns_class != "healthy":
                return BoundedActionPublicationResult(
                    publication_result="namespace_integrity_indeterminate",
                    local_publication_result="absent",
                    request_intent_digest_match=False,
                    current_evidence_class="not_inspected",
                    namespace_integrity_class=ns_class,
                    target_aggregate_class="not_inspected",
                    new_publication_allowed=False,
                )
            bundle = load_bounded_action_proposal_bundle(proposal_id, base_dir=base_dir)
            if bundle.proposal is None:
                return BoundedActionPublicationResult(
                    publication_result="precondition_failed",
                    local_publication_result="absent",
                    request_intent_digest_match=False,
                    current_evidence_class="not_inspected",
                    namespace_integrity_class=ns_class,
                    target_aggregate_class="not_inspected",
                    new_publication_allowed=False,
                )
            if bundle.proposal.get("record_digest") != expected_proposal_record_digest:
                return BoundedActionPublicationResult(
                    publication_result="expected_proposal_digest_mismatch",
                    local_publication_result="conflict",
                    request_intent_digest_match=False,
                    current_evidence_class="not_inspected",
                    namespace_integrity_class=ns_class,
                    target_aggregate_class="not_inspected",
                    new_publication_allowed=False,
                )
            successor_run_id = bundle.proposal["successor_run_id"]
            source_run_id = bundle.proposal["source_run_id"]
            with successor_coord_lock(locks, successor_run_id, base_dir):
                pc, ec, agg_class, agg_digest = _scan_target_aggregate(successor_run_id, base_dir)
                if terminal_kind == "escalation" and ec >= BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR:
                    return BoundedActionPublicationResult(
                        publication_result="cap_exhausted",
                        local_publication_result="absent",
                        request_intent_digest_match=False,
                        current_evidence_class="not_inspected",
                        namespace_integrity_class=ns_class,
                        target_aggregate_class=agg_class,
                        new_publication_allowed=False,
                    )
                with case_lock(locks, proposal_id, create=False) as case_fd:
                    existing, _ = _load_record_if_present(case_fd, filename)
                    if existing is not None:
                        if sha256_digest(intent_a) == existing.get("request_intent_digest"):
                            validate_record_digest(existing)
                            return _locked_identical_publication_result(
                                existing,
                                ns_class=ns_class,
                                agg_class=agg_class,
                            )
                        raise BoundedActionConflictError("terminal intent conflict")
                    if (bundle.review_decision and terminal_kind != "review") or (
                        bundle.escalation and terminal_kind != "escalation"
                    ):
                        return BoundedActionPublicationResult(
                            publication_result="conflict",
                            local_publication_result="conflict",
                            request_intent_digest_match=False,
                            current_evidence_class="not_inspected",
                            namespace_integrity_class=ns_class,
                            target_aggregate_class=agg_class,
                            new_publication_allowed=False,
                        )
                    source_ev = build_source_evidence(source_run_id, base_dir)
                    task27 = build_task27_evidence(
                        bundle.proposal["task27_evidence"]["recovery_request_id"],
                        source_run_id=source_run_id,
                        successor_run_id=successor_run_id,
                        base_dir=base_dir,
                    )
                    successor_ev = build_successor_evidence(
                        successor_run_id, source_run_id=source_run_id, task27=task27, base_dir=base_dir
                    )
                    esc_after = ec + (1 if terminal_kind == "escalation" else 0)
                    body: dict[str, Any] = {
                        "record_type": record_type,
                        "schema_version": SCHEMA_VERSION,
                        "protocol_version": PROTOCOL_VERSION,
                        "proposal_id": proposal_id,
                        "review_decision_id" if terminal_kind == "review" else "escalation_id": terminal_id,
                        "source_run_id": source_run_id,
                        "successor_run_id": successor_run_id,
                        "source_project_identity_digest": bundle.proposal["source_project_identity_digest"],
                        "source_runs_root_identity_digest": bundle.proposal["source_runs_root_identity_digest"],
                        "successor_project_identity_digest": bundle.proposal["successor_project_identity_digest"],
                        "successor_runs_root_identity_digest": bundle.proposal["successor_runs_root_identity_digest"],
                        actor_field: actor,
                        "created_at": _utc_now_iso(),
                        "expected_proposal_record_digest": expected_proposal_record_digest,
                        "proposal_predecessor_raw_digest": sha256_digest({"proposal": bundle.proposal["record_digest"]}),
                        "proposal_predecessor_semantic_digest": sha256_digest(projection_b(bundle.proposal)),
                        "proposal_subject": bundle.proposal["proposal_subject"],
                        "proposal_summary_digest": bundle.proposal["proposal_summary_digest"],
                        "proposal_risk_class": bundle.proposal["risk_class"],
                        "proposal_confidence_class": bundle.proposal["confidence_class"],
                        "proposal_reason_codes": bundle.proposal["reason_codes"],
                        "proposal_time_source_evidence_digest": sha256_digest(bundle.proposal["source_evidence"]),
                        "proposal_time_task27_evidence_digest": sha256_digest(bundle.proposal["task27_evidence"]),
                        "proposal_time_successor_evidence_digest": sha256_digest(bundle.proposal["successor_evidence"]),
                        "source_evidence": source_ev,
                        "task27_evidence": task27,
                        "successor_evidence": successor_ev,
                        "evidence_drift_classification": evidence_drift(
                            bundle.proposal.get("source_evidence"), source_ev
                        ),
                        class_field: class_value,
                        "risk_class": risk_class,
                        "confidence_class": confidence_class,
                        "reason_codes": reason_codes,
                        "reason_detail": reason_detail,
                        "request_intent_digest": sha256_digest(intent_a),
                        "namespace_integrity_snapshot_digest": ns_digest,
                        "target_aggregate_snapshot_digest": agg_digest,
                        "proposal_count_before": pc,
                        "proposal_count_after": pc,
                        "escalation_count_before": ec,
                        "escalation_count_after": esc_after,
                        "protocol_cap_max_proposals": BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR,
                        "protocol_cap_max_escalations": BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR,
                        **authority_booleans_false(),
                    }
                    if terminal_kind == "review":
                        body.pop("escalator", None)
                        body.pop("escalation_class", None)
                    else:
                        body.pop("reviewer", None)
                        body.pop("decision_class", None)
                    body["record_digest"] = compute_record_digest(body)
                    persisted = _write_immutable(case_fd, filename, body)
                    return BoundedActionPublicationResult(
                        publication_result="published_new",
                        local_publication_result="published_new",
                        request_intent_digest_match=True,
                        current_evidence_class=body["evidence_drift_classification"],
                        namespace_integrity_class=ns_class,
                        target_aggregate_class=agg_class,
                        new_publication_allowed=True,
                        record_digest=persisted.get("record_digest"),
                    )
    finally:
        release_publication_locks(locks)
