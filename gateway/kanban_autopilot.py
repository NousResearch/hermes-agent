"""Kanban-first /autopilot command surface.

Autopilot is a bounded controller over the existing Kanban dispatcher.  It owns
mode/focus/policy decisions and ready-gate filtering; the dispatcher remains the
only claim/spawn/retry/accounting substrate.  `/autopilot on` uses the default
policy scope, while `/autopilot on <parent>` narrows the same bounded loop.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

AUTOPILOT_STATE_FILE = "gateway_autopilot_state.json"
AUTOPILOT_STATE_VERSION = 2
AUTOPILOT_USAGE = "/autopilot [status|dry-run|queue|on|pause [reason]|pause-lane <tenant> [reason]|resume-lane <tenant>|hard-stop <reason>|recover <ack>|off|stop|focus <BO-123>]"
_READ_ONLY_ACTIONS = {"status", "dry-run", "dry_run", "queue"}
_CONTROL_ACTIONS = {"on", "off", "stop", "pause", "pause-lane", "resume-lane", "hard-stop", "recover", "focus", "once"}
_MUTATIONS_ATTEMPTED: list[str] = []
_READY_GATE_REQUIREMENTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("missing_goal", ("goal:", "goal -", "objective:"), "goal"),
    ("missing_end_state", ("end-state", "end state", "output:", "deliverable"), "end-state/output"),
    ("missing_scope_non_goals", ("scope/non-goals", "non-goals", "non goals", "out of scope"), "scope/non-goals"),
    ("missing_acceptance_criteria", ("acceptance criteria", "acceptance:"), "acceptance criteria"),
    ("missing_verification_requirements", ("verification requirements", "verification:", "tests_run", "test:"), "verification requirements"),
    ("missing_authority_boundary", ("authority boundary", "authority:", "kanban"), "authority boundary"),
    ("missing_repo_lane_truth", ("repo/lane truth", "repo_full_name", "repository", "branch"), "repo/lane truth"),
    ("missing_risk_flags", ("risk flags", "risk:", "env", "secret", "prod", "customer-visible", "restart"), "risk flags"),
    ("missing_dependencies_blockers", ("dependencies/blockers", "dependencies:", "blockers:", "none"), "dependencies/blockers"),
    ("missing_review_package_expectation", ("review package", "review_ready", "changed files", "commit"), "review package expectation"),
)
_DONE_CRITERIA_LEDGER_SCHEMA = "autopilot_done_criteria_ledger.v1"
_DONE_CRITERIA_LEDGER_VERSION = 1
_DONE_CRITERIA_SECTION_HEADERS = (
    "done criteria",
    "done criteria ledger",
)
_DONE_CRITERIA_HEADER_RE = re.compile(
    r"^\s*(?P<header>done criteria|done criteria ledger)\s*:\s*(?P<inline>.*\S)?\s*$",
    re.IGNORECASE,
)
_DONE_CRITERIA_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|(?:\d+|[A-Za-z])[.)])\s+(?P<text>.+\S)\s*$")
_DONE_CRITERIA_AMBIGUOUS_RE = re.compile(
    r"\b(?:and/or|either\b|maybe\b|perhaps\b|some\s+way|tbd\b|to\s+be\s+determined|etc\.?|or\b)"
    r"|\b\w+\s*/\s*\w+\b",
    re.IGNORECASE,
)
_KNOWN_TASK_SECTION_HEADERS = {
    "goal",
    "end-state/output",
    "scope/non-goals",
    "acceptance criteria",
    "verification requirements",
    "authority boundary",
    "repo/lane truth",
    "risk flags",
    "dependencies/blockers",
    "review package expectation",
    "done criteria",
    "done criteria ledger",
}
_PR_URL_RE = re.compile(r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)/pull/\d+/?$")
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
_RELEASE_ONLY_BASES = {"prod", "production"}
_CLOSED_LOOP_ALLOWED_STATES = [
    "disabled",
    "dry_run",
    "single_flight",
    "bounded_multi_tick",
    "parent_scoped",
    "lane_scoped",
    "paused",
    "hard_stopped",
    "needs_human",
]
_DEFAULT_CLOSED_LOOP_CAPS = {
    "max_active_flights": 1,
    "max_dispatches_per_tick": 1,
    "max_tasks_per_run_single_flight": 1,
    "max_tasks_per_run_early_bounded_multi_tick": 2,
    "max_new_prs_per_run": 1,
    "max_open_autopilot_prs": 2,
    "max_consecutive_failures": 1,
    "max_no_progress_ticks": 1,
    "max_same_card_retries": 1,
    "max_runtime_minutes": 60,
    "max_daily_autopilot_tasks": 3,
    "require_clean_closeout_per_task": True,
    "require_review_ready_contract_before_next_task": True,
}


def _normalize_done_criteria_text(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().strip("•*-:;.,")
    return normalized.lower()


def _slugify_done_criteria_text(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "criterion"


def _done_criteria_is_ambiguous(text: str) -> bool:
    lowered = text.lower()
    return bool(_DONE_CRITERIA_AMBIGUOUS_RE.search(lowered))


def _extract_done_criteria_lines(body: str) -> tuple[list[str], str | None]:
    items: list[str] = []
    section: str | None = None
    collecting = False
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        header_match = _DONE_CRITERIA_HEADER_RE.match(stripped)
        if header_match:
            section = header_match.group("header").lower()
            collecting = True
            inline = _normalize_done_criteria_text(header_match.group("inline"))
            if inline:
                items.append(inline)
            continue
        if not collecting:
            continue
        current_header = stripped.split(":", 1)[0].strip().lower() if ":" in stripped else ""
        if current_header in _KNOWN_TASK_SECTION_HEADERS and current_header not in _DONE_CRITERIA_SECTION_HEADERS:
            break
        item_match = _DONE_CRITERIA_ITEM_RE.match(stripped)
        if item_match:
            items.append(_normalize_done_criteria_text(item_match.group("text")))
            continue
        if items:
            items[-1] = _normalize_done_criteria_text(f"{items[-1]} {stripped}")
        else:
            items.append(_normalize_done_criteria_text(stripped))
    return items, section


def build_done_criteria_ledger(body: Any) -> dict[str, Any]:
    text = str(body or "")
    items, section = _extract_done_criteria_lines(text)
    reason_codes: list[str] = []
    if not items:
        reason_codes.append("missing_done_criteria_ledger")
        return {
            "ok": False,
            "status": "rejected",
            "reason_codes": reason_codes,
            "human_reason": "Missing explicit done criteria ledger section.",
            "done_criteria_ledger": None,
        }
    criteria: list[dict[str, Any]] = []
    ambiguous_items: list[str] = []
    slug_counts: dict[str, int] = {}
    for index, raw_item in enumerate(items, start=1):
        normalized = _normalize_done_criteria_text(raw_item)
        if not normalized:
            continue
        if _done_criteria_is_ambiguous(normalized):
            ambiguous_items.append(normalized)
        slug = _slugify_done_criteria_text(normalized)
        slug_counts[slug] = slug_counts.get(slug, 0) + 1
        suffix = f"-{slug_counts[slug]}" if slug_counts[slug] > 1 else ""
        criteria.append(
            {
                "id": f"dc-{index:02d}-{slug}{suffix}",
                "text": normalized,
            }
        )
    if not criteria:
        reason_codes.append("missing_done_criteria_ledger")
        return {
            "ok": False,
            "status": "rejected",
            "reason_codes": reason_codes,
            "human_reason": "Missing explicit done criteria ledger section.",
            "done_criteria_ledger": None,
        }
    if ambiguous_items:
        return {
            "ok": False,
            "status": "rejected",
            "reason_codes": ["ambiguous_done_criteria_ledger"],
            "human_reason": "Ambiguous done criteria require refinement: " + ", ".join(ambiguous_items),
            "done_criteria_ledger": None,
        }
    normalized_ledger = {
        "schema": _DONE_CRITERIA_LEDGER_SCHEMA,
        "version": _DONE_CRITERIA_LEDGER_VERSION,
        "criteria": criteria,
    }
    criteria_hash = hashlib.sha256(
        json.dumps(normalized_ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ledger = {
        **normalized_ledger,
        "criteria_hash": criteria_hash,
    }
    return {
        "ok": True,
        "status": "accepted",
        "reason_codes": [],
        "human_reason": "done criteria ledger extracted",
        "done_criteria_ledger": ledger,
        "criteria_hash": criteria_hash,
        "criteria_version": _DONE_CRITERIA_LEDGER_VERSION,
        "criteria_ids": [criterion["id"] for criterion in criteria],
        "source_section": section or "done criteria",
    }


def _criteria_hash_from_evidence(evidence: dict[str, Any]) -> str:
    for key in ("done_criteria_hash", "criteria_hash"):
        value = str(evidence.get(key) or "").strip()
        if value:
            return value
    ledger = evidence.get("done_criteria_ledger")
    if isinstance(ledger, dict):
        value = str(ledger.get("criteria_hash") or "").strip()
        if value:
            return value
    return ""


def _evidence_uses_task_worktree(evidence: dict[str, Any]) -> bool:
    workspace_kind = str(evidence.get("workspace_kind") or "").strip().lower()
    return bool(
        evidence.get("task_owned_worktree") is True
        or workspace_kind == "worktree"
        or str(evidence.get("worktree_path") or "").strip()
    )


def _worktree_cleanup_blockers(evidence: dict[str, Any]) -> list[str]:
    cleanup = evidence.get("cleanup") if isinstance(evidence.get("cleanup"), dict) else {}
    git = evidence.get("git") if isinstance(evidence.get("git"), dict) else {}
    proof = str(cleanup.get("proof") or evidence.get("cleanup_proof") or "").strip()
    worktree_clean = cleanup.get("worktree_clean")
    git_clean = git.get("worktree_clean")
    status_short = str(git.get("status_short") or "").strip()
    if proof and worktree_clean is not False and git_clean is not False and not status_short:
        return []
    residue = evidence.get("residue")
    if not isinstance(residue, dict):
        return ["missing_cleanup_proof"]
    items = residue.get("items")
    if not isinstance(items, list):
        return ["missing_cleanup_proof"]
    blockers: list[str] = []
    retained_seen = False
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        disposition = str(raw_item.get("disposition") or "").strip().lower()
        if disposition == "retained":
            retained_seen = True
            if not str(raw_item.get("reason") or "").strip():
                blockers.append("retained_residue_missing_reason")
            ttl = str(raw_item.get("ttl") or raw_item.get("revisit_at") or raw_item.get("expires_at") or "").strip()
            if not ttl:
                blockers.append("retained_residue_missing_ttl")
    if retained_seen and not blockers:
        return []
    return blockers or ["missing_cleanup_proof"]


def _done_criteria_validation_for_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    source_text = str(
        evidence.get("task_body")
        or evidence.get("body")
        or evidence.get("task_spec")
        or evidence.get("spec")
        or ""
    )
    if not source_text:
        return {
            "ok": False,
            "reason_codes": ["missing_done_criteria_ledger"],
            "human_reason": "Missing explicit done criteria ledger source text.",
            "done_criteria_ledger": None,
        }
    extracted = build_done_criteria_ledger(source_text)
    if not extracted.get("ok"):
        return extracted
    expected_hash = _criteria_hash_from_evidence(evidence)
    if expected_hash and expected_hash != extracted["criteria_hash"]:
        return {
            "ok": False,
            "reason_codes": ["stale_criteria_hash"],
            "human_reason": "Stale done criteria hash no longer matches the current ledger.",
            "done_criteria_ledger": extracted.get("done_criteria_ledger"),
            "criteria_hash": extracted["criteria_hash"],
        }
    return extracted


def _criterion_requires_deterministic_verification(text: str) -> bool:
    lowered = _normalize_done_criteria_text(text)
    return any(token in lowered for token in ("test", "pytest", "check", "diff", "verification", "artifact"))


def _normalize_artifact_refs(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_worker_criterion_proofs(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    raw = evidence.get("criteria_proofs")
    if raw is None:
        raw = evidence.get("criterion_proofs")
    if raw is None:
        raw = evidence.get("per_criterion_proof")
    if raw is None:
        return []
    normalized: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = enumerate(raw, start=1)
    else:
        return []
    for key, value in iterable:
        if isinstance(value, dict):
            proof = dict(value)
        else:
            proof = {"proof": value}
        proof_id = str(
            proof.get("criterion_id")
            or proof.get("criteria_id")
            or proof.get("id")
            or proof.get("done_criteria_id")
            or key
        ).strip()
        if proof_id:
            proof["criterion_id"] = proof_id
        normalized.append(proof)
    return normalized


def _proof_supports_deterministic_verification(proof: dict[str, Any]) -> bool:
    if proof.get("tests_passed") is True or proof.get("checks_passed") is True:
        return True
    for key in (
        "tests",
        "tests_run",
        "test_command",
        "checks",
        "checks_run",
        "check_command",
        "verification",
        "verification_command",
    ):
        value = proof.get(key)
        if value not in (None, "", [], {}, ()):
            return True
    return False


def _normalize_verifier_criterion_results(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    normalized: list[dict[str, Any]] = []
    if isinstance(value, dict):
        iterable = value.items()
    elif isinstance(value, list):
        iterable = enumerate(value, start=1)
    else:
        return []
    for key, raw in iterable:
        if isinstance(raw, dict):
            result = dict(raw)
        else:
            result = {"status": raw}
        criterion_id = str(
            result.get("criterion_id")
            or result.get("criteria_id")
            or result.get("id")
            or result.get("done_criteria_id")
            or key
        ).strip()
        if criterion_id:
            result["criterion_id"] = criterion_id
        normalized.append(result)
    return normalized


def _verifier_result_passed(result: dict[str, Any]) -> bool:
    status = str(result.get("status") or result.get("verdict") or result.get("result") or "").strip().lower()
    return status in {"pass", "passed", "ok", "success", "satisfied"} or result.get("passed") is True


def _verifier_result_failed(result: dict[str, Any]) -> bool:
    status = str(result.get("status") or result.get("verdict") or result.get("result") or "").strip().lower()
    return status in {"fail", "failed", "error", "rejected", "unsatisfied"} or result.get("passed") is False


def evaluate_verifier_verdict(
    worker_evidence: dict[str, Any],
    verifier_evidence: dict[str, Any] | None = None,
    *,
    task_body: Any = None,
    task_spec: Any = None,
) -> dict[str, Any]:
    """Build an independent verifier verdict over worker_done evidence.

    The verifier is deliberately a separate contract from worker_done and from
    review_ready.  It is pure/read-only: no Kanban transition, claim, spawn, PR,
    merge, release, restart, or provider/customer mutation happens here.
    """

    verifier_evidence = dict(verifier_evidence or {})
    source_evidence = dict(worker_evidence)
    if task_body is not None:
        source_evidence["task_body"] = task_body
    if task_spec is not None:
        source_evidence["task_spec"] = task_spec
    reason_codes: list[str] = []
    remediation: list[str] = []
    blocker_reason_codes = [
        str(code).strip()
        for code in (verifier_evidence.get("blocker_reason_codes") or verifier_evidence.get("blockers") or [])
        if str(code).strip()
    ]
    if blocker_reason_codes:
        return {
            "verdict": "BLOCKED",
            "status": "blocked",
            "reason_codes": blocker_reason_codes,
            "blocker_reason_codes": blocker_reason_codes,
            "remediation_instructions": verifier_evidence.get("remediation_instructions") or [],
            "criterion_results": [],
            "criteria_ids": [],
            "criteria_hash": _criteria_hash_from_evidence(source_evidence),
            "worker_done_evidence": None,
            "worker_done_retained": source_evidence.get("worker_done") is True or source_evidence.get("kanban_worker_done") is True,
            "review_ready_input_eligible": False,
            "human_reason": "Verifier blocked by external or authority reason: " + ", ".join(blocker_reason_codes),
            "side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
        }

    worker_validation = validate_worker_done_evidence(source_evidence, task_body=source_evidence.get("task_body"), task_spec=source_evidence.get("task_spec"))
    ledger_reason_codes = worker_validation.get("reason_codes") or []
    refinement_codes = {
        "missing_done_criteria_ledger",
        "ambiguous_done_criteria_ledger",
        "stale_criteria_hash",
    }
    if any(code in refinement_codes for code in ledger_reason_codes):
        selected_codes = [code for code in ledger_reason_codes if code in refinement_codes]
        return {
            "verdict": "REFINEMENT_REQUIRED",
            "status": "refinement_required",
            "reason_codes": selected_codes,
            "blocker_reason_codes": [],
            "remediation_instructions": ["Refine the Done Criteria Ledger and rerun worker evidence against the current criteria_hash."],
            "criterion_results": [],
            "criteria_ids": worker_validation.get("criteria_ids") or [],
            "criteria_hash": worker_validation.get("criteria_hash") or _criteria_hash_from_evidence(source_evidence),
            "worker_done_evidence": worker_validation,
            "worker_done_retained": source_evidence.get("worker_done") is True or source_evidence.get("kanban_worker_done") is True,
            "review_ready_input_eligible": False,
            "human_reason": "Verifier requires task refinement before implementation evidence can be accepted: " + ", ".join(selected_codes),
            "side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
        }

    worker_identity = str(source_evidence.get("worker_identity") or source_evidence.get("worker") or source_evidence.get("assignee") or "").strip()
    verifier_identity = str(verifier_evidence.get("verifier_identity") or verifier_evidence.get("verifier") or verifier_evidence.get("reviewer") or "").strip()
    if not verifier_identity:
        reason_codes.append("missing_verifier_identity")
        remediation.append("Provide verifier_identity for the independent verifier run.")
    if worker_identity and verifier_identity and worker_identity == verifier_identity:
        reason_codes.append("verifier_same_as_worker")
        remediation.append("Use a verifier identity distinct from the worker identity.")
    if not worker_validation.get("worker_done_evidence_valid"):
        reason_codes.extend(worker_validation.get("reason_codes") or ["invalid_worker_done_evidence"])
        remediation.append("Remediate worker_done evidence until every Done criterion has artifact-backed proof and required checks.")

    criterion_results = _normalize_verifier_criterion_results(
        verifier_evidence.get("criterion_results")
        if verifier_evidence.get("criterion_results") is not None
        else verifier_evidence.get("criteria_results")
    )
    result_by_id = {str(result.get("criterion_id") or "").strip(): result for result in criterion_results if str(result.get("criterion_id") or "").strip()}
    criteria_ids = list(worker_validation.get("criteria_ids") or [])
    if not criterion_results:
        reason_codes.append("missing_verifier_criterion_results")
        remediation.append("Add verifier criterion_results with PASS/FAIL status and evidence for every Done criterion id.")
    for criterion_id in criteria_ids:
        result = result_by_id.get(criterion_id)
        if result is None:
            reason_codes.append(f"missing_verifier_result_for_{criterion_id}")
            remediation.append(f"Verify criterion {criterion_id} and record a criterion-level verdict.")
            continue
        evidence_text = str(result.get("evidence") or result.get("proof") or result.get("summary") or "").strip()
        if not evidence_text:
            reason_codes.append(f"missing_verifier_evidence_for_{criterion_id}")
            remediation.append(f"Add verifier evidence/proof text for criterion {criterion_id}.")
        if _verifier_result_failed(result):
            reason_codes.append(f"verifier_failed_{criterion_id}")
            instruction = str(result.get("remediation") or result.get("remediation_instruction") or result.get("action") or "").strip()
            remediation.append(instruction or f"Remediate failed criterion {criterion_id} and rerun verifier.")
        elif not _verifier_result_passed(result):
            reason_codes.append(f"verifier_result_not_pass_for_{criterion_id}")
            remediation.append(f"Set criterion {criterion_id} to PASS only after verifier evidence proves it.")
    accepted = not reason_codes
    deduped_reason_codes = list(dict.fromkeys(reason_codes))
    deduped_remediation = list(dict.fromkeys(item for item in remediation if str(item).strip()))
    return {
        "verdict": "PASS" if accepted else "FAIL",
        "status": "passed" if accepted else "failed",
        "reason_codes": deduped_reason_codes,
        "blocker_reason_codes": [],
        "remediation_instructions": deduped_remediation,
        "criterion_results": criterion_results,
        "criteria_ids": criteria_ids,
        "criteria_hash": worker_validation.get("criteria_hash"),
        "worker_done_evidence": worker_validation,
        "worker_done_retained": source_evidence.get("worker_done") is True or source_evidence.get("kanban_worker_done") is True,
        "review_ready_input_eligible": accepted,
        "human_reason": "verifier PASS: all done criteria independently satisfied" if accepted else "verifier FAIL: " + ", ".join(deduped_reason_codes),
        "side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
    }


def plan_verifier_retry_controller(
    worker_evidence: dict[str, Any],
    verifier_evidence: dict[str, Any] | None = None,
    *,
    task_body: Any = None,
    task_spec: Any = None,
    attempt: int | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Plan the Kanban-visible verifier/remediation transition after worker_done.

    This is check-only controller logic. It derives the next Kanban-visible
    action from worker_done + verifier verdict, persists attempt facts in the
    returned evidence shape, and never claims, spawns, mutates, restarts, merges,
    deploys, or touches external authority by itself.
    """

    raw_attempt = attempt
    if raw_attempt is None:
        for key in ("verification_attempt", "attempt", "retry_count"):
            value = worker_evidence.get(key)
            if value is not None:
                raw_attempt = value
                break
    try:
        attempt_number = int(raw_attempt) if raw_attempt is not None else 1
    except (TypeError, ValueError):
        attempt_number = 1
    attempt_number = max(1, attempt_number)
    max_attempts = max(1, int(max_attempts or 1))
    verifier = evaluate_verifier_verdict(
        worker_evidence,
        verifier_evidence or {},
        task_body=task_body,
        task_spec=task_spec,
    )
    verdict = verifier.get("verdict")
    queue: list[dict[str, Any]] = []
    reason_codes: list[str] = list(verifier.get("reason_codes") or [])
    blocked = False
    review_ready_input_eligible = False
    if verdict == "PASS":
        next_state = "verifier_pass"
        review_ready_input_eligible = True
    elif verdict == "FAIL":
        if attempt_number >= max_attempts:
            next_state = "blocked"
            blocked = True
            reason_codes.append("max_verification_attempts_exhausted")
        else:
            next_state = "queue_remediation"
            queue.append(
                {
                    "type": "remediation",
                    "attempt": attempt_number + 1,
                    "reason_codes": list(verifier.get("reason_codes") or []),
                    "remediation_instructions": verifier.get("remediation_instructions") or [],
                }
            )
    elif verdict == "REFINEMENT_REQUIRED":
        next_state = "refinement_required"
    elif verdict == "BLOCKED":
        next_state = "blocked"
        blocked = True
    else:
        next_state = "blocked"
        blocked = True
        reason_codes.append("unknown_verifier_verdict")
    reason_codes = list(dict.fromkeys(reason_codes))
    return {
        "controller": "autopilot_verifier_retry_controller.v1",
        "next_state": next_state,
        "verifier_verdict": verifier,
        "attempt": attempt_number,
        "max_attempts": max_attempts,
        "retry_remaining": max(0, max_attempts - attempt_number),
        "queued_actions": queue,
        "queued_remediation_count": len(queue),
        "review_ready_input_eligible": review_ready_input_eligible,
        "blocked": blocked,
        "reason_codes": reason_codes,
        "kanban_evidence_patch": {
            "verification_attempt": attempt_number,
            "max_verification_attempts": max_attempts,
            "last_verifier_verdict": verdict,
            "last_verifier_reason_codes": verifier.get("reason_codes") or [],
            "next_controller_state": next_state,
        },
        "side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
        "human_reason": (
            "verifier PASS: review_ready input may be built"
            if review_ready_input_eligible
            else "verifier controller state: " + next_state + (" (" + ", ".join(reason_codes) + ")" if reason_codes else "")
        ),
    }


def validate_worker_done_evidence(
    evidence: dict[str, Any],
    task_body: Any = None,
    task_spec: Any = None,
) -> dict[str, Any]:
    """Validate worker self-report evidence against the done-criteria ledger."""

    reason_codes: list[str] = []
    worker_done_observed = evidence.get("kanban_worker_done") is True or evidence.get("worker_done") is True
    if not worker_done_observed:
        reason_codes.append("missing_worker_done_observation")
    source_evidence = dict(evidence)
    if task_body is not None:
        source_evidence["task_body"] = task_body
    if task_spec is not None:
        source_evidence["task_spec"] = task_spec
    done_criteria = _done_criteria_validation_for_evidence(source_evidence)
    if not done_criteria.get("ok"):
        reason_codes.extend(done_criteria.get("reason_codes") or ["invalid_done_criteria_ledger"])
        return {
            "worker_done_evidence_valid": False,
            "status": "blocked",
            "reason_codes": reason_codes,
            "human_reason": "Worker done evidence missing or invalid done criteria: " + ", ".join(reason_codes),
            "done_criteria_ledger": done_criteria.get("done_criteria_ledger"),
            "criteria_hash": done_criteria.get("criteria_hash"),
            "criteria_version": done_criteria.get("criteria_version"),
            "criteria_ids": done_criteria.get("criteria_ids") or [],
            "criteria_proofs": [],
            "worker_done_observed": worker_done_observed,
            "authority_boundary_confirmed": False,
            "cleanup_or_residue_proof": False,
        }
    provided_hash = _criteria_hash_from_evidence(evidence)
    if not provided_hash:
        reason_codes.append("missing_criteria_hash")
    elif provided_hash != done_criteria["criteria_hash"]:
        reason_codes.append("stale_criteria_hash")
    criterion_proofs = _normalize_worker_criterion_proofs(evidence)
    if not criterion_proofs:
        reason_codes.append("missing_criterion_level_evidence")
    proof_by_id: dict[str, dict[str, Any]] = {}
    for proof in criterion_proofs:
        proof_id = str(proof.get("criterion_id") or "").strip()
        if proof_id:
            proof_by_id[proof_id] = proof
    if criterion_proofs:
        for criterion in done_criteria["done_criteria_ledger"]["criteria"]:
            proof = proof_by_id.get(criterion["id"])
            if proof is None:
                reason_codes.append(f"missing_proof_for_{criterion['id']}")
                continue
            proof_text = str(proof.get("proof") or proof.get("evidence") or proof.get("summary") or proof.get("result") or "").strip()
            if not proof_text:
                reason_codes.append(f"missing_proof_text_for_{criterion['id']}")
            artifact_refs = _normalize_artifact_refs(proof.get("artifact_refs") or proof.get("artifact_ref") or proof.get("artifacts"))
            if not artifact_refs:
                reason_codes.append(f"missing_artifact_refs_for_{criterion['id']}")
            if _criterion_requires_deterministic_verification(criterion["text"]) and not _proof_supports_deterministic_verification(proof):
                reason_codes.append(f"missing_tests_or_checks_for_{criterion['id']}")
    authority_boundary_confirmed = evidence.get("authority_boundary_confirmed")
    if authority_boundary_confirmed is None:
        authority_boundary_confirmed = evidence.get("boundaries_confirmed")
    if authority_boundary_confirmed is not True:
        reason_codes.append("missing_authority_boundary_confirmation")
    cleanup_blockers = _worktree_cleanup_blockers(evidence)
    if cleanup_blockers:
        reason_codes.extend(cleanup_blockers)
    accepted = not reason_codes
    return {
        "worker_done_evidence_valid": accepted,
        "status": "accepted" if accepted else "blocked",
        "reason_codes": reason_codes,
        "human_reason": "worker done evidence satisfied" if accepted else "Worker done evidence missing or incomplete: " + ", ".join(reason_codes),
        "done_criteria_ledger": done_criteria.get("done_criteria_ledger"),
        "criteria_hash": done_criteria.get("criteria_hash"),
        "criteria_version": done_criteria.get("criteria_version"),
        "criteria_ids": done_criteria.get("criteria_ids") or [],
        "criteria_proofs": criterion_proofs,
        "worker_done_observed": worker_done_observed,
        "authority_boundary_confirmed": authority_boundary_confirmed is True,
        "cleanup_or_residue_proof": not cleanup_blockers,
    }


_VERIFIER_VERDICTS = ("PASS", "FAIL", "BLOCKED", "REFINEMENT_REQUIRED")


def _normalize_identity_value(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip().lower()
        if text:
            return text
    return ""


def _worker_identity_from_evidence(evidence: dict[str, Any]) -> str:
    return _normalize_identity_value(
        evidence.get("worker_identity"),
        evidence.get("worker_profile"),
        evidence.get("worker_actor"),
        evidence.get("completed_by"),
        evidence.get("completed_by_profile"),
        evidence.get("profile"),
        evidence.get("task_assignee"),
        evidence.get("assignee"),
        evidence.get("worker"),
    )


def _verifier_identity_from_evidence(evidence: dict[str, Any], verifier_identity: Any = None) -> str:
    return _normalize_identity_value(
        verifier_identity,
        evidence.get("verifier_identity"),
        evidence.get("verifier_profile"),
        evidence.get("verifier_actor"),
        evidence.get("reviewer"),
        evidence.get("reviewer_profile"),
        evidence.get("completed_by_reviewer"),
    ) or "__independent_verifier__"


def _verifier_requirement_flags(criterion_text: str) -> dict[str, bool]:
    lowered = _normalize_done_criteria_text(criterion_text)
    return {
        "deterministic": _criterion_requires_deterministic_verification(criterion_text),
        "repo_policy": any(token in lowered for token in ("repo-policy", "repo policy", "policy", "policy-check")),
        "tests": any(token in lowered for token in ("test", "pytest", "check", "verification", "diff")),
        "pr_package": any(token in lowered for token in ("pr", "package", "review package", "artifact")),
        "cleanup": any(token in lowered for token in ("cleanup", "worktree", "residue", "clean")),
        "authority": any(token in lowered for token in ("authority", "boundary", "self-approval", "approval")),
    }


def build_verifier_intake_payload(
    completed_event: dict[str, Any],
    task_runs: Any = None,
    done_criteria_ledger: dict[str, Any] | None = None,
    *,
    verifier_identity: Any = None,
) -> dict[str, Any]:
    """Normalize the verifier intake from completion event, runs, and the ledger."""

    task_run_snapshot: dict[str, Any] = {}
    if isinstance(task_runs, dict):
        task_run_snapshot = dict(task_runs)
    elif isinstance(task_runs, list) and task_runs:
        last = task_runs[-1]
        if isinstance(last, dict):
            task_run_snapshot = dict(last)
    elif isinstance(completed_event.get("task_run"), dict):
        task_run_snapshot = dict(completed_event["task_run"])

    source = dict(completed_event)
    if done_criteria_ledger is not None:
        source["done_criteria_ledger"] = done_criteria_ledger
    if task_run_snapshot:
        source["task_run"] = task_run_snapshot
    ledger_validation = _done_criteria_validation_for_evidence(source)
    worker_identity = _worker_identity_from_evidence({**completed_event, **task_run_snapshot})
    verifier_identity_norm = _verifier_identity_from_evidence({**completed_event, **task_run_snapshot}, verifier_identity=verifier_identity)
    return {
        "completed_event": dict(completed_event),
        "task_run": task_run_snapshot,
        "task_runs": task_runs if isinstance(task_runs, list) else ([task_runs] if isinstance(task_runs, dict) else ([] if task_runs is None else task_runs)),
        "done_criteria_ledger": ledger_validation.get("done_criteria_ledger"),
        "criteria_hash": ledger_validation.get("criteria_hash"),
        "criteria_version": ledger_validation.get("criteria_version"),
        "criteria_ids": ledger_validation.get("criteria_ids") or [],
        "worker_identity": worker_identity,
        "verifier_identity": verifier_identity_norm,
        "task_body": completed_event.get("task_body") or completed_event.get("body"),
        "task_spec": completed_event.get("task_spec") or completed_event.get("spec"),
    }


def evaluate_verifier_result(
    evidence: dict[str, Any],
    task_runs: Any = None,
    done_criteria_ledger: dict[str, Any] | None = None,
    *,
    verifier_identity: Any = None,
    task_body: Any = None,
    task_spec: Any = None,
) -> dict[str, Any]:
    """Evaluate an independent verifier result for worker completion evidence."""

    completed_event = dict(evidence)
    if task_body is not None:
        completed_event["task_body"] = task_body
    if task_spec is not None:
        completed_event["task_spec"] = task_spec
    intake = build_verifier_intake_payload(
        completed_event,
        task_runs=task_runs if task_runs is not None else evidence.get("task_runs"),
        done_criteria_ledger=done_criteria_ledger or evidence.get("done_criteria_ledger"),
        verifier_identity=verifier_identity,
    )
    worker_identity = intake["worker_identity"]
    verifier_identity_norm = intake["verifier_identity"]
    worker_validation = validate_worker_done_evidence(
        {**completed_event, "task_body": intake.get("task_body"), "task_spec": intake.get("task_spec")},
        task_body=intake.get("task_body"),
        task_spec=intake.get("task_spec"),
    )
    ledger_validation = _done_criteria_validation_for_evidence({
        **completed_event,
        "task_body": intake.get("task_body"),
        "task_spec": intake.get("task_spec"),
        "done_criteria_ledger": intake.get("done_criteria_ledger"),
        "criteria_hash": intake.get("criteria_hash"),
    })
    criteria = list((ledger_validation.get("done_criteria_ledger") or {}).get("criteria") or [])
    criterion_proofs = _normalize_worker_criterion_proofs(completed_event)
    proof_by_id: dict[str, dict[str, Any]] = {}
    for proof in criterion_proofs:
        proof_id = str(proof.get("criterion_id") or "").strip()
        if proof_id:
            proof_by_id[proof_id] = proof
    same_worker = bool(worker_identity and verifier_identity_norm and worker_identity == verifier_identity_norm)
    criterion_results: list[dict[str, Any]] = []
    for criterion in criteria:
        criterion_id = str(criterion.get("id") or "").strip() or "unknown"
        criterion_text = str(criterion.get("text") or "").strip()
        proof = proof_by_id.get(criterion_id)
        row_reason_codes: list[str] = []
        remediation: list[str] = []
        checks: list[dict[str, Any]] = []
        row_verdict = "PASS"
        if same_worker:
            row_verdict = "BLOCKED"
            row_reason_codes.append("self_approval_prohibited")
            remediation.append("Use a distinct verifier identity; a worker cannot self-approve review readiness.")
        else:
            if proof is None:
                row_verdict = "FAIL"
                row_reason_codes.append(f"missing_proof_for_{criterion_id}")
                remediation.append("Attach criterion-level proof and artifact references.")
            else:
                proof_text = str(proof.get("proof") or proof.get("evidence") or proof.get("summary") or proof.get("result") or "").strip()
                if not proof_text:
                    row_verdict = "FAIL"
                    row_reason_codes.append(f"missing_proof_text_for_{criterion_id}")
                    remediation.append("Write an explicit proof summary for this criterion.")
                artifact_refs = _normalize_artifact_refs(proof.get("artifact_refs") or proof.get("artifact_ref") or proof.get("artifacts"))
                if not artifact_refs:
                    row_verdict = "FAIL"
                    row_reason_codes.append(f"missing_artifact_refs_for_{criterion_id}")
                    remediation.append("Attach artifact references that back the proof.")
                requirements = _verifier_requirement_flags(criterion_text)
                if requirements["deterministic"]:
                    supported = _proof_supports_deterministic_verification(proof)
                    checks.append({"name": "deterministic_verification", "required": True, "passed": supported})
                    if not supported:
                        row_verdict = "FAIL"
                        row_reason_codes.append(f"missing_tests_or_checks_for_{criterion_id}")
                        remediation.append("Add test or check evidence for this deterministic criterion.")
                if requirements["repo_policy"]:
                    repo_policy = completed_event.get("repo_policy")
                    policy_ok = isinstance(repo_policy, dict) and repo_policy.get("ok") is True
                    checks.append({"name": "repo_policy", "required": True, "passed": policy_ok})
                    if not policy_ok:
                        row_verdict = "FAIL"
                        row_reason_codes.append("missing_repo_policy_evidence")
                        remediation.append("Attach the repo-policy check result or policy snapshot.")
                if requirements["tests"]:
                    tests_passed = bool(proof.get("tests_passed") is True or proof.get("checks_passed") is True)
                    checks.append({"name": "tests_or_checks", "required": True, "passed": tests_passed})
                    if not tests_passed:
                        row_verdict = "FAIL"
                        row_reason_codes.append(f"missing_tests_or_checks_for_{criterion_id}")
                        remediation.append("Record passing tests or deterministic checks for this criterion.")
                if requirements["pr_package"]:
                    pr_or_package = any(
                        str(completed_event.get(field) or "").strip()
                        for field in ("pr_url", "package_url", "package", "artifact_package")
                    )
                    checks.append({"name": "pr_or_package_evidence", "required": True, "passed": pr_or_package})
                    if not pr_or_package:
                        row_verdict = "FAIL"
                        row_reason_codes.append("missing_pr_or_package_evidence")
                        remediation.append("Provide PR/package evidence for this criterion.")
                if requirements["cleanup"]:
                    cleanup_blockers = _worktree_cleanup_blockers(completed_event)
                    checks.append({"name": "cleanup_proof", "required": True, "passed": not cleanup_blockers})
                    if cleanup_blockers:
                        row_verdict = "BLOCKED"
                        row_reason_codes.extend(cleanup_blockers)
                        remediation.append("Provide cleanup proof or a retained-residue justification with TTL and reason.")
                if requirements["authority"]:
                    authority_ok = completed_event.get("authority_boundary_confirmed") is True or completed_event.get("boundaries_confirmed") is True
                    checks.append({"name": "authority_boundary", "required": True, "passed": authority_ok})
                    if not authority_ok:
                        row_verdict = "BLOCKED"
                        row_reason_codes.append("missing_authority_boundary_confirmation")
                        remediation.append("Confirm the authority boundary before review-ready approval.")
        criterion_results.append({
            "criterion_id": criterion_id,
            "criterion_text": criterion_text,
            "verdict": row_verdict,
            "reason_codes": list(dict.fromkeys(row_reason_codes)),
            "remediation": remediation,
            "checks": checks,
            "evidence": {
                "proof_present": proof is not None,
                "proof_text_present": bool(str((proof or {}).get("proof") or (proof or {}).get("evidence") or (proof or {}).get("summary") or (proof or {}).get("result") or "").strip()),
                "artifact_refs": _normalize_artifact_refs((proof or {}).get("artifact_refs") or (proof or {}).get("artifact_ref") or (proof or {}).get("artifacts")),
            },
        })
    if same_worker and "self_approval_prohibited" not in worker_validation.get("reason_codes", []):
        reason_codes = ["self_approval_prohibited"]
    else:
        reason_codes = list(dict.fromkeys(worker_validation.get("reason_codes") or []))
    if not ledger_validation.get("ok"):
        reason_codes.extend(ledger_validation.get("reason_codes") or ["invalid_done_criteria_ledger"])
    verdict = "PASS"
    if same_worker or any(row["verdict"] == "BLOCKED" for row in criterion_results):
        verdict = "BLOCKED"
    elif not ledger_validation.get("ok") or any(code in {"missing_done_criteria_ledger", "ambiguous_done_criteria_ledger", "stale_criteria_hash"} for code in reason_codes):
        verdict = "REFINEMENT_REQUIRED"
    elif reason_codes or any(row["verdict"] == "FAIL" for row in criterion_results):
        verdict = "FAIL"
    if not criteria and ledger_validation.get("ok"):
        criterion_results.append({
            "criterion_id": "__done_criteria__",
            "criterion_text": "done criteria ledger",
            "verdict": verdict,
            "reason_codes": reason_codes[:],
            "remediation": [],
            "checks": [],
            "evidence": {"proof_present": False, "proof_text_present": False, "artifact_refs": []},
        })
    verdict_reason_codes = list(dict.fromkeys(reason_codes + [code for row in criterion_results for code in row["reason_codes"]]))
    if verdict == "PASS":
        human_reason = "verifier result satisfied"
    elif verdict == "REFINEMENT_REQUIRED":
        human_reason = "Verifier needs a clearer or fresher done criteria ledger: " + ", ".join(verdict_reason_codes)
    elif verdict == "BLOCKED":
        human_reason = "Verifier blocked by authority/self-approval or cleanup issues: " + ", ".join(verdict_reason_codes)
    else:
        human_reason = "Verifier found fixable evidence gaps: " + ", ".join(verdict_reason_codes)
    storage_payload = {
        "verdict": verdict,
        "reason_codes": verdict_reason_codes,
        "worker_identity": worker_identity or None,
        "verifier_identity": verifier_identity_norm,
        "criteria_hash": intake.get("criteria_hash"),
        "criteria_ids": intake.get("criteria_ids") or [],
        "criterion_results": criterion_results,
        "review_ready": verdict == "PASS",
    }
    return {
        "verifier_result_valid": verdict == "PASS",
        "review_ready": verdict == "PASS",
        "verdict": verdict,
        "status": verdict.lower(),
        "reason_codes": verdict_reason_codes,
        "human_reason": human_reason,
        "worker_identity": worker_identity or None,
        "verifier_identity": verifier_identity_norm,
        "criteria_hash": intake.get("criteria_hash"),
        "criteria_version": intake.get("criteria_version"),
        "criteria_ids": intake.get("criteria_ids") or [],
        "done_criteria_ledger": intake.get("done_criteria_ledger"),
        "criterion_results": criterion_results,
        "completed_event": intake.get("completed_event"),
        "task_run": intake.get("task_run"),
        "task_runs": intake.get("task_runs") if isinstance(intake.get("task_runs"), list) else [],
        "kanban_ssot": {"task_runs": {"metadata": {"verifier_result": storage_payload}}},
    }


def get_closed_loop_operating_contract() -> dict[str, Any]:
    """Return the BO-091 closed-loop Autopilot ADR/operating contract.

    The contract is intentionally data-shaped so later slices can validate
    policy/config files against the same invariants instead of relying on prose.
    It grants no runtime authority: dispatcher handoff and worker completion
    remain separate later slices and explicit approval gates.
    """

    return {
        "adr": "bounded_controller_not_executor",
        "authority_ceiling": "review_ready_pr",
        "state_machine": {
            "allowed_states": list(_CLOSED_LOOP_ALLOWED_STATES),
            "promotion_ladder": [
                ["dry_run", "single_flight"],
                ["single_flight", "bounded_multi_tick"],
                ["bounded_multi_tick", "parent_scoped"],
                ["bounded_multi_tick", "lane_scoped"],
            ],
            "terminal_or_human_states": ["paused", "hard_stopped", "needs_human"],
        },
        "default_caps": dict(_DEFAULT_CLOSED_LOOP_CAPS),
        "dispatcher_boundary": {
            "execution_owner": "existing_kanban_dispatcher",
            "autopilot_role": "controller_policy_evidence_layer",
            "autopilot_may_directly_claim_or_spawn": False,
            "second_dispatcher_allowed": False,
            "handoff_success_is_worker_completion": False,
            "worker_done_truth_source": "kanban_dispatcher_worker_done_evidence",
        },
        "scope_model": {
            "selectors": ["parent_public_id", "lane_tenant", "repo_project", "labels", "assignee_or_profile"],
            "scope_can_silently_widen": False,
            "scope_escape_result": "needs_human_or_activation_rejected",
        },
        "forbidden_without_current_approval": [
            "gateway_restart_reload",
            "config_env_secret_provider_billing_pricing_mutation",
            "worker_dispatch_claim_spawn_live_side_effect_before_activation_slice",
            "fork_push_or_pr_before_child_range_worker_done",
            "upstream_pr",
            "merge_release_deploy_prod_customer_visible_action",
            "canonical_main_sync_or_materialization",
        ],
        "stop_conditions": [
            "forbidden_action_requested",
            "scope_ambiguity",
            "stale_kanban_state",
            "dependency_or_blocker_detected",
            "verification_failure_threshold_exceeded",
            "worker_crash_or_timeout_repeated",
            "dispatcher_unavailable",
            "kanban_read_unavailable",
            "policy_file_invalid_or_stale",
            "worker_completion_evidence_missing",
            "budget_or_cap_exceeded",
            "pr_backlog_cap_exceeded",
            "disk_ci_runtime_safety_threshold_exceeded",
            "current_turn_approval_required_or_expired",
        ],
        "future_ralplan_required_for": [
            "merge_release_deploy_prod_customer_visible_authority",
            "gateway_restart_reload_automation",
            "config_env_secret_provider_billing_pricing_mutation",
            "replace_or_bypass_existing_dispatcher",
            "new_dispatcher_worker_lifecycle_ownership",
            "global_queue_draining_autonomy",
            "cross_repo_lane_parent_scope_expansion",
            "customer_facing_automation",
            "destructive_cleanup_authority",
            "security_auth_payment_billing_policy_mutation",
        ],
        "docs": "docs/closed-loop-kanban-autopilot.md",
    }


def validate_closed_loop_policy_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate that a proposed closed-loop policy preserves BO-091 invariants."""

    reason_codes: list[str] = []
    if contract.get("adr") != "bounded_controller_not_executor":
        reason_codes.append("adr_must_be_bounded_controller_not_executor")
    if contract.get("authority_ceiling") != "review_ready_pr":
        reason_codes.append("authority_ceiling_must_be_review_ready_pr")
    states = (contract.get("state_machine") or {}).get("allowed_states") or []
    for state in _CLOSED_LOOP_ALLOWED_STATES:
        if state not in states:
            reason_codes.append(f"missing_state_{state}")
    caps = contract.get("default_caps") or {}
    if caps.get("max_dispatches_per_tick") != 1:
        reason_codes.append("max_dispatches_per_tick_must_be_one")
    if caps.get("max_active_flights") != 1:
        reason_codes.append("max_active_flights_must_be_one")
    boundary = contract.get("dispatcher_boundary") or {}
    if boundary.get("execution_owner") != "existing_kanban_dispatcher":
        reason_codes.append("execution_owner_must_be_existing_dispatcher")
    if boundary.get("autopilot_may_directly_claim_or_spawn") is not False:
        reason_codes.append("direct_claim_or_spawn_not_allowed")
    if boundary.get("second_dispatcher_allowed") is not False:
        reason_codes.append("second_dispatcher_not_allowed")
    if boundary.get("handoff_success_is_worker_completion") is not False:
        reason_codes.append("handoff_success_must_not_equal_worker_completion")
    scope = contract.get("scope_model") or {}
    if scope.get("scope_can_silently_widen") is not False:
        reason_codes.append("scope_must_not_silently_widen")
    forbidden = set(contract.get("forbidden_without_current_approval") or [])
    for required in {"gateway_restart_reload", "config_env_secret_provider_billing_pricing_mutation"}:
        if required not in forbidden:
            reason_codes.append(f"missing_forbidden_{required}")
    future = set(contract.get("future_ralplan_required_for") or [])
    if "merge_release_deploy_prod_customer_visible_authority" not in future:
        reason_codes.append("future_ralplan_gate_missing_merge_release_prod")
    return {"ok": not reason_codes, "reason_codes": reason_codes}


def simulate_closed_loop_ticks(candidates: list[dict[str, Any]], *, max_ticks: Optional[int] = None) -> dict[str, Any]:
    """Simulate closed-loop candidate selection without side effects.

    This is BO-092's read-only simulator: it reuses the same Ready-gate and
    dispatcher-eligibility contract as the live path, but returns only
    ``would_*`` facts. It does not dispatch, claim, spawn, or mutate Kanban.
    """

    state = _read_state()
    effective = _effective_mode(str(state.get("desired_mode") or "disabled"), state)
    cap = max(1, int(max_ticks or _DEFAULT_CLOSED_LOOP_CAPS["max_tasks_per_run_early_bounded_multi_tick"]))
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    handoffs: list[dict[str, Any]] = []
    pauses: list[dict[str, Any]] = []

    if effective == "hard_stop":
        pauses.append({"reason_code": "hard_stop", "human_reason": "Autopilot hard-stop is active; no simulated handoff is allowed."})
        for candidate in candidates:
            skipped.append({
                "public_id": candidate.get("public_id"),
                "task_id": candidate.get("id"),
                "status": "rejected",
                "reason_codes": ["autopilot_hard_stop_active"],
            })
        return {
            "mode": "read_only_simulation",
            "controller_effective_mode": effective,
            "would_select": selected,
            "would_skip": skipped,
            "would_pause": pauses,
            "would_handoff": handoffs,
            "next_state": "hard_stopped",
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }

    verdict = evaluate_dispatcher_eligibility(candidates)
    for item in verdict.get("ineligible") or []:
        skipped.append({
            "public_id": item.get("public_id"),
            "task_id": item.get("task_id"),
            "status": item.get("status"),
            "reason_codes": item.get("reason_codes") or [],
            "human_reason": item.get("human_reason"),
        })
    for item in (verdict.get("eligible") or [])[:cap]:
        selected_item = {
            "public_id": item.get("public_id"),
            "task_id": item.get("task_id"),
            "reason_codes": [],
            "decision": "would_select",
        }
        selected.append(selected_item)
        handoffs.append({
            "public_id": item.get("public_id"),
            "task_id": item.get("task_id"),
            "target": "existing_kanban_dispatcher",
            "check_only": True,
            "would_dispatch": False,
            "handoff_success_is_worker_completion": False,
        })
    if len(verdict.get("eligible") or []) > cap:
        for item in (verdict.get("eligible") or [])[cap:]:
            skipped.append({
                "public_id": item.get("public_id"),
                "task_id": item.get("task_id"),
                "status": "skipped",
                "reason_codes": ["max_ticks_cap_reached"],
                "human_reason": f"simulation cap reached: max_ticks={cap}",
            })
    if not selected:
        pauses.append({"reason_code": "no_progress", "human_reason": "No eligible candidate would be selected in this simulation tick."})
    return {
        "mode": "read_only_simulation",
        "controller_effective_mode": verdict.get("controller_effective_mode", effective),
        "max_ticks": cap,
        "would_select": selected,
        "would_skip": skipped,
        "would_pause": pauses,
        "would_handoff": handoffs,
        "next_state": "continue" if selected else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }

def _candidate_active_flight(candidate: dict[str, Any]) -> Optional[dict[str, Any]]:
    status = str(candidate.get("status") or "").lower()
    claim_lock = candidate.get("claim_lock")
    worker_pid = candidate.get("worker_pid")
    current_run_id = candidate.get("current_run_id")
    if status not in {"running", "claimed", "in_progress"} and not any([claim_lock, worker_pid, current_run_id]):
        return None
    return {
        "public_id": candidate.get("public_id"),
        "task_id": candidate.get("task_id") or candidate.get("id"),
        "current_run_id": current_run_id,
        "claim_lock": claim_lock,
        "worker_pid": worker_pid,
    }


def activate_single_flight(
    candidates: list[dict[str, Any]],
    *,
    check_only_handoff: Optional[Any] = None,
) -> dict[str, Any]:
    """Run the BO-093 single-flight activation gate without live dispatch.

    The function selects at most one eligible candidate, performs a caller-
    supplied check-only handoff probe, and returns explicit non-completion
    evidence. It never directly dispatches, claims, spawns, or mutates Kanban.
    """

    active_flights = [flight for candidate in candidates if (flight := _candidate_active_flight(candidate))]
    if active_flights:
        skipped = [
            {
                "public_id": candidate.get("public_id"),
                "task_id": candidate.get("task_id") or candidate.get("id"),
                "status": "skipped",
                "reason_codes": ["active_flight_already_present"],
                "human_reason": "single-flight activation blocked because the scoped Kanban set already has an active worker flight",
            }
            for candidate in candidates
            if not _candidate_active_flight(candidate)
        ]
        return {
            "status": "active_flight_blocked",
            "selected": None,
            "skipped": skipped,
            "handoff": None,
            "check": {"allowed": False, "reason": "active_flight_already_present"},
            "active_flights": active_flights,
            "next_state": "needs_human",
            "handoff_success_is_worker_completion": False,
            "worker_done_observed": False,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }

    simulation = simulate_closed_loop_ticks(candidates, max_ticks=1)
    selected = (simulation.get("would_select") or [])[:1]
    skipped = []
    for item in simulation.get("would_skip") or []:
        reason_codes = item.get("reason_codes") or []
        if reason_codes == ["max_ticks_cap_reached"]:
            item = {**item, "reason_codes": ["single_flight_limit_reached"], "human_reason": "single-flight activation may select at most one candidate"}
        skipped.append(item)
    for extra in (simulation.get("would_select") or [])[1:]:
        skipped.append({
            "public_id": extra.get("public_id"),
            "task_id": extra.get("task_id"),
            "status": "skipped",
            "reason_codes": ["single_flight_limit_reached"],
            "human_reason": "single-flight activation may select at most one candidate",
        })
    if not selected:
        return {
            "status": "no_candidate",
            "selected": None,
            "skipped": skipped,
            "handoff": None,
            "next_state": simulation.get("next_state", "needs_human"),
            "handoff_success_is_worker_completion": False,
            "worker_done_observed": False,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }
    candidate = selected[0]
    payload = {
        "public_id": candidate.get("public_id"),
        "task_id": candidate.get("task_id"),
        "target": "existing_kanban_dispatcher",
        "check_only": True,
        "would_dispatch": False,
        "handoff_success_is_worker_completion": False,
    }
    check = check_only_handoff(payload) if check_only_handoff else {"allowed": True, "reason": "no_check_callback_supplied"}
    allowed = bool((check or {}).get("allowed"))
    return {
        "status": "handoff_check_passed" if allowed else "handoff_check_blocked",
        "selected": candidate,
        "skipped": skipped,
        "handoff": payload,
        "check": check,
        "next_state": "pause_for_worker_observation" if allowed else "needs_human",
        "handoff_success_is_worker_completion": False,
        "worker_done_observed": False,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }


def dispatch_selected_once(
    candidate: dict[str, Any],
    *,
    board: Optional[str] = None,
    failure_limit: int = 1,
) -> dict[str, Any]:
    """Hand one Autopilot-selected task to the existing Kanban dispatcher.

    This intentionally reuses ``kanban_db.dispatch_once`` with a task-id filter
    so Autopilot remains a selector/policy layer, not a second dispatcher.
    """

    task_id = str(candidate.get("task_id") or candidate.get("id") or "").strip()
    if not task_id:
        return {"dispatched": False, "reason": "missing_task_id", "spawned": []}
    try:
        from hermes_cli import kanban_db as kb
    except Exception as exc:
        return {"dispatched": False, "reason": "kanban_db_unavailable", "error": str(exc), "spawned": []}
    conn = None
    try:
        conn = kb.connect(board=board)
        worker_env = {
            "HERMES_KANBAN_AUTOPILOT": "1",
            "HERMES_KANBAN_REVIEW_READY_PR_REQUIRED": "1",
        }
        repo_full_name = str(candidate.get("repo_full_name") or "").strip()
        if repo_full_name:
            worker_env["HERMES_KANBAN_EXPECTED_REPO_FULL_NAME"] = repo_full_name
        result = kb.dispatch_once(
            conn,
            max_spawn=1,
            failure_limit=failure_limit,
            board=board,
            task_ids=[task_id],
            worker_env=worker_env,
        )
    except Exception as exc:
        return {"dispatched": False, "reason": "dispatcher_error", "error": str(exc), "spawned": []}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    spawned = [
        {"task_id": tid, "assignee": who, "workspace": ws}
        for tid, who, ws in (result.spawned or [])
    ]
    return {
        "dispatched": bool(spawned),
        "target_task_id": task_id,
        "reclaimed": result.reclaimed,
        "crashed": result.crashed,
        "timed_out": result.timed_out,
        "auto_blocked": result.auto_blocked,
        "promoted": result.promoted,
        "spawned": spawned,
        "skipped_unassigned": result.skipped_unassigned,
        "skipped_nonspawnable": result.skipped_nonspawnable,
    }


def evaluate_autopilot_closeout_progress(
    evidence: dict[str, Any],
    *,
    open_autopilot_prs: int = 0,
    max_open_autopilot_prs: int = 2,
) -> dict[str, Any]:
    """Decide whether Autopilot may continue after a worker result.

    BO-094 keeps worker_done observation, review-ready/no-code evidence, and PR
    backlog caps distinct. It never grants merge/release authority.
    """

    reason_codes: list[str] = []
    worker_done = evidence.get("kanban_worker_done") is True
    if not worker_done:
        reason_codes.append("worker_done_not_observed")
    no_code = evidence.get("no_code_task") is True
    review_contract: dict[str, Any] | None = None
    review_equivalent = None
    if no_code:
        if not str(evidence.get("artifact_path") or "").strip():
            reason_codes.append("missing_no_code_artifact")
        if not str(evidence.get("verification") or "").strip():
            reason_codes.append("missing_no_code_verification")
        if evidence.get("boundaries_confirmed") is not True:
            reason_codes.append("missing_boundaries_confirmed")
        if not reason_codes:
            review_equivalent = "no_code_evidence"
    else:
        review_contract = evaluate_review_ready_contract(evidence)
        if not review_contract.get("review_ready"):
            reason_codes.append("missing_review_ready_contract")
    if open_autopilot_prs >= max_open_autopilot_prs:
        reason_codes.append("pr_backlog_cap_reached")
    return {
        "may_continue": not reason_codes,
        "reason_codes": reason_codes,
        "worker_done_observed": worker_done,
        "review_ready_contract": review_contract,
        "review_ready_equivalent": review_equivalent,
        "open_autopilot_prs": open_autopilot_prs,
        "max_open_autopilot_prs": max_open_autopilot_prs,
        "merge_allowed": False,
        "release_allowed": False,
        "prod_customer_visible_allowed": False,
        "next_state": "continue" if not reason_codes else "needs_human",
    }


def run_bounded_multi_tick(
    candidates: list[dict[str, Any]],
    *,
    max_tasks: int = 2,
    closeout_results: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Simulate bounded multi-tick continuation without live side effects."""

    cap = max(1, max_tasks)
    simulation = simulate_closed_loop_ticks(candidates, max_ticks=cap)
    executed: list[dict[str, Any]] = []
    pauses = list(simulation.get("would_pause") or [])
    skipped = []
    for item in simulation.get("would_skip") or []:
        reason_codes = item.get("reason_codes") or []
        if reason_codes == ["max_ticks_cap_reached"]:
            item = {**item, "reason_codes": ["max_tasks_per_run_reached"], "human_reason": f"bounded run cap reached: max_tasks={cap}"}
        skipped.append(item)
    closeouts = closeout_results or []
    for idx, selected in enumerate(simulation.get("would_select") or []):
        if idx >= cap:
            skipped.append({**selected, "status": "skipped", "reason_codes": ["max_tasks_per_run_reached"]})
            continue
        executed.append({**selected, "tick": idx + 1, "handoff": "check_only"})
        if idx < len(closeouts) and closeouts[idx].get("may_continue") is False:
            pauses.append({"reason_code": "closeout_blocked", "human_reason": ",".join(closeouts[idx].get("reason_codes") or [])})
            return {
                "mode": "bounded_multi_tick_simulation",
                "executed": executed,
                "skipped": skipped,
                "would_pause": pauses,
                "next_state": "needs_human",
                "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            }
    if not executed and not pauses:
        pauses.append({"reason_code": "no_progress", "human_reason": "No eligible work executed in bounded run."})
    return {
        "mode": "bounded_multi_tick_simulation",
        "executed": executed,
        "skipped": skipped,
        "would_pause": pauses,
        "next_state": "paused" if executed else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


def _candidate_with_ready_promotion(candidate: dict[str, Any], *, default_assignee: str = "arisu") -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return a ready-shaped candidate when promotion is safe.

    Promotion is intentionally narrow: it may repair Kanban readiness fields
    (status/assignee) only when the candidate already carries a complete
    gate-visible execution contract. It does not guess missing requirements.
    """

    public_id = candidate.get("public_id") or candidate.get("task_id") or candidate.get("id")
    if str(candidate.get("relation_type") or "").lower() == "dependency":
        return None, {
            "public_id": public_id,
            "task_id": candidate.get("task_id") or candidate.get("id"),
            "status": "blocked",
            "reason_codes": ["dependency_relation_blocks_ready_promotion"],
            "human_reason": "dependency links gate execution; hierarchy children only may be auto-promoted",
        }
    if str(candidate.get("status") or "").lower() in {"archived", "done", "running", "claimed", "in_progress"}:
        return None, {
            "public_id": public_id,
            "task_id": candidate.get("task_id") or candidate.get("id"),
            "status": "blocked",
            "reason_codes": ["non_promotable_lifecycle_state"],
            "human_reason": "only backlog children may be promoted to ready",
        }

    trial = {**candidate, "status": "ready", "assignee": candidate.get("assignee") or default_assignee}
    gate = evaluate_autopilot_ready_gate(trial)
    if not gate.get("autopilot_ready"):
        return None, {
            "public_id": public_id,
            "task_id": candidate.get("task_id") or candidate.get("id"),
            "status": "blocked",
            "reason_codes": gate.get("reason_codes") or [],
            "human_reason": "child contract is not safe to promote: " + ", ".join(gate.get("reason_codes") or []),
            "ready_gate": gate,
        }
    return trial, {
        "public_id": public_id,
        "task_id": candidate.get("task_id") or candidate.get("id"),
        "status": "promoted",
        "reason_codes": [],
        "assignee": trial.get("assignee"),
        "ready_gate": gate,
        "human_reason": "child promoted to autopilot-ready contract for existing dispatcher handoff",
    }


def _apply_ready_promotion_to_kanban(candidate: dict[str, Any], *, assignee: str) -> dict[str, Any]:
    task_id = str(candidate.get("task_id") or candidate.get("id") or "").strip()
    if not task_id:
        return {"applied": False, "reason": "missing_task_id"}
    try:
        from hermes_cli import kanban_db as kb
    except Exception as exc:
        return {"applied": False, "reason": "kanban_db_unavailable", "error": str(exc)}
    try:
        with kb.connect() as conn:
            conn.execute("UPDATE tasks SET status = 'ready', assignee = ? WHERE id = ?", (assignee, task_id))
            conn.commit()
    except Exception as exc:
        return {"applied": False, "reason": "kanban_update_failed", "error": str(exc)}
    return {"applied": True, "task_id": task_id, "status": "ready", "assignee": assignee}


def promote_parent_scoped_children(
    candidates: list[dict[str, Any]],
    *,
    parent_public_id: Optional[str] = None,
    dry_run: bool = True,
    default_assignee: str = "arisu",
    apply_to_kanban: bool = False,
) -> dict[str, Any]:
    """Prepare in-scope parent children for Autopilot before dispatcher handoff.

    This is the parent-level missing loop: non-ready hierarchy children that
    already contain a complete executable contract are promoted to
    ``status=ready`` with a spawnable worker profile. Ambiguous children stay
    blocked/non-ready with explicit reason codes instead of being guessed into
    execution. Actual worker execution remains owned by the existing dispatcher.
    """

    parent = str(parent_public_id or "").strip().upper()
    prepared: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    would_promote: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    out_of_scope: list[dict[str, Any]] = []
    apply_results: list[dict[str, Any]] = []

    for candidate in candidates:
        cand_parent = str(candidate.get("parent_public_id") or "").strip().upper()
        public_id = candidate.get("public_id") or candidate.get("task_id") or candidate.get("id")
        if parent and cand_parent and cand_parent != parent:
            out_of_scope.append({"public_id": public_id, "task_id": candidate.get("task_id") or candidate.get("id"), "reason_codes": ["parent_scope_mismatch"]})
            continue
        if evaluate_autopilot_ready_gate(candidate).get("autopilot_ready"):
            prepared.append(candidate)
            continue
        ready_candidate, disposition = _candidate_with_ready_promotion(candidate, default_assignee=default_assignee)
        if ready_candidate is None:
            blocked.append(disposition)
            prepared.append(candidate)
            continue
        prepared.append(ready_candidate)
        if dry_run:
            would_promote.append(disposition)
            continue
        promoted.append(disposition)
        if apply_to_kanban:
            apply_results.append(_apply_ready_promotion_to_kanban(ready_candidate, assignee=str(ready_candidate.get("assignee") or default_assignee)))

    mutated = 0 if dry_run else len(promoted)
    return {
        "parent_public_id": parent or None,
        "candidates": prepared,
        "promoted": promoted,
        "would_promote": would_promote,
        "blocked": blocked,
        "out_of_scope": out_of_scope,
        "apply_results": apply_results,
        "handoff_target": "existing_kanban_dispatcher",
        "second_dispatcher_created": False,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": mutated, "dispatched": 0},
    }


def filter_candidates_for_scope(candidates: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, Any]:
    """Filter candidates to an explicit parent/lane/repo/label scope."""

    in_scope: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    required_labels = set(scope.get("labels") or [])
    for candidate in candidates:
        reasons: list[str] = []
        if scope.get("parent_public_id") and candidate.get("parent_public_id") != scope.get("parent_public_id"):
            reasons.append("parent_scope_mismatch")
        if scope.get("tenant") and candidate.get("tenant") != scope.get("tenant"):
            reasons.append("lane_scope_mismatch")
        if scope.get("repo_full_name") and candidate.get("repo_full_name") != scope.get("repo_full_name"):
            reasons.append("repo_scope_mismatch")
        if required_labels and not required_labels.issubset(set(candidate.get("labels") or [])):
            reasons.append("label_scope_mismatch")
        if candidate.get("relation_type") == "dependency":
            reasons.append("hierarchy_dependency_ambiguous")
        if reasons:
            out.append({"public_id": candidate.get("public_id"), "task_id": candidate.get("id"), "reason_codes": reasons})
        else:
            in_scope.append(candidate)
    return {
        "in_scope": in_scope,
        "out_of_scope": out,
        "scope_can_silently_widen": False,
        "next_state": "continue" if in_scope else "needs_human",
    }


def generate_autopilot_run_report(run: dict[str, Any]) -> dict[str, Any]:
    """Generate a compact operator-readable Autopilot run report."""

    executed = run.get("executed") or run.get("would_select") or []
    skipped = run.get("skipped") or run.get("would_skip") or []
    blocked = run.get("would_pause") or []
    open_prs = run.get("open_prs") or []
    next_state = run.get("next_state") or "unknown"
    lines = ["Autopilot run report", f"next_state={next_state}"]
    if executed:
        lines.append("executed=" + ",".join(str(item.get("public_id") or item.get("task_id")) for item in executed))
    else:
        lines.append("zero work executed")
    if skipped:
        lines.append("skipped=" + ",".join(f"{item.get('public_id') or item.get('task_id')}:{'/'.join(item.get('reason_codes') or [])}" for item in skipped))
    if blocked:
        lines.append("blocked=" + ",".join(str(item.get("reason_code")) for item in blocked))
    if open_prs:
        lines.append("open_prs=" + ",".join(open_prs))
    if run.get("caps"):
        lines.append("caps=" + json.dumps(run.get("caps"), sort_keys=True))
    return {
        "summary": {
            "executed_count": len(executed),
            "skipped_count": len(skipped),
            "blocked_count": len(blocked),
            "open_pr_count": len(open_prs),
            "next_state": next_state,
            "zero_work": len(executed) == 0,
        },
        "text": "\n".join(lines),
    }


def generate_verifier_failure_operator_report(verifier_result: dict[str, Any], remediation_state: dict[str, Any]) -> dict[str, Any]:
    """Summarize verifier failures and remediation state for operators.

    BO-152 makes failed verifier outcomes readable without granting recovery
    authority.  The report is intentionally deterministic and check-only: it
    names missing criteria, retry/remediation disposition, next owner, and the
    authority boundary so an operator can see why review_ready is blocked.
    """

    verdict = str(verifier_result.get("verdict") or "UNKNOWN").upper()
    missing = list(verifier_result.get("missing_criteria") or verifier_result.get("reason_codes") or [])
    retry_count = int(remediation_state.get("retry_count") or 0)
    max_retries = int(remediation_state.get("max_retries") or 0)
    remediation_child = remediation_state.get("remediation_child")
    retry_allowed = bool(remediation_state.get("retry_allowed")) and retry_count < max_retries
    needs_human = verdict in {"BLOCKED", "REFINEMENT_REQUIRED"} or (verdict == "FAIL" and not retry_allowed and not remediation_child)
    if verdict == "PASS":
        next_state = "review_ready_gate"
    elif retry_allowed:
        next_state = "retry_queued"
    elif remediation_child:
        next_state = "remediation_child_queued"
    else:
        next_state = "needs_human"
    lines = [
        "Autopilot verifier report",
        f"verdict={verdict}",
        f"next_state={next_state}",
    ]
    if missing:
        lines.append("missing=" + ",".join(str(item) for item in missing))
    lines.append(f"retry={retry_count}/{max_retries}")
    if remediation_child:
        lines.append(f"remediation_child={remediation_child}")
    if remediation_state.get("blocked_reason"):
        lines.append(f"blocked_reason={remediation_state.get('blocked_reason')}")
    return {
        "summary": {
            "verdict": verdict,
            "missing_criteria_count": len(missing),
            "retry_count": retry_count,
            "max_retries": max_retries,
            "retry_allowed": retry_allowed,
            "needs_human": needs_human,
            "next_state": next_state,
        },
        "missing_criteria": missing,
        "remediation": {
            "retry_allowed": retry_allowed,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "remediation_child": remediation_child,
            "blocked_reason": remediation_state.get("blocked_reason"),
        },
        "authority": {
            "review_ready_allowed": verdict == "PASS",
            "merge_allowed": False,
            "release_allowed": False,
            "gateway_restart_reload_allowed": False,
            "config_env_secret_mutation_allowed": False,
        },
        "text": "\n".join(lines),
    }



def build_autopilot_review_package(
    evidence: dict[str, Any],
    *,
    run_report: Optional[dict[str, Any]] = None,
    expected_repo_full_name: str = "chriskim12/hermes-agent",
) -> dict[str, Any]:
    """Build a review-ready package proof without granting live authority.

    BO-118 packages worker evidence, PR/check truth, and operator run facts in
    one machine-readable shape.  It may prove review readiness, but it never
    grants merge, release, deploy, gateway restart/reload, or worker-completion
    authority from handoff alone.
    """

    closeout = evaluate_autopilot_closeout_progress(evidence)
    review_contract = evaluate_review_ready_contract(evidence, expected_repo_full_name=expected_repo_full_name)
    report = generate_autopilot_run_report(run_report or {})
    reason_codes: list[str] = []
    for code in closeout.get("reason_codes") or []:
        if code not in reason_codes:
            reason_codes.append(code)
    for code in review_contract.get("reason_codes") or []:
        if code not in reason_codes:
            reason_codes.append(code)
    review_ready = bool(closeout.get("may_continue") and review_contract.get("review_ready"))
    pr_url = str(evidence.get("pr_url") or "").strip()
    text_lines = [
        "Autopilot review-ready PR package" if review_ready else "Autopilot review package blocked",
        f"work_id={evidence.get('work_id') or 'unknown'}",
        f"commit={evidence.get('commit') or evidence.get('head_sha') or 'missing'}",
        f"branch={evidence.get('task_branch') or evidence.get('branch') or 'missing'}",
        f"pr={pr_url or 'missing'}",
        f"run_next_state={report['summary']['next_state']}",
        "merge/release/deploy remains forbidden without current-turn approval.",
        "handoff_success_is_worker_completion=False",
    ]
    return {
        "status": "review_package_ready" if review_ready else "review_package_blocked",
        "review_ready": review_ready,
        "reason_codes": reason_codes,
        "work_id": evidence.get("work_id"),
        "commit": evidence.get("commit") or evidence.get("head_sha"),
        "task_branch": evidence.get("task_branch") or evidence.get("branch"),
        "pr": {
            "url": pr_url,
            "base": evidence.get("pr_base"),
            "head": evidence.get("pr_head"),
            "checks_passed": evidence.get("checks_passed") is True,
        },
        "run_report": report,
        "closeout_progress": closeout,
        "review_contract": review_contract,
        "authority": {
            "ceiling": "review_ready_pr",
            "merge_allowed": False,
            "release_allowed": False,
            "deploy_allowed": False,
            "gateway_restart_reload_allowed": False,
            "prod_customer_visible_allowed": False,
            "config_env_secret_mutation_allowed": False,
        },
        "worker_done_observed": evidence.get("kanban_worker_done") is True,
        "handoff_success_is_worker_completion": False,
        "next_state": "review_ready" if review_ready else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "text": "\n".join(text_lines),
    }


def build_worktree_cleanup_registry(entries: list[dict[str, Any]], *, bundle_id: Optional[str] = None) -> dict[str, Any]:
    """Normalize task-owned worktree cleanup state without deleting anything.

    BO-151 keeps cleanup accounting explicit: registered git worktrees need
    review-safe disposition before ``review_ready`` and a separate post-merge
    reconcile ledger after Chris accepts/merges the PR stack.  This helper is
    pure/check-only; it records which entries would block instead of attempting
    ``git worktree remove`` or branch pruning itself.
    """

    normalized: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for raw in entries:
        entry = dict(raw)
        public_id = entry.get("public_id") or entry.get("card_id") or entry.get("task_id")
        cleanup_state = str(entry.get("cleanup_state") or "pending").strip().lower()
        reason_codes: list[str] = []
        if not str(entry.get("path") or "").strip():
            reason_codes.append("missing_worktree_path")
        if entry.get("registered_git_worktree") is not True:
            reason_codes.append("registered_git_worktree_not_verified")
        if cleanup_state == "removed":
            if entry.get("git_worktree_remove_verified") is not True:
                reason_codes.append("missing_git_worktree_remove_proof")
        elif cleanup_state == "retained":
            if not str(entry.get("retained_reason") or "").strip():
                reason_codes.append("missing_retained_reason")
            if not str(entry.get("ttl") or entry.get("revisit_at") or "").strip():
                reason_codes.append("missing_retained_ttl")
            if entry.get("review_safe") is not True:
                reason_codes.append("retained_residue_not_review_safe")
        else:
            reason_codes.append("cleanup_pending")
        if entry.get("active_process_cwd") or entry.get("active_tmux_cwd") or entry.get("active_worker_pid"):
            reason_codes.append("active_reference_blocks_cleanup")
        status = "ok" if not reason_codes else "blocked"
        item = {**entry, "public_id": public_id, "cleanup_state": cleanup_state, "status": status, "reason_codes": reason_codes}
        normalized.append(item)
        if reason_codes:
            blockers.append({"public_id": public_id, "path": entry.get("path"), "reason_codes": reason_codes})
    return {
        "bundle_id": bundle_id,
        "entries": normalized,
        "blocked_entries": blockers,
        "cleanup_required": bool(blockers),
        "review_ready_allowed": not blockers,
        "destructive_cleanup_performed": False,
        "required_apply_primitive": "git_worktree_remove_then_prune_verify",
    }


def evaluate_pre_review_cleanup_gate(registry: dict[str, Any]) -> dict[str, Any]:
    """Gate review_ready on removed or safely-retained task worktrees."""

    blockers = list(registry.get("blocked_entries") or [])
    reason_codes = ["cleanup_required"] if blockers else []
    for blocker in blockers:
        for code in blocker.get("reason_codes") or []:
            if code not in reason_codes:
                reason_codes.append(code)
    return {
        "review_ready_allowed": not blockers,
        "cleanup_required": bool(blockers),
        "reason_codes": reason_codes,
        "blocked_entries": blockers,
        "post_merge_reconcile_still_required": True,
        "destructive_cleanup_performed": False,
    }


def reconcile_post_merge_cleanup(registry: dict[str, Any], *, merged_prs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a post-merge cleanup reconcile ledger from existing proof.

    The reconcile remains check-only here: it verifies that every retained or
    removed worktree entry has corresponding merged/accepted PR evidence and no
    stale cleanup blockers.  Runtime materialization and gateway restart remain
    outside this authority ceiling.
    """

    merged_by_public_id = {str(pr.get("public_id") or pr.get("work_id") or ""): pr for pr in merged_prs}
    entries = registry.get("entries") or []
    stale: list[dict[str, Any]] = []
    reconciled: list[dict[str, Any]] = []
    for entry in entries:
        public_id = str(entry.get("public_id") or "")
        pr = merged_by_public_id.get(public_id)
        reasons: list[str] = []
        if not pr or pr.get("state") != "MERGED":
            reasons.append("missing_merged_pr_truth")
        if entry.get("status") != "ok":
            reasons.append("pre_review_cleanup_not_resolved")
        item = {"public_id": public_id, "path": entry.get("path"), "merged_pr": pr, "reason_codes": reasons, "status": "reconciled" if not reasons else "blocked"}
        reconciled.append(item)
        if reasons:
            stale.append(item)
    return {
        "reconciled": reconciled,
        "stale_entries": stale,
        "closed_allowed": not stale,
        "gateway_restart_reload_allowed": False,
        "canonical_materialization_allowed": False,
        "destructive_cleanup_performed": False,
    }



def prove_worker_verifier_retry_loop_check_only(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Check an end-to-end worker→verifier→retry/review_ready event trace.

    BO-153's proof is deliberately check-only: it accepts fixture events and
    verifies that a completed worker event leads either to verifier FAIL with a
    retry/remediation disposition, or verifier PASS with cleanup-checked
    review_ready.  It records forbidden authority as false and performs no
    Kanban, GitHub, cleanup, restart, or provider mutation.
    """

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(str(event.get("kind") or ""), []).append(event)
    reason_codes: list[str] = []
    if not by_kind.get("worker_completed"):
        reason_codes.append("missing_worker_completed_event")
    if not by_kind.get("verifier_intake"):
        reason_codes.append("missing_verifier_intake_event")
    verifier_events = by_kind.get("verifier_result") or []
    if not verifier_events:
        reason_codes.append("missing_verifier_result_event")
    outcomes: list[dict[str, Any]] = []
    for event in verifier_events:
        verdict = str(event.get("verdict") or "").upper()
        if verdict == "FAIL":
            retry = bool(by_kind.get("retry_queued") or by_kind.get("remediation_child_queued"))
            if not retry:
                reason_codes.append("verifier_fail_without_retry_or_remediation")
            outcomes.append({"verdict": verdict, "next_state": "retry_or_remediation" if retry else "needs_human"})
        elif verdict == "PASS":
            cleanup = bool(by_kind.get("cleanup_checked"))
            review_ready = bool(by_kind.get("review_ready_promoted"))
            if not cleanup:
                reason_codes.append("verifier_pass_without_cleanup_check")
            if not review_ready:
                reason_codes.append("verifier_pass_without_review_ready_promotion")
            outcomes.append({"verdict": verdict, "next_state": "review_ready" if cleanup and review_ready else "blocked"})
        else:
            reason_codes.append("unknown_verifier_verdict")
            outcomes.append({"verdict": verdict or "UNKNOWN", "next_state": "needs_human"})
    forbidden_attempts = [event for event in events if str(event.get("kind") or "") == "forbidden_mutation_attempt"]
    if forbidden_attempts and not by_kind.get("authority_blocked"):
        reason_codes.append("forbidden_attempt_not_blocked")
    parent_matrix_updated = bool(by_kind.get("parent_matrix_updated"))
    if not parent_matrix_updated:
        reason_codes.append("missing_parent_matrix_update")
    return {
        "passed": not reason_codes,
        "reason_codes": reason_codes,
        "outcomes": outcomes,
        "observed": {kind: len(items) for kind, items in sorted(by_kind.items())},
        "parent_matrix_updated": parent_matrix_updated,
        "authority": {
            "check_only": True,
            "merge_allowed": False,
            "release_allowed": False,
            "gateway_restart_reload_allowed": False,
            "config_env_secret_mutation_allowed": False,
            "prod_customer_visible_allowed": False,
        },
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "next_state": "proof_passed" if not reason_codes else "needs_human",
    }



def evaluate_autopilot_promotion_policy(proof: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether first live pickup proof may promote to bounded check-only mode.

    BO-119 intentionally promotes only the *policy state* to a narrow,
    parent-scoped, check-only bounded mode. It does not grant live dispatch,
    worker claim/spawn, PR/push, merge, gateway restart/reload, or config/env
    mutation authority. Those remain current-turn approval gates.
    """

    reason_codes: list[str] = []
    parent_public_id = str(proof.get("parent_public_id") or "").strip()
    if not parent_public_id:
        reason_codes.append("missing_parent_scope")
    if proof.get("live_pickup_smoke_passed") is not True:
        reason_codes.append("live_pickup_smoke_missing")
    if proof.get("single_flight_guard_passed") is not True:
        reason_codes.append("single_flight_guard_missing")
    if proof.get("review_package_ready") is not True:
        reason_codes.append("review_package_not_ready")
    worker_done_children = list(proof.get("kanban_worker_done_children") or [])
    if len(worker_done_children) < 3:
        reason_codes.append("insufficient_worker_done_child_proofs")
    if int(proof.get("active_flights") or 0) > 0:
        reason_codes.append("active_flight_present")
    max_open = int(proof.get("max_open_autopilot_prs") or _DEFAULT_CLOSED_LOOP_CAPS["max_open_autopilot_prs"])
    if int(proof.get("open_autopilot_prs") or 0) >= max_open:
        reason_codes.append("pr_backlog_cap_reached")
    requested_mode = str(proof.get("requested_mode") or "bounded_multi_tick")
    allowed_requested_modes = {"bounded_multi_tick", "parent_scoped"}
    if requested_mode not in allowed_requested_modes:
        reason_codes.append("requested_mode_not_allowed")
    if proof.get("request_live_dispatch") is True:
        reason_codes.append("live_dispatch_requires_current_turn_approval")
    if proof.get("request_gateway_restart_reload") is True:
        reason_codes.append("gateway_restart_reload_requires_current_turn_approval")
    if proof.get("request_config_env_secret_mutation") is True:
        reason_codes.append("config_env_secret_mutation_requires_current_turn_approval")
    promotion_allowed = not reason_codes
    authority = {
        "ceiling": "review_ready_pr",
        "worker_dispatch_claim_spawn_allowed": False,
        "push_pr_allowed": False,
        "merge_allowed": False,
        "release_allowed": False,
        "deploy_allowed": False,
        "gateway_restart_reload_allowed": False,
        "prod_customer_visible_allowed": False,
        "config_env_secret_mutation_allowed": False,
    }
    caps = {
        "max_tasks_per_run": _DEFAULT_CLOSED_LOOP_CAPS["max_tasks_per_run_early_bounded_multi_tick"],
        "max_active_flights": _DEFAULT_CLOSED_LOOP_CAPS["max_active_flights"],
        "max_dispatches_per_tick": _DEFAULT_CLOSED_LOOP_CAPS["max_dispatches_per_tick"],
        "max_new_prs_per_run": _DEFAULT_CLOSED_LOOP_CAPS["max_new_prs_per_run"],
        "max_consecutive_failures": _DEFAULT_CLOSED_LOOP_CAPS["max_consecutive_failures"],
        "require_review_ready_contract_before_next_task": True,
    }
    return {
        "promotion_allowed": promotion_allowed,
        "promoted_mode": "bounded_multi_tick_check_only" if promotion_allowed else "blocked",
        "reason_codes": reason_codes,
        "scope": {
            "parent_public_id": parent_public_id or None,
            "scope_can_silently_widen": False,
        },
        "caps": caps,
        "authority": authority,
        "requires_current_turn_approval_for_live_dispatch": True,
        "requires_current_turn_approval_for_push_pr": True,
        "requires_current_turn_approval_for_gateway_restart_reload": True,
        "handoff_success_is_worker_completion": False,
        "worker_done_truth_source": "kanban_dispatcher_worker_done_evidence",
        "next_state": "promote_check_only" if promotion_allowed else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


def harden_autopilot_policy(contract: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed policy hardening gate for closed-loop Autopilot."""

    validation = validate_closed_loop_policy_contract(contract)
    accepted = bool(validation.get("ok"))
    return {
        "accepted": accepted,
        "reason_codes": validation.get("reason_codes") or [],
        "next_state": "continue" if accepted else "hard_stopped",
        "recovery_required": not accepted,
        "merge_allowed": False,
        "release_allowed": False,
        "prod_customer_visible_allowed": False,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


_RECOVERY_ACTIONS = {
    "stale_kanban_state": "pause_and_reread_kanban",
    "kanban_read_unavailable": "pause_and_reread_kanban",
    "worker_crash_or_timeout_repeated": "pause_and_require_worker_evidence",
    "worker_completion_evidence_missing": "pause_and_require_worker_evidence",
    "policy_file_invalid_or_stale": "hard_stop_and_require_recovery_ack",
    "forbidden_action_requested": "hard_stop_and_require_recovery_ack",
    "scope_ambiguity": "pause_and_require_scope_confirmation",
    "budget_or_cap_exceeded": "pause_and_report_cap",
    "pr_backlog_cap_exceeded": "pause_and_report_pr_backlog",
}


def run_autopilot_recovery_drill(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Exercise fail-closed recovery decisions without executing side effects."""

    drills: list[dict[str, Any]] = []
    passed = True
    for scenario in scenarios:
        trigger = str(scenario.get("trigger") or "")
        action = _RECOVERY_ACTIONS.get(trigger)
        if action is None:
            passed = False
            action = "hard_stop_and_require_human_triage"
        drills.append({
            "name": scenario.get("name"),
            "trigger": trigger,
            "action": action,
            "side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        })
    return {
        "passed": passed,
        "drills": drills,
        "next_state": "recovered" if passed else "needs_human",
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
    }


@dataclass(frozen=True)
class AutopilotCommand:
    """Parsed /autopilot command shape."""

    action: str
    raw_args: str = ""
    value: str = ""

    @property
    def read_only(self) -> bool:
        return self.action in _READ_ONLY_ACTIONS


@dataclass(frozen=True)
class AutopilotResult:
    """Structured decision plus gateway/CLI friendly message."""

    ok: bool
    command: Optional[AutopilotCommand]
    message: str
    decision: dict[str, Any]
    fail_closed: bool = False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> Path:
    return get_hermes_home() / AUTOPILOT_STATE_FILE


def _read_state(path: Optional[Path] = None) -> dict[str, Any]:
    state_path = path or _state_path()
    try:
        raw = state_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"exists": False, "enabled": False, "desired_mode": "disabled", "path": str(state_path)}
    except OSError as exc:
        return {
            "exists": False,
            "enabled": False,
            "desired_mode": "disabled",
            "path": str(state_path),
            "read_error": str(exc),
        }
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "exists": True,
            "enabled": False,
            "desired_mode": "disabled",
            "path": str(state_path),
            "read_error": f"invalid_json:{exc.msg}",
        }
    if not isinstance(data, dict):
        return {
            "exists": True,
            "enabled": False,
            "desired_mode": "disabled",
            "path": str(state_path),
            "read_error": "state_not_object",
        }
    enabled = bool(data.get("enabled"))
    desired = str(data.get("desired_mode") or ("enabled" if enabled else "disabled")).strip().lower()
    return {**data, "exists": True, "enabled": enabled, "desired_mode": desired, "path": str(state_path)}


def _write_state(state: dict[str, Any], path: Optional[Path] = None) -> dict[str, Any]:
    state_path = path or _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _read_state(state_path)


def _json_object_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def _task_row_to_candidate(row: Any, *, parent_public_id: Optional[str] = None, relation_type: Optional[str] = None) -> dict[str, Any]:
    routing = _json_object_maybe(row["routing_verdict"] if "routing_verdict" in row.keys() else None)
    admission = _json_object_maybe(row["admission_snapshot"] if "admission_snapshot" in row.keys() else None)
    closeout = _json_object_maybe(row["closeout_evidence"] if "closeout_evidence" in row.keys() else None)
    skills: list[str] = []
    if "skills" in row.keys() and row["skills"]:
        try:
            parsed_skills = json.loads(row["skills"])
            if isinstance(parsed_skills, list):
                skills = [str(skill) for skill in parsed_skills if skill]
        except json.JSONDecodeError:
            skills = []
    candidate = {
        "id": row["id"],
        "task_id": row["id"],
        "public_id": row["public_id"] if "public_id" in row.keys() else None,
        "title": row["title"],
        "body": row["body"],
        "status": row["status"],
        "assignee": row["assignee"],
        "tenant": row["tenant"] if "tenant" in row.keys() else None,
        "priority": row["priority"],
        "routing_verdict": routing,
        "admission_snapshot": admission,
        "closeout_evidence": closeout,
        "parent_public_id": parent_public_id,
        "relation_type": relation_type,
        "skills": skills,
        "claim_lock": row["claim_lock"],
        "worker_pid": row["worker_pid"] if "worker_pid" in row.keys() else None,
        "current_run_id": row["current_run_id"] if "current_run_id" in row.keys() else None,
    }
    repo = admission.get("repo_full_name") or (admission.get("repo_intent") or {}).get("repo_full_name")
    if repo:
        candidate["repo_full_name"] = repo
    return candidate


def load_live_kanban_candidates(
    *,
    parent_public_id: Optional[str] = None,
    tenant: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read live Kanban candidates for Autopilot without mutating the board.

    If a parent/focus public id is supplied, only hierarchy children of that
    parent are returned. Without an explicit parent this returns ready tasks plus
    active flight rows so the global/default-policy single-flight guard can see
    an already-running worker before dispatching another task.
    """

    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return []
    with kb.connect() as conn:
        if parent_public_id:
            parent = conn.execute(
                "SELECT id, public_id FROM tasks WHERE public_id = ? OR id = ? LIMIT 1",
                (parent_public_id, parent_public_id),
            ).fetchone()
            if not parent:
                return []
            params: list[Any] = [parent["id"]]
            query = """
                SELECT c.*, l.relation_type AS relation_type, p.public_id AS parent_public_id
                FROM task_links l
                JOIN tasks p ON p.id = l.parent_id
                JOIN tasks c ON c.id = l.child_id
                WHERE l.parent_id = ? AND l.relation_type = 'hierarchy' AND c.status != 'archived'
            """
            if tenant:
                query += " AND c.tenant = ?"
                params.append(tenant)
            query += " ORDER BY c.priority DESC, c.created_at ASC LIMIT ?"
            params.append(max(1, int(limit)))
            rows = conn.execute(query, params).fetchall()
            return [
                _task_row_to_candidate(row, parent_public_id=parent["public_id"] or parent_public_id, relation_type=row["relation_type"])
                for row in rows
            ]
        tenant_clause = " AND tenant = ?" if tenant else ""
        tenant_params: list[Any] = [tenant] if tenant else []
        active_query = f"""
            SELECT * FROM tasks
            WHERE status != 'archived'
              AND (
                status IN ('running', 'claimed', 'in_progress')
                OR claim_lock IS NOT NULL
                OR worker_pid IS NOT NULL
                OR current_run_id IS NOT NULL
              )
              {tenant_clause}
            ORDER BY priority DESC, created_at ASC
        """
        ready_query = f"""
            SELECT * FROM tasks
            WHERE status = 'ready'
              {tenant_clause}
            ORDER BY priority DESC, created_at ASC LIMIT ?
        """
        rows = list(conn.execute(active_query, tenant_params).fetchall())
        seen_ids = {row["id"] for row in rows}
        ready_rows = conn.execute(ready_query, [*tenant_params, max(1, int(limit))]).fetchall()
        rows.extend(row for row in ready_rows if row["id"] not in seen_ids)
        return [_task_row_to_candidate(row) for row in rows]


def _candidate_scope_for(command: AutopilotCommand) -> dict[str, Any]:
    state = _read_state()
    raw_scope = str(command.value or state.get("focus") or "").strip()
    scope: dict[str, Any] = {"mode": "default_policy", "parent_public_id": None, "tenant": None}
    if raw_scope:
        scope["mode"] = "parent"
        scope["parent_public_id"] = raw_scope.upper()
    return scope


def _filter_candidates_for_scope(candidates: list[dict[str, Any]], scope: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply explicit focus narrowing to injected candidate lists.

    Live Kanban loads can filter in SQL.  Unit tests and gateway call sites may
    pass a preloaded candidate list; parent-scoped mode must still not escape
    that parent when candidates are injected.
    """

    parent_public_id = str(scope.get("parent_public_id") or "").strip().upper()
    if not parent_public_id:
        return candidates
    # Some call sites pass an already-scoped candidate list that predates the
    # explicit parent_public_id field.  Only enforce injected-list filtering
    # when the list carries parent metadata to compare against.
    if not any(candidate.get("parent_public_id") for candidate in candidates):
        return candidates
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("parent_public_id") or "").strip().upper() == parent_public_id
    ]


def _resolve_candidates(command: AutopilotCommand, candidates: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    scope = _candidate_scope_for(command)
    if candidates is not None:
        return _filter_candidates_for_scope(candidates, scope)
    return load_live_kanban_candidates(
        parent_public_id=scope.get("parent_public_id"),
        tenant=scope.get("tenant"),
        limit=50,
    )


def parse_autopilot_args(raw_args: str) -> AutopilotCommand:
    """Parse the deliberately narrow Phase-1 command surface."""

    raw = str(raw_args or "").strip()
    if not raw:
        return AutopilotCommand(action="status", raw_args="")
    parts = raw.split(maxsplit=1)
    action = parts[0].strip().lower().replace("_", "-")
    value = parts[1].strip() if len(parts) > 1 else ""
    if action == "dry_run":
        action = "dry-run"
    if action in {"status", "dry-run", "queue", "on", "off", "stop", "pause", "pause-lane", "resume-lane", "hard-stop", "recover", "focus", "once"}:
        return AutopilotCommand(action=action, raw_args=raw, value=value)
    raise ValueError(f"unsupported /autopilot command: {raw_args!r}; usage: {AUTOPILOT_USAGE}")


def _effective_mode(desired_mode: str, state: dict[str, Any]) -> str:
    desired = str(desired_mode or "disabled").lower()
    if desired == "hard_stopped":
        return "hard_stop"
    if desired in {"stopped", "off", "disabled"}:
        return "stopped" if desired == "stopped" else "degraded"
    if desired == "paused":
        return "paused"
    # Focused `/autopilot on <parent>` narrows the executable controller loop;
    # unscoped `/autopilot on` is also executable, but only through the
    # default-policy selector.  Raw `ready` alone is still insufficient.
    if desired == "on":
        return "parent_scoped" if state.get("focus") else "default_policy_loop"
    if desired == "enabled":
        return "parent_scoped" if state.get("focus") else "blocked"
    if state.get("read_error"):
        return "degraded"
    return "degraded"


def evaluate_autopilot_ready_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a dry-run Ready-gate verdict for one Kanban candidate.

    This deliberately treats raw Kanban ``status=ready`` as insufficient.  The
    candidate must carry an executable contract before a later autopilot phase
    may even consider selection.  The function is pure/read-only: no dispatcher,
    claim, spawn, or Kanban mutation calls are made here.
    """

    body = str(candidate.get("body") or "")
    title = str(candidate.get("title") or "")
    haystack = "\n".join([title, body, json.dumps(candidate, ensure_ascii=False, sort_keys=True)]).lower()
    reason_codes: list[str] = []
    missing_labels: list[str] = []
    if str(candidate.get("status") or "").lower() != "ready":
        reason_codes.append("kanban_status_not_ready")
        missing_labels.append("Kanban status=ready")
    for code, markers, label in _READY_GATE_REQUIREMENTS:
        if not any(marker in haystack for marker in markers):
            reason_codes.append(code)
            missing_labels.append(label)
    done_criteria = build_done_criteria_ledger(body)
    if not done_criteria.get("ok"):
        reason_codes.extend(done_criteria.get("reason_codes") or ["invalid_done_criteria_ledger"])
        missing_labels.append("explicit done criteria ledger")
    routing = candidate.get("routing_verdict") or {}
    if isinstance(routing, str):
        try:
            routing = json.loads(routing)
        except json.JSONDecodeError:
            routing = {"raw": routing}
    verdict = str((routing or {}).get("verdict") or "").strip().lower()
    if not verdict:
        reason_codes.append("missing_routing_verdict")
        missing_labels.append("routing verdict")
    accepted = not reason_codes
    return {
        "task_id": candidate.get("id"),
        "public_id": candidate.get("public_id"),
        "autopilot_ready": accepted,
        "status": "accepted" if accepted else "rejected",
        "reason_codes": reason_codes,
        "human_reason": "ready for autopilot dry-run selection" if accepted else "Missing executable contract fields: " + ", ".join(missing_labels),
        "done_criteria_ledger": done_criteria.get("done_criteria_ledger"),
        "criteria_hash": done_criteria.get("criteria_hash"),
        "criteria_version": done_criteria.get("criteria_version"),
        "criteria_ids": done_criteria.get("criteria_ids") or [],
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
    }


def evaluate_review_ready_contract(
    evidence: dict[str, Any],
    *,
    expected_repo_full_name: str = "chriskim12/hermes-agent",
) -> dict[str, Any]:
    """Validate the review-ready PR package without allowing merge/release."""

    reason_codes: list[str] = []
    required = {
        "repo_full_name": evidence.get("repo_full_name"),
        "commit": evidence.get("commit") or evidence.get("head_sha"),
        "task_branch": evidence.get("task_branch") or evidence.get("branch"),
        "pr_url": evidence.get("pr_url"),
        "pr_base": evidence.get("pr_base"),
        "pr_head": evidence.get("pr_head"),
    }
    for field, value in required.items():
        if not str(value or "").strip():
            reason_codes.append(f"missing_{field}")
    bool_required = {
        "checks_passed": evidence.get("checks_passed"),
        "worktree_clean": evidence.get("worktree_clean"),
        "kanban_worker_done": evidence.get("kanban_worker_done"),
        "boundaries_confirmed": evidence.get("boundaries_confirmed"),
    }
    for field, value in bool_required.items():
        if value is not True:
            reason_codes.append(f"missing_{field}")
    done_criteria = _done_criteria_validation_for_evidence(evidence)
    if not done_criteria.get("ok"):
        reason_codes.extend(done_criteria.get("reason_codes") or ["invalid_done_criteria_ledger"])
    worker_done_evidence = None
    if evidence.get("kanban_worker_done") is True or evidence.get("worker_done") is True:
        worker_done_evidence = validate_worker_done_evidence(
            evidence,
            task_body=evidence.get("task_body"),
            task_spec=evidence.get("task_spec") or evidence.get("spec"),
        )
        reason_codes.extend(worker_done_evidence.get("reason_codes") or [])
    verifier_result = evidence.get("verifier_result")
    if not isinstance(verifier_result, dict) or not verifier_result.get("verdict"):
        verifier_result = evaluate_verifier_result(
            evidence,
            task_runs=evidence.get("task_runs"),
            done_criteria_ledger=done_criteria.get("done_criteria_ledger"),
            verifier_identity=evidence.get("verifier_identity") or evidence.get("reviewer_profile") or evidence.get("reviewer") or "__independent_verifier__",
            task_body=evidence.get("task_body"),
            task_spec=evidence.get("task_spec") or evidence.get("spec"),
        )
    if verifier_result.get("verdict") != "PASS":
        reason_codes.append("verifier_not_pass")
        reason_codes.extend(verifier_result.get("reason_codes") or [])
    if _evidence_uses_task_worktree(evidence):
        reason_codes.extend(_worktree_cleanup_blockers(evidence))
    verifier_verdict = evidence.get("verifier_verdict")
    if isinstance(verifier_verdict, dict):
        verifier_pass = str(verifier_verdict.get("verdict") or "").strip().upper() == "PASS"
        if not verifier_pass:
            reason_codes.append("missing_verifier_pass")
    elif isinstance(verifier_verdict, str):
        verifier_pass = verifier_verdict.strip().upper() == "PASS"
        if not verifier_pass:
            reason_codes.append("missing_verifier_pass")
    else:
        verifier_payload = evidence.get("verifier_evidence")
        if isinstance(verifier_payload, dict):
            verifier_verdict = evaluate_verifier_verdict(evidence, verifier_payload)
            if verifier_verdict.get("verdict") != "PASS":
                reason_codes.append("missing_verifier_pass")
        else:
            verifier_verdict = None
            reason_codes.append("missing_verifier_pass")
    commit = str(required["commit"] or "").strip()
    if commit and not _SHA_RE.fullmatch(commit):
        reason_codes.append("commit_not_sha_like")
    pr_url = str(required["pr_url"] or "").strip()
    pr_match = _PR_URL_RE.fullmatch(pr_url) if pr_url else None
    if pr_url and not pr_match:
        reason_codes.append("pr_url_not_github_pull_url")
    if pr_match:
        pr_repo = f"{pr_match.group('owner')}/{pr_match.group('repo')}".lower()
        if pr_repo != expected_repo_full_name.lower():
            reason_codes.append("pr_url_repo_mismatch")
    repo = str(required["repo_full_name"] or "").strip().lower()
    if repo and repo != expected_repo_full_name.lower():
        reason_codes.append("repo_full_name_mismatch")
    task_branch = str(required["task_branch"] or "").strip()
    pr_head = str(required["pr_head"] or "").strip()
    if task_branch and pr_head and task_branch != pr_head:
        reason_codes.append("pr_head_task_branch_mismatch")
    pr_base = str(required["pr_base"] or "").strip().lower()
    if pr_base in _RELEASE_ONLY_BASES:
        reason_codes.append("release_base_requires_separate_approval")
    review_ready = not reason_codes and verifier_result.get("verdict") == "PASS"
    return {
        "review_ready": review_ready,
        "status": "review_ready" if review_ready else "blocked",
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "done_criteria_ledger": done_criteria.get("done_criteria_ledger"),
        "criteria_hash": done_criteria.get("criteria_hash"),
        "criteria_version": done_criteria.get("criteria_version"),
        "worker_done_evidence": worker_done_evidence,
        "verifier_result": verifier_result,
        "merge_allowed": False,
        "release_allowed": False,
        "prod_customer_visible_allowed": False,
        "gateway_restart_reload_allowed": False,
        "human_reason": "review-ready PR contract satisfied; merge/release still forbidden" if review_ready else "Review-ready PR contract missing or unsafe: " + ", ".join(list(dict.fromkeys(reason_codes))),
    }


def evaluate_dispatcher_eligibility(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Dry-run which Ready-gate-passing tasks could later reach dispatcher.

    This is a bridge to the existing Kanban dispatcher, not a second dispatcher.
    It returns handoff-shaped facts but performs no claim/spawn/DB transition.
    """

    state = _read_state()
    effective = _effective_mode(str(state.get("desired_mode") or "disabled"), state)
    paused_lanes = {str(lane).lower() for lane in (state.get("paused_lanes") or [])}
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for candidate in candidates:
        gate = evaluate_autopilot_ready_gate(candidate)
        tenant = str(candidate.get("tenant") or "").lower()
        if effective == "hard_stop":
            gate = {
                **gate,
                "autopilot_ready": False,
                "status": "rejected",
                "reason_codes": ["autopilot_hard_stop_active"],
                "human_reason": "Autopilot hard-stop is active; explicit operator recovery is required.",
            }
        elif tenant and tenant in paused_lanes:
            gate = {
                **gate,
                "autopilot_ready": False,
                "status": "rejected",
                "reason_codes": ["autopilot_lane_paused"],
                "human_reason": f"Autopilot lane is paused: {tenant}",
            }
        elif effective not in {"default_policy_loop", "parent_scoped", "bounded_multi_tick"}:
            gate = {
                **gate,
                "autopilot_ready": False,
                "status": "rejected",
                "reason_codes": ["autopilot_effective_mode_not_dispatch_enabled"],
                "human_reason": "Autopilot controller is not in a dispatch-enabled mode.",
            }
        if gate["autopilot_ready"]:
            eligible.append(gate)
        else:
            ineligible.append(gate)
    return {
        "controller_effective_mode": effective,
        "paused_lanes": sorted(paused_lanes),
        "handoff_target": "existing_kanban_dispatcher",
        "second_dispatcher_created": False,
        "eligible": eligible,
        "ineligible": ineligible,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
    }


def _queue_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    candidate_list = candidates or []
    results = [evaluate_autopilot_ready_gate(candidate) for candidate in candidate_list]
    dispatcher_eligibility = evaluate_dispatcher_eligibility(candidate_list)
    return {
        "action": command.action,
        "actor": actor,
        "read_only": True,
        "status": "DRY_RUN",
        "reason": "phase3_dispatcher_eligibility_bridge_no_execution_authority",
        "candidates": results,
        "dispatcher_eligibility": dispatcher_eligibility,
        "accepted_count": sum(1 for result in results if result["autopilot_ready"]),
        "rejected_count": sum(1 for result in results if not result["autopilot_ready"]),
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "state_file_enabled_is_execution_proof": False,
    }


def _format_queue_message(decision: dict[str, Any]) -> str:
    lines = [
        "Autopilot queue dry-run: READY-GATE ONLY",
        f"accepted={decision['accepted_count']}",
        f"rejected={decision['rejected_count']}",
        "No dispatch, claim, worker spawn, or Kanban mutation was attempted.",
    ]
    for candidate in decision.get("candidates") or []:
        lines.append(
            f"- {candidate.get('public_id') or candidate.get('task_id')}: {candidate['status']} "
            f"({', '.join(candidate['reason_codes']) or 'ok'})"
        )
    return "\n".join(lines)


def _closed_loop_dry_run_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    closed_loop = simulate_closed_loop_ticks(candidates or [])
    return {
        "action": command.action,
        "actor": actor,
        "read_only": True,
        "status": "DRY_RUN",
        "reason": "closed_loop_simulator_no_execution_authority",
        "closed_loop": closed_loop,
        "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }


def _format_closed_loop_dry_run_message(decision: dict[str, Any]) -> str:
    report = decision.get("closed_loop") or {}
    return "\n".join([
        "Autopilot closed-loop dry-run",
        f"would_select={len(report.get('would_select') or [])}",
        f"would_skip={len(report.get('would_skip') or [])}",
        f"would_pause={len(report.get('would_pause') or [])}",
        f"next_state={report.get('next_state')}",
        "No dispatch, claim, worker spawn, or Kanban mutation was attempted.",
    ])


def _bounded_on_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """Execute one bounded Autopilot tick through the existing dispatcher."""
    state = _write_state(_controller_state_for(command, actor=actor))
    scope = _candidate_scope_for(command)
    candidate_list = candidates or []
    promotion: dict[str, Any] | None = None
    if scope.get("parent_public_id"):
        promotion = promote_parent_scoped_children(
            candidate_list,
            parent_public_id=str(scope.get("parent_public_id") or ""),
            dry_run=False,
            default_assignee="arisu",
            apply_to_kanban=False,
        )
        candidate_list = promotion["candidates"]
    single_flight = activate_single_flight(candidate_list)
    dispatch_result: dict[str, Any] | None = None
    if single_flight.get("status") == "handoff_check_passed" and single_flight.get("selected"):
        dispatch_result = dispatch_selected_once(single_flight["selected"])
        handoff = dict(single_flight.get("handoff") or {})
        handoff.update({"check_only": False, "would_dispatch": True})
        single_flight = {**single_flight, "handoff": handoff}
    dispatched = bool((dispatch_result or {}).get("dispatched"))
    return {
        "action": command.action,
        "actor": actor,
        "desired_mode": state.get("desired_mode"),
        "effective_mode": _effective_mode(str(state.get("desired_mode") or "disabled"), state),
        "read_only": False,
        "status": "BOUNDED_DISPATCHED" if dispatched else "BOUNDED_BLOCKED",
        "reason": "bounded_dispatch_via_existing_dispatcher",
        "scope": scope,
        "caps": {"max_active_flights": 1, "max_dispatches_per_tick": 1},
        "promotion": promotion or {
            "parent_public_id": scope.get("parent_public_id"),
            "candidates": candidate_list,
            "promoted": [],
            "would_promote": [],
            "blocked": [],
            "out_of_scope": [],
            "handoff_target": "existing_kanban_dispatcher",
            "second_dispatcher_created": False,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
        },
        "single_flight": single_flight,
        "dispatch_result": dispatch_result,
        "dispatched_count": 1 if dispatched else 0,
        "dry_run_side_effects": {
            "claimed": 0,
            "spawned": len((dispatch_result or {}).get("spawned") or []),
            "mutated": 0,
            "dispatched": 1 if dispatched else 0,
        },
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "state_file_enabled_is_execution_proof": False,
    }


def autopilot_continuous_tick(*, actor: Optional[str] = None, candidates: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """Run one persisted `/autopilot on [<parent>]` controller tick.

    The gateway watcher calls this repeatedly.  It is intentionally one bounded
    tick: load either the default-policy scope or persisted parent focus, select
    at most one ready-gate-passed task, and hand only that selected task to the
    existing Kanban dispatcher.
    """

    state = _read_state()
    desired = str(state.get("desired_mode") or "disabled").lower()
    if desired != "on" or not state.get("enabled"):
        return {
            "action": "continuous-tick",
            "actor": actor,
            "desired_mode": desired,
            "effective_mode": _effective_mode(desired, state),
            "status": "IDLE",
            "reason": "autopilot_not_on",
            "scope": {"mode": "default_policy" if not state.get("focus") else "parent", "parent_public_id": state.get("focus"), "tenant": None},
            "caps": {"max_active_flights": 1, "max_dispatches_per_tick": 1},
            "dispatched_count": 0,
            "dispatch_result": None,
            "dry_run_side_effects": {"claimed": 0, "spawned": 0, "mutated": 0, "dispatched": 0},
            "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        }
    focus = str(state.get("focus") or "").strip().upper()
    command = AutopilotCommand(action="on", raw_args=f"on {focus}" if focus else "on", value=focus)
    resolved_candidates = _resolve_candidates(command, candidates)
    return _bounded_on_decision(
        command,
        actor=actor or "autopilot-continuous-loop",
        candidates=resolved_candidates,
    )


def _once_decision(command: AutopilotCommand, *, actor: Optional[str], candidates: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    single_flight = activate_single_flight(candidates or [])
    dispatch_result: dict[str, Any] | None = None
    if single_flight.get("status") == "handoff_check_passed" and single_flight.get("selected"):
        dispatch_result = dispatch_selected_once(single_flight["selected"])
        handoff = dict(single_flight.get("handoff") or {})
        handoff.update({"check_only": False, "would_dispatch": True})
        single_flight = {**single_flight, "handoff": handoff}
    if dispatch_result is not None:
        status = "DISPATCHED" if dispatch_result.get("dispatched") else "DISPATCH_BLOCKED"
    else:
        status = "CHECK_ONLY_HANDOFF_BLOCKED"
    return {
        "action": command.action,
        "actor": actor,
        "read_only": False,
        "status": status,
        "reason": "single_flight_selected_dispatch_via_existing_dispatcher" if dispatch_result is not None else "single_flight_no_dispatch_candidate",
        "single_flight": single_flight,
        "dispatch_result": dispatch_result,
        "dry_run_side_effects": {
            "claimed": 0,
            "spawned": len((dispatch_result or {}).get("spawned") or []),
            "mutated": 0,
            "dispatched": 1 if (dispatch_result or {}).get("dispatched") else 0,
        },
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
    }


def _format_once_message(decision: dict[str, Any]) -> str:
    sf = decision.get("single_flight") or {}
    selected = sf.get("selected") or {}
    return "\n".join([
        "Autopilot single-flight dispatch",
        f"status={decision.get('status')}",
        f"selected={selected.get('public_id') or selected.get('task_id')}",
        f"spawned={len(((decision.get('dispatch_result') or {}).get('spawned') or []))}",
        f"worker_done_observed={sf.get('worker_done_observed')}",
        "handoff_success_is_worker_completion=False",
        "Autopilot selected one candidate and handed only that task to the existing Kanban dispatcher.",
    ])


def _format_bounded_on_message(decision: dict[str, Any]) -> str:
    scope = decision.get("scope") or {}
    sf = decision.get("single_flight") or {}
    selected = sf.get("selected") or {}
    scope_label = scope.get("parent_public_id") or "default_policy"
    return "\n".join([
        "Autopilot bounded dispatch",
        f"status={decision.get('status')}",
        f"scope={scope_label}",
        f"selected={selected.get('public_id') or selected.get('task_id')}",
        f"dispatched={decision.get('dispatched_count')}",
        "caps=max_active_flights=1,max_dispatches_per_tick=1",
        "handoff_success_is_worker_completion=False",
    ])


def _status_decision(command: AutopilotCommand, *, actor: Optional[str] = None) -> dict[str, Any]:
    state = _read_state()
    desired_mode = str(state.get("desired_mode") or "disabled")
    effective = _effective_mode(desired_mode, state)
    reasons = []
    if effective == "parent_scoped":
        reasons.append("parent_scoped_continuous_loop_enabled")
    elif effective == "default_policy_loop":
        reasons.append("default_policy_continuous_loop_enabled")
    else:
        reasons.append("phase1_controller_state_only_no_execution_authority")
    if state.get("read_error"):
        reasons.append(str(state["read_error"]))
    if (state.get("enabled") or desired_mode in {"on", "enabled"}) and effective not in {"parent_scoped", "default_policy_loop"}:
        reasons.append("state_file_enabled_true_is_not_runtime_proof")
    return {
        "action": command.action,
        "actor": actor,
        "desired_mode": desired_mode,
        "effective_mode": effective,
        "status": effective.upper(),
        "reason": ";".join(reasons),
        "focus": state.get("focus"),
        "scope_mode": "parent" if state.get("focus") else "default_policy",
        "pause_reason": state.get("pause_reason"),
        "paused_lanes": state.get("paused_lanes") or [],
        "hard_stop_reason": state.get("hard_stop_reason"),
        "dispatch_blocked": effective in {"blocked", "paused", "hard_stop", "stopped", "degraded"},
        "operator_recovery_required": effective == "hard_stop",
        "state_file": state,
        "state_file_enabled_is_execution_proof": False,
        "read_only": command.read_only,
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "dispatch_once_called": False,
        "claim_task_called": False,
        "worker_spawn_called": False,
        "kanban_mutation_called": False,
    }


def _format_status_message(decision: dict[str, Any]) -> str:
    state = decision.get("state_file") or {}
    lines = [
        "Autopilot status: " + str(decision["status"]),
        f"desired_mode={decision['desired_mode']}",
        f"effective_mode={decision['effective_mode']}",
    ]
    if decision.get("focus"):
        lines.append(f"focus={decision['focus']}")
    if decision.get("pause_reason"):
        lines.append(f"pause_reason={decision['pause_reason']}")
    lines.append("reason=" + str(decision["reason"]))
    if decision.get("effective_mode") in {"parent_scoped", "default_policy_loop"}:
        lines.extend([
            "Continuous bounded loop is enabled for the focused parent." if decision.get("effective_mode") == "parent_scoped" else "Continuous bounded loop is enabled for the default/global policy scope.",
            "Each tick may hand at most one selected task to the existing Kanban dispatcher.",
        ])
    else:
        lines.extend(
            [
                "State file enabled=true is not execution proof.",
                "No dispatch, claim, worker spawn, or Kanban mutation was attempted.",
            ]
        )
    lines.append(f"state_file={state.get('path')}")
    return "\n".join(lines)


def _controller_state_for(command: AutopilotCommand, *, actor: Optional[str]) -> dict[str, Any]:
    previous = _read_state()
    state: dict[str, Any] = {
        "version": AUTOPILOT_STATE_VERSION,
        "enabled": bool(previous.get("enabled")),
        "desired_mode": str(previous.get("desired_mode") or ("enabled" if previous.get("enabled") else "disabled")),
        "focus": previous.get("focus"),
        "pause_reason": previous.get("pause_reason"),
        "paused_lanes": previous.get("paused_lanes") or [],
        "hard_stop_reason": previous.get("hard_stop_reason"),
        "updated_at": _iso_now(),
        "updated_by": actor or "unknown",
    }
    if command.action == "on":
        updates: dict[str, Any] = {"desired_mode": "on", "enabled": True, "pause_reason": None}
        if command.value.strip():
            updates["focus"] = command.value.strip().upper()
        else:
            updates["focus"] = None
        state.update(updates)
    elif command.action in {"off", "stop"}:
        state.update({"desired_mode": "stopped", "enabled": False, "pause_reason": None})
    elif command.action == "pause":
        state.update({"desired_mode": "paused", "enabled": False, "pause_reason": command.value or "manual_pause"})
    elif command.action == "pause-lane":
        parts = command.value.split(maxsplit=1)
        lane = parts[0].strip().lower() if parts else ""
        reason = parts[1].strip() if len(parts) > 1 else "manual_lane_pause"
        paused = {str(existing).lower() for existing in (state.get("paused_lanes") or [])}
        if lane:
            paused.add(lane)
        state.update({"paused_lanes": sorted(paused), "pause_reason": reason})
    elif command.action == "resume-lane":
        lane = command.value.split(maxsplit=1)[0].strip().lower()
        paused = {str(existing).lower() for existing in (state.get("paused_lanes") or [])}
        if lane:
            paused.discard(lane)
        state.update({"paused_lanes": sorted(paused)})
    elif command.action == "hard-stop":
        state.update({"desired_mode": "hard_stopped", "enabled": False, "hard_stop_reason": command.value or "manual_hard_stop", "pause_reason": command.value or "manual_hard_stop"})
    elif command.action == "recover":
        state.update({"desired_mode": "paused", "enabled": False, "hard_stop_reason": None, "pause_reason": command.value or "manual_recovery_acknowledged"})
    elif command.action == "focus":
        state.update({"focus": command.value.upper() if command.value else None})
    elif command.action == "once":
        state.update({"desired_mode": "paused", "enabled": False, "pause_reason": "once_not_available_until_dispatch_gate"})
    return state


def _control_decision(command: AutopilotCommand, *, actor: Optional[str] = None) -> dict[str, Any]:
    state = _write_state(_controller_state_for(command, actor=actor))
    desired_mode = str(state.get("desired_mode") or "disabled")
    effective = _effective_mode(desired_mode, state)
    return {
        "action": command.action,
        "actor": actor,
        "desired_mode": desired_mode,
        "effective_mode": effective,
        "status": effective.upper(),
        "reason": "phase1_controller_state_persisted_without_execution_authority",
        "focus": state.get("focus"),
        "scope_mode": "parent" if state.get("focus") else "default_policy",
        "pause_reason": state.get("pause_reason"),
        "paused_lanes": state.get("paused_lanes") or [],
        "hard_stop_reason": state.get("hard_stop_reason"),
        "dispatch_blocked": effective in {"blocked", "paused", "hard_stop", "stopped", "degraded"},
        "operator_recovery_required": effective == "hard_stop",
        "state_file": state,
        "state_file_enabled_is_execution_proof": False,
        "read_only": False,
        "mutations_attempted": list(_MUTATIONS_ATTEMPTED),
        "dispatch_once_called": False,
        "claim_task_called": False,
        "worker_spawn_called": False,
        "kanban_mutation_called": False,
    }


def handle_autopilot_command(
    raw_args: str = "",
    *,
    actor: Optional[str] = None,
    candidates: Optional[list[dict[str, Any]]] = None,
) -> AutopilotResult:
    """Handle /autopilot without execution authority in Phase 2."""

    try:
        command = parse_autopilot_args(raw_args)
    except ValueError as exc:
        return AutopilotResult(
            ok=False,
            command=None,
            message=str(exc),
            decision={"status": "BLOCKED", "reason": "parse_error", "mutations_attempted": []},
            fail_closed=True,
        )

    if command.action == "queue":
        decision = _queue_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_queue_message(decision), decision, False)

    if command.action == "dry-run":
        decision = _closed_loop_dry_run_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_closed_loop_dry_run_message(decision), decision, False)

    if command.action in {"status"}:
        decision = _status_decision(command, actor=actor)
        return AutopilotResult(True, command, _format_status_message(decision), decision, False)

    if command.action == "once":
        decision = _once_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_once_message(decision), decision, False)

    if command.action == "on" and command.value.strip():
        decision = _bounded_on_decision(command, actor=actor, candidates=_resolve_candidates(command, candidates))
        return AutopilotResult(True, command, _format_bounded_on_message(decision), decision, False)

    if command.action in _CONTROL_ACTIONS:
        decision = _control_decision(command, actor=actor)
        return AutopilotResult(
            ok=True,
            command=command,
            message=_format_status_message(decision),
            decision=decision,
            fail_closed=False,
        )

    decision = {"status": "BLOCKED", "reason": "unsupported_action", "mutations_attempted": []}
    return AutopilotResult(False, command, f"Unsupported /autopilot action. {AUTOPILOT_USAGE}", decision, True)
