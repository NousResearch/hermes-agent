#!/usr/bin/env python3
"""Deterministic PR/CI traffic-controller posture/proof contract.

The LLM may summarize this output, but it must not independently invent
posture/proof/action combinations. This module contains no live PR/ticket IDs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

POSTURES = {"DRAIN_ACTION_APPLIED", "DRAIN_ACTION_ATTEMPTED_BUT_FAILED_CLOSED", "DRAIN_ACTION_REQUIRED_BUT_NOT_APPLIED", "ACTION_WITHHELD_PERMISSION_LIMIT", "TRUE_NO_ACTION_SAFE"}
PROOFS = {"PASS", "FAIL", "N/A"}
READBACK = {"YES", "NO", "N/A"}
METHODS = {"structured_json_atomic_write", "none"}
CLEAR_PRESSURE = {"clear", "merged", "closed", "superseded", "outside_controller_authority", "none", None}
SUCCESS_ACTIONS = {"LANE_TICKET_CREATED", "LANE_TICKET_UPDATED", "LANE_TICKET_SUPERSEDED"}
APPLIED_ACTIONS = SUCCESS_ACTIONS | {
    "FRONTIER_REVIEW_INVOKED",
    "FRONTIER_REVIEW_QUEUED_WITH_RECEIPT",
    "FRONTIER_REVIEW_ALREADY_PENDING",
    "FRONTIER_REVIEW_RECEIPT_VERIFIED",
    "FRONTIER_REVIEW_BLOCKED_NO_CONSUMER",
    "RUNNER_REPAIR_APPLIED",
    "DISPATCHER_REPAIR_RECEIPT_WRITTEN",
    "INBOX_RECOVERED",
    "READY_TO_ADVANCE_ACTION_APPLIED",
    "LANE_TICKET_REPAIRED_UPDATED",
    "REPLACEMENT_PRESSURE_TICKET_CREATED",
    "REPLACEMENT_PRESSURE_TICKET_UPDATED",
    "STALE_TICKET_SUPERSEDED",
    "DISPATCHER_NUDGE_WRITTEN",
    "READY_TO_ADVANCE_LOCAL_RECEIPT",
}
FRONTIER_ACTIONS = {
    "FRONTIER_REVIEW_INVOKED",
    "FRONTIER_REVIEW_QUEUED_WITH_RECEIPT",
    "FRONTIER_REVIEW_ALREADY_PENDING",
    "FRONTIER_REVIEW_BLOCKED_NO_CONSUMER",
    "FRONTIER_REVIEW_RECEIPT_VERIFIED",
}
FRONTIER_ESCALATION_PACKET_FIELDS = (
    "frontier_ticket_id",
    "target_model_lane",
    "org",
    "repo",
    "pr_numbers",
    "pr_head_sha",
    "blocker_type",
    "why_frontier_is_needed",
    "why_controller_cannot_drain_directly",
    "requested_output_type",
    "expected_next_action_after_frontier_response",
    "state_fingerprint",
    "created_at",
    "status",
)
REVIEW_CONSUMER_STATUSES = {"queued", "invoked", "already_pending", "blocked_no_consumer", "assigned", "not_yet_routed"}


def mentions_adversarial_review(report: dict[str, Any]) -> bool:
    haystack = " ".join(str(report.get(k) or "") for k in ("required_next_action", "next_action", "escalation_status"))
    return bool(re.search(r"\b(adversarial review|opposite-provider review)\b", haystack, re.I))


def _pick(payload: dict[str, Any], key: str, default: Any) -> Any:
    return payload.get(key, default)


def required_proofs(payload: dict[str, Any]) -> dict[str, str]:
    inbox = payload.get("inbox_repair_result") or {}
    return {
        "hard_coded_example_check": _pick(payload, "hard_coded_example_check", "PASS"),
        "lane_ticket_durable_mutation_proof": _pick(payload, "lane_ticket_durable_mutation_proof", inbox.get("lane_ticket_durable_mutation_proof", "N/A")),
        "durable_mutation_readback_verified": _pick(payload, "durable_mutation_readback_verified", inbox.get("durable_mutation_readback_verified", "N/A")),
        "lane_ticket_consumption_proof": _pick(payload, "lane_ticket_consumption_proof", inbox.get("lane_ticket_consumption_proof", "N/A")),
        "frontier_consumer_proof": _pick(payload, "frontier_consumer_proof", "N/A"),
        "hold_behind_reconciliation_performed": _pick(payload, "hold_behind_reconciliation_performed", "N/A"),
        "ready_to_advance_candidates_checked": _pick(payload, "ready_to_advance_candidates_checked", "YES"),
    }


def failed_or_unverifiable(proofs: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for key, value in proofs.items():
        if value in {"FAIL", "NO", "UNVERIFIABLE", "UNKNOWN", None}:  # type: ignore[comparison-overlap]
            failures.append(key)
    if proofs.get("ready_to_advance_candidates_checked") not in {"YES", "N/A"}:
        failures.append("ready_to_advance_candidates_checked")
    return sorted(set(failures))


def _actions(report: dict[str, Any]) -> list[str]:
    return list(report.get("actions_performed") or report.get("actions") or [])


def validate_report_semantics(report: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    posture = report.get("drain_action_posture")
    if posture and posture not in POSTURES:
        problems.append(f"unknown drain_action_posture: {posture}")
    method = report.get("durable_mutation_method", "none")
    mutation_proof = report.get("lane_ticket_durable_mutation_proof", "N/A")
    readback = report.get("durable_mutation_readback_verified", "N/A")
    actions = _actions(report)
    proof_subset = required_proofs(report)
    bad = failed_or_unverifiable(proof_subset)
    if posture == "TRUE_NO_ACTION_SAFE" and bad:
        problems.append("TRUE_NO_ACTION_SAFE cannot appear with failed/unverifiable proofs")
    if posture == "TRUE_NO_ACTION_SAFE" and report.get("lane_ticket_consumption_proof") == "FAIL":
        problems.append("TRUE_NO_ACTION_SAFE cannot appear with Lane-ticket consumption proof: FAIL")
    classifications = set(report.get("classifications") or [])
    inbox = report.get("inbox_repair_result") or {}
    if inbox.get("parse_status") == "MALFORMED" or "LANE_TICKET_INBOX_MALFORMED" in classifications:
        if "LANE_TICKET_INBOX_MALFORMED" not in classifications:
            problems.append("malformed inbox state requires LANE_TICKET_INBOX_MALFORMED")
        if posture == "TRUE_NO_ACTION_SAFE":
            problems.append("TRUE_NO_ACTION_SAFE cannot appear when lane-ticket inbox parse failed")
        if inbox.get("recovery_status") != "RECOVERED" and report.get("lane_ticket_consumption_proof") != "FAIL":
            problems.append("unrecovered malformed inbox requires Lane-ticket consumption proof: FAIL")
    if method == "none" and mutation_proof == "PASS":
        problems.append("durable mutation proof PASS requires structured_json_atomic_write")
    if method == "none" and readback == "YES":
        problems.append("readback YES requires a mutation method")
    if report.get("durable_mutation_attempted") is False:
        if mutation_proof != "N/A" or method != "none" or readback != "N/A":
            problems.append("no durable mutation attempted requires proof N/A, method none, readback N/A")
    if report.get("durable_mutation_attempted") is True and mutation_proof == "FAIL":
        if readback != "NO":
            problems.append("failed durable mutation requires readback verified NO")
        if any(action in SUCCESS_ACTIONS for action in actions):
            problems.append("failed durable mutation cannot claim create/update/supersede success")
    if posture == "TRUE_NO_ACTION_SAFE" and report.get("missing_consumer_receipts_remain") and not report.get("per_ticket_missing_consumer_repair_classified"):
        problems.append("TRUE_NO_ACTION_SAFE forbidden with unclassified missing consumer receipts")
    if report.get("lane_action_done") and report.get("live_pr_pressure_status") not in CLEAR_PRESSURE:
        if not report.get("replacement_pressure_coverage"):
            problems.append("done lane action cannot clear live PR pressure without replacement pressure coverage")
    if report.get("green_review_pending_prs"):
        if posture == "TRUE_NO_ACTION_SAFE" or report.get("live_pr_pressure_status") in CLEAR_PRESSURE:
            problems.append("green review-pending PRs must remain active pressure")
    if report.get("held_green_prs") and report.get("hold_behind_reconciliation_performed") != "YES":
        problems.append("held green PRs require hold-behind reconciliation")
    if report.get("frontier_escalation_required") and not any(action in FRONTIER_ACTIONS for action in actions):
        problems.append("event-driven frontier escalation cannot stop at a normal lane ticket")
    if mentions_adversarial_review(report):
        lane = report.get("adversarial_review_lane")
        status = report.get("adversarial_review_consumer_status")
        if not lane or not status:
            problems.append("adversarial review mention requires review lane and consumer status")
        elif status not in REVIEW_CONSUMER_STATUSES:
            problems.append(f"unknown adversarial review consumer status: {status}")
    if posture == "DRAIN_ACTION_APPLIED" and set(actions).issubset({"OBSERVATION_RECORDED", "NO_DRAIN_ACTION_APPLIED"}):
        problems.append("local report-only output cannot count as pressure release")
    return problems


def derive_contract(payload: dict[str, Any]) -> dict[str, Any]:
    inbox = payload.get("inbox_repair_result") or {}
    proofs = required_proofs(payload)
    classifications: list[str] = list(payload.get("classifications") or [])
    failure_reasons: list[str] = []
    actions = list(payload.get("actions_performed") or payload.get("actions") or [])
    withheld = list(payload.get("actions_intentionally_withheld") or [])
    trusted = bool(inbox.get("trusted_consumer_receipts", True))
    method = payload.get("durable_mutation_method", inbox.get("durable_mutation_method", "none"))
    durable_attempted = payload.get("durable_mutation_attempted")
    if durable_attempted is None:
        durable_attempted = method != "none" or proofs["lane_ticket_durable_mutation_proof"] in {"PASS", "FAIL"}

    if inbox.get("classification") == "LANE_TICKET_INBOX_MALFORMED" or inbox.get("parse_status") == "MALFORMED":
        classifications.append("LANE_TICKET_INBOX_MALFORMED")
        if inbox.get("recovery_status") == "RECOVERED":
            actions.append("INBOX_RECOVERED")
            proofs["lane_ticket_durable_mutation_proof"] = inbox.get("lane_ticket_durable_mutation_proof", "PASS")
            proofs["durable_mutation_readback_verified"] = inbox.get("durable_mutation_readback_verified", "YES")
            method = inbox.get("durable_mutation_method", "structured_json_atomic_write")
            durable_attempted = True
        else:
            trusted = False
            proofs["lane_ticket_consumption_proof"] = "FAIL"
            failure_reasons.append("lane-ticket inbox malformed; consumer receipts unverifiable")

    if durable_attempted is False:
        proofs["lane_ticket_durable_mutation_proof"] = "N/A"
        method = "none"
        proofs["durable_mutation_readback_verified"] = "N/A"

    if proofs["lane_ticket_durable_mutation_proof"] == "PASS" and method == "none":
        failure_reasons.append("durable mutation proof PASS with method none is invalid")
    if proofs["lane_ticket_durable_mutation_proof"] == "FAIL":
        actions = [action for action in actions if action not in SUCCESS_ACTIONS]
        if proofs["durable_mutation_readback_verified"] != "NO":
            proofs["durable_mutation_readback_verified"] = "NO"

    if payload.get("missing_consumer_receipts_remain") and not payload.get("per_ticket_missing_consumer_repair_classified"):
        classifications.append("MISSING_CONSUMER_RECEIPT_UNCLASSIFIED")
        failure_reasons.append("missing consumer receipts remain without valid per-ticket repair classification")
    if payload.get("lane_action_done") and payload.get("live_pr_pressure_status") not in CLEAR_PRESSURE and not payload.get("replacement_pressure_coverage"):
        classifications.append("REPLACEMENT_PRESSURE_COVERAGE_REQUIRED")
        failure_reasons.append("done lane action cannot clear live PR pressure")
    if payload.get("green_review_pending_prs"):
        classifications.append("GREEN_REVIEW_PENDING_ACTIVE_PRESSURE")
        failure_reasons.append("green review-pending PRs remain active pressure")
    if payload.get("held_green_prs") and proofs["hold_behind_reconciliation_performed"] != "YES":
        classifications.append("HOLD_BEHIND_RECONCILIATION_REQUIRED")
        failure_reasons.append("held green PRs require hold-behind reconciliation")
    if payload.get("frontier_escalation_required") and not any(action in FRONTIER_ACTIONS for action in actions):
        classifications.append("FRONTIER_CONSUMER_PROOF_REQUIRED")
        proofs["frontier_consumer_proof"] = "FAIL"
        failure_reasons.append("frontier consumer proof failed")
    if mentions_adversarial_review(payload):
        lane = payload.get("adversarial_review_lane")
        status = payload.get("adversarial_review_consumer_status")
        if not lane or not status:
            classifications.append("ADVERSARIAL_REVIEW_ROUTE_UNSPECIFIED")
            proofs["frontier_consumer_proof"] = "FAIL"
            failure_reasons.append("adversarial review next action lacks review lane and consumer status")
        elif status not in REVIEW_CONSUMER_STATUSES:
            classifications.append("ADVERSARIAL_REVIEW_CONSUMER_STATUS_INVALID")
            proofs["frontier_consumer_proof"] = "FAIL"
            failure_reasons.append("adversarial review consumer status invalid")
    if set(actions).issubset({"OBSERVATION_RECORDED", "NO_DRAIN_ACTION_APPLIED"}) and actions:
        classifications.append("LOCAL_REPORT_ONLY_NO_PRESSURE_RELEASE")
        failure_reasons.append("local report-only output cannot count as pressure release")

    bad_proofs = failed_or_unverifiable(proofs)
    if bad_proofs:
        classifications.append("TRUE_NO_ACTION_SAFE_FORBIDDEN_FAILED_OR_UNVERIFIABLE_PROOF")
        for proof in bad_proofs:
            label = proof.replace("_", " ")
            if proof == "frontier_consumer_proof":
                failure_reasons.append("frontier consumer proof failed")
            elif proof == "lane_ticket_consumption_proof":
                failure_reasons.append("lane-ticket consumption proof failed")
            else:
                failure_reasons.append(f"{label} failed or unverifiable")

    pressure = payload.get("live_pr_pressure_status", "clear")
    if pressure not in CLEAR_PRESSURE:
        failure_reasons.append("live PR pressure remains")

    action_applied = any(action in APPLIED_ACTIONS for action in actions) or bool(payload.get("action_applied"))
    tier1_applied = list(payload.get("tier1_actions_applied") or [])
    only_ready_local_receipt = bool(tier1_applied) and set(tier1_applied).issubset({"ready_to_advance_local_receipt"})
    action_attempted = bool(payload.get("action_attempted")) or bool(payload.get("tier1_actions_attempted")) or action_applied
    action_failed = bool(payload.get("action_failed")) or bool(bad_proofs)
    pressure_remains = pressure not in CLEAR_PRESSURE or bool(failure_reasons)
    permission_withheld = any("permission" in str(item).lower() or "without explicit runtime authority" in str(item).lower() or "github write authority disabled" in str(item).lower() for item in withheld)
    disabled_external_actions = list(payload.get("tier2_tier3_actions_required_but_disabled") or [])

    if action_applied and not action_failed and not only_ready_local_receipt and not (set(actions).issubset({"OBSERVATION_RECORDED", "NO_DRAIN_ACTION_APPLIED"}) and actions):
        posture = "DRAIN_ACTION_APPLIED"
    elif action_failed or (payload.get("action_attempted") and not action_applied) or (payload.get("tier1_actions_attempted") and not action_applied):
        posture = "DRAIN_ACTION_ATTEMPTED_BUT_FAILED_CLOSED"
    elif pressure_remains and (permission_withheld or disabled_external_actions or only_ready_local_receipt):
        posture = "ACTION_WITHHELD_PERMISSION_LIMIT"
    elif pressure_remains:
        posture = "DRAIN_ACTION_REQUIRED_BUT_NOT_APPLIED"
    else:
        posture = "TRUE_NO_ACTION_SAFE"

    next_action = payload.get("required_next_action") or payload.get("next_action") or "none"
    if not trusted:
        next_action = "repair lane-ticket inbox before trusting consumer receipts"
    elif failure_reasons and next_action == "none":
        next_action = "resolve deterministic contract failure before claiming no-action-safe"

    result: dict[str, Any] = {
        "drain_action_posture": posture,
        "hard_coded_example_check": proofs["hard_coded_example_check"],
        "lane_ticket_durable_mutation_proof": proofs["lane_ticket_durable_mutation_proof"],
        "durable_mutation_method": method,
        "durable_mutation_readback_verified": proofs["durable_mutation_readback_verified"],
        "lane_ticket_consumption_proof": proofs["lane_ticket_consumption_proof"],
        "frontier_consumer_proof": proofs["frontier_consumer_proof"],
        "missing_consumer_receipt_repair_attempted": payload.get("missing_consumer_receipt_repair_attempted", "N/A"),
        "hold_behind_reconciliation_performed": proofs["hold_behind_reconciliation_performed"],
        "ready_to_advance_candidates_checked": proofs["ready_to_advance_candidates_checked"],
        "actions_performed": actions,
        "actions_intentionally_withheld": withheld,
        "required_next_action": next_action,
        "invalid_combinations_rejected": [],
        "full_report_required": True,
        "trusted_consumer_receipts": trusted,
        "classifications": sorted(set(classifications)),
        "failure_reasons": sorted(set(failure_reasons)),
        "next_action": next_action,
        "inbox_repair_result": inbox,
        "durable_mutation_attempted": durable_attempted,
        "live_pr_pressure_status": pressure,
        "replacement_pressure_coverage": payload.get("replacement_pressure_coverage"),
        "frontier_escalation_required": payload.get("frontier_escalation_required", False),
        "missing_consumer_receipts_remain": payload.get("missing_consumer_receipts_remain", False),
        "per_ticket_missing_consumer_repair_classified": payload.get("per_ticket_missing_consumer_repair_classified", False),
        "lane_action_done": payload.get("lane_action_done", False),
        "green_review_pending_prs": payload.get("green_review_pending_prs", []),
        "held_green_prs": payload.get("held_green_prs", []),
        "adversarial_review_lane": payload.get("adversarial_review_lane"),
        "adversarial_review_consumer_status": payload.get("adversarial_review_consumer_status"),
        "action_authority_matrix": payload.get("action_authority_matrix", {}),
        "tier1_local_actuator_enabled": bool(payload.get("tier1_local_actuator_enabled")),
        "tier1_actions_evaluated": payload.get("tier1_actions_evaluated", []),
        "tier1_actions_applied": tier1_applied,
        "tier1_actions_attempted": payload.get("tier1_actions_attempted", []),
        "tier1_actions_withheld": payload.get("tier1_actions_withheld", []),
        "tier2_tier3_actions_required_but_disabled": disabled_external_actions,
        "tier1_action": payload.get("tier1_action") or (tier1_applied[0] if tier1_applied else "none"),
        "github_write_authority": payload.get("github_write_authority", "disabled"),
        "next_permission_limited_action": payload.get("next_permission_limited_action") or (disabled_external_actions[0] if disabled_external_actions else "none"),
        "tier1_readback_proofs": payload.get("tier1_readback_proofs", []),
        "no_duplicate_owner_proof": payload.get("no_duplicate_owner_proof", {}),
        "controller_work_inbox": payload.get("controller_work_inbox", {"done": 0, "blocked": 0, "deferred": 0, "queued": 0, "top_action": "none", "next_item": None}),
    }
    problems = validate_report_semantics(result)
    result["invalid_combinations_rejected"] = problems
    result["valid"] = not problems
    result["semantic_validation_errors"] = problems
    return result


def render_compact_report(repo: str, contract: dict[str, Any], full_report_path: str) -> str:
    try:
        import compact_report_renderer  # type: ignore
        payload = dict(contract)
        payload.setdefault("repo", repo)
        payload.setdefault("full_report_path", full_report_path)
        return compact_report_renderer.render_compact_report(payload)
    except Exception:
        action = "none durably applied"
        actions = contract.get("actions_performed") or contract.get("actions") or []
        if actions:
            action = ", ".join(actions)
        elif "LANE_TICKET_INBOX_MALFORMED" in set(contract.get("classifications") or []):
            action = "malformed inbox backup attempted; recovery failed"
        proof = (
            f"hard-coded {contract.get('hard_coded_example_check')} · "
            f"mutation {contract.get('lane_ticket_durable_mutation_proof')} · "
            f"readback {contract.get('durable_mutation_readback_verified')} · "
            f"consumption {contract.get('lane_ticket_consumption_proof')} · "
            f"frontier {contract.get('frontier_consumer_proof')}"
        )
        return "\n".join([
            f"{repo} PR/CI controller — deterministic contract",
            f"Posture: {contract.get('drain_action_posture')}",
            f"Proof: {proof}",
            f"Action: {action}",
            f"Next: {contract.get('required_next_action') or contract.get('next_action')}",
            f"Full report: {full_report_path}",
        ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--compact-repo")
    args = parser.parse_args()
    result = derive_contract(json.loads(args.input.read_text()))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(text)
    if args.compact_repo:
        print(render_compact_report(args.compact_repo, result, str(args.report or args.input)), end="")
    else:
        print(text, end="")
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
