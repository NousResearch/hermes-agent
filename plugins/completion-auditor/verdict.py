"""Deterministic evidence evaluator for completion-auditor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .classifier import Claim
from .evidence import EvidenceRef, TurnEvidence

_VERIFICATION_TOOLS = {"terminal", "execute_code"}
_MUTATION_TOOLS = {"write_file", "patch", "browser_type", "browser_click"}
_EXTERNAL_TOOLS = {"send_message", "image_generate", "text_to_speech"}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    evidence_tier: str
    contradictions: list[str] = field(default_factory=list)
    degraded: bool = False
    degrade_reason: str | None = None


def _status(item: EvidenceRef) -> str:
    return item.status.strip().lower()


def _is_success(item: EvidenceRef) -> bool:
    status = _status(item)
    if status in {"error", "failed", "failure", "exception", "timeout"}:
        return False
    if item.exit_code is not None:
        return item.exit_code == 0
    return status in {"success", "ok", "completed", "unknown"}


def _is_failure(item: EvidenceRef) -> bool:
    status = _status(item)
    return status in {"error", "failed", "failure", "exception", "timeout"} or (
        item.exit_code is not None and item.exit_code != 0
    )


def _has_matching_scope(item: EvidenceRef, claim_scope: str | None) -> bool:
    if not claim_scope:
        return True
    scope = claim_scope.rstrip("/.")
    return any(ref.rstrip("/.").endswith(scope) or scope.endswith(ref.rstrip("/.")) for ref in item.arg_refs)


def _tool_in(item: EvidenceRef, names: set[str]) -> bool:
    return item.tool_name in names


def _evaluate_verification_claim(claim: Claim, evidence: list[EvidenceRef]) -> Verdict:
    failed = [item for item in evidence if _tool_in(item, _VERIFICATION_TOOLS) and _is_failure(item)]
    if failed:
        return Verdict(
            verdict="fail",
            evidence_tier="tier_1",
            contradictions=[
                f"{item.tool_name} evidence {item.evidence_id} returned exit_code={item.exit_code} status={item.status}"
                for item in failed[:3]
            ],
        )

    direct = [
        item
        for item in evidence
        if _tool_in(item, _VERIFICATION_TOOLS)
        and _is_success(item)
        and item.exit_code == 0
        and item.command_kind == "verification"
    ]
    if direct:
        return Verdict(verdict="supported", evidence_tier="tier_1")

    weak = [item for item in evidence if _tool_in(item, _VERIFICATION_TOOLS) and _is_success(item)]
    if weak:
        return Verdict(
            verdict="weak",
            evidence_tier="tier_3",
            degraded=True,
            degrade_reason="verification evidence lacks structured exit_code=0",
        )
    return Verdict(verdict="weak", evidence_tier="tier_4")


def _evaluate_mutation_claim(claim: Claim, evidence: list[EvidenceRef]) -> Verdict:
    failed = [
        item
        for item in evidence
        if _tool_in(item, _MUTATION_TOOLS) and _has_matching_scope(item, claim.claim_scope) and _is_failure(item)
    ]
    if failed:
        return Verdict(
            verdict="fail",
            evidence_tier="tier_1",
            contradictions=[
                f"{item.tool_name} evidence {item.evidence_id} failed for claimed scope"
                for item in failed[:3]
            ],
        )

    direct = [
        item
        for item in evidence
        if _tool_in(item, _MUTATION_TOOLS) and _has_matching_scope(item, claim.claim_scope) and _is_success(item)
    ]
    if direct:
        return Verdict(verdict="supported", evidence_tier="tier_1" if claim.claim_scope else "tier_2")

    indirect = [item for item in evidence if _tool_in(item, _MUTATION_TOOLS) and _is_success(item)]
    if indirect:
        return Verdict(
            verdict="weak",
            evidence_tier="tier_3",
            degraded=bool(claim.claim_scope),
            degrade_reason="mutation evidence does not match claim_scope" if claim.claim_scope else None,
        )
    return Verdict(verdict="weak", evidence_tier="tier_4")


def _evaluate_external_claim(claim: Claim, evidence: list[EvidenceRef]) -> Verdict:
    failed = [item for item in evidence if _tool_in(item, _EXTERNAL_TOOLS) and _is_failure(item)]
    if failed:
        return Verdict(
            verdict="fail",
            evidence_tier="tier_1",
            contradictions=[f"{item.tool_name} evidence {item.evidence_id} failed" for item in failed[:3]],
        )
    direct = [item for item in evidence if _tool_in(item, _EXTERNAL_TOOLS) and _is_success(item)]
    if direct:
        return Verdict(verdict="supported", evidence_tier="tier_2")
    return Verdict(verdict="weak", evidence_tier="tier_4")


def evaluate_claim(claim: Claim | None, evidence: TurnEvidence | None) -> Verdict:
    """Return an evidence-alignment verdict for one classified final-response claim.

    This evaluator is intentionally conservative. It only marks a claim as
    ``supported`` when the hook ledger contains direct structured metadata for
    the same turn; it never claims semantic task correctness.
    """

    if claim is None:
        return Verdict(verdict="not_applicable", evidence_tier="tier_4")

    refs = evidence.evidence if evidence is not None else []
    if not refs:
        return Verdict(verdict="weak", evidence_tier="tier_4")

    if claim.claim_type in {"tested", "verified"}:
        return _evaluate_verification_claim(claim, refs)
    if claim.claim_type in {"modified", "created"}:
        return _evaluate_mutation_claim(claim, refs)
    if claim.claim_type == "deployed":
        return _evaluate_external_claim(claim, refs)

    # Broad semantic claims such as implemented/fixed/completed remain weak in
    # the MVP unless a later slice narrows them to verifier-specific semantics.
    if any(_is_failure(item) for item in refs):
        failures = [item for item in refs if _is_failure(item)]
        return Verdict(
            verdict="fail",
            evidence_tier="tier_3",
            contradictions=[
                f"{item.tool_name} evidence {item.evidence_id} failed while broad completion was claimed"
                for item in failures[:3]
            ],
        )
    return Verdict(verdict="weak", evidence_tier="tier_3")
