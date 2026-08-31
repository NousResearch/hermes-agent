"""Deterministic non-activation specialist selection decisions.

This module deliberately produces an auditable *non-dispatch* decision. It
does not read provider health, capacity, credentials, model configuration, or
time-varying profile state, and it cannot start a profile or reserve a slot.
Electron remains the future owner of any capacity admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .collaboration import ContractError, SpecialistDescriptorV1, digest, specialist_descriptor_ref


DECISION_SCHEMA = "AresSpecialistDecisionReceiptV1"


@dataclass(frozen=True)
class SpecialistDecisionReceiptV1:
    """A stable, content-addressed non-dispatch decision projection."""

    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _finalize(payload: dict[str, object]) -> SpecialistDecisionReceiptV1:
    receipt = dict(payload)
    receipt["decision_digest"] = digest(receipt)
    return SpecialistDecisionReceiptV1(receipt)


def decide_specialist_nonactivation(
    *,
    request_id: str,
    request_digest: str,
    requested_capability: str | None,
    candidates: Sequence[SpecialistDescriptorV1],
    profile_binding_refs: Mapping[str, str],
    policy_version: str,
) -> SpecialistDecisionReceiptV1:
    """Return a deterministic no-dispatch decision for a frozen candidate set.

    A matching descriptor remains blocked while it is disabled. This function
    cannot return a selected profile: that transition needs separately
    authorized value evidence plus Electron-owned admission, neither of which
    is supplied here.
    """
    if not isinstance(request_id, str) or not request_id:
        raise ContractError("INVALID_REQUEST", "request_id")
    if not isinstance(request_digest, str) or not request_digest.startswith("sha256:"):
        raise ContractError("INVALID_REQUEST", "request_digest")
    if not isinstance(policy_version, str) or not policy_version:
        raise ContractError("INVALID_POLICY", "policy_version")
    ordered = sorted(candidates, key=lambda candidate: str(candidate.to_dict()["profile_id"]))
    profile_ids = [str(candidate.to_dict()["profile_id"]) for candidate in ordered]
    if len(profile_ids) != len(set(profile_ids)):
        raise ContractError("DUPLICATE_PROFILE", "candidates")

    requested = (requested_capability or "").strip()
    base: dict[str, object] = {
        "schema": DECISION_SCHEMA,
        "request_id": request_id,
        "request_digest": request_digest,
        "policy_version": policy_version,
        "candidate_profile_ids": profile_ids,
        "selected_profile_id": None,
        "dispatch_authorized": False,
    }
    if not requested:
        return _finalize(
            {
                **base,
                "outcome": "no_specialist_needed",
                "reason_code": "NO_MATERIAL_EVIDENCE_GAP",
                "candidate_rejections": [],
            }
        )

    rejections: list[dict[str, str]] = []
    for candidate in ordered:
        descriptor = candidate.to_dict()
        profile_id = str(descriptor["profile_id"])
        binding = profile_binding_refs.get(profile_id)
        if binding != specialist_descriptor_ref(candidate):
            reason = "PROFILE_BINDING_MISMATCH"
        elif requested not in descriptor["capability_classes"]:
            reason = "CAPABILITY_NOT_MATCHED"
        elif descriptor["enabled"] is not False:
            reason = "NONACTIVATION_POLICY_DENIED"
        else:
            reason = "DESCRIPTOR_DISABLED"
        rejections.append({"profile_id": profile_id, "reason_code": reason})

    return _finalize(
        {
            **base,
            "outcome": "blocked",
            "reason_code": "NO_DISPATCH_AUTHORITY",
            "requested_capability": requested,
            "candidate_rejections": rejections,
        }
    )