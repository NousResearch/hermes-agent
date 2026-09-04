"""Provider-free orchestration for local specialist capability discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gateway.candidate_profile_requests import CandidateProfileRequests, SanitizedTaskEnvelope
from gateway.capability_registry import CapabilityRegistry, CapabilitySignature


@dataclass(frozen=True, slots=True)
class SpecialistDiscoveryDecision:
    """One fail-closed local resolution or inert candidate result."""

    status: Literal["active_match", "ambiguous", "candidate", "rejected", "unavailable"]
    profile: str | None
    candidate_request_id: str | None
    reason: str


class SpecialistDiscovery:
    """Resolve configured specialists and record bounded local gaps."""

    def __init__(self, *, db_path: Path | None = None, board: str | None = None) -> None:
        self._registry = CapabilityRegistry(db_path=db_path, board=board)
        self._candidates = CandidateProfileRequests(db_path=db_path, board=board)

    def resolve_or_request(
        self,
        signature: CapabilitySignature,
        *,
        source_key: str,
        envelope: SanitizedTaskEnvelope | None = None,
    ) -> SpecialistDiscoveryDecision:
        """Return an exact configured match or an inert local request."""
        resolution = self._registry.resolve(signature)
        if resolution.status == "active_match":
            return SpecialistDiscoveryDecision(
                status="active_match",
                profile=resolution.profile,
                candidate_request_id=None,
                reason=resolution.reason,
            )
        if resolution.status in {"ambiguous", "unavailable"}:
            return SpecialistDiscoveryDecision(
                status=resolution.status,
                profile=None,
                candidate_request_id=None,
                reason=resolution.reason,
            )

        candidate = self._candidates.open_or_reuse(
            signature,
            source_key=source_key,
            resolution=resolution,
            envelope=envelope,
        )
        if candidate.status in {"candidate", "duplicate"}:
            return SpecialistDiscoveryDecision(
                status="candidate",
                profile=None,
                candidate_request_id=candidate.request_id,
                reason=candidate.reason,
            )
        return SpecialistDiscoveryDecision(
            status="rejected",
            profile=None,
            candidate_request_id=candidate.request_id or None,
            reason=candidate.reason,
        )
