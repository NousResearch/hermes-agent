"""ORCH V4 Delivery protocol — obligations, claim, attempt, receipt, ACK family.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §10 (Delivery protocol) + §I.9 (adapter).
§13 T23-T33 test contracts.

Runtime-independent pure model. No live DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DeliveryError(ValueError):
    """Delivery protocol violation."""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# §10.5 ACK families — no global scalar rank.
ACK_FAMILIES = frozenset({"provider", "adapter", "synchronous", "local_session", "none"})

ACK_STRENGTHS = frozenset({"message_id", "adapter_acceptance", "response", "session_acceptance", "none"})


@dataclass(frozen=True)
class ManifestEntry:
    """Immutable manifest entry from orch_delivery_manifest_entries."""
    manifest_entry_key: str
    required: bool
    route_digest: str
    required_ack_family: str
    required_ack_strength: str


@dataclass
class DeliveryObligation:
    """Mutable obligation state (generation 1 starts pending)."""
    obligation_id: str
    manifest_entry_key: str
    required: bool
    route_digest: str
    required_ack_family: str
    required_ack_strength: str
    delivery_generation: int = 1
    state: str = "pending"  # pending, claimed, accepted, acked, unknown, dead_letter, cancelled
    claim_owner: str | None = None
    claim_token_hash: str | None = None
    claim_epoch: int = 0
    acceptance_attempt_id: int | None = None
    acked_at: int | None = None
    duplicate_possible: bool = False


@dataclass
class DeliveryAttempt:
    """Immutable attempt record (started → adapter_accepted/rejected/unknown)."""
    attempt_id: int
    obligation_id: str
    send_nonce: str
    payload_digest: str
    result_digest: str
    route_digest: str
    claim_owner: str
    claim_token_hash: str
    claim_epoch: int
    lifecycle_revision: int
    cancel_epoch: int
    state: str = "started"  # started, adapter_accepted, rejected, unknown
    finished_at: int | None = None
    adapter_evidence_digest: str | None = None


@dataclass(frozen=True)
class DeliveryReceipt:
    """Verified receipt bound to attempt/nonce/payload/result/route/family."""
    attempt_id: int
    obligation_id: str
    send_nonce: str
    payload_digest: str
    result_digest: str
    route_digest: str
    observed_ack_family: str
    observed_ack_strength: str
    verified: bool = True


def validate_ack_family(family: str, strength: str) -> None:
    """§10.5: Typed ACK family validation. No global rank."""
    if family not in ACK_FAMILIES:
        raise DeliveryError("invalid_ack_family")
    if strength not in ACK_STRENGTHS:
        raise DeliveryError("invalid_ack_strength")
    # Family-specific strength mapping
    expected = {
        "provider": "message_id",
        "adapter": "adapter_acceptance",
        "synchronous": "response",
        "local_session": "session_acceptance",
        "none": "none",
    }
    if strength != expected[family]:
        raise DeliveryError("ack_family_strength_mismatch")


def create_obligations_from_manifest(
    entries: list[ManifestEntry],
    *,
    origin_kind: str,
) -> list[DeliveryObligation]:
    """§10.2: Create obligations from manifest entries.

    Board-only with zero required entries creates no obligations.
    """
    if origin_kind == "board_only":
        required = [e for e in entries if e.required]
        if required:
            raise DeliveryError("board_only_has_required_obligations")
        return []  # board-only: no obligations

    obligations = []
    for i, entry in enumerate(entries):
        validate_ack_family(entry.required_ack_family, entry.required_ack_strength)
        obligations.append(DeliveryObligation(
            obligation_id=f"obl-{i+1}",
            manifest_entry_key=entry.manifest_entry_key,
            required=entry.required,
            route_digest=entry.route_digest,
            required_ack_family=entry.required_ack_family,
            required_ack_strength=entry.required_ack_strength,
        ))
    return obligations


def claim_obligation(
    obligation: DeliveryObligation,
    *,
    owner: str,
    token_hash: str,
    ttl: int,
    now: int,
) -> int:
    """§10.2: Claim CAS. Returns new claim_epoch."""
    if obligation.state != "pending":
        raise DeliveryError("claim_not_pending")
    obligation.state = "claimed"
    obligation.claim_owner = owner
    obligation.claim_token_hash = token_hash
    obligation.claim_epoch += 1
    return obligation.claim_epoch


def start_attempt(
    obligation: DeliveryObligation,
    *,
    attempt_id: int,
    send_nonce: str,
    payload_digest: str,
    result_digest: str,
    claim_owner: str,
    claim_token_hash: str,
    claim_epoch: int,
    lifecycle_revision: int,
    cancel_epoch: int,
) -> DeliveryAttempt:
    """§10.2: Pre-send intent — durable started marker."""
    if obligation.state != "claimed":
        raise DeliveryError("attempt_not_claimed")
    if obligation.claim_owner != claim_owner:
        raise DeliveryError("attempt_wrong_owner")
    if obligation.claim_token_hash != claim_token_hash:
        raise DeliveryError("attempt_wrong_token")
    if obligation.claim_epoch != claim_epoch:
        raise DeliveryError("attempt_wrong_epoch")
    return DeliveryAttempt(
        attempt_id=attempt_id,
        obligation_id=obligation.obligation_id,
        send_nonce=send_nonce,
        payload_digest=payload_digest,
        result_digest=result_digest,
        route_digest=obligation.route_digest,
        claim_owner=claim_owner,
        claim_token_hash=claim_token_hash,
        claim_epoch=claim_epoch,
        lifecycle_revision=lifecycle_revision,
        cancel_epoch=cancel_epoch,
    )


def finish_attempt(
    obligation: DeliveryObligation,
    attempt: DeliveryAttempt,
    *,
    terminal_state: str,
    evidence_digest: str | None = None,
    now: int = 0,
) -> DeliveryAttempt:
    """§10.3: CAS attempt to terminal state."""
    if attempt.state != "started":
        raise DeliveryError("attempt_already_finished")
    if attempt.obligation_id != obligation.obligation_id:
        raise DeliveryError("attempt_obligation_mismatch")
    if terminal_state not in ("adapter_accepted", "rejected", "unknown"):
        raise DeliveryError("invalid_attempt_terminal")
    attempt.state = terminal_state
    attempt.finished_at = now
    attempt.adapter_evidence_digest = evidence_digest
    return attempt


def process_receipt(
    obligation: DeliveryObligation,
    attempt: DeliveryAttempt,
    receipt: DeliveryReceipt,
    *,
    now: int = 0,
) -> None:
    """§10.3: Positive path — verify receipt, ACK obligation.

    Receipt must bind exact attempt/nonce/payload/result/route/family/strength.
    """
    if receipt.attempt_id != attempt.attempt_id:
        raise DeliveryError("receipt_attempt_mismatch")
    if receipt.obligation_id != obligation.obligation_id:
        raise DeliveryError("receipt_obligation_mismatch")
    if receipt.send_nonce != attempt.send_nonce:
        raise DeliveryError("receipt_nonce_mismatch")
    if receipt.payload_digest != attempt.payload_digest:
        raise DeliveryError("receipt_payload_mismatch")
    if receipt.result_digest != attempt.result_digest:
        raise DeliveryError("receipt_result_mismatch")
    if receipt.route_digest != obligation.route_digest:
        raise DeliveryError("receipt_route_mismatch")
    if receipt.observed_ack_family != obligation.required_ack_family:
        raise DeliveryError("cross_family_ack_satisfied")
    if receipt.observed_ack_strength != obligation.required_ack_strength:
        raise DeliveryError("ack_strength_mismatch")
    if not receipt.verified:
        raise DeliveryError("receipt_not_verified")

    # CAS: claimed or unknown → accepted → acked
    if obligation.state not in ("claimed", "unknown"):
        raise DeliveryError("ack_not_authorized")
    obligation.state = "acked"
    obligation.acceptance_attempt_id = attempt.attempt_id
    obligation.acked_at = now


def check_delivery_satisfied(
    obligations: list[DeliveryObligation],
    *,
    origin_kind: str,
) -> bool:
    """§10.6: Single authoritative completion predicate.

    Board-only: no required obligations → satisfied.
    Non-board-only: all required obligations must be acked.
    Optional obligations do not block.
    """
    if origin_kind == "board_only":
        required = [o for o in obligations if o.required]
        return len(required) == 0

    required = [o for o in obligations if o.required]
    if not required:
        return False  # non-board-only must have at least 1 required

    for o in required:
        if o.state != "acked":
            return False
        if o.acceptance_attempt_id is None:
            return False
    return True


def check_delivery_terminal_semantics(
    obligations: list[DeliveryObligation],
) -> dict[str, bool]:
    """§10.4 + T31: Required dead_letter/unknown/cancel never satisfy;
    optional non-blocking."""
    result = {}
    for o in obligations:
        if o.required:
            if o.state == "dead_letter":
                result[o.obligation_id] = False
            elif o.state == "unknown":
                result[o.obligation_id] = False
            elif o.state == "cancelled":
                result[o.obligation_id] = False
            elif o.state == "acked":
                result[o.obligation_id] = True
            else:
                result[o.obligation_id] = False
        else:
            # Optional: non-blocking
            result[o.obligation_id] = True
    return result


def authorize_resend(
    obligation: DeliveryObligation,
    *,
    new_generation: int,
) -> DeliveryObligation:
    """§10.4: Generation+1 obligation for unknown/dead-letter retry."""
    if obligation.state not in ("unknown", "dead_letter"):
        raise DeliveryError("resend_not_authorized")
    if new_generation != obligation.delivery_generation + 1:
        raise DeliveryError("invalid_resend_generation")
    return DeliveryObligation(
        obligation_id=f"{obligation.obligation_id}-g{new_generation}",
        manifest_entry_key=obligation.manifest_entry_key,
        required=obligation.required,
        route_digest=obligation.route_digest,
        required_ack_family=obligation.required_ack_family,
        required_ack_strength=obligation.required_ack_strength,
        delivery_generation=new_generation,
        state="pending",
        duplicate_possible=True,
    )


__all__ = [
    "DeliveryError", "ACK_FAMILIES", "ACK_STRENGTHS",
    "ManifestEntry", "DeliveryObligation", "DeliveryAttempt",
    "DeliveryReceipt", "validate_ack_family",
    "create_obligations_from_manifest", "claim_obligation",
    "start_attempt", "finish_attempt", "process_receipt",
    "check_delivery_satisfied", "check_delivery_terminal_semantics",
    "authorize_resend",
]
