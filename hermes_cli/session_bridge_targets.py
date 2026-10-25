"""Validate provider-issued, ticket-bound live-session target receipts.

This module deliberately has no persistence or public registration surface.
Only a concrete adapter that can authenticate its provider may eventually
persist receipts and supply a verifier. Caller-asserted eligibility flags,
ticket-shaped strings, transcript metadata, and same-UID database writes are
not authority to address a live session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Iterable


_TICKET_REF = re.compile(r"^[A-Z][A-Z0-9]{1,15}-[0-9]{3,}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_OBSERVATION_AGE = 60.0


class TargetUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TransportPolicy:
    provider: str
    backend: str
    transport: str
    delivery_mode: str
    reply_mode: str
    delivery_retraction_mode: str
    external_hermes_supported: bool
    source_status: str


# Closed registry: new transports require a source-backed adapter and tests.
TRANSPORT_POLICIES: dict[tuple[str, str], TransportPolicy] = {
    ("claude-code", "claude-team-mailbox"): TransportPolicy(
        "claude-code", "claude", "claude-team-mailbox", "safe-point-queue",
        "same-team-mailbox", "none", False, "official-intra-team-only",
    ),
    ("claude-code", "claude-peer-socket"): TransportPolicy(
        "claude-code", "claude", "claude-peer-socket", "safe-point-queue",
        "peer-reply-address", "none", False,
        "official ingress; external reply envelope undocumented",
    ),
    ("claude-code", "claude-mcp-channel"): TransportPolicy(
        "claude-code", "claude", "claude-mcp-channel", "safe-point-queue",
        "correlated-reply-tool", "none", False,
        "official research preview; startup-bound; no queued-message cancellation",
    ),
    ("codex-app", "codex-app-thread"): TransportPolicy(
        "codex-app", "codex", "codex-app-thread", "safe-point-queue",
        "thread-read-wait", "none", False,
        "official authenticated surface is callable only by the owning Codex app host",
    ),
    ("codex-app-server", "codex-app-server-owned-thread"): TransportPolicy(
        "codex-app-server", "codex", "codex-app-server-owned-thread", "steer",
        "app-server-events", "active-turn-interrupt", False,
        "official protocol; only the owning client may drive the thread",
    ),
}


@dataclass(frozen=True)
class ProviderTargetReceipt:
    receipt_id: str
    ticket_ref: str
    assignment_id: str
    attempt_id: str
    assignment_state: str
    attempt_state: str
    provider: str
    backend: str
    transport: str
    provider_instance_id: str
    profile_scope: str
    native_session_id: str
    delivery_handle: str
    agent_id: str
    team_name: str | None
    owner_identity: str
    ticket_revision: int
    authority_event_sha256: str
    authority_scope: str
    identity_state: str
    transport_authorization_state: str
    safe_delivery_state: str
    reply_state: str
    caller_cancellation_state: str
    observed_at: float
    expires_at: float
    revoked: bool
    provenance: dict[str, Any]
    binding_sha256: str
    payload_sha256: str
    provider_signature: str


@dataclass(frozen=True)
class RootsRouteAuthority:
    """The exact active Roots assignment/attempt a provider receipt must bind.

    A future concrete adapter must construct this from an authenticated Roots
    ticket read.  It is a validation input, not a caller registration surface.
    """

    ticket_ref: str
    assignment_id: str
    attempt_id: str
    owner_identity: str
    profile_scope: str
    ticket_revision: int
    authority_event_sha256: str


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetUnavailable(f"{name} is missing")
    return value.strip()


def _canonical_payload(receipt: ProviderTargetReceipt) -> bytes:
    value = asdict(receipt)
    value.pop("payload_sha256")
    value.pop("provider_signature")
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def receipt_payload_sha256(receipt: ProviderTargetReceipt) -> str:
    return hashlib.sha256(_canonical_payload(receipt)).hexdigest()


def receipt_binding_sha256(receipt: ProviderTargetReceipt) -> str:
    bound = {
        "ticket_ref": receipt.ticket_ref,
        "assignment_id": receipt.assignment_id,
        "attempt_id": receipt.attempt_id,
        "provider": receipt.provider,
        "provider_instance_id": receipt.provider_instance_id,
        "profile_scope": receipt.profile_scope,
        "native_session_id": receipt.native_session_id,
        "delivery_handle": receipt.delivery_handle,
        "agent_id": receipt.agent_id,
        "team_name": receipt.team_name,
        "owner_identity": receipt.owner_identity,
        "ticket_revision": receipt.ticket_revision,
        "authority_event_sha256": receipt.authority_event_sha256,
    }
    encoded = json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_authority(
    receipt: ProviderTargetReceipt,
    verify_provider_signature: Callable[[ProviderTargetReceipt], bool],
) -> TransportPolicy:
    ref = _required_text(receipt.ticket_ref, "ticket_ref").upper()
    if ref != receipt.ticket_ref or not _TICKET_REF.fullmatch(ref):
        raise TargetUnavailable("receipt has an invalid canonical ticket reference")
    for name in (
        "receipt_id", "assignment_id", "attempt_id", "provider", "backend",
        "transport", "provider_instance_id", "profile_scope", "native_session_id",
        "delivery_handle", "agent_id", "owner_identity", "authority_event_sha256",
        "provider_signature",
    ):
        _required_text(getattr(receipt, name), name)
    policy = TRANSPORT_POLICIES.get((receipt.provider, receipt.transport))
    if policy is None or policy.backend != receipt.backend:
        raise TargetUnavailable("receipt provider/transport/backend is not registered")
    if receipt.native_session_id == receipt.delivery_handle:
        raise TargetUnavailable("native session identity must differ from its delivery handle")
    if receipt.transport == "claude-team-mailbox" and not receipt.team_name:
        raise TargetUnavailable("Claude team mailbox receipts require a team binding")
    if (
        not isinstance(receipt.ticket_revision, int)
        or isinstance(receipt.ticket_revision, bool)
        or receipt.ticket_revision < 1
    ):
        raise TargetUnavailable("receipt requires a positive ticket revision")
    if not _SHA256.fullmatch(receipt.authority_event_sha256):
        raise TargetUnavailable("receipt requires the authoritative Roots event digest")
    if (
        not isinstance(receipt.observed_at, (int, float))
        or not isinstance(receipt.expires_at, (int, float))
        or not math.isfinite(receipt.observed_at)
        or not math.isfinite(receipt.expires_at)
        or receipt.expires_at <= receipt.observed_at
    ):
        raise TargetUnavailable("receipt has an invalid observation lifetime")
    if not isinstance(receipt.provenance, dict) or not receipt.provenance:
        raise TargetUnavailable("receipt requires provider provenance")
    if receipt.authority_scope != "answer-only-no-changes":
        raise TargetUnavailable("receipt authority is not bounded to a no-change answer")
    if receipt.binding_sha256 != receipt_binding_sha256(receipt):
        raise TargetUnavailable("receipt target binding digest does not match")
    if receipt.payload_sha256 != receipt_payload_sha256(receipt):
        raise TargetUnavailable("receipt payload digest does not match")
    if not verify_provider_signature(receipt):
        raise TargetUnavailable("receipt provider signature is not trusted")
    return policy


def resolve_provider_receipts(
    route: RootsRouteAuthority,
    receipts: Iterable[ProviderTargetReceipt],
    *,
    verify_provider_signature: Callable[[ProviderTargetReceipt], bool],
    now: float | None = None,
    require_cancellation: bool = True,
) -> ProviderTargetReceipt:
    """Resolve one provider-authenticated active attempt, or fail closed.

    Authority is checked before receipts are ordered. For each authoritative
    assignment/attempt, the newest receipt is the complete current state; an
    inactive, revoked, expired, or capability-reduced receipt is a tombstone
    and cannot expose an older positive observation.
    """
    ref = _required_text(route.ticket_ref, "ticket_ref").upper()
    if not _TICKET_REF.fullmatch(ref):
        raise ValueError("ticket_ref has an unsupported format")
    if ref != route.ticket_ref:
        raise ValueError("ticket_ref must be canonical")
    for name in (
        "assignment_id", "attempt_id", "owner_identity", "profile_scope",
        "authority_event_sha256",
    ):
        _required_text(getattr(route, name), name)
    if (
        not isinstance(route.ticket_revision, int)
        or isinstance(route.ticket_revision, bool)
        or route.ticket_revision < 1
    ):
        raise ValueError("ticket_revision must be positive")
    if not _SHA256.fullmatch(route.authority_event_sha256):
        raise ValueError("authority_event_sha256 must be a lowercase SHA-256 digest")
    when = time.time() if now is None else now
    latest: dict[
        tuple[str, str], list[tuple[ProviderTargetReceipt, TransportPolicy]]
    ] = {}
    authority_errors: list[str] = []
    for receipt in receipts:
        if receipt.ticket_ref != ref:
            continue
        try:
            policy = _verify_authority(receipt, verify_provider_signature)
        except TargetUnavailable as exc:
            authority_errors.append(str(exc))
            continue
        if (
            receipt.assignment_id != route.assignment_id
            or receipt.attempt_id != route.attempt_id
            or receipt.owner_identity != route.owner_identity
            or receipt.profile_scope != route.profile_scope
            or receipt.ticket_revision != route.ticket_revision
            or receipt.authority_event_sha256 != route.authority_event_sha256
        ):
            authority_errors.append(
                "receipt does not bind the exact current Roots assignment, "
                "attempt, owner, profile, and ticket revision"
            )
            continue
        key = (receipt.assignment_id, receipt.attempt_id)
        previous = latest.get(key)
        if previous is None or receipt.observed_at > previous[0][0].observed_at:
            latest[key] = [(receipt, policy)]
        elif receipt.observed_at == previous[0][0].observed_at:
            # Without a provider-issued monotonic sequence, distinct states at
            # the same timestamp are unordered.  Compare the signed payload,
            # not only the stable target binding: revocation and capability
            # reductions deliberately keep that binding unchanged.
            if all(
                receipt.payload_sha256 != item[0].payload_sha256
                for item in previous
            ):
                previous.append((receipt, policy))

    if any(len(candidates) > 1 for candidates in latest.values()):
        raise TargetUnavailable(
            f"ticket {ref} has conflicting latest authoritative provider targets; "
            "routing is ambiguous"
        )

    eligible: list[ProviderTargetReceipt] = []
    reasons: set[str] = set()
    for candidates in latest.values():
        receipt, policy = candidates[0]
        if receipt.revoked:
            reasons.add("latest authoritative receipt is revoked")
            continue
        if receipt.assignment_state != "active" or receipt.attempt_state != "active":
            reasons.add("latest authoritative assignment or attempt is inactive")
            continue
        runtime_states = {
            "identity_state": (receipt.identity_state, "verified"),
            "transport_authorization_state": (
                receipt.transport_authorization_state, "authorized",
            ),
            "safe_delivery_state": (receipt.safe_delivery_state, "safe-point-queue"),
            "reply_state": (receipt.reply_state, "correlated"),
        }
        unavailable = [name for name, (actual, required) in runtime_states.items() if actual != required]
        if unavailable:
            reasons.add("latest authoritative receipt lacks " + ", ".join(unavailable))
            continue
        if (
            receipt.observed_at > when
            or when - receipt.observed_at > _MAX_OBSERVATION_AGE
            or receipt.expires_at <= when
        ):
            reasons.add("latest authoritative receipt is stale or expired")
            continue
        if require_cancellation and receipt.caller_cancellation_state != "stop-wait-and-invalidate-late":
            reasons.add(
                f"{policy.transport} lacks caller_cancellation_state "
                "stop-wait-and-invalidate-late"
            )
            continue
        if not policy.external_hermes_supported:
            reasons.add(f"{policy.transport} is unavailable to external Hermes: {policy.source_status}")
            continue
        if policy.delivery_mode != "safe-point-queue":
            reasons.add(f"{policy.transport} does not provide non-disruptive queue delivery")
            continue
        if policy.reply_mode not in {"correlated-reply-tool", "thread-read-wait"}:
            reasons.add(f"{policy.transport} lacks a supported correlated reply path")
            continue
        eligible.append(receipt)

    if len(eligible) == 1:
        return eligible[0]
    if len(eligible) > 1:
        raise TargetUnavailable(
            f"ticket {ref} has multiple eligible active provider attempts; routing is ambiguous"
        )
    detail = "; ".join(sorted(reasons or set(authority_errors)))
    if not detail:
        detail = "no provider-issued authoritative receipt"
    raise TargetUnavailable(f"ticket {ref} has no eligible active session target: {detail}")
