"""Short-lived exact-intent capability leases and effect re-verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta

from .errors import LeaseError, LeaseExpired, LeaseTampered
from .model import (
    ActionIntent,
    CapabilityLease,
    DecisionOutcome,
    PolicyDecision,
    Principal,
    canonical_json,
    new_id,
    parse_utc,
    utc_now,
    utc_text,
)


_VERSION = "ht1"
_PAYLOAD_KEYS = {
    "lease_id",
    "principal_id",
    "continuity_id",
    "capability",
    "intent_digest",
    "scope_digest",
    "decision_id",
    "obligations",
    "budget_reservation_id",
    "approval_id",
    "issued_at",
    "expires_at",
    "nonce",
}


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise LeaseTampered("lease contains invalid base64url") from exc


class LeaseAuthority:
    """Issue and verify HMAC-SHA256 leases.

    Signing material is supplied by a secret resolver and never persisted in
    SQLite or serialized into receipts.
    """

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: int = 120,
        clock_skew_seconds: int = 5,
    ) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("lease signing secret must contain at least 32 bytes")
        if not 5 <= ttl_seconds <= 3600:
            raise ValueError("lease ttl_seconds must be between 5 and 3600")
        if not 0 <= clock_skew_seconds <= 60:
            raise ValueError("lease clock skew must be between 0 and 60 seconds")
        self._secret = secret
        self.ttl_seconds = ttl_seconds
        self.clock_skew_seconds = clock_skew_seconds

    def issue(
        self,
        principal: Principal,
        intent: ActionIntent,
        decision: PolicyDecision,
        *,
        now: datetime | None = None,
    ) -> tuple[CapabilityLease, str]:
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise LeaseError("authority may be issued only for an allow decision")
        if decision.capability != intent.capability:
            raise LeaseTampered("decision capability does not match intent")
        if decision.intent_digest != intent.digest:
            raise LeaseTampered("decision intent digest does not match intent")
        if decision.scope_digest != intent.scope.digest:
            raise LeaseTampered("decision scope digest does not match intent")
        if principal.principal_id != intent.scope.principal_id:
            raise LeaseTampered("principal does not own the intent scope")

        current = now or utc_now()
        lease = CapabilityLease(
            lease_id=new_id("lease"),
            principal_id=principal.principal_id,
            continuity_id=intent.scope.continuity_id,
            capability=intent.capability,
            intent_digest=intent.digest,
            scope_digest=intent.scope.digest,
            decision_id=decision.decision_id,
            obligations=decision.obligations,
            budget_reservation_id=decision.budget_reservation_id,
            approval_id=decision.approval_id,
            issued_at=utc_text(current),
            expires_at=utc_text(current + timedelta(seconds=self.ttl_seconds)),
            nonce=secrets.token_urlsafe(24),
        )
        return lease, self.encode(lease)

    def encode(self, lease: CapabilityLease) -> str:
        payload = canonical_json(lease).encode("utf-8")
        encoded = _b64encode(payload)
        signing_input = f"{_VERSION}.{encoded}".encode("ascii")
        signature = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        return f"{_VERSION}.{encoded}.{_b64encode(signature)}"

    def decode(self, token: str) -> CapabilityLease:
        if not isinstance(token, str) or len(token) > 16384:
            raise LeaseTampered("lease token has invalid shape")
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != _VERSION:
            raise LeaseTampered("lease token has unsupported version")
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        actual_signature = _b64decode(parts[2])
        expected_signature = hmac.new(
            self._secret, signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise LeaseTampered("lease signature is invalid")
        raw_payload = _b64decode(parts[1])
        if len(raw_payload) > 8192:
            raise LeaseTampered("lease payload exceeds size limit")
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LeaseTampered("lease payload is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            raise LeaseTampered("lease payload fields are invalid")
        if not isinstance(payload.get("obligations"), list) or not all(
            isinstance(item, str) for item in payload["obligations"]
        ):
            raise LeaseTampered("lease obligations are invalid")
        try:
            return CapabilityLease(
                lease_id=payload["lease_id"],
                principal_id=payload["principal_id"],
                continuity_id=payload["continuity_id"],
                capability=payload["capability"],
                intent_digest=payload["intent_digest"],
                scope_digest=payload["scope_digest"],
                decision_id=payload["decision_id"],
                obligations=tuple(payload["obligations"]),
                budget_reservation_id=payload["budget_reservation_id"],
                approval_id=payload["approval_id"],
                issued_at=payload["issued_at"],
                expires_at=payload["expires_at"],
                nonce=payload["nonce"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LeaseTampered("lease payload values are invalid") from exc

    def verify(
        self,
        token: str,
        intent: ActionIntent,
        *,
        principal_id: str | None = None,
        decision_id: str | None = None,
        now: datetime | None = None,
    ) -> CapabilityLease:
        lease = self.decode(token)
        current = now or utc_now()
        skew = timedelta(seconds=self.clock_skew_seconds)
        issued = parse_utc(lease.issued_at)
        expires = parse_utc(lease.expires_at)
        if issued > current + skew:
            raise LeaseTampered("lease issuance is in the future")
        if expires <= current - skew:
            raise LeaseExpired("capability lease has expired")
        if lease.capability != intent.capability:
            raise LeaseTampered("lease capability does not match effect")
        if lease.intent_digest != intent.digest:
            raise LeaseTampered("lease does not authorize these exact arguments")
        if lease.scope_digest != intent.scope.digest:
            raise LeaseTampered("lease does not authorize this exact scope")
        expected_principal = principal_id or intent.scope.principal_id
        if lease.principal_id != expected_principal:
            raise LeaseTampered("lease principal does not match effect actor")
        if decision_id is not None and lease.decision_id != decision_id:
            raise LeaseTampered("lease decision linkage is invalid")
        if lease.continuity_id != intent.scope.continuity_id:
            raise LeaseTampered("lease continuity does not match effect scope")
        return lease
