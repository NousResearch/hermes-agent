"""Canonical Writer-owned authority for privileged Discord route-back.

The writer owns the capability signing key and pins the Discord edge receipt
verification key.  Callers may supply an exact public-send intent and may
return exact edge protocol evidence, but they cannot assert adapter success or
readback truth themselves.

This layer is intentionally semantic-free.  It validates fixed protocol
shapes and exact bindings, returns a fresh short-lived capability for an
eligible nonterminal durable claim, and derives the legacy Canonical Writer
receipt shape from verified edge evidence.  The edge journal, not capability
freshness, is the one-mutation idempotency fence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import time
from typing import Any, Callable, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.discord_edge_protocol import (
    DiscordEdgeAuthorityKind,
    DiscordEdgeCapability,
    DiscordEdgeErrorCode,
    DiscordEdgeIntent,
    DiscordEdgeOperation,
    DiscordEdgeProtocolError,
    DiscordEdgeReceipt,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeRequest,
    SignedDiscordEdgeEnvelope,
    canonical_json_bytes,
    make_request,
    parse_request_for_reconciliation,
    sign_capability,
    verify_receipt,
    verify_request_capability_for_reconciliation,
)


def derive_routeback_edge_idempotency_key(
    *,
    case_id: str,
    canonical_idempotency_key: str,
) -> str:
    """Scope the edge's global idempotency fence to one Canonical lifecycle."""

    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id is required for Discord edge idempotency")
    if (
        not isinstance(canonical_idempotency_key, str)
        or not canonical_idempotency_key
        or len(canonical_idempotency_key.encode("utf-8")) > 256
    ):
        raise ValueError("canonical route-back idempotency key is invalid")
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "case_id": case_id,
                "canonical_idempotency_key": canonical_idempotency_key,
            }
        )
    ).hexdigest()
    return "canonical-routeback:" + digest


class DiscordEdgeWriterAuthorityError(ValueError):
    """Secret-free, fixed-code writer authority failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class VerifiedDiscordEdgeEvidence:
    """Verified edge evidence and its derived legacy writer representation."""

    request: DiscordEdgeRequest
    capability: DiscordEdgeCapability
    receipt: DiscordEdgeReceipt
    canonical_receipt: Mapping[str, Any]
    blocker_reason: str


class CanonicalWriterDiscordAuthority:
    """Issue route-back capabilities and verify edge-signed delivery truth."""

    def __init__(
        self,
        *,
        capability_private_key: Ed25519PrivateKey,
        edge_receipt_public_key: Ed25519PublicKey,
        clock_unix_ms: Callable[[], int] | None = None,
        request_timeout_seconds: int = 15,
    ) -> None:
        if not isinstance(capability_private_key, Ed25519PrivateKey):
            raise TypeError("capability_private_key must be Ed25519PrivateKey")
        if not isinstance(edge_receipt_public_key, Ed25519PublicKey):
            raise TypeError("edge_receipt_public_key must be Ed25519PublicKey")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, int)
            or not 1 <= request_timeout_seconds <= 30
        ):
            raise ValueError("request_timeout_seconds must be between 1 and 30")
        self._capability_private_key = capability_private_key
        self._edge_receipt_public_key = edge_receipt_public_key
        self._clock_unix_ms = clock_unix_ms or (lambda: int(time.time() * 1000))
        self._request_timeout_seconds = request_timeout_seconds
        self._sequence = 0
        self._last_issued_at_unix_ms = 0
        self._sequence_lock = threading.Lock()

    @property
    def capability_public_key(self) -> Ed25519PublicKey:
        return self._capability_private_key.public_key()

    @staticmethod
    def parse_public_send_intent(value: Any) -> DiscordEdgeIntent:
        """Parse the exact public-message-send intent accepted for route-back."""

        try:
            if not isinstance(value, Mapping):
                raise DiscordEdgeWriterAuthorityError(
                    "invalid_discord_edge_intent",
                    "discord_edge_intent must be an exact object",
                )
            required = {"operation", "target", "payload", "idempotency_key"}
            if set(value) != required:
                raise DiscordEdgeWriterAuthorityError(
                    "invalid_discord_edge_intent",
                    "discord_edge_intent has an invalid exact shape",
                )
            intent = DiscordEdgeIntent.from_parts(
                operation=value["operation"],
                target=value["target"],
                payload=value["payload"],
                idempotency_key=value["idempotency_key"],
            )
        except DiscordEdgeWriterAuthorityError:
            raise
        except DiscordEdgeProtocolError as exc:
            raise DiscordEdgeWriterAuthorityError(
                "invalid_discord_edge_intent",
                "discord_edge_intent failed the public Discord protocol boundary",
            ) from exc
        if intent.operation is not DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
            raise DiscordEdgeWriterAuthorityError(
                "invalid_discord_edge_intent",
                "canonical route-back permits only public.message.send",
            )
        return intent

    def issue_routeback_request(
        self,
        intent: DiscordEdgeIntent,
        *,
        authorization_id: str,
    ) -> DiscordEdgeRequest:
        """Return one full request carrying writer-owned route-back authority."""

        if intent.operation is not DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
            raise DiscordEdgeWriterAuthorityError(
                "invalid_discord_edge_intent",
                "canonical route-back permits only public.message.send",
            )
        now = self._clock_unix_ms()
        if isinstance(now, bool) or not isinstance(now, int) or now < 1:
            raise DiscordEdgeWriterAuthorityError(
                "discord_edge_authority_unavailable",
                "writer authority clock is unavailable",
            )
        with self._sequence_lock:
            now = max(now, self._last_issued_at_unix_ms + 1)
            self._last_issued_at_unix_ms = now
            self._sequence += 1
            sequence = self._sequence
        try:
            capability = sign_capability(
                self._capability_private_key,
                intent,
                authority_kind=DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
                authority_ref=authorization_id,
                issued_at_unix_ms=now,
                expires_at_unix_ms=(
                    now + self._request_timeout_seconds * 1_000
                ),
            )
            return make_request(
                intent,
                capability,
                sequence=sequence,
                timeout_seconds=self._request_timeout_seconds,
                now_unix_ms=now,
            )
        except DiscordEdgeProtocolError as exc:
            raise DiscordEdgeWriterAuthorityError(
                "discord_edge_authority_unavailable",
                "writer could not issue the bounded Discord edge request",
            ) from exc

    def verify_routeback_evidence(
        self,
        *,
        request_value: Any,
        receipt_value: Any,
        authorization_id: str,
    ) -> VerifiedDiscordEdgeEvidence:
        """Verify writer capability, edge signature, and their exact binding."""

        now = self._clock_unix_ms()
        if isinstance(now, bool) or not isinstance(now, int) or now < 1:
            raise DiscordEdgeWriterAuthorityError(
                "discord_edge_authority_unavailable",
                "writer authority clock is unavailable",
            )
        try:
            request = parse_request_for_reconciliation(request_value)
            if request.intent.operation is not DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
                raise DiscordEdgeWriterAuthorityError(
                    "invalid_discord_edge_evidence",
                    "route-back evidence is not a public message send",
                )
            capability = verify_request_capability_for_reconciliation(
                request,
                self.capability_public_key,
            )
            if (
                capability.authority_kind
                is not DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK
                or capability.authority_ref != authorization_id
            ):
                raise DiscordEdgeWriterAuthorityError(
                    "invalid_discord_edge_evidence",
                    "writer capability is not bound to this route-back authorization",
                )
            receipt_envelope = SignedDiscordEdgeEnvelope.from_mapping(
                receipt_value,
                code=DiscordEdgeErrorCode.INVALID_RECEIPT,
                label="edge receipt",
            )
            receipt = verify_receipt(
                receipt_envelope,
                self._edge_receipt_public_key,
                expected_request=request,
                expected_capability=capability,
                now_unix_ms=now,
            )
        except DiscordEdgeWriterAuthorityError:
            raise
        except (DiscordEdgeProtocolError, TypeError, ValueError) as exc:
            raise DiscordEdgeWriterAuthorityError(
                "invalid_discord_edge_evidence",
                "Discord edge request or receipt evidence is invalid",
            ) from exc

        canonical_receipt: dict[str, Any] = {}
        blocker_reason = ""
        if receipt.outcome is DiscordEdgeReceiptOutcome.VERIFIED:
            canonical_receipt = {
                "platform": "discord",
                "adapter_receipt": True,
                "receipt_readback_verified": True,
                "message_id": receipt.discord_object_id,
                "channel_id": receipt.target.channel_id,
                "content_sha256": receipt.content_sha256,
            }
        else:
            if receipt.blocker_code is None:
                raise DiscordEdgeWriterAuthorityError(
                    "invalid_discord_edge_evidence",
                    "non-verified edge receipt is missing a blocker code",
                )
            blocker_reason = (
                "discord_edge:"
                + receipt.outcome.value
                + ":"
                + receipt.blocker_code
            )
            if (
                receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
                and receipt.discord_object_id is not None
            ):
                canonical_receipt = {
                    "platform": "discord",
                    "adapter_receipt": True,
                    "receipt_readback_verified": False,
                    "message_id": receipt.discord_object_id,
                    "channel_id": receipt.target.channel_id,
                    "content_sha256": receipt.content_sha256,
                }
        return VerifiedDiscordEdgeEvidence(
            request=request,
            capability=capability,
            receipt=receipt,
            canonical_receipt=canonical_receipt,
            blocker_reason=blocker_reason,
        )


__all__ = [
    "CanonicalWriterDiscordAuthority",
    "DiscordEdgeWriterAuthorityError",
    "VerifiedDiscordEdgeEvidence",
    "derive_routeback_edge_idempotency_key",
]
