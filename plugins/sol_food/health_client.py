"""Sol-side transport client for the Health food-commit v3 contract.

Conforms to the root-accepted normative contract
``health-food-commit-idempotency-contract-v3`` (SHA-256
``2edd5bae543feb1ad8ae1ef46b40ac0fedc6a274e0af18360fb223a03d582c38``)
and to the merged Health implementation's exact envelope key sets.

Boundary rules enforced here (transport side):

- Health is the ONLY canonical commit boundary. This client is transport
  and presentation only: it never talks to Supabase, carries no Supabase
  JWT/service key/owner UUID, and the request schema has no user/owner
  field at all — the Health service derives the owner from its protected
  server-side binding.
- The endpoint is Health-local and loopback-only; this client refuses to
  construct itself against any non-loopback host.
- ``HEALTH_FOOD_COMMIT_TOKEN`` is exactly 43 canonical unpadded base64url
  ASCII characters decoding to exactly 32 bytes; blank, malformed,
  padded, or noncanonical values fail closed before any request is built.
  Comparison of decoded values is the server's job; the client validates
  canonical shape so a misconfigured value can never leave the process.
- Exactly one ``Authorization: Bearer <token>`` header; body is compact
  canonical UTF-8 JSON of at most 8,192 bytes.
- The commit envelope is built ONCE per confirmed proposal, and its
  canonical bytes + mutation identity are durably frozen BEFORE the first
  send. A timeout or lost response never authorizes a new mutation:
  reconciliation retries the exact frozen bytes and validates Health's
  immutable receipt + authenticated readback. Exact retry yields the
  original receipt with response-level ``replayed`` metadata and one
  effect.
- Acknowledgement fails closed when the receipt or the separate
  authenticated post-commit readback is missing or mismatched.
- Nothing here logs tokens, bodies, receipts, or food values — stable
  reason codes only. The value-free ``receipt_sha256`` is the only
  artifact handed back for durable linkage.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from http.client import HTTPConnection
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from plugins.sol_food.limits import (
    FOOD_CANDIDATE_MAX_ITEMS,
    HEALTH_FOOD_BODY_MAX_BYTES,
    HEALTH_FOOD_TOKEN_BYTES,
    HEALTH_FOOD_TOKEN_CHARS,
)

__all__ = [
    "HealthClientError",
    "validate_commit_token",
    "build_commit_envelope",
    "FrozenEnvelope",
    "VerifiedCommit",
    "HealthFoodClient",
]

REQUEST_SCHEMA = "health.food_commit.v1"
PAYLOAD_VERSION = "health.food_meal.v1"
RESPONSE_SCHEMA = "health.food_commit_response.v1"
RECEIPT_SCHEMA = "health.food_commit_receipt.v1"

_TOKEN_RE = re.compile(r"\A[A-Za-z0-9_-]{%d}\Z" % HEALTH_FOOD_TOKEN_CHARS)
_UUID_RE = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_PLANT_KEY_RE = re.compile(r"\A[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_TIMESTAMP_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z")
_MAX_PLANT_KEY_BYTES = 64
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_MAX_RESPONSE_BYTES = 256 * 1024

_RESPONSE_KEYS = frozenset(
    {
        "schema",
        "receipt_schema",
        "receipt_canonical_json_b64",
        "receipt_sha256",
        "replayed",
        "post_commit_readback_sha256",
        "historical_commit_verified",
        "latest_projection_consistent",
    }
)

# Stable value-free reason codes.
REASON_BAD_TOKEN = "health_client_bad_token"
REASON_BAD_ENDPOINT = "health_client_bad_endpoint"
REASON_BAD_REQUEST_SHAPE = "health_client_bad_request_shape"
REASON_REQUEST_TOO_LARGE = "health_client_request_too_large"
REASON_TRANSPORT = "health_client_transport_error"
REASON_HTTP_STATUS = "health_client_http_status"
REASON_BAD_RESPONSE = "health_client_bad_response"
REASON_RECEIPT_MISMATCH = "health_client_receipt_mismatch"
REASON_READBACK_FAILED = "health_client_readback_failed"


class HealthClientError(Exception):
    """Value-free client failure. ``retryable`` means: retry the SAME
    frozen envelope; never build a new mutation."""

    def __init__(self, reason_code: str, *, retryable: bool = False) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.retryable = retryable


def validate_commit_token(value: object) -> str:
    """Validate HEALTH_FOOD_COMMIT_TOKEN canonical shape; return it.

    43 unpadded canonical base64url chars decoding (with required internal
    padding) to exactly 32 bytes, whose unpadded re-encoding equals the
    original string. Anything else fails closed.
    """
    if not isinstance(value, str) or not _TOKEN_RE.match(value):
        raise HealthClientError(REASON_BAD_TOKEN)
    try:
        decoded = base64.urlsafe_b64decode(value + "=")
    except (ValueError, TypeError):
        raise HealthClientError(REASON_BAD_TOKEN) from None
    if len(decoded) != HEALTH_FOOD_TOKEN_BYTES:
        raise HealthClientError(REASON_BAD_TOKEN)
    if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != value:
        raise HealthClientError(REASON_BAD_TOKEN)
    return value


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True)
class FrozenEnvelope:
    """One immutable commit request: canonical bytes + identity.

    Instances are durably persisted by the caller BEFORE first send and
    replayed byte-for-byte on retry.
    """

    mutation_id: str
    entry_id: str
    operation: str
    expected_revision: int
    request_bytes: bytes
    request_sha256: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "entry_id": self.entry_id,
            "operation": self.operation,
            "expected_revision": self.expected_revision,
            "request_b64": base64.b64encode(self.request_bytes).decode("ascii"),
            "request_sha256": self.request_sha256,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "FrozenEnvelope":
        request_bytes = base64.b64decode(str(data["request_b64"]))
        digest = hashlib.sha256(request_bytes).hexdigest()
        if digest != str(data["request_sha256"]):
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
        return cls(
            mutation_id=str(data["mutation_id"]),
            entry_id=str(data["entry_id"]),
            operation=str(data["operation"]),
            expected_revision=int(data["expected_revision"]),
            request_bytes=request_bytes,
            request_sha256=digest,
        )


def _validate_items(items: Sequence[Mapping[str, Any]]) -> None:
    if not 1 <= len(items) <= FOOD_CANDIDATE_MAX_ITEMS:
        raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
    for item in items:
        if set(item.keys()) != {"plant_key", "is_plant"}:
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
        key = item["plant_key"]
        if not isinstance(key, str) or not _PLANT_KEY_RE.match(key):
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
        if len(key.encode("utf-8")) > _MAX_PLANT_KEY_BYTES:
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
        if not isinstance(item["is_plant"], bool):
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)


def build_commit_envelope(
    *,
    operation: str,
    occurred_at: str,
    items: Optional[Sequence[Mapping[str, Any]]],
    expected_revision: int,
    entry_id: Optional[str] = None,
    mutation_id: Optional[str] = None,
) -> FrozenEnvelope:
    """Build + validate one canonical ``health.food_commit.v1`` request.

    ``create`` uses ``expected_revision=0`` and a fresh ``entry_id``;
    ``replace``/``delete`` reference the accepted entry + exact revision
    (compensating correction — history is never edited). ``delete`` takes
    no payload. The request has NO owner/user field by construction.
    """
    if operation not in ("create", "replace", "delete"):
        raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
    if not isinstance(occurred_at, str) or not _TIMESTAMP_RE.match(occurred_at):
        raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
    if operation == "create" and expected_revision != 0:
        raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
    if operation != "create" and expected_revision < 1:
        raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
    if operation == "delete":
        if items is not None:
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
    else:
        if items is None:
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
        _validate_items(items)

    mid = mutation_id if mutation_id is not None else str(uuid.uuid4())
    eid = entry_id if entry_id is not None else str(uuid.uuid4())
    for candidate in (mid, eid):
        if not _UUID_RE.match(candidate):
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)

    request: Dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "mutation_id": mid,
        "entry_id": eid,
        "operation": operation,
        "expected_revision": expected_revision,
        "occurred_at": occurred_at,
        "payload_version": PAYLOAD_VERSION,
    }
    if operation != "delete":
        request["payload"] = {"items": [dict(i) for i in items or []]}
    blob = _canonical_json(request)
    if len(blob) > HEALTH_FOOD_BODY_MAX_BYTES:
        raise HealthClientError(REASON_REQUEST_TOO_LARGE)
    return FrozenEnvelope(
        mutation_id=mid,
        entry_id=eid,
        operation=operation,
        expected_revision=expected_revision,
        request_bytes=blob,
        request_sha256=hashlib.sha256(blob).hexdigest(),
    )


@dataclass(frozen=True)
class VerifiedCommit:
    """A fully verified commit acknowledgement.

    ``receipt_sha256`` is the only value handed onward for durable
    linkage (authorized value-free artifact). ``receipt`` is the decoded
    immutable receipt object for presentation-layer rendering; it
    contains no payload values by contract.
    """

    receipt_sha256: str
    replayed: bool
    receipt: Mapping[str, Any]


class HealthFoodClient:
    """Blocking loopback HTTP client for the Health food endpoint.

    Deliberately stdlib-only and connection-per-request (the server
    forces connection close per the anti-smuggling framing).
    """

    def __init__(self, endpoint_url: str, token: str, *, timeout: float = 10.0) -> None:
        parts = urlsplit(endpoint_url)
        if parts.scheme != "http":
            # Loopback-only plaintext per contract; TLS would imply a
            # non-local trust boundary this route must never have.
            raise HealthClientError(REASON_BAD_ENDPOINT)
        if parts.hostname not in _LOOPBACK_HOSTS:
            raise HealthClientError(REASON_BAD_ENDPOINT)
        if not parts.path or parts.query or parts.fragment:
            raise HealthClientError(REASON_BAD_ENDPOINT)
        if parts.username is not None or parts.password is not None:
            raise HealthClientError(REASON_BAD_ENDPOINT)
        self._host = parts.hostname
        self._port = parts.port if parts.port is not None else 80
        self._path = parts.path
        self._token = validate_commit_token(token)
        self._timeout = float(timeout)

    def commit(self, envelope: FrozenEnvelope) -> VerifiedCommit:
        """Send the frozen envelope once and fully verify the response.

        Raises :class:`HealthClientError`; ``retryable=True`` means the
        caller should later retry the SAME envelope (reconciliation),
        never mint a new mutation.
        """
        if len(envelope.request_bytes) < 1 or len(envelope.request_bytes) > HEALTH_FOOD_BODY_MAX_BYTES:
            raise HealthClientError(REASON_BAD_REQUEST_SHAPE)
        body = self._send(envelope.request_bytes)
        return self._verify(envelope, body)

    # ── transport ───────────────────────────────────────────────────────
    def _send(self, request_bytes: bytes) -> bytes:
        connection = HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            connection.putrequest("POST", self._path, skip_accept_encoding=True)
            # Exactly one Authorization and one Content-Length header.
            connection.putheader("Authorization", "Bearer " + self._token)
            connection.putheader("Content-Length", str(len(request_bytes)))
            connection.putheader("Content-Type", "application/json")
            connection.endheaders()
            connection.send(request_bytes)
            response = connection.getresponse()
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(payload) > _MAX_RESPONSE_BYTES:
                # A 200 (or 5xx) can be post-commit even when its body is
                # oversized. Preserve the exact mutation for reconciliation.
                raise HealthClientError(
                    REASON_BAD_RESPONSE,
                    retryable=response.status == 200 or response.status >= 500,
                )
            if response.status != 200:
                # Includes 5xx: safe to retry the identical envelope.
                raise HealthClientError(
                    REASON_HTTP_STATUS, retryable=response.status >= 500
                )
            return payload
        except HealthClientError:
            raise
        except OSError:
            # Timeout / refused / reset: response loss never authorizes a
            # new mutation; retry the exact frozen envelope.
            raise HealthClientError(REASON_TRANSPORT, retryable=True) from None
        finally:
            connection.close()

    # ── verification ────────────────────────────────────────────────────
    def _verify(self, envelope: FrozenEnvelope, body: bytes) -> VerifiedCommit:
        # Every failure below occurs after an HTTP 200. The server may already
        # have committed the mutation, so the only safe recovery is replaying
        # this exact idempotent envelope. Never authorize a fresh mutation.
        def ambiguous(reason: str) -> HealthClientError:
            return HealthClientError(reason, retryable=True)

        try:
            parsed = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ambiguous(REASON_BAD_RESPONSE) from None
        if not isinstance(parsed, dict) or set(parsed.keys()) != _RESPONSE_KEYS:
            raise ambiguous(REASON_BAD_RESPONSE)
        if parsed["schema"] != RESPONSE_SCHEMA or parsed["receipt_schema"] != RECEIPT_SCHEMA:
            raise ambiguous(REASON_BAD_RESPONSE)
        if type(parsed["replayed"]) is not bool:
            raise ambiguous(REASON_BAD_RESPONSE)
        receipt_sha = parsed["receipt_sha256"]
        if not isinstance(receipt_sha, str) or not _SHA256_RE.match(receipt_sha):
            raise ambiguous(REASON_BAD_RESPONSE)
        try:
            receipt_bytes = base64.b64decode(
                str(parsed["receipt_canonical_json_b64"]), validate=True
            )
        except (ValueError, TypeError):
            raise ambiguous(REASON_BAD_RESPONSE) from None
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha:
            raise ambiguous(REASON_RECEIPT_MISMATCH)
        try:
            receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ambiguous(REASON_BAD_RESPONSE) from None
        if not isinstance(receipt, dict):
            raise ambiguous(REASON_BAD_RESPONSE)
        # Receipt must bind to the exact frozen request identity.
        if (
            receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("mutation_id") != envelope.mutation_id
            or receipt.get("entry_id") != envelope.entry_id
            or receipt.get("operation") != envelope.operation
            or receipt.get("request_sha256") != envelope.request_sha256
            or receipt.get("status") != "applied"
            or receipt.get("same_transaction_readback_match") is not True
        ):
            raise ambiguous(REASON_RECEIPT_MISMATCH)
        # Separate authenticated post-commit readback gates acknowledgement:
        # missing/failed proofs fail closed (retryable — canonical state is
        # unchanged by a readback failure; retry reuses the same envelope).
        readback_sha = parsed["post_commit_readback_sha256"]
        if not isinstance(readback_sha, str) or not _SHA256_RE.match(readback_sha):
            raise HealthClientError(REASON_READBACK_FAILED, retryable=True)
        if parsed["historical_commit_verified"] is not True:
            raise HealthClientError(REASON_READBACK_FAILED, retryable=True)
        if parsed["latest_projection_consistent"] is not True:
            raise HealthClientError(REASON_READBACK_FAILED, retryable=True)
        return VerifiedCommit(
            receipt_sha256=receipt_sha,
            replayed=parsed["replayed"],
            receipt=receipt,
        )
