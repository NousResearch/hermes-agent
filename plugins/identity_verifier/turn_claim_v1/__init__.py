"""turn_claim_v1: the reference identity verifier for the connector's per-turn signed claim.

Checks the ``claim`` key of one raw relay inbound frame — a JWS in compact serialization whose
payload names the tenant, agent, acting principal, owner and administration right of the frame's
turn, bound to the frame's event by a hash — exactly as the shared wire contract pins it:

- protected header exactly ``{"alg": "EdDSA", "typ": "hermes-turn-claim+jwt", "kid": …}``;
- Ed25519 only, keys fetched from the connector's JWKS
  (``GET <iss>/.well-known/hermes-turn-claim-jwks.json``, unauthenticated), cached honouring the
  response's ``max-age`` (default 300 s), with an unknown ``kid`` causing at most one refetch
  per cache interval — the connector publishes at most the current key and one previous;
- the payload's keys and types exactly the contract's, ``aud == "hermes-relay:" + agt``, and the
  audience equal to the one this process serves (``hermes-relay:<this gateway's relay id>``);
- 30 s clock skew on ``exp``/``iat``, the event binding recomputed from the frame's own event,
  and a ``jti`` replay cache bounded to the claim's lifetime (``exp`` + skew).

A frame this verifier cannot check is refused, never served: a JWKS that cannot be fetched or a
key that cannot be found is an unknown key. Fail-closed is the point — the claim is the only
thing standing between a frame and a tenant's agent.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
from typing import Any, Callable, Mapping, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from hermes_identity_verifier import (
    BAD_HEADER,
    BAD_PAYLOAD,
    BAD_SIGNATURE,
    EVENT_MISMATCH,
    EXPIRED,
    NOT_YET_VALID,
    NO_CLAIM,
    REPLAYED,
    UNKNOWN_KID,
    WRONG_AUDIENCE,
    IdentityVerifier,
    Refusal,
    VerifiedTurn,
)
from hermes_scope import Group, Principal

logger = logging.getLogger(__name__)

# The wire contract's fixed values.
_TYP = "hermes-turn-claim+jwt"
_JWKS_PATH = "/.well-known/hermes-turn-claim-jwks.json"
_AUD_PREFIX = "hermes-relay:"
_SKEW_SECONDS = 30
_JWKS_MAX_AGE_DEFAULT = 300
_JWKS_TIMEOUT_S = 5.0
# The claim's payload: key -> the exact type its value must be (a verifier refuses unknown or
# missing keys; a bool is not an int here).
_PAYLOAD_KEYS: dict[str, type] = {
    "v": int, "iss": str, "aud": str, "iat": int, "exp": int, "jti": str,
    "ten": str, "agt": str, "act": str, "own": dict, "adm": bool, "evb": str,
}
_OWNER_KINDS = ("principal", "group", "org")
_MAX_REPLAY_ENTRIES = 10_000  # a jti is a UUIDv4: 10k live claims bound the cache hard

_FetchResult = tuple[bytes, str]  # (jwks document body, Cache-Control header value)


def _b64u_decode(part: str) -> bytes:
    """base64url, padding restored (the compact JWS and the contract's hashes are unpadded)."""
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _http_fetch_jwks(url: str) -> _FetchResult:
    """One unauthenticated JWKS GET (the connector serves it publicly). Raises whatever the
    transport raises; the verifier turns a failure into an unknown key, never into a served
    frame."""
    import urllib.request

    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=_JWKS_TIMEOUT_S) as response:
        return response.read(), str(response.headers.get("Cache-Control") or "")


def _parse_max_age(cache_control: str) -> int:
    match = re.search(r"max-age\s*=\s*(\d+)", cache_control or "", re.IGNORECASE)
    return int(match.group(1)) if match else _JWKS_MAX_AGE_DEFAULT


def event_binding(event: Any) -> str:
    """The event binding: base64url-no-padding SHA-256 over the six fields of the frame's event
    joined with newlines, each the field's string value or the empty string when absent/null.
    No JSON canonicalization anywhere; computed over the raw wire event."""
    event = event if isinstance(event, Mapping) else {}
    source = event.get("source") or {}
    source = source if isinstance(source, Mapping) else {}
    fields = [
        source.get("platform"), event.get("bot_identity"), source.get("chat_id"),
        source.get("user_id"), event.get("message_id"),
        hashlib.sha256(str(event.get("text") or "").encode("utf-8")).hexdigest(),
    ]
    joined = "\n".join("" if field is None else str(field) for field in fields)
    return base64.urlsafe_b64encode(hashlib.sha256(joined.encode("utf-8")).digest()).rstrip(b"=").decode()


class TurnClaimV1Verifier(IdentityVerifier):
    """The v1 verifier. *audience* is the relay instance this process serves
    (``hermes-relay:<relay id>``); ``None`` reads it from the deployment's relay identity, which
    the transport dials with. *now* (epoch seconds) and *fetch_jwks* are seams for tests."""

    def __init__(self, *, audience: Optional[str] = None,
                 now: Optional[Callable[[], int]] = None,
                 fetch_jwks: Optional[Callable[[str], _FetchResult]] = None) -> None:
        self._audience = audience
        self._now = now or (lambda: int(time.time()))
        self._fetch_jwks = fetch_jwks or _http_fetch_jwks
        # Per issuer: the parsed keys by kid, when the cache expires, its max-age, and when an
        # unknown kid last caused a refetch (at most one per interval).
        self._keys: dict[str, dict[str, Ed25519PublicKey]] = {}
        self._expires: dict[str, float] = {}
        self._max_age: dict[str, int] = {}
        self._unknown_kid_refetched: dict[str, float] = {}
        # jti -> when it stops being a replay (exp + skew); FIFO-capped.
        self._seen_jti: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "turn_claim_v1"

    def is_available(self) -> bool:
        """The relay id this process serves must be configured: without it no audience can be
        checked, so no claim could be admitted."""
        try:
            return self._expected_audience() is not None
        except Exception:  # health: allow BLE001 -- the relay id comes from the config backends, whose read errors are many and varied; none of them make a verifier available, and boot must report unavailability rather than enumerate every backend's failure
            return False

    def _expected_audience(self) -> Optional[str]:
        if self._audience is not None:
            return self._audience
        from gateway.relay import relay_connection_auth
        gateway_id = relay_connection_auth()[0]
        return f"{_AUD_PREFIX}{gateway_id}" if gateway_id else None

    # ── the check, in the contract's order; each step is its own method so a test can break one ──

    async def verify(self, frame: Mapping[str, Any]):
        claim = frame.get("claim") if isinstance(frame, Mapping) else None
        if not isinstance(claim, str) or not claim:
            return Refusal(NO_CLAIM)
        now = int(self._now())
        parts = claim.split(".")
        if len(parts) != 3:
            return Refusal(BAD_SIGNATURE)
        try:
            header = json.loads(_b64u_decode(parts[0]))
            payload = json.loads(_b64u_decode(parts[1]))
            signature = _b64u_decode(parts[2])
        except (ValueError, TypeError):
            # Not decodable, not JSON, or not an object: not a JWS this process can check. (A
            # base64 or JSON error is a ValueError; a non-str part a TypeError.)
            return Refusal(BAD_SIGNATURE)
        if self._bad_header(header):
            return Refusal(BAD_HEADER)
        key = await self._signing_key(payload, header, now)
        if key is None:
            return Refusal(UNKNOWN_KID)
        if not self._signature_valid(key, signature, parts):
            return Refusal(BAD_SIGNATURE)
        if self._bad_payload(payload):
            return Refusal(BAD_PAYLOAD)
        if payload["aud"] != self._expected_audience():
            return Refusal(WRONG_AUDIENCE)
        if self._expired(payload, now):
            return Refusal(EXPIRED)
        if self._not_yet_valid(payload, now):
            return Refusal(NOT_YET_VALID)
        if payload["evb"] != self._frame_event_binding(frame):
            return Refusal(EVENT_MISMATCH)
        if self._replayed(payload["jti"], payload["exp"] + _SKEW_SECONDS, now):
            return Refusal(REPLAYED)
        return self._verified_turn(payload)

    def _bad_header(self, header: Any) -> bool:
        """The protected header is exactly the contract's three keys with the contract's two
        fixed values; anything else is refused before any key is consulted."""
        return (not isinstance(header, dict) or set(header) != {"alg", "typ", "kid"}
                or header["alg"] != "EdDSA" or header["typ"] != _TYP)

    async def _signing_key(self, payload: Any, header: Mapping[str, Any],
                           now: float) -> Optional[Ed25519PublicKey]:
        """The JWKS key for the claim's ``kid``: the JWKS is fetched from the claim's ``iss``
        (the connector's public base URL) and cached per issuer honouring ``max-age``. An unknown
        ``kid`` causes at most one refetch per cache interval, so a replayed unknown kid is
        answered from the cache. Any failure to obtain the key is ``None`` (an unknown key), never
        a served frame."""
        iss = payload.get("iss") if isinstance(payload, dict) else None
        kid = header.get("kid") if isinstance(header, Mapping) else None
        if not isinstance(iss, str) or not iss or not isinstance(kid, str):
            return None
        keys = await self._jwks(iss, now)
        if kid in keys:
            return keys[kid]
        # Unknown kid: one refetch per interval, then the cache answers.
        if now - self._unknown_kid_refetched.get(iss, float("-inf")) < self._max_age.get(iss, _JWKS_MAX_AGE_DEFAULT):
            return None
        self._unknown_kid_refetched[iss] = now
        keys = await self._jwks(iss, now, refresh=True)
        return keys.get(kid)

    async def _jwks(self, iss: str, now: float, *, refresh: bool = False) -> dict[str, Ed25519PublicKey]:
        """The cached JWKS keys for *iss*, fetched when absent or past their max-age (or forced)."""
        if not refresh and self._expires.get(iss, 0.0) > now:
            return self._keys.get(iss) or {}
        url = iss.rstrip("/") + _JWKS_PATH
        try:
            body, cache_control = await asyncio.to_thread(self._fetch_jwks, url)
            document = json.loads(body)
            keys = self._parse_jwks(document)
        except Exception:
            logger.warning("turn-claim JWKS fetch failed for %s; treating every key as unknown",
                           iss, exc_info=True)
            return self._keys.get(iss) or {}
        max_age = _parse_max_age(cache_control)
        self._keys[iss] = keys
        self._max_age[iss] = max_age
        self._expires[iss] = now + max_age
        return keys

    @staticmethod
    def _parse_jwks(document: Any) -> dict[str, Ed25519PublicKey]:
        """``{"keys": [{"kty": "OKP", "crv": "Ed25519", "x": …, "kid": …, …}]}`` — every Ed25519
        key by its kid (the connector publishes at most the current key and one previous)."""
        entries = document.get("keys") if isinstance(document, dict) else None
        keys: dict[str, Ed25519PublicKey] = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            try:
                if entry.get("kty") != "OKP" or entry.get("crv") != "Ed25519":
                    continue
                kid = entry["kid"]
                keys[str(kid)] = Ed25519PublicKey.from_public_bytes(_b64u_decode(entry["x"]))
            except Exception:
                logger.warning("skipping an unparseable turn-claim JWKS entry", exc_info=True)
        return keys

    @staticmethod
    def _signature_valid(key: Ed25519PublicKey, signature: bytes, parts: list[str]) -> bool:
        try:
            key.verify(signature, f"{parts[0]}.{parts[1]}".encode("ascii"))
            return True
        except InvalidSignature:
            return False

    def _bad_payload(self, payload: Any) -> bool:
        """The payload's keys and types exactly as the contract pins them, the owner's shape, and
        the audience internally consistent with the agent it names."""
        if not isinstance(payload, dict) or set(payload) != set(_PAYLOAD_KEYS):
            return True
        if any(type(payload[key]) is not expected for key, expected in _PAYLOAD_KEYS.items()):
            return True
        if payload["v"] != 1:
            return True
        owner = payload["own"]
        if set(owner) != {"k", "id"} or owner["k"] not in _OWNER_KINDS or type(owner["id"]) is not str:
            return True
        return payload["aud"] != _AUD_PREFIX + payload["agt"]

    def _frame_event_binding(self, frame: Mapping[str, Any]) -> str:
        return event_binding(frame.get("event"))

    @staticmethod
    def _expired(payload: Mapping[str, Any], now: float) -> bool:
        """Whether the claim's lifetime is past, allowing the contract's clock skew."""
        return now > payload["exp"] + _SKEW_SECONDS

    @staticmethod
    def _not_yet_valid(payload: Mapping[str, Any], now: float) -> bool:
        """Whether the claim was minted too far in the future (the same allowed skew)."""
        return payload["iat"] > now + _SKEW_SECONDS

    def _replayed(self, jti: str, expiry: float, now: float) -> bool:
        """Whether *jti* was seen before; records it otherwise. Entries live to the claim's
        ``exp`` + skew (after that the claim is expired anyway) and the cache is FIFO-capped."""
        seen_at = self._seen_jti.get(jti)
        if seen_at is not None and seen_at > now:
            return True
        self._seen_jti[jti] = expiry
        while len(self._seen_jti) > _MAX_REPLAY_ENTRIES:
            self._seen_jti.pop(next(iter(self._seen_jti)))
        return False

    @staticmethod
    def _verified_turn(payload: Mapping[str, Any]) -> VerifiedTurn:
        owner = payload["own"]
        if owner["k"] == "principal":
            owned_by = Principal(owner["id"])
        else:
            owned_by = Group(("group:" if owner["k"] == "group" else "org:") + owner["id"])
        return VerifiedTurn(tenant=payload["ten"], agent=payload["agt"],
                            actor=Principal(payload["act"]), owner=owned_by,
                            can_administer=payload["adm"])


def register(ctx) -> None:
    ctx.register_identity_verifier(TurnClaimV1Verifier())
