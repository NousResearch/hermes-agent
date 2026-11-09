"""turn_claim_v1: the reference identity verifier for the connector's per-turn signed claim.

Checks the ``claim`` key of one raw relay inbound frame — a JWS in compact serialization whose
payload names the tenant, agent, acting principal, owner and administration right of the frame's
turn, bound to the frame's event by a hash — exactly as the shared wire contract pins it:

- protected header exactly ``{"alg": "EdDSA", "typ": "hermes-turn-claim+jwt", "kid": …}``;
- Ed25519 only. The verifier pins its trust root: it is configured with the connector's issuer
  (``hosted.turn_claim_issuer``) and fetches keys ONLY from that issuer's JWKS
  (``GET <configured issuer>/.well-known/hermes-turn-claim-jwks.json``, unauthenticated) — the
  claim's own ``iss`` never locates a key; after the payload's checks it must equal the pinned
  issuer. The well-known path is the only hop: a redirect — to another origin, or to the
  issuer's own — is a failed fetch, never followed, so no origin the pin never named can
  introduce a key. The pinned issuer itself must be an https URL, or an http one on loopback
  (a deployment's own test connector); a plaintext issuer across a network is no trust root
  at all. Keys are cached honouring the response's ``max-age`` (default 300 s), a failed refresh
  keeps the last good keys for at most an hour past their expiry (then every key is unknown),
  and fetches are single-flight and at most one per 30 s whatever the outcome — an unknown
  ``kid`` shares that budget, so rotation is picked up without any frame being able to hammer
  the connector. Single-flight means join: a frame arriving while a fetch runs awaits it
  (bounded by the fetch timeout) and answers from its result, so a cold burst of frames is
  admitted by the one fetch behind them, not refused for an empty cache;
- the payload's keys and types exactly the contract's, its value rules (a claim's lifetime is
  at most 120 s, the ``jti`` a lowercase canonical UUIDv4, the tenant, agent, actor and owner
  id non-empty, ``v == 1``), and ``aud == "hermes-relay:" + agt``; the audience must equal the
  one this process serves (``hermes-relay:<this gateway's relay id>``);
- 30 s clock skew on ``exp``/``iat``, the event binding recomputed from the frame's own event,
  and a ``jti`` replay cache that retains an entry exactly as long as the claim is not expired
  (``now <= exp + skew``): live entries are never evicted to make room — at capacity the frame
  is refused as a replay (fail closed).

A frame this verifier cannot check is refused, never served: a JWKS that cannot be fetched or a
key that cannot be found is an unknown key, and no pinned issuer means no key at all. Fail-closed
is the point — the claim is the only thing standing between a frame and a tenant's agent.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import heapq
import ipaddress
import json
import logging
import re
import time
import urllib.parse
import urllib.request
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
    WRONG_ISSUER,
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
_MAX_CLAIM_LIFETIME_S = 120
_JWKS_MAX_AGE_DEFAULT = 300
_JWKS_TIMEOUT_S = 5.0
# The key cache's own bounds: whatever a fetch returns (keys, an error, or a document without
# the wanted kid), the next fetch is at least this far away, and a failed refresh keeps the last
# good keys trusted for at most this long past their expiry.
_JWKS_FETCH_MIN_INTERVAL_S = 30.0
_JWKS_STALE_IF_ERROR_S = 3600.0
# The claim's payload: key -> the exact type its value must be (a verifier refuses unknown or
# missing keys; a bool is not an int here).
_PAYLOAD_KEYS: dict[str, type] = {
    "v": int, "iss": str, "aud": str, "iat": int, "exp": int, "jti": str,
    "ten": str, "agt": str, "act": str, "own": dict, "adm": bool, "evb": str,
}
_OWNER_KINDS = ("principal", "group", "org")
# A canonical lowercase UUIDv4: 8-4-4-4-12 hex, version nibble 4, variant 8-b.
_UUIDV4_JTI = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_MAX_REPLAY_ENTRIES = 10_000  # live claims bound the cache hard; at capacity it refuses

_FetchResult = tuple[bytes, str]  # (jwks document body, Cache-Control header value)


def _b64u_decode(part: str) -> bytes:
    """base64url, padding restored (the compact JWS and the contract's hashes are unpadded)."""
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _issuer_is_fetchable(issuer: Optional[str]) -> bool:
    """Whether the pinned issuer is a URL keys may ever be fetched from: https, or http on
    loopback (a deployment's own test connector). Anything else — plaintext http across a
    network, another scheme, no host at all — is refused: its keys would cross the network
    in the clear (or not be fetchable at all), so it is no trust root."""
    split = urllib.parse.urlsplit(issuer or "")
    if not split.hostname:
        return False
    if split.scheme == "https":
        return True
    if split.scheme != "http":
        return False
    host = split.hostname.lower()
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


class _NoHopRedirect(urllib.request.HTTPRedirectHandler):
    """The well-known path is the only source of keys: a redirect — to another origin, or to
    the issuer's own — is refused, never followed. Returning ``None`` is the handler's
    documented refusal, which surfaces the 3xx as an HTTPError."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# The one opener every JWKS fetch goes through (built on first use; a test seam can pin it).
_jwks_opener: Optional[urllib.request.OpenerDirector] = None


def _http_fetch_jwks(url: str) -> _FetchResult:
    """One unauthenticated JWKS GET (the connector serves it publicly), answered only by the
    URL it was pointed at: any 3xx — followed or bare — is a failed fetch, never a hop.
    Raises whatever the transport raises; the verifier turns a failure into an unknown key,
    never into a served frame."""
    global _jwks_opener
    if _jwks_opener is None:
        _jwks_opener = urllib.request.build_opener(_NoHopRedirect())
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _jwks_opener.open(request, timeout=_JWKS_TIMEOUT_S) as response:
        if 300 <= response.status < 400:
            raise OSError(f"the well-known document at {url} answered {response.status}; a "
                          "redirected JWKS is never fetched")
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
    """The v1 verifier. *issuer* is the pinned trust root (the connector's issuer, whose
    well-known document is the only place a key is ever fetched from) and *audience* the relay
    instance this process serves (``hermes-relay:<relay id>``); ``None`` reads either from the
    deployment's configuration, which a hosted boot refuses the verifier without. *now* (epoch
    seconds) and *fetch_jwks* are seams for tests."""

    def __init__(self, *, audience: Optional[str] = None,
                 issuer: Optional[str] = None,
                 now: Optional[Callable[[], int]] = None,
                 fetch_jwks: Optional[Callable[[str], _FetchResult]] = None) -> None:
        self._audience = audience
        self._issuer = issuer
        self._now = now or (lambda: int(time.time()))
        self._fetch_jwks = fetch_jwks or _http_fetch_jwks
        # The pinned issuer's key cache: the keys by kid from the last good fetch, when they
        # stop being fresh (fetch time + max-age), when a fetch last ran (any outcome — the
        # throttle), and the one in-flight fetch every concurrent frame joins.
        self._keys: dict[str, Ed25519PublicKey] = {}
        self._keys_fresh_until: float = 0.0
        self._last_fetch_at: float = float("-inf")
        self._in_flight_fetch: Optional[asyncio.Task] = None
        # jti -> when it stops being a replay (exp + skew); pruned only past that, never
        # evicted to make room. The heap orders the pruning.
        self._seen_jti: dict[str, float] = {}
        self._jti_expiry: list[tuple[float, str]] = []

    @property
    def name(self) -> str:
        return "turn_claim_v1"

    def is_available(self) -> bool:
        """The relay id this process serves and the connector's pinned issuer must both be
        configured — and the issuer fetchable (https, or http on loopback): without either,
        no audience and no trust root could be checked, so no claim could be admitted."""
        try:
            return self._expected_audience() is not None and self._pinned_issuer() is not None
        except Exception:  # health: allow BLE001 -- the relay id and the issuer come from the config backends, whose read errors are many and varied; none of them make a verifier available, and boot must report unavailability rather than enumerate every backend's failure
            return False

    def _expected_audience(self) -> Optional[str]:
        if self._audience is not None:
            return self._audience
        from gateway.relay import relay_connection_auth
        gateway_id = relay_connection_auth()[0]
        return f"{_AUD_PREFIX}{gateway_id}" if gateway_id else None

    def _pinned_issuer(self) -> Optional[str]:
        """The deployment's trust root (``hosted.turn_claim_issuer``): the one issuer whose
        well-known document is ever fetched. A claim never chooses it. Only a fetchable URL
        is a trust root — https, or http on loopback — so a plaintext issuer across a
        network is as good as none."""
        if self._issuer is not None:
            configured = self._issuer
        else:
            from hermes_cli.hosted_mode import _boot_or_effective_config, _hosted_setting
            configured = _hosted_setting(_boot_or_effective_config(), "turn_claim_issuer")
            if not isinstance(configured, str) or not configured.strip():
                return None
            configured = configured.strip()
        return configured if _issuer_is_fetchable(configured) else None

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
        # The key lookup depends on the header's kid and the pinned issuer alone — never on a
        # payload field — so a signed payload missing its iss is judged by the payload's own
        # checks, not as an unknown key.
        key = await self._signing_key(header, now)
        if key is None:
            return Refusal(UNKNOWN_KID)
        if not self._signature_valid(key, signature, parts):
            return Refusal(BAD_SIGNATURE)
        if self._bad_payload(payload):
            return Refusal(BAD_PAYLOAD)
        if self._wrong_issuer(payload):
            return Refusal(WRONG_ISSUER)
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

    async def _signing_key(self, header: Mapping[str, Any],
                           now: float) -> Optional[Ed25519PublicKey]:
        """The key for the claim's ``kid`` from the pinned issuer's cache. A kid the cached
        document does not name is worth one fetch per fetch window (rotation: the connector
        publishes at most the current key and one previous), then the cache answers."""
        kid = header.get("kid") if isinstance(header, Mapping) else None
        if not isinstance(kid, str) or not kid or self._pinned_issuer() is None:
            return None  # no pinned trust root: every key is unknown (fail closed)
        keys = await self._jwks(now)
        if kid in keys:
            return keys[kid]
        if now - self._last_fetch_at >= _JWKS_FETCH_MIN_INTERVAL_S:
            keys = await self._jwks(now, refresh=True)
        return keys.get(kid)

    async def _jwks(self, now: float, *, refresh: bool = False) -> dict[str, Ed25519PublicKey]:
        """The pinned issuer's keys by kid. Keys fetched at T are trusted until T + max-age; a
        refresh is fetched single-flight, at most one per fetch window whatever the outcome; a
        failed refresh keeps the last good keys for at most an hour past their expiry. One
        fetch serves every concurrent frame: a frame that arrives while a fetch is running
        awaits it and answers from its result, never from the cache the fetch has not filled
        yet."""
        if self._keys and not refresh and now < self._keys_fresh_until:
            return self._keys
        in_flight = self._in_flight_fetch
        if in_flight is not None:
            return await self._await_fetch(in_flight, now)
        if now - self._last_fetch_at < _JWKS_FETCH_MIN_INTERVAL_S:
            return self._trusted_keys(now)
        self._in_flight_fetch = asyncio.ensure_future(self._fetch(now))
        return await self._await_fetch(self._in_flight_fetch, now)

    async def _await_fetch(self, fetch: asyncio.Future,
                           now: float) -> dict[str, Ed25519PublicKey]:
        """Await the single-flight fetch, bounded by the fetch timeout: a waiter is never
        stuck behind a hung fetch — it answers with what the last good fetch left (nothing,
        on a cold cache) while the fetch itself keeps running and still lands in the cache."""
        try:
            return await asyncio.wait_for(asyncio.shield(fetch), timeout=_JWKS_TIMEOUT_S)
        except Exception:  # health: allow BLE001 -- the fetch logs its own failure where it happens; a waiter answers from the last good keys whether the fetch failed or outran the timeout, and no traceback of either would change that answer
            return self._trusted_keys(now)

    async def _fetch(self, now: float) -> dict[str, Ed25519PublicKey]:
        """The one fetch behind the single-flight gate, run by the first frame that needs it
        and awaited by every frame that arrives before it lands."""
        try:
            self._last_fetch_at = now
            url = (self._pinned_issuer() or "").rstrip("/") + _JWKS_PATH
            try:
                body, cache_control = await asyncio.to_thread(self._fetch_jwks, url)
                keys = self._parse_jwks(json.loads(body))
            except Exception:
                logger.warning("turn-claim JWKS fetch failed for %s; the last good keys serve "
                               "while they may, then every key is unknown", url, exc_info=True)
                return self._trusted_keys(now)
            self._keys = keys
            self._keys_fresh_until = now + _parse_max_age(cache_control)
            return keys
        finally:
            self._in_flight_fetch = None

    def _trusted_keys(self, now: float) -> dict[str, Ed25519PublicKey]:
        """What the last good fetch left, while it may still answer: fresh keys, or stale ones
        within the stale-if-error bound past their expiry. Empty past that bound — a key that
        cannot be freshened within the hour is no key at all."""
        if self._keys and now <= self._keys_fresh_until + _JWKS_STALE_IF_ERROR_S:
            return self._keys
        return {}

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
        """The payload's keys, types and value rules exactly as the contract pins them, the
        owner's shape, and the audience internally consistent with the agent it names."""
        if not isinstance(payload, dict) or set(payload) != set(_PAYLOAD_KEYS):
            return True
        if any(type(payload[key]) is not expected for key, expected in _PAYLOAD_KEYS.items()):
            return True
        if payload["v"] != 1:
            return True
        if not (payload["iat"] < payload["exp"] <= payload["iat"] + _MAX_CLAIM_LIFETIME_S):
            return True
        if _UUIDV4_JTI.fullmatch(payload["jti"]) is None:
            return True
        if not payload["ten"] or not payload["agt"] or not payload["act"]:
            return True
        owner = payload["own"]
        if (set(owner) != {"k", "id"} or owner["k"] not in _OWNER_KINDS
                or type(owner["id"]) is not str or not owner["id"]):
            return True
        return payload["aud"] != _AUD_PREFIX + payload["agt"]

    def _wrong_issuer(self, payload: Mapping[str, Any]) -> bool:
        """Whether the claim names an issuer other than the pinned one. The claim never chose
        its trust root; this only refuses a good signature from the wrong issuer."""
        return payload["iss"] != self._pinned_issuer()

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
        """Whether *jti* was seen before; records it otherwise. An entry lives exactly as long
        as the claim is not expired (``now <= exp + skew`` — the same boundary as ``expired``,
        so a claim is either expired or remembered, never neither), is pruned only past that,
        and is never evicted to make room: a verifier that cannot record a jti refuses the
        frame as a replay (fail closed)."""
        seen_until = self._seen_jti.get(jti)
        if seen_until is not None and now <= seen_until:
            return True
        while self._jti_expiry and self._jti_expiry[0][0] < now:
            expiry_of, recorded = heapq.heappop(self._jti_expiry)
            if self._seen_jti.get(recorded) == expiry_of:
                del self._seen_jti[recorded]
        if len(self._seen_jti) >= _MAX_REPLAY_ENTRIES:
            return True  # full of live entries: refuse rather than evict one
        self._seen_jti[jti] = expiry
        heapq.heappush(self._jti_expiry, (expiry, jti))
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
