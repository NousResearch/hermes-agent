"""The reference verifier's trust root, key cache and replay retention (beyond the vectors).

The verifier pins its trust root to the deployment's configured issuer: keys are fetched only
from that issuer's well-known document, the claim's own ``iss`` never locates anything, and a
verifier without a pinned issuer is unavailable (a hosted boot refuses it). The key cache
fetches single-flight, at most once per thirty seconds whatever the outcome, keeps the last
good keys for at most an hour past their expiry after a failed refresh, and then knows no key
at all. The replay cache never evicts a live entry: at capacity it refuses the frame as a
replay (fail closed), and a redelivered claim is remembered to exactly the ``expired`` boundary.

The claims here are minted with the shared vectors' TEST-ONLY key.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from hermes_identity_verifier import Refusal, VerifiedTurn
from plugins.identity_verifier.turn_claim_v1 import TurnClaimV1Verifier, event_binding

VECTORS = json.loads((Path(__file__).resolve().parents[1] / "conformance" / "vectors"
                     / "turn_claim_v1.json").read_text())
_KID = VECTORS["jwks"]["keys"][0]["kid"]
_JWKS_BODY = json.dumps(VECTORS["jwks"]).encode()
_ISSUER = "https://connector.example"
_AUDIENCE = "hermes-relay:inst_agent_a"
_NOW = 1_800_000_010
_EVENT = {"text": "hello", "message_type": "text",
          "source": {"platform": "telegram", "chat_id": "c-1", "user_id": "u-1", "chat_type": "dm"},
          "message_id": "m-1", "bot_identity": "bot-1"}

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_SK = serialization.load_pem_private_key(VECTORS["test_private_key_pkcs8_pem"].encode(), None)
assert isinstance(_SK, Ed25519PrivateKey)


def _b64u(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def mint(payload: dict, *, key=_SK, kid=_KID) -> str:
    header = {"alg": "EdDSA", "typ": "hermes-turn-claim+jwt", "kid": kid}
    h = _b64u(json.dumps(header, separators=(",", ":")).encode())
    p = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    return f"{h}.{p}.{_b64u(key.sign(f'{h}.{p}'.encode()))}"


def payload(**overrides) -> dict:
    claim = {"v": 1, "iss": _ISSUER, "aud": _AUDIENCE, "iat": _NOW - 10, "exp": _NOW + 110,
             "jti": str(uuid.uuid4()), "ten": "tenant_acme", "agt": "inst_agent_a",
             "act": "user_alice", "own": {"k": "principal", "id": "user_alice"}, "adm": False,
             "evb": event_binding(_EVENT)}
    claim.update(overrides)
    return claim


def frame(claim: str) -> dict:
    return {"type": "inbound", "event": _EVENT, "claim": claim}


class Clock:
    """A mutable clock the verifier reads."""

    def __init__(self, now: int = _NOW) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def fetch_log(clock):
    """A fetch seam that records every URL and serves the vectors' document, until a test
    flips it to failing (an outage) or to another document (a rotation)."""
    urls: list[str] = []
    state = {"mode": "ok"}

    def fetch(url: str) -> tuple[bytes, str]:
        urls.append(url)
        if state["mode"] == "fail":
            raise OSError("the connector's JWKS endpoint is unreachable")
        if state["mode"] == "rotated":
            rotated = json.loads(_JWKS_BODY)
            rotated["keys"].append({"kty": "OKP", "crv": "Ed25519",
                                     "x": _b64u(_OTHER_PUB), "kid": "kid-rotated",
                                     "use": "sig", "alg": "EdDSA"})
            return json.dumps(rotated).encode(), "max-age=300"
        return _JWKS_BODY, state.get("cache_control", "max-age=300")

    fetch.urls = urls
    fetch.state = state
    return fetch


_OTHER = Ed25519PrivateKey.from_private_bytes(b"\x02" * 32)
_OTHER_PUB = _OTHER.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _verifier(fetch, clock, *, audience=_AUDIENCE, issuer=_ISSUER) -> TurnClaimV1Verifier:
    return TurnClaimV1Verifier(audience=audience, issuer=issuer, now=clock, fetch_jwks=fetch)


# ── the pinned trust root ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_keys_are_never_fetched_from_the_claim_s_own_issuer(fetch_log, clock):
    """A claim naming another issuer is judged against the pinned issuer's keys alone: the
    fetcher is never pointed anywhere else, and the claim refuses as a wrong issuer."""
    fetch = fetch_log
    forged_iss = mint(payload(iss="https://attacker.example"))
    verdict = await _verifier(fetch, clock).verify(frame(forged_iss))
    assert isinstance(verdict, Refusal) and verdict.reason == "wrong_issuer"
    assert fetch.urls == [f"{_ISSUER}/.well-known/hermes-turn-claim-jwks.json"]


@pytest.mark.asyncio
async def test_a_verifier_without_a_pinned_issuer_admits_no_frame(monkeypatch, fetch_log, clock):
    """Without a configured trust root every key is unknown: the verifier fails closed rather
    than fetching from somewhere a claim could choose."""
    fetch = fetch_log
    monkeypatch.setattr("hermes_cli.hosted_mode._boot_or_effective_config",
                        lambda: {"hosted": {}})
    verifier = TurnClaimV1Verifier(audience=_AUDIENCE, now=clock, fetch_jwks=fetch)
    verdict = await verifier.verify(frame(mint(payload())))
    assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
    assert fetch.urls == []


def test_availability_needs_the_pinned_issuer(monkeypatch, fetch_log, clock):
    """``is_available`` answers whether the verifier can check frames here: the deployment's
    relay id (the audience) and its pinned issuer must both be configured."""
    fetch = fetch_log

    def config(issuer):
        monkeypatch.setattr("hermes_cli.hosted_mode._boot_or_effective_config",
                            lambda: {"hosted": {"turn_claim_issuer": issuer}})

    monkeypatch.setenv("GATEWAY_RELAY_ID", "inst_agent_a")
    config("")
    assert TurnClaimV1Verifier(now=clock, fetch_jwks=fetch).is_available() is False
    config(_ISSUER)
    assert TurnClaimV1Verifier(now=clock, fetch_jwks=fetch).is_available() is True
    monkeypatch.setenv("GATEWAY_RELAY_ID", "")
    assert TurnClaimV1Verifier(now=clock, fetch_jwks=fetch).is_available() is False


# ── the key cache ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_fetch_is_negative_cached_within_the_thirty_second_window(fetch_log, clock):
    """An unreachable JWKS endpoint is fetched once; the frames behind it, still inside the
    throttle window, are answered without another fetch — an outage cannot be amplified into
    one network round trip per inbound frame."""
    fetch = fetch_log
    fetch.state["mode"] = "fail"
    verifier = _verifier(fetch, clock)
    for _ in range(4):
        verdict = await verifier.verify(frame(mint(payload())))
        assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
    assert len(fetch.urls) == 1
    clock.now += 30
    await verifier.verify(frame(mint(payload())))  # past the window, one retry
    assert len(fetch.urls) == 2


@pytest.mark.asyncio
async def test_concurrent_cold_frames_share_one_single_flight_fetch(fetch_log, clock):
    fetch = fetch_log
    verifier = _verifier(fetch, clock)
    verdicts = await asyncio.gather(*(verifier.verify(frame(mint(payload()))) for _ in range(12)))
    assert all(isinstance(v, VerifiedTurn) for v in verdicts)
    assert len(fetch.urls) == 1


@pytest.mark.asyncio
async def test_concurrent_unknown_kid_frames_share_one_single_flight_fetch(fetch_log, clock):
    fetch = fetch_log
    verifier = _verifier(fetch, clock)
    unknown = await asyncio.gather(*(
        verifier.verify(frame(mint(payload(), kid="no-such-kid"))) for _ in range(12)))
    assert all(isinstance(v, Refusal) and v.reason == "unknown_kid" for v in unknown)
    assert len(fetch.urls) == 1


@pytest.mark.asyncio
async def test_a_new_kid_after_a_rotation_is_fetched_once_per_window(fetch_log, clock):
    """A key the cache has never seen is worth one fetch per thirty-second window: rotation
    is picked up without an unknown kid being able to hammer the connector."""
    fetch = fetch_log
    rotated_kid = "kid-rotated"
    verdict = await _verifier(fetch, clock).verify(frame(mint(payload(), kid=_KID)))
    assert isinstance(verdict, VerifiedTurn)
    fetch.state["mode"] = "rotated"  # the connector rotated; the claim's kid is new to this cache
    clock.now += 40
    verdict = await _verifier(fetch, clock).verify(frame(mint(payload(), key=_OTHER, kid=rotated_kid)))
    assert isinstance(verdict, VerifiedTurn)
    assert len(fetch.urls) == 2


@pytest.mark.asyncio
async def test_a_failed_refresh_keeps_the_last_good_keys_within_the_stale_bound(fetch_log, clock):
    """Keys fetched at T are trusted until T + max-age; when the refresh fails, the last good
    keys keep answering for at most an hour past that expiry — and no longer. Each frame carries
    a claim minted for the moment it is verified, so only the KEY cache ages here."""
    fetch = fetch_log
    verifier = _verifier(fetch, clock)
    fetch.state["cache_control"] = "max-age=1"
    assert isinstance(await verifier.verify(frame(mint(payload()))), VerifiedTurn)
    keys_expire_at = clock.now + 1
    fetch.state["mode"] = "fail"
    clock.now = keys_expire_at + 100  # past expiry, refresh failing: the stale keys still answer
    assert isinstance(await verifier.verify(frame(mint(payload(iat=clock.now - 10,
                                                                exp=clock.now + 110)))),
                      VerifiedTurn)
    clock.now = keys_expire_at + 3600  # the stale bound itself: still trusted
    assert isinstance(await verifier.verify(frame(mint(payload(iat=clock.now - 10,
                                                                exp=clock.now + 110)))),
                      VerifiedTurn)
    clock.now = keys_expire_at + 3601  # past it: every key is unknown until a fetch succeeds
    verdict = await verifier.verify(frame(mint(payload(iat=clock.now - 10,
                                                        exp=clock.now + 110))))
    assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"


# ── the replay cache ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_live_replay_entry_is_never_evicted_and_capacity_fails_closed(fetch_log, clock):
    """Ten thousand claims fill the cache inside one claim lifetime; the ten-thousand-and-first
    is refused as a replay rather than evicting a live entry, and the very first claim is still
    remembered when it is redelivered."""
    fetch = fetch_log
    verifier = _verifier(fetch, clock)
    for index in range(10_000):
        verdict = await verifier.verify(frame(mint(payload(jti=_jti(index)))))
        assert isinstance(verdict, VerifiedTurn), f"claim {index} should verify"
    overflow = await verifier.verify(frame(mint(payload(jti=_jti(10_000)))))
    assert isinstance(overflow, Refusal) and overflow.reason == "replayed"
    redelivered = await verifier.verify(frame(mint(payload(jti=_jti(0)))))
    assert isinstance(redelivered, Refusal) and redelivered.reason == "replayed"


def _jti(index: int) -> str:
    """A distinct canonical UUIDv4 per index (deterministic, so a redelivered claim is the same
    bytes): the version and variant nibbles are forced, the rest come from the index's hash."""
    import hashlib

    h = hashlib.sha256(f"replay-capacity-{index}".encode()).hexdigest()
    variant = "89ab"[int(h[15], 16) % 4]
    return f"{h[0:8]}-{h[8:12]}-4{h[12:15]}-{variant}{h[16:19]}-{h[20:32]}"


@pytest.mark.asyncio
async def test_breaking_retention_lets_an_evicted_claim_back_in(fetch_log, clock, monkeypatch):
    """The retention rules are load-bearing: with a FIFO eviction cache back in their place, the
    ten-thousand-and-first distinct claim evicts the first, and the first claim's redelivery
    is served a second time — exactly what retention exists to prevent."""
    fetch = fetch_log

    def fifo_replayed(self, jti, expiry, now):
        seen = self._seen_jti.get(jti)
        if seen is not None and seen > now:
            return True
        self._seen_jti[jti] = expiry
        from plugins.identity_verifier.turn_claim_v1 import _MAX_REPLAY_ENTRIES
        while len(self._seen_jti) > _MAX_REPLAY_ENTRIES:
            self._seen_jti.pop(next(iter(self._seen_jti)))
        return False

    monkeypatch.setattr(TurnClaimV1Verifier, "_replayed", fifo_replayed)
    verifier = _verifier(fetch, clock)
    for index in range(10_000):
        await verifier.verify(frame(mint(payload(jti=_jti(index)))))
    overflow = await verifier.verify(frame(mint(payload(jti=_jti(10_000)))))
    redelivered = await verifier.verify(frame(mint(payload(jti=_jti(0)))))
    assert isinstance(overflow, VerifiedTurn), (
        "fail-closed at capacity is load-bearing: with FIFO eviction back, the overflow claim "
        "was served after evicting a live entry")
    assert isinstance(redelivered, VerifiedTurn), (
        "retention is load-bearing: with FIFO eviction back, the original claim returned after "
        "the cache turned over")
