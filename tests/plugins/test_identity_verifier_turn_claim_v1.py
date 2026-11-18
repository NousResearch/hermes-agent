"""The reference verifier's trust root, key cache and replay retention (beyond the vectors).

The verifier pins its trust root to the deployment's configured issuer: keys are fetched only
from that issuer's well-known document — its own path, with no redirect hop followed, from an
https issuer (or an http one on loopback, a deployment's own test connector) — the claim's own
``iss`` never locates anything, and a verifier without a fetchable pinned issuer is unavailable
(a hosted boot refuses it). The key cache fetches single-flight and joins: one fetch serves
every concurrent frame, each awaiting it (bounded by the fetch timeout) and answering from its
result; fetches are at most once per thirty seconds whatever the outcome, and a failed refresh
keeps the last good keys for at most an hour past their expiry, then knows no key at all. The
replay cache never evicts a live entry: at capacity it refuses the frame as a replay (fail
closed), and a redelivered claim is remembered to exactly the ``expired`` boundary.

The claims here are minted with the shared vectors' TEST-ONLY key.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import pytest

from hermes_identity_verifier import Refusal, VerifiedTurn
from plugins.identity_verifier import turn_claim_v1
from plugins.identity_verifier.turn_claim_v1 import TurnClaimV1Verifier, event_binding
from tests.waiting import async_wait_until

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


# ── the well-known path is the only hop ──────────────────────────────────────────────────────

_JWKS_PATH = "/.well-known/hermes-turn-claim-jwks.json"
_ATTACKER_KID = "kid-attacker"
_ATTACKER_JWKS = json.dumps({"keys": [
    {"kty": "OKP", "crv": "Ed25519", "x": _b64u(_OTHER_PUB), "kid": _ATTACKER_KID,
     "use": "sig", "alg": "EdDSA"}]}).encode()


class _JWKSOrigin(BaseHTTPRequestHandler):
    """One origin answering the well-known path (and any other) with whatever the test
    pinned on it: the document itself, or a redirect whose hop would move the fetch. Every
    request the origin saw is recorded, so a test can prove a hop was never followed."""

    def do_GET(self) -> None:
        server = self.server
        assert isinstance(server, _RecordingOrigin)
        server.requests.append(self.path)
        status, location, body = server.routes.get(self.path, (200, None, b"{}"))
        self.send_response(status)
        if location is not None:
            self.send_header("Location", location)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        pass  # a probe origin stays out of the runner's output


class _RecordingOrigin(ThreadingHTTPServer):
    """A loopback HTTP origin whose answers the test pins per path, recording every request."""

    def __init__(self, routes: dict[str, tuple[int, Optional[str], bytes]]) -> None:
        super().__init__(("127.0.0.1", 0), _JWKSOrigin)
        self.requests: list[str] = []
        self.routes = dict(routes)


@contextmanager
def _jwks_origin(routes: dict[str, tuple[int, Optional[str], bytes]]):
    """A real local HTTP origin on loopback (port 0), pinned to *routes* for its answers."""
    server = _RecordingOrigin(routes)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _origin_url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.mark.asyncio
async def test_a_jwks_path_that_redirects_is_a_failed_fetch_not_a_hop(clock):
    """Keys come only from the configured issuer's own well-known path: a redirect away from
    it — to an origin whose keys would fit the claim — is a failed fetch, so the frame is
    refused as an unknown key and the second origin is never read at all."""
    with _jwks_origin({_JWKS_PATH: (200, None, _ATTACKER_JWKS)}) as attacker:
        with _jwks_origin({_JWKS_PATH: (302, None, b"{}")}) as pinned:
            pinned.routes[_JWKS_PATH] = (302, _origin_url(attacker) + _JWKS_PATH, b"{}")
            origin = _origin_url(pinned)
            # signed by the second origin's key, naming the pinned issuer: only the hop
            # could ever let it in
            claim = mint(payload(iss=origin), key=_OTHER, kid=_ATTACKER_KID)
            verdict = await TurnClaimV1Verifier(
                audience=_AUDIENCE, issuer=origin, now=clock).verify(frame(claim))
            assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
            assert pinned.requests == [_JWKS_PATH]
            assert attacker.requests == [], "no key from the second origin was ever fetched"


@pytest.mark.asyncio
async def test_a_redirect_within_the_issuer_s_own_origin_is_no_hop_either(clock):
    """The pin is the path, not just the host: a redirect that stays inside the configured
    issuer's own origin is refused all the same — the document must be served where the
    contract says, not one hop away from it."""
    with _jwks_origin({}) as pinned:
        origin = _origin_url(pinned)
        pinned.routes[_JWKS_PATH] = (307, origin + "/moved-jwks.json", b"{}")
        pinned.routes["/moved-jwks.json"] = (200, None, _JWKS_BODY)
        verdict = await TurnClaimV1Verifier(
            audience=_AUDIENCE, issuer=origin, now=clock).verify(frame(mint(payload(iss=origin))))
        assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
        assert pinned.requests == [_JWKS_PATH]


@pytest.mark.asyncio
async def test_a_loopback_http_issuer_serving_its_keys_directly_admits_the_claim(clock):
    """An http issuer on loopback — a deployment's own test connector — is a trust root when
    it serves its keys itself: what the tests above refuse is the hop, never the scheme."""
    with _jwks_origin({_JWKS_PATH: (200, None, _JWKS_BODY)}) as pinned:
        origin = _origin_url(pinned)
        verdict = await TurnClaimV1Verifier(
            audience=_AUDIENCE, issuer=origin, now=clock).verify(frame(mint(payload(iss=origin))))
        assert isinstance(verdict, VerifiedTurn)


@pytest.mark.asyncio
async def test_a_plaintext_issuer_off_loopback_is_no_trust_root_at_all(monkeypatch, fetch_log, clock):
    """A pinned issuer read over plaintext http off loopback would publish its keys to the
    network path: the verifier is unavailable (a hosted boot refuses it), nothing is ever
    fetched from it, and every claim is an unknown key. Loopback http and https are."""
    fetch = fetch_log

    def config(issuer):
        monkeypatch.setattr("hermes_cli.hosted_mode._boot_or_effective_config",
                            lambda: {"hosted": {"turn_claim_issuer": issuer}})

    monkeypatch.setenv("GATEWAY_RELAY_ID", "inst_agent_a")
    config("http://connector.internal")
    verifier = TurnClaimV1Verifier(now=clock, fetch_jwks=fetch)
    assert verifier.is_available() is False
    verdict = await verifier.verify(frame(mint(payload(iss="http://connector.internal"))))
    assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
    assert fetch.urls == []  # nothing is ever fetched over plaintext http off loopback
    for issuer in ("http://127.0.0.1:8443", "http://localhost:8443", "https://connector.example"):
        config(issuer)
        assert TurnClaimV1Verifier(now=clock, fetch_jwks=fetch).is_available() is True, issuer


# ── the single-flight fetch is joined by every concurrent frame ──────────────────────────────


@pytest.mark.asyncio
async def test_a_cold_burst_awaits_the_one_fetch_and_verifies_against_its_result(clock):
    """Single-flight means join, not just dedupe: with a fetch slow enough that a whole cold
    burst of frames arrives behind it, every frame awaits the same fetch and is admitted —
    one fetch, twelve frames, none answered from an empty cache."""
    fetches: list[str] = []

    def slow_fetch(url: str) -> tuple[bytes, str]:
        fetches.append(url)
        time.sleep(0.25)  # health: allow HX013 -- the slow fetch IS the behaviour: it must still be in flight when the whole burst is queued behind it
        return _JWKS_BODY, "max-age=300"

    verifier = TurnClaimV1Verifier(audience=_AUDIENCE, issuer=_ISSUER, now=clock,
                                   fetch_jwks=slow_fetch)
    verdicts = await asyncio.gather(*(verifier.verify(frame(mint(payload()))) for _ in range(12)))
    assert all(isinstance(v, VerifiedTurn) for v in verdicts), (
        "each frame in the burst must await the fetch, not read the still-empty cache")
    assert len(fetches) == 1


@pytest.mark.asyncio
async def test_a_waiter_is_bounded_by_the_fetch_timeout_and_the_fetch_still_lands(monkeypatch, clock):
    """A fetch that outruns the fetch timeout does not hang its waiters: they answer from
    what the last good fetch left (nothing, on a cold cache) and the frame is refused as an
    unknown key — and the fetch itself still completes and lands in the cache, so the next
    frame is admitted without a second fetch."""
    monkeypatch.setattr(turn_claim_v1, "_JWKS_TIMEOUT_S", 0.05)
    fetches: list[str] = []

    def hung_fetch(url: str) -> tuple[bytes, str]:
        fetches.append(url)
        time.sleep(0.2)  # health: allow HX013 -- outrunning the join timeout IS the behaviour under test
        return _JWKS_BODY, "max-age=300"

    verifier = TurnClaimV1Verifier(audience=_AUDIENCE, issuer=_ISSUER, now=clock,
                                   fetch_jwks=hung_fetch)
    verdict = await verifier.verify(frame(mint(payload())))
    assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
    assert len(fetches) == 1
    await async_wait_until(lambda: verifier._keys, what="the fetch the waiter stopped "
                                                        "waiting for still landed in the cache")
    assert isinstance(await verifier.verify(frame(mint(payload()))), VerifiedTurn)
    assert len(fetches) == 1  # the landed fetch answers; the window never re-fetched


# ── what each of those protections prevents, with the protection removed ──────────────────────


@pytest.mark.asyncio
async def test_following_the_hop_again_lets_another_origin_s_keys_in(clock, monkeypatch):
    """The no-hop fetch is load-bearing: with a redirect-following HTTP client back in its
    place, the pinned issuer's redirect hands the fetch to a second origin and its keys
    admit a claim the configured issuer never signed — exactly what pinning the well-known
    path exists to prevent."""
    monkeypatch.setattr(turn_claim_v1, "_jwks_opener", urllib.request.build_opener())
    with _jwks_origin({_JWKS_PATH: (200, None, _ATTACKER_JWKS)}) as attacker:
        with _jwks_origin({_JWKS_PATH: (302, None, b"{}")}) as pinned:
            pinned.routes[_JWKS_PATH] = (302, _origin_url(attacker) + _JWKS_PATH, b"{}")
            origin = _origin_url(pinned)
            claim = mint(payload(iss=origin), key=_OTHER, kid=_ATTACKER_KID)
            verdict = await TurnClaimV1Verifier(
                audience=_AUDIENCE, issuer=origin, now=clock).verify(frame(claim))
    assert isinstance(verdict, VerifiedTurn), (
        "the pinned well-known path is load-bearing: with redirects followed again, a claim "
        "signed by a second origin the pin never named was admitted")


@pytest.mark.asyncio
async def test_accepting_a_plaintext_issuer_again_publishes_keys_off_loopback(monkeypatch, fetch_log, clock):
    """The https-or-loopback rule is load-bearing: with every issuer accepted again, a
    plaintext off-loopback trust root becomes available at boot and its keys are fetched
    over plain http — exactly what the rule exists to prevent."""
    fetch = fetch_log
    monkeypatch.setattr(turn_claim_v1, "_issuer_is_fetchable", lambda issuer: bool(issuer))
    monkeypatch.setattr("hermes_cli.hosted_mode._boot_or_effective_config",
                        lambda: {"hosted": {"turn_claim_issuer": "http://connector.internal"}})
    monkeypatch.setenv("GATEWAY_RELAY_ID", "inst_agent_a")
    verifier = TurnClaimV1Verifier(now=clock, fetch_jwks=fetch)
    assert verifier.is_available() is True
    verdict = await verifier.verify(frame(mint(payload(iss="http://connector.internal"))))
    assert isinstance(verdict, VerifiedTurn), (
        "the scheme rule is load-bearing: with it removed, keys were read over plaintext "
        "http off loopback")
    assert fetch.urls == ["http://connector.internal/.well-known/hermes-turn-claim-jwks.json"]


@pytest.mark.asyncio
async def test_answering_from_the_cache_without_joining_the_fetch_refuses_the_cold_burst(
        monkeypatch, clock):
    """The join is load-bearing: with every waiter answered from the cache instead of
    awaiting the in-flight fetch, a cold burst behind one slow fetch is refused frame by
    frame for an empty cache — exactly what joining exists to prevent."""
    fetches: list[str] = []

    def slow_fetch(url: str) -> tuple[bytes, str]:
        fetches.append(url)
        time.sleep(0.25)  # health: allow HX013 -- the slow fetch IS the behaviour: the unjoined burst must pile up behind it
        return _JWKS_BODY, "max-age=300"

    async def unjoined(self, fetch, now):
        return self._trusted_keys(now)  # the cache as it stands, never the fetch's result

    monkeypatch.setattr(TurnClaimV1Verifier, "_await_fetch", unjoined)
    verifier = TurnClaimV1Verifier(audience=_AUDIENCE, issuer=_ISSUER, now=clock,
                                   fetch_jwks=slow_fetch)
    verdicts = await asyncio.gather(*(verifier.verify(frame(mint(payload()))) for _ in range(12)))
    await async_wait_until(lambda: verifier._keys, what="the one fetch the burst never "
                                                       "awaited still ran to its end")
    assert all(isinstance(v, Refusal) and v.reason == "unknown_kid" for v in verdicts), (
        "the join is load-bearing: without it, all twelve valid claims were refused behind "
        "one fetch that was never awaited")
    assert len(fetches) == 1
