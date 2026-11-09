"""The reference turn-claim verifier against the shared test vectors.

``tests/conformance/vectors/turn_claim_v1.json`` is copied verbatim from the connector's vector
generator (the two sides mint and verify against the same bytes), so running every case here
proves the reference provider agrees with the connector, case by case, reason slug by reason
slug. Each check in the provider's order is then broken one at a time and a vector must fail:
the suite cannot pass with a check silently removed. The vectors carry TEST-ONLY keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_identity_verifier import Refusal, VerifiedTurn
from hermes_scope import Group, Principal
from plugins.identity_verifier.turn_claim_v1 import TurnClaimV1Verifier, event_binding

VECTORS = json.loads((Path(__file__).parent / "vectors" / "turn_claim_v1.json").read_text())
JWKS_BODY = json.dumps(VECTORS["jwks"]).encode()


@pytest.fixture
def fetch_log():
    """A JWKS fetch seam that records every URL and always serves the vectors' document."""
    urls: list[str] = []

    def fetch(url: str) -> tuple[bytes, str]:
        urls.append(url)
        return JWKS_BODY, "max-age=300"

    return fetch, urls


def _verifier(case, fetch) -> TurnClaimV1Verifier:
    return TurnClaimV1Verifier(audience=case["audience"], now=lambda: case["now"], fetch_jwks=fetch)


def _frame(case) -> dict:
    return {"type": "inbound", "event": case["event"], "claim": case["jws"]}


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
@pytest.mark.asyncio
async def test_every_vector_case_verifies_as_expected(case, fetch_log):
    fetch, _urls = fetch_log
    verdict = await _verifier(case, fetch).verify(_frame(case))
    if case["expect"] == "valid":
        assert isinstance(verdict, VerifiedTurn), verdict
        payload = case["payload"]
        assert verdict.tenant == payload["ten"]
        assert verdict.agent == payload["agt"]
        assert verdict.actor == Principal(payload["act"])
        assert verdict.can_administer is payload["adm"]
        kind, owner_id = payload["own"]["k"], payload["own"]["id"]
        if kind == "principal":
            assert verdict.owner == Principal(owner_id)
        elif kind == "group":
            assert verdict.owner == Group("group:" + owner_id)
        else:
            assert verdict.owner == Group("org:" + owner_id)
    else:
        assert isinstance(verdict, Refusal) and verdict.reason == case["expect"], verdict


@pytest.mark.asyncio
async def test_the_jwks_is_fetched_from_the_claim_issuer_well_known_path(fetch_log):
    fetch, urls = fetch_log
    case = next(c for c in VECTORS["cases"] if c["name"] == "valid")
    await _verifier(case, fetch).verify(_frame(case))
    assert urls == ["https://connector.example/.well-known/hermes-turn-claim-jwks.json"]


@pytest.mark.asyncio
async def test_an_unknown_kid_refetches_at_most_once_per_cache_interval(fetch_log):
    fetch, urls = fetch_log
    case = next(c for c in VECTORS["cases"] if c["name"] == "unknown_kid")
    verifier = _verifier(case, fetch)
    for _ in range(3):
        verdict = await verifier.verify(_frame(case))
        assert isinstance(verdict, Refusal) and verdict.reason == "unknown_kid"
    # One fetch for the cache, one refetch for the unknown kid; the third frame is answered
    # from the cache, inside the interval.
    assert len(urls) == 2


@pytest.mark.asyncio
async def test_a_replayed_jti_is_refused_within_the_claim_lifetime(fetch_log):
    fetch, _ = fetch_log
    case = VECTORS["cases"][0]  # the "valid" case, at its own clock
    verifier = _verifier(case, fetch)
    first = await verifier.verify(_frame(case))
    second = await verifier.verify(_frame(case))
    assert isinstance(first, VerifiedTurn)
    assert isinstance(second, Refusal) and second.reason == "replayed"


def test_the_vectors_event_binding_example_hashes_byte_for_byte():
    example = VECTORS["evb_input_example"]
    assert event_binding(example["event"]) == example["evb"]


# --------------------------------------------------------------------------------------------------
# Each check in the order, broken one at a time: a vector must fail.


def _case(name: str) -> dict:
    return next(c for c in VECTORS["cases"] if c["name"] == name)


# Each mutation breaks exactly one check of the verifier's order: name -> how it is broken.
def _mutate_header(verifier, monkeypatch, case):
    monkeypatch.setattr(verifier, "_bad_header", lambda header: False)


def _mutate_kid(verifier, monkeypatch, case):
    keys = TurnClaimV1Verifier._parse_jwks(VECTORS["jwks"])

    async def any_key(payload, header, now):
        return next(iter(keys.values()))  # the kid is ignored; some key always answers

    monkeypatch.setattr(verifier, "_signing_key", any_key)


def _mutate_signature(verifier, monkeypatch, case):
    monkeypatch.setattr(verifier, "_signature_valid", lambda key, sig, parts: True)


def _mutate_payload(verifier, monkeypatch, case):
    monkeypatch.setattr(verifier, "_bad_payload", lambda payload: False)


def _mutate_aud_consistency(verifier, monkeypatch, case):
    def bad_payload(payload):
        from plugins.identity_verifier.turn_claim_v1 import _PAYLOAD_KEYS
        if not isinstance(payload, dict) or set(payload) != set(_PAYLOAD_KEYS):
            return True
        if any(type(payload[key]) is not expected for key, expected in _PAYLOAD_KEYS.items()):
            return True
        if payload["v"] != 1:
            return True
        owner = payload["own"]
        return set(owner) != {"k", "id"} or owner["k"] not in ("principal", "group", "org") \
            or type(owner["id"]) is not str

    monkeypatch.setattr(verifier, "_bad_payload", bad_payload)  # the aud/agt line removed


def _mutate_audience(verifier, monkeypatch, case):
    monkeypatch.setattr(verifier, "_expected_audience", lambda: "hermes-relay:inst_agent_b")


def _mutate_expired(verifier, monkeypatch, case):
    monkeypatch.setattr(verifier, "_expired", lambda payload, now: False)


def _mutate_not_yet_valid(verifier, monkeypatch, case):
    monkeypatch.setattr(verifier, "_not_yet_valid", lambda payload, now: False)


def _mutate_event_binding(verifier, monkeypatch, case):
    # The binding is no longer recomputed from the frame's own event: it echoes the claim.
    monkeypatch.setattr(verifier, "_frame_event_binding", lambda frame: case["payload"]["evb"])


_MUTATIONS = {
    "header": _mutate_header,
    "kid": _mutate_kid,
    "signature": _mutate_signature,
    "payload": _mutate_payload,
    "aud_consistency": _mutate_aud_consistency,
    "audience": _mutate_audience,
    "expired": _mutate_expired,
    "not_yet_valid": _mutate_not_yet_valid,
    "event_binding": _mutate_event_binding,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation,case_name,expected", [
    ("header", "wrong_alg", "bad_header"),
    ("kid", "unknown_kid", "unknown_kid"),
    ("signature", "tampered_payload", "bad_signature"),
    ("payload", "extra_payload_key", "bad_payload"),
    ("aud_consistency", "agt_not_aud", "bad_payload"),
    ("audience", "wrong_audience", "wrong_audience"),
    ("expired", "expired", "expired"),
    ("not_yet_valid", "not_yet_valid", "not_yet_valid"),
    ("event_binding", "event_mismatch_source_platform", "event_mismatch"),
])
async def test_breaking_one_check_lets_a_vector_through(mutation, case_name, expected, fetch_log, monkeypatch):
    """With the check mutated away, the vector that pins it verifies (or refuses with the wrong
    reason): the suite cannot pass with a check silently removed."""
    fetch, _ = fetch_log
    case = _case(case_name)
    verifier = _verifier(case, fetch)
    _MUTATIONS[mutation](verifier, monkeypatch, case)

    verdict = await verifier.verify(_frame(case))
    assert not (isinstance(verdict, Refusal) and verdict.reason == expected), (
        f"the {mutation} check is load-bearing: with it removed the {case_name} vector still "
        f"refused with {expected}")


@pytest.mark.asyncio
async def test_breaking_the_replay_check_lets_a_second_delivery_through(fetch_log, monkeypatch):
    fetch, _ = fetch_log
    case = VECTORS["cases"][0]
    verifier = _verifier(case, fetch)
    monkeypatch.setattr(verifier, "_replayed", lambda jti, expiry, now: False)
    first = await verifier.verify(_frame(case))
    second = await verifier.verify(_frame(case))
    assert isinstance(first, VerifiedTurn) and isinstance(second, VerifiedTurn), (
        "the replay cache is load-bearing: with it removed a redelivered claim is served twice")
