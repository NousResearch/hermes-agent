"""The hosted relay intake: the identity verifier admits every inbound frame before it runs.

The transport hands each raw frame to the configured verifier once. A refusal drops the frame
(counted and logged by its bounded reason slug, acknowledged like an unknown agent, never
routed to a default tenant or agent) and the verified identity — tenant, agent, actor, owner
and administration right, all from the connector's signed claim, never from the body — is bound
as the scope the frame's whole handling runs under. A local install consults no verifier: a
``claim`` key changes nothing, byte for byte.

The claims here are minted with the shared vectors' TEST-ONLY key, and the verifier is the real
reference provider (``turn_claim_v1``) with its clock, fetcher and audience seamed to the test.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import hosted_mode
from hermes_identity_verifier import Refusal, VerifiedTurn
from hermes_scope import LOCAL_TENANT, Principal, current_scope
from plugins.identity_verifier.turn_claim_v1 import TurnClaimV1Verifier, event_binding

VECTORS = json.loads((Path(__file__).resolve().parents[2] / "conformance" / "vectors"
                     / "turn_claim_v1.json").read_text())
_KID = VECTORS["jwks"]["keys"][0]["kid"]
_JWKS_BODY = json.dumps(VECTORS["jwks"]).encode()
NOW = 1_800_000_010
AUDIENCE = "hermes-relay:inst_agent_a"
EVENT = {"text": "hello there", "message_type": "text",
         "source": {"platform": "telegram", "chat_id": "c-42", "user_id": "u-7", "chat_type": "dm"},
         "message_id": "m-1001", "bot_identity": "bot-1"}

_SK = serialization.load_pem_private_key(VECTORS["test_private_key_pkcs8_pem"].encode(), None)
assert isinstance(_SK, Ed25519PrivateKey)
_OTHER = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"not-the-vectors-key").digest())


def _b64u(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def mint(payload: dict, *, key=_SK, kid=_KID, typ="hermes-turn-claim+jwt", alg="EdDSA") -> str:
    header = {"alg": alg, "typ": typ, "kid": kid}
    h = _b64u(json.dumps(header, separators=(",", ":")).encode())
    p = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    return f"{h}.{p}.{_b64u(key.sign(f'{h}.{p}'.encode()))}"


def claim_payload(**overrides) -> dict:
    payload = {"v": 1, "iss": "https://connector.example", "aud": AUDIENCE, "iat": NOW - 10,
               "exp": NOW + 110, "jti": "6f1c2b8e-1d2a-4c3b-9e8f-0a1b2c3d4e5f", "ten": "tenant_acme",
               "agt": "inst_agent_a", "act": "user_alice",
               "own": {"k": "group", "id": "grp_support"}, "adm": False,
               "evb": event_binding(EVENT)}
    payload.update(overrides)
    return payload


@pytest.fixture
def transport(monkeypatch):
    """A relay transport with a recording inbound handler and a recording ACK sender."""
    from gateway.relay.ws_transport import WebSocketRelayTransport

    t = WebSocketRelayTransport("ws://127.0.0.1:1/relay", "telegram", "app")
    seen = {"events": [], "scopes": [], "acks": []}

    async def inbound(event):
        seen["events"].append(event)
        seen["scopes"].append(current_scope())

    async def ack(buffer_id):
        seen["acks"].append(buffer_id)

    t._inbound = inbound
    monkeypatch.setattr(t, "_send_inbound_ack", ack)
    t._seen = seen
    return t


@pytest.fixture
def fetch():
    return lambda url: (_JWKS_BODY, "max-age=300")


def _verifier(fetch, audience=AUDIENCE):
    return TurnClaimV1Verifier(audience=audience, now=lambda: NOW, fetch_jwks=fetch)


@pytest.fixture
def hosted(monkeypatch, fetch):
    """A hosted process served by the real reference verifier, seamed to the test clock and JWKS."""
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": True})
    from gateway.relay import identity_intake

    identity_intake.reset_intake_state()
    identity_intake._RESOLVED["verifier"] = _verifier(fetch)
    return identity_intake


def _frame(claim=None, event=EVENT, **body) -> dict:
    frame = {"type": "inbound", "event": event, "bufferId": "buf-1"}
    if claim is not None:
        frame["claim"] = claim
    frame.update(body)
    return frame


# ── the gates ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_forged_claim_drops_the_frame(transport, hosted, caplog):
    """A claim signed by another key under the known kid is refused, and the frame is dropped:
    the handler never sees it, the connector's buffer still advances (the same acknowledgment an
    unknown agent gets, so a replay is not wedged), and the refusal is logged and counted by its
    bounded reason slug."""
    forged = mint(claim_payload(), key=_OTHER)
    with caplog.at_level(logging.WARNING):
        await transport._on_inbound(_frame(forged))
    assert transport._seen["events"] == []  # never routed to any tenant or agent
    assert transport._seen["acks"] == ["buf-1"]
    assert "identity verifier (bad_signature)" in caplog.text
    assert hosted.drop_counts() == {"bad_signature": 1}


@pytest.mark.asyncio
async def test_a_frame_without_a_claim_is_refused(transport, hosted, caplog):
    with caplog.at_level(logging.WARNING):
        await transport._on_inbound(_frame())
    assert transport._seen["events"] == []
    assert transport._seen["acks"] == ["buf-1"]
    assert "identity verifier (no_claim)" in caplog.text
    assert hosted.drop_counts() == {"no_claim": 1}


@pytest.mark.asyncio
async def test_the_verified_identity_is_bound_for_the_frame_s_whole_handling(transport, hosted):
    """A valid claim's tenant, agent, actor, owner and administration right are bound as the
    scope the frame runs under: a hosted scope with no home (its state lives behind the storage
    ports), unbound again once the handling returns."""
    await transport._on_inbound(_frame(mint(claim_payload())))
    [scope] = transport._seen["scopes"]
    assert scope is not None
    assert (scope.tenant, scope.agent) == ("tenant_acme", "inst_agent_a")
    assert scope.actor == Principal("user_alice")
    assert scope.owner and scope.owner.id == "group:grp_support"
    assert scope.can_administer is False
    assert scope.home() is None
    assert current_scope() is None  # the bind was the frame's, not the process's


@pytest.mark.asyncio
@pytest.mark.parametrize("admin,expected", [(True, True), (False, False)])
async def test_administration_comes_from_the_claim(transport, hosted, admin, expected):
    """The administration right a turn runs with is the claim's ``adm`` and nothing else: the
    gates that refuse key, setting, plugin and cron changes from a scope without
    ``can_administer`` read exactly this bound value."""
    await transport._on_inbound(_frame(mint(claim_payload(adm=admin))))
    [scope] = transport._seen["scopes"]
    assert scope.can_administer is expected


@pytest.mark.asyncio
async def test_body_fields_asserting_an_identity_are_never_read_for_it(transport, hosted):
    """The frame's body may assert a tenant, principal, owner, administration right or agent;
    only the verified claim counts. A body-asserted identity with no claim behind it is refused;
    with one, the bound scope is the claim's, not the body's."""
    forged_body = _frame(tenant="tenant_evil", principal="user_evil", owner="user_evil",
                        admin=True, agent="inst_victim")
    await transport._on_inbound(forged_body)
    assert transport._seen["events"] == []  # no claim: refused, never admitted on the body
    await transport._on_inbound(_frame(mint(claim_payload()),
                                       tenant="tenant_evil", principal="user_evil",
                                       owner="user_evil", admin=True, agent="inst_victim"))
    [scope] = transport._seen["scopes"]
    assert (scope.tenant, scope.agent, scope.actor, scope.can_administer) == \
        ("tenant_acme", "inst_agent_a", Principal("user_alice"), False)


@pytest.mark.asyncio
async def test_an_unknown_agent_and_a_foreign_agent_are_indistinguishable(transport, hosted, caplog):
    """A claim minted for another instance and one naming an agent that does not exist refuse
    identically: the same reason slug, the same log line, the same acknowledgment, no handler
    call — the response never tells the two apart."""
    foreign = mint(claim_payload(aud="hermes-relay:inst_agent_b", agt="inst_agent_b"))
    unknown = mint(claim_payload(aud="hermes-relay:no-such-agent", agt="no-such-agent"))
    with caplog.at_level(logging.WARNING):
        await transport._on_inbound(_frame(foreign))
        await transport._on_inbound(_frame(unknown))
    assert transport._seen["events"] == []
    assert transport._seen["acks"] == ["buf-1", "buf-1"]
    refusals = [r.message for r in caplog.records if "identity verifier" in r.getMessage()]
    assert refusals[0] == refusals[1]
    assert "wrong_audience" in refusals[0]
    assert hosted.drop_counts() == {"wrong_audience": 2}


@pytest.mark.asyncio
async def test_a_route_miss_drops_the_frame(transport, hosted):
    """A frame whose claim names an agent this instance does not serve never reaches any agent:
    the drop happens at intake, before the frame is decoded or routed."""
    await transport._on_inbound(_frame(mint(claim_payload(agt="inst_other", aud="hermes-relay:inst_other"))))
    assert transport._seen["events"] == []
    assert transport._seen["acks"] == ["buf-1"]
    assert hosted.drop_counts() == {"wrong_audience": 1}


@pytest.mark.asyncio
async def test_a_replayed_claim_is_refused(transport, hosted):
    same = mint(claim_payload())
    await transport._on_inbound(_frame(same))
    await transport._on_inbound(_frame(same))
    assert len(transport._seen["events"]) == 1  # served once; the redelivery is dropped
    assert transport._seen["acks"] == ["buf-1", "buf-1"]
    assert hosted.drop_counts() == {"replayed": 1}


@pytest.mark.asyncio
async def test_a_claim_naming_the_local_tenant_is_refused(transport, hosted):
    """The hosted process serves no local tenant (its agents live behind the storage ports, not
    a home directory): a claim naming one would bind a scope with a home, so it is dropped."""
    await transport._on_inbound(_frame(mint(claim_payload(ten=LOCAL_TENANT))))
    assert transport._seen["events"] == []
    assert hosted.drop_counts() == {"bad_payload": 1}


@pytest.mark.asyncio
async def test_a_verifier_that_crash_drops_the_frame_fail_closed(transport, hosted, caplog):
    """A verifier bug is a drop, never a served frame and never a dead reader: the frame is
    acknowledged (a replay would crash again) and the failure is logged with its traceback."""
    async def crash(frame):
        raise RuntimeError("verifier bug")

    hosted._RESOLVED["verifier"] = type("Broken", (), {
        "verify": staticmethod(crash), "name": "broken", "is_available": lambda self: True})()
    with caplog.at_level(logging.ERROR):
        await transport._on_inbound(_frame(mint(claim_payload())))
    assert transport._seen["events"] == []
    assert transport._seen["acks"] == ["buf-1"]
    assert "identity verifier failed" in caplog.text


# ── local mode: nothing changes ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_local_turn_is_byte_identical_with_a_claim_key(transport, monkeypatch, fetch):
    """On a local install the ``claim`` key is ignored: the same frame with and without a claim
    hands the handler the identical decoded event and sends the identical acknowledgment, and no
    verifier is resolved at all."""
    from hermes_cli import config_effective

    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": False})
    from gateway.relay import identity_intake

    identity_intake.reset_intake_state()

    def no_verifier_ever(name):
        raise AssertionError("a local install resolves no identity verifier")

    monkeypatch.setattr("hermes_identity_verifier.resolve_identity_verifier", no_verifier_ever)
    monkeypatch.setattr(config_effective, "load_user_config_effective",
                        lambda: {"hosted": {"identity_verifier": "should-not-be-read"}})

    bogus_claim = mint(claim_payload(ten="tenant_evil"), key=_OTHER)  # would be refused, hosted
    for claim in (bogus_claim, None):
        await transport._on_inbound(_frame(claim))
    [first, second] = transport._seen["events"]
    # Everything decoded from the frame is identical; ``timestamp`` is this process's own
    # arrival stamp (set the same way in both deliveries), so it is compared apart.
    from dataclasses import fields

    def shape(event):
        return {field.name: getattr(event, field.name) for field in fields(type(event))
                if field.name != "timestamp"}

    assert shape(first) == shape(second)
    assert first.timestamp is not None and second.timestamp is not None
    assert transport._seen["acks"] == ["buf-1", "buf-1"]
    assert identity_intake.drop_counts() == {}
