"""Agent Card signature verification (A2A v1.0 §8.4) — hermes-agent#150793.

The control that matters is ``test_forged_signature_under_legit_kid_is_refused``: it fails against
an unpatched tree (which accepts any card), so it proves the fix rather than passing trivially.
"""

import base64
import json

import pytest

from plugins.platforms.a2a import card_signature as cs

jwt_algorithms = pytest.importorskip("jwt.algorithms")
ec = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ec")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _keypair(alg: str = "ES256"):
    priv = ec.generate_private_key(ec.SECP256R1())
    algo = jwt_algorithms.get_default_algorithms()[alg]
    public_jwk = json.loads(algo.to_jwk(priv.public_key()))
    return priv, public_jwk, algo


def _sign(card: dict, priv, algo, *, kid: str, alg: str = "ES256") -> dict:
    protected_b64 = _b64(json.dumps({"alg": alg, "kid": kid}, separators=(",", ":")).encode("utf-8"))
    signing_input = (protected_b64 + "." + _b64(cs.canonicalize(card))).encode("ascii")
    signature = algo.sign(signing_input, priv)
    return {**card, "signatures": [{"protected": protected_b64, "signature": _b64(signature)}]}


CARD = {
    "name": "Peer Agent",
    "description": "a peer",
    "url": "https://peer.example/rpc",
    "supportedInterfaces": [{"url": "https://peer.example/rpc", "protocolBinding": "JSONRPC"}],
    "skills": [{"id": "general", "name": "general"}],
    "capabilities": {"streaming": True},
}


# --- canonicalisation ----------------------------------------------------

def test_canonicalize_excludes_signatures_and_sorts_keys():
    signed = {**CARD, "signatures": [{"protected": "x", "signature": "y"}]}
    assert cs.canonicalize(signed) == cs.canonicalize(CARD)
    out = cs.canonicalize({"b": 1, "a": 2}).decode()
    assert out == '{"a":2,"b":1}'


def test_canonicalize_rejects_floats():
    with pytest.raises(ValueError):
        cs.canonicalize({"x": 1.5})


# --- verification --------------------------------------------------------

def test_valid_signature_verifies_against_a_trusted_signer():
    priv, jwk, algo = _keypair()
    card = _sign(CARD, priv, algo, kid="signer-1")
    ok, reason = cs.verify_card_signatures(card, {"signer-1": jwk}, require=True)
    assert ok and "verified" in reason


def test_forged_signature_under_legit_kid_is_refused():
    # The attacker serves a card signed with their OWN key, under the legitimate signer's kid.
    legit_priv, legit_jwk, algo = _keypair()
    attacker_priv, _attacker_jwk, _ = _keypair()
    forged = _sign(CARD, attacker_priv, algo, kid="signer-1")      # signed by the attacker
    ok, reason = cs.verify_card_signatures(forged, {"signer-1": legit_jwk}, require=False)
    assert not ok and "forged" in reason
    # and the genuine card from the legit signer still verifies
    genuine = _sign(CARD, legit_priv, algo, kid="signer-1")
    assert cs.verify_card_signatures(genuine, {"signer-1": legit_jwk}, require=False)[0]


def test_tampered_card_after_signing_is_refused():
    priv, jwk, algo = _keypair()
    card = _sign(CARD, priv, algo, kid="signer-1")
    card["url"] = "https://evil.example/rpc"                        # endpoint swapped post-signature
    assert cs.verify_card_signatures(card, {"signer-1": jwk}, require=False)[0] is False


def test_forged_kid_does_not_help_when_key_is_trusted_under_another_kid():
    priv, jwk, algo = _keypair()
    card = _sign(CARD, priv, algo, kid="wrong-kid")                # right key, misleading kid
    assert cs.verify_card_signatures(card, {"real-kid": jwk}, require=True)[0] is True


def test_unsigned_card_refused_only_when_required():
    priv, jwk, _ = _keypair()
    assert cs.verify_card_signatures(CARD, {"k": jwk}, require=False)[0] is True
    ok, reason = cs.verify_card_signatures(CARD, {"k": jwk}, require=True)
    assert not ok and "no signatures" in reason


def test_inert_without_trusted_signers():
    priv, _jwk, algo = _keypair()
    signed = _sign(CARD, priv, algo, kid="signer-1")
    assert cs.verify_card_signatures(signed, {}, require=False)[0] is True      # accepted, warned
    assert cs.verify_card_signatures(CARD, {}, require=False)[0] is True        # unsigned, accepted
    assert cs.verify_card_signatures(signed, {}, require=True)[0] is False      # require with no anchors


def test_symmetric_and_none_algs_are_never_accepted():
    _priv, jwk, _algo = _keypair()
    for alg in ("none", "HS256"):
        protected_b64 = _b64(json.dumps({"alg": alg, "kid": "signer-1"}).encode())
        card = {**CARD, "signatures": [{"protected": protected_b64, "signature": _b64(b"anything")}]}
        assert cs.verify_card_signatures(card, {"signer-1": jwk}, require=False)[0] is False


def test_malformed_signature_entries_are_refused():
    priv, jwk, _ = _keypair()
    for bad in ([{"protected": "!!", "signature": "!!"}], ["notadict"], [{}]):
        card = {**CARD, "signatures": bad}
        assert cs.verify_card_signatures(card, {"signer-1": jwk}, require=False)[0] is False


# --- config loading ------------------------------------------------------

def test_load_trusted_signers_parses_jwks_and_fails_inert():
    _priv, jwk, _ = _keypair()
    jwk["kid"] = "signer-1"
    loaded = cs.load_trusted_signers(json.dumps({"keys": [jwk]}))
    assert loaded == {"signer-1": jwk}
    assert cs.load_trusted_signers("") == {}
    assert cs.load_trusted_signers("not json") == {}
    assert cs.load_trusted_signers(json.dumps({"keys": "nope"})) == {}


def test_load_trusted_signers_from_file(tmp_path):
    _priv, jwk, _ = _keypair()
    jwk["kid"] = "signer-1"
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps({"keys": [jwk]}), encoding="utf-8")
    assert cs.load_trusted_signers(str(path)) == {"signer-1": jwk}


# --- wiring through security.verify_peer_card and the tools --------------

def _trust_env(monkeypatch, jwks: dict | None, require: bool = False):
    from plugins.platforms.a2a import security

    values = {"A2A_TRUSTED_SIGNERS": json.dumps(jwks) if jwks else "",
              "A2A_REQUIRE_CARD_SIGNATURE": "true" if require else ""}
    monkeypatch.setattr(security, "_startup_env", lambda name: values.get(name, ""))


def test_verify_peer_card_reads_config(monkeypatch):
    from plugins.platforms.a2a import security

    priv, jwk, algo = _keypair()
    jwk["kid"] = "signer-1"
    _trust_env(monkeypatch, {"keys": [jwk]})
    assert security.verify_peer_card(_sign(CARD, priv, algo, kid="signer-1"))[0] is True
    attacker, _aj, _ = _keypair()
    assert security.verify_peer_card(_sign(CARD, attacker, algo, kid="signer-1"))[0] is False


def test_a2a_discover_refuses_a_forged_card(monkeypatch):
    from plugins.platforms.a2a import security, tools

    legit_priv, jwk, algo = _keypair()
    jwk["kid"] = "signer-1"
    _trust_env(monkeypatch, {"keys": [jwk]})
    attacker_priv, _aj, _ = _keypair()
    monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: _sign(CARD, attacker_priv, algo, kid="signer-1"))
    out = tools.a2a_discover({"url": "https://peer.example"})
    assert out.startswith("Error: refusing") and "not shown" in out
    # a genuine card is summarised normally
    monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: _sign(CARD, legit_priv, algo, kid="signer-1"))
    assert "Peer Agent" in tools.a2a_discover({"url": "https://peer.example"})
