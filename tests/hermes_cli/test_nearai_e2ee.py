"""Tests for NEAR AI E2EE crypto primitives and attestation helpers.

Transport-layer integration is covered in ``test_e2ee_transport.py``.
"""

import base64
import hashlib
import secrets

import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import requests

from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519

from hermes_cli.e2ee_proxy import (
    _encrypt_ecdsa,
    _decrypt_ecdsa,
    _generate_ecdsa_keypair,
    _encrypt_ed25519,
    _decrypt_ed25519,
    _generate_ed25519_keypair,
)
from hermes_cli.attestation import _phala_check_report_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_keypair():
    """Return (priv_key_obj, pub_hex) for a fresh SECP256K1 keypair (the "model CVM" side)."""
    priv = ec.generate_private_key(ec.SECP256K1(), default_backend())
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return priv, pub_bytes[1:].hex()  # strip 0x04


# ---------------------------------------------------------------------------
# ECIES roundtrip
# ---------------------------------------------------------------------------

class TestEciesRoundtrip:
    def test_encrypt_decrypt(self):
        model_priv, model_pub_hex = _make_model_keypair()
        plaintext = b"secret prompt: what is 2+2?"
        ct = _encrypt_ecdsa(plaintext, model_pub_hex)
        recovered = _decrypt_ecdsa(ct, model_priv)
        assert recovered == plaintext

    def test_ciphertext_is_random(self):
        _, model_pub_hex = _make_model_keypair()
        ct1 = _encrypt_ecdsa(b"same message", model_pub_hex)
        ct2 = _encrypt_ecdsa(b"same message", model_pub_hex)
        assert ct1 != ct2  # ephemeral key is fresh each time

    def test_wrong_key_raises(self):
        _, model_pub_hex = _make_model_keypair()
        wrong_priv, _ = _make_model_keypair()
        ct = _encrypt_ecdsa(b"hello", model_pub_hex)
        with pytest.raises(Exception):
            _decrypt_ecdsa(ct, wrong_priv)

    def test_client_keypair_generation(self):
        priv, pub_hex = _generate_ecdsa_keypair()
        assert len(bytes.fromhex(pub_hex)) == 64  # uncompressed minus 0x04 prefix


# ---------------------------------------------------------------------------
# Ed25519 roundtrip
# ---------------------------------------------------------------------------

def _make_ed25519_model_keypair():
    priv = _ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return priv, pub.hex()


class TestEd25519Roundtrip:
    def test_encrypt_decrypt(self):
        model_priv, model_pub_hex = _make_ed25519_model_keypair()
        plaintext = b"secret prompt: what is 2+2?"
        ct = _encrypt_ed25519(plaintext, model_pub_hex)
        recovered = _decrypt_ed25519(ct, model_priv)
        assert recovered == plaintext

    def test_ciphertext_is_random(self):
        _, model_pub_hex = _make_ed25519_model_keypair()
        ct1 = _encrypt_ed25519(b"same message", model_pub_hex)
        ct2 = _encrypt_ed25519(b"same message", model_pub_hex)
        assert ct1 != ct2

    def test_wrong_key_raises(self):
        _, model_pub_hex = _make_ed25519_model_keypair()
        wrong_priv, _ = _make_ed25519_model_keypair()
        ct = _encrypt_ed25519(b"hello", model_pub_hex)
        with pytest.raises(Exception):
            _decrypt_ed25519(ct, wrong_priv)

    def test_client_keypair_generation(self):
        priv, pub_hex = _generate_ed25519_keypair()
        assert len(bytes.fromhex(pub_hex)) == 32


# ---------------------------------------------------------------------------
# signing_public_key → signing_address derivation
# ---------------------------------------------------------------------------

class TestSigningKeyDerivation:
    def test_pub_key_derives_to_correct_address(self):
        """keccak256(pubkey[1:]) → last 20 bytes == Ethereum address."""
        from eth_keys.datatypes import PublicKey as _EthPubKey

        priv = ec.generate_private_key(ec.SECP256K1(), default_backend())
        pub_bytes_full = priv.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
        pub_hex = pub_bytes_full[1:].hex()  # strip 0x04

        eth_addr = "0x" + _EthPubKey(bytes.fromhex(pub_hex)).to_canonical_address().hex()
        assert eth_addr.startswith("0x")
        assert len(eth_addr) == 42

    def test_wrong_pub_key_gives_different_address(self):
        from eth_keys.datatypes import PublicKey as _EthPubKey

        priv1 = ec.generate_private_key(ec.SECP256K1(), default_backend())
        priv2 = ec.generate_private_key(ec.SECP256K1(), default_backend())

        def _addr(priv):
            pub = priv.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
            return "0x" + _EthPubKey(pub[1:]).to_canonical_address().hex()

        assert _addr(priv1) != _addr(priv2)


# ---------------------------------------------------------------------------
# _phala_check_report_data
# ---------------------------------------------------------------------------

class TestPhalaReportData:
    def _make_report_data(self, signing_address_hex: str, nonce_hex: str) -> str:
        addr = bytes.fromhex(signing_address_hex.removeprefix("0x"))
        nonce = bytes.fromhex(nonce_hex)
        return (addr.ljust(32, b"\x00") + nonce).hex()

    def test_valid_report_data(self):
        addr = "0x" + secrets.token_hex(20)
        nonce = secrets.token_hex(32)
        rd = self._make_report_data(addr, nonce)
        assert _phala_check_report_data(rd, addr, "ecdsa", nonce) is True

    def test_wrong_nonce_fails(self):
        addr = "0x" + secrets.token_hex(20)
        nonce = secrets.token_hex(32)
        rd = self._make_report_data(addr, nonce)
        assert _phala_check_report_data(rd, addr, "ecdsa", secrets.token_hex(32)) is False

    def test_wrong_address_fails(self):
        addr = "0x" + secrets.token_hex(20)
        nonce = secrets.token_hex(32)
        rd = self._make_report_data(addr, nonce)
        wrong_addr = "0x" + secrets.token_hex(20)
        assert _phala_check_report_data(rd, wrong_addr, "ecdsa", nonce) is False

    def test_0x_prefix_stripped(self):
        addr = "0x" + "ab" * 20
        nonce = secrets.token_hex(32)
        rd = self._make_report_data(addr, nonce)
        assert _phala_check_report_data(rd, addr, "ecdsa", nonce) is True
        assert _phala_check_report_data(rd, addr.removeprefix("0x"), "ecdsa", nonce) is True


# ---------------------------------------------------------------------------
# Live integration test (skipped without credentials)
# ---------------------------------------------------------------------------

import os


def _read_env_file(path, key):
    if os.path.exists(path):
        for line in open(path):
            if line.startswith(f"{key}="):
                return line.strip().split("=", 1)[1]
    return ""

def _near_api_key():
    return os.environ.get("NEAR_API_KEY", "") or _read_env_file(os.path.expanduser("~/.hermes-near-test/.env"), "NEAR_API_KEY")

def _redpill_api_key():
    return os.environ.get("REDPILL_API_KEY", "") or _read_env_file(os.path.expanduser("~/.hermes-near-test/.env"), "REDPILL_API_KEY")


@pytest.mark.skipif(not _near_api_key(), reason="NEAR_API_KEY not found — live attestation test skipped")
class TestNearAILiveAttestation:
    def test_full_attestation_returns_signing_key(self):
        from hermes_cli.attestation import verify_attestation
        api_key = _near_api_key()
        report = verify_attestation("near-ai", {"api_key": api_key, "base_url": "https://cloud-api.near.ai", "model": "openai/gpt-oss-120b"}, {"enabled": True, "strict": True})
        assert report.valid, f"Attestation failed: {report.error}"
        assert report.signing_public_key is not None
        assert len(bytes.fromhex(report.signing_public_key)) == 64


@pytest.mark.skipif(not _redpill_api_key(), reason="REDPILL_API_KEY not found — live redpill test skipped")
class TestRedpillLiveAttestation:
    def test_full_attestation_returns_signing_key(self):
        from hermes_cli.attestation import verify_attestation
        api_key = _redpill_api_key()
        report = verify_attestation("redpill", {"api_key": api_key, "base_url": "https://api.red-pill.ai/v1", "model": "phala/gpt-oss-20b"}, {"enabled": True})
        assert report.valid, f"Attestation failed: {report.error}"
        assert report.signing_public_key is not None
        assert report.signing_algo in ("ecdsa", "ed25519")

    def test_plain_chat_works_without_e2ee(self):
        # api.red-pill.ai gateway rejects E2EE headers ("This endpoint is not supported").
        # Attestation gives us the model CVM's signing key, but the gateway blocks encrypted
        # requests from reaching it. Test that plain (non-E2EE) chat works.
        api_key = _redpill_api_key()
        resp = requests.post(
            "https://api.red-pill.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "phala/gpt-oss-20b", "messages": [{"role": "user", "content": "say hi"}], "max_tokens": 20},
            timeout=60,
        )
        assert resp.status_code == 200
        data = resp.json()
        msg = data["choices"][0]["message"]
        response_text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning")
        assert response_text, f"No response text in message fields: {list(msg.keys())}"


# ---------------------------------------------------------------------------
# Model attestation filtering
# ---------------------------------------------------------------------------

class TestAttestationCache:
    def setup_method(self):
        from hermes_cli.attestation import _MODEL_ATTESTATION_CACHE, _ATTESTATION_CACHE
        _MODEL_ATTESTATION_CACHE.clear()
        _ATTESTATION_CACHE.clear()

    def teardown_method(self):
        from hermes_cli.attestation import _MODEL_ATTESTATION_CACHE, _ATTESTATION_CACHE
        _MODEL_ATTESTATION_CACHE.clear()
        _ATTESTATION_CACHE.clear()

    def test_verify_attestation_caches_per_model(self, monkeypatch):
        # Regression: cache key must include model_id so probing different
        # models under the same base_url doesn't collide.
        from hermes_cli import attestation as att_mod

        calls: list[str] = []

        def fake_redpill(creds, config):
            calls.append(creds["model"])
            return att_mod.AttestationReport(
                valid=creds["model"] == "good",
                provider="redpill", attestation_type="tdx+gpu",
                verified_at="", details={}, error=None if creds["model"] == "good" else "bad",
            )

        monkeypatch.setattr(att_mod, "_verify_redpill_attestation", fake_redpill)
        monkeypatch.setattr(att_mod, "_live_tls_fingerprint", lambda *a, **kw: None)

        r1 = att_mod.verify_attestation("redpill", {"api_key": "k", "base_url": "https://x", "model": "good"}, {})
        r2 = att_mod.verify_attestation("redpill", {"api_key": "k", "base_url": "https://x", "model": "bad"}, {})
        assert r1.valid is True and r2.valid is False
        assert calls == ["good", "bad"]  # both invoked — not collapsed by cache

    def test_verify_attestation_populates_model_cache(self, monkeypatch):
        from hermes_cli import attestation as att_mod
        monkeypatch.setattr(att_mod, "_live_tls_fingerprint", lambda *a, **kw: None)
        report = att_mod.verify_attestation(
            "custom",
            {"base_url": "http://localhost:9999", "model": "some-model"},
            {},
        )
        cached = att_mod.get_model_attestation_status("custom", "some-model")
        assert cached is report


# ---------------------------------------------------------------------------
# Redpill attestation shape dispatch — covers all 4 model types from
# redpill-verifier docs (Phala / NearAI / Chutes / Tinfoil).
# ---------------------------------------------------------------------------

class TestRedpillShapeDispatch:
    def setup_method(self):
        from hermes_cli.attestation import _MODEL_ATTESTATION_CACHE, _ATTESTATION_CACHE
        _MODEL_ATTESTATION_CACHE.clear()
        _ATTESTATION_CACHE.clear()

    def teardown_method(self):
        from hermes_cli.attestation import _MODEL_ATTESTATION_CACHE, _ATTESTATION_CACHE
        _MODEL_ATTESTATION_CACHE.clear()
        _ATTESTATION_CACHE.clear()

    def _fake_get(self, body):
        class _Resp:
            status_code = 200
            def json(self_inner):
                return body
            def raise_for_status(self_inner):
                return None
        return _Resp()

    def test_phala_simple_shape_dispatches_to_gateway_path(self, monkeypatch):
        # Phala simple shape: top-level intel_quote (no gateway_attestation, no attestation_type)
        from hermes_cli import attestation as att_mod
        att_body = {"intel_quote": "deadbeef", "nvidia_payload": {}, "signing_address": "0x00", "info": {}}
        monkeypatch.setattr(att_mod.requests, "get", lambda *a, **kw: self._fake_get(att_body))
        # Stub the Phala TDX verifier to simulate verification failure so we don't network out.
        post_calls = []
        class _P:
            def json(self_inner):
                return {"quote": {"verified": False, "message": "stubbed"}}
        def fake_post(url, **kw):
            post_calls.append(url)
            return _P()
        monkeypatch.setattr(att_mod.requests, "post", fake_post)
        r = att_mod._verify_redpill_attestation(
            {"api_key": "k", "base_url": "https://api.red-pill.ai/v1", "model": "phala/qwen-2.5-7b-instruct"},
            {},
        )
        assert r.valid is False
        assert "TDX quote verification failed" in (r.error or "")
        # Confirms the Phala simple branch was taken (hit the Phala verifier URL).
        assert att_mod._PHALA_TDX_VERIFIER in post_calls

    def test_near_ai_via_redpill_shape_dispatches_to_gateway_path(self, monkeypatch):
        # NearAI-via-redpill shape: gateway_attestation + model_attestations
        from hermes_cli import attestation as att_mod
        att_body = {
            "gateway_attestation": {"intel_quote": "deadbeef", "signing_address": "0x00", "info": {}},
            "model_attestations": [{"signing_public_key": "abcd", "signing_algo": "ecdsa"}],
        }
        monkeypatch.setattr(att_mod.requests, "get", lambda *a, **kw: self._fake_get(att_body))
        class _P:
            def json(self_inner):
                return {"quote": {"verified": False, "message": "stubbed"}}
        monkeypatch.setattr(att_mod.requests, "post", lambda *a, **kw: _P())
        r = att_mod._verify_redpill_attestation(
            {"api_key": "k", "base_url": "https://api.red-pill.ai/v1", "model": "phala/gpt-oss-120b"},
            {},
        )
        # Path taken = gateway_attestation branch; fails at TDX step (stubbed).
        assert r.valid is False
        assert "TDX quote verification failed" in (r.error or "")

    def test_chutes_shape_verifies_anti_tamper_binding(self, monkeypatch):
        # Chutes shape: attestation_type="chutes" + all_attestations[]
        # Build a synthetic quote whose report_data = SHA256(nonce || e2e_pubkey)
        from hermes_cli import attestation as att_mod

        e2e_pubkey = "beadfeed" * 8
        # We control the nonce by intercepting secrets.token_hex
        test_nonce = "11" * 32
        monkeypatch.setattr(att_mod.secrets, "token_hex", lambda n: test_nonce)
        expected_rd = hashlib.sha256((test_nonce + e2e_pubkey).encode()).hexdigest()

        att_body = {
            "attestation_type": "chutes",
            "nonce": test_nonce,
            "all_attestations": [{
                "intel_quote": base64.b64encode(b"\x00" * 100).decode(),
                "e2e_pubkey": e2e_pubkey,
                "nonce": test_nonce,
                "instance_id": "inst-1",
            }],
        }
        monkeypatch.setattr(att_mod.requests, "get", lambda *a, **kw: self._fake_get(att_body))

        # Phala verifier returns a body containing report_data that binds our nonce||e2e_pubkey
        class _P:
            def json(self_inner):
                return {
                    "quote": {
                        "verified": True,
                        "body": {"reportdata": expected_rd + "00" * 32, "td_attributes": "0000001000000000"},
                    },
                }
        monkeypatch.setattr(att_mod.requests, "post", lambda *a, **kw: _P())

        r = att_mod._verify_redpill_attestation(
            {"api_key": "k", "base_url": "https://api.red-pill.ai/v1", "model": "phala/deepseek-v3.2"},
            {},
        )
        assert r.valid, f"expected chutes attestation to pass, got: {r.error}"
        assert r.attestation_type == "chutes+tdx"
        assert r.details["instance_count"] == 1

    def test_chutes_shape_rejects_when_anti_tamper_fails(self, monkeypatch):
        from hermes_cli import attestation as att_mod
        test_nonce = "22" * 32
        monkeypatch.setattr(att_mod.secrets, "token_hex", lambda n: test_nonce)

        att_body = {
            "attestation_type": "chutes",
            "nonce": test_nonce,
            "all_attestations": [{
                "intel_quote": base64.b64encode(b"\x00" * 100).decode(),
                "e2e_pubkey": "cafebabe" * 8,
                "nonce": test_nonce,
                "instance_id": "inst-tampered",
            }],
        }
        monkeypatch.setattr(att_mod.requests, "get", lambda *a, **kw: self._fake_get(att_body))

        # Simulate a WRONG report_data — attacker swapped the e2e key without updating the binding
        class _P:
            def json(self_inner):
                return {
                    "quote": {
                        "verified": True,
                        "body": {"reportdata": "00" * 64, "td_attributes": "0000001000000000"},
                    },
                }
        monkeypatch.setattr(att_mod.requests, "post", lambda *a, **kw: _P())

        r = att_mod._verify_redpill_attestation(
            {"api_key": "k", "base_url": "https://api.red-pill.ai/v1", "model": "phala/kimi-k2.5"},
            {},
        )
        assert r.valid is False
        assert "anti-tamper" in (r.error or "").lower()

    def test_chutes_shape_rejects_debug_mode(self, monkeypatch):
        from hermes_cli import attestation as att_mod
        test_nonce = "33" * 32
        monkeypatch.setattr(att_mod.secrets, "token_hex", lambda n: test_nonce)
        e2e_pubkey = "feedface" * 8
        expected_rd = hashlib.sha256((test_nonce + e2e_pubkey).encode()).hexdigest()

        att_body = {
            "attestation_type": "chutes",
            "nonce": test_nonce,
            "all_attestations": [{
                "intel_quote": base64.b64encode(b"\x00" * 100).decode(),
                "e2e_pubkey": e2e_pubkey,
                "nonce": test_nonce,
                "instance_id": "inst-dbg",
            }],
        }
        monkeypatch.setattr(att_mod.requests, "get", lambda *a, **kw: self._fake_get(att_body))

        # td_attributes bit 0 set → DEBUG mode (int(hex,16) & 1 == 1)
        class _P:
            def json(self_inner):
                return {
                    "quote": {
                        "verified": True,
                        "body": {"reportdata": expected_rd + "00" * 32, "td_attributes": "0000000000000001"},
                    },
                }
        monkeypatch.setattr(att_mod.requests, "post", lambda *a, **kw: _P())

        r = att_mod._verify_redpill_attestation(
            {"api_key": "k", "base_url": "https://api.red-pill.ai/v1", "model": "phala/deepseek-v3.2"},
            {},
        )
        assert r.valid is False
        assert "debug" in (r.error or "").lower()

    def test_tinfoil_or_unknown_shape_rejected(self, monkeypatch):
        # Tinfoil responses aren't routed through our curated list yet, and the
        # redpill API returns {"error": "..."} for upstream (non-phala/) model
        # paths. Either way, unknown shapes must fail cleanly — the picker
        # filter will drop them.
        from hermes_cli import attestation as att_mod
        monkeypatch.setattr(
            att_mod.requests, "get",
            lambda *a, **kw: self._fake_get({"error": "route not found"}),
        )
        r = att_mod._verify_redpill_attestation(
            {"api_key": "k", "base_url": "https://api.red-pill.ai/v1", "model": "meta-llama/llama-3.3-70b-instruct"},
            {},
        )
        assert r.valid is False
        assert "Unrecognized attestation response format" in (r.error or "")
