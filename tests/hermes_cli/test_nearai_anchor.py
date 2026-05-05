"""Tests for NEAR AI static anchor enforcement (Block B-static).

Until on-chain Base RPC reads land, hermes_cli/anchors/nearai_mainnet.json
is the authority for "is this CVM permitted to serve model M". These
tests cover the four pinned fields (app_id, compose_hash, os_image_hash,
key_provider_info.id) and the not-anchored model case. Live integration
coverage is in test_nearai_e2ee.py::TestNearAILiveAttestation."""

import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from hermes_cli import attestation as att_mod
from hermes_cli.anchors import expected_for_model, load_nearai_anchor


_ANCHOR = load_nearai_anchor()
_GLM = "zai-org/GLM-5.1-FP8"


def _override_anchor(compose_hash_hex):
    """Build a fresh anchor dict that pins a synthetic compose_hash for GLM and
    install it as att_mod._NEARAI_ANCHOR (which the verifier reads at call time)."""
    new = {
        "kms_contract_addr": _ANCHOR["kms_contract_addr"],
        "kms_provider_info_id": _ANCHOR["kms_provider_info_id"],
        "os_image_hashes": list(_ANCHOR["os_image_hashes"]),
        "models": {
            _GLM: {
                "app_id": _ANCHOR["models"][_GLM]["app_id"],
                "compose_hashes": [compose_hash_hex],
            },
        },
    }
    return new


def _mock_check_tdx_quote_for(compose_hash_hex):
    """Returns a fake check_tdx_quote that yields a verified quote with a mr_config
    consistent with the attestation's app_compose hash."""
    async def _fake(_payload):
        return {
            "verified": True,
            "status": "OK",
            "advisory_ids": [],
            "mrtd": "deadbeef" * 12,
            "quote": {"body": {"mrconfig": "01" + compose_hash_hex + ("00" * 16)}},
        }
    return _fake


def _mock_check_report_data(_payload, _nonce, _intel):
    return {"binds_address": True, "embeds_nonce": True}


def _mock_check_gpu(_payload, _nonce):
    return {"verdict": "PASS", "nonce_matches": True}


def _mock_verify_domain(_atte):
    return None


def _build_payload(app_id, compose_str, os_image_hash, kpi_id, signing_pub_hex, signing_addr):
    """Construct an attestation HTTP payload that the verifier will accept up to
    the anchor check (TDX/report_data/GPU/domain mocked as success)."""
    info = {
        "app_id": app_id,
        "os_image_hash": os_image_hash,
        "key_provider_info": json.dumps({"name": "kms", "id": kpi_id}),
        "tcb_info": json.dumps({"app_compose": compose_str}),
    }
    model_att = {
        "info": info,
        "signing_address": signing_addr,
        "signing_public_key": signing_pub_hex,
        "nvidia_payload": "x",
        "intel_quote": "x",
    }
    gateway = {
        "info": {"app_id": app_id},
        "signing_address": "0xdeadbeef" + "00" * 16,
        "tls_cert_fingerprint": "ab" * 32,
        "intel_quote": "x",
    }
    return {
        "gateway_attestation": gateway,
        "model_attestations": [model_att],
        "tls_certificate": "-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----\n",
    }


def _patched_run(payload):
    """Drive _verify_near_ai_attestation through the anchor logic with all
    upstream verifier checks mocked to PASS."""
    fake_resp = type("R", (), {
        "json": lambda self: payload,
        "raise_for_status": lambda self: None,
    })()
    compose_hash = hashlib.sha256(json.loads(payload["model_attestations"][0]["info"]["tcb_info"])["app_compose"].encode()).hexdigest()
    with patch.object(att_mod.requests, "get", return_value=fake_resp), \
         patch.object(att_mod, "check_tdx_quote", _mock_check_tdx_quote_for(compose_hash)), \
         patch.object(att_mod, "check_report_data", _mock_check_report_data), \
         patch.object(att_mod, "check_gpu", _mock_check_gpu), \
         patch.object(att_mod, "verify_domain_attestation", _mock_verify_domain):
        return att_mod._verify_near_ai_attestation(
            {"api_key": "x", "base_url": "https://example.test", "model": _GLM},
            {},
        )


# A fixed signing keypair whose public derives to a known address.
# Pulled from a deterministic test vector (private = b"\x01" * 32 -> known pub).
# eth_keys: PrivateKey(b'\x01'*32).public_key
_SIGNING_PUB_HEX = "1b84c5567b126440995d3ed5aaba0565d71e1834604819ff9c17f5e9d5dd078f70beaf8f588b541507fed6a642c5ab42dfdf8120a7f639de5122d47a69a8e8d1"
_SIGNING_ADDR = "0x1a642f0e3c3af545e7acbd38b07251b3990914f1"


class _AnchorTestBase(unittest.TestCase):
    SYNTHETIC = "appcompose-fixture"

    def setUp(self):
        self.synthetic_hash = hashlib.sha256(self.SYNTHETIC.encode()).hexdigest()
        self._anchor_patcher = patch.object(
            att_mod, "_NEARAI_ANCHOR", _override_anchor(self.synthetic_hash)
        )
        self._anchor_patcher.start()

    def tearDown(self):
        self._anchor_patcher.stop()

    def _payload(self, **overrides):
        glm = att_mod._NEARAI_ANCHOR["models"][_GLM]
        defaults = dict(
            app_id=glm["app_id"],
            compose_str=self.SYNTHETIC,
            os_image_hash=att_mod._NEARAI_ANCHOR["os_image_hashes"][0],
            kpi_id=att_mod._NEARAI_ANCHOR["kms_provider_info_id"],
            signing_pub_hex=_SIGNING_PUB_HEX,
            signing_addr=_SIGNING_ADDR,
        )
        defaults.update(overrides)
        return _build_payload(**defaults)


class TestAnchorPositive(_AnchorTestBase):
    """Anchor matches → valid=True."""

    def test_matching_attestation_validates(self):
        report = _patched_run(self._payload())
        self.assertTrue(report.valid, f"expected valid; got error={report.error}")
        self.assertTrue(report.details["models"][0]["anchor_matched"])


class TestAnchorRefusals(_AnchorTestBase):
    """Each pinned field should fail closed independently."""

    def test_wrong_app_id_fails(self):
        report = _patched_run(self._payload(app_id="ff" * 20))
        self.assertFalse(report.valid)
        self.assertIn("app_id", report.error)

    def test_wrong_compose_fails(self):
        report = _patched_run(self._payload(compose_str="something-else"))
        self.assertFalse(report.valid)
        self.assertIn("compose_hash", report.error)

    def test_wrong_os_image_fails(self):
        report = _patched_run(self._payload(os_image_hash="ee" * 32))
        self.assertFalse(report.valid)
        self.assertIn("os_image_hash", report.error)

    def test_wrong_kpi_id_fails(self):
        report = _patched_run(self._payload(kpi_id="3059" + "ab" * 87))
        self.assertFalse(report.valid)
        self.assertIn("key_provider_info", report.error)

    def test_unanchored_model_fails(self):
        payload = self._payload()
        fake_resp = type("R", (), {
            "json": lambda self: payload,
            "raise_for_status": lambda self: None,
        })()
        with patch.object(att_mod.requests, "get", return_value=fake_resp), \
             patch.object(att_mod, "check_tdx_quote", _mock_check_tdx_quote_for(self.synthetic_hash)), \
             patch.object(att_mod, "check_report_data", _mock_check_report_data), \
             patch.object(att_mod, "check_gpu", _mock_check_gpu), \
             patch.object(att_mod, "verify_domain_attestation", _mock_verify_domain):
            report = att_mod._verify_near_ai_attestation(
                {"api_key": "x", "base_url": "https://example.test", "model": "not-in-anchor/foo"},
                {},
            )
        self.assertFalse(report.valid)
        self.assertIn("not-in-anchor/foo", report.error)


class TestAnchorSchema(unittest.TestCase):
    """The shipped anchor file is well-formed."""

    def test_schema(self):
        a = load_nearai_anchor()
        self.assertIn("kms_contract_addr", a)
        self.assertIn("kms_provider_info_id", a)
        self.assertIn("os_image_hashes", a)
        self.assertIsInstance(a["models"], dict)
        for model, e in a["models"].items():
            self.assertIn("app_id", e, f"{model} missing app_id")
            self.assertIn("compose_hashes", e, f"{model} missing compose_hashes")
            self.assertGreater(len(e["compose_hashes"]), 0, f"{model} empty compose_hashes")


if __name__ == "__main__":
    unittest.main()
