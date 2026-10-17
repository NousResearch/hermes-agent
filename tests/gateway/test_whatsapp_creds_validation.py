"""Tests for ``has_valid_whatsapp_creds`` — the single source of truth both the
gateway adapter and the CLI pairing wizard use to decide whether ``creds.json``
represents a genuine WhatsApp pairing.

Regression for the "enabled but not paired" restart loop: in ``--pair-only``
mode the bridge writes ``creds.json`` *after* emitting ``connected`` and then
exits, so a supervisor that kills it on ``connected`` can leave a 0-byte or
half-written file behind. That truncated file passes ``Path.exists()``, so the
wizard reports success while the gateway then rejects it every restart.
"""

import json

from gateway.platforms.whatsapp_common import has_valid_whatsapp_creds

_VALID = {
    "noiseKey": {"private": "aa", "public": "bb"},
    "signedIdentityKey": {"private": "cc", "public": "dd"},
    "registered": True,
}


def test_missing_file_is_invalid(tmp_path):
    assert has_valid_whatsapp_creds(tmp_path / "creds.json") is False


def test_empty_file_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("", encoding="utf-8")
    assert has_valid_whatsapp_creds(p) is False


def test_truncated_non_json_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("{\"noiseKey\": {\"priv", encoding="utf-8")
    assert has_valid_whatsapp_creds(p) is False


def test_json_without_identity_keys_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"registered": True}), encoding="utf-8")
    assert has_valid_whatsapp_creds(p) is False


def test_non_object_json_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps(["noiseKey", "signedIdentityKey"]), encoding="utf-8")
    assert has_valid_whatsapp_creds(p) is False


def test_directory_in_place_of_file_is_invalid(tmp_path):
    d = tmp_path / "creds.json"
    d.mkdir()
    assert has_valid_whatsapp_creds(d) is False


def test_genuine_pairing_is_valid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps(_VALID), encoding="utf-8")
    assert has_valid_whatsapp_creds(p) is True
