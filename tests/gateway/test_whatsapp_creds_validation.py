"""Regression tests for WhatsApp pairing state validation (issue #85391, Bug 2).

Existence-only ``creds.json`` checks let a 0-byte / truncated file count as
"paired". In ``--pair-only`` mode the bridge writes ``creds.json`` *after*
emitting ``connected`` and then exits on its own, so a supervisor that kills
the bridge on ``connected`` can truncate the file — leaving hundreds of
pre-key files present but ``creds.json`` empty, yet the gateway proceeds on
unusable credentials. ``_has_valid_creds`` validates content, not just
existence.
"""

import json

import pytest

from plugins.platforms.whatsapp.adapter import _has_valid_creds


_GOOD_CREDS = {
    "noiseKey": {"private": "x", "public": "y"},
    "signedIdentityKey": {"private": "a", "public": "b"},
    "me": {"id": "9955xxxx:2@s.whatsapp.net"},
}


def test_missing_file_is_invalid(tmp_path):
    assert _has_valid_creds(tmp_path / "creds.json") is False


def test_zero_byte_file_is_invalid(tmp_path):
    """The exact reported failure: a 0-byte creds.json must not pass."""
    p = tmp_path / "creds.json"
    p.write_text("", encoding="utf-8")
    assert p.exists()  # existence-only check would (wrongly) pass here
    assert _has_valid_creds(p) is False


def test_truncated_non_json_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text('{"noiseKey": {"priv', encoding="utf-8")  # half-written
    assert _has_valid_creds(p) is False


def test_json_without_identity_keys_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"registered": False}), encoding="utf-8")
    assert _has_valid_creds(p) is False


def test_non_object_json_is_invalid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("[]", encoding="utf-8")
    assert _has_valid_creds(p) is False


def test_valid_creds_pass(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps(_GOOD_CREDS), encoding="utf-8")
    assert _has_valid_creds(p) is True


def test_adapter_reexports_shared_validator():
    """The adapter's ``_has_valid_creds`` and the wizard both resolve to the
    single ``whatsapp_common`` implementation, so validation can never drift
    between the gateway reader and the pairing wizard (issue #85391)."""
    from gateway.platforms.whatsapp_common import has_valid_whatsapp_creds

    assert _has_valid_creds is has_valid_whatsapp_creds
