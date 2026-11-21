"""#138627: a truncated (0-byte) creds.json must fail preflight instead of booting a headless QR bridge."""
import json
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.run import _profile_runtime_scope
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter, _has_valid_creds

VALID_CREDS = json.dumps({
    "noiseKey": {"private": "x" * 43, "public": "y" * 32},
    "signedIdentityKey": {"private": "a" * 43, "public": "b" * 32},
    "signedPreKey": {"private": "c" * 43, "public": "d" * 32},
    "registrationId": 4029384,
    "advSecretKey": "e" * 64,
})


def _adapter(tmp_path: Path) -> WhatsAppAdapter:
    (tmp_path / "bridge.js").write_text("// stub bridge script")
    with _profile_runtime_scope(tmp_path, hydrate_secrets=False):
        return WhatsAppAdapter(PlatformConfig(extra={
            "session_path": str(tmp_path / "session"),
            "bridge_script": str(tmp_path / "bridge.js"),
        }))


def _gate_environment(monkeypatch, adapter):
    monkeypatch.setattr("plugins.platforms.whatsapp.adapter.check_whatsapp_requirements", lambda: True)
    monkeypatch.setattr("plugins.platforms.whatsapp.adapter.ensure_aiohttp", lambda: None)
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", lambda *a, **k: None)


@pytest.mark.parametrize("content", [
    b"",                                            # 0-byte truncation from the issue report
    "",                                             # same, via text
    '{"noiseKey": "' + "a" * 200,                   # killed mid-write: valid JSON prefix, truncated value
    json.dumps({"registrationId": 4029384, "advSecretKey": "e" * 64}),  # valid JSON, never finished pairing
    json.dumps(["not", "a", "creds", "dict"] * 20),  # >100 bytes of valid JSON, wrong shape
], ids=["zero-bytes", "empty-text", "truncated-json", "half-paired", "wrong-shape"])
def test_invalid_creds_fail_validation(tmp_path, content):
    path = tmp_path / "creds.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    assert _has_valid_creds(path) is False


def test_valid_creds_pass_validation(tmp_path):
    path = tmp_path / "creds.json"
    path.write_text(VALID_CREDS)
    assert _has_valid_creds(path) is True


def test_missing_creds_fails_validation(tmp_path):
    assert _has_valid_creds(tmp_path / "absent.json") is False


def test_zero_byte_creds_fails_preflight_with_named_error(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    session = adapter._session_path
    session.mkdir(parents=True)
    (session / "creds.json").write_bytes(b"")
    _gate_environment(monkeypatch, adapter)
    assert adapter._preflight() is False
    assert adapter._fatal_error_code == "whatsapp_creds_invalid"
    assert adapter._fatal_error_retryable is False


def test_valid_creds_pass_preflight(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    session = adapter._session_path
    session.mkdir(parents=True)
    (session / "creds.json").write_text(VALID_CREDS)
    _gate_environment(monkeypatch, adapter)
    assert adapter._preflight() is True
    assert adapter._fatal_error_code is None


def test_missing_creds_keeps_not_paired_error(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    adapter._session_path.mkdir(parents=True)
    _gate_environment(monkeypatch, adapter)
    assert adapter._preflight() is False
    assert adapter._fatal_error_code == "whatsapp_not_paired"
