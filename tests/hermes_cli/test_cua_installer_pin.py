"""Pinned cua-driver installer identity and digest checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.cua_installer_pin import (
    CUA_INSTALLER_REF,
    CUA_INSTALLER_SHA256_PS1,
    CUA_INSTALLER_SHA256_SH,
    CuaInstallerIntegrityError,
    cua_installer_expected_sha256,
    cua_installer_url,
    verify_cua_installer_digest,
)
from hermes_cli.tools_config import _write_verified_cua_installer


class _BytesResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_url_is_pinned_not_main():
    posix = cua_installer_url(is_windows=False)
    windows = cua_installer_url(is_windows=True)
    assert "/main/" not in posix
    assert "/main/" not in windows
    assert CUA_INSTALLER_REF in posix
    assert CUA_INSTALLER_REF in windows
    assert posix.endswith("/install.sh")
    assert windows.endswith("/install.ps1")


def test_verify_accepts_matching_digest():
    payload = b"trusted installer\n"
    expected = hashlib.sha256(payload).hexdigest()
    assert verify_cua_installer_digest(payload, expected) == expected


def test_verify_rejects_mismatch():
    with pytest.raises(CuaInstallerIntegrityError, match="sha256 mismatch"):
        verify_cua_installer_digest(b"evil", "0" * 64)


def test_write_verified_refuses_tampered_bytes(tmp_path):
    dest = tmp_path / "install.sh"
    with patch("urllib.request.urlopen", return_value=_BytesResp(b"not the pinned installer\n")):
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            _write_verified_cua_installer(str(dest), is_windows=False)
    assert not dest.exists()


def test_write_verified_keeps_matching_bytes(tmp_path):
    dest = tmp_path / "install.sh"
    payload = b"trusted installer\n"
    expected = hashlib.sha256(payload).hexdigest()
    with patch("urllib.request.urlopen", return_value=_BytesResp(payload)), patch(
        "hermes_cli.cua_installer_pin.cua_installer_expected_sha256",
        return_value=expected,
    ):
        _write_verified_cua_installer(str(dest), is_windows=False)
    assert dest.read_bytes() == payload


def test_pin_file_matches_module_constants():
    pin_path = Path(__file__).resolve().parents[2] / "scripts" / "cua_installer_pin.env"
    raw = pin_path.read_text(encoding="utf-8")
    assert f"CUA_INSTALLER_REF={CUA_INSTALLER_REF}" in raw
    assert f"CUA_INSTALLER_SHA256_SH={CUA_INSTALLER_SHA256_SH}" in raw
    assert f"CUA_INSTALLER_SHA256_PS1={CUA_INSTALLER_SHA256_PS1}" in raw
    assert cua_installer_expected_sha256(is_windows=False) == CUA_INSTALLER_SHA256_SH
    assert cua_installer_expected_sha256(is_windows=True) == CUA_INSTALLER_SHA256_PS1


def test_live_pinned_posix_installer_matches_recorded_digest():
    import urllib.request

    url = cua_installer_url(is_windows=False)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
        status = getattr(resp, "status", 200)
    assert status == 200
    assert verify_cua_installer_digest(data, CUA_INSTALLER_SHA256_SH) == CUA_INSTALLER_SHA256_SH
