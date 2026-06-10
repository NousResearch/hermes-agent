"""Regression tests for Claude Code OAuth credential source selection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agent.anthropic_adapter import read_claude_code_credentials


def _cred(access_token: str, expires_at: int) -> dict:
    return {
        "accessToken": access_token,
        "refreshToken": f"refresh-{access_token}",
        "expiresAt": expires_at,
    }


def test_expired_keychain_does_not_shadow_valid_credentials_file(tmp_path: Path):
    """Claude Code migrations can leave stale Keychain data plus fresh file data."""
    valid_expiry_ms = 4_102_444_800_000  # 2100-01-01T00:00:00Z
    expired_expiry_ms = 1_577_836_800_000  # 2020-01-01T00:00:00Z
    cred_dir = tmp_path / ".claude"
    cred_dir.mkdir()
    (cred_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": _cred("file-token", valid_expiry_ms)}),
        encoding="utf-8",
    )

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch(
            "agent.anthropic_adapter._read_claude_code_credentials_from_keychain",
            return_value={**_cred("keychain-token", expired_expiry_ms), "source": "macos_keychain"},
        ),
    ):
        creds = read_claude_code_credentials()

    assert creds is not None
    assert creds["accessToken"] == "file-token"
    assert creds["source"] == "claude_code_credentials_file"


def test_valid_keychain_still_wins_over_credentials_file(tmp_path: Path):
    valid_expiry_ms = 4_102_444_800_000  # 2100-01-01T00:00:00Z
    cred_dir = tmp_path / ".claude"
    cred_dir.mkdir()
    (cred_dir / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": _cred("file-token", valid_expiry_ms)}),
        encoding="utf-8",
    )

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch(
            "agent.anthropic_adapter._read_claude_code_credentials_from_keychain",
            return_value={**_cred("keychain-token", valid_expiry_ms), "source": "macos_keychain"},
        ),
    ):
        creds = read_claude_code_credentials()

    assert creds is not None
    assert creds["accessToken"] == "keychain-token"
    assert creds["source"] == "macos_keychain"
