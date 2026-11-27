"""``security -i`` cuts a stdin line at 4096 bytes: an oversized Claude Code Keychain item used to be
stored as invalid JSON (and Claude Code logged out). The mirror must leave it untouched instead."""

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from agent.anthropic_credentials import _keychain_mirror_command, _mirror_claude_code_credentials_to_keychain


def _fake_security_i(store):
    """Double of ``security -i``: runs ``add-generic-password -X <hex>`` but, like the real tool,
    keeps only the first 4096 bytes of the stdin line (newline included)."""

    def run(argv, **kwargs):
        # Recorded, not asserted: the mirror swallows exceptions raised inside a double.
        store["argv"] = list(argv)
        line = kwargs["input"].encode("utf-8")[:4096].decode("utf-8", "ignore")
        digits = line.split(" -X ", 1)[1].strip()
        store["blob"] = bytes.fromhex(digits[: len(digits) // 2 * 2])
        return MagicMock(returncode=0, stderr="")

    return run


def _item(mcp_bytes):
    return {"claudeAiOauth": {"accessToken": "A0", "refreshToken": "R0"}, "mcpOAuth": {"s": "m" * mcp_bytes}}


@pytest.mark.parametrize("mcp_bytes", [3 * 1024, 25 * 1024])
def test_oversized_payload_builds_no_command(mcp_bytes):
    assert _keychain_mirror_command("bob", _item(mcp_bytes)) is None


def test_payload_that_fits_is_still_written_intact():
    payload = _item(1200)
    argv, line = _keychain_mirror_command("bob", payload)
    assert len(line.encode()) <= 4096
    assert json.loads(bytes.fromhex(line.split(" -X ", 1)[1].strip())) == payload


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("mcp_bytes", [10, 1200, 3 * 1024, 25 * 1024])
def test_item_is_never_left_truncated(monkeypatch, mcp_bytes):
    existing = _item(mcp_bytes)
    old = json.dumps(existing).encode()
    store = {"blob": old}
    monkeypatch.setattr("agent.anthropic_credentials._find_claude_code_keychain_item", lambda: ("bob", existing))
    monkeypatch.setattr(subprocess, "run", _fake_security_i(store))

    _mirror_claude_code_credentials_to_keychain("A-SECRET", "R-SECRET", 1, spent_refresh_token="R0")

    if "argv" in store:  # nothing secret on argv, so nothing in ``ps``
        assert store["argv"] == ["security", "-i"]
    written = json.loads(store["blob"])  # always valid JSON: the old item or the complete new one
    if store["blob"] == old:
        assert mcp_bytes > 1200  # skipped only when it cannot fit
    else:
        assert written["claudeAiOauth"]["refreshToken"] == "R-SECRET"
        assert written["mcpOAuth"] == existing["mcpOAuth"]


@pytest.mark.platforms("macos")
def test_skip_is_logged_and_nothing_is_run(monkeypatch, caplog):
    item = ("bob", _item(25 * 1024))
    monkeypatch.setattr("agent.anthropic_credentials._find_claude_code_keychain_item", lambda: item)
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)

    with caplog.at_level("WARNING"):
        _mirror_claude_code_credentials_to_keychain("A1", "R1", 1, spent_refresh_token="R0")

    run.assert_not_called()
    assert "too large" in caplog.text
