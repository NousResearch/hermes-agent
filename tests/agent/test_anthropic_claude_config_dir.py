"""Tests for CLAUDE_CONFIG_DIR support in agent/anthropic_adapter.py.

Anthropic's docs (https://code.claude.com/docs/en/authentication) state that
on Linux/Windows, setting CLAUDE_CONFIG_DIR relocates
``.credentials.json`` under that directory instead of ``~/.claude``. These
tests cover the module-level ``_claude_code_credentials_path()`` resolver and
its use at both call sites: ``_read_claude_code_credentials_from_file`` and
``_write_claude_code_credentials``.
"""

import json
import time

import pytest

from agent.anthropic_adapter import (
    _claude_code_credentials_path,
    _read_claude_code_credentials_from_file,
    _write_claude_code_credentials,
)


def _write_credentials_file(path, access_token="tok", refresh_token="ref", expires_at=None):
    """Write a minimal valid Claude Code OAuth credentials JSON file at `path`."""
    if expires_at is None:
        expires_at = int(time.time() * 1000) + 3600_000
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "expiresAt": expires_at,
            }
        }),
        encoding="utf-8",
    )


class TestClaudeCodeCredentialsPathResolution:
    """Direct coverage of the ``_claude_code_credentials_path()`` helper."""

    def test_honors_claude_config_dir(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom-claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        # Decoy home: if the helper ignored CLAUDE_CONFIG_DIR and fell back
        # to Path.home(), the assertion below would fail against this path.
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")

        assert _claude_code_credentials_path() == custom_dir / ".credentials.json"

    def test_falls_back_to_default_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert _claude_code_credentials_path() == tmp_path / ".claude" / ".credentials.json"

    @pytest.mark.parametrize("blank_value", ["", "   ", "\t\n"])
    def test_falls_back_when_value_is_blank(self, tmp_path, monkeypatch, blank_value):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", blank_value)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        assert _claude_code_credentials_path() == tmp_path / ".claude" / ".credentials.json"

    def test_expands_tilde_in_value(self, tmp_path, monkeypatch):
        # os.path.expanduser reads $HOME directly, independent of Path.home().
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/custom-claude-config")
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")

        assert _claude_code_credentials_path() == tmp_path / "custom-claude-config" / ".credentials.json"

    def test_expands_dollar_var_in_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_CLAUDE_BASE", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "$HERMES_TEST_CLAUDE_BASE/custom-claude-config")
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")

        assert _claude_code_credentials_path() == tmp_path / "custom-claude-config" / ".credentials.json"

    def test_resolved_at_call_time_not_cached(self, tmp_path, monkeypatch):
        """The env var must be re-read on every call — no import-time caching."""
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        assert _claude_code_credentials_path() == tmp_path / ".claude" / ".credentials.json"

        later_dir = tmp_path / "set-after-first-call"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(later_dir))
        assert _claude_code_credentials_path() == later_dir / ".credentials.json"


class TestReadClaudeCodeCredentialsFromFileHonorsConfigDir:
    """``_read_claude_code_credentials_from_file`` must use the resolved path."""

    def test_reads_from_claude_config_dir(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom-claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        # Decoy default-location home with no credentials file: if the read
        # path fell back to it instead of honoring CLAUDE_CONFIG_DIR, this
        # would incorrectly return None.
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")
        _write_credentials_file(custom_dir / ".credentials.json", access_token="from-config-dir")

        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "from-config-dir"
        assert creds["source"] == "claude_code_credentials_file"

    def test_falls_back_to_default_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        _write_credentials_file(tmp_path / ".claude" / ".credentials.json", access_token="from-default")

        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "from-default"

    def test_blank_value_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "   ")
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
        _write_credentials_file(tmp_path / ".claude" / ".credentials.json", access_token="from-default-blank")

        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "from-default-blank"

    def test_tilde_in_value_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/custom-claude-config")
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")
        _write_credentials_file(
            tmp_path / "custom-claude-config" / ".credentials.json",
            access_token="from-tilde-path",
        )

        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "from-tilde-path"

    def test_returns_none_when_no_file_at_resolved_location(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty-config-dir"))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")

        assert _read_claude_code_credentials_from_file() is None


class TestWriteClaudeCodeCredentialsHonorsConfigDir:
    """``_write_claude_code_credentials`` must write to the resolved path."""

    def test_write_lands_in_claude_config_dir(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom-claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")

        _write_claude_code_credentials("new-tok", "new-ref", 99999)

        default_cred_file = tmp_path / "decoy-home" / ".claude" / ".credentials.json"
        assert not default_cred_file.exists(), "must not also write a stale duplicate at the default path"

        cred_file = custom_dir / ".credentials.json"
        assert cred_file.exists()
        data = json.loads(cred_file.read_text(encoding="utf-8"))
        assert data["claudeAiOauth"]["accessToken"] == "new-tok"
        assert data["claudeAiOauth"]["refreshToken"] == "new-ref"
        assert data["claudeAiOauth"]["expiresAt"] == 99999

    def test_round_trip_write_then_read_back(self, tmp_path, monkeypatch):
        """Write via CLAUDE_CONFIG_DIR, then confirm the read path finds it."""
        custom_dir = tmp_path / "custom-claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")
        expires_at = int(time.time() * 1000) + 3600_000

        _write_claude_code_credentials("round-trip-tok", "round-trip-ref", expires_at)
        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "round-trip-tok"
        assert creds["refreshToken"] == "round-trip-ref"
        assert creds["expiresAt"] == expires_at

    def test_preserves_existing_fields_in_claude_config_dir(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom-claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_dir))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path / "decoy-home")
        cred_file = custom_dir / ".credentials.json"
        cred_file.parent.mkdir(parents=True)
        cred_file.write_text(json.dumps({"otherField": "keep-me"}), encoding="utf-8")

        _write_claude_code_credentials("new-tok", "new-ref", 99999)

        data = json.loads(cred_file.read_text(encoding="utf-8"))
        assert data["otherField"] == "keep-me"
        assert data["claudeAiOauth"]["accessToken"] == "new-tok"
