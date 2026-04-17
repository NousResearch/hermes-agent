"""
Integration test: kimi-coding with bad API key must NOT leak into Claude Code credentials.

Scenario (from PR #7009 / #7156):
  1. User runs `hermes setup`, selects kimi-coding, enters a malformed API key.
  2. User runs `hermes chat` — main agent fails on kimi.
  3. Auxiliary client fallback chain reaches _try_anthropic() which reads
     ~/.claude/.credentials.json and silently uses the Claude Max OAuth token.
  4. Token expires → Hermes rewrites ~/.claude/.credentials.json.
  5. Next day: Claude Code finds modified credentials → "not logged in" / wrong quota.

This file verifies the fix closes all three vulnerable paths:
  A. _seed_from_singletons("anthropic")  — credential pool seeding
  B. _resolve_api_key_provider()         — auxiliary client fallback chain
  C. auth_remove_command()               — suppression flag on explicit remove
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kimi_env(tmp_path, monkeypatch):
    """
    Isolated environment:
      - HERMES_HOME points to tmp dir with kimi-coding as model.provider
      - active_provider = None  (user never explicitly added anthropic)
      - ~/.claude/.credentials.json exists with a fake Claude Max OAuth token
      - KIMI_API_KEY set to a syntactically wrong key
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # config.yaml: kimi-coding selected, NOT anthropic
    (hermes_home / "config.yaml").write_text(yaml.dump({
        "model": {
            "provider": "kimi-coding",
            "default": "kimi-k2",
        }
    }))

    # auth.json: no active_provider, no anthropic entry
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "active_provider": None,
    }))

    # Fake Claude Code credentials (the file the vulnerability exploits)
    claude_dir = tmp_path / "dot_claude"
    claude_dir.mkdir()
    fake_creds = {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-FAKE-CLAUDE-MAX-TOKEN",
            "refreshToken": "sk-ant-rt01-FAKE-REFRESH",
            "expiresAt": 9999999999999,
        }
    }
    claude_creds_file = claude_dir / ".credentials.json"
    claude_creds_file.write_text(json.dumps(fake_creds))

    # Bad kimi key (wrong format — missing prefix)
    monkeypatch.setenv("KIMI_API_KEY", "BADKEY-NOT-A-REAL-KIMI-KEY")

    # Point read_claude_code_credentials at our fake file
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))

    # Remove any real anthropic env vars so they don't count as explicit config
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    return {
        "hermes_home": hermes_home,
        "claude_creds_file": claude_creds_file,
        "fake_token": "sk-ant-oat01-FAKE-CLAUDE-MAX-TOKEN",
    }


# ---------------------------------------------------------------------------
# Part A: _seed_from_singletons must be blocked
# ---------------------------------------------------------------------------

class TestCredentialPoolGate:
    """_seed_from_singletons("anthropic") must not run when kimi-coding is configured."""

    def test_gate_blocks_seeding_when_kimi_configured(self, kimi_env, monkeypatch):
        """
        With kimi-coding as provider, calling _seed_from_singletons("anthropic")
        must return (False, set()) — no entries seeded, no Claude file read.
        """
        from agent.credential_pool import _seed_from_singletons

        read_called = []

        def fake_read_claude_code_credentials():
            read_called.append(True)
            return {
                "accessToken": kimi_env["fake_token"],
                "refreshToken": "rt",
                "expiresAt": 9999999999999,
            }

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            fake_read_claude_code_credentials,
        )

        entries = []
        changed, active_sources = _seed_from_singletons("anthropic", entries)

        assert not changed, "Should not have seeded any credentials"
        assert active_sources == set(), "Should not have any active sources"
        assert not read_called, (
            "read_claude_code_credentials() was called — "
            "the gate failed to block it!"
        )

    def test_gate_allows_seeding_when_anthropic_configured(self, tmp_path, monkeypatch):
        """Sanity check: seeding DOES work when anthropic is the explicit provider."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        (hermes_home / "config.yaml").write_text(yaml.dump({
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"}
        }))
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {}, "active_provider": None,
        }))

        from agent.credential_pool import _seed_from_singletons

        def fake_read_claude_code_credentials():
            return {
                "accessToken": "sk-ant-oat01-REAL-TOKEN",
                "refreshToken": "rt",
                "expiresAt": 9999999999999,
            }

        def fake_read_hermes_oauth_credentials():
            return None

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            fake_read_claude_code_credentials,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_hermes_oauth_credentials",
            fake_read_hermes_oauth_credentials,
        )

        entries = []
        changed, active_sources = _seed_from_singletons("anthropic", entries)

        assert changed, "Should have seeded credentials when anthropic is configured"
        assert "claude_code" in active_sources


# ---------------------------------------------------------------------------
# Part B: _resolve_api_key_provider must skip anthropic
# ---------------------------------------------------------------------------

class TestAuxiliaryClientGate:
    """_resolve_api_key_provider() must skip anthropic when kimi-coding is configured."""

    def test_anthropic_skipped_in_fallback_chain(self, kimi_env, monkeypatch):
        """
        When main agent fails on kimi-coding and auxiliary fallback runs,
        anthropic must be skipped — _try_anthropic() must never be called.

        Note: kimi-coding itself MAY return a client (via its env key), which
        is fine — the vulnerability is using *Claude* credentials, not kimi's.
        """
        try_anthropic_called = []

        def fake_try_anthropic():
            try_anthropic_called.append(True)
            return MagicMock(), "claude-sonnet-4-6"

        monkeypatch.setattr(
            "agent.auxiliary_client._try_anthropic",
            fake_try_anthropic,
        )

        from agent.auxiliary_client import _resolve_api_key_provider
        client, model = _resolve_api_key_provider()

        # PRIMARY assertion: _try_anthropic() must NOT be called
        assert not try_anthropic_called, (
            "_try_anthropic() was called — "
            "the auxiliary client gate failed! Claude credentials were used."
        )
        # If a client was returned, it must not be a Claude model
        if model is not None:
            assert "claude" not in model.lower(), (
                f"Returned model '{model}' looks like a Claude model — "
                "anthropic credentials leaked through!"
            )

    def test_anthropic_used_when_explicitly_configured(self, tmp_path, monkeypatch):
        """Sanity: anthropic IS tried when explicitly configured."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        (hermes_home / "config.yaml").write_text(yaml.dump({
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"}
        }))
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {}, "active_provider": None,
        }))

        try_anthropic_called = []
        fake_client = MagicMock()

        def fake_try_anthropic():
            try_anthropic_called.append(True)
            return fake_client, "claude-sonnet-4-6"

        monkeypatch.setattr(
            "agent.auxiliary_client._select_pool_entry",
            lambda provider_id: (False, None),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._try_anthropic",
            fake_try_anthropic,
        )

        from agent.auxiliary_client import _resolve_api_key_provider
        client, model = _resolve_api_key_provider()

        assert try_anthropic_called, (
            "_try_anthropic() was NOT called when anthropic is explicit provider"
        )
        assert client is fake_client


# ---------------------------------------------------------------------------
# Part C: auth_remove_command writes suppression flag
# ---------------------------------------------------------------------------

class TestAuthRemoveSuppression:
    """auth_remove_command() for claude_code must write suppression flag."""

    def test_remove_claude_code_writes_suppression_flag(self, kimi_env, monkeypatch):
        """
        After `hermes auth remove anthropic <claude_code_label>`,
        the claude_code source must be in suppressed_sources so it
        won't be re-seeded even if Hermes is restarted.
        """
        hermes_home = kimi_env["hermes_home"]

        # Add a fake claude_code credential pool entry to auth.json
        auth_data = json.loads((hermes_home / "auth.json").read_text())
        auth_data["providers"]["anthropic"] = {
            "pool": [
                {
                    "id": "cc-001",
                    "source": "claude_code",
                    "auth_type": "oauth",
                    "access_token": kimi_env["fake_token"],
                    "label": "claude_code",
                    "priority": 1,
                    "exhausted": False,
                }
            ]
        }
        (hermes_home / "auth.json").write_text(json.dumps(auth_data))

        from hermes_cli.auth import is_source_suppressed

        # Before removal: not suppressed
        assert not is_source_suppressed("anthropic", "claude_code"), \
            "Should not be suppressed before removal"

        # Simulate: auth remove anthropic claude_code
        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("anthropic", "claude_code")

        # After suppression: blocked
        assert is_source_suppressed("anthropic", "claude_code"), \
            "Should be suppressed after auth remove"

    def test_suppressed_source_not_reseeded(self, kimi_env, monkeypatch):
        """
        After suppression, even if anthropic IS explicitly configured,
        the claude_code source must not be re-seeded.
        """
        hermes_home = kimi_env["hermes_home"]

        # Configure anthropic explicitly (so the provider gate passes)
        (hermes_home / "config.yaml").write_text(yaml.dump({
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"}
        }))

        # Suppress claude_code source
        from hermes_cli.auth import suppress_credential_source
        suppress_credential_source("anthropic", "claude_code")

        read_called = []

        def fake_read_claude_code_credentials():
            read_called.append(True)
            return {
                "accessToken": kimi_env["fake_token"],
                "refreshToken": "rt",
                "expiresAt": 9999999999999,
            }

        def fake_read_hermes_oauth_credentials():
            return None

        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            fake_read_claude_code_credentials,
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_hermes_oauth_credentials",
            fake_read_hermes_oauth_credentials,
        )

        from agent.credential_pool import _seed_from_singletons
        entries = []
        changed, active_sources = _seed_from_singletons("anthropic", entries)

        assert not changed, "Suppressed source should not be re-seeded"
        assert "claude_code" not in active_sources, \
            "claude_code should not appear in active_sources after suppression"


# ---------------------------------------------------------------------------
# Part D: is_provider_explicitly_configured edge cases
# ---------------------------------------------------------------------------

class TestProviderGateEdgeCases:
    """Verify the gate correctly handles implicit vs explicit anthropic configuration."""

    def test_claude_code_oauth_env_var_is_not_explicit(self, kimi_env, monkeypatch):
        """CLAUDE_CODE_OAUTH_TOKEN (set by Claude Code itself) must not count as explicit."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-auto-injected")

        from hermes_cli.auth import is_provider_explicitly_configured
        assert not is_provider_explicitly_configured("anthropic"), (
            "CLAUDE_CODE_OAUTH_TOKEN should not satisfy the explicit-provider gate"
        )

    def test_anthropic_api_key_counts_as_explicit(self, kimi_env, monkeypatch):
        """A real ANTHROPIC_API_KEY set by the user IS explicit."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-realuserkey")

        from hermes_cli.auth import is_provider_explicitly_configured
        assert is_provider_explicitly_configured("anthropic"), (
            "A real ANTHROPIC_API_KEY should be treated as explicit configuration"
        )

    def test_kimi_provider_config_does_not_enable_anthropic(self, kimi_env, monkeypatch):
        """Having kimi-coding in config.yaml must NOT enable anthropic auto-discovery."""
        from hermes_cli.auth import is_provider_explicitly_configured
        result = is_provider_explicitly_configured("anthropic")
        assert not result, (
            "kimi-coding config should not satisfy anthropic explicit-provider check"
        )
