"""Regression tests for Anthropic launch-path credential resolution order.

The anthropic registry tuple (``PROVIDER_REGISTRY["anthropic"].api_key_env_vars``)
lists ANTHROPIC_API_KEY first because it also backs "is configured" checks.
The runtime credential pick, however, must follow the adapter's documented
OAuth-first order (``agent.anthropic_adapter.resolve_anthropic_token``):

    ANTHROPIC_TOKEN → CLAUDE_CODE_OAUTH_TOKEN → Claude Code credential
    store (with refresh) → ANTHROPIC_API_KEY

Without that, a stale/revoked ANTHROPIC_API_KEY exported in the shell
environment shadows a valid OAuth token and every launch fails with HTTP 401.
"""

import pytest

from hermes_cli.auth import resolve_api_key_provider_credentials

FAKE_OAUTH_TOKEN = "sk-ant-oat01-" + "g" * 60
FAKE_STALE_API_KEY = "sk-ant-api03-" + "d" * 60


@pytest.fixture(autouse=True)
def _hermetic_anthropic_sources(monkeypatch):
    """Keep tests hermetic on developer machines (especially macOS).

    ``resolve_anthropic_token`` reads the macOS Keychain and
    ``~/.claude/.credentials.json`` — both live outside the per-test
    HERMES_HOME isolation, so a developer's real Claude Code login would
    leak into assertions.  Stub both readers to "nothing found".
    """
    monkeypatch.setattr(
        "agent.anthropic_adapter._read_claude_code_credentials_from_keychain",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials",
        lambda: None,
    )
    # Make sure no var from the host shell leaks in (conftest strips
    # *_API_KEY/*_TOKEN already; explicit here for self-documentation).
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)


class TestResolveApiKeyProviderCredentialsAnthropic:
    def test_oauth_token_beats_stale_api_key(self, monkeypatch):
        """A valid CLAUDE_CODE_OAUTH_TOKEN must win over ANTHROPIC_API_KEY.

        This is the launch-path regression: the registry tuple is
        API-key-first, so the generic env-var iteration would return the
        stale key and 401 every launch.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_STALE_API_KEY)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH_TOKEN)

        creds = resolve_api_key_provider_credentials("anthropic")

        assert creds["api_key"] == FAKE_OAUTH_TOKEN
        assert creds["api_key"] != FAKE_STALE_API_KEY
        assert creds["source"] == "CLAUDE_CODE_OAUTH_TOKEN"

    def test_anthropic_token_beats_stale_api_key(self, monkeypatch):
        """ANTHROPIC_TOKEN (Hermes-persisted OAuth) also outranks the API key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_STALE_API_KEY)
        monkeypatch.setenv("ANTHROPIC_TOKEN", FAKE_OAUTH_TOKEN)

        creds = resolve_api_key_provider_credentials("anthropic")

        assert creds["api_key"] == FAKE_OAUTH_TOKEN
        assert creds["source"] == "ANTHROPIC_TOKEN"

    def test_api_key_still_used_when_only_api_key_present(self, monkeypatch):
        """The API key remains a valid last-resort credential."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_STALE_API_KEY)

        creds = resolve_api_key_provider_credentials("anthropic")

        assert creds["api_key"] == FAKE_STALE_API_KEY
        assert creds["source"] == "ANTHROPIC_API_KEY"


class TestRuntimeProviderLaunchPath:
    def test_launch_path_resolves_oauth_first(self, monkeypatch):
        """End-to-end launch resolution (hermes -z / chat) returns the OAuth
        token, not the stale API key, when both are present."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_STALE_API_KEY)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH_TOKEN)

        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="anthropic")

        assert runtime["provider"] == "anthropic"
        assert runtime["api_key"] == FAKE_OAUTH_TOKEN
        assert runtime["api_key"] != FAKE_STALE_API_KEY
