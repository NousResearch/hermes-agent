"""Tests for local/private-network provider no-auth support (#41370).

Covers:
  - ``_is_local_base_url()`` helper — RFC 1918, loopback, link-local
  - ``resolve_api_key_provider_credentials`` — local providers get placeholder
  - ``_prompt_api_key`` — setup prompt shows no-auth default for local providers
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ── _is_local_base_url ────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:1234/v1",
    "http://localhost:8080/v1",
    "http://localhost/v1",
    "https://[::1]:443/v1",
    "http://10.0.0.1:11434/v1",
    "http://10.255.255.255/v1",
    "http://172.16.0.1/v1",
    "http://172.31.255.255/v1",
    "http://192.168.1.100:8000/v1",
    "http://192.168.0.1/v1",
    "http://169.254.1.1/v1",
    "http://myserver.local/v1",
    "http://desktop.localhost/v1",
])
def test_is_local_base_url_local_addresses(url):
    from hermes_cli.auth import _is_local_base_url
    assert _is_local_base_url(url) is True


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1",
    "https://openrouter.ai/api/v1",
    "http://8.8.8.8/v1",
    "https://172.32.0.1/v1",       # outside 172.16/12 range
    "http://193.168.1.1/v1",       # public IP
    "https://my-remote-server.com/v1",
    "http://192.169.1.1/v1",       # just outside 192.168/16
    "",
    "not-a-url",
])
def test_is_local_base_url_remote_addresses(url):
    from hermes_cli.auth import _is_local_base_url
    assert _is_local_base_url(url) is False


# ── resolve_api_key_provider_credentials ──────────────────────────────────────

class TestLocalProviderNoAuth:
    """Test that providers with local base_urls get a no-auth placeholder."""

    def test_local_provider_gets_placeholder(self, monkeypatch):
        """A provider in PROVIDER_REGISTRY whose base_url points to localhost
        should get LOCAL_NOAUTH_PLACEHOLDER when no API key is set."""
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            ProviderConfig,
            LOCAL_NOAUTH_PLACEHOLDER,
            resolve_api_key_provider_credentials,
        )

        # Add a test provider with a local base_url.
        test_config = ProviderConfig(
            id="test-local-server",
            name="Test Local Server",
            auth_type="api_key",
            inference_base_url="http://192.168.1.50:11434/v1",
            api_key_env_vars=("TEST_LOCAL_API_KEY",),
        )
        monkeypatch.setitem(PROVIDER_REGISTRY, "test-local-server", test_config)
        monkeypatch.delenv("TEST_LOCAL_API_KEY", raising=False)

        creds = resolve_api_key_provider_credentials("test-local-server")

        assert creds["provider"] == "test-local-server"
        assert creds["api_key"] == LOCAL_NOAUTH_PLACEHOLDER
        assert creds["base_url"] == "http://192.168.1.50:11434/v1"

    def test_local_provider_with_key_uses_real_key(self, monkeypatch):
        """When a real API key IS set for a local provider, use it instead
        of the placeholder."""
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            ProviderConfig,
            resolve_api_key_provider_credentials,
        )

        test_config = ProviderConfig(
            id="test-local-auth",
            name="Test Local Auth",
            auth_type="api_key",
            inference_base_url="http://10.0.0.5:8080/v1",
            api_key_env_vars=("TEST_LOCAL_AUTH_KEY",),
        )
        monkeypatch.setitem(PROVIDER_REGISTRY, "test-local-auth", test_config)
        monkeypatch.setenv("TEST_LOCAL_AUTH_KEY", "my-real-key")

        creds = resolve_api_key_provider_credentials("test-local-auth")

        assert creds["api_key"] == "my-real-key"

    def test_remote_provider_without_key_raises(self, monkeypatch):
        """A remote provider without an API key should NOT get a placeholder."""
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            ProviderConfig,
            resolve_api_key_provider_credentials,
        )

        test_config = ProviderConfig(
            id="test-remote-server",
            name="Test Remote Server",
            auth_type="api_key",
            inference_base_url="https://api.example.com/v1",
            api_key_env_vars=("TEST_REMOTE_KEY",),
        )
        monkeypatch.setitem(PROVIDER_REGISTRY, "test-remote-server", test_config)
        monkeypatch.delenv("TEST_REMOTE_KEY", raising=False)

        # This should return empty api_key (not the placeholder),
        # and the caller (runtime resolver) will handle the error.
        creds = resolve_api_key_provider_credentials("test-remote-server")
        assert creds["api_key"] == ""

    def test_lmstudio_still_uses_dedicated_placeholder(self, monkeypatch):
        """LM Studio keeps its own sentinel value for backward compat."""
        from hermes_cli.auth import (
            LMSTUDIO_NOAUTH_PLACEHOLDER,
            LOCAL_NOAUTH_PLACEHOLDER,
            resolve_api_key_provider_credentials,
        )

        monkeypatch.delenv("LM_API_KEY", raising=False)
        monkeypatch.delenv("LM_BASE_URL", raising=False)

        creds = resolve_api_key_provider_credentials("lmstudio")

        assert creds["api_key"] == LMSTUDIO_NOAUTH_PLACEHOLDER
        # Should NOT be the generic placeholder.
        assert creds["api_key"] != LOCAL_NOAUTH_PLACEHOLDER

    def test_env_base_url_local_triggers_placeholder(self, monkeypatch):
        """Even if the default base_url is remote, a local env override should
        trigger the no-auth placeholder."""
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            ProviderConfig,
            LOCAL_NOAUTH_PLACEHOLDER,
            resolve_api_key_provider_credentials,
        )

        test_config = ProviderConfig(
            id="test-env-local",
            name="Test Env Local",
            auth_type="api_key",
            inference_base_url="https://api.remote.com/v1",
            api_key_env_vars=("TEST_ENV_LOCAL_KEY",),
            base_url_env_var="TEST_ENV_LOCAL_BASE_URL",
        )
        monkeypatch.setitem(PROVIDER_REGISTRY, "test-env-local", test_config)
        monkeypatch.delenv("TEST_ENV_LOCAL_KEY", raising=False)
        monkeypatch.setenv("TEST_ENV_LOCAL_BASE_URL", "http://192.168.0.10:8080/v1")

        creds = resolve_api_key_provider_credentials("test-env-local")

        assert creds["api_key"] == LOCAL_NOAUTH_PLACEHOLDER
        assert creds["base_url"] == "http://192.168.0.10:8080/v1"


# ── _prompt_api_key for local providers ───────────────────────────────────────

@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / ".env").write_text("")
    return home


def _run_prompt(existing_key, choice="", new_key="", provider_id="", pconfig_name="deepseek"):
    """Invoke _prompt_api_key with mocked input()/getpass() responses."""
    from hermes_cli import main as m
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY[pconfig_name]
    with patch("builtins.input", return_value=choice), \
         patch("hermes_cli.secret_prompt.masked_secret_prompt", return_value=new_key):
        return m._prompt_api_key(pconfig, existing_key, provider_id=provider_id)


def test_local_provider_first_time_empty_uses_placeholder(profile_env, monkeypatch):
    """First-time setup for a local provider: pressing Enter should use the
    LOCAL_NOAUTH_PLACEHOLDER instead of cancelling."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        ProviderConfig,
        LOCAL_NOAUTH_PLACEHOLDER,
    )

    test_config = ProviderConfig(
        id="test-prompt-local",
        name="Test Prompt Local",
        auth_type="api_key",
        inference_base_url="http://10.0.0.1:11434/v1",
        api_key_env_vars=("TEST_PROMPT_LOCAL_KEY",),
    )
    monkeypatch.setitem(PROVIDER_REGISTRY, "test-prompt-local", test_config)

    key, abort = _run_prompt(
        existing_key="",
        new_key="",
        provider_id="test-prompt-local",
        pconfig_name="test-prompt-local",
    )
    assert key == LOCAL_NOAUTH_PLACEHOLDER
    assert abort is False


def test_local_provider_replace_empty_keeps_existing(profile_env, monkeypatch):
    """On REPLACE with empty input, do NOT substitute the placeholder."""
    from hermes_cli.auth import PROVIDER_REGISTRY, ProviderConfig
    from hermes_cli.config import save_env_value

    test_config = ProviderConfig(
        id="test-prompt-local2",
        name="Test Prompt Local 2",
        auth_type="api_key",
        inference_base_url="http://192.168.1.1:8080/v1",
        api_key_env_vars=("TEST_PROMPT_LOCAL2_KEY",),
    )
    monkeypatch.setitem(PROVIDER_REGISTRY, "test-prompt-local2", test_config)
    save_env_value("TEST_PROMPT_LOCAL2_KEY", "my-real-key")

    key, abort = _run_prompt(
        existing_key="my-real-key",
        choice="r",
        new_key="",
        provider_id="test-prompt-local2",
        pconfig_name="test-prompt-local2",
    )
    assert key == "my-real-key"
    assert abort is False
