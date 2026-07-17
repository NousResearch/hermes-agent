"""Tests for the model wizard's credential-pool fallback.

Covers the fix for NousResearch/hermes-agent#65977: `hermes model` only
checked .env / os.environ when deciding if a provider was already
configured, so a working key in `auth.json`'s credential pool (added via
`hermes auth`, e.g. Moonshot / DeepSeek) was invisible to the wizard and
it prompted for a new key.

The fix adds `_resolve_existing_api_key` in hermes_cli.model_setup_flows,
which checks .env → os.environ → credential pool, in that order. The
wizard flows (`_model_flow_kimi`, `_model_flow_stepfun`,
`_model_flow_api_key_provider`, `_model_flow_openrouter`) now use this
helper instead of hand-rolling env-only checks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_pconfig(provider_id="deepseek", env_vars=None):
    """Minimal ProviderConfig for testing (registry-free where possible)."""
    from hermes_cli.auth import ProviderConfig

    return ProviderConfig(
        id=provider_id,
        name=provider_id.title(),
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars or [f"{provider_id.upper()}_API_KEY"]),
    )


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Isolated ~/.hermes with cleared env vars and pool, get_env_value cache
    invalidated."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "ZAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "KIMI_API_KEY",
        "KIMI_CODING_API_KEY",
        "STEPFUN_API_KEY",
        "ANTHROPIC_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)

    return home


class TestResolveExistingApiKeyCredentialPoolFallback:
    """`_resolve_existing_api_key` must surface a credential-pool entry when
    no .env / os.environ value is present."""

    def test_pool_key_returned_when_env_empty(self, isolated_hermes_home):
        """When .env and os.environ are both empty but auth.json's pool has a
        key for the provider, the helper returns it (issue #65977)."""
        from hermes_cli.model_setup_flows import _resolve_existing_api_key

        pconfig = _make_pconfig("deepseek", ["DEEPSEEK_API_KEY"])

        with patch(
            "hermes_cli.auth._resolve_api_key_provider_secret",
            return_value=("sk-pool-deepseek-xyz", "credential_pool:deepseek"),
        ):
            result = _resolve_existing_api_key("deepseek", pconfig)

        assert result == "sk-pool-deepseek-xyz"

    def test_env_value_takes_priority_over_pool(self, isolated_hermes_home, monkeypatch):
        """A key in os.environ must win over a credential-pool entry so a
        deliberate rotation via shell export isn't shadowed."""
        from hermes_cli.model_setup_flows import _resolve_existing_api_key

        pconfig = _make_pconfig("deepseek", ["DEEPSEEK_API_KEY"])
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-takes-priority")

        with patch(
            "hermes_cli.auth._resolve_api_key_provider_secret",
            return_value=("sk-pool-should-not-win", "credential_pool:deepseek"),
        ) as mock_pool_resolver:
            result = _resolve_existing_api_key("deepseek", pconfig)

        assert result == "sk-env-takes-priority"
        # Pool resolver must not even fire when env already has a value.
        mock_pool_resolver.assert_not_called()

    def test_returns_empty_when_no_source_has_key(self, isolated_hermes_home):
        """When env, os.environ, and pool are all empty, returns ''. The
        wizard will then fall through to the new-key prompt."""
        from hermes_cli.model_setup_flows import _resolve_existing_api_key

        pconfig = _make_pconfig("deepseek", ["DEEPSEEK_API_KEY"])

        with patch(
            "hermes_cli.auth._resolve_api_key_provider_secret",
            return_value=("", ""),
        ):
            result = _resolve_existing_api_key("deepseek", pconfig)

        assert result == ""

    def test_pool_resolver_exception_does_not_break_wizard(
        self, isolated_hermes_home
    ):
        """If the credential pool lookup raises, the helper must return ''
        (so the wizard prompts for a new key) rather than propagating."""
        from hermes_cli.model_setup_flows import _resolve_existing_api_key

        pconfig = _make_pconfig("deepseek", ["DEEPSEEK_API_KEY"])

        with patch(
            "hermes_cli.auth._resolve_api_key_provider_secret",
            side_effect=RuntimeError("pool read failed"),
        ):
            result = _resolve_existing_api_key("deepseek", pconfig)

        assert result == ""

    def test_existing_key_reaches_prompt_api_key(self, isolated_hermes_home, monkeypatch):
        """Integration: _model_flow_api_key_provider passes the pool key
        through to _prompt_api_key so the wizard's K/R/C offer fires instead
        of the 'No API key configured' first-time flow."""
        from hermes_cli.model_setup_flows import _model_flow_api_key_provider

        pconfig = _make_pconfig("deepseek", ["DEEPSEEK_API_KEY"])
        monkeypatch.setattr(
            "hermes_cli.auth.PROVIDER_REGISTRY",
            {"deepseek": pconfig},
        )

        captured = {}

        def fake_prompt_api_key(pconfig, existing_key, provider_id=""):
            captured["existing_key"] = existing_key
            captured["provider_id"] = provider_id
            return existing_key, True  # abort

        monkeypatch.setattr("hermes_cli.main._prompt_api_key", fake_prompt_api_key)

        with patch.object(
            __import__("hermes_cli.model_setup_flows", fromlist=["_resolve_existing_api_key"]),
            "_resolve_existing_api_key",
            return_value="sk-pool-deepseek-abc",
        ):
            config = {"model": {}}
            _model_flow_api_key_provider(config, "deepseek")

        # Must see the pool key — not empty / blank.
        assert captured["existing_key"] == "sk-pool-deepseek-abc"
        assert captured["provider_id"] == "deepseek"
