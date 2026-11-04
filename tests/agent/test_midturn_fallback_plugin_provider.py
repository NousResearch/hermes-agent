"""Mid-turn fallback reaches a plugin-registered ``external_process`` model provider.

A ``fallback_providers`` entry naming a model-provider plugin (e.g.
``claude-subscription-directsdk-experimental``) must resolve mid-turn through
``try_activate_fallback`` -> ``resolve_provider_client`` exactly as it does at startup, instead of
logging "Fallback to <plugin> failed: provider not configured" and skipping the entry.

The fake profile below is constructed with the same ``ProviderProfile`` fields the Claude DirectSDK
plugin passes: when core rejected one of them (``native_reasoning_details_type``) the plugin failed
to import, was never registered, and the mid-turn path lost it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PLUGIN = "acme-plugin-directsdk"


class _PluginClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self._custom_headers = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None))


@pytest.fixture
def plugin_provider(monkeypatch, tmp_path):
    """Register an out-of-tree external_process profile the way a user model-provider plugin does."""
    import providers
    from hermes_cli import auth
    from providers.base import ProviderProfile

    fake_cli = tmp_path / "fake-claude"
    fake_cli.write_text("#!/bin/sh\nexit 0\n")
    fake_cli.chmod(0o755)

    class _Profile(ProviderProfile):
        def create_client(self, **client_kwargs):
            return _PluginClient(**client_kwargs)

    # Same constructor surface as claude-subscription-directsdk-experimental/__init__.py.
    profile = _Profile(
        name=PLUGIN,
        display_name="Acme DirectSDK",
        description="Acme DirectSDK (test double for a model-provider plugin)",
        api_mode="chat_completions",
        auth_type="external_process",
        supports_health_check=False,
        native_reasoning_details_type=f"{PLUGIN}.native_assistant",
        env_vars=(),
        base_url=f"process://{PLUGIN}",
        process_command=str(fake_cli),
        process_args=(),
        process_command_env_vars=("ACME_DIRECTSDK_COMMAND",),
        default_aux_model="acme-small",
        fallback_models=("acme-opus", "acme-small"),
        model_aliases={"opus": "acme-opus"},
    )
    monkeypatch.delenv("ACME_DIRECTSDK_COMMAND", raising=False)
    providers.register_provider(profile)
    # Post-discovery registration: mirror into PROVIDER_REGISTRY the way plugin discovery does.
    auth.sync_plugin_provider_registry()
    try:
        yield profile
    finally:
        auth.PROVIDER_REGISTRY.pop(PLUGIN, None)
        from hermes_cli.auth_plugin_providers import PLUGIN_MIRRORED_PROVIDERS
        PLUGIN_MIRRORED_PROVIDERS.discard(PLUGIN)
        providers._REGISTRY.pop(PLUGIN, None)
        providers._SOURCES.pop(PLUGIN, None)
        providers._PROVIDER_LIST_CACHE = None


def _agent_on_failed_primary():
    """An agent whose turn started on kimi-coding with the plugin first in the fallback chain."""
    agent = MagicMock()
    agent.provider = "kimi-coding"
    agent.model = "k3"
    agent.base_url = "https://api.kimi.com/coding/v1"
    agent.api_mode = "anthropic_messages"
    agent.api_key = "primary-key"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = [{"provider": PLUGIN, "model": "acme-opus"}]
    agent._unavailable_fallback_keys = set()
    agent._primary_runtime = {"provider": "kimi-coding", "model": "k3", "base_url": agent.base_url,
                              "api_mode": "anthropic_messages", "api_key": "primary-key"}
    agent._config_context_length = None
    agent._credential_pool = None
    agent._rate_limited_until = 0
    agent._transport_cache = {}
    agent._client_kwargs = {"api_key": "primary-key", "base_url": agent.base_url}
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    agent.context_compressor = None
    return agent


def test_resolve_provider_client_builds_plugin_external_process_client(plugin_provider):
    from agent.auxiliary_client import resolve_provider_client

    client, model = resolve_provider_client(PLUGIN, model="acme-opus", raw_codex=True)

    assert isinstance(client, _PluginClient)
    assert model == "acme-opus"
    assert client.kwargs["base_url"] == f"process://{PLUGIN}"
    assert client.kwargs["command"] == plugin_provider.process_command


def test_midturn_fallback_activates_plugin_external_process_provider(plugin_provider, caplog):
    from agent.chat_completion_helpers import try_activate_fallback

    agent = _agent_on_failed_primary()
    with patch("agent.credential_pool.load_pool", return_value=None):
        activated = try_activate_fallback(agent)

    assert "provider not configured" not in caplog.text
    assert activated is True
    assert agent.provider == PLUGIN
    assert agent.model == "acme-opus"
    assert agent.base_url == f"process://{PLUGIN}"
    assert agent._unavailable_fallback_keys == set()
