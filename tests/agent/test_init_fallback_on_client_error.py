"""Regression tests for #157305: a primary whose client construction raises at
``AIAgent.__init__`` must walk ``fallback_providers`` instead of failing the turn.

The real-world trigger is a provider plugin whose ``create_client()`` raises
(``ModuleNotFoundError`` after its frozen copy was pruned). The missing-credential
path already reaches the fallback chain at init; a primary that resolves
credentials but cannot build its client did not.

Resolution-chain code, so these run with real imports against the runner's
isolated ``HERMES_HOME``: fallback entries carry inline keys and resolve through
the real router; no network call is made. The one stub is the Anthropic SDK
build (a third-party edge) where a test needs it to fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_yaml as yaml
import providers as _providers
from agent import anthropic_adapter
from agent.credential_pool import CredentialPool
from providers.base import ProviderProfile
from run_agent import AIAgent

_BROKEN = "init-broken-plugin"
_LOCAL = "http://127.0.0.1:9/v1"
_CUSTOM = {"provider": "custom", "model": "local-model", "base_url": _LOCAL, "api_key": "sk-local-test-123456"}
_ANTHROPIC = {"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "sk-ant-test-123456"}


class _BrokenPluginProfile(ProviderProfile):
    def create_client(self, **kwargs):
        raise ModuleNotFoundError("No module named 'directsdk'")


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "DEEPSEEK_API_KEY",
                 "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def broken_primary():
    """Register a process provider whose client construction always raises."""
    _providers._discover_providers()
    snapshot = (dict(_providers._REGISTRY), dict(_providers._ALIASES), _providers._PROVIDER_LIST_CACHE)
    _providers.register_provider(_BrokenPluginProfile(
        name=_BROKEN, auth_type="external_process", base_url=f"process://{_BROKEN}",
    ))
    yield
    _providers._REGISTRY.clear()
    _providers._REGISTRY.update(snapshot[0])
    _providers._ALIASES.clear()
    _providers._ALIASES.update(snapshot[1])
    _providers._PROVIDER_LIST_CACHE = snapshot[2]
    # Never roll the generation back: caches keyed on it must not reuse a stale entry.
    _providers._REGISTRY_GENERATION += 1


def _broken_primary_agent(fallback_model, **extra):
    return AIAgent(provider=_BROKEN, model="claude-opus-5-5", api_key="primary-key-1234567890",
                   base_url=f"process://{_BROKEN}", quiet_mode=True, skip_context_files=True, skip_memory=True,
                   enabled_toolsets=[], fallback_model=fallback_model, **extra)


def _keyless_primary_agent(fallback_model, **extra):
    """A primary with no credentials: init takes the router's missing-credential walk."""
    return AIAgent(provider="deepseek", model="deepseek-chat", quiet_mode=True, skip_context_files=True,
                   skip_memory=True, enabled_toolsets=[], fallback_model=fallback_model, **extra)


def _anthropic_sdk_rejecting(api_key):
    """Patch the Anthropic SDK build to fail for *api_key* only (an unusable native route)."""
    real_build = anthropic_adapter.build_anthropic_client

    def build(key, *args, **kwargs):
        if key == api_key:
            raise ImportError("anthropic SDK unavailable")
        return real_build(key, *args, **kwargs)

    return patch.object(anthropic_adapter, "build_anthropic_client", side_effect=build)


_MALFORMED = {"provider": "custom", "model": "bad-model", "base_url": "http://127.0.0.1:99999/v1",
              "api_key": "sk-bad-test-123456"}  # resolves, then fails ``_validate_base_url`` at client build
_BROKEN_ANTHROPIC = {"provider": "custom", "model": "claude-sonnet-4-5", "base_url": "http://127.0.0.1:9/anthropic",
                     "api_key": "sk-ant-broken-123456"}  # Anthropic wire by host path


@pytest.mark.parametrize("make_agent, primary, chain", [
    pytest.param(_broken_primary_agent, _BROKEN, [_MALFORMED, _CUSTOM], id="primary-client-raises"),
    pytest.param(_keyless_primary_agent, "deepseek", [_BROKEN_ANTHROPIC, _CUSTOM], id="native-fallback-raises"),
])
def test_a_route_whose_client_cannot_be_built_hands_init_to_the_next_fallback(
        broken_primary, make_agent, primary, chain):
    primary_pool = CredentialPool(primary, [])

    with _anthropic_sdk_rejecting(_BROKEN_ANTHROPIC["api_key"]):
        agent = make_agent(chain, credential_pool=primary_pool)

    assert (agent.provider, agent.model, agent._fallback_activated) == ("custom", "local-model", True)
    assert agent.api_mode == "chat_completions"
    assert agent.base_url.rstrip("/") == _LOCAL
    assert agent.api_key == "sk-local-test-123456"
    assert str(agent._client_kwargs["base_url"]).rstrip("/") == _LOCAL
    assert str(agent.client.base_url).rstrip("/") == _LOCAL
    # The primary's pool must not rotate its keys into the fallback's route.
    assert agent._credential_pool is None or agent._credential_pool.provider == "custom"


@pytest.mark.parametrize("make_agent", [
    pytest.param(_broken_primary_agent, id="primary-client-raises"),
    pytest.param(_keyless_primary_agent, id="primary-without-credentials"),
])
def test_an_anthropic_wire_fallback_is_bound_to_a_native_anthropic_client(broken_primary, make_agent):
    agent = make_agent([_ANTHROPIC])

    assert (agent.provider, agent.model, agent.api_mode) == ("anthropic", "claude-sonnet-4-5", "anthropic_messages")
    assert agent._anthropic_client is not None
    assert agent.client is None


@pytest.mark.parametrize("make_agent", [
    pytest.param(_broken_primary_agent, id="primary-client-raises"),
    pytest.param(_keyless_primary_agent, id="primary-without-credentials"),
])
def test_a_named_provider_keeps_its_declared_api_mode_on_a_plain_host(broken_primary, make_agent, monkeypatch):
    """A ``providers.<name>`` block declares its wire once; a plain host carries no Anthropic signal,
    so init must take the declared ``api_mode``, as the per-turn binder does (#33062, #81932)."""
    monkeypatch.setenv("MYRELAY_API_KEY", "sk-relay-test-123456")
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(yaml.safe_dump({"providers": {"myrelay": {
        "name": "myrelay", "base_url": "http://127.0.0.1:9", "key_env": "MYRELAY_API_KEY",
        "api_mode": "anthropic_messages", "default_model": "claude-sonnet-4-5",
    }}}))

    agent = make_agent([{"provider": "myrelay", "model": "claude-sonnet-4-5"}])

    assert (agent.provider, agent.api_mode) == ("myrelay", "anthropic_messages")
    assert agent._anthropic_client is not None
    assert agent._anthropic_base_url.rstrip("/") == "http://127.0.0.1:9"
    assert agent.client is None


@pytest.mark.parametrize("make_agent", [
    pytest.param(_broken_primary_agent, id="primary-client-raises"),
    pytest.param(_keyless_primary_agent, id="primary-without-credentials"),
])
def test_an_anthropic_suffixed_fallback_keeps_its_messages_base_path(broken_primary, make_agent):
    """The resolver rewrites a dual-surface ``/anthropic`` base to ``/v1`` unless it is told the wire;
    a Messages client on the rewritten base would post to ``/v1/messages``."""
    agent = make_agent([{"provider": "custom", "model": "claude-sonnet-4-5",
                         "base_url": "https://api.minimax.io/anthropic", "api_key": "sk-ant-relay-123456"}])

    assert agent.api_mode == "anthropic_messages"
    assert agent._anthropic_base_url.rstrip("/") == "https://api.minimax.io/anthropic"


@pytest.mark.parametrize("make_agent", [
    pytest.param(_broken_primary_agent, id="primary-client-raises"),
    pytest.param(_keyless_primary_agent, id="primary-without-credentials"),
])
def test_an_actual_fallback_binds_chat_completions_whatever_its_declared_wire(broken_primary, make_agent):
    """Actual serves chat completions only; the per-turn binder overrides every hint for it."""
    agent = make_agent([{"provider": "custom", "model": "actual-model", "base_url": "https://api.actual.inc/v1",
                         "api_key": "sk-actual-test-123456", "api_mode": "anthropic_messages"}])

    assert agent.api_mode == "chat_completions"
    assert agent.client is not None
    assert agent._anthropic_client is None


@pytest.mark.parametrize("make_agent", [
    pytest.param(_broken_primary_agent, id="primary-client-raises"),
    pytest.param(_keyless_primary_agent, id="primary-without-credentials"),
])
def test_a_named_provider_without_a_declared_wire_is_detected_from_its_anthropic_base(broken_primary, make_agent,
                                                                                     monkeypatch):
    """No ``api_mode`` declared: the resolver must still see the named block's ``/anthropic`` base
    and build a Messages client on it, as init did before it consulted entry hints."""
    monkeypatch.setenv("MMRELAY_API_KEY", "sk-mm-test-123456")
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(yaml.safe_dump({"providers": {"mmrelay": {
        "name": "mmrelay", "base_url": "https://api.minimax.io/anthropic", "key_env": "MMRELAY_API_KEY",
        "default_model": "MiniMax-M2",
    }}}))

    agent = make_agent([{"provider": "mmrelay", "model": "MiniMax-M2"}])

    assert (agent.provider, agent.api_mode) == ("mmrelay", "anthropic_messages")
    assert agent._anthropic_base_url.rstrip("/") == "https://api.minimax.io/anthropic"
