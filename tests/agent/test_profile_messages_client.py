"""``ProviderProfile.create_messages_client()`` — the Messages-wire twin of ``create_client()``.

On the OpenAI wire a profile can supply its own client (``create_client``, main agent and
auxiliary routes). The Anthropic Messages wire builds ``anthropic.Anthropic`` in
``build_anthropic_client`` without asking the profile, so a provider plugin could not ship its
own Messages transport. These tests pin the seam: the builder consults the profile when the
caller names the provider, falls through for ``None``, a broken plugin, or an anonymous call,
and the agent's builders pass the provider they are building for.
"""

from __future__ import annotations

import types

import pytest

import providers as _providers
from providers.base import ProviderProfile

pytest.importorskip("anthropic")


class _FakeMessagesClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NativeProfile(ProviderProfile):
    def create_messages_client(self, **kwargs):
        return _FakeMessagesClient(**kwargs)


class _PassThroughProfile(ProviderProfile):
    def create_messages_client(self, **kwargs):
        return None


class _ExplodingProfile(ProviderProfile):
    def create_messages_client(self, **kwargs):
        raise RuntimeError("plugin is broken")


@pytest.fixture
def register(monkeypatch):
    """Register profiles for one test on copies of every registry ``register_provider`` writes."""
    import hermes_cli.auth as _auth

    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_SOURCES", dict(_providers._SOURCES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(_auth, "PROVIDER_REGISTRY", dict(_auth.PROVIDER_REGISTRY))

    def _register(cls, name):
        _providers.register_provider(cls(name=name, api_mode="anthropic_messages",
                                         base_url=f"https://{name}.invalid/anthropic"))

    return _register


def test_default_profile_answers_none():
    assert ProviderProfile(name="plain").create_messages_client(api_key="k") is None


def test_builder_returns_the_profile_client(register):
    from agent.anthropic_adapter import build_anthropic_client

    register(_NativeProfile, "msgs-native")
    client = build_anthropic_client("sk-test", "https://msgs-native.invalid/anthropic", timeout=12.0,
                                    provider="msgs-native")
    assert isinstance(client, _FakeMessagesClient)
    assert client.kwargs == {"api_key": "sk-test", "base_url": "https://msgs-native.invalid/anthropic",
                             "timeout": 12.0, "drop_context_1m_beta": False}


def test_builder_falls_through_when_the_profile_declines(register):
    import anthropic
    from agent.anthropic_adapter import build_anthropic_client

    register(_PassThroughProfile, "msgs-plain")
    client = build_anthropic_client("sk-test", "https://msgs-plain.invalid/anthropic", provider="msgs-plain")
    assert isinstance(client, anthropic.Anthropic)


def test_a_broken_profile_is_logged_and_skipped(register, caplog):
    import anthropic
    from agent.anthropic_adapter import build_anthropic_client

    register(_ExplodingProfile, "msgs-broken")
    client = build_anthropic_client("sk-test", "https://msgs-broken.invalid/anthropic", provider="msgs-broken")
    assert isinstance(client, anthropic.Anthropic)
    assert any("failed to create a Messages client" in r.getMessage() for r in caplog.records)


def test_an_anonymous_build_never_asks_a_profile(register, monkeypatch):
    import agent.anthropic_adapter as adapter

    asked = []
    monkeypatch.setattr(adapter, "_profile_messages_client", lambda *a, **k: asked.append(a) or None)
    adapter.build_anthropic_client("sk-test", "https://api.anthropic.com")
    assert asked == []


def test_unknown_provider_builds_the_standard_client():
    import anthropic
    from agent.anthropic_adapter import build_anthropic_client

    client = build_anthropic_client("sk-test", "https://api.anthropic.com", provider="no-such-provider-xyz")
    assert isinstance(client, anthropic.Anthropic)


def test_auxiliary_wrap_uses_the_profile_client(register):
    from agent.auxiliary_client import AnthropicAuxiliaryClient, _maybe_wrap_anthropic

    register(_NativeProfile, "msgs-aux")
    plain = types.SimpleNamespace(api_key="sk-test", base_url="https://msgs-aux.invalid/anthropic")
    wrapped = _maybe_wrap_anthropic(plain, "m", "sk-test", "https://msgs-aux.invalid/anthropic",
                                    "anthropic_messages", provider="msgs-aux")
    assert isinstance(wrapped, AnthropicAuxiliaryClient)
    assert isinstance(wrapped._real_client, _FakeMessagesClient)


def test_request_local_clients_name_the_agent_provider(monkeypatch):
    import agent.anthropic_adapter as adapter
    from agent.client_lifecycle import ClientLifecycleMixin

    seen = []
    monkeypatch.setattr(adapter, "build_anthropic_client",
                        lambda *a, **k: seen.append(k.get("provider")) or object())
    host = types.SimpleNamespace(provider="msgs-agent", model="m")
    ClientLifecycleMixin._build_anthropic_client_for_key(host, ("direct", "sk", "https://x.invalid", 30.0, False))
    ClientLifecycleMixin._build_direct_anthropic_client(host, "sk", "https://x.invalid")
    assert seen == ["msgs-agent", "msgs-agent"]
