"""A provider profile can supply its own client — the registration seam.

A provider whose wire protocol is not OpenAI-over-HTTP (the ACP subprocess
shims) supplies its client via ``ProviderProfile.create_client()``, from inside
or outside the tree. These tests pin the hook through the real entry point
(``create_openai_client``), its failure isolation, and the capability flags
that let such a client opt out of the auxiliary transport/async wrappers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import providers as _providers
from providers.base import ProviderProfile


class _FakeClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True
    api_key = "k"
    base_url = "acp://seam-test"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SeamProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return _FakeClient(**kwargs)


class _ExplodingProfile(ProviderProfile):
    def create_client(self, **kwargs):
        raise RuntimeError("plugin is broken")


@pytest.fixture
def registered():
    """Register profiles for one test and restore the registry afterwards."""
    _providers._discover_providers()
    snapshot = (dict(_providers._REGISTRY), dict(_providers._ALIASES), _providers._PROVIDER_LIST_CACHE)
    yield _providers.register_provider
    _providers._REGISTRY.clear()
    _providers._REGISTRY.update(snapshot[0])
    _providers._ALIASES.clear()
    _providers._ALIASES.update(snapshot[1])
    _providers._PROVIDER_LIST_CACHE = snapshot[2]


def _agent(provider: str = ""):
    return SimpleNamespace(
        provider=provider,
        _client_log_context=lambda: "",
        _build_keepalive_http_client=lambda *a, **k: None,
    )


def _build(provider, base_url):
    from agent.agent_runtime_helpers import create_openai_client

    return create_openai_client(_agent(provider), {"api_key": "k", "base_url": base_url}, reason="t", shared=False)


def test_an_out_of_tree_profile_supplies_the_client_through_create_openai_client(registered):
    from openai import OpenAI

    registered(_SeamProfile(name="seam-test", aliases=("seam",), base_url="acp://seam-test"))
    # By name, by alias, and by base_url prefix when no provider name is set.
    assert isinstance(_build("seam-test", "acp://seam-test"), _FakeClient)
    assert isinstance(_build("seam", "acp://seam-test"), _FakeClient)
    assert isinstance(_build("", "acp://seam-test/x"), _FakeClient)
    # No hook → the standard client, untouched by the seam.
    assert isinstance(_build("openai-api", "https://api.example/v1"), OpenAI)


def test_a_broken_plugin_falls_through_instead_of_taking_the_turn_down(registered):
    from agent.agent_runtime_helpers import _provider_supplied_client

    registered(_ExplodingProfile(name="seam-boom", base_url="acp://seam-boom"))
    assert _provider_supplied_client(_agent("seam-boom"), {"api_key": "k"}) is None


def test_copilot_acp_still_gets_its_acp_client_via_its_profile():
    from agent.copilot_acp_client import CopilotACPClient

    assert isinstance(_build("copilot-acp", "acp://copilot"), CopilotACPClient)
    assert isinstance(_build("", "acp://copilot"), CopilotACPClient)


def test_skip_flags_replace_the_isinstance_checks_for_in_and_out_of_tree_clients():
    from agent.auxiliary_client import _maybe_wrap_anthropic, _to_async_client
    from agent.copilot_acp_client import CopilotACPClient
    from agent.gemini_native_adapter import GeminiNativeClient

    assert CopilotACPClient.HERMES_SKIP_TRANSPORT_WRAP and CopilotACPClient.HERMES_SKIP_ASYNC_WRAP
    assert GeminiNativeClient.HERMES_SKIP_TRANSPORT_WRAP
    assert not getattr(GeminiNativeClient, "HERMES_SKIP_ASYNC_WRAP", False)

    client = _FakeClient()  # never imported by auxiliary_client
    assert _maybe_wrap_anthropic(client, "m", "k", "acp://seam-test") is client
    assert _to_async_client(client, "m")[0] is client


def test_gemini_provider_aliases_route_to_the_native_client():
    """A `google` fallback entry must get GeminiNativeClient, not the generic
    OpenAI SDK client (#104583).

    try_activate_fallback installs the RAW entry provider (``google`` and its
    sibling aliases) on ``agent.provider``; every later client rebuild re-enters
    create_openai_client with that raw name. The native-client gate must match
    the aliases registered by the gemini profile, or the rebuilt client POSTs
    OpenAI-shaped payloads (bare top-level ``thinking_config``) at the native
    Gemini endpoint and 400s.
    """
    from agent.gemini_native_adapter import GeminiNativeClient

    native_url = "https://generativelanguage.googleapis.com/v1beta"
    # Every alias registered by plugins/model-providers/gemini plus the canonical name.
    for provider in ("gemini", "google", "google-gemini", "google-ai-studio"):
        client = _build(provider, native_url)
        assert isinstance(client, GeminiNativeClient), (
            f"provider={provider!r} on the native Gemini URL must build GeminiNativeClient, got {type(client).__name__}"
        )


def test_non_gemini_provider_on_a_gemini_url_keeps_the_generic_client():
    """Boundary: the alias set must not capture unrelated providers.

    A custom provider pointing at the Gemini endpoint is NOT the ``gemini``
    provider identity — it keeps the generic OpenAI client (pre-existing
    behavior, pinned here so widening the gate cannot silently change it).
    """
    from openai import OpenAI

    native_url = "https://generativelanguage.googleapis.com/v1beta"
    assert isinstance(_build("custom", native_url), OpenAI)
    assert isinstance(_build("some-other-provider", native_url), OpenAI)


def test_gemini_alias_set_matches_the_registered_profile_aliases():
    """Invariant: the native-route alias set and the registered gemini profile
    aliases must stay in lockstep — a new profile alias without a routing alias
    reintroduces this bug class (#104583)."""
    from agent.gemini_native_adapter import GEMINI_NATIVE_PROVIDER_NAMES
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    assert profile is not None, "the bundled gemini profile must be discoverable"
    assert GEMINI_NATIVE_PROVIDER_NAMES == frozenset({profile.name, *profile.aliases})
