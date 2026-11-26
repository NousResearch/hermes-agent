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


def test_a_broken_http_plugin_falls_through_instead_of_taking_the_turn_down(registered):
    from agent.agent_runtime_helpers import _provider_supplied_client

    registered(_ExplodingProfile(name="seam-boom", base_url="acp://seam-boom"))
    assert _provider_supplied_client(_agent("seam-boom"), {"api_key": "k"}) is None


def test_a_broken_process_profile_preserves_its_client_error(registered):
    """Regression for #139807: launch kwargs must not reach OpenAI after hook failure."""
    from agent.agent_runtime_helpers import create_openai_client

    registered(_ExplodingProfile(
        name="seam-process-boom", auth_type="external_process", base_url="process://seam-process-boom",
    ))

    with pytest.raises(RuntimeError, match="plugin is broken"):
        create_openai_client(
            _agent("seam-process-boom"),
            {"api_key": "***", "base_url": "process://seam-process-boom", "command": "provider-cli"},
            reason="test",
            shared=False,
        )


def test_skip_flags_replace_the_isinstance_checks_for_in_and_out_of_tree_clients():
    from agent.auxiliary_client import _maybe_wrap_anthropic, _to_async_client

    class _WrappedAsync:
        def __init__(self, sync_client):
            self.sync_client = sync_client

    class _SyncWithAsyncAdapter:
        HERMES_SKIP_TRANSPORT_WRAP = True
        HERMES_ASYNC_CLIENT = _WrappedAsync

    client = _FakeClient()  # never imported by auxiliary_client
    assert _maybe_wrap_anthropic(client, "m", "k", "acp://seam-test") is client
    assert _to_async_client(client, "m")[0] is client

    sync_client = _SyncWithAsyncAdapter()
    assert _maybe_wrap_anthropic(sync_client, "m", "k", "https://example.invalid/v1") is sync_client
    wrapped, model = _to_async_client(sync_client, "m")
    assert isinstance(wrapped, _WrappedAsync) and wrapped.sync_client is sync_client and model == "m"


def test_auxiliary_profile_client_enforces_strict_client_and_allows_none_fallback(registered):
    from agent.auxiliary_client import _api_key_profile_supplied_client

    class _StrictExplodingProfile(ProviderProfile):
        def create_client(self, **kwargs):
            raise RuntimeError("strict native client failed")

    class _StrictOptionalProfile(ProviderProfile):
        def create_client(self, **kwargs):
            return None if kwargs.get("base_url", "").endswith("/openai") else _FakeClient(**kwargs)

    registered(_ExplodingProfile(name="aux-non-strict-boom", base_url="https://example.invalid/v1"))
    registered(_StrictExplodingProfile(name="aux-strict-boom", base_url="https://example.invalid/v1", strict_client=True))
    registered(_StrictOptionalProfile(name="aux-strict-opt", base_url="https://example.invalid/v1", strict_client=True))

    assert _api_key_profile_supplied_client("aux-non-strict-boom", api_key="k", base_url="https://example.invalid/v1") is None
    assert _api_key_profile_supplied_client("aux-non-strict-boom", api_key="k", base_url="https://example.invalid/v1", auxiliary_chain=True) is None

    for aux_chain in (False, True):
        with pytest.raises(RuntimeError, match="strict native client failed"):
            _api_key_profile_supplied_client("aux-strict-boom", api_key="k", base_url="https://example.invalid/v1", auxiliary_chain=aux_chain)

    assert _api_key_profile_supplied_client("aux-strict-opt", api_key="k", base_url="https://example.invalid/openai", auxiliary_chain=True) is None
    client = _api_key_profile_supplied_client("aux-strict-opt", api_key="k", base_url="https://example.invalid/v1", auxiliary_chain=True)
    assert isinstance(client, _FakeClient) and client.kwargs.get("auxiliary_chain") is True
