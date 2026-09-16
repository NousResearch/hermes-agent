"""Auxiliary ``api_key`` routes honor ``ProviderProfile.create_client()`` — parity with the main agent.

``resolve_provider_client()``'s ``api_key`` branch used to build ``openai.OpenAI`` directly,
so an out-of-tree provider registered with ``auth_type="api_key"`` lost its native transport
for auxiliary tasks even though the main-agent path
(``agent_runtime_helpers._provider_supplied_client``) honors the same hook (#112384).
These tests pin the seam through the real resolution entry point: the native client
(sync + async), and the fall-through to the standard client for ordinary providers and
for a broken plugin.
"""

from __future__ import annotations

import pytest

import providers as _providers
from providers.base import ProviderProfile

_PROBE_ENV_VAR = "AUX_SEAM_PROBE_AUTH"
_PROBE_KEY = "probe-sentinel"


class _FakeNativeClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NativeProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return _FakeNativeClient(**kwargs)


class _PassThroughProfile(ProviderProfile):
    """Ordinary provider: ``create_client`` returns None so the standard client is built."""

    def create_client(self, **kwargs):
        return None


class _ExplodingProfile(ProviderProfile):
    def create_client(self, **kwargs):
        raise RuntimeError("plugin is broken")


def _probe_profile(cls, name: str) -> ProviderProfile:
    return cls(
        name=name,
        auth_type="api_key",
        env_vars=(_PROBE_ENV_VAR,),
        base_url=f"https://{name}.invalid",
        default_aux_model="probe-model",
    )


@pytest.fixture
def registered(monkeypatch):
    """Register provider profiles for one test on copies of both registries.

    Mirrors the import-time synthesis ``hermes_cli.auth`` performs for plugin ``api_key``
    profiles, which is what routes them into ``_resolve_api_key_branch`` in the first place.
    """
    import hermes_cli.auth as _auth
    from agent import secret_scope as _secret_scope

    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(_auth, "PROVIDER_REGISTRY", dict(_auth.PROVIDER_REGISTRY))
    scope_token = _secret_scope.set_secret_scope({_PROBE_ENV_VAR: _PROBE_KEY})

    def _register(profile: ProviderProfile) -> None:
        _providers.register_provider(profile)
        _auth._register_plugin_provider(profile)

    yield _register
    _secret_scope.reset_secret_scope(scope_token)


@pytest.mark.parametrize(
    "extra",
    [{"task": "title_generation"}, {"async_mode": True}],
    ids=["sync", "async-skips-AsyncOpenAI-rewrap"],
)
def test_native_profile_supplies_the_auxiliary_client(registered, extra):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_NativeProfile, "aux-seam-native"))
    client, model = resolve_provider_client("aux-seam-native", "probe-model", **extra)

    assert isinstance(client, _FakeNativeClient)
    assert model == "probe-model"
    # The hook receives the same mapping the branch would have passed to openai.OpenAI.
    assert client.kwargs == {"api_key": _PROBE_KEY, "base_url": "https://aux-seam-native.invalid"}


@pytest.mark.parametrize(
    "profile_cls", [_PassThroughProfile, _ExplodingProfile], ids=["returns-None", "raises"]
)
def test_profile_without_a_client_falls_back_to_the_standard_client(registered, profile_cls):
    from openai import OpenAI

    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(profile_cls, "aux-seam-fallback"))
    client, model = resolve_provider_client("aux-seam-fallback", "probe-model")

    # A raising plugin can only fail to provide a client, never break auxiliary resolution.
    assert isinstance(client, OpenAI)
    assert model == "probe-model"
