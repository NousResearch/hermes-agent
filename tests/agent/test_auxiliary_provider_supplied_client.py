"""Auxiliary ``api_key`` routes honor ``ProviderProfile.create_client()`` — parity with the main agent.

``resolve_provider_client()``'s ``api_key`` branch used to build ``openai.OpenAI`` directly,
so an out-of-tree provider registered with ``auth_type="api_key"`` lost its native transport
for auxiliary tasks even though the main-agent path
(``agent_runtime_helpers._provider_supplied_client``) honors the same hook (#112384).
These tests pin the seam through the real resolution entry point: the native client
(sync + async), the ``None`` fall-through for ordinary providers, and failure isolation
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
def registered(tmp_path, monkeypatch):
    """Register provider profiles for one test; restore both registries and the secret scope after.

    Mirrors the import-time synthesis ``hermes_cli.auth`` performs for plugin ``api_key``
    profiles, which is what routes them into ``_resolve_api_key_branch`` in the first place.
    """
    import hermes_cli.auth as _auth
    from agent import secret_scope as _secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _providers._discover_providers()
    providers_snapshot = (
        dict(_providers._REGISTRY),
        dict(_providers._ALIASES),
        _providers._PROVIDER_LIST_CACHE,
    )
    auth_snapshot = dict(_auth.PROVIDER_REGISTRY)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    home_token = set_hermes_home_override(str(tmp_path))
    scope_token = _secret_scope.set_secret_scope({_PROBE_ENV_VAR: _PROBE_KEY})

    yield (
        lambda profile: (
            (
                _providers.register_provider(profile),
                _auth._register_plugin_provider(profile),
            )
            and profile
        )
    )

    _secret_scope.reset_secret_scope(scope_token)
    reset_hermes_home_override(home_token)
    _providers._REGISTRY.clear()
    _providers._REGISTRY.update(providers_snapshot[0])
    _providers._ALIASES.clear()
    _providers._ALIASES.update(providers_snapshot[1])
    _providers._PROVIDER_LIST_CACHE = providers_snapshot[2]
    _auth.PROVIDER_REGISTRY.clear()
    _auth.PROVIDER_REGISTRY.update(auth_snapshot)


def test_native_profile_supplies_the_auxiliary_client(registered):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_NativeProfile, "aux-seam-native"))
    client, model = resolve_provider_client(
        "aux-seam-native", "probe-model", task="title_generation"
    )

    assert isinstance(client, _FakeNativeClient)
    assert model == "probe-model"
    # The hook receives the same mapping the branch would have passed to openai.OpenAI.
    assert client.kwargs["api_key"] == _PROBE_KEY
    assert client.kwargs["base_url"] == "https://aux-seam-native.invalid"


def test_native_profile_client_survives_the_async_route(registered):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_NativeProfile, "aux-seam-native"))
    client, model = resolve_provider_client(
        "aux-seam-native", "probe-model", async_mode=True
    )

    # HERMES_SKIP_ASYNC_WRAP: a native transport is not rebuilt as AsyncOpenAI.
    assert isinstance(client, _FakeNativeClient)
    assert model == "probe-model"


def test_profile_returning_none_falls_through_to_the_standard_client(registered):
    from openai import OpenAI

    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_PassThroughProfile, "aux-seam-passthrough"))
    client, model = resolve_provider_client("aux-seam-passthrough", "probe-model")

    assert isinstance(client, OpenAI)
    assert model == "probe-model"


def test_a_broken_profile_falls_back_to_the_standard_client(registered):
    from openai import OpenAI

    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile(_ExplodingProfile, "aux-seam-boom"))
    client, model = resolve_provider_client("aux-seam-boom", "probe-model")

    # A raising plugin can only fail to provide a client, never break auxiliary resolution.
    assert isinstance(client, OpenAI)
    assert model == "probe-model"
