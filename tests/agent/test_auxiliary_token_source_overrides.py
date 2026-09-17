"""Callable token sources survive auxiliary credential overrides (#113976).

``key_cmd`` / Entra ID runtimes carry a ``CommandTokenSource`` (a lazy callable) as their
``api_key``. Resolve paths that pin auxiliary calls to the main session's working key used to
assume a string: ``.strip()`` raised ``AttributeError`` (breaking title generation, smart
approvals and vision checks behind the working chat path) while ``str()`` coercions sent the
callable's *repr* out as the bearer token, failing auth with no usable error. These tests pin
the contract at every site named in the issue: strings are stripped, callables reach the wire
client untouched, and usage paths that need a concrete token drop callables instead of
repr-ing them.

All credential-looking values below are dummy fixtures for the stripping/passthrough
assertions; no real secret ever appears.
"""

from __future__ import annotations

import pytest

import providers as _providers
from agent.command_token_source import CommandTokenSource, normalize_token_source
from providers.base import ProviderProfile

_STR_KEY = "  dummy-probe-key  "
_STR_KEY_STRIPPED = "dummy-probe-key"
_NATIVE_TOKEN = "dummy-native-token"


def _token_source() -> CommandTokenSource:
    return CommandTokenSource("dummy-mint-command", "probe")


@pytest.mark.parametrize(
    "value, expected",
    [("  dummy-sk-live  ", "dummy-sk-live"), ("", ""), (None, ""), (1234, "")],
    ids=["strips-strings", "empty-string", "none", "non-credential-junk"],
)
def test_normalize_token_source_strings_and_junk(value, expected):
    assert normalize_token_source(value) == expected


def test_normalize_token_source_passes_callables_through_untouched():
    source = _token_source()
    assert normalize_token_source(source) is source


class _FakeNativeClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _NativeProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return _FakeNativeClient(**kwargs)


@pytest.fixture
def registered(monkeypatch):
    """Register provider profiles for one test on copies of both registries."""
    import hermes_cli.auth as _auth
    from agent import secret_scope as _secret_scope

    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(_auth, "PROVIDER_REGISTRY", dict(_auth.PROVIDER_REGISTRY))

    def _register(profile: ProviderProfile) -> None:
        _providers.register_provider(profile)
        _auth._register_plugin_provider(profile)

    yield _register
    _secret_scope.reset_secret_scope(_secret_scope.set_secret_scope({}))


def _probe_profile() -> ProviderProfile:
    return _NativeProfile(
        name="tokseam-native",
        auth_type="api_key",
        env_vars=("TOKSEAM_PROBE_AUTH",),
        base_url="https://tokseam-native.invalid",
        default_aux_model="probe-model",
    )


def test_api_key_branch_passes_callable_override_through(registered):
    """``_resolve_api_key_branch`` used to ``.strip()`` the override — AttributeError (#113976)."""
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile())
    source = _token_source()
    client, model = resolve_provider_client(
        "tokseam-native", "probe-model", explicit_api_key=source)

    assert isinstance(client, _FakeNativeClient)
    assert model == "probe-model"
    assert client.kwargs["api_key"] is source


def test_api_key_branch_still_strips_string_overrides(registered):
    from agent.auxiliary_client import resolve_provider_client

    registered(_probe_profile())
    client, _ = resolve_provider_client(
        "tokseam-native", "probe-model", explicit_api_key=_STR_KEY)

    assert client.kwargs["api_key"] == _STR_KEY_STRIPPED


def test_named_custom_branch_accepts_callable_override(monkeypatch):
    """``_resolve_named_custom_branch`` used to ``.strip()`` the override — AttributeError (#113976)."""
    from agent.auxiliary_client import _AuxProbeClientStub, aux_probe_mode, resolve_provider_client

    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: {"name": "kcc", "base_url": "https://kcc.invalid/v1"} if name == "kcc" else None,
    )
    source = _token_source()
    with aux_probe_mode():
        client, _ = resolve_provider_client("custom:kcc", "probe-model", explicit_api_key=source)

    assert isinstance(client, _AuxProbeClientStub)
    assert client.api_key is source


def test_custom_branch_reuses_main_runtime_callable_key():
    """``str(main_runtime["api_key"])`` sent the callable's repr as the bearer (#113976)."""
    from agent.auxiliary_client import _AuxProbeClientStub, aux_probe_mode, resolve_provider_client

    source = _token_source()
    with aux_probe_mode():
        client, _ = resolve_provider_client(
            "custom", "probe-model",
            main_runtime={"provider": "custom", "base_url": "https://rt.invalid/v1",
                          "api_key": source, "model": "probe-model"},
        )

    assert isinstance(client, _AuxProbeClientStub)
    assert client.api_key is source


def test_resolve_explicit_runtime_keeps_callable_override():
    """``_resolve_explicit_runtime``'s ``str()`` coercion repr'd callables into the runtime (#113976)."""
    from hermes_cli.runtime_provider import _resolve_explicit_runtime

    source = _token_source()
    runtime = _resolve_explicit_runtime(
        provider="kimi-coding", requested_provider="kimi-coding", model_cfg={},
        explicit_api_key=source, explicit_base_url=None)

    assert runtime is not None
    assert runtime["api_key"] is source


def test_azure_foundry_explicit_callable_override():
    from hermes_cli.runtime_provider_backends import _resolve_azure_foundry_runtime

    source = _token_source()
    runtime = _resolve_azure_foundry_runtime(
        requested_provider="azure-foundry", model_cfg={},
        explicit_api_key=source, explicit_base_url="https://az.invalid/v1")

    assert runtime["api_key"] is source


def test_codex_usage_drops_callable_instead_of_sending_its_repr(monkeypatch):
    """A callable is not a Codex usage token; the native tiers must resolve, never the repr (#113976)."""
    from agent import account_usage

    monkeypatch.setattr(
        account_usage, "resolve_codex_runtime_credentials",
        lambda **kwargs: {"api_key": _NATIVE_TOKEN, "base_url": "https://codex.invalid"})
    monkeypatch.setattr(account_usage, "_read_codex_tokens", lambda: {"tokens": {}})

    token, resolved_base, account_id = account_usage._resolve_codex_usage_credentials(
        None, _token_source())

    assert token == _NATIVE_TOKEN
    assert resolved_base == "https://codex.invalid"
    assert account_id is None
