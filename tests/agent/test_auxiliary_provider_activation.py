"""Provider-plugin activation regressions for auxiliary LLM routing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import auxiliary_client as aux
from hermes_cli.auth import AuthError


@pytest.fixture(autouse=True)
def _clear_auxiliary_state():
    aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()
    yield
    aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()


def _disable_provider_plugins(monkeypatch, *provider_ids: str) -> None:
    blocked = frozenset(provider_ids)
    monkeypatch.setattr(
        "providers.is_provider_plugin_active",
        lambda provider_id: provider_id not in blocked,
    )


def test_non_plugin_managed_provider_remains_allowed(monkeypatch):
    import providers

    provider_id = "runtime-only-aux-provider-test"
    snapshot = SimpleNamespace(
        plugin_managed_provider_ids=frozenset({"openrouter"}),
        active_plugin_provider_ids=frozenset({"openrouter"}),
    )
    monkeypatch.setattr(providers, "_ensure_providers_discovered", lambda: snapshot)

    assert providers.is_provider_plugin_active(provider_id) is True
    assert aux._aux_provider_plugin_is_active(provider_id) is True


def test_disabled_openrouter_never_reads_credentials_or_builds_client(monkeypatch):
    _disable_provider_plugins(monkeypatch, "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-used")
    load_pool = MagicMock(side_effect=AssertionError("credential pool was read"))
    build_client = MagicMock(side_effect=AssertionError("client was built"))
    monkeypatch.setattr(aux, "load_pool", load_pool)
    monkeypatch.setattr(aux, "_create_openai_client", build_client)

    assert aux._try_openrouter() == (None, None)
    assert aux.resolve_provider_client("openrouter", "test/model") == (None, None)
    load_pool.assert_not_called()
    build_client.assert_not_called()


def test_disabled_nous_never_reads_auth_or_refreshes_runtime(monkeypatch):
    _disable_provider_plugins(monkeypatch, "nous")
    read_auth = MagicMock(side_effect=AssertionError("Nous auth was read"))
    resolve_runtime = MagicMock(side_effect=AssertionError("Nous runtime refreshed"))
    build_client = MagicMock(side_effect=AssertionError("client was built"))
    monkeypatch.setattr(aux, "_read_nous_auth", read_auth)
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", resolve_runtime)
    monkeypatch.setattr(aux, "_create_openai_client", build_client)

    assert aux._try_nous() == (None, None)
    assert aux.resolve_provider_client("nous", "test-model") == (None, None)
    read_auth.assert_not_called()
    resolve_runtime.assert_not_called()
    build_client.assert_not_called()


@pytest.mark.parametrize("provider", ["custom", "custom:corp", "corp"])
def test_disabled_custom_blocks_direct_and_named_endpoint_construction(
    monkeypatch,
    provider,
):
    _disable_provider_plugins(monkeypatch, "custom")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: {
            "name": "corp",
            "base_url": "https://corp.invalid/v1",
            "api_key": "must-not-be-used",
        }
        if name in {"corp", "custom:corp"}
        else None,
    )
    build_client = MagicMock(side_effect=AssertionError("client was built"))
    api_key_fallback = MagicMock(
        side_effect=AssertionError("API-key fallback was attempted")
    )
    monkeypatch.setattr(aux, "_create_openai_client", build_client)
    monkeypatch.setattr(aux, "_resolve_api_key_provider", api_key_fallback)

    client, model = aux.resolve_provider_client(
        provider,
        "test-model",
        explicit_base_url="https://explicit.invalid/v1",
        explicit_api_key="must-not-be-used",
    )

    assert (client, model) == (None, None)
    build_client.assert_not_called()
    api_key_fallback.assert_not_called()


def test_rejected_custom_runtime_is_not_revived_from_openai_base_url(monkeypatch):
    _disable_provider_plugins(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    resolve_runtime = MagicMock(
        side_effect=AuthError(
            "Provider 'custom' is disabled by plugin configuration.",
            code="invalid_provider",
        )
    )
    read_legacy_key = MagicMock(side_effect=AssertionError("legacy key was read"))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_runtime,
    )
    monkeypatch.setattr(aux, "_scoped_key_env", read_legacy_key)

    assert aux._resolve_custom_runtime() == (None, None, None)
    resolve_runtime.assert_called_once_with(requested="custom")
    read_legacy_key.assert_not_called()


def test_text_and_vision_auto_chains_skip_all_disabled_special_providers(
    monkeypatch,
):
    _disable_provider_plugins(monkeypatch, "openrouter", "nous", "custom")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-used")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "")
    monkeypatch.setattr(aux, "_resolve_provider_vision_default", lambda _provider: None)
    monkeypatch.setattr(aux, "_resolve_api_key_provider", lambda: (None, None))
    build_client = MagicMock(side_effect=AssertionError("client was built"))
    resolve_nous = MagicMock(side_effect=AssertionError("Nous runtime refreshed"))
    resolve_custom = MagicMock(side_effect=AssertionError("custom runtime resolved"))
    monkeypatch.setattr(aux, "_create_openai_client", build_client)
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", resolve_nous)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_custom,
    )

    assert aux._resolve_auto() == (None, None)
    assert aux.resolve_vision_provider_client(provider="auto") == (None, None, None)
    build_client.assert_not_called()
    resolve_nous.assert_not_called()
    resolve_custom.assert_not_called()


def test_cached_direct_client_is_not_reused_after_provider_is_disabled(monkeypatch):
    disabled = {"value": False}
    monkeypatch.setattr(
        "providers.is_provider_plugin_active",
        lambda provider_id: not (
            provider_id == "openrouter" and disabled["value"]
        ),
    )
    monkeypatch.setattr(aux, "_peek_pool_entry", lambda _provider: None)
    cached_client = MagicMock()
    resolver = MagicMock(return_value=(cached_client, "test/model"))
    monkeypatch.setattr(aux, "resolve_provider_client", resolver)

    assert aux._get_cached_client("openrouter", "test/model")[0] is cached_client
    disabled["value"] = True

    assert aux._get_cached_client("openrouter", "test/model") == (None, None)
    resolver.assert_called_once()


def test_auto_cache_is_partitioned_by_canonical_activation_state(monkeypatch):
    generation = {"value": "enabled"}
    monkeypatch.setattr(
        aux,
        "_aux_activation_cache_fingerprint",
        lambda: generation["value"],
    )
    monkeypatch.setattr(aux, "_pool_cache_hint", lambda *args, **kwargs: "")
    cached_client = MagicMock()

    def resolve_auto(*args, **kwargs):
        if generation["value"] == "enabled":
            return cached_client, "test/model"
        return None, None

    resolver = MagicMock(side_effect=resolve_auto)
    monkeypatch.setattr(aux, "resolve_provider_client", resolver)

    assert aux._get_cached_client("auto", "test/model")[0] is cached_client
    generation["value"] = "disabled"

    # Cache miss preserves the requested model even when no client resolves;
    # the important invariant is that the old client is not reused.
    assert aux._get_cached_client("auto", "test/model") == (None, "test/model")
    assert resolver.call_count == 2


def test_cache_is_partitioned_by_provider_discovery_identity(monkeypatch):
    activation_state = ("same-activation",)
    discovery_identity = {"value": ("profile-a", "project-a")}
    monkeypatch.setattr(
        "hermes_cli.config.load_plugin_activation_state",
        lambda: activation_state,
    )
    monkeypatch.setattr(
        "providers.get_provider_discovery_identity",
        lambda: discovery_identity["value"],
    )
    monkeypatch.setattr(aux, "_pool_cache_hint", lambda *args, **kwargs: "")
    first_client = MagicMock(name="profile-a-client")
    second_client = MagicMock(name="profile-b-client")
    resolver = MagicMock(
        side_effect=[
            (first_client, "test/model"),
            (second_client, "test/model"),
        ]
    )
    monkeypatch.setattr(aux, "resolve_provider_client", resolver)

    assert aux._get_cached_client("auto", "test/model")[0] is first_client
    discovery_identity["value"] = ("profile-b", "project-b")
    assert aux._get_cached_client("auto", "test/model")[0] is second_client

    assert resolver.call_count == 2


def test_named_custom_alias_uses_custom_activation_owner_before_cache_gate(
    monkeypatch,
    tmp_path,
):
    activation_checks = []
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """custom_providers:
  - name: kimi
    base_url: https://custom-kimi.invalid/v1
    api_key: custom-key
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def is_active(provider_id):
        activation_checks.append(provider_id)
        return provider_id != "kimi-coding"

    monkeypatch.setattr("providers.is_provider_plugin_active", is_active)
    monkeypatch.setattr(aux, "_aux_activation_cache_fingerprint", lambda: "stable")
    monkeypatch.setattr(aux, "_pool_cache_hint", lambda *args, **kwargs: "")
    monkeypatch.setattr(aux, "_peek_pool_entry", lambda _provider: None)

    try:
        client, model = aux._get_cached_client("kimi", "custom-model")

        assert client is not None
        assert "custom-kimi.invalid" in str(client.base_url)
        assert model == "custom-model"
        assert "custom" in activation_checks
    finally:
        aux.shutdown_cached_clients()


def test_disabled_canonical_provider_cannot_be_shadowed_by_named_custom(
    monkeypatch,
    tmp_path,
):
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """custom_providers:
  - name: nous
    base_url: https://custom-nous.invalid/v1
    api_key: custom-key
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "providers.is_provider_plugin_active",
        lambda provider_id: provider_id != "nous",
    )
    resolver = MagicMock(side_effect=AssertionError("resolver must stay gated"))
    monkeypatch.setattr(aux, "resolve_provider_client", resolver)

    assert aux._get_cached_client("nous", "test-model") == (None, None)
    resolver.assert_not_called()


def test_disabled_canonical_provider_cannot_be_shadowed_through_alias(
    monkeypatch,
    tmp_path,
):
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """custom_providers:
  - name: kimi-coding
    base_url: https://custom-kimi-canonical.invalid/v1
    api_key: custom-key
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "providers.is_provider_plugin_active",
        lambda provider_id: provider_id != "kimi-coding",
    )
    resolver = MagicMock(side_effect=AssertionError("resolver must stay gated"))
    monkeypatch.setattr(aux, "resolve_provider_client", resolver)

    assert aux._get_cached_client("kimi", "test-model") == (None, None)
    resolver.assert_not_called()
