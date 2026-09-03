"""Regression coverage for hermes_cli.auth PROVIDER_REGISTRY sync (#102123).

``hermes_cli.auth`` can be imported while ``providers._discover_providers()``
is still mid-scan (a plugin's transitive import chain reaches it). Its
module-level snapshot then only sees the providers registered up to that
moment and module code never runs again — leaving the registry permanently
incomplete. These tests pin the fix: the snapshot logic lives in an idempotent
``sync_plugin_providers_to_registry()`` that discovery re-invokes after its
full scan.
"""

import sys
import types

import pytest

import providers
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    ProviderConfig,
    sync_plugin_providers_to_registry,
)


def _register_test_profile(name, *, auth_type="api_key", env_vars=("TESTSYNC_API_KEY", "TESTSYNC_BASE_URL"), aliases=()):
    profile = providers.ProviderProfile(
        name=name,
        display_name=f"Test Sync {name}",
        base_url="https://example.invalid/v1",
        env_vars=env_vars,
        auth_type=auth_type,
        aliases=aliases,
    )
    providers.register_provider(profile)
    return profile


def _cleanup_registry(names):
    for name in names:
        providers._REGISTRY.pop(name, None)
        providers._PROVIDER_LIST_CACHE = None
        config = PROVIDER_REGISTRY.pop(name, None)
        if config is not None:
            for key in [k for k, v in PROVIDER_REGISTRY.items() if v is config]:
                del PROVIDER_REGISTRY[key]


def test_sync_registers_provider_missing_from_snapshot():
    """A provider registered after auth's initial snapshot is picked up."""
    name = "test-sync-late-provider"
    _register_test_profile(name, aliases=("testsync-late",))
    PROVIDER_REGISTRY.pop(name, None)
    try:
        assert name not in PROVIDER_REGISTRY  # reproduces the #102123 symptom

        sync_plugin_providers_to_registry()

        config = PROVIDER_REGISTRY[name]
        assert isinstance(config, ProviderConfig)
        assert config.auth_type == "api_key"
        assert config.api_key_env_vars == ("TESTSYNC_API_KEY",)
        assert config.base_url_env_var == "TESTSYNC_BASE_URL"
        assert PROVIDER_REGISTRY["testsync-late"] is config
    finally:
        _cleanup_registry([name, "testsync-late"])


def test_sync_registers_external_process_provider():
    name = "test-sync-external-provider"
    _register_test_profile(name, auth_type="external_process", env_vars=())
    try:
        sync_plugin_providers_to_registry()

        config = PROVIDER_REGISTRY[name]
        assert config.auth_type == "external_process"
        assert config.inference_base_url == "https://example.invalid/v1"
    finally:
        _cleanup_registry([name])


def test_sync_is_idempotent_and_leaves_existing_entries_untouched():
    name = "test-sync-idempotent-provider"
    _register_test_profile(name)
    try:
        sync_plugin_providers_to_registry()
        first = PROVIDER_REGISTRY[name]
        first.api_key_env_vars = ("SENTINEL",)

        sync_plugin_providers_to_registry()

        assert PROVIDER_REGISTRY[name] is first
        assert first.api_key_env_vars == ("SENTINEL",)
    finally:
        _cleanup_registry([name])


def test_discovery_completion_resyncs_auth_module(monkeypatch):
    """Discovery calls back into an already-imported hermes_cli.auth (#102123)."""
    calls = []
    fake_auth = types.ModuleType("hermes_cli.auth")
    fake_auth.sync_plugin_providers_to_registry = lambda: calls.append(True)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)

    providers._sync_auth_provider_registry()

    assert calls == [True]


def test_discovery_completion_skips_unimported_auth_module(monkeypatch):
    """No sync (and no import) when hermes_cli.auth was never imported."""
    monkeypatch.delitem(sys.modules, "hermes_cli.auth", raising=False)

    providers._sync_auth_provider_registry()

    # The helper must not have imported hermes_cli.auth just to sync it.
    assert "hermes_cli.auth" not in sys.modules


@pytest.mark.parametrize("bad_module", [None, object()])
def test_sync_failure_never_raises(monkeypatch, bad_module):
    """A broken consumer or missing hook must not break discovery."""
    if bad_module is None:
        monkeypatch.delitem(sys.modules, "hermes_cli.auth", raising=False)
    else:
        broken = types.ModuleType("hermes_cli.auth")
        broken.sync_plugin_providers_to_registry = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        monkeypatch.setitem(sys.modules, "hermes_cli.auth", broken)

    providers._sync_auth_provider_registry()  # must not raise
