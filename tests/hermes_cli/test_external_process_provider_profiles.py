"""External-process model-provider profiles use the generic picker contracts."""

from __future__ import annotations

import importlib

import pytest
from providers.base import ProviderProfile


class _FakeProcessProfile(ProviderProfile):
    """A small out-of-tree-style profile: no vendor-specific core branches."""

    def __init__(self, live_models: list[str] | None):
        super().__init__(
            name="test-process-provider", display_name="Test Process Provider",
            description="Test ACP provider", base_url="acp://test-process",
            auth_type="external_process", process_command="test-process",
            process_args=("--acp",), process_command_env_vars=("TEST_PROCESS_COMMAND",),
            process_args_env_var="TEST_PROCESS_ARGS", fallback_models=("fallback-agent",),
        )
        self.live_models = live_models
        self.fetch_calls = 0
        self.fetch_kwargs: dict = {}

    def fetch_models(self, **kwargs):
        self.fetch_calls += 1
        self.fetch_kwargs = kwargs
        return self.live_models


class _IncompatibleSigProcessProfile(ProviderProfile):
    """An external-process profile whose fetch_models needs keyword-only HTTP credentials.

    The generic external-process discovery calls fetch_models() with no arguments, so this
    signature is incompatible with it and the call raises TypeError internally.
    """

    def __init__(self):
        super().__init__(
            name="test-process-provider-badsig", display_name="Test Process Provider",
            description="Test ACP provider", base_url="acp://test-process",
            auth_type="external_process", process_command="test-process",
            process_args=("--acp",), process_command_env_vars=("TEST_PROCESS_COMMAND",),
            process_args_env_var="TEST_PROCESS_ARGS", fallback_models=("fallback-agent",),
        )

    def fetch_models(self, *, api_key, base_url):
        return ["live-agent"]


@pytest.fixture
def process_profile(monkeypatch):
    """Register a profile through the same registries an external plugin uses."""
    import providers
    from hermes_cli import auth

    profile = _FakeProcessProfile(["live-agent"])
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    auth._register_plugin_provider(profile)
    yield profile


@pytest.fixture
def incompatible_process_profile(monkeypatch):
    """Register an external-process profile with an incompatible fetch_models signature."""
    import providers
    from hermes_cli import auth

    profile = _IncompatibleSigProcessProfile()
    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    auth._register_plugin_provider(profile)
    yield profile


def _canonicalize_fake_profile(monkeypatch):
    """Reload the import-time catalog extension after registering the fake profile."""
    import hermes_cli.models as models
    import hermes_cli.models_catalog_static as catalog

    catalog = importlib.reload(catalog)
    monkeypatch.setattr(models, "CANONICAL_PROVIDERS", catalog.CANONICAL_PROVIDERS)
    return catalog


def test_external_process_profile_is_catalogued_discovered_cached_and_visible(
        tmp_path, monkeypatch, process_profile):
    """A registered ACP profile reaches catalog, cached discovery, and picker rows generically."""
    from hermes_cli import auth, model_switch_providers, models

    catalog = _canonicalize_fake_profile(monkeypatch)
    assert any(entry.slug == process_profile.name for entry in catalog.CANONICAL_PROVIDERS)

    executable = tmp_path / "test-process"
    executable.write_text("", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("TEST_PROCESS_COMMAND", str(executable))
    monkeypatch.setenv("TEST_PROCESS_ARGS", "--test-mode")

    assert models.provider_model_ids(process_profile.name, force_refresh=True) == ["fallback-agent", "live-agent"]
    assert process_profile.fetch_kwargs == {}
    assert models.cached_provider_model_ids(process_profile.name, force_refresh=True) == ["fallback-agent", "live-agent"]

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr(model_switch_providers, "_credential_pool_is_usable", lambda *_a, **_k: False)
    rows = model_switch_providers.list_authenticated_providers(max_models=10)
    row = next(row for row in rows if row["slug"] == process_profile.name)
    assert row["models"] == ["fallback-agent", "live-agent"]

    first_fingerprint = models._credential_fingerprint(process_profile.name)
    monkeypatch.setenv("TEST_PROCESS_ARGS", "--other-mode")
    assert models._credential_fingerprint(process_profile.name) != first_fingerprint


def test_external_process_profile_uses_fallback_models_when_discovery_fails(monkeypatch, process_profile):
    """The generic cache path retains a profile's offline catalog when its process cannot list."""
    from hermes_cli import models

    _canonicalize_fake_profile(monkeypatch)
    process_profile.live_models = None
    assert models.provider_model_ids(process_profile.name, force_refresh=True) == ["fallback-agent"]
    assert models.cached_provider_model_ids(process_profile.name, force_refresh=True) == ["fallback-agent"]


def test_external_process_profile_incompatible_fetch_signature_degrades_gracefully(
        monkeypatch, incompatible_process_profile):
    """The generic external-process discovery calls fetch_models() with no arguments. A profile
    whose fetch_models requires keyword-only HTTP credentials raises TypeError under that call;
    provider_model_ids swallows it and degrades to the curated catalog rather than propagating
    the TypeError to the picker."""
    from hermes_cli import models

    _canonicalize_fake_profile(monkeypatch)
    # No TypeError escapes; the profile has no static catalog entry so the graceful result is empty.
    assert models.provider_model_ids(
        incompatible_process_profile.name, force_refresh=True) == []
