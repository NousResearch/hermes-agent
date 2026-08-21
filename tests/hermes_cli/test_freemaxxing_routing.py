"""Freemaxxing provider composition and profile-isolation contracts."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("FREEMAXXING_PORT", "0")

from agent.secret_scope import (  # noqa: E402
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
import providers as provider_registry  # noqa: E402
from providers import get_provider_profile  # noqa: E402

# Provider discovery is process-global and pytest may have populated it from the
# operator's HERMES_HOME before collecting this module. This contract targets
# the bundled plugin, so reset discovery and suppress user-plugin overrides.
for module_name in list(sys.modules):
    if module_name.startswith("plugins.model_providers.freemaxxing"):
        sys.modules.pop(module_name, None)
provider_registry._REGISTRY.clear()
provider_registry._ALIASES.clear()
provider_registry._PROVIDER_LIST_CACHE = None
provider_registry._discovered = False
provider_registry._user_plugins_dir = lambda: None

# Bundled model-provider plugins are loaded by the provider registry under a
# synthetic module name. Trigger canonical discovery, then consume the exact
# module objects installed by that loader. Importing the dotted synthetic name
# through importlib is not a production contract: the loader intentionally does
# not create an importable ``plugins.model_providers`` parent package.
PROFILE = get_provider_profile("freemaxxing")
assert PROFILE is not None
PLUGIN = sys.modules["plugins.model_providers.freemaxxing"]
PROXY = sys.modules["plugins.model_providers.freemaxxing.proxy"]

# Import auth only after the bundled profile has registered its scoped env-var
# contract and loopback capability.
from hermes_cli.auth import resolve_api_key_provider_credentials  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_runtime():
    set_multiplex_active(False)
    PROXY.pool.clear()
    yield
    set_multiplex_active(False)
    PROXY.pool.clear()


def test_static_catalog_is_metadata_only(monkeypatch):
    profile = get_provider_profile("freemaxxing")
    assert profile is not None
    monkeypatch.setattr(
        PLUGIN,
        "_build_pool",
        lambda: (_ for _ in ()).throw(
            AssertionError("catalog inspection must not establish runtime authority")
        ),
    )
    assert profile.fetch_models() == ["freemaxxing"]
    assert PROXY.pool.count() == 0


def test_generic_runtime_resolver_carries_the_local_capability():
    token = set_secret_scope({"FREEMAXXING_API_KEY": PLUGIN.local_token()})
    try:
        credentials = resolve_api_key_provider_credentials("freemaxxing")
    finally:
        reset_secret_scope(token)
    assert credentials["api_key"] == PLUGIN.local_token()
    assert credentials["base_url"].startswith("http://127.0.0.1:")
    assert credentials["base_url"].endswith("/v1")


def test_multiplex_mode_rejects_before_either_profile_key_is_read(monkeypatch):
    reads: list[str] = []

    def should_not_resolve_nous():
        reads.append("nous")
        return "https://example.invalid/v1", "should-not-be-read"

    monkeypatch.setattr(PLUGIN, "_resolve_nous_credentials", should_not_resolve_nous)
    set_multiplex_active(True)

    for scoped_key in ("profile-a-key", "profile-b-key"):
        token = set_secret_scope({"OPENROUTER_API_KEY": scoped_key})
        try:
            credentials = resolve_api_key_provider_credentials("freemaxxing")
            assert credentials["api_key"] == ""
            with pytest.raises(RuntimeError, match="multiplex_profiles"):
                PLUGIN._build_pool()
            with pytest.raises(RuntimeError, match="multiplex_profiles"):
                PLUGIN.ensure_proxy()
        finally:
            reset_secret_scope(token)

    assert reads == []
    assert PROXY.pool.count() == 0


def test_pool_enrolls_nous_and_optional_openrouter_but_never_huggingface(monkeypatch):
    monkeypatch.setattr(
        PLUGIN,
        "_resolve_nous_credentials",
        lambda: ("https://nous.invalid/v1", "nous-key"),
    )
    monkeypatch.setattr(
        PLUGIN,
        "_resolve_key",
        lambda names: "or-key" if "OPENROUTER_API_KEY" in names else "",
    )

    PLUGIN._build_pool()
    assert [backend.name for backend in PROXY.pool.snapshot()] == [
        "nous-portal",
        "openrouter",
    ]


def test_scope_miss_never_falls_through_to_process_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "other-profile-key")
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        assert PLUGIN._resolve_key(("OPENROUTER_API_KEY",)) == ""
    finally:
        reset_secret_scope(token)
