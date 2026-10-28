"""Provider-profile entries of OPTIONAL_ENV_VARS: config-free, same contents as before."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import config_provider_env as cpe
from hermes_cli.config_backend import ConfigBackendUnavailable
from hermes_cli.config_defaults_env import OptionalEnvVars


def _profile(name, *env_vars, auth_type="api_key"):
    return SimpleNamespace(name=name, display_name=name.title(), env_vars=env_vars, auth_type=auth_type,
                           signup_url=f"https://{name}.test/keys")


def test_every_api_key_provider_var_is_declared_in_core():
    import hermes_cli.config as cfg
    from providers import list_providers

    expected = {v for p in list_providers() if p.auth_type == "api_key" for v in p.env_vars}
    assert expected, "no api_key provider profiles registered"
    assert expected <= set(cfg.OPTIONAL_ENV_VARS)
    assert expected <= cfg.CORE_DECLARED_ENV_NAMES


def test_bundled_profiles_match_discovery_and_leave_the_registry_alone():
    import providers

    before = dict(providers._REGISTRY)
    bundled = {p.name: p for p in providers.bundled_profiles()}
    assert dict(providers._REGISTRY) == before
    discovered = {n for n in providers._REGISTRY if providers.provider_source(n) == "bundled"}
    assert discovered and set(bundled) >= discovered
    for name in discovered:
        assert bundled[name].env_vars == providers._REGISTRY[name].env_vars


def test_the_catalog_builds_under_a_backend_that_refuses_every_read(monkeypatch):
    import hermes_cli.config_backend as cb

    reads = []

    def refuse(*a, **k):
        reads.append(a)
        raise ConfigBackendUnavailable("this test's backend refuses every read")

    monkeypatch.setattr(cb, "_route", refuse)
    monkeypatch.setattr(cpe, "_CATALOG", None)
    entries = cpe._provider_entries()
    assert "ANTHROPIC_API_KEY" in entries or len(entries) > 10
    assert reads == []


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(cpe, "_STATIC_NAMES", frozenset({"CORE_KEY"}))
    monkeypatch.setattr(cpe, "_PROVIDER_NAMES", set())
    monkeypatch.setattr(cpe, "_CATALOG", None)
    t = OptionalEnvVars({"CORE_KEY": {"description": "core"}, "PLUG_KEY": {"description": "platform"}})
    t.arm(cpe._provider_entries, cpe._PROVIDER_NAMES.update)
    return t


def test_fill_runs_on_first_read_core_wins_provider_replaces_platform_first_provider_wins(table, monkeypatch):
    import providers

    calls = []
    profiles = [_profile("alpha", "CORE_KEY", "PLUG_KEY", "ALPHA_API_KEY", "ALPHA_BASE_URL"),
                _profile("beta", "ALPHA_API_KEY"), _profile("oauth", "OAUTH_TOKEN", auth_type="oauth")]
    monkeypatch.setattr(providers, "bundled_profiles", lambda: calls.append(1) or profiles)
    dict.__setitem__(table, "LATE", {})  # writes and raw dict access never fill
    assert calls == []
    assert table["CORE_KEY"] == {"description": "core"} and calls == [1]
    assert table["PLUG_KEY"]["description"] == "Alpha API key"
    assert table["ALPHA_API_KEY"]["description"] == "Alpha API key" and table["ALPHA_API_KEY"]["password"]
    assert table["ALPHA_BASE_URL"]["password"] is False and "OAUTH_TOKEN" not in table
    list(table.items())
    assert calls == [1]  # once


def test_discovery_completion_adds_registered_plugin_providers(table, monkeypatch):
    import providers

    from hermes_cli import config_defaults

    monkeypatch.setattr(providers, "bundled_profiles", lambda: [_profile("alpha", "ALPHA_API_KEY")])
    monkeypatch.setattr(config_defaults, "OPTIONAL_ENV_VARS", table)
    monkeypatch.setattr(providers, "_REGISTRY", {"alpha": _profile("alpha", "ALPHA_API_KEY"),
                                                 "plug": _profile("plug", "PLUG_API_KEY", "ALPHA_API_KEY")})
    monkeypatch.setattr(providers, "provider_source", lambda n: "bundled" if n == "alpha" else "entry_point")
    assert "ALPHA_API_KEY" in table
    cpe.sync_registered_providers()
    assert table["PLUG_API_KEY"]["description"] == "Plug API key"
    assert table["ALPHA_API_KEY"]["description"] == "Alpha API key"  # an earlier entry is kept
    assert "PLUG_API_KEY" in cpe._PROVIDER_NAMES
