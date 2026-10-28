"""Provider-profile entries of OPTIONAL_ENV_VARS: added on first read, same contents as before."""
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


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(cpe, "_STATIC_NAMES", frozenset({"CORE_KEY"}))
    monkeypatch.setattr(cpe, "_PROVIDER_NAMES", set())
    t = OptionalEnvVars({"CORE_KEY": {"description": "core"}, "PLUG_KEY": {"description": "platform"}})
    t.arm(cpe._add_provider_entries)
    return t


def test_fill_runs_on_first_read_core_wins_provider_replaces_platform_first_provider_wins(table, monkeypatch):
    import providers

    calls = []
    profiles = [_profile("alpha", "CORE_KEY", "PLUG_KEY", "ALPHA_API_KEY", "ALPHA_BASE_URL"),
                _profile("beta", "ALPHA_API_KEY"), _profile("oauth", "OAUTH_TOKEN", auth_type="oauth")]
    monkeypatch.setattr(providers, "list_providers", lambda: calls.append(1) or profiles)
    dict.__setitem__(table, "LATE", {})  # writes and raw dict access never fill
    assert calls == []
    assert table["CORE_KEY"] == {"description": "core"} and calls == [1]
    assert table["PLUG_KEY"]["description"] == "Alpha API key"
    assert table["ALPHA_API_KEY"]["description"] == "Alpha API key" and table["ALPHA_API_KEY"]["password"]
    assert table["ALPHA_BASE_URL"]["password"] is False and "OAUTH_TOKEN" not in table
    list(table.items())
    assert calls == [1]  # once


def test_unavailable_config_fails_the_read_loudly_and_the_next_read_retries(table, monkeypatch):
    import providers

    answers = [ConfigBackendUnavailable("boot in flight"), [_profile("alpha", "ALPHA_API_KEY")]]

    def list_providers():
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(providers, "list_providers", list_providers)
    with pytest.raises(ConfigBackendUnavailable):  # fail closed, never a silently partial table
        table.get("ALPHA_API_KEY")
    assert "ALPHA_API_KEY" in table and answers == []


def test_a_broken_provider_layer_is_not_retried(table, monkeypatch):
    import providers

    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("broken")

    monkeypatch.setattr(providers, "list_providers", boom)
    assert "CORE_KEY" in table and "CORE_KEY" in table
    assert calls == [1]
