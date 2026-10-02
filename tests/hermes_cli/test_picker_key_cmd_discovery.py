"""Command-auth discovery is lazy, credential-scoped, and never persists bearers."""

import copy

import pytest

from hermes_cli import model_switch_providers as picker


@pytest.mark.parametrize("route", ["providers", "custom_providers"])
@pytest.mark.parametrize("static", [{}, {"api_key": "stale"}, {"key_env": "TEST_GATEWAY_KEY"}])
def test_picker_command_auth_is_lazy_and_credential_scoped(monkeypatch, route, static):
    from agent import command_token_source
    from hermes_cli import models

    monkeypatch.setenv("TEST_GATEWAY_KEY", "stale")
    mints, probes = [], []

    def mint(command, label):
        mints.append(command)
        if command == "broken":
            raise RuntimeError("private helper output")
        return f"token-{command}", 3600

    def fetch(key, url, **kwargs):
        probes.append(key)
        return ["fallback", f"catalog-{key}"] if key.startswith("token-") else None

    monkeypatch.setattr(command_token_source, "_mint", mint)
    monkeypatch.setattr(models, "fetch_api_models", fetch)
    monkeypatch.setattr("hermes_cli.models_local.should_use_ollama_native_catalog", lambda *a, **k: False)
    entry = {"name": "Gateway", "base_url": "https://gateway.invalid/v1", "key_cmd": "tenant-a",
             "model": "fallback", "default_model": "fallback", **static}

    def rows(entries, *, probe=True):
        b = picker._PickerBuild("", "", "", None, False, False, probe, False, False, set(), {})
        if route == "providers":
            picker._lap_user_provider_rows(b, {f"gateway-{i}": e for i, e in enumerate(entries)})
        else:
            picker._lap_custom_provider_rows(b, entries)
        return b.results

    for disabled in [False, "false"]:
        assert rows([{**entry, "discover_models": disabled}])[0]["models"] == ["fallback"]
    assert rows([entry], probe=False)[0]["models"] == ["fallback"]
    assert not mints and not probes

    first = rows([entry])[0]["models"]
    assert first == ["fallback", "catalog-token-tenant-a"]
    assert mints == ["tenant-a"] and probes == ["token-tenant-a"]
    for probe in [False, True]:
        assert rows([entry], probe=probe)[0]["models"] == first
    assert mints == ["tenant-a"] and probes == ["token-tenant-a"]

    from hermes_cli.config import save_config, load_config
    other = {**entry, "key_cmd": "tenant-b", "discover_models": False}
    save_config({"custom_providers": [entry, other]})
    if route == "custom_providers":
        rows([entry])
        assert load_config()["custom_providers"][1].get("models") is None
    source = command_token_source.build_command_token_provider("tenant-a")
    empty = picker._NativePickerModelList()
    result = models.cached_fetch_api_models(
        source, entry["base_url"], force_refresh=True, fetch_models=lambda: empty)
    assert isinstance(result, picker._NativePickerModelList) and result == []
    cached_empty = models.cached_fetch_api_models(source, entry["base_url"], cache_only=True)
    assert isinstance(cached_empty, picker._NativePickerModelList) and cached_empty == []
    for probe in [False, True]:
        native_row = rows([entry], probe=probe)[0]
        assert native_row["models"] == [] and native_row["native_catalog_empty"]
    models.clear_provider_models_cache()

    distinct = rows([entry, {**entry, "key_cmd": "tenant-b"}])
    assert [r["models"] for r in distinct] == [first, ["fallback", "catalog-token-tenant-b"]]
    assert rows([{**entry, "key_cmd": "broken"}])[0]["models"] == ["fallback"]
    assert "stale" not in probes


@pytest.mark.parametrize("route", ["providers", "custom_providers"])
@pytest.mark.parametrize("discover", [True, False, "false"])
@pytest.mark.parametrize("static", [{}, {"api_key": "stale"}, {"key_env": "TEST_GATEWAY_KEY"}])
def test_setup_probe_credentials_never_become_saved_credentials(monkeypatch, route, discover, static):
    from agent import command_token_source
    from hermes_cli import config as config_module, models
    from hermes_cli import model_setup_flows_custom as setup

    monkeypatch.setenv("TEST_GATEWAY_KEY", "stale")
    mints, probes, choices = [], [], []
    entry = {"name": "Gateway", "base_url": "https://gateway.invalid/v1", "key_cmd": "tenant-a",
             "model": "fallback", "default_model": "fallback", "discover_models": discover, **static}
    config = {"model": {"default": "fallback", "provider": "custom"},
              route: {"gateway": copy.deepcopy(entry)} if route == "providers" else [copy.deepcopy(entry)]}
    config_module.save_config(config)
    from hermes_cli.main_provider_setup import _named_custom_provider_map
    info = next(iter(_named_custom_provider_map(config_module.load_config()).values()))

    def mint(command, label):
        mints.append(command)
        return "transient-bearer", 3600

    def fetch(key, url, **kwargs):
        probes.append(key)
        return ["fallback", "discovered"] if key == "transient-bearer" else None

    def pick(name, available, saved):
        choices.extend(available)
        return available[-1]

    monkeypatch.setattr(command_token_source, "_mint", mint)
    monkeypatch.setattr(models, "fetch_api_models", fetch)
    monkeypatch.setattr("hermes_cli.models_local.should_use_ollama_native_catalog", lambda *a, **k: False)
    monkeypatch.setattr(setup, "_pick_named_custom_model", pick)
    monkeypatch.setattr(setup, "_ask", lambda *a, **k: "fallback")
    setup._model_flow_named_custom(config, info)
    if discover is True:
        assert mints == ["tenant-a"] and probes == ["transient-bearer"]
        assert choices == ["fallback", "discovered"]
    else:
        assert not mints and not probes
        assert choices == ["fallback"]
    saved = config_module.load_config()
    assert "transient-bearer" not in repr(saved)
    persisted = saved[route]["gateway"] if route == "providers" else saved[route][0]
    assert persisted["key_cmd"] == entry["key_cmd"]
    assert persisted.get("api_key", "") == static.get("api_key", "")
