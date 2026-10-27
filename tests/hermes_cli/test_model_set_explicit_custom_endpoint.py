"""#145466: the Local / custom endpoint wizard posts ``provider: custom`` + a NEW ``base_url`` to
POST /api/model/set. With a named custom provider already configured, the legacy bare-``custom``
recovery bound the request to that provider: the response said ok, but its old URL stayed the
route and the new endpoint was never saved."""

import importlib
import json

import pytest

OLD_URL = "http://127.0.0.1:9/old/v1"
NEW_URL = "http://127.0.0.1:9/new/v1"

LAYOUTS = {
    "custom_providers_list": {
        "custom_providers": [{"name": "provider-a", "base_url": OLD_URL, "api_key": "key-a", "model": "model-a"}],
    },
    "providers_dict": {
        "providers": {"custom:provider-a": {"name": "provider-a", "base_url": OLD_URL, "api_key": "key-a",
                                            "default_model": "model-a"}},
    },
}


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as config_mod

    importlib.reload(config_mod)
    yield home
    importlib.reload(config_mod)


def _write(home, layout):
    cfg = {"model": {"provider": "custom:provider-a", "default": "model-a"}, **LAYOUTS[layout]}
    (home / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")


def _read(home):
    import hermes_yaml as yaml

    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _endpoint_urls(cfg):
    rows = list(cfg.get("custom_providers") or []) + list((cfg.get("providers") or {}).values())
    return {str(r.get("base_url") or "") for r in rows if isinstance(r, dict)}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("api_key", ["key-b", ""])
def test_wizard_save_with_new_url_routes_to_and_saves_the_new_endpoint(isolated_home, layout, api_key):
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    _write(isolated_home, layout)
    result = _apply_model_assignment_sync("main", "custom", "model-b", "", NEW_URL, api_key)

    assert (result["provider"], result["base_url"]) == ("custom", NEW_URL)
    cfg = _read(isolated_home)
    model = cfg["model"]
    assert (model["provider"], model["default"], model["base_url"]) == ("custom", "model-b", NEW_URL)
    # The existing provider's credential never follows the request onto the new host.
    assert model.get("api_key", "") == api_key
    # Both endpoints are listed afterwards: the new one saved, the old one kept.
    assert {OLD_URL, NEW_URL} <= _endpoint_urls(cfg)


def test_bare_custom_without_url_still_recovers_the_named_provider(isolated_home):
    """Control: legacy configs whose stored provider is a bare ``custom`` keep self-healing onto
    the first named custom provider when no endpoint is submitted."""
    from hermes_cli.web_server_config import _normalize_main_model_assignment, _prepare_main_assignment

    _write(isolated_home, "custom_providers_list")
    assert _normalize_main_model_assignment("custom", "model-a")[0] == "custom:provider-a"
    from hermes_cli.config import load_config

    _base_url, result = _prepare_main_assignment(load_config(), "custom", "model-a", "", "")
    assert (result.target_provider, result.base_url) == ("custom:provider-a", OLD_URL)


@pytest.mark.parametrize("api_key", ["key-b", ""])
def test_wizard_validation_probes_the_submitted_endpoint_not_the_saved_one(isolated_home, monkeypatch, api_key):
    """The active named provider's loopback URL sits in ``model.base_url``: validation must probe
    the submitted endpoint with the submitted key, never the saved route it is about to replace."""
    from hermes_cli import models_validate
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    cfg = {"model": {"provider": "custom:provider-a", "default": "model-a", "base_url": OLD_URL,
                     "api_key": "key-a"}, **LAYOUTS["custom_providers_list"]}
    (isolated_home / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    probes = []

    def record(model, provider, *, api_key=None, base_url=None, **_kw):
        probes.append((base_url, api_key))
        return {"accepted": True, "persist": True, "recognized": True, "message": ""}

    monkeypatch.setattr(models_validate, "validate_requested_model", record)
    _apply_model_assignment_sync("main", "custom", "model-b", "", NEW_URL, api_key)

    assert probes, "the submitted endpoint was never validated"
    assert all(url == NEW_URL for url, _ in probes), probes
    assert all((key or "") in {api_key, "no-key-required"} and key != "key-a" for _, key in probes), probes
    assert _read(isolated_home)["model"]["base_url"] == NEW_URL
