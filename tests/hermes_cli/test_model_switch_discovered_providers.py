"""A successful /v1/models probe persists into a ``providers:`` entry, not only ``custom_providers:``.

Regression for #159908: the picker probed ``providers.<name>`` entries but only ever saved into the
legacy list, so the discovered catalog was lost once the probe cache expired.
"""

import hermes_yaml as yaml
import pytest

from hermes_cli.config import load_config
from hermes_cli.model_switch_providers import _save_discovered_models_to_config, list_authenticated_providers

_URL = "https://gateway.example.com/v1"


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    import hermes_cli.config as config_mod
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MY_GATEWAY_KEY", "sk-test")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path), raising=False)
    return path


def _open_picker(monkeypatch, live_catalog: list[str]) -> dict:
    """One /model picker open on the current provider, the endpoint answering with *live_catalog*."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models",
                        lambda api_key, api_url, **kw: list(live_catalog) if api_url.rstrip("/") == _URL else None)
    rows = list_authenticated_providers(current_provider="my-gateway", user_providers=load_config()["providers"],
                                        custom_providers=[], max_models=50)
    return next(row for row in rows if row["slug"] == "my-gateway")


def test_a_live_probe_of_a_providers_entry_is_saved_and_does_not_pin_the_endpoint(config_file, monkeypatch):
    config_file.write_text(
        "providers:\n"
        "  my-gateway:\n"
        f"    api: {_URL}\n"
        "    default_model: some-model\n"
        "    key_env: MY_GATEWAY_KEY\n"
        "  other:\n"
        "    api: https://other.example.com/v1\n"
        "    default_model: other-model\n", encoding="utf-8")

    assert _open_picker(monkeypatch, ["alpha", "beta"])["models"] == ["alpha", "beta"]

    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert saved["providers"]["my-gateway"]["models"] == {"alpha": {}, "beta": {}}
    assert saved["providers"]["my-gateway"]["models_discovered"] is True
    assert "models" not in saved["providers"]["other"]
    assert "custom_providers" not in saved
    # The probe cache expires; the saved catalog is Hermes's own discovery, not a user allowlist, so the
    # next probe still runs and a wider live catalog replaces it.
    (config_file.parent / "provider_models_cache.json").unlink()
    assert _open_picker(monkeypatch, ["alpha", "beta", "gamma"])["models"] == ["alpha", "beta", "gamma"]
    assert list(yaml.safe_load(config_file.read_text(encoding="utf-8"))["providers"]["my-gateway"]["models"]) == [
        "alpha", "beta", "gamma"]


def test_a_live_probe_never_replaces_a_curated_providers_catalog(config_file):
    config_file.write_text(
        "providers:\n"
        "  my-gateway:\n"
        f"    api: {_URL}\n"
        "    models:\n"
        "      curated-model:\n"
        "        context_length: 32000\n", encoding="utf-8")

    _save_discovered_models_to_config(_URL, ["alpha"])

    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))["providers"]["my-gateway"]
    assert saved["models"] == {"curated-model": {"context_length": 32000}}
    assert "models_discovered" not in saved
