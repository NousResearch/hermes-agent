"""A successful /v1/models probe persists into a ``providers:`` entry, not only ``custom_providers:``.

Regression for #159908: the writer read only the legacy list, so a modern ``providers.<name>`` entry
never kept its discovered catalog once the probe cache expired.
"""

import hermes_yaml as yaml

from hermes_cli.config import load_config
from hermes_cli.config_providers import get_compatible_custom_providers
from hermes_cli.model_switch_providers import _entry_credentials, _save_discovered_models_to_config

_URL = "https://gateway.example.com/v1"


def _write(tmp_path, monkeypatch, text: str):
    import hermes_cli.config as config_mod
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(path), raising=False)
    return path


def test_probe_persists_into_the_matching_providers_entry_and_reads_back(tmp_path, monkeypatch):
    path = _write(tmp_path, monkeypatch, (
        "providers:\n"
        "  my-gateway:\n"
        f"    base_url: {_URL}\n"
        "    model: some-model\n"
        "    key_env: MY_GATEWAY_KEY\n"
        "  other:\n"
        "    base_url: https://other.example.com/v1\n"
        "    model: other-model\n"))
    # The picker passes the identity of the entry it probed, computed from the read view.
    probed = next(e for e in get_compatible_custom_providers(load_config()) if e["name"] == "my-gateway")

    _save_discovered_models_to_config(_URL + "/", ["alpha", "beta"],
                                      credential_identity=_entry_credentials(probed, "key_env", "api_key_env")[2])

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["providers"]["my-gateway"]["models"] == {"alpha": {}, "beta": {}}
    assert saved["providers"]["my-gateway"]["models_discovered"] is True
    assert "models" not in saved["providers"]["other"]
    assert "custom_providers" not in saved
    reread = next(e for e in get_compatible_custom_providers(load_config()) if e["name"] == "my-gateway")
    assert list(reread["models"]) == ["alpha", "beta"] and reread["models_discovered"] is True


def test_probe_never_replaces_a_curated_providers_catalog(tmp_path, monkeypatch):
    path = _write(tmp_path, monkeypatch, (
        "providers:\n"
        "  my-gateway:\n"
        f"    base_url: {_URL}\n"
        "    models:\n"
        "      curated-model:\n"
        "        context_length: 32000\n"))

    _save_discovered_models_to_config(_URL, ["alpha"])

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["providers"]["my-gateway"]
    assert saved["models"] == {"curated-model": {"context_length": 32000}}
    assert "models_discovered" not in saved
