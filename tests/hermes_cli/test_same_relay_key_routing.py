"""Same relay groups keep their credentials through selection and restoration.

Everything here is synthetic and offline; config and credential-store reads are
replaced at their I/O seams, while catalog, switch and runtime resolution are real.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.config import get_compatible_custom_providers
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.providers import resolve_provider_full


@pytest.fixture
def relay(monkeypatch):
    cfg = {"model": {"provider": "custom:relay-2", "default": "shared-model"},
           "custom_providers": [
               {"name": "Relay", "base_url": "https://relay.example/v1", "key_env": "RELAY_KEY_A", "model": "shared-model", "discover_models": False},
               {"name": "Relay", "base_url": "https://relay.example/v1", "key_env": "RELAY_KEY_B", "model": "shared-model", "discover_models": False},
           ]}
    monkeypatch.setattr(rp, "load_config", lambda: deepcopy(cfg))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: deepcopy(cfg))
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: deepcopy(cfg))
    monkeypatch.setattr(rp, "_get_model_config", lambda: deepcopy(cfg["model"]))
    monkeypatch.setattr("agent.credential_pool._load_config_safe", lambda: deepcopy(cfg))
    monkeypatch.setattr(rp, "load_pool", lambda *_a, **_kw: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setenv("RELAY_KEY_A", "synthetic-key-a")
    monkeypatch.setenv("RELAY_KEY_B", "synthetic-key-b")
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.model_switch_providers._build_curated_lists", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda **kw: {
        "accepted": True, "persist": True, "recognized": True, "message": None})
    return cfg


def _catalog(cfg):
    return [r for r in list_authenticated_providers(
        current_provider="custom:relay-2", user_providers=cfg.get("providers", {}),
        custom_providers=get_compatible_custom_providers(cfg), probe_custom_providers=False,
    ) if r.get("is_user_defined")]


@pytest.mark.parametrize("kind", ["key_env", "inline"])
def test_second_group_runtime_uses_its_own_key(relay, kind):
    if kind == "inline":
        for i, entry in enumerate(relay["custom_providers"]):
            entry.pop("key_env")
            entry["api_key"] = ["synthetic-key-a", "synthetic-key-b"][i]
    rows = _catalog(relay)
    assert {r["slug"] for r in rows} == {"custom:relay", "custom:relay-2"}
    selected = rp.resolve_runtime_provider(requested="custom:relay-2", target_model="shared-model")
    assert selected["api_key"] == "synthetic-key-b"
    result = switch_model("shared-model", current_model="shared-model", current_provider="custom:relay",
                          explicit_provider="custom:relay-2", user_providers={},
                          custom_providers=get_compatible_custom_providers(relay))
    assert result.success, result.error_message
    assert result.target_provider == "custom:relay-2"
    assert result.api_key == "synthetic-key-b"
    identity = rp.canonical_custom_identity(base_url=result.base_url,
                                           config_provider=result.target_provider, model=result.new_model)
    assert identity == "custom:relay-2"
    assert rp.resolve_runtime_provider(requested=identity)["api_key"] == "synthetic-key-b"


def test_second_group_does_not_reuse_first_pool(relay, monkeypatch):
    calls = []
    def load_pool(name):
        calls.append(name)
        return SimpleNamespace(has_credentials=lambda: name == "custom:relay",
                               select=lambda: SimpleNamespace(api_key="synthetic-key-a"))
    monkeypatch.setattr(rp, "load_pool", load_pool)
    monkeypatch.setattr(rp, "_pool_entry_api_key", lambda entry: entry.api_key)
    selected = rp.resolve_runtime_provider(requested="custom:relay-2")
    assert selected["api_key"] == "synthetic-key-b"
    assert "custom:relay" not in calls


def test_group_slug_matches_after_multiple_models_in_first_group(relay):
    duplicate = deepcopy(relay["custom_providers"][0])
    duplicate["model"] = "other-model"
    relay["custom_providers"].insert(1, duplicate)
    rows = _catalog(relay)
    assert {r["slug"] for r in rows} == {"custom:relay", "custom:relay-2"}
    second = resolve_provider_full("custom:relay-2", {}, get_compatible_custom_providers(relay))
    assert second is not None
    assert second.api_key_env_vars == ("RELAY_KEY_B",)
    assert rp.resolve_runtime_provider(requested="custom:relay-2")["api_key"] == "synthetic-key-b"


def test_environment_names_are_case_sensitive(relay):
    relay["custom_providers"][1]["key_env"] = "relay_key_a"
    assert len(get_compatible_custom_providers(relay)) == 2


def test_same_address_named_and_legacy_groups_both_visible(relay):
    second = relay["custom_providers"].pop()
    relay["providers"] = {"relay-b": {"name": "Relay", "api": second["base_url"],
                                       "key_env": "RELAY_KEY_B", "models": {"model-b": {}},
                                       "discover_models": False}}
    rows = _catalog(relay)
    assert len(rows) == 2
    assert {r["slug"] for r in rows} == {"custom:relay", "relay-b"}


def test_literal_suffix_name_does_not_collide_with_generated_slug(relay):
    relay["custom_providers"].append({"name": "Relay-2", "base_url": "https://relay.example/v1",
                                       "api_key": "synthetic-key-c", "model": "model-c", "discover_models": False})
    rows = _catalog(relay)
    assert len(rows) == 3
    assert len({r["slug"] for r in rows}) == 3
    selected_keys = {rp.resolve_runtime_provider(requested=r["slug"])["api_key"] for r in rows}
    assert selected_keys == {"synthetic-key-a", "synthetic-key-b", "synthetic-key-c"}


@pytest.mark.parametrize("kind", ["inline", "key_env"])
def test_persisted_route_and_real_pool_stay_on_second_group(tmp_path, monkeypatch, kind):
    import hermes_yaml as yaml
    from agent.credential_pool import load_pool, credential_pool_matches_provider, resolve_runtime_pool_key
    from hermes_cli.model_switch import persist_model_selection
    from hermes_cli.config import load_config

    home = tmp_path / "relay-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda **kw: {
        "accepted": True, "persist": True, "recognized": True, "message": None})
    rows = []
    for suffix in ("A", "B"):
        row = {"name": "Relay", "base_url": "https://relay.example/v1", "model": "shared-model",
               "discover_models": False}
        if kind == "inline":
            row["api_key"] = f"synthetic-{suffix}"
        else:
            row["key_env"] = f"RELAY_KEY_{suffix}"
            monkeypatch.setenv(row["key_env"], f"synthetic-{suffix}")
        rows.append(row)
    cfg = {"model": {"provider": "custom:relay", "default": "shared-model"}, "custom_providers": rows}
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = switch_model("shared-model", current_model="shared-model", current_provider="custom:relay",
                          explicit_provider="custom:relay-2", custom_providers=get_compatible_custom_providers(load_config()))
    assert result.success, result.error_message
    assert result.api_key == "synthetic-B"
    persist_model_selection(result, path)
    saved = load_config()
    assert saved["model"]["provider"] == "custom:relay-2"
    assert rp.resolve_runtime_provider(requested=saved["model"]["provider"])["api_key"] == "synthetic-B"
    pool = load_pool("custom:relay-2")
    if kind == "inline":
        assert pool.has_credentials()
        selected = pool.select()
        assert selected is not None
        assert selected.access_token == "synthetic-B"
    assert credential_pool_matches_provider(pool, "custom:relay-2", base_url=rows[1]["base_url"])
    assert not credential_pool_matches_provider("custom:relay", "custom:relay-2", base_url=rows[1]["base_url"])
    assert resolve_runtime_pool_key("custom:relay-2", rows[1]["base_url"]) == "custom:relay-2"


def test_discovered_catalog_writes_only_matching_key_group(relay, monkeypatch):
    from hermes_cli.custom_provider_identity import credential_identity
    from hermes_cli.model_switch_providers import _save_discovered_models_to_config
    saved = []
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.append(deepcopy(cfg)))
    _save_discovered_models_to_config(relay["custom_providers"][1]["base_url"], ["second-only"],
                                     credential_identity=credential_identity(relay["custom_providers"][1]))
    assert len(saved) == 1
    first, second = saved[0]["custom_providers"]
    assert "models" not in first
    assert list(second["models"]) == ["second-only"]


@pytest.mark.parametrize("kind", ["inline", "key_env", "template"])
def test_cli_wizard_keeps_both_groups_and_updates_selected_only(tmp_path, monkeypatch, kind):
    import hermes_yaml as yaml
    from hermes_cli.config import load_config
    from hermes_cli.main_provider_setup import _named_custom_provider_map
    from hermes_cli.model_setup_flows_custom import _model_flow_named_custom
    home = tmp_path / "wizard-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", lambda *a, **k: 1)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    rows = []
    for suffix in ("A", "B"):
        row = {"name": "Relay", "base_url": "https://relay.example/v1", "model": "old-model",
               "models": ["old-model", "new-model"], "discover_models": False}
        monkeypatch.setenv(f"RELAY_KEY_{suffix}", f"synthetic-{suffix}")
        if kind == "key_env":
            row["key_env"] = f"RELAY_KEY_{suffix}"
        else:
            row["api_key"] = f"${{RELAY_KEY_{suffix}}}" if kind == "template" else f"synthetic-{suffix}"
        rows.append(row)
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"provider": "custom:relay", "default": "old-model"},
                                    "custom_providers": rows}))
    catalog = _named_custom_provider_map(load_config())
    assert set(catalog) == {"custom:relay", "custom:relay-2"}
    _model_flow_named_custom({}, catalog["custom:relay-2"])
    saved = yaml.safe_load(path.read_text())
    assert saved["custom_providers"][0] == rows[0]
    assert saved["custom_providers"][1]["model"] == "new-model"
    assert saved["custom_providers"][0]["model"] == "old-model"
    assert saved["model"]["provider"] == "custom:relay-2"
    assert rp.resolve_runtime_provider(requested=saved["model"]["provider"])["api_key"] == "synthetic-B"


def test_saving_new_key_on_same_url_creates_second_group(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli.main_provider_setup import _save_custom_provider
    home = tmp_path / "new-group"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _save_custom_provider("https://relay.example/v1", "synthetic-A", "a-model", name="Relay")
    _save_custom_provider("https://relay.example/v1", "synthetic-B", "b-model", name="Relay")
    saved = yaml.safe_load((home / "config.yaml").read_text())["custom_providers"]
    assert [(entry["api_key"], entry["model"]) for entry in saved] == [
        ("synthetic-A", "a-model"), ("synthetic-B", "b-model")]
