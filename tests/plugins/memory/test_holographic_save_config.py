"""``save_config`` of the holographic provider writes config.yaml through the canonical writer.

The provider used to ``yaml.dump`` straight over config.yaml, bypassing the config lock, the
managed-mode refusal and the atomic replace. Two contracts pin the canonical path: unrelated
sections survive a provider save, and a managed install refuses the write.
"""
from __future__ import annotations

import yaml

from plugins.memory.holographic import HolographicMemoryProvider


def _provider():
    return HolographicMemoryProvider(config={})


def test_save_config_merges_into_existing_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  default: keep-me\nmemory:\n  provider: holographic\n")

    _provider().save_config({"db_path": "custom.db", "hrr_dim": "512"}, str(tmp_path))

    raw = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert raw["plugins"]["hermes-memory-store"] == {"db_path": "custom.db", "hrr_dim": "512"}
    assert raw["model"]["default"] == "keep-me"
    assert raw["memory"]["provider"] == "holographic"


def test_save_config_targets_explicit_profile_home(tmp_path, monkeypatch):
    active_home = tmp_path / "active"
    target_home = tmp_path / "target"
    active_home.mkdir()
    target_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(active_home))
    (active_home / "config.yaml").write_text("model:\n  default: active-model\n")
    (target_home / "config.yaml").write_text(
        "model:\n  default: target-model\nmemory:\n  provider: holographic\n"
    )
    active_before = (active_home / "config.yaml").read_bytes()

    _provider().save_config({"auto_extract": "true"}, str(target_home))

    assert (active_home / "config.yaml").read_bytes() == active_before
    target = yaml.safe_load((target_home / "config.yaml").read_text())
    assert target["plugins"]["hermes-memory-store"] == {"auto_extract": "true"}
    assert target["model"]["default"] == "target-model"
    assert target["memory"]["provider"] == "holographic"


def test_save_config_restores_prior_home_override(tmp_path, monkeypatch):
    active_home = tmp_path / "active"
    outer_home = tmp_path / "outer"
    target_home = tmp_path / "target"
    for home in (active_home, outer_home, target_home):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(active_home))

    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    outer_token = set_hermes_home_override(outer_home)
    try:
        _provider().save_config({"hrr_dim": "512"}, str(target_home))
        assert get_hermes_home() == outer_home
    finally:
        reset_hermes_home_override(outer_token)

    assert not (active_home / "config.yaml").exists()
    assert not (outer_home / "config.yaml").exists()
    target = yaml.safe_load((target_home / "config.yaml").read_text())
    assert target["plugins"]["hermes-memory-store"] == {"hrr_dim": "512"}


def test_save_config_respects_managed_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    before = "model:\n  default: managed\n"
    (tmp_path / "config.yaml").write_text(before)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: True)
    monkeypatch.setattr("hermes_cli.config.managed_error", lambda *_a, **_k: None)

    _provider().save_config({"db_path": "custom.db"}, str(tmp_path))

    assert (tmp_path / "config.yaml").read_text() == before
