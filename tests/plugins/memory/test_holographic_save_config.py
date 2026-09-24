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


def test_save_config_respects_managed_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    before = "model:\n  default: managed\n"
    (tmp_path / "config.yaml").write_text(before)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: True)
    monkeypatch.setattr("hermes_cli.config.managed_error", lambda *_a, **_k: None)

    _provider().save_config({"db_path": "custom.db"}, str(tmp_path))

    assert (tmp_path / "config.yaml").read_text() == before
