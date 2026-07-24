"""Regression tests for dotted YAML config updates."""

from __future__ import annotations

import builtins
import stat

import yaml

from utils import atomic_roundtrip_yaml_update


def _block_ruamel_import(monkeypatch):
    real_import = builtins.__import__

    def block_ruamel(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ruamel" or name.startswith("ruamel."):
            raise ModuleNotFoundError("No module named 'ruamel'", name="ruamel")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_ruamel)


def test_atomic_roundtrip_yaml_update_falls_back_when_ruamel_missing(monkeypatch, tmp_path):
    """Slash-command config toggles should still save without ruamel.yaml."""
    _block_ruamel_import(monkeypatch)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  default: test-model\n", encoding="utf-8")

    atomic_roundtrip_yaml_update(config_path, "display.runtime_footer.enabled", True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model"]["default"] == "test-model"
    assert config["display"]["runtime_footer"]["enabled"] is True


def test_atomic_roundtrip_yaml_update_fallback_creates_missing_file(monkeypatch, tmp_path):
    _block_ruamel_import(monkeypatch)

    config_path = tmp_path / "config.yaml"
    atomic_roundtrip_yaml_update(config_path, "display.runtime_footer.enabled", True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config == {"display": {"runtime_footer": {"enabled": True}}}


def test_atomic_roundtrip_yaml_update_fallback_preserves_file_mode(monkeypatch, tmp_path):
    _block_ruamel_import(monkeypatch)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("display:\n  skin: ares\n", encoding="utf-8")
    config_path.chmod(0o640)

    atomic_roundtrip_yaml_update(config_path, "display.runtime_footer.enabled", True)

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o640
