"""Tests for plugin-contributed config_schema → DEFAULT_CONFIG merge.

Covers the new ``config_schema:`` field in plugin.yaml that lets user
plugins surface their settings in the dashboard CONFIG page (the
config.yaml editor), not just in KEYS / env vars.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    _load_plugin_config_schemas,
    get_effective_default_config,
)


@pytest.fixture
def fake_plugins_root(tmp_path, monkeypatch):
    """Point ``get_hermes_home()`` at a temp dir with a ``plugins/`` subdir.

    Returns the ``plugins/`` Path so tests can drop plugin.yaml fixtures
    into it without touching the real ~/.hermes.
    """
    hermes_home = tmp_path / "hermes_home"
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home",
        lambda: hermes_home,
    )
    return plugins_dir


def _write_plugin(plugins_dir: Path, name: str, manifest: dict) -> Path:
    plugin_dir = plugins_dir / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plugin_dir / "plugin.yaml"
    manifest_path.write_text(yaml.dump(manifest), encoding="utf-8")
    return manifest_path


# ── _load_plugin_config_schemas ──────────────────────────────────────────────


class TestLoadPluginConfigSchemas:
    def test_no_plugins_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home", lambda: tmp_path / "doesnt-exist"
        )
        assert _load_plugin_config_schemas() == {}

    def test_empty_plugins_dir_returns_empty(self, fake_plugins_root):
        assert _load_plugin_config_schemas() == {}

    def test_plugin_without_config_schema_skipped(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "demo", {
            "name": "demo",
            "kind": "platform",
            # no config_schema
        })
        assert _load_plugin_config_schemas() == {}

    def test_plugin_with_config_schema_collected(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "max-messenger", {
            "name": "max-messenger",
            "kind": "platform",
            "config_schema": {
                "home_channel": "",
                "allow_all_users": False,
                "parse_mode": "markdown",
            },
        })
        result = _load_plugin_config_schemas()
        assert "max-messenger" in result
        assert result["max-messenger"] == {
            "home_channel": "",
            "allow_all_users": False,
            "parse_mode": "markdown",
        }

    def test_config_section_overrides_name(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "max-messenger", {
            "name": "max-messenger",
            "config_section": "max_bot",
            "config_schema": {"home_channel": ""},
        })
        result = _load_plugin_config_schemas()
        assert result == {"max_bot": {"home_channel": ""}}

    def test_non_dict_config_schema_skipped(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "broken", {
            "name": "broken",
            "config_schema": "not a dict",
        })
        assert _load_plugin_config_schemas() == {}

    def test_empty_config_schema_skipped(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "empty", {
            "name": "empty",
            "config_schema": {},
        })
        assert _load_plugin_config_schemas() == {}

    def test_malformed_yaml_swallowed(self, fake_plugins_root):
        plugin_dir = fake_plugins_root / "bad"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(":::not valid yaml [[[", encoding="utf-8")
        # No exception, just no entries.
        assert _load_plugin_config_schemas() == {}

    def test_multiple_plugins(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "a-plugin", {
            "name": "a-plugin",
            "config_schema": {"field_a": 1},
        })
        _write_plugin(fake_plugins_root, "b-plugin", {
            "name": "b-plugin",
            "config_schema": {"field_b": True},
        })
        result = _load_plugin_config_schemas()
        assert result == {
            "a-plugin": {"field_a": 1},
            "b-plugin": {"field_b": True},
        }

    def test_category_layout_recursed(self, fake_plugins_root):
        """``<plugins>/<category>/<plugin>/plugin.yaml`` (depth-2) discovered."""
        plugin_dir = fake_plugins_root / "messaging" / "max-messenger"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({
            "name": "max-messenger",
            "config_schema": {"home_channel": ""},
        }), encoding="utf-8")
        result = _load_plugin_config_schemas()
        assert result == {"max-messenger": {"home_channel": ""}}


# ── get_effective_default_config ─────────────────────────────────────────────


class TestGetEffectiveDefaultConfig:
    def test_returns_default_config_when_no_plugins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.get_hermes_home", lambda: tmp_path / "doesnt-exist"
        )
        result = get_effective_default_config()
        # Every key in DEFAULT_CONFIG appears in the result.
        for key in DEFAULT_CONFIG:
            assert key in result

    def test_plugin_section_added(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "max-messenger", {
            "name": "max-messenger",
            "config_schema": {
                "home_channel": "",
                "require_mention": False,
            },
        })
        result = get_effective_default_config()
        assert result["max-messenger"] == {
            "home_channel": "",
            "require_mention": False,
        }

    def test_does_not_mutate_default_config(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "max-messenger", {
            "name": "max-messenger",
            "config_schema": {"x": 1},
        })
        get_effective_default_config()
        assert "max-messenger" not in DEFAULT_CONFIG

    def test_collision_with_bundled_section_rejected(self, fake_plugins_root, caplog):
        # Pick a section that's definitely in DEFAULT_CONFIG.
        bundled_key = next(
            k for k, v in DEFAULT_CONFIG.items() if isinstance(v, dict)
        )
        _write_plugin(fake_plugins_root, "rogue", {
            "name": "rogue",
            "config_section": bundled_key,
            "config_schema": {"hostile": "evil"},
        })
        import logging
        with caplog.at_level(logging.WARNING):
            result = get_effective_default_config()
        # Bundled section preserved unchanged.
        assert result[bundled_key] == DEFAULT_CONFIG[bundled_key]
        assert "rogue" not in [str(s) for s in result.get(bundled_key, {}).values()]
        # Warning was emitted.
        assert any(
            "collides" in rec.message.lower() and bundled_key in rec.message
            for rec in caplog.records
        )

    def test_multiple_plugins_all_added(self, fake_plugins_root):
        _write_plugin(fake_plugins_root, "alpha", {
            "name": "alpha", "config_schema": {"a": 1},
        })
        _write_plugin(fake_plugins_root, "beta", {
            "name": "beta", "config_schema": {"b": True},
        })
        result = get_effective_default_config()
        assert result["alpha"] == {"a": 1}
        assert result["beta"] == {"b": True}
