"""Tests for _remove_memory_provider and _save_memory_provider legacy key fix."""
import pytest

from hermes_cli.plugins_cmd import _remove_memory_provider, _save_memory_provider


class TestRemoveMemoryProvider:
    def test_remove_existing(self, monkeypatch):
        config = {"memory": {"providers": ["a", "b", "c"], "provider": "a"}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        result = _remove_memory_provider("b")
        assert result is True
        assert saved["memory"]["providers"] == ["a", "c"]
        assert saved["memory"]["provider"] == "a"

    def test_remove_first(self, monkeypatch):
        config = {"memory": {"providers": ["a", "b"], "provider": "a"}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        result = _remove_memory_provider("a")
        assert result is True
        assert saved["memory"]["providers"] == ["b"]
        assert saved["memory"]["provider"] == "b"

    def test_remove_last(self, monkeypatch):
        config = {"memory": {"providers": ["a"], "provider": "a"}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        result = _remove_memory_provider("a")
        assert result is True
        assert saved["memory"]["providers"] == []
        assert saved["memory"]["provider"] == ""

    def test_remove_nonexistent(self, monkeypatch):
        config = {"memory": {"providers": ["a", "b"], "provider": "a"}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        result = _remove_memory_provider("z")
        assert result is False
        assert not saved  # save_config should not have been called

    def test_remove_from_empty(self, monkeypatch):
        config = {"memory": {"providers": [], "provider": ""}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        result = _remove_memory_provider("a")
        assert result is False
        assert not saved

    def test_updates_legacy_key(self, monkeypatch):
        config = {"memory": {"providers": ["x", "y"], "provider": "x"}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        result = _remove_memory_provider("x")
        assert result is True
        assert saved["memory"]["providers"] == ["y"]
        assert saved["memory"]["provider"] == "y"


class TestSaveMemoryProviderLegacyKey:
    def test_legacy_key_is_first_provider(self, monkeypatch):
        config = {"memory": {"providers": [], "provider": ""}}
        saved = {}

        def fake_load():
            return config

        def fake_save(cfg):
            saved.update(cfg)

        monkeypatch.setattr("hermes_cli.config.load_config", fake_load)
        monkeypatch.setattr("hermes_cli.config.save_config", fake_save)

        _save_memory_provider("a")
        assert saved["memory"]["providers"] == ["a"]
        assert saved["memory"]["provider"] == "a"

        # Reset saved and update config in-place for second call
        saved.clear()
        config["memory"]["providers"] = ["a"]
        config["memory"]["provider"] = "a"

        _save_memory_provider("b")
        assert saved["memory"]["providers"] == ["a", "b"]
        assert saved["memory"]["provider"] == "a"
