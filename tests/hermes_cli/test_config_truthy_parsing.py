"""Regression tests for quoted-string config boolean parsing.

Config values like ``enabled: "false"`` (quoted strings in YAML) were
treated as ``True`` by ``bool()`` because any non-empty string is truthy
in Python.  These tests verify that affected config paths now use
``is_truthy_value()`` which correctly handles ``"false"``, ``"0"``,
``"off"`` as False.
"""
import pytest

from agent import curator
from agent import curator_backup


# ── curator.is_enabled ──────────────────────────────────────────────

class TestCuratorEnabled:
    def test_unquoted_false_disables(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {"enabled": False})
        assert curator.is_enabled() is False

    def test_quoted_false_disables(self, monkeypatch):
        """Regression: ``enabled: "false"`` must disable the curator."""
        monkeypatch.setattr(curator, "_load_config", lambda: {"enabled": "false"})
        assert curator.is_enabled() is False

    def test_quoted_zero_disables(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {"enabled": "0"})
        assert curator.is_enabled() is False

    def test_quoted_off_disables(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {"enabled": "off"})
        assert curator.is_enabled() is False

    def test_default_enabled_when_missing(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {})
        assert curator.is_enabled() is True

    def test_quoted_true_enables(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {"enabled": "true"})
        assert curator.is_enabled() is True


# ── curator.get_prune_builtins ──────────────────────────────────────

class TestCuratorPruneBuiltins:
    def test_quoted_false_disables_prune(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {"prune_builtins": "false"})
        assert curator.get_prune_builtins() is False

    def test_default_prune_enabled(self, monkeypatch):
        monkeypatch.setattr(curator, "_load_config", lambda: {})
        assert curator.get_prune_builtins() is True


# ── curator_backup.is_enabled ───────────────────────────────────────

class TestCuratorBackupEnabled:
    def test_quoted_false_disables_backup(self, monkeypatch):
        monkeypatch.setattr(curator_backup, "_load_config", lambda: {"enabled": "false"})
        assert curator_backup.is_enabled() is False

    def test_default_backup_enabled(self, monkeypatch):
        monkeypatch.setattr(curator_backup, "_load_config", lambda: {})
        assert curator_backup.is_enabled() is True
