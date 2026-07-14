"""Tests for plugin secret-source first-process re-pull (#64177)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import PluginManager


def test_refresh_secret_sources_noop_without_plugin_sources(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    def _list():
        return []

    monkeypatch.setattr(
        "agent.secret_sources.registry.list_sources", _list, raising=False
    )
    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", _list)

    def _boom_reset():
        called["reset"] += 1

    def _boom_load(**kwargs):
        called["load"] += 1

    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache", _boom_reset
    )
    monkeypatch.setattr("hermes_cli.env_loader.load_hermes_dotenv", _boom_load)

    mgr._refresh_secret_sources_after_discovery()
    assert called["reset"] == 0
    assert called["load"] == 0


def test_refresh_secret_sources_repulls_when_plugin_enabled(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    class _Src:
        name = "myvault"

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", lambda: [_Src()])

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )

    def _reset():
        called["reset"] += 1

    def _load(**kwargs):
        called["load"] += 1

    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache", _reset
    )
    monkeypatch.setattr("hermes_cli.env_loader.load_hermes_dotenv", _load)

    mgr._refresh_secret_sources_after_discovery()
    assert called["reset"] == 1
    assert called["load"] == 1


def test_discover_and_load_invokes_refresh(monkeypatch):
    mgr = PluginManager()
    hits = {"n": 0}
    monkeypatch.setattr(PluginManager, "_discover_and_load_inner", lambda self: None)
    monkeypatch.setattr(
        PluginManager,
        "_refresh_secret_sources_after_discovery",
        lambda self: hits.__setitem__("n", hits["n"] + 1),
    )
    mgr.discover_and_load()
    assert hits["n"] == 1
