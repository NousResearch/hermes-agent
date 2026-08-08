"""Tests for plugin secret-source first-process re-pull (#64177)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION,
    FetchResult,
    SecretSource,
)
from hermes_cli.plugins import PluginManager


class _StubSource(SecretSource):
    """Minimal spec-compliant plugin source for tests."""

    api_version = SECRET_SOURCE_API_VERSION
    shape = "bulk"

    def __init__(self, name: str = "myvault", scheme: str | None = None):
        self.name = name
        self.scheme = scheme

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult(secrets={})


class _CustomActivationSource(_StubSource):
    """Ignores ``enabled`` and activates when a custom key is present."""

    def is_enabled(self, cfg: dict) -> bool:
        return bool(isinstance(cfg, dict) and cfg.get("vault_id"))


def test_refresh_secret_sources_noop_without_plugin_sources(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


def test_refresh_secret_sources_noop_when_only_builtins(monkeypatch):
    """Bundled sources must never trigger a re-pull."""
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(
        reg, "list_sources", lambda: [_StubSource(name="bitwarden")]
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"bitwarden": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


def test_refresh_secret_sources_repulls_when_plugin_enabled(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", lambda: [_StubSource()])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 1, "load": 1}


def test_refresh_respects_custom_is_enabled(monkeypatch):
    """A source with custom activation (no ``enabled`` key) is re-pulled."""
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", lambda: [_CustomActivationSource()])
    # No `enabled` key at all — only the source's custom contract decides.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"vault_id": "abc123"}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 1, "load": 1}


def test_refresh_skips_custom_source_when_not_activated(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", lambda: [_CustomActivationSource()])
    # `enabled: true` but the custom contract ignores it and requires vault_id.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


def test_refresh_skips_source_whose_is_enabled_raises(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    class _Boom(_StubSource):
        def is_enabled(self, cfg: dict) -> bool:
            raise RuntimeError("boom")

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_sources", lambda: [_Boom()])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


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


def test_real_plugin_source_discovery_applies_dotenv(monkeypatch, tmp_path):
    """End-to-end: a real plugin source registered via discovery triggers a
    re-pull that flows through the registry's is_enabled contract."""
    import agent.secret_sources.registry as reg

    reg._reset_registry_for_tests()

    # Register a real plugin source the way PluginContext.register_secret_source
    # would (lands in registry.register_source).
    plugin_source = _StubSource(name="tmpvault", scheme="tmpvault")
    assert reg.register_source(plugin_source) is True
    assert any(getattr(s, "name", "") == "tmpvault" for s in reg.list_sources())

    applied = {"reset": 0, "load": 0}
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda: applied.__setitem__("reset", applied["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: applied.__setitem__("load", applied["load"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"tmpvault": {"enabled": True}}},
    )

    mgr = PluginManager()
    mgr._refresh_secret_sources_after_discovery()

    assert applied == {"reset": 1, "load": 1}
    reg._reset_registry_for_tests()
