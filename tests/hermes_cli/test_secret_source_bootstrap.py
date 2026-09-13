"""Tests for plugin secret-source first-process re-pull (#64177)."""
from __future__ import annotations

import os
from pathlib import Path

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

    monkeypatch.setattr(reg, "list_plugin_sources", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
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

    reg._reset_registry_for_tests()
    assert reg.list_plugin_sources() == []
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"bitwarden": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
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

    monkeypatch.setattr(reg, "list_plugin_sources", lambda: [_StubSource()])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 1, "load": 1}


def test_refresh_reconciles_once_when_the_last_plugin_source_is_removed(monkeypatch):
    """Removal regression (#107695 review): ``discover_and_load(force=True)`` unloads the old
    registration first, so a discovery that finds no enabled plugin source used to return before the
    cache reset and the installed-scope refresh — the per-home snapshot and the current scope kept the
    removed plugin's names. After a discovery that DID re-apply plugin sources, the next one that finds
    none must reconcile exactly once; a home that never had a plugin source stays a no-op."""
    mgr = PluginManager()
    called = {"reset": 0, "load": 0, "scope": 0}

    import agent.secret_sources.registry as reg

    sources = [_StubSource()]
    monkeypatch.setattr(reg, "list_plugin_sources", lambda: list(sources))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"secrets": {"myvault": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.env_loader.reset_secret_source_cache",
                        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1))
    monkeypatch.setattr("hermes_cli.env_loader.load_hermes_dotenv",
                        lambda **kw: called.__setitem__("load", called["load"] + 1))
    monkeypatch.setattr("agent.secret_scope.refresh_installed_secret_scope",
                        lambda *a, **kw: called.__setitem__("scope", called["scope"] + 1) or True)

    mgr._refresh_secret_sources_after_discovery()          # plugin source present and enabled
    assert called == {"reset": 1, "load": 1, "scope": 1}

    sources.clear()                                         # the plugin is gone (force-reload unloaded it)
    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 2, "load": 2, "scope": 2}    # reconciled once so its names drop out

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 2, "load": 2, "scope": 2}    # and not again: nothing left to reconcile


def test_refresh_retries_removal_cleanup_after_a_failed_attempt(monkeypatch):
    """The reconcile marker must survive a failed cleanup (#107695 review on f5f88d5058): clearing it
    before the fallible reset/reload/refresh left the removed plugin's credential active while every
    later no-source discovery returned early. It clears only once cleanup succeeds."""
    mgr = PluginManager()
    calls = {"load": 0}
    fail = {"on": True}

    import agent.secret_sources.registry as reg

    sources = [_StubSource()]
    monkeypatch.setattr(reg, "list_plugin_sources", lambda: list(sources))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"secrets": {"myvault": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.env_loader.reset_secret_source_cache", lambda *a, **kw: None)
    monkeypatch.setattr("agent.secret_scope.refresh_installed_secret_scope", lambda *a, **kw: True)

    def _load(**kw):
        calls["load"] += 1
        if fail["on"]:
            raise RuntimeError("reload blew up")

    monkeypatch.setattr("hermes_cli.env_loader.load_hermes_dotenv", _load)

    fail["on"] = False
    mgr._refresh_secret_sources_after_discovery()   # enabled: marker set
    assert calls["load"] == 1

    sources.clear()
    fail["on"] = True
    mgr._refresh_secret_sources_after_discovery()   # removal cleanup attempt fails
    assert calls["load"] == 2
    assert mgr._plugin_secret_sources_reconciled is True   # NOT cleared by a failed attempt

    fail["on"] = False
    mgr._refresh_secret_sources_after_discovery()   # retried, succeeds
    assert calls["load"] == 3
    assert mgr._plugin_secret_sources_reconciled is False

    mgr._refresh_secret_sources_after_discovery()   # nothing left to reconcile
    assert calls["load"] == 3


def test_refresh_respects_custom_is_enabled(monkeypatch):
    """A source with custom activation (no ``enabled`` key) is re-pulled."""
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(
        reg, "list_plugin_sources", lambda: [_CustomActivationSource()]
    )
    # No `enabled` key at all — only the source's custom contract decides.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"vault_id": "abc123"}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
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

    monkeypatch.setattr(
        reg, "list_plugin_sources", lambda: [_CustomActivationSource()]
    )
    # `enabled: true` but the custom contract ignores it and requires vault_id.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
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

    monkeypatch.setattr(reg, "list_plugin_sources", lambda: [_Boom()])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
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
    """A cold process discovers a real plugin and applies its credential."""
    import agent.secret_sources.registry as reg
    from hermes_cli import env_loader

    reg._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "fixture-secret-source"
    plugin_dir.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [fixture-secret-source]\n"
        "secrets:\n"
        "  fixturevault:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: fixture-secret-source\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from agent.secret_sources.base import (\n"
        "    SECRET_SOURCE_API_VERSION, FetchResult, SecretSource,\n"
        ")\n\n"
        "class FixtureVault(SecretSource):\n"
        "    name = 'fixturevault'\n"
        "    label = 'Fixture vault'\n"
        "    api_version = SECRET_SOURCE_API_VERSION\n"
        "    shape = 'bulk'\n\n"
        "    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:\n"
        "        return FetchResult(secrets={\n"
        "            'HERMES_TEST_PLUGIN_BOOTSTRAP': 'from-plugin',\n"
        "        })\n\n"
        "def register(ctx):\n"
        "    ctx.register_secret_source(FixtureVault())\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TEST_PLUGIN_BOOTSTRAP", raising=False)

    try:
        PluginManager().discover_and_load()

        assert os.environ["HERMES_TEST_PLUGIN_BOOTSTRAP"] == "from-plugin"
        assert [source.name for source in reg.list_plugin_sources()] == [
            "fixturevault"
        ]
    finally:
        os.environ.pop("HERMES_TEST_PLUGIN_BOOTSTRAP", None)
        reg._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()
