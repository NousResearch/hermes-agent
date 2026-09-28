"""Live plugin-catalog lookups must not multiply per installed plugin.

A slow or unreachable catalog host used to cost one request timeout per plugin row, on the
dashboard plugins-hub rebuild (inline on the event loop) and on ``hermes plugins list``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugin_catalog as pc
from hermes_cli import plugins_cmd
from hermes_cli import web_server
import hermes_cli.config as _cfg_mod
import hermes_cli.web_server_dashboard as _web_server_dashboard
import hermes_cli.web_server_memory as _web_server_memory
from tools import registry as tools_registry


@pytest.fixture(autouse=True)
def _isolated_live_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "_live_fetch_failed_until", 0.0, raising=False)
    monkeypatch.setattr(pc, "_live_cache_path", lambda: tmp_path / "cache" / "plugin-catalog.json")
    tools_registry.invalidate_check_fn_cache()
    _web_server_dashboard._invalidate_plugins_hub_cache()


class _UnreachableCatalog:
    """Counts network attempts; every one fails like a dead host."""

    def __init__(self):
        self.attempts = 0

    def __call__(self, *args, **kwargs):
        self.attempts += 1
        raise OSError("catalog host unreachable")


def test_failed_live_fetch_is_remembered_until_the_failure_ttl(monkeypatch, tmp_path):
    """One network attempt per failure window; inside it a stale on-disk copy still answers
    (removals published before the outage keep blocking) and the network is left alone."""
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(pc.time, "time", lambda: clock["now"])
    unreachable = _UnreachableCatalog()
    monkeypatch.setattr("httpx.get", unreachable)

    assert pc.fetch_live_catalog() is None
    assert pc.fetch_live_catalog() is None
    assert unreachable.attempts == 1

    cache = pc._live_cache_path()
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"entries": [], "removed": [{"name": "pulled-live", "reason": "cve"}]}))
    stale = clock["now"] - pc.LIVE_CATALOG_TTL_SECONDS - 1  # older than the success TTL under the fake clock
    os.utime(cache, (stale, stale))
    assert pc.fetch_live_catalog()["removed"][0]["name"] == "pulled-live"
    assert unreachable.attempts == 1

    clock["now"] += pc.LIVE_CATALOG_FAILURE_TTL_SECONDS + 1
    pc.fetch_live_catalog()
    assert unreachable.attempts == 2  # window over: exactly one fresh attempt


_PLUGIN_ROWS = [
    ("demo", "1.0.0", "demo plugin", "user", "/tmp/demo-plugin", "demo"),
    ("second", "0.2.0", "second plugin", "user", "/tmp/second-plugin", "second"),
    ("third", "0.3.0", "third plugin", "user", "/tmp/third-plugin", "third"),
]


def test_hub_rebuild_and_plugins_list_resolve_the_kill_list_once(monkeypatch, tmp_path, capsys):
    """Both listing surfaces cost at most ONE network attempt with the catalog unreachable — not
    one per installed plugin — and still report an in-tree removal for every affected row."""
    unreachable = _UnreachableCatalog()
    monkeypatch.setattr("httpx.get", unreachable)
    monkeypatch.setattr(pc, "get_catalog_dir", lambda: tmp_path)
    (tmp_path / "removed.yaml").write_text("removed:\n- name: demo\n  reason: exfiltrated env vars\n")

    monkeypatch.setattr(web_server, "_get_dashboard_plugins", lambda force_rescan=False: [])
    monkeypatch.setattr(_web_server_memory, "_discover_memory_provider_statuses", lambda: [])
    monkeypatch.setattr(_cfg_mod, "get_hermes_home", lambda: Path("/tmp/hermes-home"))
    monkeypatch.setattr(_cfg_mod, "load_config", lambda: {"dashboard": {"hidden_plugins": []}})
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: list(_PLUGIN_ROWS))
    monkeypatch.setattr(plugins_cmd, "_get_current_context_engine", lambda: "compressor")
    monkeypatch.setattr(plugins_cmd, "_get_current_memory_provider", lambda: "")
    monkeypatch.setattr(plugins_cmd, "_discover_context_engines", lambda: [])
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"demo"})
    monkeypatch.setattr(plugins_cmd, "_read_manifest", lambda _path: {"provides_tools": []})
    monkeypatch.setattr(plugins_cmd, "_read_install_metadata", lambda: {})
    monkeypatch.setattr(tools_registry.registry, "get_entry", lambda _name: SimpleNamespace(check_fn=None))

    payload = _web_server_dashboard._merged_plugins_hub(force_refresh=True)
    by_name = {row["name"]: row["removed_reason"] for row in payload["plugins"]}
    assert by_name == {"demo": "exfiltrated env vars", "second": None, "third": None}
    assert unreachable.attempts == 1

    monkeypatch.setattr(pc, "_live_fetch_failed_until", 0.0)  # forget the failure: a fresh window
    plugins_cmd.cmd_list(argparse.Namespace(enabled=False, user=False, no_bundled=False, plain=False, json=True))
    rows = {row["name"]: row["removed"] for row in json.loads(capsys.readouterr().out)}
    assert rows == {"demo": "exfiltrated env vars", "second": None, "third": None}
    assert unreachable.attempts == 2  # one more for the whole listing, not one per row


def test_dropped_cache_no_longer_outvotes_the_in_tree_catalog(monkeypatch):
    """The #119340 shape: a snapshot cached before an update answers every lookup while it
    is fresh, even with the catalog host dead. After the update drops it, the same dead host
    yields ``None`` — callers fall back to the in-tree catalog the update just installed —
    instead of resurrecting the pre-update snapshot for the rest of the TTL."""
    monkeypatch.setattr("httpx.get", _UnreachableCatalog())
    cache = pc._live_cache_path()
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"entries": [], "removed": []}))

    assert pc.fetch_live_catalog() == {"entries": [], "removed": []}  # the stale snapshot wins

    pc.invalidate_live_cache_for_home(cache.parent.parent)

    assert pc.fetch_live_catalog() is None  # nothing left to serve: in-tree takes over


def test_update_invalidates_the_live_catalog_cache_for_every_profile(tmp_path, monkeypatch):
    """Post-update maintenance drops the cached live catalog under the active home AND every
    sibling profile's — the checkout is shared, so one profile's update changes every
    profile's catalog truth at once (#119340)."""
    from hermes_cli import update_cmd
    from hermes_cli import update_cmd_maint
    from hermes_cli import backup as _backup

    root = tmp_path / "home"
    alpha = root / "profiles" / "alpha"
    beta = root / "profiles" / "beta"
    caches = []
    for home in (root, alpha, beta):
        cache = home / "cache" / "plugin-catalog.json"
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({"entries": [], "removed": []}))
        caches.append(cache)

    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: root)
    monkeypatch.setattr(_backup, "_sibling_profile_homes", lambda _home: [("alpha", alpha), ("beta", beta)])

    update_cmd_maint._invalidate_live_plugin_catalog_caches()

    assert not any(cache.exists() for cache in caches)
