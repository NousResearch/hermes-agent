"""Plugins hub must not multiply live-catalog fetches (issue #113677 fix contract).

A hub rebuild annotates every installed plugin with the kill-list reason; resolving the kill list
per row cost one synchronous catalog HTTPS request per candidate, so a slow/unreachable catalog
host stalled the dashboard event loop for minutes. These tests pin the two halves of the fix:
one kill-list resolution per rebuild, and a remembered failed fetch within a short TTL window.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugin_catalog as pc
from hermes_cli import plugins_cmd_catalog as pcc
from hermes_cli import web_server
import hermes_cli.config as _cfg_mod
import hermes_cli.web_server_dashboard as _web_server_dashboard
import hermes_cli.web_server_memory as _web_server_memory
from hermes_cli import plugins_cmd
from tools import registry as tools_registry


_PLUGIN_ROWS = [
    ("demo", "1.0.0", "demo plugin", "user", "/tmp/demo-plugin", "demo"),
    ("second", "0.2.0", "second plugin", "user", "/tmp/second-plugin", "second"),
]


@pytest.fixture(autouse=True)
def _reset_live_fetch_state(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "_live_fetch_failed_until", {})
    monkeypatch.setattr(
        pc, "_live_cache_path", lambda: tmp_path / "cache" / "plugin-catalog.json"
    )
    tools_registry.invalidate_check_fn_cache()
    _web_server_dashboard._invalidate_plugins_hub_cache()


class _UnreachableCatalog:
    """Counts network attempts; every attempt blocks like a real timeout, then fails."""

    def __init__(self):
        self.attempts = 0

    def __call__(self, *args, **kwargs):
        self.attempts += 1
        raise OSError("catalog host unreachable")


def _patch_hub_dependencies(monkeypatch, *, rows=None):
    rows = rows if rows is not None else list(_PLUGIN_ROWS)
    monkeypatch.setattr(
        web_server, "_get_dashboard_plugins", lambda force_rescan=False: []
    )
    monkeypatch.setattr(
        _web_server_memory, "_discover_memory_provider_statuses", lambda: []
    )
    monkeypatch.setattr(_cfg_mod, "get_hermes_home", lambda: Path("/tmp/hermes-home"))
    monkeypatch.setattr(
        _cfg_mod, "load_config", lambda: {"dashboard": {"hidden_plugins": []}}
    )
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: rows)
    monkeypatch.setattr(
        plugins_cmd, "_get_current_context_engine", lambda: "compressor"
    )
    monkeypatch.setattr(plugins_cmd, "_get_current_memory_provider", lambda: "")
    monkeypatch.setattr(plugins_cmd, "_discover_context_engines", lambda: [])
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"demo"})
    monkeypatch.setattr(
        plugins_cmd, "_read_manifest", lambda _path: {"provides_tools": []}
    )
    monkeypatch.setattr(
        tools_registry.registry,
        "get_entry",
        lambda _name: SimpleNamespace(check_fn=None),
    )


def test_hub_rebuild_issues_one_network_attempt_then_none(monkeypatch):
    """An unreachable catalog host costs ONE timeout per rebuild (not one per installed plugin),
    and rebuilds inside the failure window cost none at all."""
    unreachable = _UnreachableCatalog()
    monkeypatch.setattr("httpx.get", unreachable)

    _patch_hub_dependencies(monkeypatch)

    payload = _web_server_dashboard._merged_plugins_hub(force_refresh=True)
    assert len(payload["plugins"]) == len(_PLUGIN_ROWS)
    assert all(row["removed_reason"] is None for row in payload["plugins"])
    assert unreachable.attempts == 1

    _web_server_dashboard._invalidate_plugins_hub_cache()
    _web_server_dashboard._merged_plugins_hub(force_refresh=True)
    assert unreachable.attempts == 1  # remembered failure: no second timeout


def test_failed_fetch_is_remembered_only_within_ttl(monkeypatch):
    clock = {"now": 1_000_000.0}
    attempts = {"count": 0}

    def fake_get(url, **kwargs):
        attempts["count"] += 1
        raise OSError("catalog host unreachable")

    monkeypatch.setattr(pc.time, "time", lambda: clock["now"])
    monkeypatch.setattr("httpx.get", fake_get)

    assert pc.fetch_live_catalog() is None
    pc.fetch_live_catalog()  # remembered failure: served without a second attempt
    assert attempts["count"] == 1

    clock["now"] += pc.LIVE_CATALOG_FAILURE_TTL_SECONDS + 1
    assert (
        pc.fetch_live_catalog() is None
    )  # TTL expired: one fresh attempt reaches the network
    assert attempts["count"] == 2


def test_failure_window_prefers_stale_cache_over_in_tree(monkeypatch, tmp_path):
    """Inside the failure window a previously fetched on-disk cache still answers — removals
    published before the outage keep blocking without any network traffic."""
    cache = tmp_path / "cache" / "plugin-catalog.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps({
            "entries": [],
            "removed": [{"name": "pulled-live", "reason": "cve"}],
        })
    )
    monkeypatch.setattr("httpx.get", _UnreachableCatalog())

    data = pc.fetch_live_catalog()
    assert data is not None and data["removed"][0]["name"] == "pulled-live"
    assert pc.fetch_live_catalog() is not None  # still served from the stale cache


def test_batch_annotation_matches_per_plugin_lookup(monkeypatch, tmp_path):
    """The batch resolver returns exactly what per-plugin ``removed_annotation`` returns, across
    name matches, sidecar catalog names and repo URLs."""
    kill_list = [
        pc.RemovedEntry(
            name="evil", repo="https://github.com/x/evil.git", reason="malware"
        ),
        pc.RemovedEntry(name="pulled-live", reason="cve"),
    ]
    monkeypatch.setattr(pc, "load_removed_list", lambda *a, **k: kill_list)
    monkeypatch.setattr(pc, "live_removed_list", lambda: [])

    sidecar_dir = tmp_path / "installed-from-catalog"
    sidecar_dir.mkdir()
    (sidecar_dir / pcc.CATALOG_SIDECAR).write_text(
        json.dumps({
            "catalog_name": "pulled-live",
            "repo": "https://github.com/x/other",
            "sha": "0" * 40,
            "tier": "community",
        })
    )
    repo_dir = tmp_path / "installed-from-repo"
    repo_dir.mkdir()
    (repo_dir / pcc.CATALOG_SIDECAR).write_text(
        json.dumps({
            "catalog_name": "unknown",
            "repo": "https://github.com/x/EVIL/",
            "sha": "0" * 40,
            "tier": "community",
        })
    )
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()

    plugins = [
        ("evil", plain_dir),
        ("catalog-install", sidecar_dir),
        ("repo-install", repo_dir),
        ("fine", plain_dir),
    ]
    batch = pcc.removed_annotation_batch(plugins, pc.resolved_removed_entries())

    for name, dir_path in plugins:
        assert batch[name] == pcc.removed_annotation(name, dir_path)
    assert batch["evil"] == "malware"
    assert batch["catalog-install"] == "cve"
    assert (
        batch["repo-install"] == "malware"
    )  # repo match is .git/trailing-slash insensitive
    assert batch["fine"] is None


def test_hub_rebuild_annotates_from_preloaded_kill_list(monkeypatch, tmp_path):
    """The hub surfaces removal reasons from ONE resolved kill list even when the live catalog
    is unreachable — an installed removed plugin is still reported, at hub-rebuild speed."""
    monkeypatch.setattr("httpx.get", _UnreachableCatalog())
    monkeypatch.setattr(pc, "get_catalog_dir", lambda: tmp_path)
    (tmp_path / "removed.yaml").write_text(
        "removed:\n- name: demo\n  reason: exfiltrated env vars\n"
    )

    _patch_hub_dependencies(monkeypatch)
    payload = _web_server_dashboard._merged_plugins_hub(force_refresh=True)

    by_name = {row["name"]: row["removed_reason"] for row in payload["plugins"]}
    assert by_name["demo"] == "exfiltrated env vars"
    assert by_name["second"] is None


def test_cmd_list_resolves_kill_list_once(monkeypatch, tmp_path, capsys):
    """``hermes plugins list`` resolves the kill list ONCE per listing, not per row: a slow or
    unreachable live catalog must cost one resolution (one network attempt at most) regardless
    of how many plugins are installed."""
    import argparse

    calls = {"load_removed_list": 0, "live_removed_list": 0}
    kill_list = [pc.RemovedEntry(name="pulled-plugin", reason="security review")]

    def counted_load_removed(*args, **kwargs):
        calls["load_removed_list"] += 1
        return kill_list

    def counted_live_removed():
        calls["live_removed_list"] += 1
        return []

    monkeypatch.setattr(pc, "load_removed_list", counted_load_removed)
    monkeypatch.setattr(pc, "live_removed_list", counted_live_removed)
    monkeypatch.setattr(
        plugins_cmd, "_discover_all_plugins",
        lambda: [(f"plugin-{i}", "1.0", "", "user", tmp_path / str(i), f"plugin-{i}")
                 for i in range(3)]
        + [("pulled-plugin", "1.0", "", "user", tmp_path / "pulled", "pulled-plugin")],
    )
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"plugin-0"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_read_install_metadata", lambda: {})
    monkeypatch.setattr(pc, "_live_cache_path", lambda: tmp_path / "cache" / "plugin-catalog.json")

    plugins_cmd.cmd_list(argparse.Namespace(
        enabled=False, user=False, no_bundled=False, plain=False, json=True))

    assert calls["load_removed_list"] == 1
    assert calls["live_removed_list"] == 1
    rows = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row["removed"] for row in rows}
    assert by_name["pulled-plugin"] == "security review"
    assert by_name["plugin-0"] is None
