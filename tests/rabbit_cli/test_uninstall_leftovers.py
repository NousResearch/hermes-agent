"""Uninstall sweeps every leftover: both LaunchAgent label patterns, macOS
Library app-store entries and the XDG cache/data dirs outside RABBIT_HOME
(#62209 — uninstalls reported success yet launchd respawned the gateway and
Library/XDG dirs survived until manually removed)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rabbit_cli import uninstall


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


# ── LaunchAgent sweep: both label patterns ──────────────────────────────


@pytest.fixture
def launch_agents(fake_home, monkeypatch):
    agents = fake_home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    monkeypatch.setattr(
        uninstall, "get_launchd_plist_path",
        lambda: agents / "ai.rabbit.gateway.plist", raising=False,
    )
    # _remove_launchd_gateway imports it from rabbit_cli.gateway at call time.
    import rabbit_cli.gateway as gateway
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: agents / "ai.rabbit.gateway.plist")
    return agents


def _fake_launchctl(monkeypatch, calls):
    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(uninstall.subprocess, "run", fake_run)


def test_remove_launchd_gateway_sweeps_both_label_patterns(launch_agents, monkeypatch):
    current = launch_agents / "ai.rabbit.gateway.plist"
    profiled = launch_agents / "ai.rabbit.gateway-work.plist"
    legacy = launch_agents / "io.nousresearch.rabbit-agent.gateway.plist"
    unrelated = launch_agents / "com.other.agent.plist"
    for p in (current, profiled, legacy, unrelated):
        p.write_text("<plist/>", encoding="utf-8")

    calls: list[list[str]] = []
    _fake_launchctl(monkeypatch, calls)

    assert uninstall._remove_launchd_gateway() is True
    assert not current.exists() and not profiled.exists() and not legacy.exists()
    assert unrelated.exists()  # never touch another app's agent
    labels = {c[-1].rsplit("/", 1)[-1] for c in calls if c[1] == "bootout"}
    assert labels == {"ai.rabbit.gateway", "ai.rabbit.gateway-work",
                      "io.nousresearch.rabbit-agent.gateway"}
    assert all(c[1] == "unload" for c in calls if c[1] == "unload")


def test_remove_launchd_gateway_returns_false_when_no_plists(launch_agents, monkeypatch):
    calls: list[list[str]] = []
    _fake_launchctl(monkeypatch, calls)
    assert uninstall._remove_launchd_gateway() is False
    assert calls == []


# ── macOS Library + XDG leftovers ────────────────────────────────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Library layout")
def test_remove_desktop_app_leftovers_removes_rabbit_library_entries(fake_home):
    library = fake_home / "Library"
    ours = []
    for parent, name in (
        ("Caches", "com.seven0070.rabbit"),
        ("Logs", "Rabbit"),
        ("WebKit", "com.seven0070.rabbit"),
        ("HTTPStorages", "com.seven0070.rabbit"),
        ("Saved Application State", "com.seven0070.rabbit.savedState"),
    ):
        d = library / parent / name
        d.mkdir(parents=True)
        ours.append(d)
    (library / "Caches" / "com.other.vendor").mkdir(parents=True)  # untouched

    removed = uninstall.remove_desktop_app_leftovers(full_uninstall=False)

    assert all(not d.exists() for d in ours)
    assert (library / "Caches" / "com.other.vendor").exists()
    assert len(removed) == len(ours)


def test_xdg_cache_removed_and_data_mode_semantics(fake_home):
    cache = fake_home / ".cache" / "rabbit"
    data = fake_home / ".local" / "share" / "rabbit"
    cache.mkdir(parents=True)
    data.mkdir(parents=True)

    assert uninstall._xdg_leftover_paths(full_uninstall=False) == ([cache], [data])
    assert uninstall._xdg_leftover_paths(full_uninstall=True) == ([cache, data], [])

    removed = uninstall.remove_desktop_app_leftovers(full_uninstall=False)
    assert cache in removed and not cache.exists()
    assert data.exists()  # user data is preserved in keep-data mode


def test_xdg_data_removed_on_full_uninstall(fake_home):
    cache = fake_home / ".cache" / "rabbit"
    data = fake_home / ".local" / "share" / "rabbit"
    cache.mkdir(parents=True)
    data.mkdir(parents=True)

    removed = uninstall.remove_desktop_app_leftovers(full_uninstall=True)

    assert not cache.exists() and not data.exists()
    assert cache in removed and data in removed
