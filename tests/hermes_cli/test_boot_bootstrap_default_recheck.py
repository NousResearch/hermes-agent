"""A revision already maintained still re-asks the default-plugin step on every start: the live
catalog can add a default without a new revision. Other steps stay one attempt per revision."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import hermes_cli.plugin_catalog as pc
import pm.plugins_state
from hermes_cli import boot_bootstrap, default_plugins, post_update


@pytest.fixture
def install(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    tree = tmp_path / "payload"
    tree.mkdir()
    (tree / "install-stamp.json").write_text(
        json.dumps({"commit": "aaaa1111", "payload": "full", "updateMechanism": "electron-updater"}), encoding="utf-8")
    return tree


def test_settled_revision_rechecks_default_plugins_on_every_start(install, monkeypatch):
    ran = {"default": 0, "other": 0}

    def defaults():
        ran["default"] += 1
        return {"ok": True}

    def other():
        ran["other"] += 1
        return {"ok": True}

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS",
                        (("other_step", other), ("install_default_plugins", defaults)))
    boot_bootstrap.run_boot_bootstrap(install)  # first start of the revision: every step once
    record = boot_bootstrap.record_path(install).read_bytes()
    assert ran == {"default": 1, "other": 1}

    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}
    assert ran == {"default": 3, "other": 1}, "only the default-plugin step reruns on a settled revision"
    assert boot_bootstrap.record_path(install).read_bytes() == record, "a recheck is not recorded"


def test_a_failing_recheck_never_stops_a_start(install, monkeypatch):
    calls = []

    def defaults():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("disk full")
        return {"ok": True}

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("install_default_plugins", defaults),))
    boot_bootstrap.run_boot_bootstrap(install)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}
    assert len(calls) == 2


def test_cli_recheck_never_fetches_network_when_cache_missing_or_expired(install, tmp_path, monkeypatch):
    """When cache/plugin-catalog.json is missing or past TTL, step_install_default_plugins makes zero
    network requests and spawns the detached installer child immediately."""
    home = tmp_path / ".hermes"
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(default_plugins, "_identity", lambda: "aaaa1111")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(pc, "fetch_live_catalog", lambda **_k: pytest.fail("CLI boot recheck must not touch network"))
    monkeypatch.setattr(pc, "_download_live_catalog", lambda *_a, **_k: pytest.fail("HTTP fetch during CLI recheck"))

    spawned: list = []
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: spawned.append(h) or True)

    # Settled marker exists for this revision, but cache/plugin-catalog.json is missing.
    default_plugins._write_settled(
        home / "cache" / "default-plugins.settled",
        "aaaa1111",
        default_plugins._defaults_hash([]),
        default_names=[],
        live_cache_sig="stale:1",
    )
    res_missing = default_plugins.step_install_default_plugins()
    assert res_missing == {"ok": True, "spawned": True, "owed": True}
    assert len(spawned) == 1

    # Now place an EXPIRED cache/plugin-catalog.json (> LIVE_CATALOG_TTL_SECONDS old).
    cache_file = home / "cache" / "plugin-catalog.json"
    cache_file.write_text(json.dumps({"entries": [], "removed": []}), encoding="utf-8")
    expired = time.time() - pc.LIVE_CATALOG_TTL_SECONDS - 60
    os.utime(cache_file, (expired, expired))

    res_expired = default_plugins.step_install_default_plugins()
    assert res_expired == {"ok": True, "spawned": True, "owed": True}
    assert len(spawned) == 2


def test_cli_recheck_skips_remote_config_backend_without_network_or_exit(install, tmp_path, monkeypatch):
    """With HERMES_CONFIG_BACKEND=remote, step_install_default_plugins skips before fetch_policy
    and _recheck_default_plugins neither touches the network nor exits on ConfigBackendUnavailable."""
    from hermes_cli.config_backend import ConfigBackendUnavailable

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "remote")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home])
    monkeypatch.setattr(
        default_plugins,
        "fetch_policy",
        lambda _h: (_ for _ in ()).throw(ConfigBackendUnavailable("remote config plane unreachable")),
    )
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: pytest.fail("must not spawn for remote backend"))

    assert default_plugins.step_install_default_plugins() == {"ok": True, "skipped": "remote-config-backend"}

    # Boot bootstrap recheck on an already-maintained revision also succeeds without exiting.
    monkeypatch.setattr(
        post_update,
        "BOOT_HOME_STEPS",
        (("install_default_plugins", default_plugins.step_install_default_plugins),),
    )
    boot_bootstrap.run_boot_bootstrap(install)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}


def test_recheck_catches_config_backend_unavailable_system_exit(install, monkeypatch):
    """ConfigBackendUnavailable inherits SystemExit; _recheck_default_plugins must never abort CLI startup."""
    from hermes_cli.config_backend import ConfigBackendUnavailable

    calls = []

    def defaults():
        calls.append(1)
        if len(calls) > 1:
            raise ConfigBackendUnavailable("config plane down")
        return {"ok": True}

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("install_default_plugins", defaults),))
    boot_bootstrap.run_boot_bootstrap(install)
    assert boot_bootstrap.run_boot_bootstrap(install) == {"home": "skipped"}
    assert len(calls) == 2


def test_settled_marker_fallback_spawns_for_unrecorded_home(install, tmp_path, monkeypatch):
    """When _fast_settled_hit returns False (e.g. a profile re-enabled lazy installs and never recorded
    the defaults), the settled-hash fallback must still check _pending_names and spawn_install()."""
    home = tmp_path / ".hermes"
    profile_home = home / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")

    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delenv("HERMES_CONFIG_BACKEND", raising=False)
    monkeypatch.setattr(default_plugins, "_identity", lambda: "aaaa1111")
    monkeypatch.setattr(pm.plugins_state, "dependency_homes", lambda: [home, profile_home])
    monkeypatch.setattr(pc, "get_catalog_dir", lambda: tmp_path / "empty-in-tree")
    monkeypatch.setattr(pc, "_in_tree_catalog_time", None)

    entry = {
        "name": "hermes-security",
        "repo": "https://github.com/NousResearch/hermes-official-plugins",
        "sha": "a" * 40,
        "description": "security plugin",
        "maintainer": "NousResearch",
        "tier": "official",
        "default": True,
    }
    cache_file = home / "cache" / "plugin-catalog.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"generated_at": "2099-01-01T00:00:00+00:00", "entries": [entry], "removed": []}),
        encoding="utf-8",
    )

    # Default home already recorded the plugin, but profile_home has neither installed nor recorded it.
    (home / "config.yaml").write_text("_default_plugins_installed:\n  - hermes-security\n", encoding="utf-8")
    cached = default_plugins._cached_default_entries()
    assert cached is not None and [e.name for e in cached] == ["hermes-security"]
    default_plugins._write_settled(
        home / "cache" / "default-plugins.settled",
        "aaaa1111",
        default_plugins._defaults_hash(cached),
        default_names=["hermes-security"],
        live_cache_sig=default_plugins._live_cache_signature(),
    )

    spawned: list = []
    monkeypatch.setattr(default_plugins, "spawn_install", lambda h=None: spawned.append(h) or True)

    assert default_plugins._fast_settled_hit(home, [home, profile_home]) is False
    res = default_plugins.step_install_default_plugins()
    assert res == {"ok": True, "spawned": True, "owed": True}
    assert len(spawned) == 1
