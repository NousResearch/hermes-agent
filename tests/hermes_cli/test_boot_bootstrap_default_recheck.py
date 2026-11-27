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
