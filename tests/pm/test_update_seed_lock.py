"""Regression for #136694: an update whose recorded seed lock vanished must not crash."""

from __future__ import annotations

from pathlib import Path

import pytest

import pm.environment
import pm.environments
import pm.lock
import pm.native_build
import pm.plugins_state
import pm.workspace
from pm.packages import Venv


class _Stop(Exception):
    pass


class _Environment:
    env: dict = {}

    def create(self):
        pass


def _apply_seed(tmp_path, monkeypatch, recorded: Path) -> Path:
    project = tmp_path / "checkout"
    project.mkdir()
    (project / "uv.lock").write_text("checkout lock")
    seen = {}

    class Facts:
        def __init__(self, path, *, strict=False):
            pass

        def get(self, name):
            return {"environment": str(recorded.parent.parent / "venv"), "resolved_lock": str(recorded)}

    def lock_and_sync(members, extras, *, seed_lock, **kwargs):
        seen["seed"] = seed_lock
        seed_lock.read_bytes()  # what the real lock_and_sync does first
        raise _Stop

    monkeypatch.setattr(Venv, "project_root", lambda self: project)
    monkeypatch.setattr(pm.environments, "install_state_dir", lambda p: tmp_path / "state")
    monkeypatch.setattr(pm.environments, "runtime_facts_path", lambda p: tmp_path / "facts.json")
    monkeypatch.setattr(pm.environment, "managed_environment", lambda *a, **kw: _Environment())
    monkeypatch.setattr(pm.native_build, "source_build_environment", lambda p: {})
    monkeypatch.setattr(pm.plugins_state, "enabled_plugins_ordered", lambda **kw: [])
    monkeypatch.setattr(pm.lock, "Facts", Facts)
    monkeypatch.setattr(pm.workspace, "lock_and_sync", lock_and_sync)
    with pytest.raises(_Stop):
        Venv().apply([], plugin_dirs=[tmp_path / "plugin"])
    return seen["seed"]


def test_update_falls_back_to_checkout_lock_when_recorded_seed_is_missing(tmp_path, monkeypatch):
    missing = tmp_path / "state" / "environments" / "old" / "workspace" / "uv.lock"
    assert _apply_seed(tmp_path, monkeypatch, missing) == tmp_path / "checkout" / "uv.lock"


def test_update_keeps_recorded_seed_when_present(tmp_path, monkeypatch):
    recorded = tmp_path / "state" / "environments" / "old" / "workspace" / "uv.lock"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("recorded lock")
    assert _apply_seed(tmp_path, monkeypatch, recorded) == recorded
