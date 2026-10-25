"""The staged-uv bootstrap shortcut must only run on the pinned interpreter.

A historical-updater takeover reaches ``runtime_python()`` on the old app venv
Python (3.11/3.12) while the pinned uv is already staged in the tool store
without facts. Handing ``sys.executable`` to the PM venv builder then fails
``uv sync`` on ``requires-python ==3.14.*`` on every retry, so the update never
completes (#125565). The shortcut is therefore gated on the lockfile's pinned
python (major, minor).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from pm.runtime import _pinned_minor, runtime_python


def _stage_uv(store: Path) -> Path:
    from pm.packages import Uv
    from pm.store import current_target

    target = current_target()
    entry = store / Uv().store_entry("bootstrap-fixture", target)
    binary = Uv().binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    return binary


def _write_lock(tmp_path: Path, python_version: str | None) -> Path:
    packages = {
        "uv": {"version": "bootstrap-fixture", "artifacts": {"any": {
            "url": "https://must-not-fetch.invalid/uv.tar.gz", "sha256": "a" * 64,
        }}},
    }
    if python_version is not None:
        packages["python"] = {"version": python_version, "artifacts": {"any": {
            "url": "https://must-not-fetch.invalid/python.tar.gz", "sha256": "b" * 64,
        }}}
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": 1, "packages": packages}))
    return lock


@pytest.fixture
def bootstrap_harness(tmp_path, monkeypatch):
    """Staged uv without facts, recorded toolchain/prepare calls."""
    from pm import runtime

    store = tmp_path / "tools"
    staged = _stage_uv(store)
    lock = _write_lock(tmp_path, "3.11.0+old-venv")

    calls: dict[str, object] = {"toolchain": None, "prepare": None}
    sentinel_uv, sentinel_python = tmp_path / "sentinel-uv", tmp_path / "sentinel-python"

    def fake_toolchain(*, realize: bool = True, explicit: bool = False):
        if not realize:
            return None  # no uv/python facts in the store
        calls["toolchain"] = explicit
        return sentinel_uv, sentinel_python

    def fake_prepare(uv, python, destination, **kwargs):
        calls["prepare"] = (uv, python)
        return tmp_path / "prepared"

    monkeypatch.setattr("pm.paths.lockfile_path", lambda: lock)
    monkeypatch.setattr("pm.paths.store_root", lambda: store)
    monkeypatch.setattr("pm._uv._toolchain", fake_toolchain)
    monkeypatch.setattr(runtime, "prepare_runtime", fake_prepare)
    monkeypatch.setattr(runtime, "_resident_runtime", lambda: None)
    monkeypatch.setattr(runtime, "is_runtime", lambda: False)
    return type("Harness", (), {
        "staged": staged, "lock": lock, "calls": calls,
        "sentinel": (sentinel_uv, sentinel_python),
        "runtime": runtime,
    })()


@pytest.mark.platforms("posix")
def test_staged_uv_shortcut_falls_through_on_mismatched_minor(bootstrap_harness):
    """Old-venv takeover: staged uv + wrong interpreter must not use sys.executable."""
    harness = bootstrap_harness
    assert harness.staged.is_file()
    result = runtime_python()
    assert result.name == "prepared"
    assert harness.calls["toolchain"] is True, "mismatched minor must provision explicitly"
    assert harness.calls["prepare"] == harness.sentinel
    assert harness.calls["prepare"][1] != Path(sys.executable)


@pytest.mark.platforms("posix")
def test_staged_uv_shortcut_used_on_matching_minor(bootstrap_harness):
    """Bootstrap Python of the pinned minor keeps the TLS-first shortcut."""
    harness = bootstrap_harness
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    _write_lock(harness.lock.parent, f"{here}.0+same-minor")
    result = runtime_python()
    assert result.name == "prepared"
    assert harness.calls["toolchain"] is None, "matched minor must not provision"
    assert harness.calls["prepare"] == (harness.staged, Path(sys.executable))


@pytest.mark.platforms("posix")
def test_staged_uv_without_python_pin_provisions(bootstrap_harness):
    """No python pin in the lock: cannot vouch for sys.executable, provision."""
    harness = bootstrap_harness
    _write_lock(harness.lock.parent, None)
    runtime_python()
    assert harness.calls["toolchain"] is True
    assert harness.calls["prepare"] == harness.sentinel


def test_pinned_minor_parses_lock_spelling():
    assert _pinned_minor("3.14.7+2026090x") == (3, 14)
    assert _pinned_minor("3.14.0") == (3, 14)
    assert _pinned_minor(None) is None
    assert _pinned_minor("") is None
    assert _pinned_minor("unparsable") is None
    assert _pinned_minor("3.four.1") is None
