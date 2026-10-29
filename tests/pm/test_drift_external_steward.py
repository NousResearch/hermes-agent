"""An external steward's PATH tools are its install of record: drift must not
demand store copies a sealed tree cannot place.

Regression: on a Nix-managed install the first lazy tool install creates
facts.json, after which PM vouches for the whole lock closure — and every
steward-shipped PATH tool (ffmpeg, node, npm, ripgrep) started reporting
"not installed or outdated" on every boot, with a remedy only the builder
can perform. See ``pm.install.drift``."""

import os
from pathlib import Path

import pytest

from pm import install, paths, registry
from pm.lock import Lockfile
from pm.packages import BinaryPackage
from pm.store import current_target


@pytest.fixture
def lock_package(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    (tmp_path / "store").mkdir()
    # A lazy install already happened: PM now vouches for the closure.
    (tmp_path / "store" / "facts.json").write_text('{"schema": 1, "packages": {}}')
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    monkeypatch.setattr(paths, "install_root", lambda: tmp_path / "tree")
    monkeypatch.setattr(registry, "_packages", {})
    package = BinaryPackage()
    package.name = "tool"
    package.binary_rel = {"posix": "bin/tool", "win32": "tool.exe"}
    registry._packages["tool"] = package
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin("tool", "1.0", {"any": {"url": "https://example.invalid/tool",
                                         "sha256": "0" * 64}})
    lock.save()
    return tmp_path


def _steward_ships(monkeypatch, tmp_path, package) -> None:
    """The steward's PATH carries the tool's binary (its exact target name)."""
    binary = package.binary(Path("."), current_target())
    assert binary is not None
    shipped = tmp_path / "steward-bin" / binary.name
    shipped.parent.mkdir(parents=True, exist_ok=True)
    shipped.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(shipped, 0o755)
    monkeypatch.setenv("PATH", str(shipped.parent))


def _stamp(tmp_path, mechanism: str) -> None:
    (tmp_path / "tree").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tree" / "install-stamp.json").write_text(
        f'{{"updateMechanism": "{mechanism}"}}')


def test_steward_path_tool_is_not_drift(lock_package, monkeypatch):
    _steward_ships(monkeypatch, lock_package, registry._packages["tool"])
    _stamp(lock_package, "external")
    assert install.check() == []


def test_path_tool_still_drifts_when_self_managed(lock_package, monkeypatch):
    # The PATH skip is a stewardship contract, not PATH amnesty: a
    # self-managed install still owes the store copy the updater re-verifies.
    _steward_ships(monkeypatch, lock_package, registry._packages["tool"])
    _stamp(lock_package, "self")
    assert install.check() == ["tool: not installed or outdated"]


def test_steward_tool_off_path_still_drifts(lock_package, monkeypatch):
    # Silence covers only tools the steward actually ships.
    monkeypatch.setenv("PATH", str(lock_package / "empty-bin"))
    _stamp(lock_package, "external")
    assert install.check() == ["tool: not installed or outdated"]
