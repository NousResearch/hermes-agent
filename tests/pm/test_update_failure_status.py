"""A failed update lookup is not a current result or permission to apply."""
import importlib
from argparse import Namespace

import pytest

from pm import cli, paths, registry
from pm.lock import Lockfile
from pm.package import Package
from pm.store import current_target


class UpdateFixture(Package):
    def __init__(self, name, versions):
        self.name = name
        self.versions = versions
        self.lookups = 0

    def missing_reason(self, target):
        return None if target == current_target() else "not a fixture target"

    def latest_versions(self, target, locked=None):
        self.lookups += 1
        if isinstance(self.versions, Exception):
            raise self.versions
        return self.versions


def prepare(tmp_path, monkeypatch, packages):
    lock = Lockfile(tmp_path / "lock.json")
    for package in packages:
        lock.set_pin(package.name, "1.0", {})
        monkeypatch.setitem(registry._packages, package.name, package)
    lock.save()
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))

    def forbidden(*args, **kwargs):
        pytest.fail("lookup-only result reached a mutation or dependency refresh")

    for name in ("_pin_artifacts", "_install_names", "_run_live", "lock_project"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(importlib.import_module("pm.ensure"), "sync_venv", forbidden)
    return lock


@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize("mixed", [True, False])
def test_failed_resolution_stops_every_apply_path(tmp_path, monkeypatch, capsys, check, mixed):
    failed = UpdateFixture("failed-lookup", TimeoutError("fixture index unavailable"))
    healthy = UpdateFixture("healthy-lookup", ["2.0"])
    packages = [failed, healthy] if mixed else [failed]
    lock = prepare(tmp_path, monkeypatch, packages)
    before = lock.path.read_bytes()
    args = Namespace(names=[p.name for p in packages], target=None, check=check, uv=True, npm=True)
    assert cli.cmd_update(args) == 1
    assert "fixture index unavailable" in capsys.readouterr().out
    assert lock.path.read_bytes() == before
    assert not (tmp_path / "tools").exists()
    if mixed:
        assert healthy.lookups == 1


@pytest.mark.parametrize("versions", [["1.0"], []])
def test_current_and_manual_results_are_successful_without_writes(tmp_path, monkeypatch, versions):
    package = UpdateFixture("no-update", versions)
    lock = prepare(tmp_path, monkeypatch, [package])
    before = lock.path.read_bytes()
    for check in (True, False):
        args = Namespace(names=[package.name], target=None, check=check, uv=False, npm=False)
        assert cli.cmd_update(args) == 0
    assert lock.path.read_bytes() == before
    assert not (tmp_path / "tools").exists()
