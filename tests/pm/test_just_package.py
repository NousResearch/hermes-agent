"""just is a developer tool: the test-environment install provisions it, a user install never does."""

from __future__ import annotations

import argparse
import importlib
import json

import pytest

import pm.cli
from pm.lock import Lockfile
from pm.paths import lockfile_path
from pm.registry import get_package, source_install_packages


@pytest.fixture()
def requested(monkeypatch):
    names: list[str] = []

    def fake_install_names(batch, target=None, *, verify=True):
        names.extend(batch)
        return 0

    monkeypatch.setattr(pm.cli, "_install_names", fake_install_names)
    monkeypatch.setattr(pm.cli, "_install_python_environments", lambda *args, **kwargs: 0)
    monkeypatch.setattr(importlib.import_module("pm.install"), "activate", lambda **kwargs: [])
    monkeypatch.setattr(importlib.import_module("pm.environments"), "record_activation_inputs",
                        lambda *args, **kwargs: None)
    return names


@pytest.mark.parametrize("test_environment,provisioned", [(None, False), ("", True)])
def test_only_the_test_environment_install_requests_just(requested, test_environment, provisioned):
    lockfile = Lockfile(lockfile_path())
    assert "just" in lockfile.names()
    assert "just" not in source_install_packages(lockfile.names())
    args = argparse.Namespace(names=None, tools_only=False, test_environment=test_environment)
    assert pm.cli.cmd_install(args) == 0
    assert ("just" in requested) is provisioned


def test_a_lockfile_without_just_still_installs_without_it(requested, tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps({"schema": 1, "packages": {
        name: Lockfile(lockfile_path())._packages[name] for name in ("python", "uv")}}), encoding="utf-8")
    monkeypatch.setattr(importlib.import_module("pm.install"), "_lockfile", lambda: Lockfile(lock_path))
    monkeypatch.setattr(pm.cli, "_lockfile", lambda: Lockfile(lock_path))
    args = argparse.Namespace(names=None, tools_only=False, test_environment="")
    assert pm.cli.cmd_install(args) == 0
    assert "just" not in requested


def test_every_locked_just_artifact_is_the_url_fetch_url_builds():
    lockfile = Lockfile(lockfile_path())
    package = get_package("just")
    version = lockfile.version("just")
    assert version
    locked = {target: [row["url"] for row in lockfile.artifacts("just", target)]
              for target in lockfile.pinned_artifacts("just")}
    assert locked
    for target, urls in locked.items():
        assert urls == package.fetch_urls(version, target), target


def test_a_desktop_bundle_never_stages_just():
    from scripts.bundles import native

    assert "just" in Lockfile(lockfile_path()).names()
    assert "just" not in native._bundle_package_names()


def _fake_binary(tmp_path, name: str = "just"):
    """A file whose header no architecture check can read, and that exits 1 when run."""
    entry = tmp_path / "entry"
    entry.mkdir()
    binary = entry / name
    binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)
    return entry


def test_termux_just_is_checked_by_file_evidence_and_never_run(tmp_path, monkeypatch):
    # On Termux the static musl just is checked from its file only; running it is
    # left to the device. The probe would fail (the fake exits 1) if it ran.
    import pm.packages

    monkeypatch.setattr(pm.packages, "current_target", lambda: "linux-arm64-bionic")
    entry = _fake_binary(tmp_path)
    assert get_package("just").verify(entry, "linux-arm64-bionic") == ""
    assert get_package("just").verify(tmp_path / "missing", "linux-arm64-bionic") != ""

