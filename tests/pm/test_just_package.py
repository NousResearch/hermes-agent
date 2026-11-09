"""just is a developer tool: the test-environment install provisions it, a user install never does."""

from __future__ import annotations

import argparse
import importlib

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
