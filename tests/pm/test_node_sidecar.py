"""Plugin-local npm installs retain PM's paired Node and consent policy."""

import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess

import pytest

import pm
from pm.package import InstallError, Runner, compose_env
from pm.workspace import install_node_sidecar


def test_no_package_json_never_acquires_npm(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a dependency-free plugin must not acquire npm")
    monkeypatch.setattr(pm, "ensure", unexpected)
    assert install_node_sidecar(tmp_path) is None


@pytest.mark.parametrize("explicit", [False, True])
def test_acquisition_receives_consent_and_reports_refusal(tmp_path, monkeypatch, explicit):
    import importlib

    monkeypatch.setattr(importlib.import_module("pm.ensure"), "lazy_installs_allowed", lambda: True)
    (tmp_path / "package.json").write_text('{}')
    calls = []
    def refused(name, **kwargs):
        calls.append((name, kwargs))
        raise InstallError(name, "acquisition refused")
    monkeypatch.setattr(pm, "ensure", refused)
    reason = install_node_sidecar(tmp_path, explicit=explicit)
    assert "acquisition refused" in reason
    assert calls == [("npm", {"explicit": explicit})]


@pytest.mark.platforms("posix")
def test_real_npm_uses_paired_node_with_empty_ambient_path(tmp_path, monkeypatch):
    npm, node = shutil.which("npm"), shutil.which("node")
    if not npm or not node:
        pytest.skip("npm and node are required")
    # npm's real JS entrypoint uses /usr/bin/env node. Its paired Node lives
    # in a different PATH entry, just as the two PM packages do.
    npm_cli = next(iter(Path(npm).resolve().parent.parent.glob("lib/npm*/bin/npm-cli.js")), Path(npm).resolve())
    npm_bin, node_bin = tmp_path / "npm/bin", tmp_path / "node/bin"
    npm_bin.mkdir(parents=True)
    node_bin.mkdir(parents=True)
    wrapper = npm_bin / "npm"
    wrapper.write_text(f'#!/bin/sh\nexec node {shlex.quote(str(npm_cli))} "$@"\n')
    wrapper.chmod(0o755)
    (node_bin / "node").symlink_to(node)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    context = Runner("npm", compose_env([
        {"PATH": [str(node_bin)]}, {"PATH": [str(npm_bin)]},
    ]))
    calls = []
    def acquire(name, **kwargs):
        calls.append((name, kwargs))
        return context
    monkeypatch.setattr(pm, "ensure", acquire)
    dependency, plugin = tmp_path / "side-dep", tmp_path / "plugin"
    dependency.mkdir()
    plugin.mkdir()
    (dependency / "package.json").write_text(json.dumps({"name": "side-dep", "version": "1.0.0", "main": "index.js"}))
    (dependency / "index.js").write_text("module.exports = 'isolated';")
    manifest = {"name": "plugin", "version": "1.0.0", "dependencies": {"side-dep": "file:../side-dep"}}
    (plugin / "package.json").write_text(json.dumps(manifest))
    ambient = dict(os.environ)
    assert install_node_sidecar(plugin, explicit=True) is None
    lock = (plugin / "package-lock.json").read_bytes()
    # ci must remove an extraneous directory; another `install` would not
    # reliably prove that the lockfile selected the clean-install path.
    stale = plugin / "node_modules/stale-file"
    stale.write_text("remove me")
    assert install_node_sidecar(plugin, explicit=True) is None
    assert not stale.exists()
    assert (plugin / "package-lock.json").read_bytes() == lock
    result = context.run(["node", "-e", "console.log(require('side-dep'))"], cwd=plugin, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
    assert not (tmp_path / "node_modules").exists()
    assert dict(os.environ) == ambient
    assert calls == [("npm", {"explicit": True})] * 2


def test_npm_process_failure_is_reported(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text('{}')
    binary = tmp_path / ("npm.cmd" if os.name == "nt" else "npm")
    binary.write_text("process boundary fixture")
    binary.chmod(0o755)
    class FailedRunner:
        env = {"PATH": str(tmp_path)}
        def run(self, command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "ERESOLVE dependency conflict")
    monkeypatch.setattr(pm, "ensure", lambda *args, **kwargs: FailedRunner())
    reason = install_node_sidecar(tmp_path, explicit=True)
    assert "exited 1" in reason and "ERESOLVE" in reason


def test_on_demand_sidecar_install_respects_lazy_refusal(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text('{}')
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    binary = tmp_path / ("npm.cmd" if os.name == "nt" else "npm")
    binary.write_text("process boundary fixture")
    binary.chmod(0o755)
    class InstalledRunner:
        env = {"PATH": str(tmp_path)}
        def run(self, *args, **kwargs):
            pytest.fail("installed npm must not mutate sidecars when on-demand installs are disabled")
    monkeypatch.setattr(pm, "ensure", lambda *args, **kwargs: InstalledRunner())
    reason = install_node_sidecar(tmp_path)
    assert reason and "disabled" in reason