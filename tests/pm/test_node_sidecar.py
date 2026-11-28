"""Plugin-local npm installs retain PM's paired Node and consent policy."""

import json
import os
from pathlib import Path
import shutil
import shlex

import pytest

import pm
from pm.package import InstallError, Runner
from pm.workspace import install_node_sidecar

_REAL_HERMES_HOME = Path.home() / ".hermes"  # Captured before per-test HOME isolation.


def test_no_package_json_never_acquires_npm(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a dependency-free plugin must not acquire npm")
    monkeypatch.setattr(pm, "ensure", unexpected)
    assert install_node_sidecar(tmp_path) is None


@pytest.mark.parametrize("explicit", [False, True])
def test_acquisition_receives_consent_and_reports_refusal(tmp_path, monkeypatch, explicit):
    import importlib

    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(importlib.import_module("pm.install"), "lazy_installs_allowed", lambda: True)
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
    if any(Path(executable).absolute().is_relative_to(_REAL_HERMES_HOME)
           for executable in (npm, node)):
        pytest.skip("requires npm and Node outside the real Hermes home")
    # npm's real JS entrypoint uses /usr/bin/env node. Its paired Node lives
    # in a different PATH entry, just as the two PM packages do.
    npm_cli = next(iter(Path(npm).resolve().parent.parent.glob("lib/npm*/bin/npm-cli.js")), Path(npm).resolve())
    from pm import paths
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import current_target, tree_digest

    store, target = tmp_path / "tools", current_target()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    packages = [get_package(name) for name in ("node", "npm")]
    binaries = {}
    for package in packages:
        entry = store / package.store_entry("test", target)
        binary = package.binary(entry, target)
        assert binary is not None
        binary.parent.mkdir(parents=True)
        binaries[package.name] = binary
    wrapper = binaries["npm"]
    wrapper.write_text(f'#!/bin/sh\nexec node {shlex.quote(str(npm_cli))} "$@"\n')
    wrapper.chmod(0o755)
    binaries["node"].symlink_to(node)
    lockfile = Lockfile(paths.lockfile_path())
    facts = Facts(paths.facts_path())
    for package in packages:
        entry = store / package.store_entry("test", target)
        lockfile.set_pin(package.name, "test", {})
        facts.record(package.name, "test", entry.name, package.env(entry, target), store,
                     target=target, digest=tree_digest(entry))
    lockfile.save()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    context = Runner("npm", pm.env_for("npm"))
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
    (plugin / "package.json").write_text("{malformed")
    reason = install_node_sidecar(plugin, explicit=True)
    assert "exited" in reason and "JSON" in reason



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
