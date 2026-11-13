"""Plugin-local pnpm installs retain PM's paired Node and consent policy."""

import json
import os
from pathlib import Path
import shlex
import shutil

import pytest

import pm
from pm.package import InstallError, Runner, compose_env
from pm.workspace import install_node_sidecar

_REAL_HERMES_HOME = Path.home() / ".hermes"  # Captured before per-test HOME isolation.


def test_no_package_json_never_acquires_pnpm(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("a dependency-free plugin must not acquire pnpm")
    monkeypatch.setattr(pm, "ensure", unexpected)
    assert install_node_sidecar(tmp_path) is None


@pytest.mark.parametrize("explicit", [False, True])
def test_acquisition_receives_consent_and_reports_refusal(tmp_path, monkeypatch, explicit):
    import importlib

    monkeypatch.setattr(importlib.import_module("pm.install"), "lazy_installs_allowed", lambda: True)
    (tmp_path / "package.json").write_text('{}')
    calls = []
    def refused(name, **kwargs):
        calls.append((name, kwargs))
        raise InstallError(name, "acquisition refused")
    monkeypatch.setattr(pm, "ensure", refused)
    reason = install_node_sidecar(tmp_path, explicit=explicit)
    assert "acquisition refused" in reason
    assert calls == [("pnpm", {"explicit": explicit})]


@pytest.mark.platforms("posix")
def test_real_pnpm_uses_paired_node_with_empty_ambient_path(tmp_path, monkeypatch):
    pnpm, node = shutil.which("pnpm"), shutil.which("node")
    if not pnpm or not node:
        pytest.skip("pnpm and node are required")
    if any(Path(executable).absolute().is_relative_to(_REAL_HERMES_HOME)
           for executable in (pnpm, node)):
        pytest.skip("requires pnpm and Node outside the real Hermes home")
    # pnpm resolves lifecycle scripts through Node; its paired Node lives in a
    # different PATH entry, just as the two PM packages do.
    pnpm_bin, node_bin = tmp_path / "pnpm/bin", tmp_path / "node/bin"
    pnpm_bin.mkdir(parents=True)
    node_bin.mkdir(parents=True)
    wrapper = pnpm_bin / "pnpm"
    wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(str(pnpm))} "$@"\n')
    wrapper.chmod(0o755)
    (node_bin / "node").symlink_to(node)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    context = Runner("pnpm", compose_env([
        {"PATH": [str(node_bin)]}, {"PATH": [str(pnpm_bin)]},
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
    (dependency / "index.js").write_text("module.exports = 'isolated';\n")
    manifest = {"name": "plugin", "version": "1.0.0", "dependencies": {"side-dep": "file:../side-dep"}}
    (plugin / "package.json").write_text(json.dumps(manifest))
    ambient = dict(os.environ)
    assert install_node_sidecar(plugin, explicit=True) is None
    assert (plugin / "pnpm-lock.yaml").is_file()
    result = context.run(["node", "-e", "console.log(require('side-dep'))"], cwd=plugin, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"
    assert not (tmp_path / "node_modules").exists()
    assert dict(os.environ) == ambient
    assert calls == [("pnpm", {"explicit": True})]
    (plugin / "package.json").write_text("{malformed")
    reason = install_node_sidecar(plugin, explicit=True)
    assert reason is not None and "exited" in reason


def test_on_demand_sidecar_install_respects_lazy_refusal(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text('{}')
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    binary = tmp_path / ("pnpm.cmd" if os.name == "nt" else "pnpm")
    binary.write_text("process boundary fixture")
    binary.chmod(0o755)
    class InstalledRunner:
        env = {"PATH": str(tmp_path)}
        def run(self, *args, **kwargs):
            pytest.fail("installed pnpm must not mutate sidecars when on-demand installs are disabled")
    monkeypatch.setattr(pm, "ensure", lambda *args, **kwargs: InstalledRunner())
    reason = install_node_sidecar(tmp_path)
    assert reason and "disabled" in reason
