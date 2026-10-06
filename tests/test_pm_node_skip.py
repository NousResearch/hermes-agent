"""Regression for #130574: skipping the managed Node runtime when the host provides one.

Behavioral contracts (no snapshots):
- ``node``/``npm`` are declinable defaults traveling as one ecosystem:
  ``--without node`` implies ``--without npm``; naming either opts both back in.
- ``hermes pm install --without node`` verifies the host toolchain against
  ``package.json`` engines and refuses when it cannot satisfy them.
- Host mode resolves node-ecosystem commands through the host PATH instead of
  the managed entry (``find_node_executable`` / ``with_hermes_node_path`` /
  source builds / terminal PATH entries).
- ``hermes doctor`` warns when the managed Node shadows a host Node, and
  reports host mode (usable or not) instead of the managed entry.
- The installers expose the opt-out (``install.sh --skip-node``,
  ``install.ps1 -SkipNode``) as PM's persisted ``--without node``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

import hermes_constants
import pm.cli
from pm.defaults import default_packages, expand_declined, expand_opt_in

pytestmark = pytest.mark.platforms("posix")

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- defaults: node/npm travel as one ecosystem ---

def test_node_and_npm_are_declinable_defaults():
    from pm.defaults import default_package_names

    allowed = default_package_names()
    assert {"node", "npm"} <= set(allowed)


def test_without_node_implies_without_npm():
    assert expand_declined(["node"]) == ["node", "npm"]
    assert expand_declined(["npm"]) == ["npm"]
    assert expand_declined(["agent-browser", "node"]) == ["agent-browser", "node", "npm"]


def test_naming_either_half_opts_the_ecosystem_back_in():
    assert expand_opt_in(["node"]) == ["node", "npm"]
    assert expand_opt_in(["npm"]) == ["node", "npm"]
    assert expand_opt_in(["agent-browser"]) == ["agent-browser"]


def test_legacy_node_only_record_still_skips_npm():
    names = pm.cli._lockfile().names()
    selected = default_packages(names, target="linux-x64", declined_names=frozenset({"node"}))
    assert "node" not in selected and "npm" not in selected
    selected = default_packages(names, target="linux-x64", declined_names=frozenset())
    assert {"node", "npm"} <= set(selected)


# --- cmd_install: --without node verifies the host and records both ---

@pytest.fixture()
def install_spy(monkeypatch):
    calls: dict = {"names": []}

    def fake_install_names(names, target=None, *, verify=True):
        calls["names"].extend(names)
        return 0

    def fake_sync_venv(extras=None, **kwargs):
        return None

    def fake_activate(**kwargs):
        return []

    monkeypatch.setattr(pm.cli, "_install_names", fake_install_names)
    monkeypatch.setattr(importlib.import_module("pm.install"), "sync_venv", fake_sync_venv)
    monkeypatch.setattr(importlib.import_module("pm.install"), "activate", fake_activate)
    return calls


def test_without_node_records_the_ecosystem_and_installs_nothing_managed(install_spy, monkeypatch, capsys):
    from pm.defaults import declined

    monkeypatch.setattr(hermes_constants, "verify_host_node",
                        lambda: (True, "host node v26.7.0, host npm 12.0.2"))
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["node"])) == 0
    # The opt-out is persisted (later bare installs and `hermes update` keep it)...
    assert declined() == frozenset({"node", "npm"})
    # ...so this install carries neither managed entry.
    assert "node" not in install_spy["names"] and "npm" not in install_spy["names"]
    assert "using host Node" in capsys.readouterr().out


def test_without_node_refuses_an_unsatisfiable_host(install_spy, monkeypatch, capsys):
    from pm.defaults import declined

    monkeypatch.setattr(hermes_constants, "verify_host_node",
                        lambda: (False, "host node v20.0.0 does not satisfy engines.node"))
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["node"])) == 1
    assert declined() == frozenset()
    assert install_spy["names"] == []
    assert "--without node needs a usable host Node" in capsys.readouterr().out


def test_explicit_node_install_opts_the_ecosystem_back_in(install_spy):
    from pm.defaults import declined, record_declined

    record_declined(add=["node", "npm"])
    assert pm.cli.cmd_install(argparse.Namespace(names=["node"], tools_only=False)) == 0
    assert install_spy["names"] == ["node"]
    assert declined() == frozenset()


def test_default_install_still_carries_node_fatally(install_spy, monkeypatch):
    # Hosts without an opt-out keep the managed runtime exactly as today:
    # node/npm install through the fatal path, not the warn-only defaults.
    monkeypatch.setattr(hermes_constants, "verify_host_node",
                        lambda: (_ for _ in ()).throw(AssertionError("no host check on a default install")))
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False)) == 0
    assert {"node", "npm"} <= set(install_spy["names"])


# --- host verification against package.json engines ---

def _write_host_tool(bin_dir: Path, name: str, version: str) -> None:
    script = bin_dir / name
    script.write_text(f"#!/bin/sh\necho '{version}'\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_verify_host_node_accepts_a_satisfying_toolchain(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_host_tool(bin_dir, "node", "v26.7.0")
    _write_host_tool(bin_dir, "npm", "12.0.2")
    monkeypatch.setenv("PATH", str(bin_dir))
    ok, detail = hermes_constants.verify_host_node()
    assert ok, detail
    assert "v26.7.0" in detail and "12.0.2" in detail


def test_verify_host_node_rejects_an_old_host_node(tmp_path, monkeypatch):
    engines = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))["engines"]
    assert engines["node"]  # the rejection below is about this floor, not a hardcoded version
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_host_tool(bin_dir, "node", "v20.0.0")
    _write_host_tool(bin_dir, "npm", "12.0.2")
    monkeypatch.setenv("PATH", str(bin_dir))
    ok, detail = hermes_constants.verify_host_node()
    assert not ok
    assert "engines.node" in detail


def test_verify_host_node_reports_a_missing_host_npm(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_host_tool(bin_dir, "node", "v26.7.0")
    monkeypatch.setenv("PATH", str(bin_dir))
    ok, detail = hermes_constants.verify_host_node()
    assert not ok
    assert "npm" in detail


# --- host-mode resolution: PATH instead of the managed entry ---

def test_find_node_executable_prefers_the_host_when_declined(tmp_path, monkeypatch):
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    _write_host_tool(host_dir, "node", "v26.7.0")
    _write_host_tool(host_dir, "npm", "12.0.2")
    monkeypatch.setenv("PATH", str(host_dir))
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: True)
    assert hermes_constants.find_node_executable("node") == str(host_dir / "node")
    assert hermes_constants.find_node_executable("npm") == str(host_dir / "npm")


def test_find_node_executable_stays_managed_without_opt_out(monkeypatch):
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: False)
    assert hermes_constants.find_node_executable("not-a-node-command") is None


def test_with_hermes_node_path_returns_the_caller_env_when_declined(monkeypatch):
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: True)
    base = {"PATH": "/host/bin", "KEPT": "1"}
    assert hermes_constants.with_hermes_node_path(base) == base
    assert hermes_constants.with_hermes_node_path(None) == dict(os.environ)


def test_source_build_env_uses_the_host_without_provisioning(monkeypatch):
    from hermes_cli import source_build

    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: True)
    monkeypatch.setattr(hermes_constants, "verify_host_node", lambda: (True, "host node v26.7.0"))
    import pm as pm_facade

    def forbidden_ensure(*args, **kwargs):
        raise AssertionError("host mode must not provision the managed entry")

    monkeypatch.setattr(pm_facade, "ensure", forbidden_ensure)
    env = source_build.source_build_env(base_env={"FROM": "caller"})
    assert env["FROM"] == "caller"
    assert env["CI"] == "1"


def test_managed_runtime_path_entries_skip_node_when_declined(monkeypatch):
    from tools.environments import local as local_env

    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: True)
    import pm as pm_facade

    def forbidden_env_for(*args, **kwargs):
        raise AssertionError("host mode must not compose the managed node entry")

    monkeypatch.setattr(pm_facade, "env_for", forbidden_env_for)
    assert all("node" not in entry and "npm" not in entry
               for entry in local_env._managed_runtime_path_entries())


# --- hermes doctor: two-Node warning + host-mode rows ---

def _quiet_browser(monkeypatch):
    from hermes_cli import doctor_tools

    monkeypatch.setattr(doctor_tools, "_check_agent_browser", lambda should_fix: False)
    monkeypatch.setattr(doctor_tools, "_check_lightpanda", lambda: None)


def test_doctor_warns_when_managed_node_shadows_the_host(monkeypatch, capsys):
    from hermes_cli import doctor_tools

    _quiet_browser(monkeypatch)
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: False)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path",
                        lambda name: Path(f"/store/{name}/bin/{name}"))
    monkeypatch.setattr(doctor_tools, "_host_node_on_path_excluding_managed",
                        lambda: "/home/user/.nvm/versions/node/v26.7.0/bin/node")
    finding = doctor_tools._check_node_and_browser(False)
    out = capsys.readouterr().out
    assert "managed Node shadows host Node" in out
    assert "--skip-node" in out
    assert finding.issues == []


def test_doctor_stays_quiet_with_only_the_managed_node(monkeypatch, capsys):
    from hermes_cli import doctor_tools

    _quiet_browser(monkeypatch)
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: False)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path",
                        lambda name: Path(f"/store/{name}/bin/{name}"))
    monkeypatch.setattr(doctor_tools, "_host_node_on_path_excluding_managed", lambda: None)
    finding = doctor_tools._check_node_and_browser(False)
    out = capsys.readouterr().out
    assert "shadows host" not in out
    assert finding.issues == []


def test_doctor_reports_a_healthy_host_runtime(monkeypatch, capsys):
    from hermes_cli import doctor_tools

    _quiet_browser(monkeypatch)
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: True)
    monkeypatch.setattr(hermes_constants, "verify_host_node",
                        lambda: (True, "host node v26.7.0 at /host/bin/node"))
    finding = doctor_tools._check_node_and_browser(False)
    out = capsys.readouterr().out
    assert "(host;" in out
    assert finding.issues == []


def test_doctor_flags_an_unusable_host_runtime(monkeypatch, capsys):
    from hermes_cli import doctor_tools

    _quiet_browser(monkeypatch)
    monkeypatch.setattr(hermes_constants, "is_node_host_mode", lambda: True)
    monkeypatch.setattr(hermes_constants, "verify_host_node",
                        lambda: (False, "host node not found on PATH"))
    finding = doctor_tools._check_node_and_browser(False)
    out = capsys.readouterr().out
    assert "Host Node.js unusable" in out
    assert finding.issues and "Host Node.js unusable" in finding.issues[0]


# --- installers hand the opt-out to PM ---

def test_install_sh_skip_node_becomes_the_pm_opt_out(tmp_path):
    bash = shutil.which("bash")
    assert bash
    script = REPO_ROOT / "scripts/install.sh"
    help_text = subprocess.run([bash, str(script), "--help"],
                               capture_output=True, text=True, timeout=10).stdout
    assert "--skip-node" in help_text
    record = tmp_path / "pm-argv"
    boot = tmp_path / "boot-python"
    boot.write_text(f"#!{bash}\nprintf \"%s\\n\" \"$@\" > {record}\n", encoding="utf-8")
    boot.chmod(0o755)
    runner = ('source "$1" "${@:4}" --manifest; INSTALL_DIR="$2"; FIXTURE_PY="$3"; '
              'bootstrap_python() { boot_py="$FIXTURE_PY"; }; bootstrap_pm')
    result = subprocess.run([bash, "-c", runner, "test", str(script), str(tmp_path), str(boot), "--skip-node"],
                            env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert record.read_text(encoding="utf-8").splitlines() == ["-m", "pm.cli", "install", "--without", "node"]


def test_install_ps1_wires_skip_node_to_pm():
    text = (REPO_ROOT / "scripts/install.ps1").read_text(encoding="utf-8-sig")
    assert "[switch]$SkipNode" in text
    assert "install.sh --skip-node" in text
    assert "$pmArgs += @('--without', 'node')" in text
