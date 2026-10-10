"""Plugin Doctor must re-check declared deps inside pm's staged runtime venv (#134469).

The doctor probe runs in-process, so when the gateway execs the staged runtime venv
while the CLI was launched from another interpreter (e.g. the legacy source-tree
venv), an in-process dependency check blesses packages the gateway cannot import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugin_dev


def _deps_plugin(tmp_path: Path) -> Path:
    plugin = tmp_path / "deps-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "name: deps-plugin\n"
        "python_dependencies:\n"
        "  - mnemosyne-memory>=0.3,<0.4\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    return plugin


def _fake_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "runtime-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /python\n", encoding="utf-8")
    return venv


def _patch_pm_venv(monkeypatch, venv_dir: Path) -> None:
    monkeypatch.setattr(
        "pm.packages.Venv", lambda: SimpleNamespace(
            venv_dir=lambda: venv_dir, project_root=lambda: venv_dir.parent))
    monkeypatch.setattr(
        "pm.environments.running_from_selected_environment", lambda root: False)


def test_doctor_warns_when_runtime_venv_lacks_declared_deps(tmp_path, monkeypatch) -> None:
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(plugin_dev, "_staged_runtime_venv", lambda: venv)
    monkeypatch.setattr(
        plugin_dev, "_deps_missing_from_venv", lambda v, dists: ["mnemosyne-memory"])

    report = plugin_dev.doctor_plugin(_deps_plugin(tmp_path))

    messages = "\n".join(f.message for f in report.findings)
    assert "missing from the staged runtime venv" in messages
    assert "mnemosyne-memory" in messages


def test_doctor_stays_silent_when_runtime_venv_has_declared_deps(tmp_path, monkeypatch) -> None:
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(plugin_dev, "_staged_runtime_venv", lambda: venv)
    monkeypatch.setattr(plugin_dev, "_deps_missing_from_venv", lambda v, dists: [])

    report = plugin_dev.doctor_plugin(_deps_plugin(tmp_path))

    messages = "\n".join(f.message for f in report.findings)
    assert "staged runtime venv" not in messages


def test_doctor_skips_runtime_probe_without_staged_venv(tmp_path, monkeypatch) -> None:
    probes: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(plugin_dev, "_staged_runtime_venv", lambda: None)
    monkeypatch.setattr(
        plugin_dev, "_deps_missing_from_venv",
        lambda v, dists: probes.append((v, dists)) or [])

    plugin_dev.doctor_plugin(_deps_plugin(tmp_path))

    assert probes == []


def test_staged_venv_resolved_when_process_runs_outside(tmp_path, monkeypatch) -> None:
    venv = _fake_venv(tmp_path)
    _patch_pm_venv(monkeypatch, venv)

    assert plugin_dev._staged_runtime_venv() == venv


def test_staged_venv_none_when_process_runs_inside(tmp_path, monkeypatch) -> None:
    venv = _fake_venv(tmp_path)
    _patch_pm_venv(monkeypatch, venv)
    monkeypatch.setattr(
        "pm.environments.running_from_selected_environment", lambda root: True)

    assert plugin_dev._staged_runtime_venv() is None


def test_staged_venv_requires_provisioned_marker(tmp_path, monkeypatch) -> None:
    intended = tmp_path / "intended-venv"
    intended.mkdir()
    _patch_pm_venv(monkeypatch, intended)

    assert plugin_dev._staged_runtime_venv() is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell shim")
def test_deps_probe_runs_the_venv_interpreter(tmp_path) -> None:
    venv = _fake_venv(tmp_path)
    shim = venv / "bin" / "python"
    shim.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n", encoding="utf-8")
    shim.chmod(0o755)

    missing = plugin_dev._deps_missing_from_venv(venv, ["pytest>=8", "no-such-dist-zqx-134469"])

    assert missing == ["no-such-dist-zqx-134469"]
