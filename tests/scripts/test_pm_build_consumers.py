"""Build adapters consume selected Python environments, never install commands."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def local_toolchain(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    uv = shutil.which("uv")
    assert uv, "build integration tests require the dev-shell toolchain"
    monkeypatch.setattr("pm.client.is_runtime", lambda: True)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))


def test_ci_setup_exports_no_installer_path_or_policy(tmp_path, monkeypatch, local_toolchain):
    import importlib
    from types import SimpleNamespace
    from scripts.ci import setup_toolchain

    monkeypatch.setattr(setup_toolchain, "packages", lambda _: ["python", "uv"])
    manager = importlib.import_module("pm.ensure")
    monkeypatch.setattr(manager, "ensure", lambda *args, **kwargs: None)

    def environment(*names, base_env=None):
        assert "uv" not in names
        return {"PATH": str(Path(sys.executable).parent)}

    def package(name):
        assert name != "uv", "CI must not resolve an installer executable"
        return SimpleNamespace(binary=lambda *args: Path(sys.executable))

    monkeypatch.setattr(manager, "env_for", environment)
    monkeypatch.setattr("pm.registry.get_package", package)
    monkeypatch.setattr("pm.lock.Facts", lambda path: SimpleNamespace(get=lambda name: {"entry": name, "version": "fixture"}))
    files = {name: tmp_path / name for name in ("GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH")}
    for name, path in files.items():
        monkeypatch.setenv(name, str(path))
    setup_toolchain.install(SimpleNamespace(toolchain="python", home=tmp_path / "ci"))
    outputs = dict(line.split("=", 1) for line in files["GITHUB_OUTPUT"].read_text(encoding="utf-8").splitlines())
    exported = dict(line.split("=", 1) for line in files["GITHUB_ENV"].read_text(encoding="utf-8").splitlines())
    assert "uv-path" not in outputs
    assert not any(name.startswith("UV_") for name in exported)
    assert exported["HERMES_PYTHON"] == outputs["python-path"]
    assert Path(outputs["python-path"]).parent.as_posix() in files["GITHUB_PATH"].read_text(encoding="utf-8").replace("\\", "/")
    subprocess.run([outputs["python-path"], "-I", "-c", "import sys; assert sys.prefix != sys.base_prefix"], check=True)


def test_ci_packages_exports_python_and_preserves_real_child_exit(tmp_path, monkeypatch, local_toolchain):
    import pm
    from scripts.ci import python_packages
    from tests.pm.test_workspace_build_inputs import _wheel

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "ci_probe", "1.0")
    python = pm.build_requirements_environment(
        ["ci-probe==1.0"], out=tmp_path / "prepared", python=Path(sys.executable),
        wheelhouse=wheels, offline=True, explicit=True,
    )

    def selected(name, requirements, *, explicit):
        assert name == "ci-tools" and requirements == ["ci-probe==1.0"] and explicit
        return python

    monkeypatch.setattr(pm, "ensure_environment", selected)
    exports, path = tmp_path / "environment", tmp_path / "path"
    monkeypatch.setenv("GITHUB_ENV", str(exports))
    monkeypatch.setenv("GITHUB_PATH", str(path))
    assert python_packages.main(["ci-probe==1.0"]) == 0
    values = dict(line.split("=", 1) for line in exports.read_text(encoding="utf-8").splitlines())
    assert values["HERMES_PYTHON"] == str(python)
    assert values["VIRTUAL_ENV"] == str(python.parent.parent)
    assert not any(name.startswith("UV_") for name in values)
    assert path.read_text(encoding="utf-8").strip() == str(python.parent)
    assert python_packages.main([
        "ci-probe==1.0", "--", "-I", "-c", "import ci_probe; raise SystemExit(37)",
    ]) == 37


def test_termux_gate_checks_real_offline_wheels_and_application_uses_same_graph(tmp_path, local_toolchain):
    from pm.package import InstallError
    from scripts.termux import build_wheels, build_environment
    from tests.pm.test_workspace_build_inputs import _wheel

    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    _wheel(wheels, "native_probe", "1.0")
    resolved = tmp_path / "resolved.txt"
    resolved.write_text(
        'native-probe\t==1.0\t\t\n'
        'missing-windows-only\t==1.0\tsys_platform == "win32"\t\n'
        'nemo-relay\t==1.0\t\t\n', encoding="utf-8",
    )
    # This adapter runs on the actual host; no platform faking is needed.
    if os.name == "nt":
        resolved.write_text('native-probe\t==1.0\t\t\n', encoding="utf-8")
    build_wheels.wheelhouse_gates(resolved, wheels, ["native-probe"])
    requirements = tmp_path / "requirements.txt"
    build_wheels.write_reqs_file(resolved, requirements)
    build_environment.application(tmp_path, requirements, Path(sys.executable))
    python = tmp_path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-I", "-c", "import native_probe"], check=True)
    for wheel in wheels.iterdir():
        wheel.unlink()
    with pytest.raises(InstallError):
        build_wheels.wheelhouse_gates(resolved, wheels, ["native-probe"])
