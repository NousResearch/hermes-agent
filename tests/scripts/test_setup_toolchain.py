"""The CI bootstrap reads PM's pins without importing installed dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pm.lock import Lockfile
from pm.paths import lockfile_path
from pm.store import current_target
from tests.pm._fixtures import build_worker, client, isolated_python  # noqa: F401


@pytest.mark.parametrize("test_environment", [False, True], ids=["runtime", "tests"])
def test_development_setup_keeps_test_groups_out_of_the_runtime(tmp_path, monkeypatch, test_environment, build_worker):
    from types import SimpleNamespace
    import shutil

    from scripts.ci import setup_toolchain
    from pm import lock_project
    from tests.pm._fixtures import _wheel

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "test_only_dep", "1.0")
    _wheel(wheels, "dev_only_dep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="ci-test-environment"\nversion="1"\nrequires-python=">=3.11"\n'
        '[project.optional-dependencies]\nall=[]\n'
        '[dependency-groups]\ndev=["dev-only-dep==1.0"]\ntest=["test-only-dep==1.0"]\n'
        '[tool.uv]\npackage=false\nno-index=true\ndefault-groups=[]\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    lock_project(core, python=Path(sys.executable), offline=True, explicit=True)
    home = tmp_path / "ci-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pm.paths.repo_root", lambda: core)
    files = {name: tmp_path / name for name in ("GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH")}
    for name, file in files.items():
        monkeypatch.setenv(name, str(file))

    setup_toolchain.dependencies(SimpleNamespace(extras=[], home=home, test_environment=test_environment))

    outputs = dict(line.split("=", 1) for line in files["GITHUB_OUTPUT"].read_text(encoding="utf-8").splitlines())
    result = subprocess.run(
        [outputs["python-path"], "-I", "-c", "import importlib.util; print(importlib.util.find_spec('test_only_dep') is not None)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(test_environment)
    probe = subprocess.run(
        [outputs["python-path"], "-I", "-c",
         "import importlib.util; print(importlib.util.find_spec('dev_only_dep') is not None)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == str(test_environment)
    assert Path(outputs["venv"]).is_relative_to(home)
    assert not (core / ".venv").exists()
    from pm.environments import runtime_facts_path
    assert runtime_facts_path(core).exists() != test_environment
    if not test_environment:
        from pm import build_environment

        bundle_python = build_environment(source=core, out=tmp_path / "bundle-env",
                                          all_extras=True, no_install_project=True, explicit=True)
        bundle = subprocess.run(
            [bundle_python, "-I", "-c",
             "import importlib.util; assert importlib.util.find_spec('dev_only_dep') is None; "
             "assert importlib.util.find_spec('test_only_dep') is None"],
            cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )
        assert bundle.returncode == 0, bundle.stderr


@pytest.mark.parametrize("toolchain,names", [
    ("python", {"python", "uv"}),
    ("node", {"node", "npm"}),
    ("all", {"python", "uv", "node", "npm"}),
])
def test_stdlib_bootstrap_exports_the_pm_lock(toolchain, names, tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "output"
    envfile = tmp_path / "environment"
    home = tmp_path / "runner state"
    env = {**os.environ, "GITHUB_OUTPUT": str(output), "GITHUB_ENV": str(envfile)}
    result = subprocess.run(
        [sys.executable, "-S", str(root / "scripts/ci/setup_toolchain.py"),
         "prepare", "--toolchain", toolchain, "--home", str(home)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8-sig").splitlines())
    lock = Lockfile(lockfile_path())
    assert json.loads(values["packages"]) == sorted(names)
    assert values["target"] == current_target()
    for name in names:
        expected = lock.version(name)
        assert values[f"{name}-version"] == (expected.partition("+")[0] if name == "python" else expected)
    exported = dict(line.split("=", 1) for line in envfile.read_text(encoding="utf-8-sig").splitlines())
    assert Path(exported["HERMES_HOME"]) == home
    assert Path(exported["HERMES_RUNTIME_DIR"]).is_relative_to(home)
    assert not Path(exported["HERMES_RUNTIME_DIR"]).exists(), "prepare must not provision before cache restore"


@pytest.mark.parametrize("extras", ['"dev"', '{}', '[1]', '["dev\\nHERMES_HOME=bad"]', '["--all"]'])
def test_invalid_extras_do_not_export_or_install(extras, tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "output"
    home = tmp_path / "state"
    result = subprocess.run(
        [sys.executable, "-S", str(root / "scripts/ci/setup_toolchain.py"), "prepare",
         "--home", str(home), "--extras", extras],
        cwd=tmp_path, env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode != 0
    assert "extras must be a JSON array of extra names" in result.stderr
    assert not output.exists()
    assert not home.exists()
