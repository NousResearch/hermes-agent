"""Startup restores recorded dependencies before importing application packages."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.lock import Facts, Lockfile
from pm.runtime import runtime_environment
from pm.store import current_target, sha256_file, tree_digest
from tests.pm.test_workspace_build_inputs import _wheel

@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))



@pytest.mark.parametrize("marker_name", [".update-incomplete", ".lazy-refresh-incomplete", None, "manual", "baseline"])
def test_bootstrap_repairs_before_dependency_activation(tmp_path, monkeypatch, marker_name):
    import pm.paths as paths
    from hermes_cli.runtime_paths import selected_venv, site_packages

    engine = importlib.import_module("pm.ensure")
    repo = Path(__file__).resolve().parents[2]
    core = tmp_path / "app"
    core.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    for name in ("hermes_bootstrap.py", "hermes_constants.py"):
        shutil.copy2(repo / name, core / name)
    shutil.copytree(repo / "pm", core / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    cli = core / "hermes_cli"
    cli.mkdir()
    for name in ("__init__.py", "runtime_paths.py", "runtime_state.py", "_early_recovery.py", "_parser.py"):
        shutil.copy2(repo / "hermes_cli" / name, cli / name)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "startup_dep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="startup-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["startup-dep==1.0"]\n[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    uv = shutil.which("uv")
    assert uv, "startup recovery requires real uv"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    monkeypatch.setattr(engine, "lazy_installs_allowed", lambda: True)
    clean = {**runtime_environment(), "UV_PYTHON": sys.executable, "UV_OFFLINE": "1"}
    clean.pop("UV_NO_CONFIG", None)
    subprocess.run([uv, "lock"], cwd=core, env=clean, capture_output=True, check=True, timeout=60)
    engine.sync_venv([], explicit=True, plugin_dirs=[])
    old = selected_venv(core)
    shutil.rmtree(site_packages(old))
    if marker_name == "baseline":
        from pm.features import write_features

        # A shipped baseline has its feature declaration but no mutable selection.
        paths.runtime_facts_path().unlink()
        write_features([], tmp_path)
    marker = core / (".update-incomplete" if marker_name in {"manual", "baseline"} else marker_name) if marker_name else None
    if marker:
        marker.write_text('{"attempts":3}' if marker_name == "manual" else '{"attempts":0}', encoding="utf-8")

    # Select real executables through isolated fixture facts, without downloads.
    lock = Lockfile(core / "pm" / "lock.json")
    tools = tmp_path / "tools"
    facts = Facts(tools / "facts.json")
    uv_entry = tools / "uv"
    uv_entry.mkdir(parents=True)
    shutil.copy2(uv, uv_entry / Path(uv).name)
    python_entry = Path(sys.base_prefix)
    if os.name != "nt":
        # A system Python's prefix can be /usr: never hash that entire tree.
        python_entry = tools / "python"
        (python_entry / "bin").mkdir(parents=True)
        (python_entry / "bin/python3").symlink_to(Path(sys._base_executable).resolve())
    from pm.registry import get_package

    target = current_target()
    for name, directory in (("uv", uv_entry), ("python", python_entry)):
        package = get_package(name)
        binary = package.binary(directory, target)
        assert not package.verify(directory, target), (
            f"startup recovery requires native {target} tools: {binary}"
        )
        digest = sha256_file(binary)
        lock.set_pin(name, "fixture", {target: {"url": binary.as_uri(), "sha256": digest}})
        facts.record(name, "fixture", str(directory), {}, tools,
                     target=target, artifacts=[digest], digest=tree_digest(directory))
    lock.save()
    launcher = core / "launch.py"
    launcher.write_text(
        'import hermes_bootstrap\nimport startup_dep\nprint("APP_STARTED", startup_dep.__version__)\n',
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(core)}
    env.pop("PYTEST_CURRENT_TEST", None)  # this child owns an isolated copied installation
    if marker_name == "manual":
        repaired = subprocess.run([sys.executable, "-S", "-m", "pm.cli", "repair"], cwd=tmp_path,
                                  env=env, capture_output=True, text=True, timeout=90)
        assert repaired.returncode == 0, repaired.stdout + repaired.stderr
        assert not marker.exists()
    result = subprocess.run(
        [sys.executable, "-S", str(launcher)], cwd=tmp_path,
        env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "APP_STARTED 1.0"
    assert marker is None or not marker.exists()
    assert selected_venv(core) != old
    assert old.is_dir()
    if marker_name != "manual":
        assert "repaired" in result.stderr.lower()
