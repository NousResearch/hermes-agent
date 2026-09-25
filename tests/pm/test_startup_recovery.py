"""Bare startup repairs through real PM workers before activating dependencies.

The copied install is not a self-managed git checkout: source-update completion
must defer to recorded-graph recovery, not resolve today's application inputs.
"""
from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.lock import Facts, Lockfile
from pm.plugin_inputs import Members
from pm.runtime import runtime_environment
from pm.store import current_target, tree_digest
from tests.pm._fixtures import _wheel

@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Path.home's monkeypatch does not cross the subprocess boundary. PM's
    # machine cache and platform-specific home lookup must be isolated there too.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)


@pytest.mark.parametrize("marker_name", [".update-incomplete", ".lazy-refresh-incomplete", None, "manual", "baseline"])
def test_bootstrap_repairs_before_dependency_activation(tmp_path, monkeypatch, marker_name):
    import pm.paths as paths
    from pm.environments import selected_venv, site_packages

    engine = importlib.import_module("pm.install")
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
    # Include the real preimport protocol, including its ownership check. Do
    # not stub prepare_launch: the same files are also saved in PM's workspace.
    for name in ("__init__.py", "runtime_state.py", "_early_recovery.py",
                 "_parser.py", "venv_sync.py", "steward.py"):
        shutil.copy2(repo / "hermes_cli" / name, cli / name)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "startup_dep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="startup-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["startup-dep==1.0"]\n'
        '[project.optional-dependencies]\nstartup-extra=[]\n'
        '[tool.uv]\npackage=false\nno-index=true\noffline=true\n'
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
    extras = ["startup-extra"]
    engine.sync_venv(extras, explicit=True, plugins=Members([]))
    old = selected_venv(core)
    recorded = Facts(paths.runtime_facts_path(), strict=True).get("venv")
    assert recorded is not None
    saved_lock = Path(recorded["resolved_lock"]).read_bytes()
    saved_project = Path(recorded["resolved_lock"]).with_name("pyproject.toml").read_bytes()
    old_site = site_packages(old)
    activated_old = tmp_path / "activated-damaged-dependencies"
    if marker_name in {".update-incomplete", ".lazy-refresh-incomplete"}:
        # Partial damage leaves executable .pth hooks behind. Merely succeeding
        # eventually is insufficient: startup must never activate this tree.
        shutil.rmtree(old_site / "startup_dep")
        (old_site / "old_dependency.pth").write_text(
            f"import pathlib; pathlib.Path({str(activated_old)!r}).touch()\n", encoding="utf-8",
        )
    else:
        shutil.rmtree(old_site)
    if marker_name == "baseline":
        from pm.features import write_features

        # A shipped baseline has its feature declaration but no mutable selection.
        paths.runtime_facts_path().unlink()
        write_features(extras, tmp_path)
    else:
        # Recovery must replay the saved workspace and lock, not discover a new
        # graph from broken/changed live inputs or silently lose recorded extras.
        (core / "pyproject.toml").write_text("broken [project\n", encoding="utf-8")
        (core / "uv.lock").write_text("broken lock [\n", encoding="utf-8")
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
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        lock.set_pin(name, "fixture", {target: {"url": binary.as_uri(), "sha256": digest}})
        facts.record(name, "fixture", str(directory), {}, tools,
                     target=target, artifacts=[digest], digest=tree_digest(directory))
    lock.save()
    launcher = core / "launch.py"
    launcher.write_text(
        'import importlib.util, json\n'
        'assert importlib.util.find_spec("startup_dep") is None\n'
        'import hermes_bootstrap\nimport startup_dep\n'
        'print(json.dumps({"version": startup_dep.__version__, "file": startup_dep.__file__}))\n',
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(core), "HERMES_PYTHON_SRC_ROOT": str(core)}
    env.pop("PYTEST_CURRENT_TEST", None)  # this child owns an isolated copied installation
    control = subprocess.run(
        [sys.executable, "-S", "-c", f"import site; site.addsitedir({str(old_site)!r}); import startup_dep"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert control.returncode != 0 and "No module named 'startup_dep'" in control.stderr
    if old_site.is_dir():
        assert activated_old.is_file(), "positive control did not execute the old activation hook"
        activated_old.unlink()
    receipts = home / "logs" / "update_receipts"
    before_receipts = set(receipts.glob("pm_*.json"))
    if marker_name == "manual":
        assert marker is not None
        refused = subprocess.run(
            [sys.executable, "-S", str(launcher)], cwd=tmp_path,
            env=env, capture_output=True, text=True, timeout=90,
        )
        assert refused.returncode != 0
        assert "retry limit reached" in refused.stderr
        assert selected_venv(core) == old
        assert marker.exists()
        assert set(receipts.glob("pm_*.json")) == before_receipts
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
    output = json.loads(result.stdout)
    assert output["version"] == "1.0"
    selected = selected_venv(core)
    assert Path(output["file"]).is_relative_to(site_packages(selected))
    assert not activated_old.exists(), "damaged dependencies activated before repair"
    assert marker is None or not marker.exists()
    assert selected != old
    assert old.is_dir()
    repaired_fact = Facts(paths.runtime_facts_path(), strict=True).get("venv")
    assert repaired_fact is not None
    assert repaired_fact["extras"] == extras
    assert Path(repaired_fact["environment"]) == selected
    repaired_lock = Path(repaired_fact["resolved_lock"])
    assert repaired_lock.read_bytes() == saved_lock
    assert repaired_lock.with_name("pyproject.toml").read_bytes() == saved_project
    receipt_path, = set(receipts.glob("pm_*.json")) - before_receipts
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["outcome"] == "ok"
    assert receipt["venv_rebuild"]["ok"] is True
    assert receipt["feature_list"] == extras
    if marker_name != "manual":
        assert "repaired" in result.stderr.lower()

    # A completed repair is durable. The next bare launch must not run PM again
    # or replace the selection (even though the live manifests remain broken).
    facts_bytes = paths.runtime_facts_path().read_bytes()
    completed_receipts = set(receipts.glob("pm_*.json"))
    again = subprocess.run(
        [sys.executable, "-S", str(launcher)], cwd=tmp_path,
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert again.returncode == 0, again.stdout + again.stderr
    assert json.loads(again.stdout) == output
    assert paths.runtime_facts_path().read_bytes() == facts_bytes
    assert set(receipts.glob("pm_*.json")) == completed_receipts
