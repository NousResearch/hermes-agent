"""Command diagnostics use the same selection and launch contract as setup."""

import json
from argparse import Namespace
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hermes_cli import _launchers, doctor, doctor_platform
from hermes_cli.runtime_paths import install_state_dir, site_packages


def _tree(tmp_path, monkeypatch):
    home = tmp_path / "data"
    home.mkdir()
    project = tmp_path / "source"
    project.mkdir()
    (project / ".install_method").write_text("git", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, doctor_platform._check_command_installation),))
    command = tmp_path / ".local" / "bin" / "hermes"
    command.parent.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(command.parent))
    return project, home, command


def _generation(project):
    selected = install_state_dir(project) / "environments" / "selected" / "venv"
    site_packages(selected).mkdir(parents=True)
    (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (install_state_dir(project) / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8"
    )
    return selected


def _pm_source(project, home):
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "hermes", "hermes_bootstrap.py", "hermes_constants.py", "hermes_cli/__init__.py",
        "hermes_cli/runtime_paths.py", "hermes_cli/runtime_state.py",
        "hermes_cli/_early_recovery.py", "hermes_cli/_parser.py",
    ):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    (project / "hermes_cli/main.py").write_text(
        "def main():\n    import selected_probe\n    print(selected_probe.VALUE)\n    return 0\n",
        encoding="utf-8",
    )
    interpreter = home / "tools" / "python-fixture" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys._base_executable).resolve())
    (home / "tools" / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "python-fixture"}}}), encoding="utf-8"
    )
    return interpreter


@pytest.mark.platforms("posix")
def test_pm_generation_does_not_require_a_legacy_console_script(tmp_path, monkeypatch, capsys):
    project, home, command = _tree(tmp_path, monkeypatch)
    selected = _generation(project)
    _pm_source(project, home)
    assert _launchers.stage_launcher("hermes", project, command.parent) == command

    doctor.run_doctor(Namespace(fix=True))

    out = capsys.readouterr().out
    assert "All checks passed" in out
    assert not (project / "venv").exists()
    assert not (selected / "bin" / "hermes").exists()
    assert command.is_file() and not command.is_symlink()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("prior", ["missing", "legacy"])
def test_pm_fix_publishes_a_generation_aware_launcher(tmp_path, monkeypatch, capsys, prior):
    project, home, command = _tree(tmp_path, monkeypatch)
    _pm_source(project, home)
    selected = _generation(project)
    (site_packages(selected) / "selected_probe.py").write_text("VALUE = 'selected'\n", encoding="utf-8")
    stale = project / "venv" / "bin" / "hermes"
    stale.parent.mkdir(parents=True)
    stale.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    stale.chmod(0o755)
    if prior == "legacy":
        command.symlink_to(stale)

    doctor.run_doctor(Namespace(fix=True))

    assert "Fixed 1 issue" in capsys.readouterr().out
    assert not command.is_symlink()
    result = subprocess.run([str(command)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "selected"
    assert stale.is_file()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("kind", ["wrapper", "symlink"])
def test_fix_preserves_user_managed_commands(tmp_path, monkeypatch, capsys, kind):
    project, home, command = _tree(tmp_path, monkeypatch)
    _pm_source(project, home)
    _generation(project)
    wrapper = tmp_path / "custom-wrapper"
    body = "#!/bin/sh\nexit 42\n"
    wrapper.write_text(body, encoding="utf-8")
    wrapper.chmod(0o755)
    if kind == "symlink":
        command.symlink_to(wrapper)
    else:
        shutil.copy2(wrapper, command)

    doctor.run_doctor(Namespace(fix=True))

    out = capsys.readouterr().out
    assert "Fixed 1 issue" not in out
    assert command.read_text(encoding="utf-8") == body
    if kind == "symlink":
        assert command.is_symlink() and command.resolve() == wrapper
        assert "manual" in out.lower()


@pytest.mark.parametrize("layout", ["legacy", "generation"])
@pytest.mark.parametrize("active", [True, False])
def test_doctor_reports_selected_import_tree_not_interpreter_prefix(tmp_path, monkeypatch, capsys, layout, active):
    import importlib
    import pm.paths

    project, _home, _command = _tree(tmp_path, monkeypatch)
    if layout == "generation":
        selected = _generation(project)
    else:
        selected = project / "venv"
        site_packages(selected).mkdir(parents=True)
        (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    monkeypatch.setattr(pm.paths, "repo_root", lambda: project)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, doctor_platform._check_python_environment),))
    # A venv prefix with no selected imports is not evidence of activation;
    # conversely PM's base interpreter can import the selected tree directly.
    monkeypatch.setattr(sys, "prefix", sys.base_prefix if active else str(selected))
    if active:
        module = site_packages(selected) / "doctor_selected_probe.py"
        module.write_text("VALUE = 'selected'\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(site_packages(selected)))
        monkeypatch.delitem(sys.modules, "doctor_selected_probe", raising=False)
        loaded = importlib.import_module("doctor_selected_probe")
        assert Path(loaded.__file__) == module
        monkeypatch.delitem(sys.modules, "doctor_selected_probe")

    doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert str(selected) in out
    assert ("active in this process" in out) is active
    assert ("runs outside it" in out) is not active


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("method, remedy", [
    ("git", "hermes pm repair"), ("nix", "Nix"), ("docker", "docker pull"), ("apt", "pkg upgrade"),
])
def test_remedies_and_launcher_repairs_respect_install_owner(tmp_path, monkeypatch, capsys, method, remedy):
    project, _home, command = _tree(tmp_path, monkeypatch)
    (project / ".install_method").write_text(method, encoding="utf-8")
    entry = project / "venv" / "bin" / "hermes"
    entry.parent.mkdir(parents=True)
    entry.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    entry.chmod(0o755)
    monkeypatch.setattr(doctor_platform, "_PACKAGES", (("doctor_missing_dependency_probe", "Required dependency", False),))
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", (
        (None, doctor_platform._check_required_packages), (None, doctor_platform._check_command_installation),
    ))

    doctor.run_doctor(Namespace(fix=True))

    out = capsys.readouterr().out
    assert remedy in out
    assert "pip install" not in out
    if method == "git":
        assert command.is_symlink() and command.resolve() == entry
    else:
        assert not command.exists()
        assert "hermes pm repair" not in out
