"""Command diagnostics use the same selection and launch contract as setup."""

import json
from argparse import Namespace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hermes_cli import _launchers, doctor, doctor_platform
from pm.environments import install_state_dir, site_packages


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
        "hermes_agent/__init__.py", "hermes_agent/compat/__init__.py", "hermes_agent/compat/finder.py",
        "hermes_agent/compat/moves.py",
        "pm/environments.py", "pm/filesystem.py", "hermes_cli/runtime_state.py",
        "hermes_cli/_early_recovery.py", "hermes_cli/_parser.py",
        "hermes_cli/venv_sync.py", "hermes_cli/steward.py", "hermes_cli/stderr_timestamp.py",
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
    result = subprocess.run([str(command)], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False)
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


def _bundled_tree(tmp_path, monkeypatch, *, app_dir: Path | None = None, cli_name: str = "hermes",
                  with_manifest: bool = True, with_runtime_stamp: bool = True, on_path: bool = True):
    home = tmp_path / "data"
    home.mkdir(exist_ok=True)
    user_home = tmp_path / "user"
    user_home.mkdir(exist_ok=True)
    payload_root = (app_dir or (tmp_path / "Applications" / "Hermes Agent Canary.app" / "Contents" / "Resources")) / "agent-payload"
    project = payload_root / "hermes-agent"
    project.mkdir(parents=True)
    (project / "hermes").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (project / "hermes").chmod(0o755)
    bin_dir = payload_root / "bin"
    bin_dir.mkdir(parents=True)
    launcher = bin_dir / cli_name
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    if cli_name != "hermes":
        (bin_dir / "hermes").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        (bin_dir / "hermes").chmod(0o755)
    runtime = {
        "repoDir": "hermes-agent",
        "toolsDir": "tools",
        "storePython": "tools/python-fixture/bin/python3",
        "sitePackages": "venv/lib/python3.14/site-packages",
        "commands": {"hermes": f"bin/{cli_name}"},
    }
    stamp = {"payload": "bundled", "distribution": "desktop-app", "updateMechanism": "electron-updater"}
    if with_runtime_stamp:
        stamp["runtime"] = runtime
    (project / "install-stamp.json").write_text(json.dumps(stamp), encoding="utf-8")
    if with_manifest:
        (payload_root / "manifest.json").write_text(
            json.dumps({"schema": 1, "repo": "hermes-agent", "venv": "venv", "store": "tools", "runtime": runtime}),
            encoding="utf-8",
        )
    interpreter = payload_root / "tools" / "python-fixture" / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys._base_executable).resolve())
    (payload_root / "tools" / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "python-fixture"}}}), encoding="utf-8"
    )
    venv_dir = payload_root / "venv"
    site_packages(venv_dir).mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(payload_root / "tools"))
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("ZDOTDIR", raising=False)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setattr(doctor, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, doctor_platform._check_command_installation),))
    link_dir = user_home / ".local" / "bin"
    link_dir.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(link_dir) if on_path else "/usr/local/bin:/usr/bin:/bin")
    return project, launcher, link_dir


@pytest.mark.platforms("posix")
def test_bundled_payload_accepts_bundle_bin_symlink_macos_layout(tmp_path, monkeypatch, capsys):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch)
    (link_dir / "hermes").symlink_to(launcher)

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 0
    assert "All checks passed" in out
    assert "points to wrong target" not in out
    assert str(launcher) in out
    assert "~/.local/bin/hermes → correct target" in out


@pytest.mark.platforms("posix")
def test_bundled_payload_accepts_minimal_stamp_without_manifest(tmp_path, monkeypatch, capsys):
    app_dir = tmp_path / "local" / "share" / "hermes" / "bundles" / "v0.23.1-abc123"
    _project, launcher, link_dir = _bundled_tree(
        tmp_path, monkeypatch, app_dir=app_dir, with_manifest=False, with_runtime_stamp=False,
    )
    (link_dir / "hermes").symlink_to(launcher)

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 0
    assert "All checks passed" in out
    assert "points to wrong target" not in out
    assert "~/.local/bin/hermes → correct target" in out


@pytest.mark.platforms("posix")
def test_bundled_payload_respects_qualified_canary_command_name(tmp_path, monkeypatch, capsys):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, cli_name="hermes-canary")
    other = tmp_path / "stable-hermes"
    other.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (link_dir / "hermes").symlink_to(other)
    (link_dir / "hermes-canary").symlink_to(launcher)

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 0
    assert "All checks passed" in out
    assert "~/.local/bin/hermes-canary → correct target" in out
    assert (link_dir / "hermes").resolve() == other.resolve()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("prior", ["missing", "dangling_bundle", "superseded_appimage"])
def test_bundled_payload_fix_symlinks_bundle_trampoline(tmp_path, monkeypatch, capsys, prior):
    bundles = tmp_path / "bundles"
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, app_dir=bundles / "build-2")
    command = link_dir / "hermes"
    if prior == "dangling_bundle":
        command.symlink_to(tmp_path / "Old.app" / "Contents" / "Resources" / "agent-payload" / "bin" / "hermes")
    elif prior == "superseded_appimage":
        old = bundles / "build-1" / "agent-payload" / "bin" / "hermes"
        old.parent.mkdir(parents=True)
        old.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        command.symlink_to(old)

    rc = doctor.run_doctor(Namespace(fix=True))

    out = capsys.readouterr().out
    assert rc == 0
    assert "Fixed 1 issue" in out
    assert command.is_symlink()
    assert command.resolve() == launcher.resolve()


@pytest.mark.platforms("posix")
def test_bundled_fix_keeps_a_command_published_after_the_check(tmp_path, monkeypatch, capsys):
    _project, _launcher, link_dir = _bundled_tree(tmp_path, monkeypatch)
    command = link_dir / "hermes"
    other = "#!/bin/sh\necho other installer\n"
    real_symlink = os.symlink

    def other_installer_publishes_first(src, dst, *args, **kwargs):
        if not os.path.lexists(command):
            command.write_text(other, encoding="utf-8")
        return real_symlink(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "symlink", other_installer_publishes_first)
    rc = doctor.run_doctor(Namespace(fix=True))

    out = capsys.readouterr().out
    assert rc == 1
    assert not command.is_symlink() and command.read_text(encoding="utf-8") == other
    assert "Fixed 1 issue" not in out


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("foreign_kind", ["other_tool", "other_app"])
def test_bundled_payload_fix_preserves_live_foreign_symlink(tmp_path, monkeypatch, capsys, foreign_kind):
    _project, _launcher, link_dir = _bundled_tree(tmp_path, monkeypatch)
    command = link_dir / "hermes"
    if foreign_kind == "other_tool":
        foreign = tmp_path / "other-tool" / "hermes"
    else:
        foreign = tmp_path / "Other.app" / "Contents" / "Resources" / "agent-payload" / "bin" / "hermes"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    command.symlink_to(foreign)

    rc = doctor.run_doctor(Namespace(fix=True))

    out = capsys.readouterr().out
    assert rc == 1
    assert "Fixed 1 issue" not in out
    assert command.is_symlink() and command.resolve() == foreign.resolve()
    assert "manual" in out.lower()


@pytest.mark.platforms("posix")
def test_command_installation_warns_when_local_bin_missing_from_path(tmp_path, monkeypatch, capsys):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, on_path=False)
    (link_dir / "hermes").symlink_to(launcher)

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 1
    assert "~/.local/bin/hermes → correct target" in out
    assert "~/.local/bin is not on your PATH" in out
    assert "~/.zprofile" in out
    assert 'export PATH="$HOME/.local/bin:$PATH"' in out


_DOCTOR_AFTER_STARTUP = """
import os, sys
from argparse import Namespace
from pathlib import Path
from pm.environments import activate_dependencies

project = Path(sys.argv[1])
activate_dependencies(project)
print("startup PATH=" + os.environ["PATH"], file=sys.stderr)
from hermes_cli import doctor, doctor_platform
doctor.PROJECT_ROOT = project
doctor.HERMES_HOME = Path(os.environ["HERMES_HOME"])
doctor.DOCTOR_CHECKS = ((None, doctor_platform._check_command_installation),)
sys.exit(doctor.run_doctor(Namespace(fix=False)))
"""


def _doctor_after_startup(project: Path, inherited: str) -> subprocess.CompletedProcess:
    """Doctor's command check in a child that boots like `hermes doctor`: real activation first."""
    repo = Path(__file__).resolve().parents[2]
    # The payload venv stands in for the bundle's own dependencies.
    (site_packages(project.parent / "venv") / "test-deps.pth").write_text(
        "\n".join(entry for entry in sys.path if entry.endswith("site-packages")) + "\n", encoding="utf-8"
    )
    env = {**os.environ, "PATH": inherited, "HOME": str(Path.home()), "PYTHONPATH": str(repo)}
    return subprocess.run([sys.executable, "-c", _DOCTOR_AFTER_STARTUP, str(project)],
                          cwd=repo, env=env, capture_output=True, text=True, check=False)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("inherited, rc", [
    ("system", 1),  # startup's own <payload>/bin must not hide the missing ~/.local/bin
    ("payload_bin", 0),  # the user put the bundle's bin dir on PATH themselves
    ("other_link", 0),  # another PATH dir links to the launcher; the venv's own hermes must not shadow it
])
def test_path_check_judges_the_path_doctor_inherited(tmp_path, monkeypatch, inherited, rc):
    project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, on_path=False)
    (link_dir / "hermes").symlink_to(launcher)
    venv_hermes = project.parent / "venv" / "bin" / "hermes"
    venv_hermes.parent.mkdir(parents=True)
    venv_hermes.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    venv_hermes.chmod(0o755)
    other_bin = tmp_path / "other-bin"
    other_bin.mkdir()
    (other_bin / "hermes").symlink_to(launcher)
    first = {"system": [], "payload_bin": [str(launcher.parent)], "other_link": [str(other_bin)]}[inherited]

    done = _doctor_after_startup(project, os.pathsep.join([*first, "/usr/bin", "/bin"]))

    assert f"startup PATH={launcher.parent}{os.pathsep}{venv_hermes.parent}{os.pathsep}" in done.stderr
    assert done.returncode == rc, done.stdout + done.stderr
    assert ("~/.local/bin is not on your PATH" in done.stdout) == (rc == 1)


@pytest.mark.platforms("posix")
def test_path_check_survives_a_path_entry_that_cannot_resolve(tmp_path, monkeypatch, capsys):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, on_path=False)
    (link_dir / "hermes").symlink_to(launcher)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    resolve = Path.resolve

    def resolve_like_python_3_11(self, strict=False):
        # 3.11 and 3.12 raise RuntimeError on a symlink loop; 3.13+ hand the path back.
        if self == loop:
            raise RuntimeError(f"Symlink loop from {str(self)!r}")
        return resolve(self, strict)

    monkeypatch.setattr(Path, "resolve", resolve_like_python_3_11)
    monkeypatch.setenv("PATH", f"{loop}:{launcher.parent}")

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "is not on your PATH" not in out


@pytest.mark.platforms("posix")
def test_command_installation_path_check_respects_zdotdir_and_configured_rc(tmp_path, monkeypatch, capsys):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, on_path=False)
    (link_dir / "hermes").symlink_to(launcher)
    zdotdir = Path.home() / ".config" / "zsh"
    zdotdir.mkdir(parents=True)
    monkeypatch.setenv("ZDOTDIR", str(zdotdir))
    (Path.home() / ".zprofile").write_text('export PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")
    (zdotdir / ".zprofile").write_text('# export PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")

    rc_missing = doctor.run_doctor(Namespace(fix=False))
    out_missing = capsys.readouterr().out
    assert rc_missing == 1
    assert f"{zdotdir}/.zprofile" in out_missing

    (zdotdir / ".zprofile").write_text(
        '# Hermes Agent command\n'
        'case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac\n',
        encoding="utf-8",
    )
    rc_configured = doctor.run_doctor(Namespace(fix=False))
    out_configured = capsys.readouterr().out
    assert rc_configured == 0
    assert "⚠" not in out_configured
    assert "configured in" in out_configured


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("bash_profile, rc, verdict", [
    (False, 0, "configured in ~/.profile"),
    # A login bash reads the first of ~/.bash_profile, ~/.bash_login, ~/.profile, so ~/.profile never runs.
    (True, 1, "~/.local/bin is not on your PATH"),
])
def test_bash_path_check_trusts_only_the_login_file_bash_reads(tmp_path, monkeypatch, capsys, bash_profile, rc, verdict):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, on_path=False)
    (link_dir / "hermes").symlink_to(launcher)
    monkeypatch.setenv("SHELL", "/bin/bash")
    (Path.home() / ".profile").write_text('export PATH="$HOME/.local/bin:$PATH"\n', encoding="utf-8")
    if bash_profile:
        (Path.home() / ".bash_profile").write_text("alias ll='ls -l'\n", encoding="utf-8")

    assert doctor.run_doctor(Namespace(fix=False)) == rc

    out = capsys.readouterr().out
    assert verdict in out
    assert "~/.profile" in out


@pytest.mark.platforms("posix")
def test_command_installation_path_check_accepts_trailing_slash_or_other_path_symlink(tmp_path, monkeypatch, capsys):
    _project, launcher, link_dir = _bundled_tree(tmp_path, monkeypatch, on_path=False)
    (link_dir / "hermes").symlink_to(launcher)
    monkeypatch.setenv("PATH", f"{link_dir}/:/usr/bin")

    rc_slash = doctor.run_doctor(Namespace(fix=False))
    assert rc_slash == 0
    assert "is not on your PATH" not in capsys.readouterr().out

    alt_bin = tmp_path / "usr-local-bin"
    alt_bin.mkdir()
    (alt_bin / "hermes").symlink_to(launcher)
    monkeypatch.setenv("PATH", f"{alt_bin}:/usr/bin")

    rc_alt = doctor.run_doctor(Namespace(fix=False))
    assert rc_alt == 0
    assert "is not on your PATH" not in capsys.readouterr().out


@pytest.mark.platforms("posix")
def test_source_checkout_with_sibling_bin_is_not_treated_as_bundled(tmp_path, monkeypatch, capsys):
    project, home, command = _tree(tmp_path, monkeypatch)
    _pm_source(project, home)
    _generation(project)
    decoy = project.parent / "bin" / "hermes"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    decoy.chmod(0o755)
    command.symlink_to(decoy)

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 1
    assert "points to wrong target" in out


@pytest.mark.platforms("posix")
def test_source_install_warns_when_local_bin_missing_from_path_even_when_launcher_exists(tmp_path, monkeypatch, capsys):
    project, home, command = _tree(tmp_path, monkeypatch)
    _generation(project)
    _pm_source(project, home)
    assert _launchers.stage_launcher("hermes", project, command.parent) == command
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    monkeypatch.setenv("SHELL", "/bin/bash")

    rc = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert rc == 1
    assert "~/.local/bin is not on your PATH" in out

    (tmp_path / ".bashrc").write_text(
        '# Hermes Agent command\n'
        'case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac\n',
        encoding="utf-8",
    )
    rc_after_rc = doctor.run_doctor(Namespace(fix=False))
    out_after_rc = capsys.readouterr().out
    assert rc_after_rc == 0
    assert "⚠" not in out_after_rc
    assert "configured in" in out_after_rc
