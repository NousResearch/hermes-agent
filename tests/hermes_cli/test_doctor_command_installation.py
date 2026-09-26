"""Command Installation doctor check must judge entry points per execution layout.

A PM-managed runtime can execute doctor FROM a workspace snapshot. pm.workspace
never copies the extensionless root ``hermes`` launcher into a workspace, so the
check must accept the environment's own console script as the entry point there
instead of demanding a file that layout cannot contain.
"""

from pathlib import Path


def _install_layout(monkeypatch, tmp_path, *, root_launcher, venv_script, pm):
    """Point the check at a synthetic install layout; return the doctor module."""
    from hermes_cli import doctor_platform

    project = tmp_path / "project"
    venv = tmp_path / "env-venv"
    (project / "hermes_cli").mkdir(parents=True)
    venv.mkdir()
    if root_launcher:
        (project / "hermes").write_text("#!/bin/sh\n")
    if venv_script:
        (venv / "bin").mkdir()
        (venv / "bin" / "hermes").write_text("#!/bin/sh\n")

    monkeypatch.setattr("hermes_cli.doctor.PROJECT_ROOT", project)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda root: "git")
    monkeypatch.setattr("pm.environments.base_venv", lambda root: tmp_path / "base-venv")
    monkeypatch.setattr("pm.environments.selected_venv", lambda root: venv)
    if pm:
        monkeypatch.setattr(
            "hermes_cli._launchers.resolve_store_python", lambda root: tmp_path / "store-python"
        )
    else:
        monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root: None)
    # Isolate the ~/.local/bin symlink expectations from the real home.
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    (tmp_path / "fakehome").mkdir()
    return doctor_platform


def test_workspace_run_accepts_the_environment_console_script(tmp_path, capsys, monkeypatch):
    doctor_platform = _install_layout(monkeypatch, tmp_path, root_launcher=False, venv_script=True, pm=True)
    f = doctor_platform._check_command_installation(False)
    out = capsys.readouterr().out

    assert f"Hermes entry point exists ({tmp_path / 'env-venv' / 'bin' / 'hermes'})" in out
    assert "entry point not found" not in out
    assert f.manual_issues == [], "a PM workspace layout was reported as a broken launcher"


def test_checkout_run_keeps_using_the_root_launcher(tmp_path, capsys, monkeypatch):
    doctor_platform = _install_layout(monkeypatch, tmp_path, root_launcher=True, venv_script=False, pm=True)
    f = doctor_platform._check_command_installation(False)
    out = capsys.readouterr().out

    assert f"Hermes entry point exists ({tmp_path / 'project' / 'hermes'})" in out
    assert f.manual_issues == []


def test_missing_everywhere_still_reports_the_manual_issue(tmp_path, capsys, monkeypatch):
    doctor_platform = _install_layout(monkeypatch, tmp_path, root_launcher=False, venv_script=False, pm=True)
    f = doctor_platform._check_command_installation(False)
    out = capsys.readouterr().out

    assert "Hermes entry point not found" in out
    assert any("launcher" in i for i in f.manual_issues), "a genuinely missing launcher raised no issue"


def test_non_pm_venv_run_checks_the_venv_script(tmp_path, capsys, monkeypatch):
    doctor_platform = _install_layout(monkeypatch, tmp_path, root_launcher=False, venv_script=True, pm=False)
    f = doctor_platform._check_command_installation(False)
    out = capsys.readouterr().out

    assert f"Hermes entry point exists ({tmp_path / 'env-venv' / 'bin' / 'hermes'})" in out
    assert f.manual_issues == []