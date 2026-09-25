"""``hermes update`` from a PM environment's venv must not advise a reinstall (#122627).

A committed PM environment's venv imports ``hermes_cli`` from the generation's
``workspace/`` snapshot — a build copy that deliberately carries no ``.git``. Before
this fix both git gates (``_cmd_update_check`` and ``_prepare_git_command``) reported
the generic "Not a git repository. Please reinstall" advice, which points users at a
healthy install. These tests pin the PM-workspace guidance and the unchanged generic
message for every other .git-less tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import update_cmd


def _make_pm_workspace(tmp_path: Path, key: str = "0123456789abcdef") -> tuple[Path, Path]:
    """Build ``<installs>/<install-key>/environments/<gen>/workspace`` and return it
    next to its fake installs root, matching pm.environments' layout."""
    installs = tmp_path / "installs"
    generation = installs / key / "environments" / "gen1"
    workspace = generation / "workspace"
    workspace.mkdir(parents=True)
    (generation / "venv").mkdir()
    (generation / ".lease-managed").touch()
    return workspace, installs


# ---------- apply path (_prepare_git_command) ----------

def test_prepare_git_command_reports_pm_workspace_instead_of_reinstall(
    monkeypatch, tmp_path, capsys
):
    workspace, installs = _make_pm_workspace(tmp_path)
    monkeypatch.setattr("pm.environments.installs_root", lambda: installs)
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", workspace)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._prepare_git_command()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "PM environment's workspace" in out
    assert "Please reinstall" not in out
    assert "install.sh" not in out


def test_prepare_git_command_keeps_generic_reinstall_advice_for_other_trees(
    monkeypatch, tmp_path, capsys
):
    bare = tmp_path / "plain-install"
    bare.mkdir()
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", bare)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._prepare_git_command()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Not a git repository. Please reinstall:" in out


# ---------- check path (_cmd_update_check) ----------

def test_check_gate_reports_pm_workspace_instead_of_generic_error(
    monkeypatch, tmp_path, capsys
):
    workspace, installs = _make_pm_workspace(tmp_path)
    monkeypatch.setattr("pm.environments.installs_root", lambda: installs)
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", workspace)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda: "git")

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._cmd_update_check(branch="main")

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "PM environment's workspace" in out
    assert "cannot check for updates" not in out


def test_check_gate_keeps_generic_error_for_other_trees(monkeypatch, tmp_path, capsys):
    bare = tmp_path / "plain-install"
    bare.mkdir()
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", bare)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda: "git")

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._cmd_update_check(branch="main")

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "✗ Not a git repository — cannot check for updates." in out


# ---------- shape self-check (_pm_workspace_guidance) ----------

def test_guidance_rejects_workspace_without_generation_siblings(monkeypatch, tmp_path):
    """A directory named ``workspace`` with no venv/.lease-managed sibling is not PM."""
    workspace, installs = _make_pm_workspace(tmp_path)
    (workspace.parent / "venv").rmdir()
    (workspace.parent / ".lease-managed").unlink()
    monkeypatch.setattr("pm.environments.installs_root", lambda: installs)

    assert update_cmd._pm_workspace_guidance(workspace) is None


def test_guidance_rejects_generation_outside_pm_installs_root(monkeypatch, tmp_path):
    """A matching shape that lives outside PM's installs root is not a PM workspace."""
    workspace, installs = _make_pm_workspace(tmp_path)
    monkeypatch.setattr("pm.environments.installs_root", lambda: tmp_path / "elsewhere")

    assert update_cmd._pm_workspace_guidance(workspace) is None


def test_guidance_rejects_non_hex_install_key(monkeypatch, tmp_path):
    """The key directory above ``environments`` must carry install-key shape."""
    workspace, installs = _make_pm_workspace(tmp_path, key="not-an-install-key")
    monkeypatch.setattr("pm.environments.installs_root", lambda: installs)

    assert update_cmd._pm_workspace_guidance(workspace) is None


def test_guidance_rejects_non_workspace_name(tmp_path):
    assert update_cmd._pm_workspace_guidance(tmp_path) is None
