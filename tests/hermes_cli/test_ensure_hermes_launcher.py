"""`hermes update` / `hermes doctor --fix` self-heal the primary ``hermes`` launcher.

Issue #76421: one layout-aware check/repair helper
(``hermes_cli.launcher_repair``) shared by the updater and doctor. Repairs
missing launchers, dangling/wrong legacy symlinks, and recognized stale
Hermes-managed wrappers in the ONE command dir the installer selects;
preserves user-managed wrappers with a warning; never writes through a
symlink (#21454).  The rewritten shim is the exact installer form: venv
interpreter + checked-in ``hermes`` entrypoint (NOT the uv console script,
which needs ``realpath`` — absent on stock macOS).

All fixtures are isolated under tmp_path — bin dirs and the command-dir
selector are always injected, so the real ``~/.local/bin`` and
``/usr/local/bin`` are never touched.
"""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

def _lr():
    """Lazy import: on the pre-fix tree launcher_repair does not exist, and
    sabotage must record real test FAILURES (ImportError inside the test
    body), not a collection error."""
    from hermes_cli import launcher_repair as m
    return m


@pytest.fixture
def project(tmp_path):
    """A fake project checkout: entrypoint + venv interpreter + console script."""
    root = tmp_path / "project"
    entrypoint = root / "hermes"
    root.mkdir(parents=True)
    entrypoint.write_text("#!/usr/bin/env python\n# entrypoint\n", encoding="utf-8")
    entrypoint.chmod(0o755)
    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_bytes(b"")
    venv_python.chmod(0o755)
    console_script = venv_bin / "hermes"
    console_script.write_text("#!/usr/bin/env python\n# console script\n", encoding="utf-8")
    console_script.chmod(0o755)
    return root


@pytest.fixture
def bin_dir(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    return d


def _venv_python(project: Path) -> Path:
    return project / "venv" / "bin" / "python"


def _entrypoint(project: Path) -> Path:
    return project / "hermes"


def _installer_shim(project: Path) -> str:
    """The exact form scripts/install.sh writes (venv python + entrypoint)."""
    return _lr().managed_shim_text(_venv_python(project), _entrypoint(project))


def _ensure(project: Path, **kwargs):
    return _lr().ensure_hermes_launcher(
        _venv_python(project), _entrypoint(project), project, **kwargs
    )


# ---------------------------------------------------------------------------
# managed_launcher_status / repair_launcher / ensure_hermes_launcher
# ---------------------------------------------------------------------------


def test_missing_launcher_created_as_installer_form_shim(bin_dir, project):
    launcher = bin_dir / "hermes"
    assert _lr().managed_launcher_status(launcher, project) == _lr().MISSING

    actions = _ensure(project, bin_dirs=[bin_dir])

    assert launcher.is_file() and not launcher.is_symlink()
    assert launcher.read_text(encoding="utf-8") == _installer_shim(project)
    shim = launcher.read_text(encoding="utf-8")
    # Installer form: venv INTERPRETER + checked-in entrypoint, and the
    # inherited-Python-env guards — NOT the uv console script.
    assert f'exec "{_venv_python(project)}" "{_entrypoint(project)}" "$@"' in shim
    assert "unset PYTHONPATH" in shim and "unset PYTHONHOME" in shim
    assert os.access(launcher, os.X_OK)
    assert any("Installed" in a for a in actions)
    assert _lr().managed_launcher_status(launcher, project) == _lr().HEALTHY


def test_dangling_symlink_replaced_with_shim(bin_dir, project):
    launcher = bin_dir / "hermes"
    launcher.symlink_to(bin_dir / "gone")  # dangling legacy symlink
    assert _lr().managed_launcher_status(launcher, project) == _lr().WRONG_SYMLINK

    actions = _ensure(project, bin_dirs=[bin_dir])

    assert launcher.is_file() and not launcher.is_symlink()
    assert launcher.read_text(encoding="utf-8") == _installer_shim(project)
    assert os.access(launcher, os.X_OK)
    assert any("symlink" in a for a in actions)


def test_wrong_target_symlink_replaced_with_shim(bin_dir, project, tmp_path):
    launcher = bin_dir / "hermes"
    wrong = tmp_path / "wrong_hermes"
    wrong.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    launcher.symlink_to(wrong)
    assert _lr().managed_launcher_status(launcher, project) == _lr().WRONG_SYMLINK

    _ensure(project, bin_dirs=[bin_dir])

    assert launcher.is_file() and not launcher.is_symlink()
    assert launcher.read_text(encoding="utf-8") == _installer_shim(project)
    # Never written through the symlink: the wrong target is untouched.
    assert wrong.read_text(encoding="utf-8") == "#!/usr/bin/env python\n"


def test_symlink_into_current_project_left_alone(bin_dir, project):
    launcher = bin_dir / "hermes"
    launcher.symlink_to(project / "venv" / "bin" / "hermes")
    assert _lr().managed_launcher_status(launcher, project) == _lr().HEALTHY

    actions = _ensure(project, bin_dirs=[bin_dir])

    assert actions == []
    assert launcher.is_symlink()


def test_current_installer_form_wrapper_is_healthy(bin_dir, project):
    """macOS-shaped regression: the CURRENT installer form (venv/bin/python +
    checked-in entrypoint) must classify healthy and stay byte-identical —
    the console-script-based classifier used to call this stale."""
    launcher = bin_dir / "hermes"
    content = _installer_shim(project)
    launcher.write_text(content, encoding="utf-8")
    launcher.chmod(0o755)

    assert _lr().managed_launcher_status(launcher, project) == _lr().HEALTHY
    actions = _ensure(project, bin_dirs=[bin_dir])

    assert actions == []
    assert launcher.read_bytes() == content.encode()


def test_removed_venv_wrapper_is_stale_and_rewritten(bin_dir, project, tmp_path):
    """A recognized wrapper exec'ing a REMOVED venv is stale and gets
    rewritten into the current installer form."""
    old_root = tmp_path / "old_project"
    (old_root / "venv" / "bin").mkdir(parents=True)
    (old_root / "venv" / "bin" / "python").write_bytes(b"")
    (old_root / "hermes").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    launcher = bin_dir / "hermes"
    launcher.write_text(_lr().managed_shim_text(
        old_root / "venv" / "bin" / "python", old_root / "hermes"
    ), encoding="utf-8")
    launcher.chmod(0o755)
    shutil.rmtree(old_root)  # venv removed → exec target gone

    assert _lr().managed_launcher_status(launcher, project) == _lr().STALE_MANAGED
    actions = _ensure(project, bin_dirs=[bin_dir])

    assert launcher.is_file() and not launcher.is_symlink()
    assert launcher.read_text(encoding="utf-8") == _installer_shim(project)
    assert os.access(launcher, os.X_OK)
    assert any("stale" in a for a in actions)


def test_wrapper_pointing_at_other_project_is_stale(bin_dir, project, tmp_path):
    """Exec targets exist but are not under the current project root → stale."""
    other = tmp_path / "other_project"
    (other / "venv" / "bin").mkdir(parents=True)
    (other / "venv" / "bin" / "python").write_bytes(b"")
    (other / "hermes").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    launcher = bin_dir / "hermes"
    launcher.write_text(_lr().managed_shim_text(
        other / "venv" / "bin" / "python", other / "hermes"
    ), encoding="utf-8")

    assert _lr().managed_launcher_status(launcher, project) == _lr().STALE_MANAGED


def test_user_wrapper_preserved_with_warning(bin_dir, project):
    launcher = bin_dir / "hermes"
    content = "#!/bin/sh\n# my own hermes wrapper\nexec /opt/custom/hermes --mine\n"
    launcher.write_text(content, encoding="utf-8")
    assert _lr().managed_launcher_status(launcher, project) == _lr().USER_WRAPPER

    actions = _ensure(project, bin_dirs=[bin_dir])

    # Byte-identical, and a warning was surfaced.
    assert launcher.read_bytes() == content.encode()
    assert any("untouched" in a for a in actions)
    assert not any("✓" in a for a in actions)


def test_unrecognized_bash_wrapper_is_user_wrapper(bin_dir, project):
    """Conservatism: bash + exec but no managed markers → warn, keep."""
    launcher = bin_dir / "hermes"
    content = "#!/usr/bin/env bash\nexec /opt/custom/tool run\n"
    launcher.write_text(content, encoding="utf-8")

    assert _lr().managed_launcher_status(launcher, project) == _lr().USER_WRAPPER
    actions = _ensure(project, bin_dirs=[bin_dir])
    assert launcher.read_bytes() == content.encode()
    assert any("untouched" in a for a in actions)


def test_unwritable_bin_dir_skipped(bin_dir, project):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permissions")
    bin_dir.chmod(0o555)
    try:
        actions = _ensure(project, bin_dirs=[bin_dir])
        assert actions == []
        assert not (bin_dir / "hermes").exists()
    finally:
        bin_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# select_command_dir — the ONE layout, mirroring install.sh
# ---------------------------------------------------------------------------


def test_selector_termux_via_termux_version(tmp_path, monkeypatch):
    prefix = tmp_path / "termux" / "usr"
    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", str(prefix))
    assert _lr().select_command_dir() == prefix / "bin"


def test_selector_termux_via_prefix_path(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    assert _lr().select_command_dir() == Path("/data/data/com.termux/files/usr/bin")


def test_selector_fhs_when_root_on_linux(monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(_lr().sys, "platform", "linux")
    monkeypatch.setattr(_lr().os, "geteuid", lambda: 0)
    assert _lr().select_command_dir() == Path("/usr/local/bin")


def test_selector_user_local_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(_lr().os, "geteuid", lambda: 1000)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert _lr().select_command_dir() == tmp_path / ".local" / "bin"


def test_ensure_repairs_exactly_one_dir(bin_dir, project, tmp_path, monkeypatch):
    """No sweep: with the default dir selection, only the selector's dir is
    repaired — a launcher in any other candidate dir is left alone, so an
    FHS install can never gain a second launcher in ~/.local/bin."""
    other_dir = tmp_path / "other_bin"
    other_dir.mkdir()
    stale_content = _lr().managed_shim_text(
        tmp_path / "gone" / "venv" / "bin" / "python", tmp_path / "gone" / "hermes"
    )
    (other_dir / "hermes").write_text(stale_content, encoding="utf-8")

    monkeypatch.setattr(_lr(), "select_command_dir", lambda: bin_dir)
    actions = _ensure(project)  # default bin_dirs → [_lr().select_command_dir()]

    assert (bin_dir / "hermes").read_text(encoding="utf-8") == _installer_shim(project)
    assert (other_dir / "hermes").read_text(encoding="utf-8") == stale_content
    assert all(str(bin_dir) in a for a in actions)


# ---------------------------------------------------------------------------
# update_cmd wrapper
# ---------------------------------------------------------------------------


def test_update_wrapper_uses_installer_form(tmp_path, project, monkeypatch):
    """_ensure_hermes_launcher() resolves PROJECT_ROOT's venv python +
    entrypoint and repairs via the shared helper (selector sandboxed)."""
    from hermes_cli import main as hm
    from hermes_cli.update_cmd import _ensure_hermes_launcher

    bin_dir = tmp_path / ".local" / "bin"
    monkeypatch.setattr(hm, "PROJECT_ROOT", project)
    monkeypatch.setattr(_lr(), "select_command_dir", lambda: bin_dir)

    _ensure_hermes_launcher()

    launcher = bin_dir / "hermes"
    assert launcher.is_file() and not launcher.is_symlink()
    assert launcher.read_text(encoding="utf-8") == _installer_shim(project)
    assert os.access(launcher, os.X_OK)


# ---------------------------------------------------------------------------
# Wire-level: the no-new-commits update path calls the helper
# ---------------------------------------------------------------------------


def _git_side_effect(cmd, **kwargs):
    joined = " ".join(str(c) for c in cmd)
    if "rev-parse" in joined and "--abbrev-ref" in joined:
        return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
    if "rev-parse" in joined and "--verify" in joined:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if "rev-list" in joined:
        return subprocess.CompletedProcess(cmd, 0, stdout="0\n", stderr="")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_no_new_commits_update_calls_launcher_selfheal(capsys):
    """commit_count == 0 must still run the launcher self-heal before the
    'Already up to date!' return (issue #76421)."""
    from hermes_cli import main as hm
    from hermes_cli.main import cmd_update

    with patch("shutil.which", return_value=None), patch(
        "subprocess.run", side_effect=_git_side_effect
    ), patch(
        "hermes_cli.managed_uv.ensure_uv", return_value=None
    ), patch(
        "hermes_cli.managed_uv.update_managed_uv", return_value=None
    ), patch.object(
        hm, "_get_origin_url", return_value=None
    ), patch.object(
        hm, "_sync_with_upstream_if_needed"
    ), patch(
        "hermes_cli.update_cmd._ensure_hermes_launcher"
    ) as heal_mock:
        cmd_update(SimpleNamespace())

    heal_mock.assert_called_once_with()
    assert "Already up to date!" in capsys.readouterr().out
