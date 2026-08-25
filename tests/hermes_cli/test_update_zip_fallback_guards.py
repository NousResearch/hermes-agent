"""ZIP fallback must not fire on dependency failures or clobber a dirty tree.

Issue #87304: on Windows the update ``try`` spans git pull *and* ``uv pip
install``. A locked ``hermes.exe`` makes the install exit 2, the handler
prints ``Git update failed``, and ``_update_via_zip`` replaces every
top-level entry except ``venv`` / ``node_modules`` / ``.git`` / ``.env`` —
permanently deleting uncommitted edits and untracked files. The git pull
has already succeeded by then, so the ZIP cannot fix the actual failure.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


def _cpe(cmd, returncode=2, stderr="", stdout="") -> subprocess.CalledProcessError:
    exc = subprocess.CalledProcessError(returncode, cmd)
    exc.stderr = stderr
    exc.stdout = stdout
    return exc


# ---------------------------------------------------------------------------
# Stage classification + ZIP gating
# ---------------------------------------------------------------------------


def test_uv_pip_install_is_a_dependency_failure_not_git():
    exc = _cpe([r"C:\venv\Scripts\uv.exe", "pip", "install", "-e", "."])
    assert update_cmd._called_process_error_is_git(exc) is False
    assert update_cmd._called_process_error_is_python_dep_install(exc) is True
    assert update_cmd._format_update_failure_stage(exc) == (
        "Python dependency install failed"
    )


def test_venv_pip_install_is_a_dependency_failure():
    exc = _cpe([r"C:\venv\Scripts\python.exe", "-m", "pip", "install", "-e", "."])
    assert update_cmd._called_process_error_is_python_dep_install(exc) is True
    assert update_cmd._called_process_error_is_git(exc) is False


def test_ensurepip_is_a_dependency_failure():
    exc = _cpe([r"C:\venv\Scripts\python.exe", "-m", "ensurepip", "--upgrade"])
    assert update_cmd._called_process_error_is_python_dep_install(exc) is True
    assert update_cmd._format_update_failure_stage(exc) == (
        "Python dependency install failed"
    )


def test_git_pull_is_classified_as_git():
    exc = _cpe(["git", "-c", "windows.appendAtomically=false", "pull"], returncode=1)
    assert update_cmd._called_process_error_is_git(exc) is True
    assert update_cmd._called_process_error_is_python_dep_install(exc) is False
    assert update_cmd._format_update_failure_stage(exc) == "Git update failed"


def test_git_exe_path_is_still_git():
    exc = _cpe([r"C:\Program Files\Git\cmd\git.exe", "fetch", "origin", "main"])
    assert update_cmd._called_process_error_is_git(exc) is True


def test_unknown_command_gets_generic_stage():
    exc = _cpe(["npm", "install"], returncode=1)
    assert update_cmd._format_update_failure_stage(exc) == "Update step failed"


def test_windows_dep_failure_does_not_zip_fallback(monkeypatch):
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    exc = _cpe([r"C:\venv\Scripts\uv.exe", "pip", "install", "-e", "."])
    assert update_cmd._should_zip_fallback_on_update_error(exc) is False


def test_windows_git_failure_still_zips(monkeypatch):
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    exc = _cpe(["git", "pull"], returncode=1)
    assert update_cmd._should_zip_fallback_on_update_error(exc) is True


def test_posix_git_failure_does_not_zip(monkeypatch):
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    exc = _cpe(["git", "pull"], returncode=1)
    assert update_cmd._should_zip_fallback_on_update_error(exc) is False


def test_error_tail_prints_last_lines(capsys):
    stderr = "\n".join(f"line-{i}" for i in range(20))
    exc = _cpe(["uv", "pip", "install"], stderr=stderr)
    update_cmd._print_called_process_error_tail(exc)
    out = capsys.readouterr().out
    assert "Last output:" in out
    assert "line-19" in out
    assert "line-0" not in out
    assert "line-7" not in out
    assert "line-8" in out


# ---------------------------------------------------------------------------
# Dirty-tree overlay guard
# ---------------------------------------------------------------------------


def _porcelain_run(stdout: str, returncode: int = 0):
    def fake_run(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        if "status" in joined and "--porcelain" in joined:
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


def test_zip_overlay_allowed_without_git(tmp_path):
    assert update_cmd._zip_overlay_block_reason(tmp_path) is None


def test_zip_overlay_blocked_on_modified_file(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        update_cmd.subprocess, "run", _porcelain_run(" M hermes_cli/update_cmd.py\n")
    )
    reason = update_cmd._zip_overlay_block_reason(tmp_path)
    assert reason is not None
    assert "uncommitted" in reason


def test_zip_overlay_blocked_on_untracked_file(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(update_cmd.subprocess, "run", _porcelain_run("?? notes.md\n"))
    reason = update_cmd._zip_overlay_block_reason(tmp_path)
    assert reason is not None
    assert "untracked" in reason


def test_zip_overlay_blocked_when_git_status_fails(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        _porcelain_run("", returncode=128),
    )
    reason = update_cmd._zip_overlay_block_reason(tmp_path)
    assert reason is not None
    assert "could not check" in reason


def test_zip_overlay_allowed_on_clean_git_checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(update_cmd.subprocess, "run", _porcelain_run(""))
    assert update_cmd._zip_overlay_block_reason(tmp_path) is None


def test_update_via_zip_aborts_before_download_when_dirty(
    tmp_path, monkeypatch, capsys
):
    """The live tree must not be touched, and the ZIP must not be fetched."""
    fake_root = tmp_path / "install"
    fake_root.mkdir()
    (fake_root / ".git").mkdir()
    local = fake_root / "keep-me.txt"
    local.write_text("local work\n", encoding="utf-8")
    untracked_dir = fake_root / "agent" / "scratch"
    untracked_dir.mkdir(parents=True)
    (untracked_dir / "wip.py").write_text("print('wip')\n", encoding="utf-8")

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        _porcelain_run(" M keep-me.txt\n?? agent/scratch/wip.py\n"),
    )

    with patch("urllib.request.urlretrieve") as download:
        with pytest.raises(SystemExit) as exc_info:
            hermes_main._update_via_zip(SimpleNamespace(branch=None))

    assert exc_info.value.code == 1
    download.assert_not_called()
    assert local.read_text(encoding="utf-8") == "local work\n"
    assert (untracked_dir / "wip.py").read_text(encoding="utf-8") == "print('wip')\n"
    out = capsys.readouterr().out
    assert "ZIP fallback refused" in out
    assert "Downloading latest version" not in out


# ---------------------------------------------------------------------------
# Pre-swap TOCTOU re-check
# ---------------------------------------------------------------------------


def test_status_uses_untracked_files_all(tmp_path, monkeypatch):
    """A user git config hiding untracked files must not blind the guard."""
    (tmp_path / ".git").mkdir()
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    update_cmd._zip_overlay_block_reason(tmp_path)
    assert seen and "--untracked-files=all" in seen[0]


def test_staging_artifact_lines_are_recognized():
    is_artifact = update_cmd._is_zip_staging_artifact_status_line
    assert is_artifact("?? agent.hermes-update-staging/")
    assert is_artifact("?? cli.py.hermes-update-staging")
    assert is_artifact("?? tools.hermes-update-old/")
    # Nested user files under a staging-lookalike directory don't match the
    # top-level test only when the TOP level itself is not an artifact.
    assert not is_artifact("?? agent/scratch/wip.py")
    assert not is_artifact(" M hermes_cli/update_cmd.py")
    assert not is_artifact("?? notes.hermes-update-staging.txt")


def test_recheck_ignores_own_staging_artifacts(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        _porcelain_run("?? agent.hermes-update-staging/\n?? cli.py.hermes-update-old\n"),
    )
    assert (
        update_cmd._zip_overlay_block_reason(tmp_path, ignore_staging_artifacts=True)
        is None
    )
    # Without the flag the same output still refuses (pre-download check).
    assert update_cmd._zip_overlay_block_reason(tmp_path) is not None


def test_recheck_still_blocks_user_files_amid_staging_artifacts(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        _porcelain_run("?? agent.hermes-update-staging/\n?? my-notes.md\n"),
    )
    reason = update_cmd._zip_overlay_block_reason(
        tmp_path, ignore_staging_artifacts=True
    )
    assert reason is not None
