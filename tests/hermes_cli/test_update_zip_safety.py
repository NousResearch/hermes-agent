"""Regression coverage for Windows update fallback safety (#87304)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


def _fake_main(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        PROJECT_ROOT=root,
        _is_windows=lambda: True,
        sys=sys,
    )


def test_dependency_failure_is_not_classified_as_git_update() -> None:
    """A failed uv/pip command must not authorize the ZIP fallback."""
    assert not update_cmd._is_git_update_failure(
        subprocess.CalledProcessError(
            2,
            ["C:/hermes/venv/Scripts/uv.exe", "pip", "install", "-e", "."],
        )
    )
    assert not update_cmd._is_git_update_failure(
        subprocess.CalledProcessError(2, ["python", "-m", "pip", "install", "."])
    )


def test_git_failure_is_classified_for_windows_fallback() -> None:
    assert update_cmd._is_git_update_failure(
        subprocess.CalledProcessError(1, ["C:/Program Files/Git/bin/git.exe", "rev-list"])
    )
    assert update_cmd._is_git_update_failure(
        subprocess.CalledProcessError(1, "git fetch origin main")
    )


def test_fallback_policy_allows_only_windows_git_failures(monkeypatch, tmp_path):
    fake_main = _fake_main(tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)

    git_failure = subprocess.CalledProcessError(1, ["git.exe", "fetch"])
    dependency_failure = subprocess.CalledProcessError(
        2, ["uv.exe", "pip", "install", "-e", "."]
    )
    assert update_cmd._should_zip_fallback_on_update_error(git_failure) is True
    assert update_cmd._should_zip_fallback_on_update_error(dependency_failure) is False

    fake_main._is_windows = lambda: False
    assert update_cmd._should_zip_fallback_on_update_error(git_failure) is False


def test_fallback_policy_rejects_non_process_errors(monkeypatch, tmp_path):
    fake_main = _fake_main(tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    assert update_cmd._should_zip_fallback_on_update_error(OSError("locked")) is False


def test_zip_fallback_refuses_uncommitted_and_untracked_files(tmp_path, monkeypatch, capsys):
    """The guard must fail before a ZIP replacement can delete source-tree work."""
    (tmp_path / ".git").mkdir()
    fake_main = _fake_main(tmp_path)
    fake_main._resolve_update_branch = lambda _args: "main"
    fake_main._capture_active_tool_dependencies = lambda: {}
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=" M hermes_cli/update_cmd.py\n?? scratch/new-tool.py\n",
            stderr="",
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "urllib.request.urlretrieve",
        lambda *_args, **_kwargs: pytest.fail("dirty ZIP fallback must not download"),
    )

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._update_via_zip(SimpleNamespace())

    assert exc_info.value.code == 1
    assert calls == [
        [
            "git",
            "-c",
            "windows.appendAtomically=false",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored=all",
        ]
    ]
    output = capsys.readouterr().out
    assert "local changes" in output
    assert "uncommitted, untracked, or ignored" in output
    assert "scratch/new-tool.py" in output


def test_zip_fallback_refuses_ignored_source_files_but_allows_preserved_dirs(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / ".git").mkdir()
    fake_main = _fake_main(tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="!! .cache/local-source.py\n!! .env\n!! ./venv/Lib/site-packages/pkg.py\n",
            stderr="",
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._ensure_zip_update_checkout_is_clean()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert ".cache/local-source.py" in output
    assert ".env" not in output
    assert "venv/Lib/site-packages/pkg.py" not in output


def test_zip_fallback_allows_clean_git_checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    fake_main = _fake_main(tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    # No exception means the existing clean-checkout ZIP behavior remains
    # available; non-git installs are covered by the early return in the guard.
    update_cmd._ensure_zip_update_checkout_is_clean()


def test_zip_guard_fails_closed_when_git_status_fails(tmp_path, monkeypatch, capsys):
    (tmp_path / ".git").mkdir()
    fake_main = _fake_main(tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._ensure_zip_update_checkout_is_clean()

    assert exc_info.value.code == 1
    assert "cannot verify checkout state" in capsys.readouterr().out
