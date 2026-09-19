"""A pulled syntax error must restore the previous runnable Git tree."""

import subprocess
import sys

import pytest

from hermes_cli import main, update_cmd


def test_pull_rolls_back_broken_critical_file_and_accepts_corrected_retry(tmp_path, monkeypatch, capsys):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "hermes_constants.py"
    source.write_text("print('runnable')\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "working")
    previous = git("rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    assert update_cmd._capture_head_sha(["git"], tmp_path) == previous
    # Missing critical files are legitimate after refactors, not syntax failures.
    assert update_cmd._validate_critical_files_syntax(tmp_path) == (True, None, None)

    source.write_text("<<<<<<< HEAD\n", encoding="utf-8")
    git("commit", "-am", "broken upstream")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("reset", "--hard", previous)

    def pull():
        return update_cmd._pull_updates(
            ["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False,
        )

    with pytest.raises(SystemExit) as failure:
        pull()
    assert failure.value.code == 1
    assert "syntax error" in capsys.readouterr().out
    assert git("rev-parse", "HEAD") == previous
    assert subprocess.run([sys.executable, str(source)], capture_output=True, text=True, check=True).stdout == "runnable\n"

    git("reset", "--hard", "origin/main")
    source.write_text("print('corrected')\n", encoding="utf-8")
    git("commit", "-am", "corrected upstream")
    corrected = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", corrected)
    git("reset", "--hard", previous)
    assert pull() == previous
    assert git("rev-parse", "HEAD") == corrected
    assert subprocess.run([sys.executable, str(source)], capture_output=True, text=True, check=True).stdout == "corrected\n"