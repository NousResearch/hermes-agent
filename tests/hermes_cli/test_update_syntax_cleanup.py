"""Scratch cleanup must not change the post-pull syntax verdict (#129238)."""

import errno
import os
import subprocess

import pytest

from hermes_cli import main, update_cmd


@pytest.mark.parametrize("broken", [False, True])
def test_pull_preserves_syntax_verdict_when_scratch_cleanup_fails(tmp_path, monkeypatch, broken):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "hermes_constants.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "working")
    previous = git("rev-parse", "HEAD")
    source.write_text("<<<<<<< HEAD\n" if broken else "VALUE = 2\n", encoding="utf-8")
    git("commit", "-am", "upstream")
    upstream = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", upstream)
    git("reset", "--hard", previous)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    real_rmdir = os.rmdir
    cleanup_attempts = []

    def busy_scratch(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)).startswith("hermes-syntax-check-"):
            cleanup_attempts.append(path)
            raise OSError(errno.ENOTEMPTY, "scratch directory still pending deletion", path)
        return real_rmdir(path, *args, **kwargs)

    with monkeypatch.context() as cleanup_patch:
        cleanup_patch.setattr(os, "rmdir", busy_scratch)
        outcome = None
        try:
            update_cmd._pull_updates(
                ["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
                discard_local_changes=False, keep_stash=False,
            )
        except (OSError, SystemExit) as exc:
            outcome = exc

    # Clean the intentionally retained, empty scratch directories outside the fault.
    for path in set(cleanup_attempts):
        real_rmdir(path)
    assert cleanup_attempts, "the real scratch directory teardown must be exercised"
    assert not isinstance(outcome, OSError), f"cleanup replaced the syntax verdict: {outcome}"
    if broken:
        assert isinstance(outcome, SystemExit) and outcome.code == 1
        assert git("rev-parse", "HEAD") == previous
    else:
        assert outcome is None
        assert git("rev-parse", "HEAD") == upstream
    assert not (tmp_path / "__pycache__").exists()
