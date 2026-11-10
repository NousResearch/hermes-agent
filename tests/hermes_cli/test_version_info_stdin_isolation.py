"""Regression for #152551: Git provenance must not inherit managed-worker stdin.

The managed worker reserves stdin for bootstrap and control frames. Both the
version lookup and its dirty-status probe must give Git an independent EOF
rather than the worker's live control pipe (reported by roelvde).
"""

import shutil
import subprocess

import pytest

from hermes_cli.version_info import _git_version_info, _run_git


def test_git_identity_probes_do_not_inherit_worker_control_stdin(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is required for a provenance probe")

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(repo)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )

    real_popen = subprocess.Popen
    git_children = []

    def observe_child(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "git":
            git_children.append((tuple(argv), kwargs.get("stdin")))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", observe_child)

    assert _run_git(repo, "rev-parse", "--git-dir")
    assert _git_version_info(repo).source == "git"

    assert any(command[:2] == ("git", "rev-parse") for command, _ in git_children)
    assert any("--no-optional-locks" in command for command, _ in git_children)
    assert all(stdin == subprocess.DEVNULL for _, stdin in git_children), (
        "a Git identity/status child inherited the managed worker's stdin control pipe: "
        f"{git_children!r}"
    )
