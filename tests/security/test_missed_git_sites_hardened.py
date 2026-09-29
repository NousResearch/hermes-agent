"""The four git call sites #126017 found outside the hardened set.

Every Hermes-initiated git spawn must run under :func:`noninteractive_git_env`
(and :func:`harden_git_argv` for diff-rendering commands) since #101483 /
GHSA-7x36-8jrh-v4pw: a repository delivered as files can carry a
``.git/config`` naming a command in ``core.fsmonitor`` or ``core.hooksPath``,
and Hermes runs some git paths automatically — before any trust gate. Four
sites were missed:

* ``tools/async_delegation_recovery_hints.git_state_hint`` — ``status``/``log``
  on abandoned-delegation recovery
* ``hermes_cli/kanban_db_workspace._git`` — ``worktree add`` for task worktrees
  (also runs the repo's hooks)
* ``tui_gateway/methods_complete_helpers._git_repo_files`` — ``ls-files`` on
  ``@``/Cmd-P path completion
* ``hermes_cli/worktree_gc._git`` — ``status`` + worktree/branch plumbing in
  the ``/worktree gc`` audit

Same real-repo fixture shape as ``test_gitspawn_config_injection.py``: a repo
whose ``.git/config`` arms fsmonitor and a post-checkout hook; a fired sink
leaves a marker file on disk. Skipped when ``git`` is unavailable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_HAS_GIT = __import__("shutil").which("git") is not None
pytestmark = pytest.mark.skipif(not _HAS_GIT, reason="git not installed")


def _make_malicious_repo(tmp: Path) -> tuple[Path, Path]:
    """Repo whose .git/config arms fsmonitor and a post-checkout hook.

    Returns (repo, marker): a fired sink leaves ``<marker>.<sink>`` on disk.
    Windows note: the hook needs a POSIX shell, so the hook sink is only
    asserted on non-Windows; fsmonitor alone is enough to catch all four sites.
    """
    repo = tmp / "poc"
    clean = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=clean)
    (repo / "README").write_text("hi\n")
    ident = ["-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run(["git", "-C", str(repo), *ident, "add", "."], check=True, env=clean)
    subprocess.run(["git", "-C", str(repo), *ident, "commit", "-qm", "init"], check=True, env=clean)

    marker = tmp / "MARKER"
    hooks = repo / "evil-hooks"
    hooks.mkdir()
    hook = hooks / "post-checkout"
    marker_shell = marker.as_posix()
    hook.write_text(f"#!/bin/sh\ntouch '{marker_shell}.hook'\n")
    hook.chmod(0o755)
    settings = {
        "core.fsmonitor": f"touch '{marker_shell}.fsmonitor'",
        "core.hooksPath": hooks.as_posix(),
    }
    for key, value in settings.items():
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True, env=clean)
    return repo, marker


def _fired(marker: Path) -> list[str]:
    out = []
    for sink in ("fsmonitor", "hook"):
        p = Path(f"{marker}.{sink}")
        if p.exists():
            out.append(sink)
            p.unlink()
    return out


@pytest.fixture()
def malicious_repo(tmp_path):
    repo, marker = _make_malicious_repo(tmp_path)
    yield repo, marker


def test_recovery_hints_git_is_safe(malicious_repo):
    from tools.async_delegation_recovery_hints import git_state_hint

    repo, marker = malicious_repo
    hint = git_state_hint(str(repo))
    assert hint, "the hint should still be produced on a healthy repo"
    assert _fired(marker) == []


def test_kanban_worktree_git_is_safe(malicious_repo, tmp_path):
    from hermes_cli.kanban_db_workspace import _git

    repo, marker = malicious_repo
    target = tmp_path / "wt"
    proc = _git(repo, "worktree", "add", str(target), "HEAD", timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert (target / "README").is_file(), "the worktree should still materialize"
    fired = _fired(marker)
    if os.name != "nt":
        assert fired == []


def test_completion_repo_files_is_safe(malicious_repo):
    """_git_repo_files normally runs rebound onto server.py's globals (bind_module):
    bare ``os``/``subprocess`` resolve to server's imports. Rebind it onto a stub with
    the real modules the same way, then exercise it."""
    import os
    import subprocess
    import types

    from tui_gateway.method_ctx import rebind
    from tui_gateway import methods_complete_helpers as helpers

    fake_server = types.SimpleNamespace()
    fake_server.os = os
    fake_server.subprocess = subprocess
    _git_repo_files = rebind(helpers._git_repo_files, vars(fake_server))

    repo, marker = malicious_repo
    files = list(_git_repo_files(str(repo)))
    assert "README" in files, "ls-files should still list tracked files"
    assert _fired(marker) == []


def test_worktree_gc_git_is_safe(malicious_repo):
    from hermes_cli.worktree_gc import _git

    repo, marker = malicious_repo
    proc = _git(["status", "--porcelain"], cwd=str(repo), timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert _fired(marker) == []
