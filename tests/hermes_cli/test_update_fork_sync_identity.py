"""The fork sync acts on ONE upstream commit (review R6 m2).

``_sync_with_upstream_if_needed`` fetches ``refs/remotes/upstream/main`` and fast-forwards to it. The
short name ``upstream/main`` also names a LOCAL branch of that name (git resolves ``refs/heads``
before ``refs/remotes``), so a count by short name and a merge by full ref could each see a
different commit: the sync printed "1 behind" and merged 2 commits.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hermes_cli.update_cmd_git import _sync_with_upstream_if_needed


def _env(home: Path) -> dict:
    return {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(home),
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8", env=_env(cwd)).stdout.strip()


def test_a_local_branch_named_upstream_main_never_changes_what_the_sync_counts_or_merges(
        tmp_path, monkeypatch, capsys):
    for name, value in _env(tmp_path).items():
        if name.startswith("GIT_") or name == "HOME":
            monkeypatch.setenv(name, value)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    commits = []
    for i in range(3):
        (upstream / "f.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(upstream, "add", "f.txt")
        _git(upstream, "commit", "-qm", f"c{i}")
        commits.append(_git(upstream, "rev-parse", "HEAD"))
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(upstream), str(origin))
    _git(origin, "update-ref", "refs/heads/main", commits[0])
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "remote", "add", "upstream", str(upstream))
    # The shadow: a local branch literally named upstream/main, one commit behind upstream.
    _git(clone, "fetch", "-q", "upstream")
    _git(clone, "branch", "upstream/main", commits[1])

    assert _sync_with_upstream_if_needed(["git"], clone, assume_yes=True) is True
    out = capsys.readouterr().out
    assert "Fork is 2 commit(s) behind upstream" in out, out
    assert _git(clone, "rev-parse", "HEAD") == commits[2]
