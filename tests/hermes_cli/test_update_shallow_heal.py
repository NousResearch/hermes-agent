"""A shallow checkout is unshallowed (commit graph only) before the updater's bounded fetch.

From a depth-1 install far behind main, a plain ``git fetch origin main`` drags in ~the whole
history and cannot finish inside the 300s network cap (#123254); a ``--depth`` pre-fetch grafts
origin/main so merge-base reports orphan divergence (#123346). Real local git repositories.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import os

import pytest

from hermes_cli.gitlock import convert_treeless_checkout, heal_shallow_history


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def _origin(tmp_path: Path, commits: int) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "uploadpack.allowFilter", "true")
    for i in range(commits):
        (origin / "f.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(origin, "add", "f.txt")
        _git(origin, "commit", "-qm", f"c{i}")
    return origin


def test_depth1_clone_of_a_tagless_origin_gets_its_commit_graph(tmp_path):
    origin = _origin(tmp_path, 3)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{origin}", str(clone))

    assert heal_shallow_history(clone, "main") is True

    assert _git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"
    assert _git(clone, "rev-list", "--count", "HEAD").stdout.strip() == "3"
    # History really was missing: fetched as the blobless partial clone installers make (#129712).
    assert _git(clone, "config", "remote.origin.partialclonefilter").stdout.strip() == "blob:none"
    # The clone's own depth-1 pack is a partial-clone pack now too, or git 2.53+ crashes every
    # later fetch in pack-objects (#124272).
    packs = list((clone / ".git" / "objects" / "pack").glob("pack-*.pack"))
    assert packs and all(p.with_suffix(".promisor").exists() for p in packs)


def test_depth_prefetch_on_a_full_clone_restores_ancestry_and_keeps_it_full(tmp_path):
    origin = _origin(tmp_path, 2)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    (origin / "f.txt").write_text("next\n", encoding="utf-8")
    _git(origin, "commit", "-qam", "next")
    _git(clone, "fetch", "-q", "--depth", "1", "origin", "main")
    assert _git(clone, "merge-base", "HEAD", "origin/main").returncode != 0

    assert heal_shallow_history(clone, "main") is True

    assert _git(clone, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode == 0
    assert _git(clone, "config", "--get", "remote.origin.promisor").returncode != 0


def test_treeless_checkout_gets_every_tree_once_so_history_walks_stay_offline(tmp_path):
    origin = _origin(tmp_path, 5)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--filter=tree:0", f"file://{origin}", str(clone))
    offline = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    walk = ["git", "log", "--format=%H", "--", "f.txt"]
    assert subprocess.run(walk, cwd=clone, env=offline, capture_output=True).returncode != 0

    assert convert_treeless_checkout(clone) is True

    assert _git(clone, "config", "remote.origin.partialclonefilter").stdout.strip() == "blob:none"
    walked = subprocess.run(walk, cwd=clone, env=offline, capture_output=True, text=True)
    assert walked.returncode == 0 and len(walked.stdout.split()) == 5
    assert convert_treeless_checkout(clone) is False


def test_failed_conversion_keeps_the_checkout_treeless_for_a_retry(tmp_path):
    origin = _origin(tmp_path, 2)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--filter=tree:0", f"file://{origin}", str(clone))
    _git(clone, "remote", "set-url", "origin", f"file://{tmp_path / 'gone'}")

    with pytest.raises(subprocess.CalledProcessError):
        convert_treeless_checkout(clone)

    assert _git(clone, "config", "remote.origin.partialclonefilter").stdout.strip() == "tree:0"
