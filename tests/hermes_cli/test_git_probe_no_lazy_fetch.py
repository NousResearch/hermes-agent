"""Read-only git probes never lazy-fetch from a partial clone's promisor remote.

In a blobless clone, asking about an object the clone has not fetched (the fresh upstream tip the
startup update check compares against) makes git download it — for a real install, the whole
commit/tree history — and the probe's timeout kills only its own git, orphaning the fetch.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from hermes_cli import plugin_catalog, source_check
from hermes_cli._subprocess_compat import bounded_git_probe

_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _git(*args, cwd=None, env=_ENV):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, env=env,
                          check=True, capture_output=True, text=True).stdout.strip()


def _git_supports_no_lazy_fetch() -> bool:
    m = re.search(r"(\d+)\.(\d+)", _git("--version"))
    return bool(m) and (int(m[1]), int(m[2])) >= (2, 44)


pytestmark = pytest.mark.skipif(not _git_supports_no_lazy_fetch(), reason="GIT_NO_LAZY_FETCH needs git >= 2.44")


@pytest.fixture
def partial_clone(tmp_path: Path):
    """(clone, local_head, unfetched_upstream_tip) for a blobless clone of a local upstream."""
    seed, up, clone = tmp_path / "seed", tmp_path / "up.git", tmp_path / "clone"
    _git("init", "-q", "-b", "main", str(seed))
    for i in range(2):
        (seed / f"f{i}.txt").write_text(f"v{i}\n" * 50, encoding="utf-8")
        _git("add", "-A", cwd=seed)
        _git("commit", "-qm", f"c{i}", cwd=seed)
    _git("clone", "-q", "--bare", str(seed), str(up))
    _git("config", "uploadpack.allowFilter", "true", cwd=up)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=up)
    _git("clone", "-q", "--filter=blob:none", "--no-checkout", up.as_uri(), str(clone))
    (seed / "new.txt").write_text("upstream moved\n", encoding="utf-8")
    _git("add", "-A", cwd=seed)
    _git("commit", "-qm", "upstream", cwd=seed)
    _git("push", "-q", str(up), "main", cwd=seed)
    return clone, _git("rev-parse", "HEAD", cwd=clone), _git("rev-parse", "main", cwd=up)


def _fetched(clone: Path, sha: str) -> bool:
    env = {**_ENV, "GIT_NO_LAZY_FETCH": "1"}
    return subprocess.run(["git", "cat-file", "-e", sha], cwd=clone, env=env, capture_output=True).returncode == 0


def test_update_check_ancestry_probe_never_fetches_from_the_promisor(partial_clone):
    clone, head, upstream_tip = partial_clone

    # The exact ancestry probe check_for_updates runs before it falls back to the compare API.
    assert not source_check._git_ok(["merge-base", "--is-ancestor", upstream_tip, head], cwd=clone)
    assert not _fetched(clone, upstream_tip), "the update check downloaded upstream history"
    # Control: an upstream tip already in local history still reads as up to date.
    parent = _git("rev-parse", "HEAD~1", cwd=clone)
    assert source_check._git_ok(["merge-base", "--is-ancestor", parent, head], cwd=clone)


def test_bounded_git_probe_never_fetches_from_the_promisor(partial_clone):
    clone, head, upstream_tip = partial_clone

    assert bounded_git_probe(["git", "-C", str(clone), "log", "-1", "--format=%H", upstream_tip], timeout=10) == ""
    assert not _fetched(clone, upstream_tip), "a session-start git probe downloaded upstream history"
    assert bounded_git_probe(["git", "-C", str(clone), "log", "-1", "--format=%H", head], timeout=10) == head


@pytest.fixture
def treeless_clone(tmp_path: Path):
    """(seed, clone, catalog_commit) — the installer's ``--filter=tree:0`` clone: only HEAD's tree is local,
    and ``plugin-catalog/`` last changed two commits back."""
    seed, up, clone = tmp_path / "seed", tmp_path / "up.git", tmp_path / "clone"
    _git("init", "-q", "-b", "main", str(seed))
    (seed / "plugin-catalog").mkdir()
    (seed / "plugin-catalog" / "catalog.yaml").write_text("entries: []\n", encoding="utf-8")
    for i in range(3):
        (seed / "other.txt").write_text(f"v{i}\n", encoding="utf-8")
        _git("add", "-A", cwd=seed)
        date = f"2026-01-0{i + 1}T00:00:00Z"  # distinct commit times, so the control below discriminates
        _git("commit", "-qm", f"c{i}", cwd=seed, env={**_ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})
    _git("clone", "-q", "--bare", str(seed), str(up))
    _git("config", "uploadpack.allowFilter", "true", cwd=up)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=up)
    _git("clone", "-q", "--filter=tree:0", up.as_uri(), str(clone))
    return seed, clone, _git("rev-parse", "HEAD~2", cwd=seed)


def test_in_tree_catalog_time_never_fetches_from_the_promisor(treeless_clone, monkeypatch):
    seed, clone, catalog_commit = treeless_clone
    parent_tree = _git("rev-parse", "HEAD~1^{tree}", cwd=seed)
    assert not _fetched(clone, parent_tree)  # precondition: the walk's next tree is not local

    monkeypatch.setattr(plugin_catalog, "_in_tree_catalog_time", -1.0)
    monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: clone / "plugin-catalog")
    assert plugin_catalog.in_tree_catalog_time() is None
    assert not _fetched(clone, parent_tree), "dating the in-tree plugin catalog downloaded history"

    # Control: with the history local, the catalog is dated by the commit that last touched it.
    monkeypatch.setattr(plugin_catalog, "_in_tree_catalog_time", -1.0)
    monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: seed / "plugin-catalog")
    assert plugin_catalog.in_tree_catalog_time() == float(_git("log", "-1", "--format=%ct", catalog_commit, cwd=seed))
