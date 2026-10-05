"""Progressive partial-clone pack cleanup (#129712), on real git: a blobless clone of a local upstream."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

import hermes_cli.git_pack_tidy as tidy

_GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
_OFFLINE = {**_GIT_ENV, "GIT_NO_LAZY_FETCH": "1"}


def _git(*args: str, cwd: Path, env: dict = _GIT_ENV) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, env=env,
                          check=True, capture_output=True, text=True).stdout.strip()


def _packs(repo: Path) -> list[Path]:
    return sorted((repo / ".git" / "objects" / "pack").glob("pack-*.pack"))


@pytest.fixture
def clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A tree:0 clone with one on-demand pack per file it read, plus one holding a fetched-by-id commit."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    seed, up, repo = tmp_path / "seed", tmp_path / "up.git", tmp_path / "clone"
    _git("init", "-q", "-b", "main", str(seed), cwd=tmp_path)
    for i in range(6):
        (seed / f"f{i}.txt").write_text(f"v{i}\n", encoding="utf-8")
        _git("add", "-A", cwd=seed)
        _git("commit", "-qm", f"c{i}", cwd=seed)
    _git("switch", "-q", "-c", "side", cwd=seed)
    _git("commit", "-q", "--allow-empty", "-m", "side", cwd=seed)
    side = _git("rev-parse", "side", cwd=seed)
    _git("clone", "-q", "--bare", str(seed), str(up), cwd=tmp_path)
    for key in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        _git("config", key, "true", cwd=up)
    _git("clone", "-q", "--filter=tree:0", "--no-checkout", "--single-branch", "-b", "main", up.as_uri(), str(repo),
         cwd=tmp_path)
    _git("config", "maintenance.auto", "false", cwd=repo)
    for i in range(6):
        _git("cat-file", "-p", _git("rev-parse", f"HEAD:f{i}.txt", cwd=repo), cwd=repo)
    _git("cat-file", "-t", side, cwd=repo)  # an on-demand fetch that brings a commit
    _git("update-ref", "refs/heads/keep-side", side, cwd=repo)
    old = time.time() - 2 * 3600
    for entry in (repo / ".git" / "objects" / "pack").iterdir():
        os.utime(entry, (old, old))
    return repo, side


def test_erases_commit_free_on_demand_packs_and_keeps_what_refs_need(clone: tuple[Path, str]) -> None:
    repo, side = clone
    before = _packs(repo)

    result = tidy.tidy_partial_clone_packs(repo)

    after = _packs(repo)
    assert result.erased > 0 and len(after) == len(before) - result.erased and result.freed_bytes > 0
    assert _git("cat-file", "-t", side, cwd=repo, env=_OFFLINE) == "commit", "a ref points into that pack"
    assert _git("rev-list", "--count", "HEAD", cwd=repo, env=_OFFLINE) == "6"
    # Erased objects are copies of what the promisor serves: git fetches them again on demand.
    assert _git("cat-file", "-p", "HEAD:f3.txt", cwd=repo) == "v3"
    assert tidy.tidy_partial_clone_packs(repo).erased == 0


def test_merges_down_to_the_target_without_losing_objects(clone: tuple[Path, str],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    repo, side = clone
    monkeypatch.setattr(tidy, "_MIN_PACK_AGE_SECONDS", 10 ** 9)  # nothing is old enough to erase
    monkeypatch.setattr(tidy, "PACK_COUNT_TARGET", 1)
    assert len(_packs(repo)) > 2

    result = tidy.tidy_partial_clone_packs(repo)

    assert result.erased == 0 and result.packs_left == 1
    assert _packs(repo)[0].with_suffix(".promisor").exists()
    assert _git("cat-file", "-t", side, cwd=repo, env=_OFFLINE) == "commit"
    assert _git("cat-file", "-p", "HEAD:f3.txt", cwd=repo, env=_OFFLINE) == "v3"


def test_a_pack_that_will_not_unlink_stays_whole_or_is_finished_later(clone: tuple[Path, str],
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    repo, side = clone
    pack_dir = repo / ".git" / "objects" / "pack"
    _git("multi-pack-index", "write", cwd=repo)
    real_unlink = Path.unlink
    locked = {".pack"}  # a reader holds the payload (Windows refuses to unlink a mapped file)

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.parent == pack_dir and self.suffix in locked:
            raise PermissionError(13, "in use", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    before = _packs(repo)
    assert tidy.tidy_partial_clone_packs(repo).erased == 0
    assert _packs(repo) == before and all(p.with_suffix(".idx").exists() for p in before), "a failed unlink changed a pack"

    locked.clear()
    locked.add(".promisor")  # now the payload goes and a later part sticks
    tidy.tidy_partial_clone_packs(repo)
    assert not list(pack_dir.glob("multi-pack-index*")), "the multi-pack-index still names erased packs"
    assert _git("cat-file", "-t", side, cwd=repo, env=_OFFLINE) == "commit"
    assert _git("rev-list", "--count", "HEAD", cwd=repo, env=_OFFLINE) == "6"
    assert any(not p.with_suffix(".pack").exists() for p in pack_dir.glob("pack-*.promisor"))

    locked.clear()
    tidy.tidy_partial_clone_packs(repo)
    assert all(p.with_suffix(".pack").exists() for p in pack_dir.glob("pack-*.*") if p.suffix != ".pack"), \
        "leftovers of an interrupted erase survived the retry"
