"""Progressive partial-clone pack cleanup (#129712), on real git: partial clones of a local upstream."""

from __future__ import annotations

import os
import subprocess
import sys
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


def _objects(repo: Path) -> set[str]:
    """Every object the clone holds locally (what no cleanup may lose)."""
    return set(_git("cat-file", "--batch-all-objects", "--batch-check=%(objectname)", cwd=repo, env=_OFFLINE).split())


def _age(repo: Path) -> None:
    old = time.time() - 2 * 3600
    for entry in (repo / ".git" / "objects" / "pack").iterdir():
        os.utime(entry, (old, old))


def _upstream(tmp_path: Path, contents: list[bytes]) -> Path:
    seed, up = tmp_path / "seed", tmp_path / "up.git"
    _git("init", "-q", "-b", "main", str(seed), cwd=tmp_path)
    for i, data in enumerate(contents):
        (seed / f"f{i}").write_bytes(data)
        _git("add", "-A", cwd=seed)
        _git("commit", "-qm", f"c{i}", cwd=seed)
    _git("clone", "-q", "--bare", str(seed), str(up), cwd=tmp_path)
    for key in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        _git("config", key, "true", cwd=up)
    return up


@pytest.fixture
def clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A treeless install converted to blobless: its on-demand tree packs now duplicate the refetch
    pack, its on-demand blob packs hold the only copies, and a local branch needs both. The upstream
    is then rewritten and pruned, so nothing the clone lost could be fetched again."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    up = _upstream(tmp_path, [f"v{i}\n".encode() for i in range(6)])
    repo = tmp_path / "clone"
    _git("clone", "-q", "--filter=tree:0", "--no-checkout", "-b", "main", up.as_uri(), str(repo), cwd=tmp_path)
    _git("config", "maintenance.auto", "false", cwd=repo)
    _git("config", "gc.auto", "0", cwd=repo)
    for i in range(6):
        _git("cat-file", "-p", f"HEAD:f{i}", cwd=repo)  # one tree pack, then one blob pack, per file
    _git("update-ref", "refs/heads/local",
         _git("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "local", cwd=repo), cwd=repo)
    _git("fetch", "-q", "--refetch", "--filter=blob:none", "origin", "main", cwd=repo)
    _git("update-ref", "refs/heads/main", _git("commit-tree", "4b825dc642cb6eb9a060e54bf8d69288fbee4904",
                                              "-m", "rewrite", cwd=up), cwd=up)
    _git("gc", "-q", "--prune=now", cwd=up)
    _age(repo)
    return repo


def test_erases_only_packs_whose_every_object_has_another_copy(clone: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = _packs(clone)
    for pack in before:  # what killed fetches leave behind; git's own repack never frees these
        pack.with_suffix(".keep").write_text("fetch-pack 4242 on host\n", encoding="utf-8")
    head = _git("rev-parse", "HEAD", cwd=clone)
    listings = {p: _git("verify-pack", "-v", str(p.with_suffix(".idx")), cwd=clone, env=_OFFLINE) for p in before}
    clone_pack = min((p for p in before if head in listings[p]), key=lambda p: len(listings[p]))  # commits only
    clone_pack.with_suffix(".keep").write_text("pinned by hand\n", encoding="utf-8")
    _age(clone)
    held = _objects(clone)
    biggest = max(before, key=lambda p: p.stat().st_size).with_suffix(".idx")  # where most copies live
    real_init = tidy._Index.__init__

    def busy_biggest(self: object, idx: Path) -> None:
        if idx == biggest:
            raise OSError("busy")
        real_init(self, idx)

    with monkeypatch.context() as patched:  # without the index holding the copies, nothing is proven unique
        patched.setattr(tidy._Index, "__init__", busy_biggest)
        tidy.tidy_partial_clone_packs(clone)

    result = tidy.tidy_partial_clone_packs(clone)

    assert result.erased > 0 and len(_packs(clone)) == len(before) - result.erased and result.freed_bytes > 0
    assert clone_pack.exists(), "a pack pinned by hand was erased"
    assert _objects(clone) == held, "an erased pack held the only copy of an object"
    assert _git("cat-file", "-p", "local:f3", cwd=clone, env=_OFFLINE) == "v3"
    assert tidy.tidy_partial_clone_packs(clone).erased == 0


def test_merges_down_to_the_target_into_one_pack_whatever_the_size_limit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    up = _upstream(tmp_path, [os.urandom(400 * 1024) for _ in range(6)])
    repo = tmp_path / "clone"
    _git("clone", "-q", "--filter=blob:none", "--no-checkout", "-b", "main", up.as_uri(), str(repo), cwd=tmp_path)
    _git("config", "maintenance.auto", "false", cwd=repo)
    _git("config", "pack.packSizeLimit", "1m", cwd=repo)  # would split the merge into several packs
    for i in range(6):
        _git("cat-file", "-s", f"HEAD:f{i}", cwd=repo)  # one on-demand pack per blob
    _age(repo)
    held = _objects(repo)
    monkeypatch.setattr(tidy, "PACK_COUNT_TARGET", 1)
    # The merge writes the object store during `hermes update`: it is an updater git, so it runs in
    # the update's custody holding the checkout lock fd (a killed update's orphan keeps the lock).
    from hermes_cli import update_custody

    spawned, real_run = [], update_custody.run
    monkeypatch.setattr(update_custody, "run", lambda argv, *, inherit_lock=False, **kw: (
        spawned.append((update_custody.git_subcommand(list(argv)[1:]), inherit_lock))
        or real_run(argv, inherit_lock=inherit_lock, **kw)))

    result = tidy.tidy_partial_clone_packs(repo)

    assert result.packs_left == 1 and _packs(repo)[0].with_suffix(".promisor").exists()
    assert _objects(repo) == held
    assert ("pack-objects", True) in spawned


_KILLED_MID_ERASE = """
import os, sys
from pathlib import Path
import hermes_cli.git_pack_tidy as tidy
real = Path.unlink
def unlink(self, missing_ok=False):
    real(self, missing_ok=missing_ok)
    if self.suffix == ".pack":
        os._exit(9)  # die between a pack's payload and the rest of its files
Path.unlink = unlink
tidy.tidy_partial_clone_packs(Path(sys.argv[1]))
"""


@pytest.mark.parametrize("midx_layout", [[], ["--incremental"]])
def test_a_refused_or_killed_erase_leaves_git_reading_and_is_finished_later(
        clone: Path, monkeypatch: pytest.MonkeyPatch, midx_layout: list) -> None:
    pack_dir = clone / ".git" / "objects" / "pack"
    _git("multi-pack-index", "write", *midx_layout, cwd=clone)
    held, before = _objects(clone), _packs(clone)
    real_unlink = Path.unlink

    def refuse_payload(self: Path, missing_ok: bool = False) -> None:
        if self.parent == pack_dir and self.suffix == ".pack":
            raise PermissionError(13, "in use", str(self))  # a reader maps it (Windows)
        real_unlink(self, missing_ok=missing_ok)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", refuse_payload)
        assert tidy.tidy_partial_clone_packs(clone).erased == 0
    assert _packs(clone) == before and all(p.with_suffix(".idx").exists() for p in before)

    _git("multi-pack-index", "write", *midx_layout, cwd=clone)
    assert list(pack_dir.glob("multi-pack-index*"))
    child = subprocess.run([sys.executable, "-c", _KILLED_MID_ERASE, str(clone)], capture_output=True, text=True, encoding="utf-8",
                           env={**_GIT_ENV, "PYTHONPATH": str(Path(tidy.__file__).resolve().parents[1])})
    assert child.returncode == 9, child.stderr
    assert not list(pack_dir.glob("multi-pack-index*")), "a killed erase left a multi-pack-index naming a lost pack"
    assert _objects(clone) == held
    assert _git("cat-file", "-p", "local:f3", cwd=clone, env=_OFFLINE) == "v3"

    # The killed run's lock died with it. A live holder still excludes every other run.
    holder = tidy._take_lock(clone / ".git")
    assert holder is not None, "a killed tidy left its lock held"
    try:
        assert tidy.tidy_partial_clone_packs(clone) == tidy.TidyResult(), "ran beside a live lock holder"
    finally:
        os.close(holder)
    tidy.tidy_partial_clone_packs(clone)
    assert all(p.with_suffix(".pack").exists() for p in pack_dir.glob("pack-*.*")), \
        "leftovers of an interrupted erase survived the next run"
    assert _objects(clone) == held
