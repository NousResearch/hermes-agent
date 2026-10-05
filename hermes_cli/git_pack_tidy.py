"""Progressive, time-boxed cleanup of a partial clone's promisor packfiles (#129712).

A partial clone downloads objects when git first needs them, and git asks for them without telling
the server what it already has, so the same trees (and, for history downloads, commits) land again
and again, each download as its own packfile that nothing removes: ``.git`` reached 39 GiB on a
~1 GiB repository. A full ``git gc`` over that is all-or-nothing and on a big checkout runs for tens
of minutes, so ``hermes update`` instead spends at most ``TIDY_BUDGET_SECONDS`` per run and every
unit of work it finishes is kept:

1. Erase promisor packs whose every object is also in another promisor pack. Nothing is lost, and
   nothing depends on the remote still serving it: the rule is proven against the local packs alone.
2. While more than ``PACK_COUNT_TARGET`` promisor packs remain, merge the smallest few into one
   with ``git pack-objects --stdin-packs``; the merged pack holds each object once.

Each erase and each merge is atomic for git (a pack exists for git only while both its ``.idx`` and
``.pack`` do), so an update that runs out of budget, fails or is killed leaves the rest for the next.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from hermes_cli._subprocess_compat import (
    NO_LAZY_FETCH_ENV,
    bounded_probe_run,
    noninteractive_git_env,
    windows_hide_flags,
)

logger = logging.getLogger(__name__)

TIDY_BUDGET_SECONDS = 60
PACK_COUNT_TARGET = 50
# A pack this young may still be in use by the git that just fetched it; index-pack also writes a
# new pack's .promisor/.keep before its .pack, so a young marker without a payload may be a live fetch.
_MIN_PACK_AGE_SECONDS = 60 * 60
_MERGE_BATCH_BYTES = 256 * 1024 * 1024
# Kept in .git/: packs proven to hold an object no other pack has (valid until a new pack appears,
# since removing packs never adds copies) and the merge batch size, halved after a merge that ran out
# of time. Remembering both is what guarantees progress: no update repeats work an earlier one finished.
_STATE_FILE = "hermes-pack-tidy.json"
# Two tidies at once could each erase a pack because the other still has its copies.
_LOCK_FILE = "hermes-pack-tidy.lock"
# Payload first: git loads a pack only when both .idx and .pack exist, so once .pack is gone the pack is
# gone for git, and a failure on .pack itself (a reader still maps it on Windows) changes nothing.
_PACK_SUFFIXES = (".pack", ".idx", ".rev", ".bitmap", ".mtimes", ".promisor", ".keep")
_IDX_V2 = b"\xfftOc\x00\x00\x00\x02"
_FANOUT = 8
_NAMES = _FANOUT + 256 * 4


@dataclass
class TidyResult:
    erased: int = 0
    freed_bytes: int = 0
    merged: int = 0
    packs_left: int = 0
    out_of_time: bool = False


class _Index:
    """A version-2 SHA-1 pack index (what a GitHub clone writes), searched in place."""

    def __init__(self, idx: Path) -> None:
        with open(idx, "rb") as fh:
            self._map = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        if self._map[:8] != _IDX_V2:
            self.close()
            raise ValueError(f"{idx.name}: not a v2 SHA-1 pack index")
        self.count = struct.unpack_from(">I", self._map, _FANOUT + 255 * 4)[0]

    def oids(self) -> Iterator[bytes]:
        for i in range(self.count):  # lazily: a big index must not run past the deadline before its first check
            yield self._map[_NAMES + i * 20: _NAMES + i * 20 + 20]

    def __contains__(self, oid: bytes) -> bool:
        lo = struct.unpack_from(">I", self._map, _FANOUT + (oid[0] - 1) * 4)[0] if oid[0] else 0
        hi = struct.unpack_from(">I", self._map, _FANOUT + oid[0] * 4)[0]
        while lo < hi:
            mid = (lo + hi) // 2
            name = self._map[_NAMES + mid * 20: _NAMES + mid * 20 + 20]
            if name == oid:
                return True
            if name < oid:
                lo = mid + 1
            else:
                hi = mid
        return False

    def close(self) -> None:
        self._map.close()  # Windows refuses to unlink a mapped file


def _git_env() -> Dict[str, str]:
    return {**noninteractive_git_env(), **NO_LAZY_FETCH_ENV}


def _promisor_packs(pack_dir: Path) -> List[Path]:
    return [p for p in pack_dir.glob("pack-*.pack")
            if p.with_suffix(".promisor").exists() and p.with_suffix(".idx").exists()]


def _pinned(pack: Path, cutoff: float) -> bool:
    """Whether a ``.keep`` pins the pack. fetch-pack and index-pack write ``<cmd> <pid> on <host>`` and
    delete it once the fetch has updated its refs, so an old one of theirs belongs to a fetch that was
    killed; it pins nothing, and git's own repack never touches the pack while it exists."""
    keep = pack.with_suffix(".keep")
    try:
        return not (keep.read_text(encoding="utf-8-sig", errors="replace").startswith(("fetch-pack ", "index-pack "))
                    and keep.stat().st_mtime < cutoff)
    except FileNotFoundError:
        return False
    except OSError:
        return True


def _drop_midx(pack_dir: Path) -> None:
    """A multi-pack-index names the packs it covers; dropping it before any pack goes means no crash
    can leave one naming a deleted pack. git rebuilds it on its own maintenance.

    The incremental layout (``multi-pack-index.d/``) is read through its chain file, so once the chain
    is gone git ignores the layers and removing them is only tidying."""
    (pack_dir / "multi-pack-index").unlink(missing_ok=True)
    layers = pack_dir / "multi-pack-index.d"
    (layers / "multi-pack-index-chain").unlink(missing_ok=True)
    shutil.rmtree(layers, ignore_errors=True)


def _remove_pack(pack: Path) -> int:
    """Delete one pack's files, payload first; returns the bytes freed.

    Raises on the first part that will not go. Leftovers after the payload are invisible to git and
    :func:`_sweep_remnants` retries them on a later run."""
    _drop_midx(pack.parent)
    freed = 0
    for suffix in _PACK_SUFFIXES:
        part = pack.with_suffix(suffix)
        try:
            size = part.stat().st_size
            if os.name == "nt":
                os.chmod(part, 0o666)  # git writes packs read-only; Windows refuses to unlink those
            part.unlink()
            freed += size
        except FileNotFoundError:
            continue
    return freed


def _sweep_remnants(pack_dir: Path) -> None:
    """Finish deletions an earlier run could not complete: pack files whose payload is already gone."""
    cutoff = time.time() - _MIN_PACK_AGE_SECONDS
    for part in pack_dir.glob("pack-*.*"):
        try:
            if part.suffix in _PACK_SUFFIXES[1:] and not part.with_suffix(".pack").exists() \
                    and part.stat().st_mtime < cutoff:
                _remove_pack(part)
        except OSError:
            logger.debug("pack remnant %s still in use", part.name, exc_info=True)


def _load_state(pack_dir: Path) -> dict:
    try:
        state = json.loads((pack_dir.parent.parent / _STATE_FILE).read_text(encoding="utf-8-sig"))
        kept, seen = set(state.get("kept", [])), set(state.get("seen", []))
        merge_bytes = int(state.get("merge_bytes", _MERGE_BATCH_BYTES))
    except (OSError, ValueError, TypeError, AttributeError):
        kept, seen, merge_bytes = set(), set(), _MERGE_BATCH_BYTES
    if {p.stem for p in pack_dir.glob("pack-*.pack")} - seen:
        kept = set()  # a new pack may hold the other copy of what a kept pack holds
    return {"kept": kept, "merge_bytes": merge_bytes}


def _save_state(pack_dir: Path, state: dict) -> None:
    live = sorted(p.stem for p in pack_dir.glob("pack-*.pack"))
    data = {"kept": sorted(state["kept"] & set(live)), "seen": live, "merge_bytes": state["merge_bytes"]}
    try:
        (pack_dir.parent.parent / _STATE_FILE).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.debug("could not record pack tidy state in %s", pack_dir, exc_info=True)


def _is_redundant(oids: Iterable[bytes], others: List[_Index], deadline: float) -> Optional[bool]:
    """Whether every oid is in one of ``others``; None when the deadline passed first."""
    hit = 0
    for n, oid in enumerate(oids):
        if n % 1024 == 0 and time.monotonic() > deadline:
            return None
        if oid in others[hit]:
            continue
        for j, other in enumerate(others):  # objects cluster: try the pack that had the last one first
            if j != hit and oid in other:
                hit = j
                break
        else:
            return False
    return True


def _erase_redundant_packs(pack_dir: Path, deadline: float, result: TidyResult, state: dict) -> None:
    cutoff = time.time() - _MIN_PACK_AGE_SECONDS
    sizes: Dict[Path, int] = {}
    candidates: List[Path] = []
    for pack in _promisor_packs(pack_dir):
        try:
            st = pack.stat()
        except OSError:
            continue
        sizes[pack] = st.st_size
        if pack.stem not in state["kept"] and st.st_mtime < cutoff and not _pinned(pack, cutoff):
            candidates.append(pack)
    if not candidates:
        return
    indexes: Dict[str, _Index] = {}
    try:
        for pack in sizes:
            try:
                indexes[pack.stem] = _Index(pack.with_suffix(".idx"))
            except (OSError, ValueError):  # unreadable now: neither a candidate nor a copy, retried next run
                logger.debug("pack index %s unreadable", pack.name, exc_info=True)
        by_size = sorted(sizes, key=lambda p: sizes[p], reverse=True)  # big packs hold most copies
        for pack in sorted(candidates, key=lambda p: sizes[p]):  # small first: cheapest to prove, likeliest copies
            if pack.stem not in indexes:
                continue
            others = [indexes[p.stem] for p in by_size if p.stem != pack.stem and p.stem in indexes]
            verdict = _is_redundant(indexes[pack.stem].oids(), others, deadline) if others else False
            if verdict is None:
                result.out_of_time = True
                return
            if not verdict:
                state["kept"].add(pack.stem)
                continue
            indexes.pop(pack.stem).close()
            try:
                result.freed_bytes += _remove_pack(pack)
                result.erased += 1
            except OSError as exc:
                logger.warning("Could not erase redundant pack %s (a later update retries): %s", pack.name, exc)
    finally:
        for index in indexes.values():
            index.close()


def _merge_smallest_packs(repo_root: Path, pack_dir: Path, deadline: float, result: TidyResult,
                          state: dict) -> None:
    """Merge the smallest promisor packs, a batch at a time, until the count is under target."""
    staging = pack_dir.parent.parent / "hermes-tidy-staging"  # outside objects/: git counts strays there as garbage
    cutoff = time.time() - _MIN_PACK_AGE_SECONDS
    while True:
        packs = sorted((p.stat().st_size, p) for p in _promisor_packs(pack_dir) if not _pinned(p, cutoff))
        if len(packs) <= PACK_COUNT_TARGET:
            return
        batch, total = [], 0
        for size, pack in packs:
            if len(batch) >= 2 and total + size > state["merge_bytes"]:
                break
            batch.append(pack)
            total += size
        left = deadline - time.monotonic()
        if left <= 0:
            result.out_of_time = True
            return
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            # packSizeLimit=0 whatever the user's config says: inputs retire only after everything
            # they held is published, and that is one output pack.
            done = bounded_probe_run(
                ["git", "-c", "pack.packSizeLimit=0", "pack-objects", "-q", "--stdin-packs", str(staging / "pack")],
                timeout=left, cwd=str(repo_root), env=_git_env(),
                input="".join(p.name + "\n" for p in batch))
            if done is None:
                result.out_of_time = True
                state["merge_bytes"] = max(state["merge_bytes"] // 2, 1)  # next update tries a smaller batch
                return
            outputs = list(staging.glob("pack-*.idx"))
            if done.returncode != 0 or len(outputs) != 1:
                logger.warning("Merging %d packs failed (%d outputs): %s", len(batch), len(outputs),
                               done.stderr.strip()[-300:])
                return
            new = outputs[0].with_suffix("")
            # Payload last: git ignores an index whose .pack is absent, so a kill before the final move
            # leaves only metadata, which _sweep_remnants removes (a lone .pack would be kept forever).
            (pack_dir / (new.name + ".promisor")).write_bytes(
                b"".join(p.with_suffix(".promisor").read_bytes() for p in batch))
            if new.with_suffix(".rev").exists():
                os.replace(new.with_suffix(".rev"), pack_dir / (new.name + ".rev"))
            os.replace(new.with_suffix(".idx"), pack_dir / (new.name + ".idx"))
            os.replace(new.with_suffix(".pack"), pack_dir / (new.name + ".pack"))
            for pack in batch:
                if pack.stem != new.name:
                    try:
                        _remove_pack(pack)
                    except OSError as exc:  # its objects are in the merged pack; a later run retries
                        logger.warning("Could not retire merged pack %s: %s", pack.name, exc)
            result.merged += len(batch)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _take_lock(git_dir: Path) -> Optional[Path]:
    """The tidy lock, or None while another tidy holds it. A lock older than an hour is a killed run's
    (a run lasts about ``TIDY_BUDGET_SECONDS``); renaming it away first means one taker wins."""
    lock = git_dir / _LOCK_FILE
    for _attempt in range(2):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return lock
        except FileExistsError:
            try:
                if lock.stat().st_mtime > time.time() - _MIN_PACK_AGE_SECONDS:
                    return None
                taken = git_dir / f"{_LOCK_FILE}.stale-{os.getpid()}"
                os.replace(lock, taken)
                if taken.stat().st_mtime > time.time() - _MIN_PACK_AGE_SECONDS:
                    # Another taker replaced the stale lock between our check and the rename: this is
                    # its live lock. Hand it back unless a third run has already locked.
                    try:
                        os.link(taken, lock)
                    except OSError:
                        pass
                    taken.unlink()
                    return None
                taken.unlink()
            except OSError:
                return None
    return None


def tidy_partial_clone_packs(repo_root: Path, *, budget_seconds: float = TIDY_BUDGET_SECONDS) -> TidyResult:
    """Spend at most ``budget_seconds`` erasing redundant and merging small promisor packs. Never raises."""
    from hermes_cli.gitlock import _partial_clone_filter

    result = TidyResult()
    pack_dir = Path(repo_root) / ".git" / "objects" / "pack"
    try:
        if not pack_dir.is_dir() or _partial_clone_filter(repo_root, creationflags=windows_hide_flags()) is None:
            return result
        deadline = time.monotonic() + budget_seconds
        lock = _take_lock(pack_dir.parent.parent)
        if lock is None:
            return result
        try:
            state = _load_state(pack_dir)
            try:
                _sweep_remnants(pack_dir)
                _erase_redundant_packs(pack_dir, deadline, result, state)
                if not result.out_of_time:
                    _merge_smallest_packs(repo_root, pack_dir, deadline, result, state)
            finally:
                _save_state(pack_dir, state)
        finally:
            lock.unlink(missing_ok=True)
        result.packs_left = len(list(pack_dir.glob("pack-*.pack")))
    except Exception:
        logger.warning("partial-clone pack tidy failed in %s", repo_root, exc_info=True)
    return result
