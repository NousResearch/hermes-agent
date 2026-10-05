"""Progressive, time-boxed cleanup of a partial clone's on-demand packfiles (#129712).

A partial clone (the installer's ``--filter=tree:0`` checkout) downloads trees and blobs when git
first needs them, and every such download lands as its own small packfile that nothing removes:
``.git`` reached 39 GiB on a ~1 GiB repository. A full ``git gc`` over that is all-or-nothing and
on a big checkout runs for tens of minutes, so ``hermes update`` instead spends at most
``TIDY_BUDGET_SECONDS`` per run and every unit of work it finishes is kept:

1. Erase on-demand packs that hold no commits. Their objects are copies of what the promisor
   remote serves, so deleting them frees the disk and git fetches anything it needs again. A pack
   holding commits is kept, because a ref may point into it.
2. While more than ``PACK_COUNT_TARGET`` promisor packs remain, merge the smallest few into one
   with ``git pack-objects --stdin-packs`` (cost proportional to those packs, no reachability walk).

Each erased pack and each merge is atomic, so an update that runs out of budget leaves the rest for
the next one.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from hermes_cli._subprocess_compat import (
    NO_LAZY_FETCH_ENV,
    bounded_probe_run,
    noninteractive_git_env,
    windows_hide_flags,
)

logger = logging.getLogger(__name__)

TIDY_BUDGET_SECONDS = 60
PACK_COUNT_TARGET = 50
# A pack this young may still be in use by the git that just fetched it.
_MIN_PACK_AGE_SECONDS = 60 * 60
_MERGE_BATCH_BYTES = 256 * 1024 * 1024
# Kept in .git/: per-pack "holds commits" verdicts (pack names are content hashes, so a verdict never
# goes stale) and the merge batch size, halved after a merge that ran out of time. Remembering both
# is what guarantees progress: no update repeats work an earlier one could not finish.
_STATE_FILE = "hermes-pack-tidy.json"
# Payload first: git loads a pack only when both .idx and .pack exist, so once .pack is gone the pack is
# gone for git, and a failure on .pack itself (a reader still maps it on Windows) changes nothing.
_PACK_SUFFIXES = (".pack", ".idx", ".rev", ".bitmap", ".mtimes", ".promisor", ".keep")


@dataclass
class TidyResult:
    erased: int = 0
    freed_bytes: int = 0
    merged: int = 0
    packs_left: int = 0
    out_of_time: bool = False


def _git_env() -> Dict[str, str]:
    return {**noninteractive_git_env(), **NO_LAZY_FETCH_ENV}


def _oids_in_pack_order(idx: Path) -> List[str]:
    """Object ids of a version-2 SHA-1 pack index (every pack a clone of GitHub writes), in the order
    they sit in the pack. git writes commits first, so a pack that holds any shows one up front."""
    data = idx.read_bytes()
    if data[:8] != b"\xfftOc\x00\x00\x00\x02":
        raise ValueError(f"{idx.name}: not a v2 pack index")
    count = struct.unpack(">I", data[8 + 255 * 4: 8 + 256 * 4])[0]
    names = 8 + 256 * 4
    offsets = struct.unpack(f">{count}I", data[names + count * 24: names + count * 28])  # after names + CRCs
    order = sorted(range(count), key=lambda i: offsets[i])  # MSB-set (past 2 GiB) entries sort last
    return [data[names + i * 20: names + (i + 1) * 20].hex() for i in order]


def _holds_commits(repo_root: Path, pack: Path, timeout: float) -> bool:
    """Whether the pack holds a commit. A pack git cannot classify in time counts as holding one:
    keeping a pack only costs disk, erasing one a ref points into costs history."""
    oids = _oids_in_pack_order(pack.with_suffix(".idx"))
    deadline = time.monotonic() + timeout
    for chunk in (oids[:256], oids[256:]):
        left = deadline - time.monotonic()
        if not chunk:
            continue
        result = bounded_probe_run(
            ["git", "cat-file", "--batch-check=%(objecttype)"], timeout=max(left, 0.1), cwd=str(repo_root),
            env=_git_env(), input="\n".join(chunk) + "\n")
        if left <= 0 or result is None or result.returncode != 0:
            return True
        if "commit" in result.stdout.split():
            return True
    return False


def _remove_pack(pack: Path) -> int:
    """Delete one pack's files, payload first; returns the bytes freed.

    Raises on the first part that will not go. Leftovers after the payload are invisible to git and
    :func:`_sweep_remnants` retries them on a later run."""
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
    cutoff = time.time() - _MIN_PACK_AGE_SECONDS  # index-pack writes .promisor/.keep before its .pack
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
        return {"commits": dict(state.get("commits", {})),
                "merge_bytes": int(state.get("merge_bytes", _MERGE_BATCH_BYTES))}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"commits": {}, "merge_bytes": _MERGE_BATCH_BYTES}


def _save_state(pack_dir: Path, state: dict) -> None:
    live = {p.stem for p in pack_dir.glob("pack-*.pack")}
    state = {**state, "commits": {k: v for k, v in state["commits"].items() if k in live}}
    try:
        (pack_dir.parent.parent / _STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        logger.debug("could not record pack tidy state in %s", pack_dir, exc_info=True)


def _erase_on_demand_packs(repo_root: Path, pack_dir: Path, deadline: float, result: TidyResult,
                           verdicts: Dict[str, bool]) -> None:
    cutoff = time.time() - _MIN_PACK_AGE_SECONDS
    candidates = []
    for marker in pack_dir.glob("pack-*.promisor"):
        pack = marker.with_suffix(".pack")
        try:
            # An on-demand fetch writes an empty marker; a ref fetch lists the refs it fetched.
            if marker.stat().st_size == 0 and pack.stat().st_mtime < cutoff and not pack.with_suffix(".keep").exists():
                candidates.append((pack.stat().st_mtime, pack))
        except OSError:
            continue
    for _mtime, pack in sorted(candidates):
        left = deadline - time.monotonic()
        if left <= 0:
            result.out_of_time = True
            return
        if pack.stem not in verdicts:
            try:
                verdicts[pack.stem] = _holds_commits(repo_root, pack, left)
            except (OSError, ValueError):
                verdicts[pack.stem] = True
        if verdicts[pack.stem]:
            continue
        try:
            result.freed_bytes += _remove_pack(pack)
            result.erased += 1
        except OSError as exc:
            logger.warning("Could not erase on-demand pack %s (a later update retries): %s", pack.name, exc)


def _merge_smallest_packs(repo_root: Path, pack_dir: Path, deadline: float, result: TidyResult,
                          state: dict) -> None:
    """Merge the smallest promisor packs, a batch at a time, until the count is under target."""
    staging = pack_dir.parent.parent / "hermes-tidy-staging"  # outside objects/: git counts strays there as garbage
    while True:
        packs = sorted((p.stat().st_size, p) for p in pack_dir.glob("pack-*.pack")
                       if p.with_suffix(".promisor").exists() and not p.with_suffix(".keep").exists())
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
            done = bounded_probe_run(
                ["git", "pack-objects", "-q", "--stdin-packs", str(staging / "pack")],
                timeout=left, cwd=str(repo_root), env=_git_env(),
                input="".join(p.name + "\n" for p in batch))
            if done is None:
                result.out_of_time = True
                state["merge_bytes"] = max(state["merge_bytes"] // 2, 1)  # next update tries a smaller batch
                return
            if done.returncode != 0:
                logger.warning("Merging %d packs failed: %s", len(batch), done.stderr.strip()[-300:])
                return
            new = next(staging.glob("pack-*.idx")).with_suffix("")
            # Marker first, index last: the merged pack is a promisor pack from the moment git sees it.
            (pack_dir / (new.name + ".promisor")).write_bytes(
                b"".join(p.with_suffix(".promisor").read_bytes() for p in batch))
            for suffix in (".pack", ".rev", ".idx"):
                if new.with_suffix(suffix).exists():
                    os.replace(new.with_suffix(suffix), pack_dir / (new.name + suffix))
            for pack in batch:
                if pack.stem != new.name:
                    try:
                        _remove_pack(pack)
                    except OSError as exc:  # its objects are in the merged pack; a later run retries
                        logger.warning("Could not retire merged pack %s: %s", pack.name, exc)
            result.merged += len(batch)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def tidy_partial_clone_packs(repo_root: Path, *, budget_seconds: float = TIDY_BUDGET_SECONDS) -> TidyResult:
    """Spend at most ``budget_seconds`` erasing and merging on-demand packs. Never raises."""
    from hermes_cli.gitlock import _partial_clone_filter

    result = TidyResult()
    pack_dir = Path(repo_root) / ".git" / "objects" / "pack"
    try:
        if not pack_dir.is_dir() or _partial_clone_filter(repo_root, creationflags=windows_hide_flags()) is None:
            return result
        deadline = time.monotonic() + budget_seconds
        state = _load_state(pack_dir)
        before = {p.name for p in pack_dir.glob("pack-*.pack")}
        try:
            _sweep_remnants(pack_dir)
            _erase_on_demand_packs(repo_root, pack_dir, deadline, result, state["commits"])
            if not result.out_of_time:
                _merge_smallest_packs(repo_root, pack_dir, deadline, result, state)
        finally:
            _save_state(pack_dir, state)
            if before - {p.name for p in pack_dir.glob("pack-*.pack")}:
                # A multi-pack-index names the packs it covers; git rebuilds one on its own maintenance.
                for midx in pack_dir.glob("multi-pack-index*"):
                    midx.unlink(missing_ok=True)
        result.packs_left = len(list(pack_dir.glob("pack-*.pack")))
    except Exception:
        logger.warning("partial-clone pack tidy failed in %s", repo_root, exc_info=True)
    return result
