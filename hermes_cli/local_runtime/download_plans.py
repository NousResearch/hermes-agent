"""Durable model-download recipes; callbacks and running jobs stay process-local."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pm.downloader import Download, Source, _Remote
from pm.download_state import collect_download_plans, partial_lock
from pm.filesystem import durable_write_bytes


def _root() -> Path:
    from pm.paths import partials_root

    return partials_root() / ".model-plans"


def plan_key(plan) -> str:
    identity = [(url, str(dest)) for url, dest, _ in plan]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def plan_lock(key: str, *, wait: bool = False, cancelled=None, on_wait=None):
    return partial_lock(_root(), key, wait=wait, cancelled=cancelled, on_wait=on_wait)


def save_plan(job: dict, plan) -> str:
    key = plan_key(plan)
    record = {"target": job["target"], "model_id": job.get("model_id"),
              "total_bytes": job.get("total_bytes"),
              "files": [(url, str(dest), size) for url, dest, size in plan]}
    durable_write_bytes(_root() / f"{key}.json", json.dumps(record).encode())
    return key


def remove_plan(key: str) -> None:
    try:
        (_root() / f"{key}.json").unlink(missing_ok=True)
    except OSError as exc:
        # Recipe cleanup must not turn an atomically published model into a
        # failed download, or hide the original transfer error.
        logging.getLogger(__name__).warning("could not remove model download recipe: %s", exc)


def discard_plan(key: str) -> bool:
    with plan_lock(key) as acquired:
        if not acquired:
            return False
        (_root() / f"{key}.json").unlink(missing_ok=True)
        return True


def has_plan(key: str) -> bool:
    return (_root() / f"{key}.json").exists()


def pending_plans():
    """Yield abandoned recipes while holding their cross-process ownership."""
    from hermes_cli.local_runtime.bootstrap import models_dir, assets_dir

    collect_download_plans(_root().parent)
    parents = {models_dir().resolve(), assets_dir().resolve()}
    for path in sorted(_root().glob("*.json")):
        if len(path.stem) != 64 or any(char not in "0123456789abcdef" for char in path.stem):
            continue
        with plan_lock(path.stem) as acquired:
            if not acquired:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8-sig"))
                plan = [(url, Path(dest), size) for url, dest, size in record["files"]]
                if (not plan or not isinstance(record["target"], str)
                        or record.get("model_id") is not None and not isinstance(record["model_id"], str)
                        or any(not isinstance(url, str) or type(size) is not int or size < 0
                               or dest.parent.resolve() not in parents or dest.suffix.lower() != ".gguf"
                               for url, dest, size in plan)
                        or path.stem != plan_key(plan)):
                    continue
                yield path.stem, record, plan
            except (OSError, ValueError, TypeError, KeyError):
                continue  # A damaged recipe must not invent a destination or a job.


def retained_progress(plan) -> tuple[int, int, dict]:
    """Local durable coverage; the transfer re-probes identity before reusing it."""
    sources = [Source(url, dest) for url, dest, _ in plan]
    download = Download(sources)
    ranges = {}
    total = 0
    for source, (_, _, estimate) in zip(sources, plan):
        covered = []
        size = estimate
        if source.dest.is_file():
            size = source.dest.stat().st_size
            covered = [(0, size)]
        else:
            side = download.partials_dir / f"{download._key(source.url)}.ranges"
            try:
                record = json.loads(side.read_text(encoding="utf-8-sig"))
                if type(record["total"]) is int and record["total"] > 0:
                    size = record["total"]
                    covered = download._partial_ranges(source, _Remote(size, True, record["etag"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass  # No valid checkpoint is zero reusable coverage, not the preallocation.
        ranges[str(source.dest)] = covered
        total += size
    done = sum(end - start for rows in ranges.values() for start, end in rows)
    return done, total, ranges
