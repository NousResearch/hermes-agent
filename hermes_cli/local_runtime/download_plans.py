"""Durable model-download plans; callbacks and running jobs stay process-local."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pm.downloader import Download, Source
from pm.download_state import collect_download_plans, discard_partial, is_state_key, partial_lock
from pm.filesystem import durable_write_bytes


def _root() -> Path:
    from pm.paths import partials_root

    return partials_root() / ".model-plans"


def plan_key(plan, setup=None) -> str:
    identity = (("setup", setup["profile"], setup["model_id"], setup["backend"] if setup["model_id"] is None else None)
                if setup else [(url, str(dest)) for url, dest, _ in plan])
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def plan_lock(key: str, *, wait: bool = False, cancelled=None, on_wait=None):
    return partial_lock(_root(), key, wait=wait, cancelled=cancelled, on_wait=on_wait)


def save_plan(job: dict, plan) -> str:
    key = plan_key(plan, job.get("setup"))
    record = {"target": job["target"], "model_id": job.get("model_id"),
              "total_bytes": job.get("total_bytes"),
              "files": [(url, str(dest), size) for url, dest, size in plan]}
    if job.get("setup"):
        record["setup"] = job["setup"]
    record["partial_urls"] = list(job.get("partial_urls", ()))
    durable_write_bytes(_root() / f"{key}.json", json.dumps(record).encode())
    return key


def checkpoint(job: dict) -> None:
    """Save a parked job's display state without changing its operation intent."""
    path = _root() / f"{job['plan_key']}.json"
    try:
        record = json.loads(path.read_bytes())
        record["partial_urls"] = list(job.get("partial_urls", ()))
        record["state"] = {key: job[key] for key in ("done_bytes", "total_bytes", "ranges", "restart_runtime") if key in job}
        durable_write_bytes(path, json.dumps(record).encode())
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("could not checkpoint parked download: %s", exc)


def remove_plan(key: str) -> None:
    try:
        (_root() / f"{key}.json").unlink(missing_ok=True)
    except OSError as exc:
        # Plan cleanup must not turn an atomically published model into a
        # failed download, or hide the original transfer error.
        logging.getLogger(__name__).warning("could not remove model download plan: %s", exc)


def discard_plan(key: str) -> bool:
    with plan_lock(key) as acquired:
        if not acquired:
            return False
        path = _root() / f"{key}.json"
        if path.is_file():
            record = json.loads(path.read_bytes())
            for url in [row[0] for row in record["files"]] + record.get("partial_urls", []):
                discard_partial(_root().parent, url, excluding_plan=key)
            path.unlink(missing_ok=True)
        return True


def has_plan(key: str) -> bool:
    return (_root() / f"{key}.json").exists()


def _valid_setup(setup) -> bool:
    return (isinstance(setup, dict) and set(setup) == {"model_id", "profile", "backend", "variant"}
            and isinstance(setup["profile"], str) and isinstance(setup["backend"], str)
            and all(value is None or isinstance(value, str) for value in (setup["model_id"], setup["variant"])))


def pending_plans():
    """Yield abandoned plans while holding their cross-process ownership."""
    from hermes_cli.local_runtime.bootstrap import models_dir, assets_dir

    collect_download_plans(_root().parent)
    parents = {models_dir().resolve(), assets_dir().resolve()}
    for path in sorted(_root().glob("*.json")):
        if not is_state_key(path.stem):
            continue
        with plan_lock(path.stem) as acquired:
            if not acquired:
                continue
            try:
                record = json.loads(path.read_bytes())
                plan = [(url, Path(dest), size) for url, dest, size in record["files"]]
                setup = record.get("setup")
                if (not plan and not setup or not isinstance(record["target"], str)
                        or record.get("model_id") is not None and not isinstance(record["model_id"], str)
                        or any(not isinstance(url, str) or type(size) is not int or size < 0
                               or dest.parent.resolve() not in parents or dest.suffix.lower() != ".gguf"
                               for url, dest, size in plan)
                        or setup is not None and not _valid_setup(setup)
                        or not isinstance(record.get("state", {}), dict)
                        or not isinstance(record.get("partial_urls", []), list)
                        or any(not isinstance(url, str) for url in record.get("partial_urls", []))
                        or path.stem != plan_key(plan, setup)):
                    continue
                yield path.stem, record, plan
            except (OSError, ValueError, TypeError, KeyError):
                continue  # A damaged plan must not invent a destination or a job.


def retained_progress(plan) -> tuple[int, int, dict]:
    """Local durable coverage; the transfer re-probes identity before reusing it."""
    sources = [Source(url, dest) for url, dest, _ in plan]
    return Download(sources).retained_progress([size for _, _, size in plan])
