"""Shared-metrics facts for ``hermes update`` runs and Desktop self-updates.

``hermes.update.run`` / ``hermes.update.stage`` are DERIVED from the final update receipt in the
process that finalizes it (``update_receipt.finalize_update_receipt``): the receipt already carries
the outcome, the stage marks with timestamps, the admission refusal and the fleet matrix, so no
stage is instrumented for metrics. A pre-pull interpreter must never import pulled code, so when
it is the finalizer it parks the receipt under the store dir (stdlib-only code in update_receipt)
and :func:`report_pending_updates` records it on the next Hermes start.

Desktop's packaged updaters (electron-updater, App Installer, Store) never run ``hermes update``;
Desktop reports their outcome through the ``shared_metrics.update_run`` RPC instead. Desktop's
source-checkout hand-off DOES run ``hermes update``; that receipt is tagged ``initiator=desktop``
and counted here, never by the RPC, so no run is counted twice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PENDING_DIRNAME = "pending_updates"
# One empty file per update_id already counted in this profile: the completion child and the
# parent's parked copy can both finalize the same run.
RECORDED_DIRNAME = "recorded_updates"
_RECORDED_KEEP = 64
_UPDATE_ID = re.compile(r"[0-9a-f]{8,64}")
# Receipt pm/Desktop outcome words → hermes.update.run outcome; anything else failed.
_RUN_OUTCOMES = {"success": "success", "refused": "refused", "noop": "noop"}
_FLEET_BAD_STATES = frozenset({"stale", "down"})
# Desktop mechanism (updater strategy kind) → apply_mode.
_DESKTOP_APPLY_MODES = {
    "electron-updater": "package", "app-installer": "package", "microsoft-store": "package",
    "windows-handoff": "git", "posix-handoff": "git",
}


def _epoch(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return datetime.fromisoformat(str(value)).timestamp()
    except (OverflowError, TypeError, ValueError):
        return None


def _elapsed_ms(start: Any, end: Any) -> float | None:
    first, last = _epoch(start), _epoch(end)
    return None if first is None or last is None or last < first else (last - first) * 1000


def _receipt_stages(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage marks in order, each with a duration since the previous mark (or the run start)."""
    from .shared_metrics_contract import UPDATE_STAGES

    stages: list[dict[str, Any]] = []
    previous = receipt.get("started_at")
    for mark in receipt.get("stages") or ():
        if not isinstance(mark, dict) or mark.get("name") not in UPDATE_STAGES:
            continue
        stages.append({**mark, "duration_ms": _elapsed_ms(previous, mark.get("at"))})
        previous = mark.get("at") or previous
    fleet = receipt.get("fleet")
    # The fleet matrix is only passed to finalize by the post-restart verification.
    if isinstance(fleet, list) and fleet and not any(s["name"] == "verify" for s in stages):
        bad = any(isinstance(row, dict) and row.get("state") in _FLEET_BAD_STATES for row in fleet)
        stages.append({
            "name": "verify", "outcome": "failed" if bad else "success",
            "duration_ms": _elapsed_ms(previous, receipt.get("finished_at")),
        })
    return stages


def _failed_stage(stages: list[dict[str, Any]]) -> str:
    """Where a failed/refused run stopped: the failing stage, else the stage it never reached."""
    from .shared_metrics_contract import UPDATE_STAGE_ORDER

    if not stages:
        return "other"
    last = stages[-1]
    terminal = last["name"] == "verify" or (last["name"] == "restart" and last.get("outcome") == "skipped")
    if last.get("outcome") == "failed" or terminal:
        failed = [s["name"] for s in stages if s.get("outcome") == "failed"]
        return failed[-1] if failed else "other"
    index = UPDATE_STAGE_ORDER.index(last["name"])
    return UPDATE_STAGE_ORDER[index + 1] if index + 1 < len(UPDATE_STAGE_ORDER) else "other"


def _apply_mode(receipt: dict[str, Any], stages: list[dict[str, Any]]) -> str:
    for stage in stages:
        if stage["name"] == "apply" and stage.get("mode") in {"git", "zip"}:
            return stage["mode"]
    # An admission refusal means the install is owned by something else (Docker, Nix, a package).
    if any(isinstance(s, dict) and s.get("name") == "admission" and not s.get("ok") for s in receipt.get("steps") or ()):
        return "external"
    return "unknown"


def update_receipt_fields(receipt: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, str]]] | None:
    """Bounded hermes.update.run + hermes.update.stage dimensions for one FINAL receipt."""
    from .shared_metrics_contract import update_duration_bucket, version_age_bucket

    if not isinstance(receipt, dict) or not receipt.get("finished_at"):
        return None
    stages = _receipt_stages(receipt)
    outcome = _RUN_OUTCOMES.get(str(receipt.get("outcome") or ""), "failed")
    if outcome == "success" and any(s["name"] == "apply" and s.get("outcome") == "skipped" for s in stages):
        outcome = "noop"
    started = _epoch(receipt.get("started_at"))
    committed = _epoch((receipt.get("pre_update") or {}).get("commit_date"))
    age_ms = None if started is None or committed is None else (started - committed) * 1000
    run = {
        "apply_mode": _apply_mode(receipt, stages),
        "duration_bucket": update_duration_bucket(_elapsed_ms(receipt.get("started_at"), receipt.get("finished_at"))),
        "failed_stage": _failed_stage(stages) if outcome in {"failed", "refused"} else "none",
        "from_version_age_bucket": version_age_bucket(age_ms),
        "kind": "desktop" if receipt.get("initiator") == "desktop" else "cli",
        "outcome": outcome,
    }
    if run["outcome"] == "refused" and not stages:
        run["failed_stage"] = "none"
    stage_rows = [
        {
            "duration_bucket": update_duration_bucket(stage["duration_ms"]),
            "outcome": stage.get("outcome") if stage.get("outcome") in {"success", "failed", "skipped"} else "failed",
            "stage": stage["name"],
        }
        for stage in stages
    ]
    return run, stage_rows


def _collection_on() -> bool:
    """Cheap pre-gate so a disabled install never loads the Relay runtime; `_emit` re-checks."""
    from hermes_cli.config import read_raw_config_readonly

    config: Any = read_raw_config_readonly() or {}
    for key in ("telemetry", "shared_metrics"):
        config = config.get(key) if isinstance(config, dict) else None
    return isinstance(config, dict) and config.get("enabled") is True


def _claim_update_id(update_id: Any) -> tuple[bool, Path | None]:
    """``(first, latch)``: first is False when this profile already counted ``update_id``."""
    from hermes_constants import get_hermes_home

    if not isinstance(update_id, str) or not _UPDATE_ID.fullmatch(update_id):
        return True, None  # nothing stable to dedupe on
    directory = get_hermes_home() / "telemetry" / "shared_metrics" / RECORDED_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    latch = directory / update_id
    try:
        os.close(os.open(latch, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False, None
    try:
        for stale in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[_RECORDED_KEEP:]:
            stale.unlink(missing_ok=True)
    except OSError:  # a concurrent prune won; the next claim prunes again
        pass
    return True, latch


def record_update_receipt(receipt: dict[str, Any], *, wait_saved: bool = False) -> bool:
    """Record one run row plus its stage rows from a final receipt, once per update_id.

    ``wait_saved`` (parked receipts) blocks until the rows are in the store; False means the caller
    must keep the receipt for a later retry. True also covers "nothing to record". Never raises.
    """
    try:
        if not _collection_on():
            return True
        from . import shared_metrics_contract as contract
        from .shared_metrics_events import _emit, emit_saved

        derived = update_receipt_fields(receipt)
        if derived is None:
            return True
        first, latch = _claim_update_id(receipt.get("update_id"))
        if not first:
            return True
        run, stage_rows = derived
        rows = [(contract.UPDATE_RUN_MARK, run), *((contract.UPDATE_STAGE_MARK, row) for row in stage_rows)]
        if not wait_saved:
            for mark, row in rows:
                _emit(mark, lambda row=row: row)
            return True
        if emit_saved(rows):
            return True  # even partly: a retry would count again the rows that did land
        if latch is not None:
            latch.unlink(missing_ok=True)  # nothing landed: let the retry count it
        return False
    except Exception:
        logger.debug("Update shared metrics not recorded", exc_info=True)
        return False


def pending_updates_dir(home: Path) -> Path:
    return home / "telemetry" / "shared_metrics" / PENDING_DIRNAME


def purge_pending_updates(home: Path) -> None:
    shutil.rmtree(pending_updates_dir(home), ignore_errors=True)


def report_pending_updates() -> None:
    """Record receipts a pre-pull interpreter parked (it must not import pulled code). Never raises."""
    try:
        from hermes_constants import get_hermes_home

        from .shared_metrics_process import _claim, settle_claim

        directory = pending_updates_dir(get_hermes_home())
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or ".json" not in path.name:
                continue
            claimed = _claim(path)  # a concurrent start that loses the rename records nothing
            if claimed is None:
                continue
            try:
                receipt = json.loads(claimed.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                receipt = None
            saved = not isinstance(receipt, dict) or record_update_receipt(receipt, wait_saved=True)
            settle_claim(claimed, path, saved)
    except Exception:
        logger.debug("Pending update shared metrics not reported", exc_info=True)


def desktop_update_fields(
    *, outcome: Any, failed_stage: Any, duration_ms: Any, mechanism: Any, from_commit_date: Any = None,
) -> dict[str, str]:
    """hermes.update.run dims for a Desktop packaged self-update (RPC-reported)."""
    from .shared_metrics_contract import (
        DESKTOP_UPDATE_STAGES, UPDATE_OUTCOMES, update_duration_bucket, version_age_bucket,
    )

    outcome_value = str(outcome or "").strip().lower()
    outcome_value = outcome_value if outcome_value in UPDATE_OUTCOMES else "failed"
    stage = str(failed_stage or "").strip().lower()
    commit = _epoch(from_commit_date)
    # Desktop sends seconds since epoch (install stamp commitDate); age is measured now.
    age_ms = None if commit is None else (time.time() - commit) * 1000
    return {
        "apply_mode": _DESKTOP_APPLY_MODES.get(str(mechanism or "").strip().lower(), "unknown"),
        "duration_bucket": update_duration_bucket(duration_ms),
        "failed_stage": (stage if stage in DESKTOP_UPDATE_STAGES else "other") if outcome_value == "failed" else "none",
        "from_version_age_bucket": version_age_bucket(age_ms),
        "kind": "desktop",
        "outcome": outcome_value,
    }


def record_desktop_update(**raw: Any) -> None:
    """Emit one Desktop self-update run through the enabled() gate. Never raises."""
    from . import shared_metrics_contract as contract
    from .shared_metrics_events import _emit

    _emit(contract.UPDATE_RUN_MARK, desktop_update_fields, **raw)
