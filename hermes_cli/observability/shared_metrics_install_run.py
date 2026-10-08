"""hermes.install.run: fresh installs through scripts/install.sh and scripts/install.ps1.

The installer runs before the user is asked about shared metrics and must never send anything, so a
full-ladder run only leaves ONE small local receipt (closed tokens plus two epoch timestamps) under
the profile's store dir. A later Hermes start that runs the process-exit reporter reads it:

- collection off: ``shared_metrics_process.begin_process`` (and every opt-out answer) deletes every
  receipt unreported, the same rule as parked update receipts;
- collection on, sending off: the row is recorded now (it stays on this machine);
- collection on, sending on: the row is recorded only on a day whose package the sender's consent
  gate can ever pass, i.e. a send consent window opened at or before today 00:00Z. On the opt-in day
  itself (setup asked during the install) the receipt waits for the first start on a later day;
- a receipt older than ``MAX_RECEIPT_AGE_SECONDS``, unparseable, or with any value outside the
  contract is deleted unreported (an old receipt would count under whatever version records it).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PENDING_DIRNAME = "pending_installs"
# One empty file per receipt id already counted: a delete that failed after the row was saved must
# not count the same install again on the next start.
RECORDED_DIRNAME = "recorded_installs"
_RECORDED_KEEP = 64
_RECEIPT_ID = re.compile(r"[0-9a-f]{32}")
_RECEIPT_KEYS = frozenset({"id", "installer", "outcome", "failed_stage", "failure_class", "started_at", "finished_at"})
# A receipt no start recorded within a week is dropped: the row would carry the recording client's
# version, not the installer's, and per-version install failure rates would drift onto later releases.
MAX_RECEIPT_AGE_SECONDS = 7 * 86_400


def pending_installs_dir(home: Path) -> Path:
    return home / "telemetry" / "shared_metrics" / PENDING_DIRNAME


def purge_pending_installs(home: Path) -> None:
    shutil.rmtree(pending_installs_dir(home), ignore_errors=True)


def _epoch(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def install_run_fields(receipt: Any) -> dict[str, str] | None:
    """hermes.install.run dims for one receipt, or None when anything is outside the contract."""
    from . import shared_metrics_contract as contract

    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        return None
    if not isinstance(receipt["id"], str) or not _RECEIPT_ID.fullmatch(receipt["id"]):
        return None
    started, finished = _epoch(receipt["started_at"]), _epoch(receipt["finished_at"])
    if started is None or finished is None or finished < started:
        return None
    fields = {
        "duration_bucket": contract.update_duration_bucket((finished - started) * 1000),
        "failed_stage": receipt["failed_stage"],
        "failure_class": receipt["failure_class"],
        "installer": receipt["installer"],
        "outcome": receipt["outcome"],
    }
    if not contract.counter_dimensions_are_valid(contract.INSTALL_RUN_METRIC, fields):
        return None
    # A success names no stage and no class; a failure names both.
    succeeded = fields["outcome"] == "success"
    if succeeded != (fields["failed_stage"] == "none") or succeeded != (fields["failure_class"] == "none"):
        return None
    return fields


def _recorded_latch(home: Path, receipt_id: str) -> Path:
    return home / "telemetry" / "shared_metrics" / RECORDED_DIRNAME / receipt_id


def _mark_recorded(latch: Path) -> None:
    """Latch a receipt id AFTER its row is saved: a process that dies between the latch and the save
    (a short-lived start whose reporter thread is cut off at exit) must not drop the install."""
    try:
        latch.parent.mkdir(parents=True, exist_ok=True)
        latch.touch()
        for stale in sorted(latch.parent.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[_RECORDED_KEEP:]:
            stale.unlink(missing_ok=True)
    except OSError:  # a concurrent prune won; the next latch prunes again
        pass


def _today_start() -> str:
    """Today's package ``period_start``, in the store's stamp format (``shared_metrics._isoformat``)."""
    from .shared_metrics import _isoformat, _utc_now

    return _isoformat(datetime.combine(_utc_now().date(), datetime.min.time(), tzinfo=timezone.utc))


def consented_day(home: Path) -> bool:
    """Whether a row recorded now lands in a package the user agreed to: collection on, and either
    sending off (it stays local) or a send consent window opened at or before today 00:00Z. The
    sender's ``CONSENT_GATE_SQL`` passes a day's package only when ``period_start >= opened_at``, so a
    row recorded on the opt-in day itself would sit in a package that is never sent."""
    from hermes_cli.config import read_raw_config_readonly

    from .shared_metrics_send_config import resolve_send_config

    send = resolve_send_config(read_raw_config_readonly() or {})
    if not send.enabled:
        return False
    if not send.send:
        return True
    from .shared_metrics import SharedMetricsStore

    root = home / "telemetry" / "shared_metrics"
    try:
        with SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM send_consent_windows WHERE closed_at IS NULL AND opened_at <= ? LIMIT 1",
                (_today_start(),),
            ).fetchone()
    except sqlite3.Error:  # a busy store: try again on a later start
        return False
    return row is not None


def _claim_receipt(path: Path) -> Path | None:
    """Rename a receipt so exactly one reporter owns it. Unlike the process-marker claim it never
    reads the file first: an unparseable receipt is claimed too, so it can be deleted."""
    from .shared_metrics_process import _REPORTING, _claimer_alive

    if path.name.endswith(_REPORTING) and _claimer_alive(path):
        return None
    claimed = path.with_name(f"{path.name.split('.json')[0]}.json.{os.getpid()}{_REPORTING}")
    try:
        os.replace(path, claimed)
    except OSError:  # a concurrent reporter won
        return None
    return claimed


def _stale(receipt: dict) -> bool:
    return time.time() - receipt["finished_at"] > MAX_RECEIPT_AGE_SECONDS


def report_pending_installs(home: Path) -> None:
    """Record each receipt once, on a consented day (see the module docstring). Never raises."""
    try:
        from agent import relay_runtime

        from . import shared_metrics_contract as contract
        from .shared_metrics_events import emit_saved
        from .shared_metrics_process import settle_claim

        directory = pending_installs_dir(home)
        if not directory.is_dir():
            return
        # With Relay instrumentation off the store reports a row "settled" without recording it, which
        # would latch and delete the receipt uncounted: leave everything for a start that records.
        if not relay_runtime.relay_instrumentation_enabled():
            return
        reportable = consented_day(home)
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or ".json" not in path.name:
                continue
            claimed = _claim_receipt(path)  # a concurrent start that loses the rename records nothing
            if claimed is None:
                continue
            try:
                receipt = json.loads(claimed.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                receipt = None
            fields = install_run_fields(receipt)
            if fields is None or not isinstance(receipt, dict) or _stale(receipt):  # never to be counted
                settle_claim(claimed, path, True)
                continue
            latch = _recorded_latch(home, receipt["id"])
            if latch.exists():  # counted before; only the delete had failed
                settle_claim(claimed, path, True)
                continue
            if not reportable:  # kept for the first start on a fully consented day
                settle_claim(claimed, path, False)
                continue
            # The claim rename already keeps concurrent starts apart; a busy store keeps the receipt.
            saved = emit_saved([(contract.INSTALL_RUN_MARK, fields)]) == 1
            if saved:
                _mark_recorded(latch)
            settle_claim(claimed, path, saved)
    except Exception:
        logger.debug("Pending install receipts not reported", exc_info=True)
