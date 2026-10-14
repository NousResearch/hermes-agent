"""hermes.process.exit: how Hermes processes (and watchdog-killed turns) end.

A crash or SIGKILL cannot record itself. Each long-lived entrypoint drops a small marker under its
profile's store dir at start and restamps it on the way out (``clean``, ``crash`` + class from the
excepthook, ``watchdog`` from a hard-exit watchdog). The NEXT Hermes start in that profile reports
every marker whose owner is gone; one still ``running`` whose pid fails the canonical start-time
liveness check was killed. Reporting runs on a daemon thread so startup never waits on the Relay
runtime; a claimed-but-unreported marker (the reporter died mid-way) is reclaimed later.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MARKER_DIRNAME = "process_markers"
_REPORTING = ".reporting"
# Most specific first: ModuleNotFoundError is an ImportError, RecursionError a RuntimeError.
_CRASH_CLASSES = (
    (MemoryError, "memory_error"), (ImportError, "import_error"), (OSError, "os_error"),
    (RuntimeError, "runtime_error"),
)
_STATE: dict[str, Any] = {}
_STATE_LOCK = threading.Lock()


def crash_class(exc: BaseException | None) -> str:
    return next((label for kind, label in _CRASH_CLASSES if isinstance(exc, kind)), "other")


def markers_dir(home: Path) -> Path:
    return home / "telemetry" / "shared_metrics" / MARKER_DIRNAME


def _write_marker(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, path)


def stamp_exit(exit_kind: str, crash: str = "none") -> None:
    """Restamp this process's marker with how it is ending. Never raises; no-op when unmarked."""
    try:
        with _STATE_LOCK:
            record = _STATE.get("record")
            path = _STATE.get("path")
            if record is None or path is None or record.get("state") != "running":
                return
            record = {**record, "state": exit_kind, "crash_class": crash}
            _STATE["record"] = record
            _write_marker(path, record)
    except Exception:
        logger.debug("Process exit marker not stamped", exc_info=True)


def _excepthook(previous: Any) -> Any:
    def hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            stamp_exit("crash", crash_class(exc))
        previous(exc_type, exc, tb)

    return hook


def purge_pending_receipts(home: Path) -> None:
    """Collection is off: drop the files kept for a later start to count (parked update receipts,
    installer receipts). Every opt-out answer calls this too, not only an opted-out start, so a
    receipt never survives a "No" to be counted after a later opt-in. Never raises."""
    try:
        from .shared_metrics_install_run import purge_pending_installs
        from .shared_metrics_update import purge_pending_updates

        purge_pending_updates(home)
        purge_pending_installs(home)
    except Exception:
        logger.debug("Pending shared-metrics receipts not purged", exc_info=True)


def begin_process(kind: str) -> None:
    """Mark this process as a running ``kind`` and report dead predecessors. Once per process."""
    try:
        from hermes_constants import get_hermes_home

        from .shared_metrics_desktop import ONBOARDING_LATCH_DIRNAME
        from .shared_metrics_setup import markers_dir as setup_markers_dir
        from .shared_metrics_update import _collection_on

        if _STATE:
            return
        _STATE["kind"] = kind  # watchdog turn rows name the surface even when this home is off
        if not _collection_on():
            home = get_hermes_home()
            purge_pending_receipts(home)  # parked while on / left by the installer, never counted once off
            latches = home / "telemetry" / "shared_metrics" / ONBOARDING_LATCH_DIRNAME  # an opt-out outside Desktop
            for directory in (markers_dir(home), setup_markers_dir(home), latches):  # likewise pending exits/setups
                shutil.rmtree(directory, ignore_errors=True)
            return
        from gateway.status import get_process_start_time

        home = get_hermes_home()
        directory = markers_dir(home)
        directory.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        path = directory / f"{kind}-{pid}.json"
        record = {"kind": kind, "pid": pid, "start_time": get_process_start_time(pid), "state": "running"}
        with _STATE_LOCK:
            _write_marker(path, record)
            _STATE.update(path=path, record=record)
        sys.excepthook = _excepthook(sys.excepthook)
        atexit.register(stamp_exit, "clean")
        threading.Thread(
            target=_report_dead_markers, args=(home, path), name="hermes-exit-metrics", daemon=True,
        ).start()
    except Exception:
        logger.debug("Process exit marker not started", exc_info=True)


def current_process_kind() -> str:
    return _STATE.get("kind") or "other"


def _claimer_alive(path: Path) -> bool:
    from gateway.status import runtime_status_pid_is_live

    try:
        claimer = int(path.name.rsplit(".", 2)[-2])
    except (IndexError, ValueError):
        return False
    return runtime_status_pid_is_live({"pid": claimer})


def _claim(path: Path) -> Path | None:
    """Rename a reportable marker so exactly one reporter owns it; None when not reportable."""
    from gateway.status import runtime_status_pid_is_live

    if path.name.endswith(_REPORTING):
        if _claimer_alive(path):
            return None
        record_path = path
    else:
        record_path = None
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record_path is None and record.get("state") == "running" and runtime_status_pid_is_live(record):
        return None
    claimed = path.with_name(f"{path.name.split('.json')[0]}.json.{os.getpid()}{_REPORTING}")
    try:
        os.replace(path, claimed)
    except OSError:  # a concurrent reporter won
        return None
    return claimed


def process_exit_fields(record: dict[str, Any]) -> dict[str, str]:
    from .shared_metrics_contract import CRASH_CLASSES, PROCESS_EXIT_KINDS, PROCESS_KINDS

    state = record.get("state")
    exit_kind = "killed" if state == "running" or state not in PROCESS_EXIT_KINDS else state
    crash = record.get("crash_class") if exit_kind == "crash" else "none"
    kind = record.get("kind")
    return {
        "crash_class": crash if crash in CRASH_CLASSES else "other",
        "exit_kind": exit_kind,
        "process_kind": kind if kind in PROCESS_KINDS else "other",
    }


def settle_claim(claimed: Path, original: Path, saved: bool) -> None:
    """Delete a claimed file once its row is saved; otherwise hand it back for the next start."""
    try:
        if saved:
            claimed.unlink(missing_ok=True)
        else:
            os.replace(claimed, original)
    except OSError:  # a leftover claim is reclaimed once this reporter is gone
        logger.debug("Claimed shared-metrics file not settled", exc_info=True)


def _report_dead_markers(home: Path, own: Path) -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from . import shared_metrics_contract as contract
    from .shared_metrics_events import emit_saved

    token = set_hermes_home_override(home)  # a thread does not inherit the profile binding
    try:
        for path in sorted(markers_dir(home).iterdir()):
            if path == own or path.name.startswith(".") or ".json" not in path.name:
                continue
            claimed = _claim(path)
            if claimed is None:
                continue
            try:
                record = json.loads(claimed.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                record = None
            rows = [(contract.PROCESS_EXIT_MARK, process_exit_fields(record))] if isinstance(record, dict) else []
            settle_claim(claimed, path, emit_saved(rows) == len(rows))
        from .shared_metrics_setup import report_abandoned_setups
        from .shared_metrics_update import report_pending_updates

        report_pending_updates()
        report_abandoned_setups(home)
        # ---- iuf c2 ----
        from .shared_metrics_install_run import report_pending_installs

        report_pending_installs(home)
        # ---- end iuf c2 ----
    except Exception:
        logger.debug("Dead process markers not reported", exc_info=True)
    finally:
        reset_hermes_home_override(token)


def arm_turn(agent: Any) -> None:
    """At turn entry (profile scope bound): the owning home for an off-thread watchdog abort, and
    a fresh once-per-turn latch shared by both turn watchdogs."""
    try:
        from hermes_constants import get_hermes_home

        agent._metrics_turn_home = str(get_hermes_home())
        agent._metrics_watchdog_abort_counted = False
    except Exception:
        logger.debug("Turn exit metrics not armed", exc_info=True)


def record_watchdog_turn_abort(agent: Any) -> None:
    """One exit_kind=watchdog row per turn a watchdog killed, in the turn's own profile. Never raises."""
    try:
        home = getattr(agent, "_metrics_turn_home", None)
        # No armed turn means no known owning profile: never guess the scope.
        if not home or getattr(agent, "_metrics_watchdog_abort_counted", True):
            return
        agent._metrics_watchdog_abort_counted = True
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        from . import shared_metrics_contract as contract
        from .shared_metrics_events import _emit

        token = set_hermes_home_override(home)
        try:
            _emit(contract.PROCESS_EXIT_MARK, lambda: {
                "crash_class": "none", "exit_kind": "watchdog", "process_kind": current_process_kind(),
            })
        finally:
            reset_hermes_home_override(token)
    except Exception:
        logger.debug("Watchdog turn abort not recorded", exc_info=True)
