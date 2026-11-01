"""Content-free telemetry for processes Hermes keeps outside any turn: Bot Desktop screens and the child
process groups the owner-death supervisor reaps.

Most of these moments happen where no exporter runs: a screen's own idle check (a short-lived process
the launcher starts), the supervisor reaping after Hermes died, a dashboard backend. :func:`record`
emits in-process when this process exports, and otherwise leaves the event in the home's spool
(``<HERMES_HOME>/monitoring/spool``), which the exporting gateway drains. A gateway that exports creates
the spool of each home it serves and keeps it (records written while it is down wait for it); one that
does not drain them (export or diagnostics off, or a failed start) removes them (:func:`set_spooling`),
and a spool nobody drains stops at ``_MAX_SPOOLED``.
The shell writers (``launcher.sh``, ``process_supervisor.sh``) write the same JSON shape by hand.

Kept cheap to import: a screen's idle check and the owner-death supervisor's spawn path load it.

Every field is a closed vocabulary or a small count: an event that does not validate is dropped.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from agent.monitoring.events import ManagedProcessEvent

logger = logging.getLogger(__name__)

BOT_DESKTOP = "bot_desktop"
OWNED_CHILDREN = "owned_children"
# subsystem -> action -> allowed outcomes; actors and reasons are shared closed sets.
_VOCABULARY: dict[str, dict[str, frozenset[str]]] = {
    BOT_DESKTOP: {
        "idle_stop": frozenset({"stopped", "kept"}),
        "lease_handback": frozenset({"released", "kept_newer", "write_failed"}),
        "self_stop_off": frozenset({"sleep_failed"}),
    },
    OWNED_CHILDREN: {
        "orphan_reap": frozenset({"terminated", "killed"}),
        "supervisor_exit": frozenset({"restarted", "not_restarted"}),
        "supervisor_unavailable": frozenset({"spawn_failed", "registration_failed"}),
    },
}
_ACTORS = frozenset({"watcher", "screen", "supervisor", "owner"})
_REASONS = frozenset({"idle", "abandoned_lease"})
_MAX_COUNT = 1_000_000
# A spool the gateway stopped draining (it died, export was turned off elsewhere) stops growing here. The shell
# writers (launcher.sh, process_supervisor.sh) enforce the same number; a test keeps them equal.
_MAX_SPOOLED = 1_000
_MAX_FILE_BYTES = 4096
# A record's own time is kept only inside this window around now (a broken clock, a hand-written 5, or a value
# past OTLP's 64-bit range, which fails its whole export batch, are stamped with the drain's time instead).
_TS_PAST_NS = 30 * 86_400 * 10**9
_TS_FUTURE_NS = 86_400 * 10**9
# A writer killed between creating its temp file and renaming it leaves the temp file behind.
_ABANDONED_TMP_S = 60


def spool_dir(home: Optional[Path] = None) -> Path:
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    return Path(home) / "monitoring" / "spool"


def _validated(data: Any) -> Optional[ManagedProcessEvent]:
    """The event, or None when any field falls outside the vocabulary (a hand-written or corrupt file)."""
    if not isinstance(data, dict) or data.get("event", ManagedProcessEvent.EVENT) != ManagedProcessEvent.EVENT:
        return None
    subsystem, action, outcome, actor, reason, count = (
        data.get(k) for k in ("subsystem", "action", "outcome", "actor", "reason", "count"))
    if not all(isinstance(v, str) for v in (subsystem, action, outcome)):
        return None
    outcomes = _VOCABULARY.get(subsystem, {}).get(action)
    if outcomes is None or outcome not in outcomes:
        return None
    if actor is not None and (not isinstance(actor, str) or actor not in _ACTORS):
        return None
    if reason is not None and (not isinstance(reason, str) or reason not in _REASONS):
        return None
    if count is not None and (type(count) is not int or not 0 <= count <= _MAX_COUNT):
        return None
    ts_ns, now = data.get("ts_ns"), time.time_ns()
    if type(ts_ns) is not int or not now - _TS_PAST_NS <= ts_ns <= now + _TS_FUTURE_NS:
        ts_ns = now
    return ManagedProcessEvent(subsystem=subsystem, action=action, outcome=outcome, actor=actor, reason=reason,
                               count=count, ts_ns=ts_ns)


def record(subsystem: str, action: str, outcome: str, *, actor: Optional[str] = None,
           reason: Optional[str] = None, count: Optional[int] = None) -> None:
    """Report one managed-process moment. Never raises and never blocks on more than one small file write."""
    try:
        event = _validated({"subsystem": subsystem, "action": action, "outcome": outcome, "actor": actor,
                            "reason": reason, "count": count})
        if event is None:
            logger.debug("managed-process event outside the vocabulary dropped: %s/%s/%s", subsystem, action, outcome)
            return
        from agent.monitoring import emitter
        bus = emitter.get_emitter()
        if bus.enabled:
            bus.emit(event)
            return
        _publish(spool_dir(), f"{event.ts_ns}-{os.getpid()}.json", json.dumps(event.to_dict()))
    except Exception:
        logger.debug("managed-process telemetry record failed", exc_info=True)


def _publish(directory: Path, name: str, payload: str) -> None:
    """Write one record into an EXISTING spool, atomically (temp file + rename, so the drainer never reads a
    partial one). Never creates the directory: it existing is the permission to collect, and a writer racing
    the export being turned off must not bring it back. A full or missing spool drops the event."""
    try:
        if sum(1 for _ in directory.iterdir()) >= _MAX_SPOOLED:  # temp files too: a dying writer's count
            return
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".record-", suffix=".tmp")
    except OSError:
        return  # no spool (export off for this home) or not writable
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:  # owns fd from here: closed on every failure below
            if hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), 0o644)  # mkstemp's 0600 would lock out a gateway draining as another user
            f.write(payload)
        os.replace(tmp, directory / name)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def drain_spool(homes: Iterable[Path]) -> list[ManagedProcessEvent]:
    """Collect and remove every spooled event of ``homes``, oldest first. A file that does not validate is
    removed as well: it can never become valid."""
    events: list[ManagedProcessEvent] = []
    for home in distinct_homes(homes):
        directory = spool_dir(home)
        try:
            paths = list(directory.glob("*.json"))
            _remove_abandoned_temp_files(directory)
        except OSError:
            continue
        for path in paths:
            try:
                with path.open("rb") as f:
                    raw = f.read(_MAX_FILE_BYTES + 1)  # an oversized file costs at most this much memory
                path.unlink()
            except OSError:
                continue
            try:  # per file: one malformed record must not cost the valid ones drained with it
                event = _validated(json.loads(raw)) if len(raw) <= _MAX_FILE_BYTES else None
            except (ValueError, RecursionError):  # not JSON, or nested past the parser's depth
                event = None
            if event is not None:
                events.append(event)
    events.sort(key=lambda event: event.ts_ns)
    return events


def _remove_abandoned_temp_files(directory: Path) -> None:
    """Temp files (``.*.tmp``) a writer died holding. A live writer renames its file within milliseconds."""
    cutoff = time.time() - _ABANDONED_TMP_S
    for path in directory.glob(".*.tmp"):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < cutoff:
                path.unlink()


def distinct_homes(homes: Iterable[Path]) -> list[Path]:
    """``homes`` without repeats of one home under another spelling (a multiplexer's served set also lists the
    launch profile): a screen or a record must be counted once."""
    from hermes_constants import hermes_home_key
    return list({hermes_home_key(home): Path(home) for home in homes}.values())


def set_spooling(homes: Iterable[Path], enabled: bool) -> None:
    """Create (export on) or remove (export off) the spool of each home. Fail-open per home."""
    for home in distinct_homes(homes):
        directory = spool_dir(home)
        try:
            for leftover in directory.parent.glob(f"{directory.name}.removing-*"):
                shutil.rmtree(leftover, ignore_errors=True)
            if enabled:
                from hermes_constants import mkdir_under_hermes_home
                mkdir_under_hermes_home(directory)  # never brings back a profile deleted meanwhile
            elif directory.is_dir():
                # Out of service first, in one rename: a writer publishing while the tree is deleted lands in
                # the retired directory (or fails), never in one that keeps authorizing collection.
                retired = directory.with_name(f"{directory.name}.removing-{os.getpid()}-{time.time_ns()}")
                os.replace(directory, retired)
                shutil.rmtree(retired, ignore_errors=True)
        except OSError:
            logger.debug("managed-process spool %s could not be %s", directory, "created" if enabled else "removed",
                         exc_info=True)


def _bot_desktop_metrics(homes: Iterable[Path]) -> list:  # of GatewayMetric (imported here: it pulls agent.redact)
    from agent.monitoring.gateway_health import GatewayMetric
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.bot_desktop import lease, runtime
    running = human_held = 0
    for home in distinct_homes(homes):
        token = set_hermes_home_override(home)
        try:
            if runtime.is_running_locally():
                running += 1
                human_held += lease.get().holder == lease.HUMAN
        finally:
            reset_hermes_home_override(token)
    return [GatewayMetric("hermes.bot_desktop.screens.running", running, {}),
            GatewayMetric("hermes.bot_desktop.screens.human_held", human_held, {})]


def _owned_children_metrics() -> list:  # of GatewayMetric
    from agent.monitoring.gateway_health import GatewayMetric
    from hermes_cli.process_identity import supervised_group_count
    return [GatewayMetric("hermes.owned_children.groups", supervised_group_count(), {})]


def build_managed_process_metrics(homes: Callable[[], Iterable[Path]]) -> list:  # of GatewayMetric
    """The managed-process gauges. Each group is fail-open on its own, with a WARNING naming only the
    exception type (it could carry a path), so a reader that regresses is visible instead of silently gone."""
    metrics: list = []
    for name, build in (("bot-desktop", lambda: _bot_desktop_metrics(homes())), ("owned-children", _owned_children_metrics)):
        try:
            metrics.extend(build())
        except Exception as exc:
            logger.warning("%s gauges unavailable; not exported (error_type=%s)", name, type(exc).__name__)
            logger.debug("%s gauges traceback", name, exc_info=True)
    return metrics


__all__ = [
    "BOT_DESKTOP",
    "OWNED_CHILDREN",
    "build_managed_process_metrics",
    "distinct_homes",
    "drain_spool",
    "record",
    "set_spooling",
    "spool_dir",
]
