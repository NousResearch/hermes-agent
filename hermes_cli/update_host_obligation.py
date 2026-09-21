"""Host-scoped update→restart obligation for ``hermes update``.

Multiplex-only (Teknium ruling): exactly ONE ``hermes gateway run`` per host serves every
profile, so "this pull still owes the fleet a restart" is a property of the HOST, not of one
profile's ``HERMES_HOME``. The legacy ``$HERMES_HOME/fleet_restart_pending`` marker was
per-home: ``hermes -p coder update`` armed and cleared coder's copy while restarting the
SHARED process, and every other profile's CLI could neither see nor discharge that obligation
— it simply armed its own and re-killed the same host process.

The record therefore lives beside the host rendezvous record, in
:func:`gateway.host_rendezvous.host_state_dir` (``$HERMES_GATEWAY_LOCK_DIR`` else
``$XDG_STATE_HOME/hermes/gateway-locks``) — the one cross-profile, per-OS-user state root the
tree already has. It is written once per host, read by every profile's CLI, and cleared once.

The same "one host process, not one per profile" identity is what
:func:`collapse_units_to_host_processes` applies to enumerated systemd units: leftover
per-profile ``hermes-gateway-<p>.service`` units on a multiplexed host all point at the same
live ``MainPID``, so restarting each one restarts the host process N times.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("hermes_cli.update_cmd")

#: One file per OS user, beside ``host-gateway.json`` / ``host-serve.json``.
HOST_OBLIGATION_NAME = "host-update-restart.json"

_RECORD_VERSION = 1


def host_obligation_path() -> Optional[Path]:
    """Path of the host obligation record, or ``None`` when the host state dir is unresolvable."""
    try:
        from gateway.host_rendezvous import host_state_dir

        return host_state_dir() / HOST_OBLIGATION_NAME
    except Exception:  # pragma: no cover - import/env failure must never break the updater
        logger.debug("Host obligation path unavailable", exc_info=True)
        return None


def read_host_obligation() -> Optional[dict]:
    """The published obligation record, or ``None`` when absent/corrupt/foreign-versioned."""
    path = host_obligation_path()
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _RECORD_VERSION:
        return None
    return payload


def host_obligation_present() -> bool:
    """True when the record FILE exists, parseable or not.

    Fail-closed: a corrupt record is an obligation whose terms are unknown, never a discharged
    one — the restart is still owed and the reader falls back to "no recorded inventory".
    """
    path = host_obligation_path()
    if path is None:
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def amend_host_obligation(**fields: Any) -> None:
    """Merge ``fields`` into the armed record (test/diagnostic surface). Never raises."""
    record = read_host_obligation()
    path = host_obligation_path()
    if record is None or path is None:
        return
    record.update(fields)
    try:
        from utils import atomic_json_write

        atomic_json_write(path, record, mode=0o600)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not amend host update-restart obligation: %s", exc)


def write_host_obligation(
    *, expected_sha: str = "", runtimes: Optional[list] = None, profile: str = ""
) -> bool:
    """Arm the host obligation. True when it was written. Never raises.

    Re-arming from a second profile for the SAME pulled SHA keeps the existing record (and its
    ``restarted`` proof) instead of resetting it: the host owes one restart, not one per profile.
    """
    path = host_obligation_path()
    if path is None:
        return False
    existing = read_host_obligation()
    if existing is not None and expected_sha and existing.get("expected_sha") == expected_sha:
        # Same pull, second profile: the host owes ONE restart, so keep the standing record (and
        # any proof that the restart already happened) rather than resetting it. A later arm that
        # carries the owed inventory still upgrades it — an inventory-less record owes no set.
        if runtimes is None:
            return True
        inventory = {"version": 1, "runtimes": runtimes}
        if existing.get("inventory") != inventory:
            amend_host_obligation(inventory=inventory)
        return True
    payload: dict[str, Any] = {
        "version": _RECORD_VERSION,
        "started": time.time(),
        "pid": os.getpid(),
        "armed_by_profile": profile or "",
        "expected_sha": expected_sha or "",
    }
    if runtimes is not None:
        payload["inventory"] = {"version": 1, "runtimes": runtimes}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from utils import atomic_json_write

        atomic_json_write(path, payload, mode=0o600)
    except Exception as exc:
        logger.debug("Could not write host update-restart obligation: %s", exc)
        return False
    return True


def clear_host_obligation() -> None:
    """Discharge the obligation for the whole host. Never raises."""
    path = host_obligation_path()
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Could not clear host update-restart obligation: %s", exc)


def obligation_fields() -> Optional[dict[str, str]]:
    """The obligation in the legacy ``key=value`` field shape, or ``None`` when unarmed.

    Keeps one parser for both sources: the fields a reader needs (``expected_sha``, the
    serialized ``inventory``) are identical whether they came from the host record or from an
    in-flight legacy per-home marker.
    """
    record = read_host_obligation()
    if record is None:
        return None
    fields = {"expected_sha": str(record.get("expected_sha") or "")}
    inventory = record.get("inventory")
    if inventory is not None:
        fields["inventory"] = json.dumps(inventory)
    return fields


def mark_host_restart_completed(sha: str) -> None:
    """Record that the host process was restarted onto ``sha``. Never raises."""
    record = read_host_obligation()
    path = host_obligation_path()
    if record is None or path is None:
        return
    record["restarted"] = {"sha": sha or "", "pid": os.getpid(), "at": time.time()}
    try:
        from utils import atomic_json_write

        atomic_json_write(path, record, mode=0o600)
    except Exception as exc:
        logger.debug("Could not stamp host restart completion: %s", exc)


def host_restart_already_completed(sha: Optional[str]) -> bool:
    """True when THIS host obligation was already restarted onto ``sha``.

    The guard that makes the catch-up restart idempotent per host: a second profile running
    ``hermes update`` must attach to the first restart's outcome, never kill the shared
    multiplexer again.
    """
    record = read_host_obligation()
    if record is None or not sha:
        return False
    restarted = record.get("restarted")
    return isinstance(restarted, dict) and str(restarted.get("sha") or "") == sha


def collapse_units_to_host_processes(
    units: Iterable[str], main_pid: Callable[[str], int]
) -> tuple[list[str], dict[str, str]]:
    """Split enumerated units into ``(restart, {legacy_unit: covering_unit})``.

    Units resolving to the same live ``MainPID`` are ONE host process; restarting each of them
    restarts that process N times, which on a multiplexed host is an N-fold outage triggered by
    leftover per-profile units. A unit with no readable main PID (inactive, unprivileged scope)
    keeps its own restart: identity that cannot be proved is never collapsed away.
    """
    restart: list[str] = []
    covered: dict[str, str] = {}
    owner_by_pid: dict[int, str] = {}
    for unit in units:
        try:
            pid = int(main_pid(unit) or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid <= 0:
            restart.append(unit)
            continue
        owner = owner_by_pid.get(pid)
        if owner is None:
            owner_by_pid[pid] = unit
            restart.append(unit)
        else:
            covered[unit] = owner
    return restart, covered
