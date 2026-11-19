"""Cron: per-occurrence process containment and teardown.

Invariant (kit card t_d66ef645): no process spawned by a cron occurrence may
outlive the agent session, and no store write may happen outside the
supervised run's lifetime.

Why a cgroup, not just PDEATHSIG/process groups:

- ``PR_SET_PDEATHSIG`` is not inherited across ``fork()`` — it covers only the
  direct child. Provider launchers fork their own children and call
  ``setsid()`` (new session), which escapes every process-group kill. The
  2026-11-14 Morgan orphan ran exactly through that hole.
- A systemd transient *scope* is a cgroup: ``fork()``, ``setsid()`` and new
  sessions all stay inside it. ``systemctl --user stop`` kills the whole
  cgroup regardless of session/group topology.

How the pieces fit:

- ``begin(execution_id)`` creates the per-execution registry under
  ``$HERMES_HOME/cron/containment/<execution_id>/``. The caller publishes the
  registry dir and the host service unit into the run's session ContextVars
  (``gateway/session_context._VAR_MAP``), which the terminal tool bridges
  into every child-process env — so the occurrence's driver can discover them
  without any global ``os.environ`` mutation (concurrent jobs never clobber).
- The driver re-execs itself under ``systemd-run --user --scope`` and
  registers the scope unit here; everything the occurrence spawns lands in
  that cgroup.
- ``end(execution_id)`` synchronously stops and verifies every registered
  unit BEFORE the cron session is finalized: from ``run_job``'s ``finally``
  on every exit path (success, exception, cancellation), and — when the
  conversation thread outlives ``run_job`` (#102827 deferred teardown) — from
  the worker future's done callback, so teardown follows the conversation's
  real end, not the verdict.
- Crash backstop: the driver creates scopes bound to the systemd service
  hosting this process (``host_service_unit()``, e.g. ``hermes-gateway.service``),
  so a service stop/crash takes the scopes with it. Where no host unit is
  detectable, registries stay on disk and ``sweep_stale`` cleans them on the
  next fire once the execution is no longer running.

Containment is best-effort at this layer by design: a host without a systemd
user manager still runs cron jobs (``begin`` returns ``None`` and no
containment env is exported). The *invariant carrier* is the driver-side
supervisor, which fails closed — it refuses to launch roles it cannot
contain. This module only makes the strong containment available where the
platform supports it.

Residual risk, stated honestly: a SIGKILLed host process on a host with no
detectable service unit leaves scopes running until the next fire's sweep.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Overridable for tests (monkeypatch), like other external-command seams.
SYSTEMCTL = "systemctl"

_UNITS_FILENAME = "units"
_UNIT_PREFIX = "hermes-cron-"
# Bounded teardown: the synchronous stop must never hold the dispatch guard.
_STOP_TIMEOUT_SECONDS = 10.0
_KILL_SETTLE_SECONDS = 2.0
_PROC_CGROUP_ROOT = Path("/sys/fs/cgroup")


def containment_root() -> Path:
    """Per-profile root so a multiplexed gateway never touches another home."""
    return get_hermes_home().resolve() / "cron" / "containment"


def _validate_execution_id(execution_id: object) -> str:
    execution_id = str(execution_id or "").strip()
    if not execution_id or "/" in execution_id or "\\" in execution_id or execution_id in {".", ".."}:
        raise ValueError(f"Not a usable execution id: {execution_id!r}")
    return execution_id


def registry_dir(execution_id: object) -> Path:
    return containment_root() / _validate_execution_id(execution_id)


def new_unit_name(execution_id: object) -> str:
    """Fresh transient-scope name, unique per dispatch. Valid systemd unit name
    (alnum, ``-``, ``_``, ``.``) and traceable back to its execution."""
    return f"{_UNIT_PREFIX}{_validate_execution_id(execution_id)}-{uuid.uuid4().hex[:8]}.scope"


def host_service_unit() -> str:
    """The systemd user service unit hosting THIS process, or "" when the
    process does not run inside a user-service cgroup (CLI, tests)."""
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            return _parse_host_unit(handle.readline().strip())
    except OSError:
        return ""


def _parse_host_unit(cgroup_line: str) -> str:
    """Pure parser so the user-service detection is directly testable."""
    # cgroup v2: "0::<path>"; the unit is the last component under the user slice.
    path = cgroup_line.split("::", 1)[-1] if "::" in cgroup_line else ""
    if not path:
        return ""
    unit = path.rstrip("/").rsplit("/", 1)[-1]
    if unit.endswith(".service") and "user@" in path:
        return unit
    return ""


def begin(execution_id: object) -> Optional[dict[str, str]]:
    """Create the per-execution registry. Returns ``{"dir", "host_unit"}`` for
    the caller to publish into session ContextVars, or ``None`` when the
    registry cannot be provided (invalid id, unwritable home) — in which case
    NO containment env may be exported and the driver fails closed."""
    try:
        directory = registry_dir(execution_id)
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
    except (OSError, ValueError) as exc:
        logger.warning("cron containment: registry unavailable (%s) — exporting no containment env", exc)
        return None
    return {"dir": str(directory), "host_unit": host_service_unit()}


def register_unit(execution_id: object, unit: str) -> bool:
    """Record a containment unit for this execution. Returns False when the
    registry is gone (already torn down) or the write failed."""
    try:
        directory = registry_dir(execution_id)
        directory.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return False
    if not _valid_unit_name(unit):
        logger.warning("cron containment: refusing to register malformed unit %r", unit)
        return False
    try:
        existing = _read_units(directory)
        if unit not in existing:
            _append_unit(directory, unit)
        return True
    except OSError as exc:
        logger.warning("cron containment: could not register %s: %s", unit, exc)
        return False


def _valid_unit_name(unit: str) -> bool:
    if not unit.startswith(_UNIT_PREFIX) or not unit.endswith(".scope"):
        return False
    return all(c.isalnum() or c in "-_." for c in unit)


def _read_units(directory: Path) -> list[str]:
    path = directory / _UNITS_FILENAME
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _append_unit(directory: Path, unit: str) -> None:
    path = directory / _UNITS_FILENAME
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(unit + "\n")


def registered_units(execution_id: object) -> list[str]:
    try:
        return _read_units(registry_dir(execution_id))
    except (OSError, ValueError):
        return []


def unit_state(unit: str) -> tuple[str, Optional[str]]:
    """(ActiveState, ControlGroup path or None) for a transient unit."""
    try:
        proc = subprocess.run(
            [SYSTEMCTL, "--user", "show", "--property=ActiveState", "--property=ControlGroup",
             unit],
            capture_output=True, text=True, timeout=_STOP_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error:{type(exc).__name__}", None
    if proc.returncode != 0 and not proc.stdout.strip():
        return f"error:exit{proc.returncode}", None
    # Property ORDER is not guaranteed by `systemctl show` (ControlGroup may
    # precede ActiveState) — parse key=value pairs, never positional lines.
    props: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        name, _, value = line.partition("=")
        if name in ("ActiveState", "ControlGroup"):
            props[name] = value.strip()
    return props.get("ActiveState", "unknown"), (props.get("ControlGroup", "") or None)


def cgroup_procs(cgroup: Optional[str]) -> list[int]:
    """PIDs still inside a cgroup path. Missing path == empty (already gone)."""
    if not cgroup:
        return []
    path = _PROC_CGROUP_ROOT / cgroup.lstrip("/") / "cgroup.procs"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [int(pid) for pid in handle.read().split() if pid.isdigit()]
    except (OSError, ValueError):
        return []


def stop_unit(unit: str) -> str:
    """Stop one transient scope and verify its cgroup is empty.

    Returns ``"clean"`` (stopped/verified empty), ``"gone"`` (already dead),
    or ``"leftover:<detail>"`` / ``"error:<detail>"`` — anything not clean is
    reported as NOT dead, never swallowed as success."""
    state, cgroup = unit_state(unit)
    if state.startswith("error:"):
        return state
    if state in ("inactive", "failed", "not-found", "unknown", "inactive (dead)"):
        if not cgroup_procs(cgroup):
            return "gone" if state in ("not-found", "unknown") else "clean"
    try:
        subprocess.run([SYSTEMCTL, "--user", "stop", unit],
                       capture_output=True, text=True,
                       timeout=_STOP_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        pass
    except OSError as exc:
        return f"error:{type(exc).__name__}"
    state, cgroup = unit_state(unit)
    if cgroup_procs(cgroup):
        # Escalate once: SIGKILL the whole cgroup, then re-verify.
        try:
            subprocess.run([SYSTEMCTL, "--user", "kill", "--signal=SIGKILL", unit],
                           capture_output=True, text=True,
                           timeout=_STOP_TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(_KILL_SETTLE_SECONDS)
        state, cgroup = unit_state(unit)
    if state.startswith("error:"):
        return state
    if cgroup_procs(cgroup):
        return f"leftover:{state}"
    return "clean"


def end(execution_id: object) -> dict[str, list[str]]:
    """Synchronously stop and verify EVERY unit registered for this execution.

    Called before the cron session is finalized. Never raises: a teardown
    failure is returned, not thrown — the caller logs it and the registry dir
    is kept so the next fire's sweep can retry."""
    report: dict[str, list[str]] = {"stopped": [], "leftover": []}
    try:
        directory = registry_dir(execution_id)
    except ValueError as exc:
        logger.warning("cron containment: bad execution id: %s", exc)
        return report
    for unit in _read_units(directory):
        status = stop_unit(unit)
        if status in ("clean", "gone"):
            report["stopped"].append(unit)
        else:
            report["leftover"].append(f"{unit}: {status}")
            logger.error("cron containment: execution %s unit %s NOT clean: %s",
                         execution_id, unit, status)
    if not report["leftover"]:
        shutil.rmtree(directory, ignore_errors=True)
    return report


def sweep_stale(still_running: Callable[[str], bool]) -> int:
    """Tear down registries whose execution is no longer running.

    Called at the start of a fire (lazy sweep): covers executions whose host
    process died without teardown on hosts without a bindable service unit.
    Returns the number of executions swept."""
    swept = 0
    try:
        entries = list(containment_root().iterdir())
    except OSError:
        return 0
    for directory in entries:
        if not directory.is_dir():
            continue
        execution_id = directory.name
        try:
            if still_running(execution_id):
                continue
        except Exception:  # probe failure must never block the fire
            logger.debug("cron containment: liveness probe failed for %s", execution_id, exc_info=True)
            continue
        report = end(execution_id)
        if report["stopped"] or not report["leftover"]:
            swept += 1
        if report["leftover"]:
            logger.error("cron containment: stale sweep for %s left survivors: %s",
                         execution_id, report["leftover"])
    return swept
