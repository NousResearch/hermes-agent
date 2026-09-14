"""Restart supervised gateway profiles from a clean Python generation.

The normal update command keeps executing in the interpreter that started before
``git pull``.  This module is deliberately small: it imports no gateway code
itself and launches the regular per-profile gateway command in a new
interpreter.  It is used only after the in-process restart phase has raised, so
that the recovery path cannot inherit the stale ``sys.modules`` graph that
caused the failure.

Outcome vocabulary (deliberately conservative):

- ``verified``          — the relaunch command exited 0 AND the profile's
  systemd unit was independently observed ``active`` afterwards.  This is the
  only outcome that may claim supervisor coverage.
- ``relaunch_attempted`` — the relaunch command exited 0 but no independent
  supervisor observation was possible (non-systemd supervisor, ``systemctl``
  missing, or the unit probe was inconclusive).  ``rc == 0`` from
  ``gateway restart`` is not proof that the new code generation is running,
  so this outcome must never be treated as verified coverage.
- ``failed``            — the relaunch command errored, timed out, or exited
  non-zero.

The pass covers two runtime families, because the in-process restart phase
covers both and an abort can strand either one (#92145):

- **gateway profiles**, relaunched through the existing per-profile
  ``hermes_cli.main -p <profile> gateway restart`` command; and
- **``hermes-serve*`` systemd units**, restarted directly through
  ``systemctl``.  ``hermes serve`` is not a gateway profile and has no
  per-profile relaunch command, but it is the runtime that hosts
  ``tui_gateway.server``: the process the original report saw answering every
  chat turn with an ``ImportError`` for a symbol that existed on disk.  The
  unit family is enumerated from systemd itself rather than from the update
  inventory, so a manually launched or Desktop-owned ``hermes serve`` — which
  has no relaunch authority — can never enter this path.

Serve-unit identity is always ``<scope>/<unit>`` (``user/hermes-serve``,
``system/hermes-serve``).  The two managers can each own a unit of the same
name and they are different processes: an identity that projects the scope
away lets one settled unit suppress recovery of the other, and lets one
scope's outcome speak for the other's.  Scope is therefore never dropped and
reconstructed — not in the skip payload, not in ``verified``/``failed``, not
in the receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

_RECOVERY_ENV = "HERMES_UPDATE_RESTART_RECOVERY"
_GATEWAY_MARKERS = ("_HERMES_GATEWAY", "HERMES_GATEWAY", "HERMES_GATEWAY_MODE")
_PROFILE_RESTART_TIMEOUT = 90
_VERIFY_TIMEOUT = 15
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SUPERVISOR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_UNIT_RE = re.compile(r"^hermes-serve(-[a-z0-9][a-z0-9_-]{0,63})?\.service$")
_SERVE_UNIT_PATTERN = "hermes-serve*"
_SCOPE_LABELS = ("user", "system")
_UNIT_RESTART_TIMEOUT = 60
_UNIT_SETTLE_ATTEMPTS = 10
_UNIT_SETTLE_DELAY = 1.0


def _run_quiet(run: Callable[..., Any], argv: list[str], *, timeout: int, **extra: Any) -> Any | None:
    """Run ``argv`` capturing text output; ``None`` when it errors or times out."""
    try:
        return run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=timeout, **extra,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _stdout(result: Any) -> str:
    return (getattr(result, "stdout", "") or "").strip()


def _succeeded(result: Any) -> bool:
    return result is not None and getattr(result, "returncode", 1) == 0


def _child_environment() -> dict[str, str]:
    """Return an environment that cannot self-identify as the gateway owner."""
    env = os.environ.copy()
    for marker in _GATEWAY_MARKERS:
        env.pop(marker, None)
    env[_RECOVERY_ENV] = "1"
    return env


def _run_profile_restart(profile: str, *, run: Callable[..., Any]) -> bool:
    """Run one profile restart without inheriting the updater's process state."""
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "env": _child_environment()}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kwargs["start_new_session"] = True
    argv = [sys.executable, "-m", "hermes_cli.main", "-p", profile, "gateway", "restart"]
    return _succeeded(_run_quiet(run, argv, timeout=_PROFILE_RESTART_TIMEOUT, **kwargs))


def _systemd_unit_candidates(profile: str) -> tuple[str, ...]:
    """Unit names the existing systemd gateway lifecycle produces per profile."""
    if profile == "default":
        return ("hermes-gateway.service", "gateway.service", "gateway-default.service")
    return (f"hermes-gateway-{profile}.service", f"gateway-{profile}.service")


def _systemd_verified_active(profile: str, *, run: Callable[..., Any]) -> bool:
    """True only when systemd itself reports the profile's unit active — the observation separating
    ``verified`` from ``relaunch_attempted``. Any failure (no ``systemctl``, probe error, unit not
    ``active``) means we could NOT verify, never that the restart failed."""
    systemctl = shutil.which("systemctl")
    return bool(systemctl) and any(
        _unit_is_active([systemctl, "--user"], unit, run=run, require_rc0=True)
        for unit in _systemd_unit_candidates(profile)
    )


def restart_profiles(
    profiles: Iterable[str], *, supervisors: Mapping[str, str] | None = None, run: Callable[..., Any] = subprocess.run
) -> dict[str, list[str]]:
    """Restart the supplied profiles (only ones whose inventory identified a service supervisor).

    A profile only lands in ``verified`` when its supervisor is systemd and ``systemctl --user is-
    active`` independently confirms the unit after the relaunch command succeeded.
    """
    supervisors = supervisors or {}
    result: dict[str, list[str]] = {"verified": [], "relaunch_attempted": [], "failed": []}
    for profile in sorted({p for p in profiles if isinstance(p, str) and p}):
        if not _run_profile_restart(profile, run=run):
            bucket = "failed"
        elif supervisors.get(profile) == "systemd" and _systemd_verified_active(profile, run=run):
            bucket = "verified"
        else:
            bucket = "relaunch_attempted"
        result[bucket].append(profile)
    return result


def _systemctl_scopes() -> list[tuple[str, list[str]]]:
    """``(label, systemctl argv)`` for the user and system scopes (the pair the in-process phase walks), or nothing.

    ``systemctl`` comes from ``shutil.which`` so this module never imports a Hermes platform helper —
    importing the freshly pulled tree is exactly what aborted the phase that called us. Scopes carry
    their label because the same unit name in both managers is two different processes.
    """
    systemctl = shutil.which("systemctl")
    if not systemctl or sys.platform != "linux":
        return []
    return [("user", [systemctl, "--user"]), ("system", [systemctl])]


def _listed_serve_units(scope: list[str], *, run: Callable[..., Any]) -> list[str]:
    """Serve units systemd knows about in one scope, validated by name."""
    result = _run_quiet(
        run,
        scope + ["list-units", _SERVE_UNIT_PATTERN, "--plain", "--no-legend", "--no-pager"],
        timeout=_VERIFY_TIMEOUT,
    )
    if result is None:
        return []
    # The glob is a systemd pattern, not a name gate (`hermes-serve*` also matches
    # `hermes-server.service`): require the exact base unit or the profile family.
    units: list[str] = []
    for line in (getattr(result, "stdout", "") or "").splitlines():
        parts = line.split()
        if parts and _UNIT_RE.fullmatch(parts[0]) and parts[0] not in units:
            units.append(parts[0])
    return units


def _unit_main_pid(scope: list[str], unit: str, *, run: Callable[..., Any]) -> int:
    """The unit's ``MainPID`` via ``systemctl show``; ``0`` when absent or unreadable."""
    result = _run_quiet(run, scope + ["show", unit, "--property=MainPID", "--value"], timeout=_VERIFY_TIMEOUT)
    try:
        return int(_stdout(result) or 0) if _succeeded(result) else 0
    except ValueError:
        return 0


def _unit_is_active(scope: list[str], unit: str, *, run: Callable[..., Any], require_rc0: bool = False) -> bool:
    result = _run_quiet(run, scope + ["is-active", unit], timeout=_VERIFY_TIMEOUT)
    if result is None or (require_rc0 and not _succeeded(result)):
        return False
    return _stdout(result) == "active"


def _serve_unit_replaced(
    scope: list[str], unit: str, previous_pid: int, *, run: Callable[..., Any], sleep: Callable[[float], Any]
) -> bool:
    """Did the unit come back on a NEW main process?

    ``restart`` returning 0 is not evidence: a live process can keep serving the pre-update
    generation while every status command reports success. A changed ``MainPID`` on an ``active``
    unit is the observation that the old interpreter and its stale ``sys.modules`` are gone.
    """
    for attempt in range(_UNIT_SETTLE_ATTEMPTS):
        if attempt:
            sleep(_UNIT_SETTLE_DELAY)
        if _unit_is_active(scope, unit, run=run):
            current = _unit_main_pid(scope, unit, run=run)
            if current > 0 and current != previous_pid:
                return True
    return False


def _split_skip_entry(entry: Any) -> tuple[str, str]:
    """``(scope_label, base_unit)`` of a skip entry in either the mapping or ``scope/unit`` shape."""
    if isinstance(entry, Mapping):
        scope_label = str(entry.get("scope") or "")
        unit = str(entry.get("unit") or "")
    else:
        scope_label, sep, unit = str(entry).partition("/")
        if not sep:
            scope_label, unit = "", scope_label
    return scope_label, unit.removesuffix(".service")


def _normalized_skips(skip_units: Iterable[Any]) -> tuple[set[tuple[str, str]], set[str]]:
    """Split already-settled units into scope-qualified and legacy entries (a bare ``"hermes-serve"``
    carries no scope and cannot say WHICH of two same-named processes was settled)."""
    qualified: set[tuple[str, str]] = set()
    legacy: set[str] = set()
    for entry in skip_units or ():
        scope_label, base = _split_skip_entry(entry)
        if base:
            (qualified.add((scope_label, base)) if scope_label in _SCOPE_LABELS else legacy.add(base))
    return qualified, legacy


def restart_serve_units(
    *, skip_units: Iterable[Any] = (), run: Callable[..., Any] = subprocess.run, sleep: Callable[[float], Any] = time.sleep
) -> dict[str, list[str]]:
    """Restart every active ``hermes-serve*`` systemd unit from this process.

    Units are enumerated from systemd, never from the update inventory, so a manually launched or
    Desktop-owned ``hermes serve`` (no unit) structurally cannot be touched here.
    """
    skipped_qualified, skipped_legacy = _normalized_skips(skip_units)
    # (scope, base unit) -> replaced?  The same unit name in the user and the system
    # scope is two processes; each is proven, reported and accounted for on its own.
    outcomes: dict[tuple[str, str], bool] = {}
    seen: set[tuple[str, str]] = set()
    for scope_label, scope in _systemctl_scopes():
        for unit in _listed_serve_units(scope, run=run):
            base = unit.removesuffix(".service")
            target = (scope_label, base)
            if target in seen or target in skipped_qualified or base in skipped_legacy:
                continue
            seen.add(target)
            if not _unit_is_active(scope, unit, run=run):
                continue  # not running: nothing serves a stale generation from it
            previous_pid = _unit_main_pid(scope, unit, run=run)
            # No readable main PID: a replacement can't be observed, so it can't be claimed
            # (restarting blind and reporting success is the failure mode this module removes).
            # A failed restart includes the unprivileged system-scope case; no sudo probe —
            # an unverifiable unit must read as failed so the update stays explicitly incomplete.
            outcomes[target] = previous_pid > 0 and _succeeded(
                _run_quiet(run, scope + ["--no-ask-password", "restart", unit], timeout=_UNIT_RESTART_TIMEOUT)
            ) and _serve_unit_replaced(scope, unit, previous_pid, run=run, sleep=sleep)
    # ``<scope>/<unit>`` is the only identity this module reports for a serve unit.
    return {
        "verified": sorted(f"{scope}/{base}" for (scope, base), ok in outcomes.items() if ok),
        "failed": sorted(f"{scope}/{base}" for (scope, base), ok in outcomes.items() if not ok),
    }


def _parse_payload(stream) -> tuple[list[str], dict[str, str], bool, list[str]]:
    payload = json.load(stream)
    if not isinstance(payload, dict):
        payload = {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("recovery payload must contain a profiles list")
    if any(not isinstance(profile, str) or not _PROFILE_ID_RE.fullmatch(profile) for profile in profiles):
        raise ValueError("recovery profiles contain an invalid profile id")
    raw_supervisors = payload.get("supervisors")
    supervisors: dict[str, str] = {}
    if raw_supervisors is not None:
        if not isinstance(raw_supervisors, dict) or any(
            not isinstance(profile, str)
            or not isinstance(supervisor, str)
            or not _PROFILE_ID_RE.fullmatch(profile)
            or not _SUPERVISOR_RE.fullmatch(supervisor)
            for profile, supervisor in raw_supervisors.items()
        ):
            raise ValueError("recovery supervisors map is invalid")
        supervisors = dict(raw_supervisors)
    raw_serve = payload.get("serve_units")
    recover_serve = False
    skip_units: list[str] = []
    if raw_serve is not None:
        if not isinstance(raw_serve, dict):
            raise ValueError("recovery serve_units block is invalid")
        recover_serve = bool(raw_serve.get("recover"))
        raw_skip = raw_serve.get("skip") or []
        if not isinstance(raw_skip, list) or any(not isinstance(entry, (str, dict)) for entry in raw_skip):
            raise ValueError("recovery serve_units skip list is invalid")
        for entry in raw_skip:
            # A skip entry names one already-settled process. The qualified shape carries the systemd
            # scope (`hermes-serve.service` can exist in BOTH managers; settling one says nothing about
            # the other). A bare string is the legacy shape a pre-update interpreter can still send,
            # read as scope-agnostic by `restart_serve_units`.
            if isinstance(entry, dict):
                if not isinstance(entry.get("unit"), str):
                    raise ValueError("recovery serve_units skip list is invalid")
                if not isinstance(entry.get("scope"), str):
                    entry = {"unit": entry["unit"]}
            scope_label, base = _split_skip_entry(entry)
            # Only shapes systemd can produce for this family (a skip is a name filter, never a command
            # argument). An unrecognized scope DROPS the entry rather than raising: a dropped skip costs at
            # most one extra restart-and-verify; honouring an unreadable one could suppress recovery.
            if (scope_label and scope_label not in _SCOPE_LABELS) or not _UNIT_RE.fullmatch(f"{base}.service"):
                continue
            skip_units.append(f"{scope_label}/{base}" if scope_label else base)
    return profiles, supervisors, recover_serve, skip_units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.stdin:
        parser.error("this command is an internal update-recovery entry point")

    try:
        profiles, supervisors, recover_serve, skip_units = _parse_payload(sys.stdin)
        result = restart_profiles(profiles, supervisors=supervisors)
        result["serve_units"] = (
            restart_serve_units(skip_units=skip_units) if recover_serve else {"verified": [], "failed": []}
        )
    except (ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "error": str(exc), "verified": [], "relaunch_attempted": [], "failed": [],
            "serve_units": {"verified": [], "failed": []},
        }))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 1 if result["failed"] or result["serve_units"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
