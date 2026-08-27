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
_UNIT_RESTART_TIMEOUT = 60
_UNIT_SETTLE_ATTEMPTS = 10
_UNIT_SETTLE_DELAY = 1.0


def _profile_command(profile: str) -> list[str]:
    """Build a parameterized restart command for exactly one profile."""
    return [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "-p",
        profile,
        "gateway",
        "restart",
    ]


def _child_environment() -> dict[str, str]:
    """Return an environment that cannot self-identify as the gateway owner."""
    env = os.environ.copy()
    for marker in _GATEWAY_MARKERS:
        env.pop(marker, None)
    env[_RECOVERY_ENV] = "1"
    return env


def _run_profile_restart(
    profile: str,
    *,
    run: Callable[..., Any],
) -> bool:
    """Run one profile restart without inheriting the updater's process state."""
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
        "timeout": _PROFILE_RESTART_TIMEOUT,
        "env": _child_environment(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True

    try:
        result = run(_profile_command(profile), **kwargs)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return getattr(result, "returncode", 1) == 0


def _systemd_unit_candidates(profile: str) -> tuple[str, ...]:
    """Unit names the existing systemd gateway lifecycle produces per profile."""
    if profile == "default":
        return (
            "hermes-gateway.service",
            "gateway.service",
            "gateway-default.service",
        )
    return (
        f"hermes-gateway-{profile}.service",
        f"gateway-{profile}.service",
    )


def _systemd_verified_active(profile: str, *, run: Callable[..., Any]) -> bool:
    """Return True only when systemd itself reports the profile's unit active.

    This is the observation that separates ``verified`` from
    ``relaunch_attempted``.  Any failure here (no ``systemctl``, probe error,
    unit not ``active``) means we could NOT verify — never that the restart
    failed.
    """
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    for unit in _systemd_unit_candidates(profile):
        try:
            result = run(
                [systemctl, "--user", "is-active", unit],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=_VERIFY_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if (
            getattr(result, "returncode", 1) == 0
            and (getattr(result, "stdout", "") or "").strip() == "active"
        ):
            return True
    return False


def restart_profiles(
    profiles: Iterable[str],
    *,
    supervisors: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, list[str]]:
    """Restart the supplied profiles and return per-profile terminal results.

    The caller supplies only profiles whose inventory identified a service
    supervisor.  Manual gateways are intentionally excluded before this module
    is called: killing one without a relaunch authority would turn stale code
    into an outage.

    A profile only lands in ``verified`` when its supervisor is systemd and
    ``systemctl --user is-active`` independently confirms the unit after the
    relaunch command succeeded.  Every other zero-exit relaunch is reported as
    ``relaunch_attempted`` — the code cannot observe supervisor coverage for
    those paths and must not claim it.
    """
    supervisors = supervisors or {}
    normalized = sorted(
        {profile for profile in profiles if isinstance(profile, str) and profile}
    )
    verified: list[str] = []
    relaunch_attempted: list[str] = []
    failed: list[str] = []
    for profile in normalized:
        if not _run_profile_restart(profile, run=run):
            failed.append(profile)
            continue
        if supervisors.get(profile) == "systemd" and _systemd_verified_active(
            profile, run=run
        ):
            verified.append(profile)
        else:
            relaunch_attempted.append(profile)
    return {
        "verified": verified,
        "relaunch_attempted": relaunch_attempted,
        "failed": failed,
    }


def _systemctl_scopes() -> list[list[str]]:
    """``systemctl`` invocations for the user and system scopes, or nothing.

    Mirrors the scope pair the in-process restart phase walks. ``systemctl``
    is resolved through ``shutil.which`` so this module never has to import
    any Hermes platform helper — importing the freshly pulled tree is exactly
    what aborted the phase that called us.
    """
    systemctl = shutil.which("systemctl")
    if not systemctl or sys.platform != "linux":
        return []
    return [[systemctl, "--user"], [systemctl]]


def _listed_serve_units(scope: list[str], *, run: Callable[..., Any]) -> list[str]:
    """Serve units systemd knows about in one scope, validated by name."""
    try:
        result = run(
            scope
            + [
                "list-units",
                _SERVE_UNIT_PATTERN,
                "--plain",
                "--no-legend",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_VERIFY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    units: list[str] = []
    for line in (getattr(result, "stdout", "") or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        # The glob is a systemd pattern, not a name gate: `hermes-serve*` also
        # matches the unrelated `hermes-server.service`. Require the exact
        # base unit or the hyphenated profile family, same shape as the
        # in-process phase's own name gate.
        if _UNIT_RE.fullmatch(parts[0]) and parts[0] not in units:
            units.append(parts[0])
    return units


def _unit_property(
    scope: list[str], unit: str, prop: str, *, run: Callable[..., Any]
) -> str | None:
    """One ``systemctl show`` property, or ``None`` when it cannot be read."""
    try:
        result = run(
            scope + ["show", unit, f"--property={prop}", "--value"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_VERIFY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    return (getattr(result, "stdout", "") or "").strip()


def _unit_main_pid(scope: list[str], unit: str, *, run: Callable[..., Any]) -> int:
    """The unit's ``MainPID``; ``0`` when absent or unreadable."""
    raw = _unit_property(scope, unit, "MainPID", run=run)
    try:
        return int(raw or 0)
    except ValueError:
        return 0


def _unit_is_active(scope: list[str], unit: str, *, run: Callable[..., Any]) -> bool:
    try:
        result = run(
            scope + ["is-active", unit],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_VERIFY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (getattr(result, "stdout", "") or "").strip() == "active"


def _serve_unit_replaced(
    scope: list[str],
    unit: str,
    previous_pid: int,
    *,
    run: Callable[..., Any],
    sleep: Callable[[float], Any],
) -> bool:
    """Did the unit come back on a NEW main process?

    ``restart`` returning 0 is not evidence: the whole point of #92145 is that
    a live process can keep serving the pre-update generation while every
    status command reports success. A changed ``MainPID`` on an ``active``
    unit is the observation that the old interpreter — and its stale
    ``sys.modules`` — is gone.
    """
    for attempt in range(_UNIT_SETTLE_ATTEMPTS):
        if attempt:
            sleep(_UNIT_SETTLE_DELAY)
        if not _unit_is_active(scope, unit, run=run):
            continue
        current = _unit_main_pid(scope, unit, run=run)
        if current > 0 and current != previous_pid:
            return True
    return False


def restart_serve_units(
    *,
    skip_units: Iterable[str] = (),
    run: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], Any] = time.sleep,
) -> dict[str, list[str]]:
    """Restart every active ``hermes-serve*`` systemd unit from this process.

    ``hermes serve`` hosts ``tui_gateway.server`` and is restarted by the
    in-process phase alongside the gateway units, but it is not a gateway
    profile: no ``gateway restart`` command reaches it. When the phase aborts
    part-way — systemd lists ``hermes-gateway.service`` before
    ``hermes-serve.service``, so the gateway is typically already done — the
    serve unit is the one left holding generation-N modules over a
    generation-N+1 checkout.

    Units are enumerated from systemd, never from the update inventory. That
    keeps the relaunch authority requirement structural: a manually launched
    or Desktop-owned ``hermes serve`` owns no unit and therefore cannot be
    touched here.

    Returns ``{"verified": [...], "failed": [...]}`` keyed by base unit name
    (no ``.service`` suffix), matching the vocabulary the restart phase
    already uses for ``restarted_services``.
    """
    skipped = {str(name).removesuffix(".service") for name in skip_units}
    # base unit name -> replaced?  A unit name can exist in BOTH the user and
    # the system scope; each is a separate process and each must be proven.
    # Worst outcome wins, so one unprovable scope cannot be masked by the
    # other reporting success.
    outcomes: dict[str, bool] = {}
    seen: set[tuple[str, str]] = set()
    for scope in _systemctl_scopes():
        scope_key = " ".join(scope)
        for unit in _listed_serve_units(scope, run=run):
            base = unit.removesuffix(".service")
            if (scope_key, base) in seen or base in skipped:
                continue
            seen.add((scope_key, base))
            if not _unit_is_active(scope, unit, run=run):
                # Not running: nothing is serving a stale generation from it.
                continue
            previous_pid = _unit_main_pid(scope, unit, run=run)
            if previous_pid <= 0:
                # Active with no readable main process: a replacement cannot
                # be observed, so it cannot be claimed. Restarting blind and
                # reporting success is the failure mode this module exists to
                # remove.
                outcomes[base] = False
                continue
            try:
                result = run(
                    scope + ["--no-ask-password", "restart", unit],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=_UNIT_RESTART_TIMEOUT,
                )
            except (OSError, subprocess.TimeoutExpired):
                outcomes[base] = False
                continue
            if getattr(result, "returncode", 1) != 0:
                # Includes the unprivileged system-scope case. We do not probe
                # for sudo here: an unverifiable unit must read as failed so
                # the update stays explicitly incomplete.
                outcomes[base] = False
                continue
            replaced = _serve_unit_replaced(
                scope, unit, previous_pid, run=run, sleep=sleep
            )
            outcomes[base] = outcomes.get(base, True) and replaced
    return {
        "verified": sorted(base for base, ok in outcomes.items() if ok),
        "failed": sorted(base for base, ok in outcomes.items() if not ok),
    }


def _parse_payload(stream) -> tuple[list[str], dict[str, str], bool, list[str]]:
    payload = json.load(stream)
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, list):
        raise ValueError("recovery payload must contain a profiles list")
    if any(
        not isinstance(profile, str) or not _PROFILE_ID_RE.fullmatch(profile)
        for profile in profiles
    ):
        raise ValueError("recovery profiles contain an invalid profile id")
    raw_supervisors = payload.get("supervisors") if isinstance(payload, dict) else None
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
    raw_serve = payload.get("serve_units") if isinstance(payload, dict) else None
    recover_serve = False
    skip_units: list[str] = []
    if raw_serve is not None:
        if not isinstance(raw_serve, dict):
            raise ValueError("recovery serve_units block is invalid")
        recover_serve = bool(raw_serve.get("recover"))
        raw_skip = raw_serve.get("skip") or []
        if not isinstance(raw_skip, list) or any(
            not isinstance(unit, str) for unit in raw_skip
        ):
            raise ValueError("recovery serve_units skip list is invalid")
        # Only the shapes systemd can actually produce for this family; a
        # skip entry is a name filter, never a command argument.
        skip_units = [
            unit
            for unit in raw_skip
            if _UNIT_RE.fullmatch(unit if unit.endswith(".service") else f"{unit}.service")
        ]
    return profiles, supervisors, recover_serve, skip_units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdin",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if not args.stdin:
        parser.error("this command is an internal update-recovery entry point")

    try:
        profiles, supervisors, recover_serve, skip_units = _parse_payload(sys.stdin)
        result = restart_profiles(profiles, supervisors=supervisors)
        result["serve_units"] = (
            restart_serve_units(skip_units=skip_units)
            if recover_serve
            else {"verified": [], "failed": []}
        )
    except (ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "verified": [],
                    "relaunch_attempted": [],
                    "failed": [],
                    "serve_units": {"verified": [], "failed": []},
                }
            )
        )
        return 2

    print(json.dumps(result, sort_keys=True))
    if result["failed"] or result["serve_units"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
