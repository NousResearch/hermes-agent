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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from typing import Any

_RECOVERY_ENV = "HERMES_UPDATE_RESTART_RECOVERY"
_GATEWAY_MARKERS = ("_HERMES_GATEWAY", "HERMES_GATEWAY", "HERMES_GATEWAY_MODE")
_PROFILE_RESTART_TIMEOUT = 90
_VERIFY_TIMEOUT = 15
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SUPERVISOR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


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


def _parse_payload(stream) -> tuple[list[str], dict[str, str]]:
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
    return profiles, supervisors


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
        profiles, supervisors = _parse_payload(sys.stdin)
        result = restart_profiles(profiles, supervisors=supervisors)
    except (ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "verified": [],
                    "relaunch_attempted": [],
                    "failed": [],
                }
            )
        )
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
