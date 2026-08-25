"""Restart supervised gateway profiles from a clean Python generation.

The normal update command keeps executing in the interpreter that started before
``git pull``.  This module is deliberately small: it imports no gateway code
itself and launches the regular per-profile gateway command in a new
interpreter.  It is used only after the in-process restart phase has raised, so
that the recovery path cannot inherit the stale ``sys.modules`` graph that
caused the failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from typing import Any

_RECOVERY_ENV = "HERMES_UPDATE_RESTART_RECOVERY"
_GATEWAY_MARKERS = ("_HERMES_GATEWAY", "HERMES_GATEWAY", "HERMES_GATEWAY_MODE")
_PROFILE_RESTART_TIMEOUT = 90
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


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


def restart_profiles(
    profiles: Iterable[str], *, run: Callable[..., Any] = subprocess.run
) -> dict[str, list[str]]:
    """Restart the supplied profiles and return per-profile terminal results.

    The caller supplies only profiles whose inventory identified a service
    supervisor.  Manual gateways are intentionally excluded before this module
    is called: killing one without a relaunch authority would turn stale code
    into an outage.
    """
    normalized = sorted(
        {profile for profile in profiles if isinstance(profile, str) and profile}
    )
    succeeded: list[str] = []
    failed: list[str] = []
    for profile in normalized:
        if _run_profile_restart(profile, run=run):
            succeeded.append(profile)
        else:
            failed.append(profile)
    return {"succeeded": succeeded, "failed": failed}


def _parse_payload(stream) -> list[str]:
    payload = json.load(stream)
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, list):
        raise ValueError("recovery payload must contain a profiles list")
    if any(
        not isinstance(profile, str) or not _PROFILE_ID_RE.fullmatch(profile)
        for profile in profiles
    ):
        raise ValueError("recovery profiles contain an invalid profile id")
    return profiles


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
        profiles = _parse_payload(sys.stdin)
        result = restart_profiles(profiles)
    except (ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc), "succeeded": [], "failed": []}))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
