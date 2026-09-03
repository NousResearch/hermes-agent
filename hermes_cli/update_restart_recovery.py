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


def _systemctl_scopes() -> list[tuple[str, list[str]]]:
    """``systemctl`` invocations for the user and system scopes, or nothing.

    Mirrors the scope pair the in-process restart phase walks. ``systemctl``
    is resolved through ``shutil.which`` so this module never has to import
    any Hermes platform helper — importing the freshly pulled tree is exactly
    what aborted the phase that called us.

    Each scope is returned with its label because ``hermes-serve.service`` in
    the user manager and ``hermes-serve.service`` in the system manager are
    two different processes. Every identity this module produces or consumes
    stays qualified by that label; the bare unit name is never the key.
    """
    systemctl = shutil.which("systemctl")
    if not systemctl or sys.platform != "linux":
        return []
    return [("user", [systemctl, "--user"]), ("system", [systemctl])]


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


def _qualified(scope_label: str, base: str) -> str:
    """The only identity this module reports for a serve unit."""
    return f"{scope_label}/{base}"


def _normalized_skips(
    skip_units: Iterable[Any],
) -> tuple[set[tuple[str, str]], set[str]]:
    """Split already-settled units into scope-qualified and legacy entries.

    Qualified entries (``{"scope": "user", "unit": "hermes-serve"}`` or the
    equivalent ``"user/hermes-serve"`` string) suppress exactly one process.

    A bare ``"hermes-serve"`` carries no scope and therefore cannot say WHICH
    of two same-named processes was settled. It is honoured across both scopes
    because that is all the information it contains — the caller in this tree
    always sends the qualified shape, and this branch exists only for a payload
    written by a pre-update interpreter that had no scope to send.
    """
    qualified: set[tuple[str, str]] = set()
    legacy: set[str] = set()
    for entry in skip_units or ():
        if isinstance(entry, Mapping):
            scope_label = str(entry.get("scope") or "")
            base = str(entry.get("unit") or "").removesuffix(".service")
        else:
            scope_label, sep, unit = str(entry).partition("/")
            if not sep:
                scope_label, unit = "", scope_label
            base = unit.removesuffix(".service")
        if not base:
            continue
        if scope_label in _SCOPE_LABELS:
            qualified.add((scope_label, base))
        else:
            legacy.add(base)
    return qualified, legacy


def restart_serve_units(
    *,
    skip_units: Iterable[Any] = (),
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

    Every identity in and out of this function is scope-qualified. The user
    manager and the system manager can each own a ``hermes-serve.service``,
    and they are different processes: projecting the scope away would let one
    already-settled unit suppress recovery of the other, and would let one
    scope's success describe the other's outcome.

    Returns ``{"verified": [...], "failed": [...]}`` whose entries are
    ``<scope>/<base unit>`` (no ``.service`` suffix), e.g.
    ``user/hermes-serve``.
    """
    skipped_qualified, skipped_legacy = _normalized_skips(skip_units)
    # (scope, base unit) -> replaced?  A unit name can exist in BOTH the user
    # and the system scope; each is a separate process and each is proven,
    # reported and accounted for on its own.
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
                # Not running: nothing is serving a stale generation from it.
                continue
            previous_pid = _unit_main_pid(scope, unit, run=run)
            if previous_pid <= 0:
                # Active with no readable main process: a replacement cannot
                # be observed, so it cannot be claimed. Restarting blind and
                # reporting success is the failure mode this module exists to
                # remove.
                outcomes[target] = False
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
                outcomes[target] = False
                continue
            if getattr(result, "returncode", 1) != 0:
                # Includes the unprivileged system-scope case. We do not probe
                # for sudo here: an unverifiable unit must read as failed so
                # the update stays explicitly incomplete.
                outcomes[target] = False
                continue
            outcomes[target] = _serve_unit_replaced(
                scope, unit, previous_pid, run=run, sleep=sleep
            )
    return {
        "verified": sorted(
            _qualified(*target) for target, ok in outcomes.items() if ok
        ),
        "failed": sorted(
            _qualified(*target) for target, ok in outcomes.items() if not ok
        ),
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
            not isinstance(entry, (str, dict)) for entry in raw_skip
        ):
            raise ValueError("recovery serve_units skip list is invalid")
        for entry in raw_skip:
            # A skip entry names one already-settled process. The qualified
            # shape carries the systemd scope, because `hermes-serve.service`
            # can exist in BOTH managers and settling one says nothing about
            # the other. A bare string is the legacy shape a pre-update
            # interpreter can still send; it is kept, and read as
            # scope-agnostic by `restart_serve_units`.
            if isinstance(entry, dict):
                scope_label = entry.get("scope")
                unit = entry.get("unit")
                if not isinstance(unit, str):
                    raise ValueError("recovery serve_units skip list is invalid")
                if not isinstance(scope_label, str):
                    scope_label = ""
            else:
                scope_label, _, unit = entry.rpartition("/")
            # Only the shapes systemd can actually produce for this family; a
            # skip entry is a name filter, never a command argument. An
            # unrecognized scope drops the entry rather than raising: dropping
            # a skip can only ever cause one more restart-and-verify, while
            # honouring an unreadable one could suppress recovery of a stale
            # process.
            if scope_label and scope_label not in _SCOPE_LABELS:
                continue
            if not _UNIT_RE.fullmatch(
                unit if unit.endswith(".service") else f"{unit}.service"
            ):
                continue
            base = unit.removesuffix(".service")
            skip_units.append(f"{scope_label}/{base}" if scope_label else base)
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
