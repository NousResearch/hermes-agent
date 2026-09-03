"""Fresh-process recovery after the update's in-process restart phase aborts.

``hermes update`` performs its fleet restart in the interpreter that started
before ``git pull``.  When that phase raises — the module graph it is running
no longer matches the checkout on disk — this module owns everything that
happens next: which runtimes a clean child may relaunch, what counts as proof
that a relaunch actually replaced the old generation, which pre-update
processes are still alive, and whether any of that adds up to a recovery that
may call itself complete (#92145).

It is deliberately a separate owner from ``update_cmd``: the abort path has
its own vocabulary (``verified`` / ``relaunch_attempted`` / ``failed``, serve
units, survivors) and its own fail-closed contract, and the update monolith
must not grow another authority surface (review on #96235).  The child process
itself lives in :mod:`hermes_cli.update_restart_recovery`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def _serve_unit_recovery_available() -> bool:
    """Can a fresh process restart ``hermes-serve*`` units on this host?"""
    return sys.platform == "linux" and bool(shutil.which("systemctl"))


def _surviving_pre_update_serve_runtimes(plan) -> list[dict]:
    """Pre-update serve/dashboard runtimes that are STILL the same process.

    Identity is the process incarnation ``(pid, create_time)``, never the PID
    alone. ``ledger_entries()`` re-verifies that pair on every read and prunes
    anything that is gone, so a live entry is a real process — but the numeric
    PID can be reused, and a serve that was restarted correctly can come back
    on the same number. Comparing PIDs alone would then report the successor
    as the pre-update survivor and keep the update incomplete forever
    (#92145 review).

    Anything still here after the recovery pass is a live runtime on the
    pre-update code generation, which is precisely the unsafe state #92145
    reports. Fail closed on missing evidence: an unreadable ledger, or an
    incarnation neither side can produce, counts the runtime as surviving.
    """
    planned: dict[int, dict] = {}
    try:
        for runtime in getattr(plan, "runtimes", ()) or ():
            if getattr(runtime, "kind", None) not in ("serve", "dashboard"):
                continue
            pid = getattr(runtime, "pid", None)
            if not isinstance(pid, int) or pid <= 0:
                continue
            detail = getattr(runtime, "detail", None)
            created = detail.get("create_time") if isinstance(detail, dict) else None
            planned[pid] = {
                "pid": pid,
                "kind": str(getattr(runtime, "kind", "")),
                "profile": str(getattr(runtime, "profile", "")),
                "supervisor": str(getattr(runtime, "supervisor", "")),
                "_create_time": created if isinstance(created, (int, float)) else None,
            }
    except Exception as exc:
        logger.debug("Could not read planned serve runtimes: %s", exc)
        return []
    if not planned:
        return []
    try:
        from hermes_cli.process_identity import ledger_entries

        live: dict[int, float | None] = {}
        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            pid = entry.get("pid")
            if not isinstance(pid, int):
                continue
            created = entry.get("create_time")
            live[pid] = created if isinstance(created, (int, float)) else None
    except Exception as exc:
        logger.debug("Serve/dashboard survivor probe failed: %s", exc)
        return sorted(
            (_without_incarnation(row) for row in planned.values()),
            key=lambda row: row["pid"],
        )
    survivors = []
    for pid, row in planned.items():
        if pid not in live:
            continue
        planned_created = row["_create_time"]
        live_created = live[pid]
        if (
            planned_created is not None
            and live_created is not None
            and abs(float(live_created) - float(planned_created)) >= 2.0
        ):
            # Same number, different process: the pre-update runtime is gone
            # and something new registered under its PID. Not a survivor.
            continue
        survivors.append(_without_incarnation(row))
    return sorted(survivors, key=lambda row: row["pid"])


def _without_incarnation(row: dict) -> dict:
    """The operator-facing survivor row (incarnation is a matching key only)."""
    return {key: value for key, value in row.items() if key != "_create_time"}


def _qualified_serve_skips(skip_units) -> list[dict]:
    """Scope-qualify the units the aborted phase already settled.

    ``restarted_scoped_units`` records ``<scope>/<unit>`` because
    ``hermes-serve.service`` can exist in BOTH the user and the system
    manager, and those are two different processes. Handing the fresh child a
    bare unit name would make one settled scope suppress recovery of the other
    — the stale one would never be restarted and nothing downstream would say
    so. Entries that carry no scope (a payload built by a pre-update
    interpreter) are forwarded without one and read as scope-agnostic there.
    """
    rows: list[dict] = []
    for entry in sorted(skip_units or ()):
        scope, sep, unit = str(entry).partition("/")
        if sep and scope in ("user", "system") and unit:
            rows.append({"scope": scope, "unit": unit})
        elif entry:
            rows.append({"unit": str(entry)})
    return rows


def _recover_gateway_restart_after_abort(
    plan,
    *,
    gateway_mode: bool,
    skip_profiles: set[str] | None = None,
    skip_units: set[str] | None = None,
) -> dict[str, list]:
    """Retry supervised gateway restarts from a clean Python process.

    ``hermes update`` normally performs the fleet restart in the interpreter
    that started before ``git pull``.  If that phase raises while importing the
    new tree, a warning alone leaves the old gateway alive against new files on
    disk.  The recovery boundary launches the existing per-profile
    ``gateway restart`` command through a new interpreter, preserving its
    platform-specific drain and service-manager logic without inheriting the
    stale ``sys.modules`` graph.

    Only profiles classified as supervisor-owned by the pre-update inventory
    are handed off.  A manual gateway must remain running and be reported for
    explicit operator action rather than being killed without a relaunch
    authority; serve/dashboard runtimes from the spawn ledger are likewise
    recorded as skipped with a reason instead of vanishing from the pass.
    The returned protocol is persisted in the update receipt so operators can
    distinguish a spawn failure from a per-profile failure.

    The same child additionally restarts active ``hermes-serve*`` systemd
    units (#92145).  ``hermes serve`` hosts ``tui_gateway.server`` and is
    restarted by the in-process phase alongside the gateway units, but no
    per-profile ``gateway restart`` command reaches it — so an abort used to
    leave it holding the pre-pull module graph with nothing left to notice.

    ``skip_units`` names the units the aborted phase already settled, as
    ``<scope>/<unit>``.  The scope is part of the identity, not decoration:
    ``hermes-serve.service`` can exist in both the user and the system
    manager as two different processes, and an unqualified token would let a
    settled one suppress recovery of a stale one (review on #96235).

    Outcome honesty: ``verified`` means the fresh child independently observed
    the profile's systemd unit active after the relaunch.  A zero exit from
    ``gateway restart`` alone is NOT observed proof that the new code
    generation is serving, so those outcomes are reported as
    ``relaunch_attempted`` and never claim supervisor coverage.
    """
    from hermes_cli.update_cmd import _gateway_recovery_partition

    candidates, skipped = _gateway_recovery_partition(
        plan, skip_profiles=skip_profiles
    )
    profiles = sorted(candidates)
    recover_serve = _serve_unit_recovery_available()
    _empty_serve: dict[str, list] = {"verified": [], "failed": []}
    if not profiles and not recover_serve:
        return {
            "requested": [],
            "verified": [],
            "relaunch_attempted": [],
            "failed": [],
            "skipped": skipped,
            "serve_units": dict(_empty_serve),
        }

    def _all_failed() -> dict[str, list]:
        return {
            "requested": profiles,
            "verified": [],
            "relaunch_attempted": [],
            "failed": profiles,
            "skipped": skipped,
            "serve_units": dict(_empty_serve),
        }

    command = [
        sys.executable,
        "-m",
        "hermes_cli.update_restart_recovery",
        "--stdin",
    ]
    env = os.environ.copy()
    env["HERMES_UPDATE_RESTART_RECOVERY"] = "1"
    for marker in ("_HERMES_GATEWAY", "HERMES_GATEWAY", "HERMES_GATEWAY_MODE"):
        env.pop(marker, None)

    # A gateway-triggered update may run inside the gateway's systemd cgroup.
    # Put the recovery process in a transient user scope before it asks systemd
    # to restart that gateway, otherwise KillMode can terminate the recovery
    # process together with the old service. If systemd-run is unavailable,
    # fail closed rather than pretending the in-cgroup child is independent.
    if gateway_mode and sys.platform == "linux":
        systemd_run = shutil.which("systemd-run")
        if not systemd_run:
            logger.warning("Cannot isolate fresh gateway recovery from the gateway cgroup")
            return _all_failed()
        command = [
            systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--",
            *command,
        ]

    kwargs = {
        "input": json.dumps(
            {
                "profiles": profiles,
                "supervisors": candidates,
                "serve_units": {
                    "recover": recover_serve,
                    "skip": _qualified_serve_skips(skip_units),
                },
            }
        ),
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
        "env": env,
        # Gateway profiles run sequentially at up to 90s each; the serve pass
        # adds its own restart + settle budget on top, so give the child room
        # for both rather than killing a recovery that was working.
        "timeout": max(180, 30 + 90 * len(profiles) + (150 if recover_serve else 0)),
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True

    try:
        result = subprocess.run(command, **kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Fresh gateway restart recovery failed: %s", exc)
        return _all_failed()

    if result.returncode != 0:
        logger.warning("Fresh gateway restart recovery exited %s", result.returncode)
        return _all_failed()

    try:
        recovery_result = json.loads(result.stdout or "")
        verified = recovery_result.get("verified")
        relaunch_attempted = recovery_result.get("relaunch_attempted")
        failed = recovery_result.get("failed")
        raw_serve = recovery_result.get("serve_units") or dict(_empty_serve)
    except (AttributeError, TypeError, ValueError):
        logger.warning("Fresh gateway restart recovery returned invalid JSON")
        return _all_failed()

    serve_units = dict(_empty_serve)
    if (
        isinstance(raw_serve, dict)
        and isinstance(raw_serve.get("verified"), list)
        and isinstance(raw_serve.get("failed"), list)
        and all(
            isinstance(unit, str)
            for unit in (*raw_serve["verified"], *raw_serve["failed"])
        )
    ):
        serve_units = {
            "verified": sorted(raw_serve["verified"]),
            "failed": sorted(raw_serve["failed"]),
        }
    elif recover_serve:
        # An unreadable serve block cannot be read as "nothing to do": the
        # units it describes are exactly the ones that host tui_gateway and
        # may still be serving the pre-update generation.
        logger.warning("Fresh recovery returned an invalid serve-unit result")
        serve_units = {"verified": [], "failed": ["<unreadable>"]}

    buckets = (verified, relaunch_attempted, failed)
    reported: list[str] = []
    if all(isinstance(bucket, list) for bucket in buckets):
        reported = [*verified, *relaunch_attempted, *failed]
    if (
        not all(isinstance(bucket, list) for bucket in buckets)
        or any(not isinstance(profile, str) for profile in reported)
        or set(reported) != set(profiles)
        or len(reported) != len(set(reported))
    ):
        logger.warning("Fresh gateway restart recovery returned incomplete profiles")
        return _all_failed()

    if verified:
        print(
            "  ✓ Restarted supervised gateway(s) in a fresh process"
            " (systemd-verified active): " + ", ".join(sorted(verified))
        )
    if relaunch_attempted:
        print(
            "  ⚠ Relaunch attempted in a fresh process but not"
            " supervisor-verified (check these gateways manually): "
            + ", ".join(sorted(relaunch_attempted))
        )
    if serve_units["verified"]:
        print(
            "  ✓ Restarted serve unit(s) in a fresh process"
            " (new main PID observed): "
            + ", ".join(serve_units["verified"])
        )
    if serve_units["failed"]:
        print(
            "  ⚠ Could not verify a replacement for serve unit(s): "
            + ", ".join(serve_units["failed"])
        )
    return {
        "requested": profiles,
        "verified": sorted(verified),
        "relaunch_attempted": sorted(relaunch_attempted),
        "failed": sorted(failed),
        "skipped": skipped,
        "serve_units": serve_units,
    }


def _warn_stale_serve_runtimes(rows) -> None:
    """Name the serve/dashboard processes that survived on pre-update code.

    The original #92145 report is a user watching every chat turn fail with an
    ``ImportError`` for a symbol that imports fine on disk, with nothing in the
    terminal naming the responsible process. ``hermes serve`` hosts
    ``tui_gateway.server``; when its unit was never restarted it keeps the
    pre-pull ``sys.modules`` graph and there is no gateway row anywhere that
    reveals it. Print the PIDs and the exact command that fixes it.
    """
    if not rows:
        return
    print(
        "  ⚠ These serve/dashboard processes still run pre-update code"
        " (they started before the checkout changed):"
    )
    for row in rows:
        supervisor = row.get("supervisor") or "unknown"
        print(
            f"      pid {row.get('pid')} — {row.get('kind')}"
            f" (profile {row.get('profile') or 'default'}, {supervisor})"
        )
    print(
        "    Restart them before using Hermes again, e.g."
        " `systemctl --user restart hermes-serve.service`"
        " or by relaunching `hermes serve` / the Desktop app."
    )


def _abort_recovery_is_complete(
    *,
    planned_gateway_profiles,
    covered_gateway_profiles,
    recovery_result,
    stale_runtime_rows,
) -> bool:
    """May a fresh-process recovery clear the incomplete flag?

    Only when EVERY inventoried runtime family is accounted for. The gateway
    leg alone is not enough (#92145): the post-update read-back
    (``collect_fleet_versions``) is gateway-only — it reads each profile's
    ``gateway_state.json`` / control socket — so a ``hermes serve`` still
    holding the pre-update ``sys.modules`` graph is invisible to both the
    recovery pass and the verification that follows it. Clearing the flag on
    gateway coverage alone is exactly how an update reported success while
    every chat turn kept failing with an ``ImportError`` for a symbol that
    imports fine on disk.

    Fail closed on each leg:

    * every planned gateway profile is covered, with nothing failed and
      nothing merely ``relaunch_attempted`` (rc 0 is not observed coverage);
    * no ``hermes-serve*`` unit failed to produce a verified replacement; and
    * no serve/dashboard process from the pre-update inventory is still the
      same process (``stale_runtime_rows``).

    ``planned_gateway_profiles`` being empty is deliberately NOT completeness:
    with no gateway leg to prove, the caller's own fail-closed contract
    (``_restart_phase_failure_is_incomplete``) plus the stale-runtime rows
    decide the outcome.
    """
    if not planned_gateway_profiles:
        return False
    if not set(planned_gateway_profiles) <= set(covered_gateway_profiles):
        return False
    result = recovery_result or {}
    if result.get("failed") or result.get("relaunch_attempted"):
        return False
    if (result.get("serve_units") or {}).get("failed"):
        return False
    return not stale_runtime_rows
