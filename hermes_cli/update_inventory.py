"""Runtime inventory + update plan for the fleet-update pipeline (#91277 Phase 2).

One read-only pass that answers, BEFORE any mutation: what Hermes runtimes
are running on this machine, how is each one deployed, which of them will
this update touch, and how will each be restarted?

This is the "plan" phase of the transactional deployment model (#88683):

    plan → snapshot → apply → restart-per-kind → verify → report

The module is deliberately side-effect free — every collector is a probe
over primitives that already exist (`find_profile_gateway_processes`,
`_get_service_pids`, `gateway_state.json` code stamps from #91283,
`detect_install_method`) — so `hermes update --plan` can run on a live
fleet with zero risk, and the update receipt can embed the inventory
without changing update behavior.

Deployment kinds (the concept most fleet-update bugs were missing):

    git      — source checkout; updatable in place via `hermes update`
    docker   — published image; NOT updatable in place (pull + recreate)
    nix/apt  — package-manager owned; updatable via the manager only
    unknown  — no marker; treated as in-place updatable (legacy default)

Supervisors (how a runtime is restarted after code changes):

    systemd / launchd — restart via the service manager (fleet-wide)
    desktop           — Desktop app supervises `hermes serve`; it respawns
    manual            — plain process; SIGTERM + watcher/manual relaunch
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeRecord:
    """One running (or expected) Hermes runtime on this machine."""

    kind: str                     # gateway | dashboard | serve
    profile: str                  # profile name ("default", ...)
    pid: Optional[int] = None     # live PID when known
    supervisor: str = "manual"    # systemd | launchd | desktop | manual
    code_sha: Optional[str] = None       # stamped running-code sha (#91283)
    code_version: Optional[str] = None
    restart_via: str = ""         # human-readable restart mechanism
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdatePlan:
    """The full pre-update picture: install shape + runtimes + actions."""

    install_method: str = "unknown"       # git | docker | nix | apt | ...
    updatable_in_place: bool = True
    update_mechanism: str = "hermes update"
    expected_sha: Optional[str] = None    # current checkout HEAD (pre-pull)
    expected_version: Optional[str] = None
    profiles: list = field(default_factory=list)
    runtimes: list = field(default_factory=list)  # list[RuntimeRecord]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtimes"] = [
            r.to_dict() if isinstance(r, RuntimeRecord) else r
            for r in self.runtimes
        ]
        return payload


def _detect_supervisor_for_pid(pid: int, service_pids: set) -> str:
    """Classify how a live gateway PID is supervised."""
    if pid in service_pids:
        try:
            from hermes_cli.gateway import is_macos, supports_systemd_services

            if supports_systemd_services():
                return "systemd"
            if is_macos():
                return "launchd"
        except Exception:
            pass
        return "service"
    return "manual"


def _restart_mechanism(supervisor: str, profile: str) -> str:
    """Machine-readable restart mechanism id for a runtime.

    THE policy table (#91277 Phase 2): restart execution consumes these ids
    via :func:`match_runtime_outcomes` / the update's restart phase, and the
    receipt records per-runtime outcomes against them. Display strings are
    derived by :func:`describe_restart_mechanism` — never the other way
    around.
    """
    if supervisor == "systemd":
        return "systemd"
    if supervisor == "launchd":
        return "launchd"
    if supervisor == "desktop":
        return "desktop"
    if supervisor == "manual-serve":
        return "respawn-argv"
    return "manual"


def describe_restart_mechanism(mechanism: str, profile: str) -> str:
    """Human-readable description of a restart mechanism id."""
    if mechanism == "systemd":
        return "systemctl restart (drain-first SIGUSR1 when supported)"
    if mechanism == "launchd":
        return "launchctl kickstart -k (drain-first, per-label domain)"
    if mechanism == "desktop":
        return "Desktop app respawns its serve backend"
    if mechanism == "respawn-argv":
        return "stop before code swap, relaunch with recorded launch args"
    if profile != "default":
        return f"hermes -p {profile} gateway restart"
    return "hermes gateway restart"


def collect_runtime_inventory() -> UpdatePlan:
    """Build the pre-update plan. Read-only; never raises.

    Every collector degrades independently — a probe failure yields fewer
    rows, not an exception. The result is embeddable in the update receipt
    and printable via :func:`print_update_plan`.
    """
    plan = UpdatePlan()

    # --- install shape / deployment kind ---------------------------------
    try:
        from hermes_cli.config import (
            detect_install_method,
            get_managed_system,
            recommended_update_command_for_method,
        )

        method = detect_install_method()
        plan.install_method = method
        managed = get_managed_system()
        if managed:
            plan.install_method = managed
        plan.updatable_in_place = method in ("git", "unknown") and not managed
        plan.update_mechanism = recommended_update_command_for_method(method)
    except Exception as exc:
        logger.debug("Install-method probe failed: %s", exc)

    # --- expected code identity (pre-pull) --------------------------------
    try:
        from hermes_cli.build_info import get_code_identity

        identity = get_code_identity(refresh=True)
        plan.expected_sha = identity.get("sha")
        plan.expected_version = identity.get("version")
    except Exception as exc:
        logger.debug("Code-identity probe failed: %s", exc)

    # --- profiles ----------------------------------------------------------
    profile_homes: list[tuple[str, Path]] = []
    try:
        from hermes_cli.profiles import (
            _get_default_hermes_home,
            _get_profiles_root,
            _PROFILE_ID_RE,
        )

        default_home = _get_default_hermes_home()
        if default_home.is_dir():
            profile_homes.append(("default", default_home))
        root = _get_profiles_root()
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if (
                    entry.is_dir()
                    and entry.name != "default"
                    and _PROFILE_ID_RE.match(entry.name)
                ):
                    profile_homes.append((entry.name, entry))
        plan.profiles = [name for name, _ in profile_homes]
    except Exception as exc:
        logger.debug("Profile enumeration failed: %s", exc)

    # --- service-managed PIDs (fleet-wide) ---------------------------------
    service_pids: set = set()
    try:
        from hermes_cli.gateway import _get_service_pids

        service_pids = _get_service_pids(all_profiles=True) or set()
    except Exception as exc:
        logger.debug("Service-PID probe failed: %s", exc)

    # --- per-profile gateways (PID files + runtime status stamps) ----------
    seen_pids: set[int] = set()
    try:
        from gateway.status import _pid_exists, read_runtime_status

        for profile, home in profile_homes:
            # Prefer the gateway-owned control socket (#92091): identity
            # declared by the process itself, including its own supervisor
            # provenance — no argv/PID inference. Scan fallback below.
            identity = None
            try:
                from gateway.control_socket import identify_gateway

                identity = identify_gateway(home)
            except Exception:
                identity = None
            if identity:
                try:
                    sock_pid = int(identity.get("pid"))
                except (TypeError, ValueError):
                    sock_pid = None
                if sock_pid is not None:
                    if sock_pid in seen_pids:
                        # One multiplex gateway can answer identify for
                        # several profile homes — one runtime record per
                        # process, not per home.
                        continue
                    seen_pids.add(sock_pid)
                    declared = identity.get("supervisor")
                    supervisor = (
                        str(declared)
                        if declared
                        else _detect_supervisor_for_pid(sock_pid, service_pids)
                    )
                    sock_sha = identity.get("code_sha")
                    plan.runtimes.append(
                        RuntimeRecord(
                            kind="gateway",
                            profile=profile,
                            pid=sock_pid,
                            supervisor=supervisor,
                            code_sha=str(sock_sha) if sock_sha else None,
                            code_version=identity.get("code_version"),
                            restart_via=_restart_mechanism(supervisor, profile),
                        )
                    )
                    continue
            record = read_runtime_status(home / "gateway_state.json")
            pid: Optional[int] = None
            code_sha = code_version = None
            if record:
                try:
                    pid = int(record.get("pid"))
                except (TypeError, ValueError):
                    pid = None
                code_sha = record.get("code_sha")
                code_version = record.get("code_version")
            if pid is None or not _pid_exists(pid):
                continue
            seen_pids.add(pid)
            supervisor = _detect_supervisor_for_pid(pid, service_pids)
            plan.runtimes.append(
                RuntimeRecord(
                    kind="gateway",
                    profile=profile,
                    pid=pid,
                    supervisor=supervisor,
                    code_sha=str(code_sha) if code_sha else None,
                    code_version=code_version,
                    restart_via=_restart_mechanism(supervisor, profile),
                )
            )
    except Exception as exc:
        logger.debug("Gateway-state inventory failed: %s", exc)

    # PID-file mapped gateways not covered by a runtime-status record
    try:
        from hermes_cli.gateway import find_profile_gateway_processes

        for proc in find_profile_gateway_processes():
            if proc.pid in seen_pids:
                continue
            seen_pids.add(proc.pid)
            supervisor = _detect_supervisor_for_pid(proc.pid, service_pids)
            plan.runtimes.append(
                RuntimeRecord(
                    kind="gateway",
                    profile=proc.profile,
                    pid=proc.pid,
                    supervisor=supervisor,
                    restart_via=_restart_mechanism(supervisor, proc.profile),
                )
            )
    except Exception as exc:
        logger.debug("PID-file gateway inventory failed: %s", exc)

    # Serve/dashboard backends from the spawn ledger (#63206). These are the
    # runtimes the gateway collectors above can never see: a manually
    # launched `hermes serve --host <ip>` for a remote Desktop, or a
    # long-lived `hermes dashboard`. Every serve/dashboard registers itself
    # (with structured host/port/profile since #63206) at startup, and
    # ledger_entries() live-verifies (pid, create_time) so PID reuse never
    # fabricates a row. Desktop-supervised backends are classified by their
    # recorded spawner still being alive — those restart via the Desktop's
    # own respawn, not ours.
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            purpose = entry.get("purpose")
            if purpose not in ("serve", "dashboard"):
                continue
            pid = entry.get("pid")
            if not isinstance(pid, int) or pid in seen_pids:
                continue
            seen_pids.add(pid)
            has_live_spawner = spawner_is_dead(entry) is False
            supervisor = "desktop" if has_live_spawner else "manual-serve"
            profile = str(entry.get("profile") or "default")
            plan.runtimes.append(
                RuntimeRecord(
                    kind=str(purpose),
                    profile=profile,
                    pid=pid,
                    supervisor=supervisor,
                    restart_via=_restart_mechanism(supervisor, profile),
                    detail={
                        "argv": entry.get("argv") or "",
                        "host": entry.get("host") or "",
                        "port": entry.get("port"),
                    },
                )
            )
    except Exception as exc:
        logger.debug("Serve/dashboard ledger inventory failed: %s", exc)

    return plan


def print_update_plan(plan: UpdatePlan) -> None:
    """Human-readable plan — what the update will touch and how."""
    print("Update plan:")
    print(f"  Install: {plan.install_method}", end="")
    if plan.expected_version:
        print(f" (v{plan.expected_version}", end="")
        if plan.expected_sha:
            print(f" @ {plan.expected_sha[:8]}", end="")
        print(")", end="")
    print()
    if not plan.updatable_in_place:
        print("  ⚠ This install is NOT updatable in place.")
        print(f"    Update via: {plan.update_mechanism}")
    profiles = ", ".join(plan.profiles) if plan.profiles else "(none found)"
    print(f"  Profiles: {profiles}")
    if not plan.runtimes:
        print("  Running Hermes services: none detected — code swap only.")
        return
    print(f"  Running services to restart ({len(plan.runtimes)}):")
    for runtime in plan.runtimes:
        sha = f" @ {runtime.code_sha[:8]}" if runtime.code_sha else ""
        print(
            f"    • {runtime.kind} [{runtime.profile}] pid {runtime.pid}"
            f" — {runtime.supervisor}{sha}"
        )
        print(
            "      restart: "
            f"{describe_restart_mechanism(runtime.restart_via, runtime.profile)}"
        )


def match_runtime_outcomes(
    plan: "UpdatePlan",
    *,
    restarted_services: list,
    relaunched_profiles: list,
    externally_supervised_profiles: list,
    killed_pids: set,
    failed_units: list,
) -> list[dict[str, Any]]:
    """Reconcile the plan's runtimes against what the restart phase DID.

    #91277 Phase 2 (restart via declared mechanism): the platform restart
    branches each re-discover their own targets, so a runtime the plan saw
    can be missed entirely with no signal. This cross-checks every planned
    runtime against the phase's bookkeeping and returns one outcome row per
    runtime::

        {"kind", "profile", "pid", "mechanism", "outcome"}

    outcome: ``restarted`` (service restarted / profile relaunched /
    handed to external supervisor), ``stopped`` (pid killed, watcher or
    operator relaunches), ``failed`` (in the phase's failed/stale list) or
    ``unaccounted`` — the plan saw it and NO bookkeeping mentions it: the
    blind-spot tripwire (same philosophy as the fleet matrix's DOWN row).
    Never raises; on any probe error returns what it has.
    """
    outcomes: list[dict[str, Any]] = []
    try:
        failed_set = {str(u) for u in (failed_units or [])}
        restarted_set = {str(s) for s in (restarted_services or [])}
        relaunched = set(relaunched_profiles or [])
        external = set(externally_supervised_profiles or [])
        killed = {int(p) for p in (killed_pids or set())}

        for runtime in plan.runtimes:
            r = runtime if isinstance(runtime, RuntimeRecord) else None
            if r is None:
                continue
            outcome = "unaccounted"
            if r.profile in relaunched or r.profile in external:
                outcome = "restarted"
            elif r.pid is not None and r.pid in killed:
                outcome = "stopped"
            elif any(
                r.profile in unit or (r.profile == "default" and "hermes-gateway" in unit)
                for unit in failed_set
            ):
                outcome = "failed"
            elif any(
                r.profile in svc or (r.profile == "default" and "hermes-gateway" in svc)
                for svc in restarted_set
            ):
                outcome = "restarted"
            outcomes.append(
                {
                    "kind": r.kind,
                    "profile": r.profile,
                    "pid": r.pid,
                    "mechanism": r.restart_via,
                    "outcome": outcome,
                }
            )
    except Exception as exc:
        logger.debug("Runtime-outcome reconciliation failed: %s", exc)
    return outcomes


def report_unaccounted_runtimes(outcomes: list[dict[str, Any]]) -> bool:
    """Print a loud warning for runtimes the restart phase never touched.

    Returns True when at least one planned runtime is unaccounted — the
    caller escalates exactly like a STALE/DOWN fleet row (exit 1): a runtime
    the plan promised to restart, silently missed, is the class this phase
    exists to kill.
    """
    missed = [o for o in outcomes if o.get("outcome") == "unaccounted"]
    if not missed:
        return False
    print()
    print("  ⚠ Planned runtimes the restart phase never touched:")
    for o in missed:
        print(
            f"    ✗ {o['kind']} [{o['profile']}] pid {o['pid']}"
            f" — planned mechanism: {o['mechanism']}"
        )
    print("    Restart them manually, then verify:")
    print("      hermes gateway restart                # active profile")
    print("      hermes -p <profile> gateway restart   # named profile")
    return True


def record_plan_in_receipt(plan: UpdatePlan) -> None:
    """Attach the inventory to the active update receipt. Never raises."""
    try:
        import hermes_cli.update_receipt as ur

        if ur._current is not None:
            ur._current.data["plan"] = plan.to_dict()
    except Exception as exc:
        logger.debug("Could not record plan in receipt: %s", exc)
