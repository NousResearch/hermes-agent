"""Runtime inventory + update plan for the fleet-update pipeline.

One read-only pass answering, BEFORE any mutation: which Hermes runtimes run on this machine, how
each is deployed, which ones this update touches, and how each restarts. Every collector is a
side-effect-free probe, so ``hermes update --plan`` is safe on a live fleet.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeRecord:
    """One running (or expected) Hermes runtime on this machine."""

    kind: str                     # gateway | dashboard | serve
    profile: str
    pid: Optional[int] = None
    supervisor: str = "manual"    # systemd | launchd | desktop | windows-service | service | manual | manual-serve
    code_sha: Optional[str] = None       # stamped running-code sha
    # See #91283.
    code_version: Optional[str] = None
    restart_via: str = ""         # mechanism id, see _RESTART_MECHANISMS
    detail: dict = field(default_factory=dict)


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
        return asdict(self)  # recursive: RuntimeRecord entries become dicts


def _detect_supervisor_for_pid(pid: int, service_pids: set, windows_service_pids: set | None = None) -> str:
    """Classify how a live gateway PID is supervised."""
    if windows_service_pids and pid in windows_service_pids:
        # SCM-supervised Windows gateway: the update pause machinery stops the SERVICE via sc.exe
        # instead of killing the child, so reconciliation must plan it under its own mechanism id.
        # See #91277.
        return "windows-service"
    if pid not in service_pids:
        return "manual"
    with suppress(Exception):
        from hermes_cli.gateway import is_macos, supports_systemd_services

        if supports_systemd_services():
            return "systemd"
        if is_macos():
            return "launchd"
    return "service"


# THE restart policy table: restart execution consumes these ids via match_runtime_outcomes / the
# update's restart phase, and the receipt records per-runtime outcomes against them. Display
# strings are derived by describe_restart_mechanism — never the other way around.
_RESTART_MECHANISMS = {
    "systemd": "systemd", "launchd": "launchd", "desktop": "desktop",
    "windows-service": "windows-service", "manual-serve": "respawn-argv",
}

_MECHANISM_DESCRIPTIONS = {
    "systemd": "systemctl restart (drain-first SIGUSR1 when supported)",
    "launchd": "launchctl kickstart -k (drain-first, per-label domain)",
    "desktop": "Desktop app respawns its serve backend",
    "windows-service": "sc.exe stop before venv mutation, sc.exe start after update",
    "respawn-argv": "stop before code swap, relaunch with recorded launch args",
}

_SERVE_KINDS = ("serve", "dashboard")


def _restart_mechanism(supervisor: str, profile: str) -> str:
    """Machine-readable restart mechanism id for a runtime.

    THE policy table (#91277 Phase 2): restart execution consumes these ids via
    :func:`match_runtime_outcomes` / the update's restart phase, and the receipt records per-runtime
    outcomes against them. Display strings are derived by :func:`describe_restart_mechanism` — never the
    other way around.
    """
    return _RESTART_MECHANISMS.get(supervisor, "manual")


def describe_restart_mechanism(mechanism: str, profile: str) -> str:
    """Human-readable description of a restart mechanism id."""
    return _MECHANISM_DESCRIPTIONS.get(mechanism) or (
        f"hermes -p {profile} gateway restart" if profile != "default" else "hermes gateway restart"
    )


def _runtime(
    kind: str, profile: str, pid: Optional[int], supervisor: str,
    code_sha: Any = None, code_version: Any = None, **extra: Any,
) -> RuntimeRecord:
    """A :class:`RuntimeRecord` with ``restart_via`` derived from its supervisor."""
    return RuntimeRecord(
        kind=kind, profile=profile, pid=pid, supervisor=supervisor,
        code_sha=str(code_sha) if code_sha else None, code_version=code_version,
        restart_via=_restart_mechanism(supervisor, profile), **extra,
    )


@contextmanager
def _probe(label: str):
    """Run one inventory collector; a failure is logged at debug and yields fewer rows, never an exception."""
    try:
        yield
    except Exception as exc:
        logger.debug("%s failed: %s", label, exc)


def _collect_install_shape(plan: UpdatePlan) -> None:
    with _probe("Install-method probe"):
        from hermes_cli.config import detect_install_method, get_managed_system, recommended_update_command_for_method

        method = detect_install_method()
        managed = get_managed_system()
        plan.install_method = managed or method
        plan.updatable_in_place = method in ("git", "unknown") and not managed
        # Baked image provenance is authoritative when present: a bind-mounted checkout inside a
        # container can look like `git` while the running filesystem is an immutable image.
        # Fail-closed: an invalid marker still flips the plan to not-updatable.
        with _probe("Image provenance probe"):
            # See #91277.
            from hermes_cli.image_provenance import read_image_provenance

            provenance = read_image_provenance()
            if provenance is not None:
                plan.updatable_in_place = False
                if provenance.valid and provenance.manager:
                    plan.install_method = provenance.manager
        plan.update_mechanism = recommended_update_command_for_method(method)


def _supervisor_classifier() -> Callable[[int], str]:
    """``pid -> supervisor`` over the service-PID sets; each probe degrades to an empty set."""
    service_pids: set = set()
    with _probe("Service-PID probe"):
        from hermes_cli.gateway import _get_service_pids

        service_pids = _get_service_pids(all_profiles=True) or set()
    # Windows SCM services (no-op off Windows): the update's pause phase stops these via `sc.exe
    # stop` / restarts via `sc.exe start`, so the plan must carry the matching mechanism id.
    # --- SCM-supervised gateway PIDs (Windows) ------------------------------
    # find_windows_gateway_services() maps validated gateway PIDs through process ancestry to running SCM
    # service PIDs (no-op off Windows). See #91277.
    windows_service_pids: set = set()
    with _probe("Windows SCM service-ownership probe"):
        from hermes_cli.gateway import find_windows_gateway_services

        windows_service_pids = {int(service.gateway_pid) for service in find_windows_gateway_services()}
    return lambda pid: _detect_supervisor_for_pid(pid, service_pids, windows_service_pids)


def _collect_gateway_runtimes(plan: UpdatePlan, profile_homes: list, seen: set[int]) -> None:
    """Per-profile gateways: control-socket identity first (declared by the process itself, including
    supervisor provenance — no argv/PID inference), ``gateway_state.json`` fallback, then PID-file
    mapped gateways no status record covers."""
    supervisor = _supervisor_classifier()
    with _probe("Gateway-state inventory"):
        from gateway.status import _pid_exists, read_runtime_status
        from hermes_cli.update_receipt import _socket_identity

        for profile, home in profile_homes:
            sock = _socket_identity(home)
            if sock is not None:
                pid, record = sock
                if pid in seen:
                    continue  # one multiplex gateway answers identify for several homes — one record per process
                seen.add(pid)
                declared = record.get("supervisor")
                sup = str(declared) if declared else supervisor(pid)
            else:
                record = read_runtime_status(home / "gateway_state.json") or {}
                try:
                    pid = int(record.get("pid"))
                except (TypeError, ValueError):
                    continue
                if not _pid_exists(pid):
                    continue
                seen.add(pid)
                sup = supervisor(pid)
            plan.runtimes.append(_runtime("gateway", profile, pid, sup, record.get("code_sha"), record.get("code_version")))
    with _probe("PID-file gateway inventory"):
        from hermes_cli.gateway import find_profile_gateway_processes

        for proc in find_profile_gateway_processes():
            if proc.pid not in seen:
                seen.add(proc.pid)
                plan.runtimes.append(_runtime("gateway", proc.profile, proc.pid, supervisor(proc.pid)))


def _collect_ledger_runtimes(plan: UpdatePlan, seen: set[int]) -> None:
    """Serve/dashboard backends from the spawn ledger — runtimes the gateway collectors can never see
    (a manual `hermes serve --host <ip>` for a remote Desktop, a long-lived `hermes dashboard`).
    ledger_entries() live-verifies (pid, create_time) so PID reuse never fabricates a row. Desktop-
    supervised backends (spawner still alive) restart via the Desktop's own respawn, not ours."""
    with _probe("Serve/dashboard ledger inventory"):
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            purpose, pid = entry.get("purpose"), entry.get("pid")
            if purpose not in _SERVE_KINDS or not isinstance(pid, int) or pid in seen:
                continue
            seen.add(pid)
            # detail.create_time: process incarnation, not just the numeric PID — a post-update
            # survivor probe comparing PIDs alone calls a NEW serve that reused the number a survivor.
            plan.runtimes.append(_runtime(
                str(purpose), str(entry.get("profile") or "default"), pid,
                "desktop" if spawner_is_dead(entry) is False else "manual-serve",
                detail={
                    "argv": entry.get("argv") or "", "host": entry.get("host") or "",
                    "port": entry.get("port"), "create_time": entry.get("create_time"),
                },
            ))


def collect_runtime_inventory() -> UpdatePlan:
    """Build the pre-update plan. Read-only; never raises — every collector degrades independently.

    The result is embeddable in the update receipt and printable via :func:`print_update_plan`.
    """
    plan = UpdatePlan()
    _collect_install_shape(plan)
    with _probe("Code-identity probe"):
        from hermes_cli.build_info import get_code_identity

        identity = get_code_identity(refresh=True)
        plan.expected_sha = identity.get("sha")
        plan.expected_version = identity.get("version")
    profile_homes: list = []
    with _probe("Profile enumeration"):
        from hermes_cli.update_receipt import _profile_homes

        profile_homes = _profile_homes()
        plan.profiles = [name for name, _ in profile_homes]
    seen: set[int] = set()
    _collect_gateway_runtimes(plan, profile_homes, seen)
    _collect_ledger_runtimes(plan, seen)
    return plan


def print_update_plan(plan: UpdatePlan) -> None:
    """Human-readable plan — what the update will touch and how."""
    print("Update plan:")
    install = f"  Install: {plan.install_method}"
    if plan.expected_version:
        install += f" (v{plan.expected_version}" + (f" @ {plan.expected_sha[:8]}" if plan.expected_sha else "") + ")"
    print(install)
    if not plan.updatable_in_place:
        print("  ⚠ This install is NOT updatable in place.")
        print(f"    Update via: {plan.update_mechanism}")
    print(f"  Profiles: {', '.join(plan.profiles) if plan.profiles else '(none found)'}")
    if not plan.runtimes:
        print("  Running Hermes services: none detected — code swap only.")
        return
    print(f"  Running services to restart ({len(plan.runtimes)}):")
    for runtime in plan.runtimes:
        sha = f" @ {runtime.code_sha[:8]}" if runtime.code_sha else ""
        print(f"    • {runtime.kind} [{runtime.profile}] pid {runtime.pid} — {runtime.supervisor}{sha}")
        print(f"      restart: {describe_restart_mechanism(runtime.restart_via, runtime.profile)}")


def _serve_unit_matches_profile(profile: str, unit: object) -> bool:
    """Does *unit* name a ``hermes-serve*``/``hermes-dashboard*`` unit for *profile*? (OWN vocabulary;
    the gateway's ``hermes-gateway*`` names never cover serve/dashboard runtimes.)

    Exact names only — ``work`` must not claim ``hermes-serve-workbench`` — and a scope prefix
    (``user/hermes-serve``) is tolerated because the restart phase records scope-qualified identities in
    some lists. See #100479.
    """
    name = str(unit).removesuffix(".service").rsplit("/", 1)[-1]
    suffix = "" if profile == "default" else f"-{profile}"
    return name in {f"hermes-serve{suffix}", f"hermes-dashboard{suffix}"}


def _gateway_service_matches_profile(profile: str, service: object) -> bool:
    """Match an exact gateway service/label (systemd/launchd/s6 shapes) to a profile.

    Never substring-match: ``foo`` must not claim ``hermes-gateway-foobar.service``.
    Launchd labels are ``ai.hermes.gateway`` / ``ai.hermes.gateway-<profile>`` — they do
    not contain the substring ``hermes-gateway``, so a successful macOS kickstart must
    still credit the planned default gateway. A scope prefix (``user/hermes-gateway``,
    ``gui/501/ai.hermes.gateway``) is stripped the same way serve units are.
    """
    name = str(service).removesuffix(".service").rsplit("/", 1)[-1]
    if profile == "default":
        return name in {"hermes-gateway", "ai.hermes.gateway", "gateway", "gateway-default"}
    return name in {f"hermes-gateway-{profile}", f"ai.hermes.gateway-{profile}", f"gateway-{profile}"}


def _gateway_named_in(r: RuntimeRecord, names: set) -> bool:
    # Gateway-only vocabulary: a serve/dashboard that merely shares the profile is a
    # different process. Exact label match (systemd + launchd + s6), not substring.
    return any(_gateway_service_matches_profile(r.profile, name) for name in names)


def match_runtime_outcomes(
    plan: "UpdatePlan", *, restarted_services: list, relaunched_profiles: list,
    externally_supervised_profiles: list, killed_pids: set, failed_units: list,
    stale_serve_pids: "set | None" = None,
) -> list[dict[str, Any]]:
    """Reconcile the plan's runtimes against what the restart phase DID.

    The platform restart branches each re-discover their own targets, so a runtime the plan saw can
    be missed with no signal. Returns one ``{kind, profile, pid, mechanism, outcome}`` row per
    planned runtime; outcome is ``restarted``, ``stopped``, ``failed`` or ``unaccounted`` (no
    bookkeeping mentions it — the blind-spot tripwire). Never raises. Serve/dashboard runtimes are
    reconciled in their OWN vocabulary and never borrow the gateway's outcome: with
    ``stale_serve_pids`` a pre-update serve whose incarnation is gone counts as ``restarted``, one
    still alive is ``unaccounted``; without the probe an untouched serve stays ``unaccounted``.

    See #91277.
    They never borrow the gateway's outcome: ``relaunched_profiles`` and ``hermes-gateway*`` name a
    different process that shares the profile, nothing more. See #100479.
    """
    outcomes: list[dict[str, Any]] = []
    try:
        failed_set = {str(u) for u in (failed_units or [])}
        restarted_set = {str(s) for s in (restarted_services or [])}
        relaunched = set(relaunched_profiles or []) | set(externally_supervised_profiles or [])
        killed = {int(p) for p in (killed_pids or set())}
        stale_serves = {int(p) for p in stale_serve_pids} if stale_serve_pids is not None else None

        def _outcome(r: RuntimeRecord) -> str:
            killed_here = r.pid is not None and r.pid in killed
            if r.kind in _SERVE_KINDS:
                if killed_here:
                    return "stopped"
                if any(_serve_unit_matches_profile(r.profile, u) for u in failed_set):
                    return "failed"
                if stale_serves is not None:
                    # Incarnation-verified: the pre-update process is gone (replaced by its unit / the
                    # dashboard cleanup respawn / the Desktop app) or it is still alive on pre-update code.
                    return "unaccounted" if r.pid in stale_serves else "restarted"
                return "restarted" if any(_serve_unit_matches_profile(r.profile, s) for s in restarted_set) else "unaccounted"
            if r.profile in relaunched:
                return "restarted"
            if killed_here:
                return "stopped"
            if _gateway_named_in(r, failed_set):
                return "failed"
            return "restarted" if _gateway_named_in(r, restarted_set) else "unaccounted"

        for r in plan.runtimes:
            if isinstance(r, RuntimeRecord):
                outcomes.append(
                    {"kind": r.kind, "profile": r.profile, "pid": r.pid, "mechanism": r.restart_via, "outcome": _outcome(r)}
                )
    except Exception as exc:
        logger.debug("Runtime-outcome reconciliation failed: %s", exc)
    return outcomes


def report_unaccounted_runtimes(outcomes: list[dict[str, Any]]) -> bool:
    """Print a loud warning for runtimes the restart phase never touched.

    Returns True when at least one planned runtime is unaccounted; the caller escalates like a
    STALE/DOWN fleet row (exit 1) — a promised restart silently missed is the class this phase
    exists to kill.
    """
    missed = [o for o in outcomes if o.get("outcome") == "unaccounted"]
    if not missed:
        return False
    print()
    print("  ⚠ Planned runtimes the restart phase never touched:")
    for o in missed:
        print(f"    ✗ {o['kind']} [{o['profile']}] pid {o['pid']} — planned mechanism: {o['mechanism']}")
    print("    Restart them manually, then verify:")
    if any(o.get("kind") not in _SERVE_KINDS for o in missed):
        print("      hermes gateway restart                # active profile")
        print("      hermes -p <profile> gateway restart   # named profile")
    if any(o.get("kind") in _SERVE_KINDS for o in missed):
        # A serve/dashboard is not reachable by any `gateway restart` command: name the process, not the wrong verb.
        # See #100479.
        print("      systemctl --user restart hermes-serve.service   # unit-managed serve")
        print("      relaunch `hermes serve` / `hermes dashboard` / the Desktop app")
    return True


def record_plan_in_receipt(plan: UpdatePlan) -> None:
    """Attach the inventory to the active update receipt. Never raises."""
    try:
        import hermes_cli.update_receipt as ur

        if ur._current is not None:
            ur._current.data["plan"] = plan.to_dict()
    except Exception as exc:
        logger.debug("Could not record plan in receipt: %s", exc)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
