"""Gateway supervisor observation, runtime status, safe stop and orphan reaping.

Extracted from hermes_cli.gateway; function bodies preserve the pinned base.
Collaborators are rebound from that facade per call to preserve its patch surface.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _systemd_unit_is_active(system: bool) -> bool:
    """``systemctl is-active`` == "active" for the installed unit in ``system`` scope, else False."""
    from hermes_cli.gateway import (
        _CAPTURE_TEXT,
        _run_systemctl,
        get_service_name,
        get_systemd_unit_path,
        subprocess,
    )
    if not get_systemd_unit_path(system=system).exists():
        return False
    try:
        result = _run_systemctl(["is-active", get_service_name()], system=system, timeout=10, **_CAPTURE_TEXT)
    except (RuntimeError, subprocess.TimeoutExpired):
        return False
    return result.stdout.strip() == "active"


def _probe_systemd_service_running(system: bool = False) -> tuple[bool, bool]:
    from hermes_cli.gateway import _select_systemd_scope, _systemd_unit_is_active
    selected_system = _select_systemd_scope(system)
    return selected_system, _systemd_unit_is_active(selected_system)


def _parse_kv_pairs(items) -> dict[str, str]:
    """``{key: value}`` from ``KEY=VALUE`` strings (later keys win; values stripped)."""
    return {k: v.strip() for k, v in (item.split("=", 1) for item in items if "=" in item)}


def _systemctl_show(properties: tuple[str, ...], *, system: bool) -> dict[str, str]:
    """``systemctl show --property a,b`` for the gateway unit as ``{key: value}``; {} on failure."""
    from hermes_cli.gateway import (
        _CAPTURE_TEXT,
        _parse_kv_pairs,
        _run_systemctl,
        _select_systemd_scope,
        get_service_name,
        subprocess,
    )
    try:
        result = _run_systemctl(
            ["show", get_service_name(), "--no-pager", "--property", ",".join(properties)],
            system=_select_systemd_scope(system), timeout=10, **_CAPTURE_TEXT,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return {}
    return _parse_kv_pairs(result.stdout.splitlines()) if result.returncode == 0 else {}


def _unit_environment_value(unit_path: Path, name: str) -> str | None:
    """Value of one ``Environment="NAME=…"`` directive in the unit file at *unit_path*, with
    systemd's ``\\"``/``\\\\``/``%%`` quoting undone; None when the file or the key is absent."""
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        body = line.strip()
        if not body.startswith("Environment="):
            continue
        body = body[len("Environment=") :].strip()
        if body.startswith('"') and body.endswith('"'):
            body = body[1:-1].replace('\\"', '"').replace("\\\\", "\\").replace("%%", "%")
        if body.startswith(f"{name}="):
            return body.split("=", 1)[1].strip() or None
    return None


def _hermes_home_pinned_by_unit(unit_path: Path) -> str | None:
    """``HERMES_HOME`` pinned by the unit file at *unit_path*, or None when absent/unreadable."""
    from hermes_cli.gateway import _unit_environment_value
    return _unit_environment_value(unit_path, "HERMES_HOME")


def _hermes_home_from_systemd_unit_file(system: bool = False) -> str | None:
    """``HERMES_HOME`` from the on-disk unit file - what refresh/compare already read, and reliable under ``sudo``."""
    from hermes_cli.gateway import _hermes_home_pinned_by_unit, get_systemd_unit_path
    return _hermes_home_pinned_by_unit(get_systemd_unit_path(system=system))


def _sync_hermes_home_from_systemd_unit(system: bool) -> None:
    """Adopt a system-scope unit's ``HERMES_HOME``: under ``sudo`` it is stripped and HOME=/root, so
    get_hermes_home() would pick the wrong profile for runtime-status/PID reads."""
    from hermes_cli.gateway import _hermes_home_from_systemd_unit_file, _parse_kv_pairs, _systemctl_show, os
    if not system:
        return
    # On-disk unit first; ``systemctl show`` for units that only exist in the manager.
    unit_home = (_hermes_home_from_systemd_unit_file(system=True) or "").strip()
    if not unit_home:
        env_line = _systemctl_show(("Environment",), system=True).get("Environment", "")
        unit_home = _parse_kv_pairs(env_line.split()).get("HERMES_HOME", "").strip()
    if unit_home and os.environ.get("HERMES_HOME", "").strip() != unit_home:
        os.environ["HERMES_HOME"] = unit_home


def _read_systemd_unit_properties(
    system: bool = False,
    properties: tuple[str, ...] = ("ActiveState", "SubState", "Result", "ExecMainStatus", "MainPID"),
) -> dict[str, str]:
    """Return selected ``systemctl show`` properties for the gateway unit."""
    from hermes_cli.gateway import _systemctl_show
    return _systemctl_show(properties, system=system)


def _positive_pid(value) -> int | None:
    """``int(value)`` when it parses and is > 0, else None."""
    try:
        pid = int(value or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _systemd_main_pid_from_props(props: dict[str, str]) -> int | None:
    from hermes_cli.gateway import _positive_pid
    return _positive_pid(props.get("MainPID", "0") or "0")


def _runtime_state_pid(state: dict | None) -> int:
    """``pid`` recorded in a runtime-status dict; 0 when absent, unparsable, or non-positive."""
    from hermes_cli.gateway import _positive_pid
    return _positive_pid((state or {}).get("pid", 0)) or 0


def _systemd_main_pid(system: bool = False) -> int | None:
    from hermes_cli.gateway import _read_systemd_unit_properties, _systemd_main_pid_from_props
    return _systemd_main_pid_from_props(_read_systemd_unit_properties(system=system))


def _read_gateway_runtime_status() -> dict | None:
    try:
        from gateway.status import read_runtime_status
        state = read_runtime_status()
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _systemd_cli_bits(system: bool) -> tuple[str, str, str]:
    """``(sudo_prefix, scope_flag, user_flag)`` for printed hints: ``("sudo ", " --system", "")`` in
    system scope, ``("", "", "--user ")`` in user scope."""
    return ("sudo ", " --system", "") if system else ("", "", "--user ")


def _wait_for_systemd_service_restart(
    *,
    system: bool = False,
    previous_pid: int | None = None,
    timeout: float | None = None,
    replacement_observed: list[bool] | None = None,
) -> bool:
    """Wait for the gateway service to become active after a restart handoff."""
    from hermes_cli.gateway import (
        _print_systemd_start_limit_wait,
        _read_gateway_runtime_status,
        _read_systemd_unit_properties,
        _runtime_state_pid,
        _service_scope_label,
        _systemd_cli_bits,
        _systemd_main_pid_from_props,
        _systemd_restart_wait_timeout,
        _systemd_unit_is_start_limited,
        get_service_name,
        time,
    )
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    if timeout is None:
        timeout = _systemd_restart_wait_timeout(system=system)
    deadline = time.monotonic() + timeout
    printed_runtime_wait = False

    while time.monotonic() < deadline:
        props = _read_systemd_unit_properties(system=system)
        active_state = props.get("ActiveState", "")
        sub_state = props.get("SubState", "")
        try:
            from gateway.status import get_running_pid
            new_pid = get_running_pid()
        except Exception:
            new_pid = None
        new_pid = new_pid or _systemd_main_pid_from_props(props)

        runtime_state = _read_gateway_runtime_status()
        runtime_pid = _runtime_state_pid(runtime_state)
        if (
            previous_pid is not None
            and replacement_observed is not None
            and not replacement_observed
            and any(p > 0 and p != previous_pid for p in (new_pid or 0, runtime_pid))
        ):
            replacement_observed.append(True)

        if active_state == "active" and new_pid and (previous_pid is None or new_pid != previous_pid):
            if runtime_pid != new_pid:
                runtime_state = _read_gateway_runtime_status()
                if runtime_state and _runtime_state_pid(runtime_state) != new_pid:
                    runtime_state = None
            gateway_state = (runtime_state or {}).get("gateway_state")
            if gateway_state in ("running", "degraded"):
                print(f"✓ {scope_label} service restarted (PID {new_pid})")
                if gateway_state == "degraded":
                    # Serving, but a configured platform is parked or retrying: a real restart, not a
                    # failure — say so instead of waiting out the timeout and reporting one.
                    print(f"⚠ {scope_label} gateway is DEGRADED — see `hermes gateway status`")
                return True
            if gateway_state == "startup_failed":
                reason = (runtime_state or {}).get("exit_reason") or "startup failed"
                print(
                    f"⚠ {scope_label} service process restarted (PID {new_pid}), but gateway startup failed: {reason}"
                )
                return False
            if not printed_runtime_wait:
                print(f"⏳ {scope_label} service process started (PID {new_pid}); waiting for gateway runtime...")
                printed_runtime_wait = True

        if active_state == "activating" and sub_state == "auto-restart":
            time.sleep(1)
            continue

        if _systemd_unit_is_start_limited(props):
            _print_systemd_start_limit_wait(system=system)
            return False

        time.sleep(2)

    sudo, _, user_flag = _systemd_cli_bits(system)
    print(
        f"⚠ {scope_label} service did not become active within {int(timeout)}s.\n"
        f"  Check status: {sudo}hermes gateway status\n"
        f"  Check logs:   journalctl {user_flag}-u {svc} -l --since '2 min ago'"
    )
    return False


def _systemd_restart_wait_timeout(system: bool = False) -> float:
    """Cover systemd's relaunch delays before applying the runtime wait floor."""
    from hermes_cli.gateway import _read_systemd_unit_properties
    from gateway.shutdown_forensics import parse_systemd_duration_to_us
    props = _read_systemd_unit_properties(system=system, properties=("RestartUSec", "TimeoutStartUSec"))
    supervisor_budget = 0.0
    for name in ("RestartUSec", "TimeoutStartUSec"):
        raw = props.get(name, "")
        duration_us = int(raw) if raw.isdigit() else parse_systemd_duration_to_us(raw)
        if duration_us is not None:
            supervisor_budget += duration_us / 1_000_000
    return 60.0 + supervisor_budget


def _systemd_unit_is_start_limited(props: dict[str, str]) -> bool:
    return "start-limit-hit" in (props.get("Result", "").lower(), props.get("SubState", "").lower())


def _systemd_error_indicates_start_limit(exc: subprocess.CalledProcessError) -> bool:
    parts: list[str] = []
    for attr in ("stderr", "stdout", "output"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(value.decode(errors="replace") if isinstance(value, bytes) else str(value))
    text = "\n".join(parts).lower()
    return "start-limit-hit" in text or "start request repeated too quickly" in text or "start-limit" in text


def _systemd_service_is_start_limited(system: bool = False) -> bool:
    from hermes_cli.gateway import _read_systemd_unit_properties, _systemd_unit_is_start_limited
    return _systemd_unit_is_start_limited(_read_systemd_unit_properties(system=system))


def _print_systemd_start_limit_wait(system: bool = False) -> None:
    from hermes_cli.gateway import _service_scope_label, _systemd_cli_bits, get_service_name
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)
    print(f"⏳ {scope_label} service is temporarily rate-limited by systemd.")
    print("  systemd is refusing another immediate start after repeated exits.")
    print(f"  Wait for the start-limit window to expire, then run: {sudo}hermes gateway restart{scope_flag}")
    print(f"  Or clear the failed state manually: systemctl {user_flag}reset-failed {svc}")
    print(f"  Check logs: journalctl {user_flag}-u {svc} -l --since '5 min ago'")


def _recover_pending_systemd_restart(system: bool = False, previous_pid: int | None = None) -> bool:
    """Recover a planned service restart that is stuck in systemd state."""
    from hermes_cli.gateway import (
        GATEWAY_SERVICE_RESTART_EXIT_CODE,
        _read_systemd_unit_properties,
        _run_systemctl,
        _service_scope_label,
        _wait_for_systemd_service_restart,
        get_service_name,
    )
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return False

    try:
        from gateway.status import read_runtime_status
    except Exception:
        return False

    if not (read_runtime_status() or {}).get("restart_requested"):
        return False

    active_state = props.get("ActiveState", "")
    if active_state == "activating" and props.get("SubState", "") == "auto-restart":
        print("⏳ Service restart already pending — waiting for systemd relaunch...")
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    if active_state == "failed" and (
        props.get("ExecMainStatus", "") == str(GATEWAY_SERVICE_RESTART_EXIT_CODE)
        or props.get("Result", "") == "exit-code"
    ):
        svc = get_service_name()
        print(f"↻ Clearing failed state for pending {_service_scope_label(system)} service restart...")
        _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        _run_systemctl(["start", svc], system=system, check=False, timeout=90)
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    return False


def _parse_launchd_pid_from_list_output(output: str) -> int | None:
    """PID from ``launchctl list <label>`` (``"PID" = <n>;``); None if absent (registered, not running)
    or non-positive (crashed)."""
    from hermes_cli.gateway import _positive_pid
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(('"PID"', "PID")) and "=" in stripped:
            return _positive_pid(stripped.split("=", 1)[1].strip().rstrip(";").strip('"'))
    return None


def _parse_launchd_pid_from_print_output(output: str) -> int | None:
    """Live PID from ``launchctl print`` (first ``pid = <N>`` line wins); None if absent or non-positive."""
    from hermes_cli.gateway import _positive_pid
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            return _positive_pid(stripped[len("pid = "):].strip())
    return None


def _launchd_print_service_pid(domain: str, label: str) -> tuple[bool, int | None]:
    """``(loaded, pid)`` for ``domain/label`` via ``launchctl print`` (domain-explicit; ``launchctl list``
    infers it from caller context). ``TimeoutExpired`` propagates: a wedged launchctl is not "unloaded".

    Domain-explicit on purpose: legacy ``launchctl list`` infers its domain from the caller's execution
    context, which is exactly the ambiguity that sank the first fleet-restart attempt (#41403 review).
    ``TimeoutExpired`` propagates — fleet-restart callers own per-label failure accounting (a wedged
    launchctl call must be reported, not read as "unloaded").
    """
    from hermes_cli.gateway import _CAPTURE_TEXT, _parse_launchd_pid_from_print_output, subprocess
    try:
        result = subprocess.run(["launchctl", "print", f"{domain}/{label}"], timeout=5, **_CAPTURE_TEXT)
    except FileNotFoundError:
        return (False, None)
    if result.returncode != 0:
        return (False, None)
    return (True, _parse_launchd_pid_from_print_output(result.stdout))


def _launchd_service_registered(label: str, *, timeout: int = 5) -> bool:
    """True when launchd knows ``label`` (``launchctl list`` exit 0). Domain-agnostic, so still true on
    macOS 26+ hosts whose per-user domains reject management. FileNotFoundError/TimeoutExpired propagate."""
    from hermes_cli.gateway import _CAPTURE_TEXT, subprocess
    result = subprocess.run(["launchctl", "list", label], timeout=timeout, **_CAPTURE_TEXT)
    return result.returncode == 0


def _locate_launchd_gateway_service(label: str) -> tuple[str | None, int | None]:
    """``(domain, pid)`` for ``label``, probing ``gui/<uid>`` then ``user/<uid>``. Never uses the current
    profile's cached ``_launchd_domain()`` — a fleet can mix domains. ``TimeoutExpired`` propagates."""
    from hermes_cli.gateway import _launchd_print_service_pid, os
    uid = os.getuid()  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    for domain in (f"gui/{uid}", f"user/{uid}"):
        loaded, pid = _launchd_print_service_pid(domain, label)
        if loaded:
            return (domain, pid)
    return (None, None)


def _probe_launchd_service_running() -> bool:
    """True when the plist exists AND launchd is running a process for the current label."""
    from hermes_cli.gateway import (
        _launchctl_label_supervising_process,
        get_launchd_label,
        get_launchd_plist_path,
    )
    return get_launchd_plist_path().exists() and _launchctl_label_supervising_process(get_launchd_label())


def _s6_gateway_snapshot(gateway_pids: tuple[int, ...]) -> GatewayRuntimeSnapshot | None:
    """Snapshot for an s6-supervised container gateway, or None when s6 isn't the service manager."""
    from hermes_cli.gateway import GatewayRuntimeSnapshot, _current_profile_name
    from hermes_cli.service_manager import detect_service_manager, get_service_manager
    if detect_service_manager() != "s6":
        return None
    service_name = f"gateway-{_current_profile_name()}"
    mgr = get_service_manager()
    service_installed = service_running = False
    try:
        service_dir = getattr(mgr, "scandir", None)
        if service_dir is not None:
            service_installed = (service_dir / service_name).is_dir()
    except Exception:
        service_installed = False
    if service_installed:
        try:
            service_running = bool(mgr.is_running(service_name))
        except Exception:
            service_running = False
    return GatewayRuntimeSnapshot(
        manager="s6 (container supervisor)",
        service_installed=service_installed,
        service_running=service_running,
        gateway_pids=gateway_pids,
        service_scope="s6",
    )


def get_gateway_runtime_snapshot(system: bool = False) -> GatewayRuntimeSnapshot:
    """Return a unified view of gateway liveness for the current profile."""
    from hermes_cli.gateway import (
        GatewayRuntimeSnapshot,
        _probe_launchd_service_running,
        _probe_systemd_service_running,
        _s6_gateway_snapshot,
        _service_scope_label,
        find_gateway_pids,
        get_launchd_plist_path,
        get_systemd_unit_path,
        is_linux,
        is_macos,
        is_termux,
        supports_systemd_services,
    )
    gateway_pids = tuple(find_gateway_pids())
    if is_termux():
        return GatewayRuntimeSnapshot(manager="Termux / manual process", gateway_pids=gateway_pids)

    from hermes_constants import is_container
    if is_linux() and is_container():
        # Report s6 supervision under our /init; other container runtimes keep "docker (foreground)".
        try:
            snapshot = _s6_gateway_snapshot(gateway_pids)
            if snapshot is not None:
                return snapshot
        except Exception:
            pass  # Fall through to the legacy label on any detection error.
        return GatewayRuntimeSnapshot(manager="docker (foreground)", gateway_pids=gateway_pids)

    if supports_systemd_services():
        selected_system, service_running = _probe_systemd_service_running(system=system)
        scope_label = _service_scope_label(selected_system)
        return GatewayRuntimeSnapshot(
            manager=f"systemd ({scope_label})",
            service_installed=get_systemd_unit_path(system=selected_system).exists(),
            service_running=service_running,
            gateway_pids=gateway_pids,
            service_scope=scope_label,
        )

    if is_macos():
        return GatewayRuntimeSnapshot(
            manager="launchd",
            service_installed=get_launchd_plist_path().exists(),
            service_running=_probe_launchd_service_running(),
            gateway_pids=gateway_pids,
            service_scope="launchd",
        )

    return GatewayRuntimeSnapshot(manager="manual process", gateway_pids=gateway_pids)


def _format_gateway_pids(pids: tuple[int, ...] | list[int], *, limit: int | None = 3) -> str:
    rendered = [str(pid) for pid in (pids if limit is None else pids[:limit]) if pid > 0]
    if limit is not None and len(pids) > limit:
        rendered.append("...")
    return ", ".join(rendered)


def _print_gateway_process_mismatch(snapshot: GatewayRuntimeSnapshot) -> None:
    from hermes_cli.gateway import _format_gateway_pids, _launchd_unsupported_marker_exists
    if not snapshot.has_process_service_mismatch:
        return
    print()
    pids_line = f"  PID(s): {_format_gateway_pids(snapshot.gateway_pids, limit=None)}"
    # Managed detached fallback (launchd exit-5 path) vs. a genuinely manual run.
    if _launchd_unsupported_marker_exists():
        print("⚠ Gateway is running as a detached fallback process — launchd cannot supervise it")
        print(pids_line)
        print("  Auto-start at login and auto-restart on crash are NOT available.")
        print("  Stop it with: hermes gateway stop")
    else:
        print("⚠ Gateway process is running for this profile, but the service is not active")
        print(pids_line)
        print("  This is usually a manual foreground/tmux/nohup run, so `hermes gateway`")
        print("  can refuse to start another copy until this process stops.")


def _print_multiplex_standalone_reason() -> None:
    """The boot guard kept an unset-default gateway standalone: say so in status, with the remedy."""
    from hermes_cli.gateway_multiplex_mode import recorded_standalone_warning_lines
    for line in recorded_standalone_warning_lines():
        print(line)


def _print_served_ingress_urls(profile: str | None = None) -> None:
    """Callback URLs of inbound-port platforms the live multiplexer serves for secondary profiles
    (the value to paste into the Twilio / LINE / Teams / BlueBubbles console)."""
    try:
        from hermes_cli.gateway_multiplex_served import format_ingress_url_lines, served_profile_ingress_urls
        urls = served_profile_ingress_urls(profile)
    except Exception:
        return
    if not urls:
        return
    print()
    print("Inbound callback URLs on the shared listener:")
    for name, per_platform in sorted(urls.items()):
        for line in format_ingress_url_lines(per_platform, indent=f"  {name}/" if not profile else "  "):
            print(line)


def _print_unserved_shared_ingress(profile: str | None) -> None:
    """Shared-ingress platforms (WhatsApp/Relay) this served profile enabled that the multiplexer runs
    only on the default profile — the ``whatsapp: not served under multiplex`` line."""
    try:
        from hermes_cli.gateway_multiplex_served import served_profile_unserved_platforms
        unserved = served_profile_unserved_platforms(profile or "")
    except Exception:
        return
    if not unserved:
        return
    print()
    for platform, reason in sorted(unserved.items()):
        print(f"  ⚠ {platform}: {reason}")
    print("  Enable it on the default profile (shared ingress serves every profile), or disable it here.")


def _print_other_profiles_gateway_status() -> None:
    """Print other profiles' running gateways at the bottom of ``hermes gateway status``."""
    from hermes_cli.gateway import find_profile_gateway_processes
    try:
        from hermes_cli.profiles import get_active_profile_name
        current = get_active_profile_name()
        other_processes = [p for p in find_profile_gateway_processes() if p.profile != current]
        if not other_processes:
            return
        print()
        print("Other profiles:")
        for proc in other_processes:
            print(f"  ✓ {proc.profile:<16s} — PID {proc.pid}")
    except Exception:
        pass


def _print_duplicate_credential_warnings() -> None:
    """The migrate preflight's duplicate-credential findings, so ``gateway status`` explains a parked
    or racing bot (and why the fleet will not fold) with the same words as ``migrate --dry-run``."""
    from hermes_cli.gateway import contextlib
    with contextlib.suppress(Exception):
        from hermes_cli.gateway_migrate import duplicate_credential_findings
        lines = duplicate_credential_findings()
        if lines:
            print()
            for line in lines:
                print(f"⚠ {line}")


def _gateway_list() -> None:
    """List every profile and whether its gateway is running."""
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    try:
        from hermes_cli.profiles import list_profiles, get_active_profile_name
    except Exception:
        print("Unable to list profiles.")
        return

    profiles = list_profiles()
    if not profiles:
        print("No profiles found.")
        return

    current = get_active_profile_name()

    print("Gateways:")
    for prof in profiles:
        marker = "✓" if prof.gateway_running else "✗"
        label = prof.name + (" (current)" if prof.name == current else "")
        parts = [f"  {marker} {label:<24s}"]
        if prof.gateway_running:
            pid = None
            try:
                from gateway.status import get_running_pid
                pid = get_running_pid(prof.path / "gateway.pid", cleanup_stale=False)
            except Exception:
                pass
            if pid:
                parts.append(f"PID {pid}")
            elif named_profile_served_by_running_multiplexer(prof.name):
                parts.append("served by the default multiplexer")
        else:
            parts.append("not running")
        print(" — ".join(parts))


def kill_gateway_processes(force: bool = False, exclude_pids: set | None = None, all_profiles: bool = False) -> int:
    """Kill running gateway processes (force-kill if ``force``); ``exclude_pids`` skips e.g. just-
    restarted service PIDs. Returns count killed."""
    from hermes_cli.gateway import _capture_gateway_argv, find_gateway_pids, terminate_pid
    killed = 0
    for pid in find_gateway_pids(exclude_pids=exclude_pids, all_profiles=all_profiles):
        try:
            expected_start_time = None
            if force:
                # Re-verify the LIVE cmdline at kill time: a PID recycled since the scan must never be tree-killed.
                # Re-verify at kill time, not just scan time: the cmdline match inside find_gateway_pids()
                # is stale by the time we get here, and a recycled PID could otherwise be tree-killed
                # (#89614 class). _capture_gateway_argv re-reads the LIVE cmdline and returns None for
                # anything that no longer looks like a gateway — refuse those.
                if _capture_gateway_argv(pid) is None:
                    continue
                from gateway.status import get_process_start_time
                expected_start_time = get_process_start_time(pid)
            terminate_pid(pid, force=force, expected_start_time=expected_start_time)
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
        except OSError as exc:
            print(f"Failed to kill PID {pid}: {exc}")
    return killed


_REAPER_SUPERVISOR_WALK_LIMIT = 12


def _reaper_candidate_is_supervisor_owned(pid: int) -> bool:
    """True when ``pid``'s parent chain reaches ``services.exe`` (Task Scheduler-owned gateway). Windows-only
    reaper backstop: ``_get_service_pids()`` is empty there, so a Scheduled-Task gateway with a stale
    pidfile would look like an orphan. Fail-open once the Task's bootstrap parent exits. Not applied
    on POSIX, where everything descends from PID 1 and would look supervised.

    See #83683, #86098.
    This check is deliberately NOT applied on POSIX: there, every process has PID 1 (launchd / init /
    systemd) in its ancestry — and a genuine orphan is *reparented directly to PID 1* — so supervisor-name
    ancestry carries zero signal and would spare every orphan the reaper exists to kill (#51325, 75936).
    POSIX supervised gateways are already covered pidfile- independently by the ``_get_service_pids()``
    exclusion.
    """
    from hermes_cli.gateway import _REAPER_SUPERVISOR_WALK_LIMIT, contextlib, is_windows
    if not is_windows():
        return False
    try:
        import psutil  # type: ignore
        parent = psutil.Process(pid).parent()
        for _ in range(_REAPER_SUPERVISOR_WALK_LIMIT):
            if parent is None:
                break
            with contextlib.suppress(Exception):
                if (parent.name() or "").lower() == "services.exe":
                    return True
            parent = parent.parent()
    except Exception:
        pass
    return False


def _reap_unsupervised_gateway_orphans(extra_exclude: set | None = None) -> bool:
    """Kill no-supervisor gateway orphans the pidfile/runtime record can't see. On WSL/no-systemd hosts
    the restart fallback runs the gateway in-process under a ``gateway restart`` argv; a stale pidfile
    then lets a live orphan keep the webhook port while a restart stacks a duplicate. No-op where a
    supervisor exists (there ``gateway restart`` is a transient command). ``extra_exclude``: already killed."""
    from hermes_cli.gateway import (
        _await_gateway_exit,
        _force_kill_survivors,
        _reaper_candidate_is_supervisor_owned,
        _reaper_exclusion_pids,
        _windows_scheduled_task_supervises,
        contextlib,
        find_gateway_pids,
        is_windows,
        os,
        signal,
        supports_systemd_services,
    )
    try:
        supervised_host = supports_systemd_services()
    except Exception:
        supervised_host = True
    if supervised_host:
        return False

    # Task Scheduler is a supervisor too; its state beats a parent-chain walk (broken once the bootstrap exits).
    # A Scheduled Task gateway whose conhost/VBS bootstrap has already exited is invisible to
    # `_reaper_candidate_is_supervisor_owned` (the parent chain breaks before services.exe, fail-open), yet
    # it is alive and supervised. After that launcher exits the task is typically Ready, not Running —
    # treating only Running as supervised still kills the detached gateway on every desktop serve start
    # (#86098, #87001).
    if is_windows():
        try:
            from hermes_cli.gateway_windows import get_task_name  # profile-aware task name
            _task_name = get_task_name()
        except Exception:
            _task_name = "Hermes_Gateway"
        if _windows_scheduled_task_supervises(_task_name):
            return False

    from gateway.status import _pid_exists, get_process_start_time, write_planned_stop_marker
    own = _reaper_exclusion_pids(extra_exclude)
    try:
        # On Windows also drop Task Scheduler-owned candidates (the pidfile-less gap).
        orphans = [
            p for p in find_gateway_pids(exclude_pids=own) if p and p > 0 and not _reaper_candidate_is_supervisor_owned(p)
        ]
    except Exception:
        return False
    if not orphans:
        return False

    # Pin each orphan's start time now: the delayed SIGKILL must never hit a recycled PID.
    # Pin each orphan's identity NOW: the cmdline scan above matched at scan-time only, and the SIGKILL
    # escalation below fires seconds later. A PID recycled inside that window must never be force-killed
    # (#89614 class). Fingerprint capture is best-effort — SIGTERM below proceeds regardless (it targets the
    # process verified by the scan an instant ago), but the delayed SIGKILL requires a still-matching
    # fingerprint.
    orphan_identity: dict[int, int] = {}
    for pid in orphans:
        start = get_process_start_time(pid)
        if start is not None:
            orphan_identity[pid] = start

    reaped = False
    for pid in orphans:
        with contextlib.suppress(Exception):
            write_planned_stop_marker(pid)
        # ``os.kill(..., SIGTERM)`` maps to TerminateProcess on Windows, so it
        # would kill the gateway before its marker watcher can drain and close
        # state cleanly. Let the bounded survivor wait below escalate instead.
        if is_windows():
            reaped = True
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"⚠ Permission denied to kill orphaned gateway PID {pid}")
            continue
        reaped = True

    # Wait, then force-kill survivors so the replacement can bind the port cleanly.
    # Fail-closed: SIGKILL only a PID that still names the process fingerprinted at scan time.
    _force_kill_survivors([
        pid for pid in _await_gateway_exit(orphans, pid_exists=_pid_exists)
        if pid in orphan_identity and get_process_start_time(pid) == orphan_identity[pid]
    ])
    return reaped


def _reaper_exclusion_pids(extra_exclude: set | None) -> set[int]:
    """PIDs the orphan reaper must never kill: self, caller extras, service-managed, recorded."""
    from hermes_cli.gateway import _get_service_pids, contextlib, os
    own = {os.getpid()} | (extra_exclude or set())
    # Service-managed gateways are never orphans (on macOS supports_systemd_services() is False, so a
    # launchd gateway would otherwise be SIGTERM'd); all_profiles because the scan sees siblings too.
    with contextlib.suppress(Exception):
        # This covers macOS launchd (supports_systemd_services() is False there, so without this the launchd
        # gateway looks like an unsupervised orphan and gets SIGTERM'd, causing launchd to restart it — or
        # leaving it down under KeepAlive.SuccessfulExit=false) and any systemd unit reachable from a host
        # that got past the gate above (#83683, #85344).
        # all_profiles=True: the reaper's process scan sees every profile's gateway (and on macOS the
        # now-working ps fallback surfaces sibling launchd gateways, #73626), so the service exclusion must
        # cover the whole ai.hermes.gateway* fleet — not just the current profile's label — or a sibling
        # profile's launchd gateway is misclassified as an unsupervised orphan and reaped. Same class as the
        # update-sweep fix in #74075.
        own |= _get_service_pids(all_profiles=True)
    # Exempt the recorded gateway PID and its parent chain (on Windows the Scheduled-Task bootstrap's
    # ``gateway run`` argv matches the scan; killing it takes the gateway down). Use the RAW pidfile +
    # lock records, not only the validated probe: get_running_pid returns None on any validation
    # hiccup — exactly when a healthy standalone gateway would be hard-killed (Windows SIGTERM is
    # TerminateProcess, no drain). For a KILL exclusion list a stale PID at worst spares one process;
    # a false negative kills a live gateway. The probe still supplies the runtime-status fallback PID.
    try:
        from gateway.status import _pid_from_record, _read_gateway_lock_record, _read_pid_record, get_running_pid
        recorded_pids = {_pid_from_record(rec) for rec in (_read_pid_record(), _read_gateway_lock_record())}
        recorded_pids.add(get_running_pid(cleanup_stale=False))
        for recorded in recorded_pids:
            if not recorded or recorded <= 0:
                continue
            own.add(recorded)
            try:
                import psutil  # type: ignore
                parent = psutil.Process(recorded).parent()
                while parent is not None:
                    own.add(parent.pid)
                    parent = parent.parent()
            except Exception:
                pass
    except Exception:
        pass
    return own


# A retiring gateway runs a PASSIVE WAL checkpoint in ``SessionDB.close()``; a SIGKILL mid-checkpoint
# corrupts ``state.db``. It keeps serving while we wait, so a long grace only delays the port bind.
_ORPHAN_EXIT_GRACE_SECONDS = 30.0
_ORPHAN_EXIT_POLL_SECONDS = 0.2


def _await_gateway_exit(
    pids, *, pid_exists, sleep=None, grace_s: float = _ORPHAN_EXIT_GRACE_SECONDS, poll_s: float = _ORPHAN_EXIT_POLL_SECONDS
):
    """Poll up to *grace_s* for *pids* to exit; return survivors. ``pid_exists``/``sleep`` injectable for tests."""
    from hermes_cli.gateway import time
    if sleep is None:
        sleep = time.sleep
    survivors = list(pids)
    for _ in range(max(1, int(grace_s / poll_s))):
        survivors = [p for p in survivors if pid_exists(p)]
        if not survivors:
            break
        sleep(poll_s)
    else:
        # Re-check after the LAST sleep, or a recycled PID could get the SIGKILL.
        survivors = [p for p in survivors if pid_exists(p)]
    return survivors


def _force_kill_survivors(survivors, *, kill=None) -> None:
    """SIGKILL processes that outlasted the grace period, loudly — a force-kill can tear the store, so
    it must leave a trace."""
    from hermes_cli.gateway import _ORPHAN_EXIT_GRACE_SECONDS, contextlib, logger, os, signal
    kill = kill or os.kill
    for pid in survivors:
        logger.warning(
            "Gateway PID %s did not exit within %.0fs of the stop request (SIGTERM, or the planned-stop "
            "marker on Windows) — sending "
            "SIGKILL. A kill during a WAL checkpoint can corrupt state.db; "
            "the next start will run an integrity check.",
            pid, _ORPHAN_EXIT_GRACE_SECONDS,
        )
        with contextlib.suppress((ProcessLookupError, PermissionError, OSError)):
            kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def _mark_planned_stop(pid: int | None = None) -> None:
    """Best-effort planned-stop marker for ``pid`` (default: the recorded gateway PID)."""
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker
        if pid is None:
            pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass


def stop_profile_gateway() -> bool:
    """Stop only this profile's gateway via its PID file; True if a process was stopped. Without a
    supervisor the pidfile can be stale while a live orphan holds the webhook port, so fall back to
    the orphan-aware scan rather than stacking a duplicate.

    Even when the pid file is valid and points to the current gateway, older orphans may linger from prior
    restarts that overwrote the pid file before the old process exited. After killing the recorded PID, also
    sweep for any remaining orphans so each restart produces at most one live gateway (#75936).
    """
    from hermes_cli.gateway import (
        _mark_planned_stop,
        _reap_unsupervised_gateway_orphans,
        is_windows,
        logger,
        os,
        signal,
        time,
    )
    try:
        from gateway.status import get_running_pid, remove_pid_file
    except ImportError:
        return False

    pid = get_running_pid()
    if pid is None:
        return _reap_unsupervised_gateway_orphans()

    if is_windows():
        # Windows maps SIGTERM to TerminateProcess. The marker watcher is the
        # gateway's graceful-stop IPC, so wait for it before force-killing a
        # wedged process.
        from gateway.status import get_process_start_time
        from hermes_cli.gateway_windows import (
            _drain_gateway_pid,
            _force_terminate_known_gateway_pids,
            _windows_stop_drain_timeout,
        )

        # Capture identity BEFORE the drain (as _escalate_wedged_gateway does): if the PID is
        # recycled during the wait, terminate_pid's start-time mismatch refuses the taskkill.
        expected_start_time = get_process_start_time(pid)
        if not _drain_gateway_pid(pid, _windows_stop_drain_timeout()):
            _force_terminate_known_gateway_pids({pid: expected_start_time})
    else:
        _mark_planned_stop(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already gone
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
            return False

    # ``_pid_exists``, NOT ``os.kill(pid, 0)`` (TerminateProcess on Windows).
    from gateway.status import _pid_exists
    for _ in range(20):
        if not _pid_exists(pid):
            break
        time.sleep(0.5)

    if get_running_pid() is None:
        remove_pid_file()

    # Reap orphans from prior restarts whose pidfile entry was overwritten; skip the PID just killed.
    try:
        # Exclude the PID we just killed so the sweep doesn't double-kill a process that's still tearing
        # down — _reap_unsupervised_gateway_orphans already excludes our own PID. See #75936.
        _reap_unsupervised_gateway_orphans(extra_exclude={pid} if pid else None)
    except Exception as exc:
        logger.debug("orphan reap after stop_profile_gateway failed: %s", exc)
    return True
