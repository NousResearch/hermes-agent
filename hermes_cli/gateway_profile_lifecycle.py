"""Named-profile lifecycle inside the host gateway, never a service-manager action."""
from __future__ import annotations

import logging

from hermes_constants import get_hermes_home, get_default_hermes_root
from hermes_cli.profiles import parked_marker_path, profile_is_parked, profile_is_standalone, profiles_to_serve


logger = logging.getLogger(__name__)


def ack_confirmed(answer, key, name):
    return isinstance(answer, dict) and answer.get(key) == name and not answer.get("error")


def ack_failure(answer):
    if isinstance(answer, dict):
        return answer.get("error") or "host operation is still pending"
    return "host control socket did not answer"


def mark_planned_stop(pid: int | None = None) -> None:
    """Best-effort planned-stop marker for ``pid`` (default: the recorded gateway PID)."""
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker
        if pid is None:
            pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass


def host_scope_for_all_verb(owner):
    """Home a ``--all`` verb acts as: the live host owner's, else the default root.

    ``--all`` names the ONE host gateway, whichever profile invoked the CLI — but the active home
    decides the service name, the Windows task/launcher and the detached child's ``--profile``, so
    with the host DOWN ``-p X gateway start|restart --all`` re-launched X's own gateway, which the
    named-profile guard then refused (exit 78): the host stayed down after a Desktop update.
    """
    from hermes_cli.gateway_migrate import _home_env
    return _home_env(owner.home if owner is not None else get_default_hermes_root())


def profile_lifecycle(command: str, args) -> bool:
    """True when a named-profile command was handled (including an unconfirmed request)."""
    from hermes_cli import gateway as gw
    from gateway.control_socket import request_unserve_profile, request_serve_profile_hot

    name = gw._current_profile_name()
    if not name or name == "default" or getattr(args, "all", False) or getattr(args, "force", False):
        return False
    home = get_hermes_home()
    if profile_is_standalone(home):
        # gateway.standalone wins: the host never serves (or parks) an opted-out profile, so its
        # verbs keep addressing its own gateway process even while a stale host record lists it.
        return False
    marker = parked_marker_path(home)
    if command == "restart" and profile_is_parked(home) and not gw.find_gateway_pids():
        # `stop` parked the profile AND dropped it from the host's served set, so the unserve/serve
        # path below finds no host serving it and would fall through to a standalone restart.
        # Unpark like `start` does; a live --force gateway keeps its own restart (pids non-empty).
        command = "start"
    if command == "start":
        if not profile_is_parked(home):
            return False
        marker.unlink()
        owner = gw._host_multiplexer_for_all_verb()
        if owner is None:
            from hermes_cli.gateway_multiplex_served import live_default_gateway_pid
            if live_default_gateway_pid() is None:
                return False  # Unpark even when today's normal start path must start the host first.
        host_home = owner.home if owner is not None else get_default_hermes_root()
        answer = request_serve_profile_hot(host_home, name)
        if ack_confirmed(answer, "served", name):
            print(f"Profile '{name}' served by the host gateway.")
        else:
            print(f"Profile '{name}' unparked, but serving was not confirmed: {ack_failure(answer)}.")
            print("The host retries on its next rescan (within 30s). Check gateway status.")
        return True

    # A separate --force gateway still owns its normal process/service lifecycle.
    if gw.find_gateway_pids():
        return False
    owner = gw._served_by_another_host_gateway()
    if owner is None and not gw.named_profile_served_by_running_multiplexer():
        return False
    host_home = owner.home if owner is not None else get_default_hermes_root()
    if command == "stop":
        # Persist intent BEFORE teardown: the periodic rescan must not re-add this profile.
        marker.touch()
        answer = request_unserve_profile(host_home, name)
        if ack_confirmed(answer, "unserved", name):
            print(f"Profile '{name}' parked; its bots and cron are stopped. "
                  f"Start again with: hermes -p {name} gateway start")
        else:
            print(f"Profile '{name}' parked, but immediate stop was not confirmed: {ack_failure(answer)}.")
            print("The host drops it on its next rescan (within 30s).")
        return True

    answer = request_unserve_profile(host_home, name)
    if not ack_confirmed(answer, "unserved", name):
        print(f"Profile '{name}' restart was not confirmed: {ack_failure(answer)}.")
        return True
    answer = request_serve_profile_hot(host_home, name)
    if ack_confirmed(answer, "served", name):
        print(f"Profile '{name}' restarted by the host gateway.")
    else:
        print(f"Profile '{name}' stopped, but serving was not confirmed: {ack_failure(answer)}.")
        print("The host retries on its next rescan (within 30s). Check gateway status.")
    return True


def print_parked_status() -> bool:
    """A parked satellite is still installed, but is not a running gateway."""
    from hermes_cli import gateway as gw
    name = gw._current_profile_name()
    if name and name != "default" and profile_is_parked(get_hermes_home()):
        print(f"Profile '{name}': parked (hermes -p {name} gateway start)")
        return True
    if not name or name == "default":
        parked = [profile for profile, home in profiles_to_serve(True, include_parked=True)
                  if profile != "default" and profile_is_parked(home)]
        if parked:
            owner = gw.host_multiplexer_serving()
            if owner is not None:
                print(f"Served profiles: {', '.join(owner.profiles)}")
            for profile in parked:
                print(f"Profile '{profile}': parked (hermes -p {profile} gateway start)")
    return False


def owned_gateway_pid() -> int | None:
    """This profile's own gateway pid, only on positive, fresh ownership evidence; else None.

    A served profile's runtime record carries the host multiplexer's pid (#141133), so the recorded
    pid alone never authorizes a stop. Every ``find_gateway_pids()`` caller (stop, uninstall's kill,
    ``service_manager.is_running``, cron readiness, Windows start polls) reads it through here.
    """
    from gateway.status import get_running_pid, _get_process_hermes_home

    try:
        pid = get_running_pid()
        if pid is None or not _pid_owned_by_home(pid, _get_process_hermes_home()):
            return None
        # The runtime may have switched owners while the OS/control probes ran.
        return pid if get_running_pid() == pid else None
    except Exception:
        # An inaccessible rendezvous or process is never permission to kill it.
        logger.debug("Could not prove gateway process ownership", exc_info=True)
        return None


def _pid_owned_by_home(pid: int, home) -> bool:
    """Ownership ladder, cheapest proof first; the control socket is dialled only for the host's pid.

    ``gateway status`` asks once per profile and every served profile's record names the host, so the
    host record (a file read plus an incarnation check) settles those without a dial.
    """
    from gateway import host_attach, host_rendezvous as hr
    from gateway.status import _same_hermes_home
    from hermes_cli.dashboard_procs import _hermes_home_for_pid

    record = hr.read_record(hr.ROLE_GATEWAY)
    is_host = record is not None and record.pid == pid and hr.record_names_live_owner(record)
    if is_host and not _same_hermes_home(host_attach.record_home(record), home):
        return False  # the live host, launched from another home
    if _launch_record_owns(pid, home):
        return True
    if is_host:
        host_attach.invalidate_host_gateway_cache()
        owner = host_attach.host_gateway()
        # A verified identity answer from the host's own home needs no argv profile (bare launch).
        if owner is not None and owner.pid == pid and owner.served_known:
            return _same_hermes_home(owner.home, home)
    process_home = _hermes_home_for_pid(pid, require_launch_identity=True)
    return process_home is not None and _same_hermes_home(process_home, home)


def _launch_record_owns(pid: int, home) -> bool:
    """``<home>/gateway.pid`` names *pid*'s exact incarnation as launched from *home*.

    A gateway writes that file once, about its own launch home (the host never writes one into a
    served profile), so it pins the profile resolved at launch: a ``gateway run --force`` that relied
    on the sticky ``active_profile`` stays stoppable after the sticky choice changes, which argv and
    environment alone cannot prove.
    """
    from gateway.status import (
        _GATEWAY_KIND, _get_process_start_time, _pid_from_record, _read_pid_record, _same_hermes_home)

    record = _read_pid_record() or {}
    started, recorded_home = record.get("start_time"), record.get("hermes_home")
    return (record.get("kind") == _GATEWAY_KIND and _pid_from_record(record) == pid
            and started is not None and started == _get_process_start_time(pid)
            and isinstance(recorded_home, str) and _same_hermes_home(recorded_home, home))


def report_profile_stop_failure() -> None:
    """Do not report absence when a live PID was refused or the probe failed."""
    from gateway.status import get_running_pid
    from hermes_cli import gateway as gw

    try:
        absent = get_running_pid() is None and not gw.find_gateway_pids()
    except Exception:
        logger.debug("Could not confirm gateway absence after stop", exc_info=True)
        absent = False
    if absent:
        print("✗ No gateway running for this profile")
        return
    gw.print_error("Gateway stop was not confirmed; a gateway may still be running. "
                   "Check gateway status and process ownership.")
    _print_host_hint()
    raise SystemExit(1)


def _print_host_hint() -> None:
    """Name the refused pid when it is the host gateway serving this profile (#141133)."""
    from gateway import host_rendezvous as hr
    from gateway.status import get_running_pid

    try:
        pid, record = get_running_pid(), hr.read_record(hr.ROLE_GATEWAY)
    except Exception:
        return
    if pid is not None and record is not None and record.pid == pid:
        print(f"  PID {pid} is the host gateway serving this profile; stopping it would take every "
              f"served profile offline. Stop the host instead: hermes gateway stop --all")
