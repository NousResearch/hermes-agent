"""Named-profile lifecycle inside the host gateway, never a service-manager action."""
from __future__ import annotations

from hermes_constants import get_hermes_home, get_default_hermes_root
from hermes_cli.profiles import parked_marker_path, profile_is_parked, profile_is_standalone, profiles_to_serve


def _confirmed(answer, key, name):
    return isinstance(answer, dict) and answer.get(key) == name and not answer.get("error")


def _failure(answer):
    if isinstance(answer, dict):
        return answer.get("error") or "host operation is still pending"
    return "host control socket did not answer"


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
    if command in ("start", "restart") and profile_is_parked(home):
        # A prior stop removed the profile from the host's served set, so `restart` finds no
        # server to unserve/reserve through the shared path below and must unpark exactly like
        # `start` does.
        marker.unlink()
        owner = gw._host_multiplexer_for_all_verb()
        if owner is None:
            from hermes_cli.gateway_multiplex_served import live_default_gateway_pid
            if live_default_gateway_pid() is None:
                return False  # Unpark even when today's normal start path must start the host first.
        host_home = owner.home if owner is not None else get_default_hermes_root()
        answer = request_serve_profile_hot(host_home, name)
        if _confirmed(answer, "served", name):
            print(f"Profile '{name}' served by the host gateway.")
        else:
            print(f"Profile '{name}' unparked, but serving was not confirmed: {_failure(answer)}.")
            print("The host retries on its next rescan (within 30s). Check gateway status.")
        return True
    if command == "start":
        return False

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
        if _confirmed(answer, "unserved", name):
            print(f"Profile '{name}' parked; its bots and cron are stopped. "
                  f"Start again with: hermes -p {name} gateway start")
        else:
            print(f"Profile '{name}' parked, but immediate stop was not confirmed: {_failure(answer)}.")
            print("The host drops it on its next rescan (within 30s).")
        return True

    answer = request_unserve_profile(host_home, name)
    if not _confirmed(answer, "unserved", name):
        print(f"Profile '{name}' restart was not confirmed: {_failure(answer)}.")
        return True
    answer = request_serve_profile_hot(host_home, name)
    if _confirmed(answer, "served", name):
        print(f"Profile '{name}' restarted by the host gateway.")
    else:
        print(f"Profile '{name}' stopped, but serving was not confirmed: {_failure(answer)}.")
        print("The host retries on its next rescan (within 30s). Check gateway status.")
    return True


def print_parked_status() -> bool:
    """A parked satellite is still installed, but is not a running gateway — unless a `--force`
    standalone gateway bypassed parking and is live for this profile, which the marker alone
    cannot see: the caller must still read runtime state for that case."""
    from hermes_cli import gateway as gw
    name = gw._current_profile_name()
    home = get_hermes_home()
    if name and name != "default" and profile_is_parked(home):
        from hermes_cli.web_server_gateway import _has_own_gateway
        if _has_own_gateway(home):
            print(f"Profile '{name}': parking marker present, but a gateway is live for this "
                  f"profile (started with --force?).")
            return False
        print(f"Profile '{name}': parked (hermes -p {name} gateway start)")
        return True
    if not name or name == "default":
        parked = [(profile, phome) for profile, phome in profiles_to_serve(True, include_parked=True)
                  if profile != "default" and profile_is_parked(phome)]
        if parked:
            from hermes_cli.web_server_gateway import _has_own_gateway
            owner = gw.host_multiplexer_serving()
            if owner is not None:
                print(f"Served profiles: {', '.join(owner.profiles)}")
            for profile, phome in parked:
                if _has_own_gateway(phome):
                    print(f"Profile '{profile}': parking marker present, but a gateway is live for "
                          f"this profile (started with --force?).")
                else:
                    print(f"Profile '{profile}': parked (hermes -p {profile} gateway start)")
    return False
