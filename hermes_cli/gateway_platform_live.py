"""Tell the running gateway that a profile's platform configuration changed, so it starts, restarts or
stops that profile's adapters now (``reconcile-platforms`` control verb) instead of at a restart.

Shared by the dashboard/Desktop messaging save (``web_routers/messaging.py``) and ``hermes config set``.
The gateway that serves a profile is the multiplexer at the default root when it serves that named
profile, else the profile's own gateway. ``None`` = no gateway answered, or one predating the verb.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def reconcile_running_gateway_platforms(profile_name: str, home: Path, *, timeout: float = 15.0) -> Optional[dict]:
    """``reconcile-platforms`` answer of the gateway serving ``profile_name`` (whose home is ``home``)."""
    from gateway.control_socket import query_gateway_control
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    from hermes_constants import get_default_hermes_root
    served_by_host = profile_name != "default" and named_profile_served_by_running_multiplexer(profile_name)
    gateway_home = get_default_hermes_root() if served_by_host else Path(home)
    return query_gateway_control(gateway_home, "reconcile-platforms", params={"home": str(home)}, timeout=timeout)


def set_config_value_live(key: str, value: str, force: bool = False) -> None:
    """``hermes config set``: write the value, then print what a running gateway did with it, if anything."""
    from hermes_cli.config import set_config_value
    set_config_value(key, value, force)
    note = config_set_live_note(key)
    if note:
        print(note)


def config_set_live_note(key: str) -> Optional[str]:
    """After ``hermes config set <key>``: nudge the running gateway only when the key can change a
    platform (a platform's env setting such as a bot token, or a ``platforms.*`` / ``gateway.*`` path)
    and describe only what it did. Any other key (``model``, ``terminal.backend``) never contacts a
    gateway. None when nothing ran or changed, so no hint is printed that is not true."""
    if not _can_change_a_platform(key):
        return None
    from hermes_cli.gateway import _current_profile_name
    from hermes_constants import get_hermes_home
    answer = reconcile_running_gateway_platforms(_current_profile_name() or "default", get_hermes_home())
    if answer and answer.get("pending"):
        return _pending_note(answer)
    if not answer or not answer.get("changed"):
        return None
    return _live_note(answer)


def _can_change_a_platform(key: str) -> bool:
    """A ``platforms.*`` / ``gateway.*`` path, or an env setting a messaging platform owns: declared
    outright by an adapter, or under a platform's env prefixes (what a platform's build print reads)."""
    if "." in key:
        return key.split(".", 1)[0] in {"platforms", "gateway"}
    from hermes_cli.config_env_routing import is_env_setting_key
    if not is_env_setting_key(key):
        return False
    from hermes_cli.profile_channels import declared_channel_env_keys, platform_env_prefixes, platform_ids
    name = key.upper()
    return name in declared_channel_env_keys() or any(
        name.startswith(prefix) for pid in platform_ids() for prefix in platform_env_prefixes(pid))


def _pending_note(answer: dict) -> str:
    """The gateway's verb window closed before its apply finished: say what it is still doing."""
    parts = [f"{verb} {', '.join(names)}" for verb, names in (
        ("starting", answer.get("starting")), ("restarting", answer.get("restarting")),
        ("stopping", answer.get("stopping"))) if names]
    if not parts:
        return "  Running gateway is still applying it; `hermes logs` shows the outcome"
    still = "still connecting" if parts[0].startswith("starting") and len(parts) == 1 else "still in progress"
    return f"  Running gateway is {'; '.join(parts)} ({still})"


def _live_note(answer: dict) -> str:
    """One truthful line per outcome from the gateway's own verdict: what it now serves, then (separately)
    what it tried and could not bring up, with its reason when it gave one."""
    states, reasons = answer.get("platforms") or {}, answer.get("reasons") or {}
    down = [p for p in (*(answer.get("started") or ()), *(answer.get("restarted") or ()))
            if states.get(p) == "unavailable"]
    parts = [f"{verb} {', '.join(applied)}" for verb, names in (
        ("started", answer.get("started")), ("restarted", answer.get("restarted")),
        ("stopped", answer.get("stopped"))) if (applied := [p for p in names or () if p not in down])]
    lines = [f"  Running gateway applied it now: {'; '.join(parts)}"] if parts else []
    lines += [f"  ⚠ Running gateway could not start {p}: {reasons.get(p) or 'see `hermes logs`'}" for p in down]
    return "\n".join(lines)
