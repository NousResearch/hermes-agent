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


def config_set_live_note(key: str) -> Optional[str]:
    """After ``hermes config set <key>``: nudge the running gateway when the key can change a platform
    (an env setting such as a bot token, or a ``platforms.*`` block) and describe only what it did.
    None when nothing ran or changed, so no hint is printed that is not true."""
    from hermes_cli.gateway import _current_profile_name
    from hermes_constants import get_hermes_home
    if not (key.isupper() or key.split(".", 1)[0] in {"platforms", "gateway"}):
        return None
    answer = reconcile_running_gateway_platforms(_current_profile_name() or "default", get_hermes_home())
    if not answer or not answer.get("changed"):
        return None
    states = answer.get("platforms") or {}
    parts = [f"{verb} {', '.join(names)}" for verb, names in (
        ("started", answer.get("started")), ("restarted", answer.get("restarted")),
        ("stopped", answer.get("stopped"))) if names]
    down = [p for p in (*(answer.get("started") or ()), *(answer.get("restarted") or ()))
            if states.get(p) == "unavailable"]
    note = f"  Running gateway applied it now: {'; '.join(parts)}"
    if down:
        note += f"\n  ⚠ {', '.join(down)} could not start; see `hermes logs`"
    return note
