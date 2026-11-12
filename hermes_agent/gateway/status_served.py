"""Which runtime record describes a profile the host multiplexer may serve.

One rule for ``/api/status`` and the messaging card so the two surfaces never pick different records
for the same profile. Read-only: it selects and projects existing ``gateway_state.json`` records and
writes nothing. ``hermes_agent.gateway.status`` is resolved at call time so its test seams
(``monkeypatch.setattr(hermes_agent.gateway.status, ...)``) still intercept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def local_liveness_pid(liveness: Any) -> Optional[int]:
    """``liveness.pid`` when it was proven on THIS host; None for a health-probe answer, whose PID
    belongs to the remote container and may collide with an unrelated local one."""
    return None if liveness.source == "health" else liveness.pid


def served_profile_runtime(
    home: Path, own_runtime: Optional[dict[str, Any]], *, gateway_pid: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """The multiplexer's record projected onto ``home``'s profile (bare platform keys, the shape a
    standalone gateway writes) when it is that profile's live runtime record, else None.

    - ``own_runtime`` wins whenever it proves a live gateway of ``home`` -- an own ``--force`` one, or
      the host's own record on the host itself, which must stay whole so other profiles'
      ``<profile>:<platform>`` failures remain visible there;
    - a missing/unreadable multiplexer record (``(pid, {})``) is no record;
    - ``gateway_pid`` (a PID the caller proved on this host) must BE the multiplexer, so a gateway of
      the profile's own holding its lock keeps its record.
    """
    from hermes_agent.gateway import status

    if own_runtime is not None and status.get_runtime_status_running_pid(own_runtime, expected_home=home) is not None:
        return None
    served = status.multiplexer_liveness_for_profile(home)
    if served is None or not served[1]:
        return None
    mux_pid, record = served
    if gateway_pid is not None and mux_pid != gateway_pid:
        return None
    # Fold on the profile NAME, never ``home.name``: the default root's basename is ``.hermes`` (or
    # any custom HERMES_HOME), so its flat keys never matched (#123088).
    name = status.profile_name_for_home(home) or "default"
    return {**record, "platforms": status.profile_platforms_from_multiplexer(record, name)}


def served_profile_mirror_platforms(runtime: Optional[dict[str, Any]], profile: str) -> dict[str, Any]:
    """Entries a served ``profile`` mirrors from the DEFAULT (api_server/webhook listeners, and the
    Relay adapter). The multiplexer never builds those adapters for a secondary (``hermes_agent.gateway.run_adapters``
    skips them: ``SHARED_LISTENER_MIRROR_PLATFORMS``), so the record has no ``<profile>:api_server``
    entry and every reader fell through to ``pending_restart`` — "Restart needed" forever while
    ``/p/<profile>/v1/...`` answered. A listener is mirrored only while live, with the ``/p/<profile>``
    URL the client must call; Relay in every state, since the default's failure is the profile's too.
    """
    from hermes_agent.gateway.config import (
        SERVED_PROFILE_MIRROR_PLATFORMS, SHARED_LISTENER_MIRROR_PATHS, SHARED_LISTENER_MIRROR_PLATFORMS,
    )
    plats = (runtime or {}).get("platforms")
    if not profile or profile == "default" or not isinstance(plats, dict):
        return {}
    mirrored: dict[str, Any] = {}
    for name in sorted(SERVED_PROFILE_MIRROR_PLATFORMS):
        entry = plats.get(name)
        if not isinstance(entry, dict):
            continue
        # A dead default listener has no /p/<profile> surface to mirror. A dead default Relay IS the
        # profile's Relay state: its one adapter serves every profile, so its failure is theirs too.
        if name in SHARED_LISTENER_MIRROR_PLATFORMS and entry.get("state") not in {"connected", "connecting", "retrying"}:
            continue
        # api_server and webhook bind separate ports; each mirror hangs off its own listener. A record
        # from an older gateway carries no ``listener_base``: connected, URL unknown.
        base = entry.get("listener_base")
        url = f"{base}/p/{profile}{SHARED_LISTENER_MIRROR_PATHS.get(name, '')}" if isinstance(base, str) and base else None
        mirrored[name] = {k: v for k, v in entry.items() if k != "listener_base"}
        mirrored[name].update(ingress_url=url, mirrored_from="default")
    return mirrored
