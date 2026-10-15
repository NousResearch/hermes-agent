"""Sibling of ``gateway/status.py`` for the multiplexer's platform-record projections.

Split out of ``status.py`` under its FILE_LINES ratchet (moved code keeps its cap; see root
AGENTS.md). Themes here read the ONE multiplexer record — ``gateway_state.json`` of the home
that launched it — and project its entries for a served profile.
"""

from __future__ import annotations

from typing import Any, Optional


def shared_listener_mirror_platforms(runtime: Optional[dict[str, Any]], profile: str) -> dict[str, Any]:
    """Entries for the api_server/webhook mirrors a served ``profile`` gets from the DEFAULT's
    listener. The multiplexer never builds those adapters for a secondary (``gateway.run_adapters``
    skips them: ``SHARED_LISTENER_MIRROR_PLATFORMS``), so the record has no ``<profile>:api_server``
    entry and every reader fell through to ``pending_restart`` — "Restart needed" forever while
    ``/p/<profile>/v1/...`` answered. Only a live default entry is mirrored; its state is the profile's
    state, plus the ``/p/<profile>`` URL the client must actually call.
    """
    from gateway.config import SHARED_LISTENER_MIRROR_PATHS, SHARED_LISTENER_MIRROR_PLATFORMS
    plats = (runtime or {}).get("platforms")
    if not profile or profile == "default" or not isinstance(plats, dict):
        return {}
    mirrored: dict[str, Any] = {}
    for name in sorted(SHARED_LISTENER_MIRROR_PLATFORMS):
        entry = plats.get(name)
        if not isinstance(entry, dict) or entry.get("state") not in {"connected", "connecting", "retrying"}:
            continue
        # api_server and webhook bind separate ports; each mirror hangs off its own listener. A record
        # from an older gateway carries no ``listener_base``: connected, URL unknown.
        base = entry.get("listener_base")
        url = f"{base}/p/{profile}{SHARED_LISTENER_MIRROR_PATHS.get(name, '')}" if isinstance(base, str) and base else None
        mirrored[name] = {k: v for k, v in entry.items() if k != "listener_base"}
        mirrored[name].update(ingress_url=url, mirrored_from="default")
    return mirrored
