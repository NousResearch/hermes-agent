"""Which daemon a named-profile client must start when nothing serves its home.

A ``<root>/profiles/<name>`` home that the default multiplexer serves has no daemon of its own.
When the multiplexer is down (restart, update, crash) the profile looks ``absent`` and the ensure
loop would spawn a per-profile gateway for it — a second owner that answers this client and then
blocks the multiplexer's next all-or-nothing reserve (``Cannot reserve gateway profiles``). The
client must start the multiplexer instead, so the start target is decided here from the same facts
the multiplexer boots on: the default profile's explicit ``gateway.multiplex_profiles`` flag, or —
when the flag is unset and settled at boot — the served set the last multiplexer recorded in the
root's ``gateway_state.json`` (a standalone default gateway clears it).
"""
from __future__ import annotations

from pathlib import Path


def multiplexer_root_for(home: Path) -> Path | None:
    """Default root that may multiplex *home*, when *home* is a named profile under it."""
    from hermes_constants import named_profile_home
    if named_profile_home(home) is None:
        return None
    root = home.parent.parent
    return root if root != home else None


def multiplexer_serves_home(home: Path) -> Path | None:
    """The root whose (possibly stopped) multiplexer serves *home*, else None.

    ``gateway.host_topology.named_profile_served_by_running_multiplexer`` answers the live question;
    this is its offline twin for the moment no gateway runs: an explicit ``true`` on the default
    profile, or a recorded ``served_profiles`` naming this profile (the boot-time verdict of an unset
    flag). An explicit ``false``, or no evidence, keeps the per-profile daemon.
    """
    root = multiplexer_root_for(home)
    if root is None:
        return None
    from gateway.multiplex_mode import explicit_multiplex_flag
    flag = explicit_multiplex_flag(root)
    if flag is True:
        return root
    if flag is False:
        return None
    from gateway.status import read_runtime_status
    from profiles.names import normalize_profile_name
    served = (read_runtime_status(root / "gateway_state.json") or {}).get("served_profiles")
    if not isinstance(served, list):
        return None
    names = {normalize_profile_name(str(p)) for p in served if p}
    return root if normalize_profile_name(home.name) in names else None
