"""Who owns the gateway role on THIS host, and which profiles does that one process serve?

Multiplex-only (Teknium ruling): exactly ONE ``hermes gateway run`` per host, multiplexing every
profile. Every reporting surface (doctor, ``cron status``, ``claw``, the dashboard liveness ladder)
used to ask a per-PROFILE question instead — "does *my* profile own a gateway process?" — and a
profile that is SERVED by the host gateway answered "no". That produced three user-visible lies:
"not running", "Per-profile gateways: 0/3 up", and a silently skipped destructive-action warning.

:func:`host_gateway_topology` answers the real question once, from
:mod:`gateway.host_rendezvous` (the authoritative host record the winning process publishes),
falling back to the default home's recorded ``served_profiles`` for a gateway that predates the
record. ``default`` is just another served profile here, never a special owner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def _normalized(name: Optional[str]) -> str:
    if not name:
        return ""
    try:
        from hermes_cli.profiles import normalize_profile_name

        return normalize_profile_name(name)
    except Exception:
        return str(name).strip().lower()


@dataclass(frozen=True)
class HostGatewayTopology:
    """The single gateway process on this host plus the profile roster it multiplexes.

    ``source`` names the rung that answered (``host_record`` / ``served_record``) — for logs and
    tests only; never branch product behaviour on it.
    """

    pid: int
    profiles: tuple[str, ...]
    source: str

    def serves(self, profile_name: Optional[str]) -> bool:
        """True when this host process ticks/serves ``profile_name`` (``default`` included)."""
        wanted = _normalized(profile_name)
        return bool(wanted) and wanted in {_normalized(p) for p in self.profiles}

    def describe(self) -> str:
        """``the host gateway (PID 42) serving profiles default, coder`` — one shared phrasing so
        doctor, cron status and the state.db holder lines cannot drift apart."""
        roster = ", ".join(self.profiles) if self.profiles else "an unknown profile set"
        return f"the host gateway (PID {self.pid}) serving profiles {roster}"


def _from_host_record() -> Optional[HostGatewayTopology]:
    from gateway import host_rendezvous as hr

    record = hr.read_record(hr.ROLE_GATEWAY)  # already drops stale/PID-reused records
    if record is None or not record.pid:
        return None
    # A record whose (pid, createTime) cannot be POSITIVELY matched is a candidate, never an owner
    # (``host_rendezvous`` contract): reporting an unprovable record as the live host gateway would
    # turn a leftover record into a permanent "running" lie on every surface.
    if not hr.liveness_is_proven(record):
        return None
    return HostGatewayTopology(pid=int(record.pid), profiles=tuple(record.profiles), source="host_record")


def _from_served_record() -> Optional[HostGatewayTopology]:
    """A gateway started before the host record existed still publishes ``served_profiles`` into
    the default home's ``gateway_state.json``; that plus a proven-live PID is the same fact."""
    from hermes_cli.gateway_multiplex_served import live_default_gateway_pid, recorded_served_profiles

    pid = live_default_gateway_pid()
    if pid is None:
        return None
    served = recorded_served_profiles() or []
    roster = ["default"] + [str(p) for p in served if _normalized(p) != "default"]
    return HostGatewayTopology(pid=int(pid), profiles=tuple(roster), source="served_record")


def host_gateway_topology() -> Optional[HostGatewayTopology]:
    """The one live host gateway and its served profiles, or None when no gateway owns the role."""
    for rung in (_from_host_record, _from_served_record):
        try:
            topology = rung()
        except Exception:
            logger.debug("host gateway topology rung %s failed", rung.__name__, exc_info=True)
            continue
        if topology is not None:
            return topology
    return None


def host_gateway_serving(profile_name: Optional[str] = None) -> Optional[HostGatewayTopology]:
    """The host gateway when it serves ``profile_name`` (default: the active profile), else None."""
    topology = host_gateway_topology()
    if topology is None:
        return None
    if profile_name is None:
        try:
            from hermes_cli.profiles import get_active_profile_name

            profile_name = get_active_profile_name()
        except Exception:
            return None
    return topology if topology.serves(profile_name) else None
