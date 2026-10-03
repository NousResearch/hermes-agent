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

from dataclasses import dataclass
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _normalized(name: Optional[str]) -> str:
    if not name:
        return ""
    from profiles.names import normalize_profile_name

    return normalize_profile_name(name)


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
    # turn a leftover record into a permanent "running" lie on every surface. A record with no
    # recorded createTime is exactly that — ``_same_incarnation`` treats ``None`` as "matches", so
    # liveness_is_proven() would bless ANY process that happens to hold the recorded PID today.
    if record.create_time is None or not hr.liveness_is_proven(record):
        return None
    return HostGatewayTopology(pid=int(record.pid), profiles=tuple(record.profiles), source="host_record")


def _from_served_record() -> Optional[HostGatewayTopology]:
    """A gateway started before the host record existed still publishes ``served_profiles`` into
    the default home's ``gateway_state.json``; that plus a proven-live PID is the same fact."""
    from gateway.served_profiles import live_default_gateway_pid, recorded_served_profiles

    pid = live_default_gateway_pid()
    if pid is None:
        return None
    served = recorded_served_profiles() or []
    roster = ["default"] + [str(p) for p in served if _normalized(p) != "default"]
    return HostGatewayTopology(pid=int(pid), profiles=tuple(roster), source="served_record")


def host_gateway_topology() -> Optional[HostGatewayTopology]:
    """The one live host gateway and its served profiles, or None when no gateway owns the role."""
    for rung in (_from_host_record, _from_served_record):
        topology = rung()
        if topology is not None:
            return topology
    return None


def host_gateway_serving(profile_name: Optional[str] = None) -> Optional[HostGatewayTopology]:
    """The host gateway when it serves ``profile_name`` (default: the active profile), else None."""
    topology = host_gateway_topology()
    if topology is None:
        return None
    if profile_name is None:
        from profiles.current import get_active_profile_name

        profile_name = get_active_profile_name()
    return topology if topology.serves(profile_name) else None


def _current_profile_name() -> str | None:
    from hermes_constants import get_hermes_home, profile_name_for_home

    return profile_name_for_home(get_hermes_home())


def host_multiplexer_serving(profile_name: str | None = None):
    """The live host owner when it serves ``profile_name`` using the control-channel roster."""
    try:
        from gateway.host_attach import host_gateway_serving as live_host_gateway_serving

        name = profile_name if profile_name is not None else _current_profile_name()
        return live_host_gateway_serving(name) if name else None
    except Exception:
        logger.debug("Host multiplexer probe failed", exc_info=True)
        return None


def served_by_another_host_gateway(profile_name: str | None = None):
    """The host gateway serving ``profile_name`` when it is not this home\'s own process."""
    gateway = host_multiplexer_serving(profile_name)
    if gateway is None:
        return None
    try:
        from gateway.status import _get_process_hermes_home, _same_hermes_home

        if _same_hermes_home(gateway.home, _get_process_hermes_home()):
            return None
    except Exception:
        logger.debug("Host multiplexer home comparison failed", exc_info=True)
    return gateway


def named_profile_served_by_running_multiplexer(profile_name: str | None = None) -> bool:
    """True when the live host multiplexer serves the requested named profile."""
    try:
        suffix = profile_name if profile_name is not None else _current_profile_name()
    except Exception:
        return False
    if not suffix or suffix == "default":
        return False

    if host_multiplexer_serving(suffix) is not None:
        return True

    try:
        from hermes_constants import get_default_hermes_root

        default_root = get_default_hermes_root()
    except Exception:
        return False

    try:
        from gateway.served_profiles import live_default_gateway_pid, recorded_served_profiles
        from profiles.names import normalize_profile_name

        if live_default_gateway_pid() is None:
            return False
        recorded = recorded_served_profiles(default_root)
        if recorded is not None:
            return normalize_profile_name(suffix) in {
                normalize_profile_name(profile) for profile in recorded
            }

        from gateway.multiplex_mode import explicit_multiplex_flag

        return explicit_multiplex_flag(default_root) is True
    except Exception:
        logger.debug("Multiplexer-serving probe failed", exc_info=True)
        return False
