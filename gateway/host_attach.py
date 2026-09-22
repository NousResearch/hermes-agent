"""Is there ONE live host gateway, and does it already serve this profile?

Multiplex-only (Teknium ruling): exactly one MULTIPLEXING ``hermes gateway run`` per host, serving
every profile; standalone per-profile gateways coexist until that migration is forced (#109417).
The lifecycle verbs therefore answer a different question than they used to — not "does THIS home
hold a ``gateway.pid``?" but "is the host process live, and is this profile in its served set?" —
and when it is not, they ask that process to serve the profile instead of starting a second one.
Five outcomes, in order:

* ``ATTACH``       — a live host gateway already serves this profile. Nothing to start; exit 0.
* ``RESCAN``→ATTACH — it does not serve it yet: ask it to reconcile ``profiles/`` now (control
  socket ``rescan-profiles``) and attach once the answer includes us.
* ``REPLACE_HOST`` — ``--replace`` names the host process as the target, whichever home launched it.
* ``REFUSE``       — a live MULTIPLEXING gateway exists and cannot be made to serve this profile.
  Never start a second one silently.
* ``START``        — no live owner, or the owner answers ``multiplex: False``: it is another
  profile's standalone gateway (the documented one-process-per-profile topology), not a
  multiplexer that excluded us, so this profile runs its own gateway beside it as it always did.

**The attach channel is the OWNER's control socket, never ours.** Ordering matters: the owner
publishes its rendezvous record when it claims its PID file and binds its control socket a moment
later (``gateway/run.py``: claim → socket), so for a short window the record exists and the channel
does not. A reader that took "no socket" for "no owner" would start exactly the second gateway this
module prevents — but a reader that took the RECORD's word for the served set is worse: the
claim-time record is published before the process knows what it will serve, so a supervised unit
for a profile nobody serves would stand down forever. Hence the split:

* the record proves an OWNER exists (PID + createTime), and that alone never yields ATTACH;
* the served set comes ONLY from a live ``identify`` answer, waited for a bounded
  :data:`ATTACH_CHANNEL_WAIT_S`;
* owner present + served set unknown is a TRANSIENT verdict — do not start, do not park.

Nothing here depends on the *calling* process having started anything.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: How long a caller waits for the owner's control socket after seeing its record (see module doc).
ATTACH_CHANNEL_WAIT_S = 5.0
_CHANNEL_POLL_S = 0.25

START = "start"
ATTACH = "attach"
REFUSE = "refuse"
REPLACE_HOST = "replace-host"


def _normalize(name: str) -> str:
    try:
        from hermes_cli.profiles import normalize_profile_name

        return normalize_profile_name(name or "default")
    except Exception:
        return (name or "default").strip().lower()


def profile_name_for_home(home: Path | str) -> str:
    """Profile a home belongs to; the root/default home is ``'default'`` (not ``None``)."""
    from gateway.status import _profile_name_for_home

    return _profile_name_for_home(Path(home)) or "default"


@dataclass(frozen=True)
class HostGateway:
    """The one live host gateway: who it is, where it was launched from, what it serves."""

    pid: int
    home: Path
    profiles: tuple[str, ...]
    #: False when the owner has not answered ``identify`` yet: an owner exists, but which profiles
    #: it serves is UNKNOWN. Never conflate that with "serves nothing" — see the module doc.
    served_known: bool = True
    #: True once the owner has said ``multiplex: False``: a per-profile gateway that cannot be asked
    #: to serve anyone else — see ``START`` in the module doc.
    standalone: bool = False

    def serves(self, profile: str) -> bool:
        if not self.served_known:
            return False
        wanted = _normalize(profile)
        return any(_normalize(p) == wanted for p in self.profiles)

    @property
    def profile_label(self) -> str:
        return profile_name_for_home(self.home)

    def describe(self) -> str:
        if not self.served_known:
            served = "not published yet (its control socket has not answered)"
        else:
            served = ", ".join(self.profiles) if self.profiles else "nothing"
        return f"PID {self.pid} (launched by profile '{self.profile_label}'; serves: {served})"


def _record_home(record) -> Path:
    """Home the owner was launched from. Records written before the field existed fall back to the
    default root — the home every pre-record multiplexer ran under."""
    from hermes_constants import get_default_hermes_root

    return Path(record.home) if getattr(record, "home", "") else Path(get_default_hermes_root())


def _identify(home: Path) -> Optional[dict]:
    try:
        from gateway.control_socket import identify_gateway

        return identify_gateway(home)
    except Exception:
        logger.debug("host gateway identify failed for %s", home, exc_info=True)
        return None


def _served_from_identity(identity: dict) -> tuple[str, ...]:
    """Served set from a live ``identify``. A STANDALONE gateway publishes no ``served_profiles``;
    it serves its own profile and nothing else, which is not the same as "unknown"."""
    served = identity.get("served_profiles")
    if isinstance(served, list) and served:
        return tuple(str(p) for p in served)
    return (str(identity.get("profile") or "default"),)


def _identity_matches(identity, record, home: Path) -> bool:
    """Is this ``identify`` answer really the record's owner?

    PID alone is not enough: a record naming an arbitrary home makes us dial whatever listens
    there, so the answer must also agree about the home it was launched from.
    """
    if not isinstance(identity, dict) or identity.get("pid") != record.pid:
        return False
    reported = identity.get("hermes_home")
    if not reported:
        return True  # older gateway: PID + a socket keyed by this home is all it can prove
    try:
        from gateway.status import _same_hermes_home

        return bool(_same_hermes_home(Path(str(reported)), home))
    except Exception:
        return str(reported) == str(home)


#: A CLI invocation asks this question once per profile (``gateway status`` across N profiles,
#: doctor, the lifecycle guards); a gateway PROCESS asks it for the life of the process, so the
#: memo is time-bounded rather than permanent. Writes invalidate it eagerly.
HOST_GATEWAY_CACHE_TTL_S = 2.0
_cached_probe: Optional[tuple[float, Optional[HostGateway]]] = None


def invalidate_host_gateway_cache() -> None:
    """Forget the memoized probe (called by ``host_rendezvous`` on every record write)."""
    global _cached_probe
    _cached_probe = None


def _probe_host_gateway(wait_for_channel: float) -> Optional[HostGateway]:
    from gateway import host_rendezvous as hr

    record = hr.read_record(hr.ROLE_GATEWAY)
    if record is None:
        return None
    # Liveness BEFORE the dial. A record we cannot prove live must not make us open a socket at an
    # address it chose; proving the PID first is also what keeps a stale record from naming a peer.
    if not hr.liveness_is_proven(record):
        return None
    home = _record_home(record)
    deadline = time.monotonic() + max(0.0, wait_for_channel)
    while True:
        identity = _identify(home)
        if _identity_matches(identity, record, home):
            return HostGateway(record.pid, home, _served_from_identity(identity))
        if time.monotonic() >= deadline:
            break
        time.sleep(_CHANNEL_POLL_S)
    # An owner exists and has not answered: the served set is UNKNOWN, never the record's word.
    return HostGateway(record.pid, home, (), served_known=False)


def host_gateway(*, wait_for_channel: float = 0.0) -> Optional[HostGateway]:
    """The one live host gateway, or ``None``.

    The served set comes from the owner's control socket and nowhere else; a record with no live
    answer behind it yields ``served_known=False`` — an owner whose served set nobody knows yet.
    """
    global _cached_probe
    now = time.monotonic()
    if wait_for_channel <= 0 and _cached_probe is not None and now - _cached_probe[0] < HOST_GATEWAY_CACHE_TTL_S:
        return _cached_probe[1]
    result = _probe_host_gateway(wait_for_channel)
    _cached_probe = (time.monotonic(), result)
    return result


def host_gateway_serving(profile: str, *, wait_for_channel: float = 0.0) -> Optional[HostGateway]:
    """The host gateway when it is live AND serves ``profile`` — true for ``default`` too."""
    gateway = host_gateway(wait_for_channel=wait_for_channel)
    return gateway if gateway is not None and gateway.serves(profile) else None


def request_serve_profile(profile: str, *, timeout: float = 8.0,
                          owner: Optional[HostGateway] = None) -> Optional[HostGateway]:
    """Ask the live host gateway to reconcile ``profiles/`` now; return it once it serves
    ``profile``. ``None`` when nobody answered or a multiplexer's roster still excludes the profile.
    An owner that answers ``multiplex: False`` comes back flagged ``standalone``: it cannot take the
    profile, and it is not a multiplexer that refused — the caller runs beside it, as before."""
    gateway = owner if owner is not None else host_gateway(wait_for_channel=ATTACH_CHANNEL_WAIT_S)
    if gateway is None or gateway.serves(profile):
        return gateway
    try:
        from gateway.control_socket import rescan_gateway_profiles

        answer = rescan_gateway_profiles(gateway.home, timeout=timeout)
    except Exception:
        logger.debug("host gateway rescan failed", exc_info=True)
        return None
    if not isinstance(answer, dict):
        return None
    served = answer.get("served_profiles")
    rescanned = HostGateway(
        gateway.pid, gateway.home,
        tuple(str(p) for p in served) if isinstance(served, list) else (),
        standalone=answer.get("multiplex") is False)
    return rescanned if rescanned.standalone or rescanned.serves(profile) else None


@dataclass(frozen=True)
class HostAttachDecision:
    outcome: str
    message: str
    owner: Optional[HostGateway] = None
    #: True when the verdict is a RUNTIME observation ("someone else serves me right now", "the
    #: owner has not answered yet") rather than a config-derived permanent refusal. A supervisor
    #: must RETRY a transient verdict; parking the unit on one strands the profile forever.
    transient: bool = False


def attach_message(gateway: HostGateway, profile: str) -> str:
    return (
        f"✓ The host gateway already serves profile '{profile}' — nothing to start.\n"
        f"  {gateway.describe()}\n"
        f"  One gateway per host serves every profile; manage it with "
        f"`hermes -p {gateway.profile_label} gateway restart`.")


def _unknown_served_message(gateway: HostGateway, profile: str) -> str:
    return (
        f"⏳ A gateway already owns this host and has not published its served set yet.\n"
        f"   {gateway.describe()}\n"
        f"   Whether it will serve profile '{profile}' is unknown, so starting a second gateway\n"
        f"   now could double-bind this profile's platforms. Nothing was started; this is a\n"
        f"   transient state and a service supervisor will retry.\n"
        f"   Take the host over:  hermes gateway run --replace\n"
        f"   Start anyway:        hermes gateway run --force")


def _refuse_message(gateway: HostGateway, profile: str) -> str:
    return (
        f"❌ A gateway already owns this host and will not serve profile '{profile}'.\n"
        f"   {gateway.describe()}\n"
        f"   Exactly one gateway per host serves every profile, so starting a second one\n"
        f"   would double-bind this profile's platforms.\n"
        f"   Fold this profile into it:   hermes gateway migrate --multiplex\n"
        f"   Or take the host over:       hermes gateway run --replace\n"
        f"   Or start one anyway:         hermes gateway run --force")


def decide(our_home: Path, *, replace: bool = False) -> HostAttachDecision:
    """Attach, rescan-then-attach, replace or refuse — never a second gateway beside a multiplexer.

    Never raises: a broken probe degrades to ``START``, i.e. exactly the pre-rendezvous behaviour.
    """
    profile = profile_name_for_home(our_home)
    try:
        gateway = host_gateway()
    except Exception:
        logger.debug("host gateway probe failed; starting as before", exc_info=True)
        return HostAttachDecision(START, "")
    if gateway is None or gateway.pid == os.getpid():
        return HostAttachDecision(START, "")
    if gateway.serves(profile):
        if replace:
            return HostAttachDecision(REPLACE_HOST, "", gateway)
        return HostAttachDecision(ATTACH, attach_message(gateway, profile), gateway, transient=True)
    if not gateway.served_known:
        # Give the owner its bounded window to answer before judging it: during the boot race the
        # record lands a moment before the control socket binds.
        waited = host_gateway(wait_for_channel=ATTACH_CHANNEL_WAIT_S)
        if waited is None:
            return HostAttachDecision(START, "")
        gateway = waited
        if gateway.serves(profile):
            return HostAttachDecision(ATTACH, attach_message(gateway, profile), gateway, transient=True)
    try:
        attached = request_serve_profile(profile, owner=gateway)
    except Exception:
        logger.debug("host gateway rescan request failed", exc_info=True)
        attached = None
    if attached is not None and attached.serves(profile):
        if replace:
            return HostAttachDecision(REPLACE_HOST, "", attached)
        return HostAttachDecision(ATTACH, attach_message(attached, profile), attached, transient=True)
    if attached is not None and attached.standalone:
        # One-process-per-profile fleet: the owner is another profile's standalone gateway. Refusing
        # here exits 78, which every supervisor treats as permanent — on a launchd fleet that parked
        # every unit but the first to claim the host lock. Start beside it; the host-lock claim logs
        # the topology and the `gateway migrate --multiplex` path stays the way to converge.
        logger.warning(
            "Another profile's standalone gateway owns this host (%s); starting profile '%s' beside it%s. "
            "Fold every profile onto one gateway with: hermes gateway migrate --multiplex",
            attached.describe(), profile,
            " without replacing the other profile" if replace else "")
        return HostAttachDecision(START, "")
    if replace:
        # The owner is a multiplexer (or did not answer): explicit takeover still targets the host
        # role. Only a positively identified cross-profile standalone owner degrades to START.
        return HostAttachDecision(REPLACE_HOST, "", gateway)
    if not gateway.served_known:
        # The owner never answered, so we know only that it exists. ATTACH here (on the record's
        # word) parked a supervised unit against a served set nobody had committed to yet.
        return HostAttachDecision(
            REFUSE, _unknown_served_message(gateway, profile), gateway, transient=True)
    return HostAttachDecision(REFUSE, _refuse_message(gateway, profile), gateway)
