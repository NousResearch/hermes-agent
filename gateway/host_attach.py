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
* ``REPLACE_HOST`` — ``--replace`` targets the host process when it serves this profile, whichever
  home launched it. An owner that has not published its served set yet is targeted too, but the
  ownership guard (``run._replace_target_belongs_to_other_profile``) can then only prove it from
  THIS home's pid record, so from another home the replace is refused (exit 1).
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
    it serves its own profile and nothing else, which is not the same as "unknown".

    The unified runtime publishes each served profile as ``{"profile_id", "home"}`` (what
    ``hermes_cli.gateway_runtime`` matches homes against); older gateways published bare names.
    Both spell the same roster, so a dict entry is reduced to its profile name here."""
    served = identity.get("served_profiles")
    if isinstance(served, list) and served:
        names = []
        for entry in served:
            if isinstance(entry, dict):
                home = entry.get("home") or entry.get("profile_id")
                names.append(profile_name_for_home(str(home)) if home else "default")
            else:
                names.append(str(entry))
        return tuple(names)
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
        f"   Take the host over (only from the home that launched it):  hermes gateway run --replace\n"
        f"   Start anyway:  hermes gateway run --force")


def _refuse_message(gateway: HostGateway, profile: str) -> str:
    from hermes_cli.gateway_migrate import MIGRATE_COMMAND

    return (
        f"❌ A gateway already owns this host and will not serve profile '{profile}'.\n"
        f"   {gateway.describe()}\n"
        f"   Exactly one gateway per host serves every profile, so starting a second one\n"
        f"   would double-bind this profile's platforms.\n"
        f"   Fold this profile into it:   {MIGRATE_COMMAND}\n"
        f"   Or start one anyway:         hermes gateway run --force\n"
        f"   (--replace only replaces an owner that serves this profile, so it would not take this one over.)")


def standalone_rescan_message(profile: str) -> str:
    return (
        f"The host gateway still serves profile '{profile}'; gateway.standalone is not live yet. "
        "Wait for the host gateway to rescan (<=30s), or send the rescan-profiles control verb "
        "to the host gateway before starting this profile's gateway.")


def _coexisting_gateways(owner: Optional[HostGateway]):
    """A standalone lock owner can hide a multiplexer launched beside it.

    Use the existing per-home liveness and control channels, not the single host
    record, to ask every running profile gateway what it actually serves.
    """
    from gateway.status import live_gateway_pid_for_home
    from hermes_cli.profiles import profiles_to_serve

    seen = {os.getpid()}
    if owner is not None:
        seen.add(owner.pid)
        yield owner
    for _name, home in profiles_to_serve(True, include_standalone=True, include_parked=True):
        pid = live_gateway_pid_for_home(home)
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        peer = HostGateway(pid, home, (), served_known=False)
        identity = _identify(home)
        if isinstance(identity, dict) and _identity_matches(identity, peer, home):
            peer = HostGateway(pid, home, _served_from_identity(identity),
                               standalone=identity.get("multiplex") is False)
        yield peer


def standalone_attach_decision(our_home: Path, owner: Optional[HostGateway]) -> Optional[HostAttachDecision]:
    """An opt-out permits coexistence only after every live gateway confirms we are unserved.

    Shared by the initial attach check and the lock-losing race check.
    """
    from hermes_cli.profiles import profile_is_standalone

    if not profile_is_standalone(our_home):
        return None
    profile = profile_name_for_home(our_home)
    for peer in _coexisting_gateways(owner):
        if not peer.served_known:
            return HostAttachDecision(REFUSE, _unknown_served_message(peer, profile), peer, transient=True)
        if peer.serves(profile):
            return HostAttachDecision(REFUSE, standalone_rescan_message(profile), peer, transient=True)
    logger.info("Profile '%s' is standalone by config; starting beside the host multiplexer", profile)
    return HostAttachDecision(START, "", owner)


def decide(our_home: Path, *, replace: bool = False) -> HostAttachDecision:
    """Attach, rescan-then-attach, replace or refuse; configured standalone profiles may coexist.

    Never raises: a broken probe degrades to ``START``, i.e. exactly the pre-rendezvous behaviour.
    """
    profile = profile_name_for_home(our_home)
    try:
        gateway = host_gateway()
    except Exception:
        logger.debug("host gateway probe failed; starting as before", exc_info=True)
        return HostAttachDecision(START, "")
    if gateway is None or gateway.pid == os.getpid():
        return standalone_attach_decision(our_home, None) or HostAttachDecision(START, "")
    if replace and (gateway.serves(profile) or not gateway.served_known):
        # An owner known not to serve us is another profile's gateway: replacing it is always refused
        # (_replace_target_belongs_to_other_profile fails closed) and the gateway exits, so on a
        # one-process-per-profile fleet, whose generated units all carry --replace, every unit but
        # the lock holder respawn-storms. Such an owner takes the non-replace path below instead.
        return HostAttachDecision(REPLACE_HOST, "", gateway)
    if gateway.served_known:
        standalone = standalone_attach_decision(our_home, gateway)
        if standalone is not None:
            return standalone
    if gateway.serves(profile):
        return HostAttachDecision(ATTACH, attach_message(gateway, profile), gateway, transient=True)
    if not gateway.served_known:
        # Give the owner its bounded window to answer before judging it: during the boot race the
        # record lands a moment before the control socket binds.
        waited = host_gateway(wait_for_channel=ATTACH_CHANNEL_WAIT_S)
        if waited is None:
            return HostAttachDecision(START, "")
        gateway = waited
        standalone = standalone_attach_decision(our_home, gateway)
        if standalone is not None:
            return standalone
        if gateway.serves(profile):
            return HostAttachDecision(ATTACH, attach_message(gateway, profile), gateway, transient=True)
    try:
        attached = request_serve_profile(profile, owner=gateway)
    except Exception:
        logger.debug("host gateway rescan request failed", exc_info=True)
        attached = None
    if attached is not None and attached.serves(profile):
        return HostAttachDecision(ATTACH, attach_message(attached, profile), attached, transient=True)
    if attached is not None and attached.standalone:
        # One-process-per-profile fleet: the owner is another profile's standalone gateway. Refusing
        # here exits 78, which every supervisor treats as permanent — on a launchd fleet that parked
        # every unit but the first to claim the host lock. Start beside it. decide() runs twice per
        # start (CLI guard + start_gateway), so this is INFO; the host-lock claim in run.py logs the
        # one WARNING with the `gateway migrate --multiplex` converge hint.
        from hermes_cli.gateway_migrate import MIGRATE_COMMAND

        logger.info(
            "Another profile's standalone gateway owns this host (%s); starting profile '%s' beside it. "
            "Fold every profile onto one gateway with: %s",
            attached.describe(), profile, MIGRATE_COMMAND)
        return HostAttachDecision(START, "")
    if not gateway.served_known:
        # The owner never answered, so we know only that it exists. ATTACH here (on the record's
        # word) parked a supervised unit against a served set nobody had committed to yet.
        return HostAttachDecision(
            REFUSE, _unknown_served_message(gateway, profile), gateway, transient=True)
    return HostAttachDecision(REFUSE, _refuse_message(gateway, profile), gateway)
