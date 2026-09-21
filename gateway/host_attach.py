"""Is there ONE live host gateway, and does it already serve this profile?

Multiplex-only (Teknium ruling): exactly one ``hermes gateway run`` per host, multiplexing every
profile. The lifecycle verbs therefore answer a different question than they used to — not "does
THIS home hold a ``gateway.pid``?" but "is the host process live, and is this profile in its served
set?" — and when it is not, they ask that process to serve the profile instead of starting a second
one. Four outcomes, in order:

* ``ATTACH``       — a live host gateway already serves this profile. Nothing to start; exit 0.
* ``RESCAN``→ATTACH — it does not serve it yet: ask it to reconcile ``profiles/`` now (control
  socket ``rescan-profiles``) and attach once the answer includes us.
* ``REPLACE_HOST`` — ``--replace`` names the host process as the target, whichever home launched it.
* ``REFUSE``       — a live host gateway exists and cannot be made to serve this profile. Never
  start a second one silently.

**The attach channel is the OWNER's control socket, never ours.** Ordering matters: the owner
publishes its rendezvous record when it claims its PID file and binds its control socket a moment
later (``gateway/run.py``: claim → socket), so for a short window the record exists and the channel
does not. A reader that took "no socket" for "no owner" would start exactly the second gateway this
module prevents. Hence: the record's own ``profiles`` list answers ATTACH with no channel at all,
and only the RESCAN path needs the channel — it waits a bounded :data:`ATTACH_CHANNEL_WAIT_S` for it
to appear. Nothing here depends on the *calling* process having started anything.
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

    def serves(self, profile: str) -> bool:
        wanted = _normalize(profile)
        return any(_normalize(p) == wanted for p in self.profiles)

    @property
    def profile_label(self) -> str:
        return profile_name_for_home(self.home)

    def describe(self) -> str:
        served = ", ".join(self.profiles) if self.profiles else "unknown"
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


def host_gateway(*, wait_for_channel: float = 0.0) -> Optional[HostGateway]:
    """The one live host gateway, or ``None``.

    The served set comes from the owner's control socket when it answers (live and authoritative),
    else from the record it published (available from the PID claim onward).
    """
    from gateway import host_rendezvous as hr

    record = hr.read_record(hr.ROLE_GATEWAY)
    if record is None:
        return None
    home = _record_home(record)
    deadline = time.monotonic() + max(0.0, wait_for_channel)
    while True:
        identity = _identify(home)
        if isinstance(identity, dict) and identity.get("pid") == record.pid:
            return HostGateway(record.pid, home, _served_from_identity(identity))
        if time.monotonic() >= deadline:
            break
        time.sleep(_CHANNEL_POLL_S)
    # No channel yet (or ever): the record still proves an owner when PID + createTime match.
    if not hr.liveness_is_proven(record):
        return None
    return HostGateway(record.pid, home, tuple(record.profiles))


def host_gateway_serving(profile: str, *, wait_for_channel: float = 0.0) -> Optional[HostGateway]:
    """The host gateway when it is live AND serves ``profile`` — true for ``default`` too."""
    gateway = host_gateway(wait_for_channel=wait_for_channel)
    return gateway if gateway is not None and gateway.serves(profile) else None


def request_serve_profile(profile: str, *, timeout: float = 8.0) -> Optional[HostGateway]:
    """Ask the live host gateway to reconcile ``profiles/`` now; return it once it serves
    ``profile``. ``None`` when nobody answered or the answer still excludes the profile."""
    gateway = host_gateway(wait_for_channel=ATTACH_CHANNEL_WAIT_S)
    if gateway is None or gateway.serves(profile):
        return gateway
    try:
        from gateway.control_socket import rescan_gateway_profiles

        answer = rescan_gateway_profiles(gateway.home, timeout=timeout)
    except Exception:
        logger.debug("host gateway rescan failed", exc_info=True)
        return None
    if not isinstance(answer, dict) or answer.get("multiplex") is False:
        return None
    served = answer.get("served_profiles")
    rescanned = HostGateway(
        gateway.pid, gateway.home,
        tuple(str(p) for p in served) if isinstance(served, list) else ())
    return rescanned if rescanned.serves(profile) else None


@dataclass(frozen=True)
class HostAttachDecision:
    outcome: str
    message: str
    owner: Optional[HostGateway] = None


def attach_message(gateway: HostGateway, profile: str) -> str:
    return (
        f"✓ The host gateway already serves profile '{profile}' — nothing to start.\n"
        f"  {gateway.describe()}\n"
        f"  One gateway per host serves every profile; manage it with "
        f"`hermes -p {gateway.profile_label} gateway restart`.")


def _refuse_message(gateway: HostGateway, profile: str) -> str:
    return (
        f"❌ A gateway already owns this host and will not serve profile '{profile}'.\n"
        f"   {gateway.describe()}\n"
        f"   Exactly one gateway per host serves every profile, so starting a second one\n"
        f"   would double-bind this profile's platforms.\n"
        f"   Fold this profile into it:   hermes gateway migrate --multiplex\n"
        f"   Or take the host over:       hermes gateway run --replace")


def decide(our_home: Path, *, replace: bool = False) -> HostAttachDecision:
    """Attach, rescan-then-attach, replace or refuse — never a second gateway.

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
    if replace:
        # --replace is explicit authority over the host role; the target is the host process,
        # whichever home launched it.
        return HostAttachDecision(REPLACE_HOST, "", gateway)
    if gateway.serves(profile):
        return HostAttachDecision(ATTACH, attach_message(gateway, profile), gateway)
    try:
        attached = request_serve_profile(profile)
    except Exception:
        logger.debug("host gateway rescan request failed", exc_info=True)
        attached = None
    if attached is not None:
        return HostAttachDecision(ATTACH, attach_message(attached, profile), attached)
    return HostAttachDecision(REFUSE, _refuse_message(gateway, profile), gateway)
