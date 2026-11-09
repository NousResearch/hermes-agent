"""Inbound identity verification port: who one relay-delivered frame is for.

A shared (hosted) process serves many tenants' agents over one relay socket, and a frame's body
asserts nothing this process may trust: the connector that owns the bots proves, per delivered
inbound frame, who the frame is for — a signed claim naming the tenant, the agent, the acting
principal, the agent's owner and whether that principal may administer the agent. The verifier
plugin checks that claim and returns what it proved; the relay intake binds it as the scope of
the frame's whole turn, and every body field that asserts a tenant, principal, owner,
administration right or agent is ignored — only the verified claim counts.

The verifier, not core, owns the signature scheme and the keys: core defines this interface and
resolves a plugin by explicit name (``hosted.identity_verifier``), the same shape
``hermes_state_port`` uses for session storage — bundled ``plugins/identity_verifier/<name>/``
first, then ``$HERMES_HOME/plugins/<name>/``, in-process only (a user provider is refused under
``plugins.isolation: host``), and a name that is missing, refused or unavailable is an error,
never a fallback to another verifier. A local install consults no verifier at all.

:func:`verify` is async because checking a claim may need keys fetched over the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

from hermes_scope import Fence, Group, Principal, Scope, TurnScope

# Every reason a verifier refuses a frame. Bounded and content-free: a refusal is counted and
# logged by its slug and says nothing about tenants, agents or principals.
NO_CLAIM = "no_claim"
BAD_SIGNATURE = "bad_signature"
BAD_HEADER = "bad_header"
UNKNOWN_KID = "unknown_kid"
EXPIRED = "expired"
NOT_YET_VALID = "not_yet_valid"
WRONG_AUDIENCE = "wrong_audience"
REPLAYED = "replayed"
BAD_PAYLOAD = "bad_payload"
EVENT_MISMATCH = "event_mismatch"

REFUSAL_REASONS = frozenset({
    NO_CLAIM, BAD_SIGNATURE, BAD_HEADER, UNKNOWN_KID, EXPIRED, NOT_YET_VALID,
    WRONG_AUDIENCE, REPLAYED, BAD_PAYLOAD, EVENT_MISMATCH,
})


@dataclass(frozen=True)
class Refusal:
    """A frame this process must not serve. *reason* is one of :data:`REFUSAL_REASONS`; the
    intake drops the frame (counted and logged by it), never routing it to a default tenant
    or agent, with the same response as an unknown agent."""

    reason: str

    def __post_init__(self) -> None:
        if self.reason not in REFUSAL_REASONS:
            raise ValueError(f"unknown refusal reason {self.reason!r}")


@dataclass(frozen=True)
class VerifiedTurn:
    """What the signed claim proved for one delivered inbound frame: the identity of the frame's
    whole turn. :meth:`scope` is what the intake binds while the frame runs; :meth:`turn_scope`
    is the same identity as the turn the frame opens, once admission has claimed it (the claim
    carries no session, turn or lease: those are Hermes's own, and the fence is issued by turn
    admission, never by a signature)."""

    tenant: str
    agent: str
    actor: Principal
    owner: Principal | Group
    can_administer: bool

    def scope(self) -> Scope:
        """The bound form: a hosted scope naming the owner and the administration right the
        claim granted (it has no home directory; its state lives behind the storage ports)."""
        return Scope(self.tenant, self.agent, self.actor, owner=self.owner,
                     can_administer=self.can_administer)

    def turn_scope(self, session: str, turn: str, fence: Fence) -> TurnScope:
        """The turn the verified frame opens, once admission claims it under *fence*."""
        return TurnScope(self.tenant, self.agent, self.actor, session, turn, fence,
                         owner=self.owner, can_administer=self.can_administer)


class IdentityVerifier(ABC):
    """The plugin kind: checks a delivered inbound frame's signed claim. ``name`` and ``verify``
    are required; growth stays additive, like every provider ABC."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"turn_claim_v1"``."""

    def is_available(self) -> bool:
        """Whether this verifier can run here. No network calls; boot refuses an unavailable
        verifier before the process serves."""
        return True

    @abstractmethod
    async def verify(self, frame: Mapping[str, Any]) -> VerifiedTurn | Refusal:
        """Check one raw relay inbound frame (the whole frame dict, its ``claim`` key included)
        and return what the claim proved, or the refusal that drops it. Never raises for a frame
        it cannot admit: a verifier that cannot check returns a refusal, so no unverified frame
        is ever served."""


def resolve_identity_verifier(name: str) -> IdentityVerifier:
    """The verifier named *name*. Raises ``LookupError`` with the reason for a name that is
    missing, refused or unavailable: a hosted frame is admitted by the named verifier or by
    nobody, never by a fallback. There is no local verifier — a local install consults none."""
    name = (name or "").strip()
    if not name:
        raise LookupError("no identity verifier is configured")
    from plugins.identity_verifier import IdentityVerifierRefused, load_identity_verifier
    try:
        verifier = load_identity_verifier(name)
    except IdentityVerifierRefused as refused:
        raise LookupError(f"identity verifier {name!r}: {refused}") from refused
    if verifier is None:
        raise LookupError(f"identity verifier {name!r}: no such identity verifier")
    if not verifier.is_available():
        raise LookupError(f"identity verifier {name!r} is not available here")
    return verifier


def identity_verifier_name(config: Optional[Mapping[str, Any]] = None) -> str:
    """The configured verifier's name (``hosted.identity_verifier``), straight from the effective
    config when *config* is ``None``. Empty when unset: the caller decides what that means (boot
    refuses; nothing falls back)."""
    if config is None:
        from hermes_cli.config_effective import load_user_config_effective
        config = load_user_config_effective()
    hosted = config.get("hosted") if isinstance(config, Mapping) else None
    if not isinstance(hosted, Mapping):
        return ""
    return str(hosted.get("identity_verifier") or "").strip()
