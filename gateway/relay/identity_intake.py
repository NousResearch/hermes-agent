"""Hosted relay intake: the identity verifier admits every inbound frame before anything runs.

On a hosted process one socket carries many tenants' agents, and a frame's body asserts nothing
this process may trust — so the transport hands each raw inbound frame to the configured
verifier (``hosted.identity_verifier``) exactly once, before the frame is decoded or handled:

- a refusal drops the frame: counted and logged by its bounded reason slug, acknowledged like an
  unknown agent (a replay would be refused again), never routed to a default tenant or agent, and
  the response is the same bytes for an unknown agent and a foreign one;
- the verified identity is what the frame's whole turn runs under: the transport binds it as the
  bound scope around the frame's handling, and the turn admission that claims the turn keeps the
  same tenant, agent, actor, owner and administration right (``VerifiedTurn.turn_scope``).

A local install never consults a verifier: :func:`verify_inbound_frame` answers ``None`` and the
frame runs exactly as it always did.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Union

from hermes_identity_verifier import BAD_PAYLOAD, Refusal, VerifiedTurn, resolve_identity_verifier
from hermes_scope import LOCAL_TENANT
from hermes_scope_cache import tenant_neutral

logger = logging.getLogger(__name__)

# The verifier this process serves, resolved once from the deployment's config: identical for
# every tenant (an operator names one connector-side scheme for the whole process), never a
# fallback. The per-reason drop counters are bounded the same way: one count per slug.
_RESOLVED = tenant_neutral({"verifier": None}, reason="the identity verifier is named once per process by deployment config")
_DROPS = tenant_neutral({}, reason="per-reason drop counts are the deployment's own intake signal, identical for every tenant")


def _hosted_verifier():
    """The verifier configured for this process (``hosted.identity_verifier``), resolved once."""
    if _RESOLVED["verifier"] is None:
        from hermes_identity_verifier import identity_verifier_name
        _RESOLVED["verifier"] = resolve_identity_verifier(identity_verifier_name())
    return _RESOLVED["verifier"]


def reset_intake_state() -> None:
    """Forget the resolved verifier and the counters (tests reconfigure the deployment)."""
    _RESOLVED["verifier"] = None
    _DROPS.clear()


async def verify_inbound_frame(frame: Mapping[str, Any]) -> Optional[VerifiedTurn | Refusal]:
    """Verify one raw relay inbound frame. ``None`` when this process is not hosted: no verifier
    is consulted and the frame runs exactly as it always did."""
    from hermes_cli.hosted_mode import is_hosted

    if not is_hosted():
        return None
    return await _hosted_verifier().verify(frame)


def admit_verified(verdict: Optional[VerifiedTurn | Refusal]) -> Optional[VerifiedTurn]:
    """The turn a verified frame opens, or ``None`` when the frame is dropped. A refusal is
    counted and logged by its bounded reason slug; ``None`` input is a verifier failure the caller
    has already logged. A hosted process serves no local tenant (its agents live behind the
    storage ports, not a home directory), so a claim naming one is refused; body fields
    asserting a tenant never reach here at all."""
    if verdict is None:
        return None
    if isinstance(verdict, VerifiedTurn):
        if verdict.tenant == LOCAL_TENANT:
            note_drop(BAD_PAYLOAD)
            return None
        return verdict
    note_drop(verdict.reason)
    return None


def note_drop(reason: str) -> None:
    """Count and log one dropped frame by its bounded reason slug, the same way an unknown agent
    is refused: acknowledged, never routed anywhere."""
    _DROPS[reason] = int(_DROPS.get(reason, 0)) + 1
    logger.warning("relay inbound refused by the identity verifier (%s); acknowledged, not replayed",
                   reason)


UNDECODABLE_EVENT = "undecodable_event"


def note_undecodable_event() -> None:
    """Count and log one frame whose event could not be decoded into Hermes types after a valid
    claim: dropped and acknowledged like any refusal, and the reader keeps reading (a replay
    would fail the same way). Bounded and content-free, like every other drop count."""
    _DROPS[UNDECODABLE_EVENT] = int(_DROPS.get(UNDECODABLE_EVENT, 0)) + 1
    logger.warning("relay inbound refused after a verified claim: its event could not be decoded "
                   "(%s); acknowledged, not replayed", UNDECODABLE_EVENT)


def drop_counts() -> dict[str, int]:
    """The per-reason drop counts so far (a deployment's own intake signal; a read, not a reset)."""
    return {reason: int(count) for reason, count in _DROPS.items()}


def bind_verified(turn: VerifiedTurn):
    """The scope the frame's whole turn runs under: the verified identity, bound with no home,
    secrets or terminal policy (a hosted scope's state lives behind the storage ports)."""
    from agent.runtime_scope import bind_state

    return bind_state(turn.scope())
