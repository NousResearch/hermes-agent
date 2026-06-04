"""Transport selection seam for the Kanban external-coder lane.

This is the single decision point that answers: *should this worker invocation
use the opt-in ACP transport, or fall back to the existing PTY (``claude -p``)
lane?*  It is intentionally pure and dependency-free so it can be unit-tested
without importing ``acp`` or launching anything.

Decision rules (all opt-in, **disabled by default**):

1. Default -> :data:`TRANSPORT_PTY`.  With no env opt-in the caller keeps the
   current ``claude -p`` behaviour unchanged.
2. ``KANBAN_WORKER_TRANSPORT=acp`` requests the ACP transport.  It is only
   *honoured* when both:

   * the optional ``acp`` extra is importable, **and**
   * the explicit launch guard ``HERMES_ACP_ALLOW_LAUNCH=1`` is set.

3. If ACP is requested but cannot run, the decision is a **fallback** to PTY
   with a recorded ``refusal`` reason (safe degrade).  A caller that wants
   strict behaviour raises :class:`acp_client.AcpClientUnavailable` with that
   message instead of falling back.

Provenance: adapted from prototype ``72bd6be09`` (``acp_client/transport.py``,
task ``t_7514d8c1``) onto current ``main``.  The shape is unchanged; the only
adaptation is that :func:`acp_available` is resolved from the current ``main``
``acp_client`` package surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

TRANSPORT_PTY = "pty"
TRANSPORT_ACP = "acp"

TRANSPORT_ENV_VAR = "KANBAN_WORKER_TRANSPORT"
# Defence-in-depth: even if the transport env requests ACP, no external
# subprocess is launched unless this guard is explicitly set.  This keeps an
# accidental ``KANBAN_WORKER_TRANSPORT=acp`` in a shared profile from ever
# spawning a real ``claude``/``codex`` process.
LAUNCH_GUARD_ENV_VAR = "HERMES_ACP_ALLOW_LAUNCH"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ACP_ALIASES = frozenset({"acp", "acp-client", "acp_client"})


@dataclass(frozen=True)
class TransportDecision:
    """Outcome of resolving which lane transport to use.

    Attributes:
        transport: ``"pty"`` or ``"acp"`` — the lane the caller should run.
        requested: The raw transport the environment asked for (may differ from
            ``transport`` when a requested ACP run safely fell back to PTY).
        reason: Human-readable explanation, suitable for a log line / writeback.
        refusal: Set when ACP was requested but cannot run; ``None`` otherwise.
            A caller that wants strict behaviour raises with this message
            instead of falling back.
    """

    transport: str
    requested: str
    reason: str
    refusal: Optional[str] = None

    @property
    def is_acp(self) -> bool:
        return self.transport == TRANSPORT_ACP

    @property
    def fell_back(self) -> bool:
        return self.requested == TRANSPORT_ACP and self.transport == TRANSPORT_PTY


def _normalize(value: Optional[str]) -> str:
    raw = (value or "").strip().lower()
    if raw in _ACP_ALIASES:
        return TRANSPORT_ACP
    # Treat everything else (incl. empty, "pty", "claude", "default") as PTY.
    return TRANSPORT_PTY


def resolve_transport(
    env: Optional[Mapping[str, str]] = None,
    *,
    acp_available_fn=None,
) -> TransportDecision:
    """Resolve the lane transport from environment, defaulting to PTY.

    Args:
        env: Environment mapping to read (defaults to ``os.environ``).
        acp_available_fn: Injectable predicate returning whether the ``acp``
            extra is importable.  Defaults to
            :func:`acp_client.acp_available`.  Injectable so tests never need
            the real dependency.

    Returns:
        A :class:`TransportDecision`.  Never raises — refusal is carried as data
        so the caller chooses fallback-vs-refuse.
    """
    env = os.environ if env is None else env
    requested = _normalize(env.get(TRANSPORT_ENV_VAR))

    if requested != TRANSPORT_ACP:
        return TransportDecision(
            transport=TRANSPORT_PTY,
            requested=TRANSPORT_PTY,
            reason=(
                f"{TRANSPORT_ENV_VAR} not set to 'acp'; using default PTY lane "
                "(claude -p)."
            ),
        )

    if acp_available_fn is None:
        from acp_client import acp_available as acp_available_fn  # lazy import

    if not acp_available_fn():
        msg = (
            "ACP transport requested but the optional 'acp' "
            "(agent-client-protocol) extra is not installed. Install with "
            "'pip install hermes-agent[acp]' or unset "
            f"{TRANSPORT_ENV_VAR}. Falling back to the default PTY lane."
        )
        return TransportDecision(
            transport=TRANSPORT_PTY,
            requested=TRANSPORT_ACP,
            reason=msg,
            refusal=msg,
        )

    guard = (env.get(LAUNCH_GUARD_ENV_VAR) or "").strip().lower()
    if guard not in _TRUTHY:
        msg = (
            f"ACP transport requested but launch guard {LAUNCH_GUARD_ENV_VAR} is "
            f"not enabled. Set {LAUNCH_GUARD_ENV_VAR}=1 to permit launching an "
            "external ACP agent. Falling back to the default PTY lane."
        )
        return TransportDecision(
            transport=TRANSPORT_PTY,
            requested=TRANSPORT_ACP,
            reason=msg,
            refusal=msg,
        )

    return TransportDecision(
        transport=TRANSPORT_ACP,
        requested=TRANSPORT_ACP,
        reason="ACP transport enabled and launch guard set; using ACP lane.",
    )
