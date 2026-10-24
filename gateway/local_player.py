"""Turn-bound plugin player routing for the canonical Desktop gateway."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class _Lease:
    turn: Any
    agent: Any
    live: Any
    policy: Any
    active: bool = True


_current: ContextVar[_Lease | None] = ContextVar("desktop_player_turn", default=None)


def bind(turn, agent):
    """Capture the exact GUI turn; copied tool contexts share its revocable lease."""
    from gateway.session_policy import active_policy
    authority, session_id, _generation = turn._approval_owner
    lease = _Lease(turn, agent, authority.sessions.get(session_id), active_policy())
    return lease, _current.set(lease)


def release(token):
    lease, context_token = token
    lease.active = False
    _current.reset(context_token)


def _live(lease, durable_sid):
    from gateway.session_policy import active_policy
    authority, session_id, generation = lease.turn._approval_owner
    adopted_generation, adopted_agent = authority.adopted.get(session_id, (None, None))
    return (
        lease.active and lease.live is not None and type(generation) is int
        and lease.policy is not None and lease.policy is active_policy() and lease.policy.source == "gui"
        and authority.sessions.get(session_id) is lease.live
        and authority.executing.get(session_id) == generation
        and adopted_generation == generation and adopted_agent is lease.agent
        and str(getattr(lease.agent, "session_id", "") or "") == durable_sid
        and not getattr(lease.agent, "_interrupt_requested", False)
    )


def request(session_id: str, operation: dict):
    """None means a legacy transport; a bound canonical turn never falls back."""
    lease = _current.get()
    if lease is None:
        return None
    unavailable = {"error": "desktop-player-unavailable"}
    if not _live(lease, session_id):
        return unavailable
    from gateway.session_server_requests import ask_viewer
    canonical_id = lease.turn._approval_owner[1]
    answer = ask_viewer(lease.live.event_stream, canonical_id, "player.agent.request", {"operation": operation},
                        timeout=45, replay=False, cancelled=lambda: not _live(lease, session_id))
    return answer if answer and _live(lease, session_id) else unavailable
