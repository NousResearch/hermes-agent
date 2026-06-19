"""Shared base for the realtime and streaming call brains.

Both handlers need the same session policy — caller allowlist, session-scope key,
meeting transcript, agent consult, group-call gate, greeting/outbound state. This
base (``BaseTeamsCallHandler``) holds that once so the two handlers only implement
what differs (the realtime model vs the STT→agent→TTS loop). Also owns the
process-global pending-outbound registry (call-back correlation, with TTL).
"""

from __future__ import annotations

import logging
import time

from . import group_call_gate, protocol
from .agent_consult import AgentConsult
from .bridge_server import CallSession, CallSessionHandler
from .config import TeamsVoiceConfig, caller_allowed
from .meeting import MeetingTranscript

logger = logging.getLogger(__name__)

# The inbound call that requests a callback and the outbound leg that delivers it
# are *different* WebSocket connections, so the pending spoken result is keyed by
# the worker's callId here. Entries carry a TTL so a never-answered call-back can't
# leak its result string indefinitely.
_PENDING_OUTBOUND: dict[str, tuple[str, float]] = {}
_PENDING_TTL_S = 600.0


def _pending_prune() -> None:
    now = time.monotonic()
    for k in [k for k, (_t, exp) in _PENDING_OUTBOUND.items() if exp <= now]:
        _PENDING_OUTBOUND.pop(k, None)


def _pending_set(call_id: str, text: str) -> None:
    _pending_prune()
    _PENDING_OUTBOUND[call_id] = (text, time.monotonic() + _PENDING_TTL_S)


def _pending_pop(call_id: str) -> str | None:
    _pending_prune()
    entry = _PENDING_OUTBOUND.pop(call_id, None)
    return entry[0] if entry else None


class BaseTeamsCallHandler(CallSessionHandler):
    """Common session policy shared by the realtime + streaming handlers."""

    def __init__(self, bridge_config: TeamsVoiceConfig | None = None) -> None:
        self._bridge = bridge_config
        self._require_recording = bridge_config.require_recording_status if bridge_config else True
        self._session: CallSession | None = None
        self._caller: protocol.CallerInfo | None = None
        self._thread_id = ""
        self._outbound = False
        self._greeted = False
        self._pending_greeting: str | None = None
        self._meeting = MeetingTranscript()
        self._consult = AgentConsult()
        wake = tuple(bridge_config.wake_phrases) if (bridge_config and bridge_config.wake_phrases) else ("assistant", "hermes")
        self._gate_cfg = group_call_gate.GroupCallGateConfig(wake_phrases=wake)
        self._last_addressed_ms: float | None = None

    # ── shared helpers ────────────────────────────────────────────────────────

    def _first_name(self) -> str:
        name = (self._caller.display_name if self._caller else "") or ""
        return name.strip().split(" ")[0] if name.strip() else ""

    def _recording_ok(self, session: CallSession) -> bool:
        return (not self._require_recording) or session.recording_active

    async def _begin_session(self, session: CallSession, msg: protocol.SessionStart) -> bool:
        """Common ``session.start``: state + allowlist + scope. False = rejected."""
        await CallSessionHandler.on_session_start(self, session, msg)
        self._session = session
        self._caller = msg.caller
        self._thread_id = msg.thread_id
        self._outbound = (msg.direction or "").lower() == "outbound"
        if self._outbound:  # delivery leg of a call-back
            self._pending_greeting = _pending_pop(msg.call_id)
        elif self._bridge and not caller_allowed(
            self._bridge, msg.caller.aad_id, msg.caller.display_name
        ):
            logger.info("[teams_voice] caller not allowlisted; rejecting %s", session.call_id)
            await session._ws.close()
            return False
        scope = self._bridge.session_scope if self._bridge else "per-call"
        if scope == "per-thread":
            key = msg.thread_id or msg.call_id
        elif scope == "per-aad":
            key = msg.caller.aad_id or msg.call_id
        else:
            key = msg.call_id
        self._consult = AgentConsult(session_id=f"teams:{key}")
        return True

    def _group_decision(self, transcript: str, now_ms: float):
        """``(is_group, GateDecision)`` for a finished caller turn."""
        is_group = (self._session.human_count if self._session else 0) >= 2
        decision = group_call_gate.should_respond_to_group_turn(
            transcript=transcript, is_group=is_group, config=self._gate_cfg,
            last_addressed_at_ms=self._last_addressed_ms, now_ms=now_ms,
        )
        return is_group, decision

    async def _safe_expression(self, emotion: str) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_expression(emotion)
        except Exception:  # noqa: BLE001 — cosmetic
            pass
