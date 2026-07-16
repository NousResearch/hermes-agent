"""Safe compact-fallback helpers for Context Health Phase 7.

This module intentionally does not implement Phase 8 rehydrate.  It builds a
sanitized continuity packet and user-facing HOLD payload for compact/compression
exhaustion paths so Hermes stops instead of asking the user to manage `/new` or
retry `/compress` as the primary recovery path.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


_SAFETY_BOUNDARY = (
    "raw transcript excluded; unrelated A/B task material excluded; "
    "secret, token, password, and private body content excluded"
)


@dataclass(frozen=True)
class CompactFallbackHold:
    """Sanitized continuity metadata for a compact-exhaustion HOLD."""

    reason: str
    session_id: str
    task_id: str
    message_count: int
    approx_tokens: Optional[int]
    rehydrate_status: str
    safety_boundary: str = _SAFETY_BOUNDARY

    def as_packet(self) -> Dict[str, Any]:
        packet = asdict(self)
        packet["type"] = "context_health_compact.continuity_packet"
        packet["raw_transcript_included"] = False
        packet["unrelated_context_included"] = False
        packet["secret_token_password_private_body_included"] = False
        return packet


def build_safe_compact_hold_result(
    hold_result: Dict[str, Any],
    *,
    api_calls: int = 0,
) -> Dict[str, Any]:
    """Assemble a run_conversation-safe compact HOLD result.

    Some Hermes surfaces expect a `messages` key in run_conversation results,
    but compact fallback HOLD must not return the raw transcript.  Use a
    sanitized assistant-only message containing the safe HOLD response.
    """

    final_response = str(hold_result.get("final_response") or "Context Health compact fallback HOLD")
    return {
        **hold_result,
        "messages": [{"role": "assistant", "content": final_response}],
        "api_calls": max(0, int(api_calls or 0)),
    }


def build_compact_exhaustion_hold(
    *,
    reason: str,
    session_id: str = "",
    task_id: str = "default",
    message_count: int = 0,
    approx_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a safe HOLD result for compact fallback exhaustion.

    The returned payload is deliberately metadata-only.  It never copies the raw
    transcript, unrelated A/B needles, or token/password/secret/private body
    material into the HOLD response or continuity packet.  Actual same-window
    rehydrate remains a Phase 8 responsibility; Phase 7 only creates a safe
    re-entry packet and stops the unsafe retry path.
    """

    hold = CompactFallbackHold(
        reason=reason,
        session_id=session_id or "unknown",
        task_id=task_id or "default",
        message_count=max(0, int(message_count or 0)),
        approx_tokens=approx_tokens if isinstance(approx_tokens, int) else None,
        rehydrate_status="phase8_not_executed_requires_user_approval",
    )
    packet = hold.as_packet()
    response = (
        "Context Health compact fallback HOLD: compression is exhausted and "
        "Hermes created a sanitized continuity_packet for approved re-entry. "
        "The packet excludes raw transcript content, unrelated A/B task context, "
        "and secret/token/password/private body material. Actual rehydrate is "
        "not executed in Phase 7 and requires a separate approval."
    )
    return {
        "final_response": response,
        "error": response,
        "completed": False,
        "partial": True,
        "failed": True,
        "compression_exhausted": True,
        "context_health_compact": True,
        "safe_hold": True,
        "continuity_packet": packet,
        "rehydrate": "hold_for_phase8_approval",
    }
