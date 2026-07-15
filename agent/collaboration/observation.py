"""Observation layer — captures structured signals from every agent turn.

Fire-and-forget writes to the fleet graph via GraphClient. Never blocks
the agent loop.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.collaboration.graph_client import GraphClient, GraphMemory, get_client

logger = logging.getLogger(__name__)

# Correction types
CORRECTION_REDIRECT = "redirect"
CORRECTION_TERSE = "terse_fix"
CORRECTION_SCOPE = "scope_shrink"
CORRECTION_FORMAT = "format_preference"
CORRECTION_REJECTION = "rejection"
CORRECTION_LANGUAGE = "language_switch"


@dataclass
class CorrectionSignal:
    """A detected user correction signal."""
    type: str
    weight: float
    snippet: str
    turn_number: int = 0
    session_id: str = ""


@dataclass
class TurnObservation:
    """Structured observation from a single agent turn."""
    session_id: str
    turn_number: int

    # User message features
    user_msg_length: int = 0
    user_msg_word_count: int = 0
    user_msg_question: bool = False
    user_msg_imperative: bool = False
    user_msg_terse: bool = False

    # Response features
    response_length: int = 0
    response_format: str = "prose"  # prose, table, code, bullet, mixed, terse
    response_tool_count: int = 0
    response_iterations: int = 0

    # Task context
    task_type: str = ""
    task_confidence: float = 0.0

    # Corrections
    correction_detected: bool = False
    correction_type: str = ""
    correction_signal: str = ""

    # Tool usage
    tools_used: List[str] = field(default_factory=list)
    tool_call_count: int = 0

    # Timing
    response_latency_ms: int = 0

    def to_graph_memory(self) -> GraphMemory:
        """Convert to a graph memory node for storage."""
        # Build content summary
        parts = [
            f"session:{self.session_id} turn:{self.turn_number}",
            f"task:{self.task_type or 'unknown'}",
        ]
        if self.correction_detected:
            parts.append(f"correction:{self.correction_type}")
        if self.tools_used:
            parts.append(f"tools:{','.join(self.tools_used[:5])}")
        parts.append(f"format:{self.response_format}")

        tags = [self.task_type] if self.task_type else []
        tags.append(f"format:{self.response_format}")
        if self.correction_detected:
            tags.append("correction")
            tags.append(f"correction:{self.correction_type}")
        if self.user_msg_terse:
            tags.append("terse")
        if self.response_tool_count > 0:
            tags.append("tool_heavy")

        return GraphMemory(
            content=" | ".join(parts),
            category="observation",
            tags=tags,
            memory_type="short_term",
            confidence=0.8,
            session_id=self.session_id,
            ttl=86400 * 7,  # 7 days
        )


# ---------------------------------------------------------------------------
# Correction detector
# ---------------------------------------------------------------------------


def detect_correction(user_msg: str, agent_last_response: str = "") -> Optional[CorrectionSignal]:
    """Classify the user's message as a correction or return None.

    Runs post-turn. Must be fast (<1ms) — no external calls.
    """
    if not user_msg:
        return None

    msg = user_msg.strip()
    msg_lower = msg.lower()

    # Redirect: pattern like "not <thing>!" or "!<thing>"
    if msg_lower.startswith("not ") or "!" in msg[:20]:
        return CorrectionSignal(type=CORRECTION_REDIRECT, weight=1.0, snippet=msg[:120])

    # Terse command: single/two-word directive after a response
    words = msg.split()
    if 1 <= len(words) <= 3 and msg[-1] in ".!:":
        terse_triggers = {"go", "do", "now", "fix", "stop", "implement",
                          "just do it", "do it", "move", "execute"}
        lowered = msg_lower.rstrip(".!:")
        if lowered in terse_triggers:
            return CorrectionSignal(type=CORRECTION_TERSE, weight=0.9, snippet=msg[:120])

    # Scope shrink: "ensure X has Y"
    if msg_lower.startswith("ensure "):
        return CorrectionSignal(type=CORRECTION_SCOPE, weight=0.8, snippet=msg[:120])

    # Rejection
    if msg_lower.strip(".!") in ("no", "wrong", "stop", "nope", "incorrect",
                                  "not what i asked", "didn't ask for that"):
        return CorrectionSignal(type=CORRECTION_REJECTION, weight=1.0, snippet=msg[:120])

    # Language switch
    if "english" in msg_lower and any(w in msg_lower for w in ("use", "speak", "write")):
        return CorrectionSignal(type=CORRECTION_LANGUAGE, weight=0.9, snippet=msg[:120])

    return None


# ---------------------------------------------------------------------------
# Response format classifier
# ---------------------------------------------------------------------------


def classify_response_format(response_text: str, tool_calls: int) -> str:
    """Classify response format: prose, table, code, bullet, mixed, terse."""
    if not response_text:
        return "prose"

    # Only classify as terse if it's genuinely brief AND has no structure
    if tool_calls == 0 and len(response_text) < 50 and "```" not in response_text and "|" not in response_text and "\n- " not in response_text and not response_text.startswith("#"):
        return "terse"

    has_code = "```" in response_text
    has_table = "|---" in response_text or "| ---" in response_text
    has_bullets = response_text.count("\n- ") > 1 or response_text.count("\n* ") > 1
    has_header = any(response_text.startswith(c) for c in ("#", "##", "###"))

    # Count distinct structural signals
    signal_count = sum([has_code, has_table, has_bullets, has_header])

    if signal_count >= 2:
        return "mixed"
    if has_table:
        return "table"
    if has_code:
        return "code"
    if has_bullets:
        return "bullet"
    return "prose"


# ---------------------------------------------------------------------------
# Observation capture
# ---------------------------------------------------------------------------


def capture_observation(
    session_id: str,
    turn_number: int,
    user_message: str,
    response_text: str,
    response_tool_count: int,
    response_iterations: int,
    tools_used: List[str],
    task_type: str,
    task_confidence: float,
    latency_ms: int,
    client: Optional[GraphClient] = None,
) -> None:
    """Capture a turn observation to the fleet graph. Fire-and-forget.

    This is the main entry point for the observation layer. Called from
    the turn prologue (post-turn).

    All errors are caught and logged — never blocks the agent.
    """
    try:
        if not session_id or not user_message:
            return

        _client = client or get_client()
        if not _client.is_available():
            return

        # Detect correction
        correction = detect_correction(user_message)

        # Classify format
        fmt = classify_response_format(response_text, response_tool_count)

        # Build observation
        obs = TurnObservation(
            session_id=session_id,
            turn_number=turn_number,
            user_msg_length=len(user_message),
            user_msg_word_count=len(user_message.split()),
            user_msg_question=user_message.strip().endswith("?"),
            user_msg_imperative=user_message.strip().endswith(".") and len(user_message.split()) > 1,
            user_msg_terse=len(user_message.split()) <= 3,
            response_length=len(response_text),
            response_format=fmt,
            response_tool_count=response_tool_count,
            response_iterations=response_iterations,
            task_type=task_type,
            task_confidence=task_confidence,
            correction_detected=correction is not None,
            correction_type=correction.type if correction else "",
            correction_signal=correction.snippet if correction else "",
            tools_used=tools_used or [],
            tool_call_count=len(tools_used or []),
            response_latency_ms=latency_ms,
        )

        # Write to graph
        memory = obs.to_graph_memory()
        _client.create_memory(memory)

        # Also log corrections as decisions for prominence
        if correction and correction.weight >= 0.8:
            _client.create_decision(
                description=f"User correction: {correction.type} — {correction.snippet[:200]}",
                context=f"turn {turn_number} in session {session_id}",
                outcome="observed",
                confidence=correction.weight,
                tags=["collaboration", "correction", correction.type],
                severity="low",
            )

    except Exception as e:
        logger.debug("Observation capture failed (non-fatal): %s", e)
