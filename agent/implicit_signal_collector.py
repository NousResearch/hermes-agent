"""Collect implicit quality signals from conversation patterns.

Monitors the conversation flow and records signals without requiring
explicit user feedback. These signals are used to retrain the routing
and reasoning effort classifiers over time.

Signal types:
- routing_quality: Was routing to cheap/primary correct?
- reasoning_effort: Was the effort level appropriate?
- skill_match: Was the skill suggestion used?

How signals are derived:
- Continuation after cheap route (3+ turns) → routing_quality=0.8
- User says "wrong"/"try again" after cheap route → routing_quality=0.0
- User says "think harder"/"more detail" → reasoning_effort too low
- Conversation proceeds normally → current effort was fine
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from hermes_state import SessionDB

logger = logging.getLogger(__name__)

# Patterns that suggest the previous response was inadequate
_CORRECTION_PATTERNS = re.compile(
    r"\b(?:wrong|incorrect|that'?s\s+not|no\s+that'?s|try\s+again|"
    r"not\s+what\s+I\s+(?:asked|meant|wanted)|"
    r"use\s+(?:the\s+)?(?:strong|primary|better|good)\s+model|"
    r"that\s+doesn'?t\s+(?:work|help|make\s+sense)|"
    r"you'?re\s+(?:wrong|confused|hallucinating))\b",
    re.IGNORECASE,
)

_EFFORT_UPGRADE_PATTERNS = re.compile(
    r"\b(?:think\s+(?:harder|more|deeper|carefully)|"
    r"more\s+(?:detail|thorough|careful|rigorous)|"
    r"elaborate|expand\s+on|go\s+deeper|"
    r"that'?s\s+too\s+(?:brief|short|simple)|"
    r"I\s+need\s+more)\b",
    re.IGNORECASE,
)


class ImplicitSignalCollector:
    """Collects quality signals from conversation flow patterns."""

    def __init__(self, db: "SessionDB"):
        self._db = db
        self._last_routing_info: Optional[Dict] = None
        self._last_reasoning_effort: Optional[str] = None
        self._last_skills_suggested: Optional[List[str]] = None
        self._turns_since_route: int = 0
        self._session_id: Optional[str] = None

    def set_session(self, session_id: str) -> None:
        """Set the current session for signal attribution."""
        self._session_id = session_id

    def on_routing_decision(
        self,
        routed_model: str,
        routing_reason: str,
        message_text: str,
        routing_decision_id: Optional[int] = None,
    ) -> None:
        """Called when a model routing decision is made."""
        self._last_routing_info = {
            "model": routed_model,
            "reason": routing_reason,
            "message": message_text,
            "decision_id": routing_decision_id,
            "timestamp": time.time(),
        }
        self._turns_since_route = 0

    def on_reasoning_effort_set(self, effort: str) -> None:
        """Called when reasoning effort level is applied."""
        self._last_reasoning_effort = effort

    def on_skills_suggested(self, skill_names: List[str]) -> None:
        """Called when skills are matched/suggested for the conversation."""
        self._last_skills_suggested = list(skill_names)

    def on_skill_used(self, skill_name: str) -> None:
        """Called when a skill is actually used (tool invoked)."""
        if self._last_skills_suggested and skill_name in self._last_skills_suggested:
            self._db.log_implicit_signal(
                signal_type="skill_match",
                signal_value=1.0,
                signal_source="implicit_usage",
                context_id=skill_name,
                session_id=self._session_id,
                metadata=f'{{"skill": "{skill_name}"}}',
            )

    def on_user_message(self, message_text: str) -> None:
        """Called on each user message to check for implicit correction/continuation signals."""
        text = (message_text or "").strip()
        if not text:
            return

        self._turns_since_route += 1

        # Check for correction signals (negative feedback)
        if _CORRECTION_PATTERNS.search(text):
            self._emit_correction_signals(text)
            return

        # Check for effort upgrade signals
        if _EFFORT_UPGRADE_PATTERNS.search(text):
            self._emit_effort_signal(text, signal="too_low")
            return

        # Positive continuation signal: conversation proceeding normally
        # after a cheap route for 3+ turns = routing was fine
        if (
            self._last_routing_info
            and self._last_routing_info.get("reason") in ("simple_turn", "classifier")
            and self._turns_since_route >= 3
        ):
            self._db.log_implicit_signal(
                signal_type="routing_quality",
                signal_value=0.8,
                signal_source="implicit_continuation",
                context_id=str(self._last_routing_info.get("decision_id", "")),
                session_id=self._session_id,
                message_text=self._last_routing_info.get("message", ""),
            )
            # Reset so we don't double-count
            self._last_routing_info = None

    def _emit_correction_signals(self, user_text: str) -> None:
        """User corrected the previous response — negative signal."""
        if self._last_routing_info:
            routing_reason = self._last_routing_info.get("reason", "")
            is_cheap = routing_reason in ("simple_turn", "classifier")
            if is_cheap:
                self._db.log_implicit_signal(
                    signal_type="routing_quality",
                    signal_value=0.0,
                    signal_source="implicit_correction",
                    context_id=str(self._last_routing_info.get("decision_id", "")),
                    session_id=self._session_id,
                    message_text=self._last_routing_info.get("message", ""),
                )
                self._last_routing_info = None

    def _emit_effort_signal(self, user_text: str, signal: str) -> None:
        """User explicitly asked for more reasoning effort."""
        if self._last_reasoning_effort:
            self._db.log_implicit_signal(
                signal_type="reasoning_effort",
                signal_value=0.0,
                signal_source="implicit_correction",
                session_id=self._session_id,
                message_text=user_text[:500],
                metadata=f'{{"previous_effort": "{self._last_reasoning_effort}"}}',
            )

    def on_session_end(self) -> None:
        """Called when a session ends. Emit any pending signals.

        Skills that were suggested but never used get a mild negative signal.
        """
        if self._last_skills_suggested:
            for skill_name in self._last_skills_suggested:
                self._db.log_implicit_signal(
                    signal_type="skill_match",
                    signal_value=0.3,
                    signal_source="implicit_skip",
                    context_id=skill_name,
                    session_id=self._session_id,
                    metadata=f'{{"skill": "{skill_name}", "note": "suggested but not used"}}',
                )
            self._last_skills_suggested = None
