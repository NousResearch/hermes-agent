"""CollaborationLearningManager — orchestrates the 4-layer framework.

Entry point for the agent. Manages observation capture, periodic extraction,
adaptation injection, and feedback tracking.

Usage from agent_init.py / turn_context.py:

    manager = CollaborationLearningManager()
    manager.initialize(agent, session_id=...)

    # Post-turn:
    manager.observe_turn(...)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from agent.collaboration.graph_client import get_client
from agent.collaboration.observation import capture_observation
from agent.collaboration.extraction import CollaborationExtractor
from agent.collaboration.adaptation import (
    load_preference_summary,
    format_preference_entries,
)
from agent.collaboration.feedback import CorrectionRateTracker, DriftDetector

logger = logging.getLogger(__name__)

# Extract preferences every N turns
_EXTRACTION_INTERVAL = 15

# Minimum turns needed before extraction is meaningful
_MIN_TURNS_FOR_EXTRACTION = 5


class CollaborationLearningManager:
    """Orchestrates observation, extraction, adaptation, feedback."""

    def __init__(self):
        self._client = None
        self._extractor = None
        self._tracker = None
        self._drift = None

        # Session state
        self._session_id = ""
        self._turn_count = 0
        self._correction_count = 0
        self._last_extraction_turn = 0
        self._enabled = True

        # Cached preference summary (refreshed per session start)
        self._preference_summary = ""
        self._preference_entries: List[Dict] = []

    def initialize(self, session_id: str) -> None:
        """Initialize for a new session.

        Loads existing preferences from the graph for adaptation.
        """
        self._session_id = session_id
        self._turn_count = 0
        self._correction_count = 0

        if not self._client:
            self._client = get_client()

        if not self._client.is_available():
            logger.debug("Collaboration learning: graph API unavailable")
            self._enabled = False
            return

        self._extractor = CollaborationExtractor(self._client)
        self._tracker = CorrectionRateTracker(client=self._client)
        self._drift = DriftDetector(self._tracker, self._client)

        # Load preferences for adaptation
        self._preference_summary = load_preference_summary()
        self._preference_entries = format_preference_entries()

        if self._preference_summary:
            logger.debug("Collaboration: loaded %d preference entries", len(self._preference_entries))

        # Check for drift at session start
        try:
            drift = self._drift.check()
            if drift:
                logger.info("Collaboration drift detected: %s", drift.description)
                if drift.severity == "high":
                    self._trigger_recalibration()
        except Exception:
            pass

    def observe_turn(
        self,
        user_message: str,
        response_text: str,
        tool_calls: int,
        tool_iterations: int,
        tools_used: List[str],
        task_type: str,
        task_confidence: float,
        latency_ms: int,
    ) -> None:
        """Capture a turn observation. Fire-and-forget.

        Called post-turn. May trigger extraction if enough turns have passed.
        """
        if not self._enabled or not self._session_id:
            return

        self._turn_count += 1

        # Capture observation to graph
        capture_observation(
            session_id=self._session_id,
            turn_number=self._turn_count,
            user_message=user_message,
            response_text=response_text,
            response_tool_count=tool_calls,
            response_iterations=tool_iterations,
            tools_used=tools_used,
            task_type=task_type,
            task_confidence=task_confidence,
            latency_ms=latency_ms,
            client=self._client,
        )

        # Track corrections locally for session stats
        from agent.collaboration.observation import detect_correction
        corr = detect_correction(user_message, response_text)
        if corr:
            self._correction_count += 1

        # Periodic extraction
        if (self._turn_count - self._last_extraction_turn) >= _EXTRACTION_INTERVAL \
                and self._turn_count >= _MIN_TURNS_FOR_EXTRACTION:
            self._run_extraction()

    def _run_extraction(self) -> None:
        """Run pattern extraction and update preferences."""
        if not self._extractor:
            return

        try:
            model = self._extractor.extract()
            self._last_extraction_turn = self._turn_count

            # Update cached preferences
            self._preference_summary = load_preference_summary()
            self._preference_entries = format_preference_entries()

            logger.debug("Collaboration extraction completed: %d preferences",
                         len([p for p in model.preferences.values() if p.confidence > 0.3]))
        except Exception as e:
            logger.debug("Collaboration extraction failed: %s", e)

    def _trigger_recalibration(self) -> None:
        """Force recalibration of preferences (e.g. on drift detection)."""
        logger.info("Collaboration: recalibrating preferences due to drift")
        if self._extractor:
            try:
                self._extractor.extract(force=True)
                self._preference_summary = load_preference_summary()
                self._preference_entries = format_preference_entries()
            except Exception as e:
                logger.debug("Recalibration failed: %s", e)

    def on_session_end(self) -> None:
        """Called when the session ends. Records stats, runs final extraction."""
        if not self._enabled:
            return

        # Record correction rate
        if self._tracker and self._turn_count >= _MIN_TURNS_FOR_EXTRACTION:
            try:
                self._tracker.record_session(
                    self._session_id,
                    self._correction_count,
                    self._turn_count,
                )
            except Exception as e:
                logger.debug("Failed to record session stats: %s", e)

        # Final extraction
        self._run_extraction()

    # -- Adaptation outputs -------------------------------------------------

    @property
    def preference_summary(self) -> str:
        """Compact preference summary for system prompt injection."""
        return self._preference_summary

    @property
    def preference_entries(self) -> List[Dict]:
        """Preference entries for memory tool injection."""
        return self._preference_entries

    @property
    def enabled(self) -> bool:
        return self._enabled
