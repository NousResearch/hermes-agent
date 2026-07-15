"""Pattern extraction layer — distills observations into preferences.

Runs at session end and periodically mid-session. Reads observations from
the fleet graph and produces structured preference signals with confidence
scores.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.collaboration.graph_client import GraphClient, GraphMemory, get_client

logger = logging.getLogger(__name__)

# Minimum evidence to form a preference
_MIN_SAMPLE_SIZE = 3

# Confidence floor — below this, preferences are not applied
_CONFIDENCE_FLOOR = 0.6

# Preference memory TTL (90 days)
_PREFERENCE_TTL = 86400 * 90


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class PreferenceSignal:
    """A structured preference with confidence and evidence."""
    dimension: str           # e.g. "verbosity", "format", "execution_style"
    value: str               # e.g. "concise", "tables_first", "just_go"
    confidence: float        # 0.0-1.0
    sample_size: int = 0
    correction_rate_before: float = 0.0
    correction_rate_after: float = 0.0
    last_observed: str = ""


@dataclass
class PreferenceModel:
    """The full learned preference model for a user."""
    version: int = 1
    last_updated: str = ""
    preferences: Dict[str, PreferenceSignal] = field(default_factory=dict)
    confidence_threshold: float = _CONFIDENCE_FLOOR


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


class CollaborationExtractor:
    """Extract collaboration preferences from graph observations.

    Reads from the fleet graph and produces a PreferenceModel.
    """

    def __init__(self, client: Optional[GraphClient] = None):
        self._client = client or get_client()
        self._model = PreferenceModel()

    def extract(self, force: bool = False) -> PreferenceModel:
        """Run full extraction: analyze all observations → produce preferences.

        Args:
            force: If True, re-extract even if model is already built.

        Returns:
            The updated PreferenceModel.
        """
        if self._model.preferences and not force:
            return self._model

        # Pull recent observations and corrections from the graph
        corrections = self._load_corrections(limit=100)
        observations = self._load_observations(limit=200)

        if not observations and not corrections:
            logger.debug("No observations to extract from")
            return self._model

        # Extract each dimension
        self._model.preferences["execution_style"] = self._extract_execution_style(
            observations, corrections
        )
        self._model.preferences["verbosity"] = self._extract_verbosity(
            observations, corrections
        )
        self._model.preferences["format"] = self._extract_format(
            observations, corrections
        )
        self._model.preferences["correction_handling"] = self._extract_correction_handling(
            corrections
        )
        self._model.preferences["tool_preference"] = self._extract_tool_preference(
            observations
        )

        self._model.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._model.version += 1

        # Persist to graph
        self._save_model()

        return self._model

    # -- Data loading -------------------------------------------------------

    def _load_corrections(self, limit: int = 100) -> List[Dict]:
        """Load correction observations from the graph."""
        return self._client.search_memories(
            query="tag:correction",
            category="observation",
            limit=limit,
        )

    def _load_observations(self, limit: int = 200) -> List[Dict]:
        """Load all observations from the graph."""
        return self._client.search_memories(
            query="category:observation",
            limit=limit,
        )

    # -- Dimension extractors -----------------------------------------------

    @staticmethod
    def _extract_execution_style(
        observations: List[Dict],
        corrections: List[Dict],
    ) -> PreferenceSignal:
        """Extract whether user prefers 'just go' vs 'plan first'."""
        terse_corrections = [
            c for c in corrections
            if "terse_fix" in c.get("tags", [])
        ]
        sample = len(terse_corrections)

        # If >30% of corrections are terse, user prefers execution over explanation
        ratio = sample / max(len(corrections), 1)
        is_just_go = ratio > 0.2

        confidence = compute_confidence(
            sample_size=sample,
            consistency=ratio,
            correction_impact=ratio,
            recency=1.0 if sample > 0 else 0.0,
        )

        return PreferenceSignal(
            dimension="execution_style",
            value="just_go" if is_just_go else "plan_first",
            confidence=confidence,
            sample_size=sample,
            last_observed=time.strftime("%Y-%m-%d"),
        )

    @staticmethod
    def _extract_verbosity(
        observations: List[Dict],
        corrections: List[Dict],
    ) -> PreferenceSignal:
        """Extract verbosity preference from response lengths before corrections."""
        # Look at response lengths for turns that triggered corrections
        terse_obs = []
        for c in corrections:
            c_type = next((t for t in c.get("tags", []) if t.startswith("correction:")), "")
            if "terse" in c_type or "redirect" in c_type:
                turn_num = _extract_turn(c.get("content", ""))
                if turn_num:
                    terse_obs.append(turn_num)

        sample = len(terse_obs)
        is_concise = sample > _MIN_SAMPLE_SIZE

        return PreferenceSignal(
            dimension="verbosity",
            value="concise" if is_concise else "standard",
            confidence=compute_confidence(sample, 0.6, 0.5, 0.5),
            sample_size=sample,
            last_observed=time.strftime("%Y-%m-%d"),
        )

    @staticmethod
    def _extract_format(
        observations: List[Dict],
        corrections: List[Dict],
    ) -> PreferenceSignal:
        """Extract preferred response format."""
        format_counts: Dict[str, int] = {}
        for obs in observations:
            content = obs.get("content", "")
            if "format:" in content:
                for part in content.split():
                    if part.startswith("format:"):
                        fmt = part.split(":", 1)[1]
                        format_counts[fmt] = format_counts.get(fmt, 0) + 1

        if not format_counts:
            return PreferenceSignal(
                dimension="format", value="prose", confidence=0.0,
                last_observed=time.strftime("%Y-%m-%d"),
            )

        preferred = max(format_counts, key=format_counts.get)
        total = sum(format_counts.values())
        consistency = format_counts[preferred] / max(total, 1)

        return PreferenceSignal(
            dimension="format",
            value=f"{preferred}_preferred",
            confidence=compute_confidence(total, consistency, 0.5, 0.5),
            sample_size=total,
            last_observed=time.strftime("%Y-%m-%d"),
        )

    @staticmethod
    def _extract_correction_handling(
        corrections: List[Dict],
    ) -> PreferenceSignal:
        """Extract how the user tends to correct the agent."""
        type_counts: Dict[str, int] = {}
        for c in corrections:
            for tag in c.get("tags", []):
                if tag.startswith("correction:"):
                    ctype = tag.split(":", 1)[1]
                    type_counts[ctype] = type_counts.get(ctype, 0) + 1

        if not type_counts:
            return PreferenceSignal(
                dimension="correction_handling", value="unknown", confidence=0.0,
                last_observed=time.strftime("%Y-%m-%d"),
            )

        top_type = max(type_counts, key=type_counts.get)
        return PreferenceSignal(
            dimension="correction_handling",
            value=top_type,
            confidence=min(type_counts[top_type] / 10, 0.9),
            sample_size=sum(type_counts.values()),
            last_observed=time.strftime("%Y-%m-%d"),
        )

    @staticmethod
    def _extract_tool_preference(
        observations: List[Dict],
    ) -> PreferenceSignal:
        """Extract task-specific tool preferences."""
        task_tools: Dict[str, Dict[str, int]] = {}
        for obs in observations:
            content = obs.get("content", "")
            task = ""
            tools = []
            for part in content.split():
                if part.startswith("task:"):
                    task = part.split(":", 1)[1]
                elif part.startswith("tools:"):
                    tools = part.split(":", 1)[1].split(",")

            if task and tools:
                task_tools.setdefault(task, {})
                for t in tools:
                    task_tools[task][t] = task_tools[task].get(t, 0) + 1

        return PreferenceSignal(
            dimension="tool_preference",
            value=json.dumps(task_tools) if task_tools else "none",
            confidence=0.5 if task_tools else 0.0,
            sample_size=sum(len(v) for v in task_tools.values()) if task_tools else 0,
            last_observed=time.strftime("%Y-%m-%d"),
        )

    # -- Persistence --------------------------------------------------------

    def _save_model(self) -> None:
        """Persist the preference model to the graph as memory nodes."""
        if not self._client.is_available():
            return

        # Delete old preferences
        old = self._client.search_memories("category:preference", limit=50)
        for m in old:
            mid = m.get("id", "")
            if mid:
                self._client.delete_memory(mid)

        # Save each preference as a graph memory
        for pref in self._model.preferences.values():
            if pref.confidence < 0.2:
                continue  # Don't persist noise
            memory = GraphMemory(
                content=f"{pref.dimension}: {pref.value} (confidence={pref.confidence:.2f}, "
                        f"samples={pref.sample_size})",
                category="preference",
                tags=["collaboration", pref.dimension],
                memory_type="long_term",
                confidence=pref.confidence,
                ttl=_PREFERENCE_TTL,
            )
            self._client.create_memory(memory)

        # Also record as a decision for prominence
        summary = "; ".join(
            f"{d}={p.value} ({p.confidence:.1f})"
            for d, p in self._model.preferences.items()
            if p.confidence >= 0.3
        )
        if summary:
            self._client.create_decision(
                description=f"Collaboration preferences extracted: {summary}",
                context="Automatic extraction from observations",
                outcome="recorded",
                confidence=0.7,
                tags=["collaboration", "preferences"],
                severity="low",
            )

    def get_high_confidence_preferences(self, threshold: float = _CONFIDENCE_FLOOR) -> Dict[str, str]:
        """Return preferences with confidence above threshold."""
        return {
            d: p.value for d, p in self._model.preferences.items()
            if p.confidence >= threshold
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_confidence(
    sample_size: int,
    consistency: float,
    correction_impact: float,
    recency: float,
) -> float:
    """Compute confidence in a preference signal.

    Requires at least _MIN_SAMPLE_SIZE observations.
    """
    if sample_size < _MIN_SAMPLE_SIZE:
        return 0.0
    if consistency < 0.3:
        return 0.0

    score = (
        0.3 * min(sample_size / 20, 1.0) +
        0.3 * consistency +
        0.3 * correction_impact +
        0.1 * recency
    )
    return max(0.0, min(score, 1.0))


def _extract_turn(content: str) -> Optional[int]:
    """Extract turn number from a graph memory content string."""
    import re
    m = re.search(r'turn:(\d+)', content)
    return int(m.group(1)) if m else None


# Need json for serialization in _extract_tool_preference
import json
