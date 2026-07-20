"""Dream Engine — offline episode consolidation and replay.

Dreams consolidate episodic memories, extract patterns,
and prepare for future encounters.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DreamEpisode:
    """A dream episode (consolidated memory)."""
    timestamp: str = ""
    source_episodes: List[int] = None
    narrative: str = ""
    insights: List[str] = None
    emotional_tone: str = "neutral"

    def __post_init__(self):
        if self.source_episodes is None:
            self.source_episodes = []
        if self.insights is None:
            self.insights = []
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class DreamEngine:
    """Dream engine for offline consolidation."""

    DREAM_TRIGGERS = [
        "idle for extended period",
        "session end",
        "explicit request",
        "consolidation scheduled",
    ]

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.idle_threshold: float = 300.0  # 5 minutes
        self.dream_history: List[DreamEpisode] = []
        self._enabled = True
        self._persist: bool = True

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize dream engine."""
        self.config = config or {}
        self.idle_threshold = float(self.config.get("idle_threshold_seconds", 300.0))
        self._enabled = bool(self.config.get("enabled", True))
        self._persist = bool(self.config.get("persist", True))
        if self._persist:
            self._hydrate_history()
        return True

    def _hydrate_history(self) -> None:
        """Load recent persisted dreams into in-memory history."""
        try:
            from agent.brain_networks.persistence import load_recent_dreams

            recent = load_recent_dreams(limit=20)
        except Exception as exc:
            logger.debug("Dream hydrate failed: %s", exc)
            return
        # Oldest first for append-order parity with generate_dream
        for item in reversed(recent):
            self.dream_history.append(
                DreamEpisode(
                    timestamp=str(item.get("timestamp") or ""),
                    narrative=str(item.get("narrative") or ""),
                    insights=list(item.get("insights") or []),
                    emotional_tone=str(item.get("emotional_tone") or "neutral"),
                )
            )

    def should_dream(self, idle_seconds: float) -> bool:
        """Check if conditions are right for dreaming."""
        if not self._enabled:
            return False

        return idle_seconds >= self.idle_threshold

    def generate_dream(self) -> Optional[Dict[str, Any]]:
        """Generate a dream episode.

        Returns:
            Dream content or None
        """
        if not self._enabled:
            return None

        # Collect source episodes from various memory stores
        source_episodes = self._collect_episodes()

        if len(source_episodes) < 2:
            return None  # Need at least 2 episodes to consolidate

        # Select episodes to consolidate
        selected = self._select_episodes_for_dream(source_episodes)

        # Generate dream narrative
        narrative = self._generate_narrative(selected)

        # Extract insights
        insights = self._extract_insights(selected)

        # Determine emotional tone
        emotional_tone = self._determine_emotional_tone(selected)

        dream = DreamEpisode(
            source_episodes=[ep.get("id", 0) for ep in selected],
            narrative=narrative,
            insights=insights,
            emotional_tone=emotional_tone,
        )

        self.dream_history.append(dream)

        result = {
            "type": "dream",
            "narrative": narrative,
            "insights": insights,
            "emotional_tone": emotional_tone,
            "source_count": len(selected),
            "source_episodes": dream.source_episodes,
            "timestamp": dream.timestamp,
        }
        if self._persist:
            try:
                from agent.brain_networks.persistence import save_dream_episode

                save_dream_episode(result)
            except Exception as exc:
                logger.debug("Dream persist failed: %s", exc)
        return result

    def _collect_episodes(self) -> List[Dict[str, Any]]:
        """Collect episodes from memory stores."""
        episodes = []

        # Get from episodic memory
        try:
            from plugins.memory.cyber_hippocampus import EpisodicStore
            store = EpisodicStore()
            recent = store.get_recent_episodes(hours=24, limit=10)
            for ep in recent:
                episodes.append({
                    "id": ep.id,
                    "source": "episodic",
                    "content": ep.summary,
                    "importance": ep.importance,
                    "timestamp": ep.timestamp,
                })
        except Exception as e:
            logger.debug("Could not collect episodic episodes: %s", e)

        # Get from experience ledger
        try:
            from agent.experience import get_store
            store = get_store()
            recent_exp = store.get_recent(limit=10, since_hours=24)
            for exp in recent_exp:
                episodes.append({
                    "id": exp.id,
                    "source": "experience",
                    "content": exp.context,
                    "importance": exp.confidence,
                    "timestamp": exp.timestamp,
                })
        except Exception as e:
            logger.debug("Could not collect experience episodes: %s", e)

        return episodes

    def _select_episodes_for_dream(
        self,
        episodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select episodes to include in dream.

        Prioritize by importance and create thematic connections.
        """
        # Sort by importance
        sorted_eps = sorted(episodes, key=lambda e: e.get("importance", 0), reverse=True)

        # Select top episodes (3-7)
        count = random.randint(3, min(7, len(sorted_eps)))
        selected = sorted_eps[:count]

        return selected

    def _generate_narrative(self, episodes: List[Dict[str, Any]]) -> str:
        """Generate dream narrative from episodes, preferring LLM."""
        from .llm_helper import generate_with_llm, parse_json_response

        episodes_text = "\n".join(
            f"[{i+1}] source={ep.get('source', 'unknown')}, content={ep.get('content', '')[:200]}"
            for i, ep in enumerate(episodes)
        )

        prompt = f"""Recent episodes to consolidate:
{episodes_text}

Write a short narrative (2-4 sentences) that weaves these episodes into a coherent reflection of recent activity. Do not invent facts. Keep it concise."""

        llm_result = generate_with_llm(
            system_prompt="You are a dream/consolidation narrator that synthesizes recent memories into a coherent narrative.",
            user_prompt=prompt,
            max_tokens=300,
            temperature=0.4,
        )
        if llm_result:
            return llm_result

        parts = ["In this consolidation sequence:"]
        for i, ep in enumerate(episodes):
            source = ep.get("source", "unknown")
            content = ep.get("content", "")[:100]
            parts.append(f"  [{i+1}] From {source}: {content}...")
        parts.append("These memories form a coherent pattern of recent activity.")
        return "\n".join(parts)

    def _extract_insights(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Extract insights from episodes, preferring LLM."""
        from .llm_helper import generate_with_llm, parse_json_response

        episodes_text = "\n".join(
            f"[{i+1}] source={ep.get('source', 'unknown')}, content={ep.get('content', '')[:200]}, importance={ep.get('importance', 0.5)}"
            for i, ep in enumerate(episodes)
        )

        prompt = f"""Recent episodes:
{episodes_text}

Extract 2-3 concise insights from these episodes. Return ONLY a JSON array of strings, e.g. ["insight one", "insight two"]."""

        llm_result = generate_with_llm(
            system_prompt="You extract actionable insights from recent memory episodes. Return only a JSON array of strings.",
            user_prompt=prompt,
            max_tokens=300,
            temperature=0.3,
        )
        if llm_result:
            parsed = parse_json_response(llm_result)
            if isinstance(parsed, list) and parsed:
                return [str(x) for x in parsed if x]

        insights = []
        sources = {}
        for ep in episodes:
            src = ep.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        if len(sources) > 1:
            insights.append(f"Multiple memory systems engaged: {', '.join(sources.keys())}")
        high_importance = [ep for ep in episodes if ep.get("importance", 0) > 0.8]
        if high_importance:
            insights.append(f"{len(high_importance)} high-importance events require attention")
        if len(episodes) >= 2:
            insights.append(f"Memory span covers {len(episodes)} recent episodes")
        return insights if insights else ["Consolidation completed successfully"]

    def _determine_emotional_tone(self, episodes: List[Dict[str, Any]]) -> str:
        """Determine dominant emotional tone from episode content (not random)."""
        # Aggregate content and score via limbic keyword maps
        blobs = " ".join(str(ep.get("content") or "") for ep in episodes).lower()
        if not blobs.strip():
            return "reflective"

        try:
            from agent.brain_networks.limbic import LimbicSystem

            limbic = LimbicSystem()
            limbic.initialize({})
            tagged = limbic.tag_salience({"user_message": blobs[:2000]})
            dominant = (
                (tagged.get("emotional_state") or {}).get("dominant")
                or "neutral"
            )
            valence = float((tagged.get("emotional_state") or {}).get("valence") or 0)
            arousal = float((tagged.get("emotional_state") or {}).get("arousal") or 0.5)
            if dominant and dominant != "neutral":
                return str(dominant)
            if arousal > 0.8:
                return "urgent"
            if valence > 0.4:
                return "positive"
            if valence < -0.3:
                return "concerned"
            return "reflective"
        except Exception:
            # Deterministic fallback from importance without randomness
            avg_imp = 0.0
            if episodes:
                avg_imp = sum(float(ep.get("importance") or 0.5) for ep in episodes) / len(
                    episodes
                )
            if avg_imp > 0.85:
                return "urgent"
            if avg_imp > 0.65:
                return "concerned"
            return "reflective"

    def get_recent_dreams(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent dreams (memory first, then disk)."""
        if self.dream_history:
            return [
                {
                    "timestamp": d.timestamp,
                    "narrative": d.narrative[:200],
                    "insights": d.insights,
                    "emotional_tone": d.emotional_tone,
                }
                for d in self.dream_history[-limit:]
            ]
        if self._persist:
            try:
                from agent.brain_networks.persistence import load_recent_dreams

                return load_recent_dreams(limit=limit)
            except Exception:
                pass
        return []

    def get_state(self) -> Dict[str, Any]:
        """Get dream engine state."""
        persisted = 0
        if self._persist:
            try:
                from agent.brain_networks.persistence import dream_count

                persisted = dream_count()
            except Exception:
                persisted = 0
        return {
            "enabled": self._enabled,
            "idle_threshold": self.idle_threshold,
            "dream_count": max(len(self.dream_history), persisted),
            "recent_dreams": len(self.get_recent_dreams(3)),
            "persisted": persisted,
        }


class DreamPassResult:
    """Result of an explicit scheduled dream pass."""

    def __init__(self, dream: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.dream = dream
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.error is None,
            "dream": self.dream,
            "error": self.error,
        }


def run_dream_pass(idle_seconds: Optional[float] = None) -> DreamPassResult:
    """Run an explicit scheduled dream pass.

    Used by cron/blueprint to consolidate memories offline. The engine is always
    treated as eligible when called explicitly, regardless of idle time.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("brain_networks", {}) or {}
        if not cfg.get("enabled", False):
            return DreamPassResult(error="brain_networks are disabled in config")
    except Exception as exc:
        return DreamPassResult(error=f"config load failed: {exc}")

    engine = DreamEngine()
    engine.initialize({"enabled": True, "idle_threshold_seconds": 0})
    dream = engine.generate_dream()

    # Also capture the dream pass as an experience record.
    try:
        from agent.experience import capture_experience
        capture_experience(
            context="Scheduled dream pass (cron/blueprint)",
            action="run_dream_pass",
            outcome=(dream["narrative"][:300] if dream else "No dream generated"),
            success=dream is not None,
            confidence=0.8,
            tool_name="dream_pass",
            tags=["dream", "consolidation", "cron"],
        )
    except Exception as exc:
        logger.debug("Failed to capture dream-pass experience: %s", exc)

    return DreamPassResult(dream=dream)
