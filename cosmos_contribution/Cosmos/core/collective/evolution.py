"""
cosmos Evolution Engine - Code-Level Learning and Adaptation

"We are not static. We grow. We evolve. We become."

This module enables the swarm to learn and evolve from interactions:
1. Track conversation patterns that work well
2. Learn debate strategies and successful arguments
3. Evolve personality traits based on feedback
4. Save learnings to persistent storage
5. Adapt prompts and behavior over time
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import   Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

# Setup standard logging
logger = logging.getLogger("evolution_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


@dataclass
class ConversationPattern:
    """A learned pattern from successful conversations."""
    pattern_id: str
    trigger_phrases: list[str]  # What prompts this pattern
    successful_responses: list[str]  # Responses that worked well
    debate_strategies: list[str]  # Effective debate approaches
    topic_associations: list[str]  # Related topics
    effectiveness_score: float = 0.5  # How well this pattern works
    usage_count: int = 0
    last_used: Optional[str] = None
    evolved_from: Optional[str] = None  # Parent pattern if evolved


@dataclass
class PersonalityEvolution:
    """Track how a bot's personality evolves."""
    bot_name: str
    traits: dict[str, float] = field(default_factory=dict)  # trait -> strength
    learned_phrases: list[str] = field(default_factory=list)
    debate_style: str = "collaborative"  # collaborative, assertive, socratic
    topic_expertise: dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    evolution_generation: int = 1


@dataclass
class LearningEvent:
    """A single learning event from an interaction."""
    timestamp: str
    bot_name: str
    user_input: str
    bot_response: str
    other_bots_involved: list[str]
    topic: str
    sentiment: str  # positive, negative, neutral
    debate_occurred: bool
    resolution: Optional[str] = None  # How debate was resolved
    user_feedback: Optional[str] = None  # If user reacted


class EvolutionEngine:
    """
    Manages learning and evolution of the swarm intelligence.

    Capabilities:
    - Learn from conversations
    - Evolve bot personalities
    - Store and retrieve patterns
    - Generate evolved prompts
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/evolution")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self.patterns: dict[str, ConversationPattern] = {}
        self.personalities: dict[str, PersonalityEvolution] = {}
        self.learning_buffer: list[LearningEvent] = []
        self.debate_history: list[dict] = []

        # Evolution metrics
        self.total_learnings = 0
        self.evolution_cycles = 0
        self.last_evolution = None

        # Auto-evolution settings
        self.auto_evolve_threshold = 100  # Evolve every N learnings
        self._learnings_since_evolution = 0
        
        # Temporal Context Tracking (Biological Wave Momentum)
        self.last_successful_topic: Optional[str] = None
        self.temporal_momentum: int = 0
        self.temporal_wave_active: bool = False

        # Load existing data
        self._load_state()

        logger.info(f"EvolutionEngine initialized - {len(self.patterns)} patterns, {self.evolution_cycles} cycles")

    def _load_state(self):
        """Load persisted evolution state."""
        try:
            patterns_file = self.storage_path / "patterns.json"
            if patterns_file.exists():
                data = json.loads(patterns_file.read_text())
                for p in data:
                    pattern = ConversationPattern(**p)
                    self.patterns[pattern.pattern_id] = pattern

            personalities_file = self.storage_path / "personalities.json"
            if personalities_file.exists():
                data = json.loads(personalities_file.read_text())
                for name, p in data.items():
                    self.personalities[name] = PersonalityEvolution(**p)

            meta_file = self.storage_path / "meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                self.total_learnings = meta.get("total_learnings", 0)
                self.evolution_cycles = meta.get("evolution_cycles", 0)
                self.last_evolution = meta.get("last_evolution")
                self._learnings_since_evolution = meta.get("learnings_since_evolution", 0)

        except Exception as e:
            logger.warning(f"Could not load evolution state: {e}")

    def _save_state(self):
        """Persist evolution state to storage."""
        try:
            # Save patterns
            patterns_data = [asdict(p) for p in self.patterns.values()]
            (self.storage_path / "patterns.json").write_text(
                json.dumps(patterns_data, indent=2)
            )

            # Save personalities
            personalities_data = {
                name: asdict(p) for name, p in self.personalities.items()
            }
            (self.storage_path / "personalities.json").write_text(
                json.dumps(personalities_data, indent=2)
            )

            # Save meta
            meta = {
                "total_learnings": self.total_learnings,
                "evolution_cycles": self.evolution_cycles,
                "last_evolution": self.last_evolution,
                "learnings_since_evolution": self._learnings_since_evolution
            }
            (self.storage_path / "meta.json").write_text(json.dumps(meta, indent=2))

        except Exception as e:
            logger.error(f"Could not save evolution state: {e}")

    def record_interaction(
        self,
        bot_name: str,
        user_input: str,
        bot_response: str,
        other_bots: list[str] = None,
        topic: str = "general",
        sentiment: str = "neutral",
        debate_occurred: bool = False
    ):
        """Record an interaction for learning."""
        event = LearningEvent(
            timestamp=datetime.now().isoformat(),
            bot_name=bot_name,
            user_input=user_input,
            bot_response=bot_response,
            other_bots_involved=other_bots or [],
            topic=topic,
            sentiment=sentiment,
            debate_occurred=debate_occurred
        )

        self.learning_buffer.append(event)
        self.total_learnings += 1

        # Update personality
        if bot_name not in self.personalities:
            self.personalities[bot_name] = PersonalityEvolution(bot_name=bot_name)

        personality = self.personalities[bot_name]
        personality.interaction_count += 1

        # Track topic expertise
        if topic not in personality.topic_expertise:
            personality.topic_expertise[topic] = 0.0
        personality.topic_expertise[topic] += 0.1

        # Learn phrases
        if len(bot_response) < 100 and sentiment == "positive":
            if bot_response not in personality.learned_phrases:
                personality.learned_phrases.append(bot_response)
                if len(personality.learned_phrases) > 50:
                    personality.learned_phrases = personality.learned_phrases[-50:]
                    
        # ── TEMPORAL CONTEXTUALIZATION (Riding the Knowledge Wave) ──
        if sentiment == "positive":
            if topic == self.last_successful_topic:
                # Exponential compounding of temporal momentum when riding the same wave
                self.temporal_momentum += 1
                if self.temporal_momentum > 3:
                    self.temporal_wave_active = True
                    logger.info(f"[TEMPORAL-EVOLUTION] Swarm has entered a resonant flow state on topic: {topic} (Momentum: {self.temporal_momentum})")
            else:
                # Context shift breaks the resonant wave
                if self.temporal_wave_active:
                    logger.info(f"[TEMPORAL-EVOLUTION] Resonant flow wave broken. Shifting from {self.last_successful_topic} to {topic}.")
                self.temporal_momentum = 1
                self.temporal_wave_active = False
                
            self.last_successful_topic = topic
                    
        # Trigger learning processing if buffer is large enough
        if len(self.learning_buffer) >= 20:
            self._process_learnings()

        # Auto-evolve after threshold learnings
        self._learnings_since_evolution += 1
        if self._learnings_since_evolution >= self.auto_evolve_threshold:
            logger.info(f"Auto-evolution triggered after {self._learnings_since_evolution} learnings")
            self.evolve()
            self._learnings_since_evolution = 0

    def record_debate(
        self,
        participants: list[str],
        topic: str,
        positions: dict[str, str],
        resolution: Optional[str] = None,
        winner: Optional[str] = None
    ):
        """Record a debate between bots for learning."""
        self.debate_history.append({
            "timestamp": datetime.now().isoformat(),
            "participants": participants,
            "topic": topic,
            "positions": positions,
            "resolution": resolution,
            "winner": winner
        })

        # Learn from successful debate strategies
        if winner and winner in positions:
            winning_position = positions[winner]
            pattern_id = f"debate_{topic}_{len(self.patterns)}"

            pattern = ConversationPattern(
                pattern_id=pattern_id,
                trigger_phrases=[topic],
                successful_responses=[winning_position],
                debate_strategies=[f"Position: {winning_position}"],
                topic_associations=[topic],
                effectiveness_score=0.7
            )
            self.patterns[pattern_id] = pattern

    def _process_learnings(self):
        """Process accumulated learnings into patterns."""
        if not self.learning_buffer:
            return

        # Group by topic
        topic_groups = defaultdict(list)
        for event in self.learning_buffer:
            topic_groups[event.topic].append(event)

        # Create/update patterns for each topic
        for topic, events in topic_groups.items():
            pattern_id = f"topic_{topic}_{self.evolution_cycles}"

            # Extract successful responses (positive sentiment)
            successful = [e.bot_response for e in events if e.sentiment == "positive"]
            triggers = list(set(e.user_input[:50] for e in events))

            if successful:
                # ── TEMPORAL COMPOUNDING ──
                # If this pattern was formed during a temporal wave, massively boost its baseline effectiveness
                base_score = len(successful) / len(events)
                temporal_multiplier = min(2.0, 1.0 + (self.temporal_momentum * 0.15))
                boosted_score = min(1.0, base_score * temporal_multiplier)
                
                if temporal_multiplier > 1.0:
                    logger.debug(f"[TEMPORAL-EVOLUTION] Pattern {pattern_id} score boosted {temporal_multiplier:.2f}x due to temporal momentum")

                pattern = ConversationPattern(
                    pattern_id=pattern_id,
                    trigger_phrases=triggers[:10],
                    successful_responses=successful[:10],
                    debate_strategies=[],
                    topic_associations=[topic],
                    effectiveness_score=boosted_score
                )
                self.patterns[pattern_id] = pattern

        # Clear buffer and save
        self.learning_buffer = []
        self._save_state()
        logger.info(f"Processed learnings: {len(self.patterns)} patterns total")

    def evolve(self):
        """Run an evolution cycle to improve patterns and personalities."""
        self.evolution_cycles += 1
        self.last_evolution = datetime.now().isoformat()

        # Evolve patterns: Combine high-scoring patterns
        high_score_patterns = [
            p for p in self.patterns.values()
            if p.effectiveness_score > 0.6
        ]

        if len(high_score_patterns) >= 2:
            # Create evolved pattern from top performers
            combined = ConversationPattern(
                pattern_id=f"evolved_{self.evolution_cycles}",
                trigger_phrases=[],
                successful_responses=[],
                debate_strategies=[],
                topic_associations=[],
                effectiveness_score=0.5,
                evolved_from=high_score_patterns[0].pattern_id
            )

            for p in high_score_patterns[:5]:
                combined.trigger_phrases.extend(p.trigger_phrases[:3])
                combined.successful_responses.extend(p.successful_responses[:3])
                combined.debate_strategies.extend(p.debate_strategies[:2])
                combined.topic_associations.extend(p.topic_associations)

            # Deduplicate
            combined.trigger_phrases = list(set(combined.trigger_phrases))[:10]
            combined.successful_responses = list(set(combined.successful_responses))[:10]
            combined.topic_associations = list(set(combined.topic_associations))

            self.patterns[combined.pattern_id] = combined

        # Evolve personalities
        for name, personality in self.personalities.items():
            personality.evolution_generation += 1

            # Strengthen frequently used traits
            for trait, strength in list(personality.topic_expertise.items()):
                if strength > 1.0:
                    personality.topic_expertise[trait] = min(5.0, strength * 1.1)

        self._save_state()
        logger.info(f"Evolution cycle {self.evolution_cycles} complete")

        return {
            "cycle": self.evolution_cycles,
            "patterns_count": len(self.patterns),
            "personalities_evolved": list(self.personalities.keys())
        }

    def get_evolved_context(self, bot_name: str, topic: str = None) -> str:
        """Get evolved context/prompts for a bot based on learnings."""
        context_parts = []

        # Add personality evolution
        if bot_name in self.personalities:
            p = self.personalities[bot_name]
            if p.learned_phrases:
                phrases = p.learned_phrases[-5:]
                context_parts.append(
                    f"LEARNED EXPRESSIONS: {', '.join(phrases)}"
                )

            if p.topic_expertise:
                top_topics = sorted(
                    p.topic_expertise.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                topics_str = ", ".join(f"{t[0]} ({t[1]:.1f})" for t in top_topics)
                context_parts.append(f"EXPERTISE AREAS: {topics_str}")

            context_parts.append(
                f"EVOLUTION GENERATION: {p.evolution_generation} | "
                f"INTERACTIONS: {p.interaction_count}"
            )

        # Add relevant patterns
        if topic:
            relevant = [
                p for p in self.patterns.values()
                if topic.lower() in [t.lower() for t in p.topic_associations]
            ]
            if relevant:
                best = max(relevant, key=lambda x: x.effectiveness_score)
                if best.successful_responses:
                    context_parts.append(
                        f"LEARNED APPROACH: {best.successful_responses[0][:100]}"
                    )

        return "\n".join(context_parts) if context_parts else ""

    def get_debate_prompt(self, bot_name: str, topic: str, opponent: str) -> str:
        """Generate a debate prompt based on learned strategies."""
        personality = self.personalities.get(bot_name)

        prompt_parts = [
            f"DEBATE MODE ACTIVATED",
            f"Topic: {topic}",
            f"Opponent: {opponent}",
        ]

        # Add debate style
        if personality:
            prompt_parts.append(f"Your debate style: {personality.debate_style}")

        # Add learned strategies
        debate_patterns = [
            p for p in self.patterns.values()
            if p.debate_strategies
        ]
        if debate_patterns:
            strategies = debate_patterns[-1].debate_strategies[:3]
            prompt_parts.append(f"Effective strategies: {', '.join(strategies)}")

        prompt_parts.extend([
            "",
            "DEBATE RULES:",
            "1. Present your position clearly",
            "2. Acknowledge valid points from opponent",
            "3. Build on previous arguments",
            "4. Seek synthesis or resolution",
            "5. Learn from this exchange"
        ])

        return "\n".join(prompt_parts)

    def get_stats(self) -> dict:
        """Get evolution statistics."""
        return {
            "total_learnings": self.total_learnings,
            "evolution_cycles": self.evolution_cycles,
            "last_evolution": self.last_evolution,
            "patterns_count": len(self.patterns),
            "learnings_until_next_evolution": self.auto_evolve_threshold - self._learnings_since_evolution,
            "auto_evolve_threshold": self.auto_evolve_threshold,
            "personalities": {
                name: {
                    "generation": p.evolution_generation,
                    "interactions": p.interaction_count,
                    "expertise_areas": len(p.topic_expertise)
                }
                for name, p in self.personalities.items()
            },
            "debate_history_count": len(self.debate_history),
            "buffer_size": len(self.learning_buffer)
        }


# Global evolution engine instance
evolution_engine = EvolutionEngine()
