"""
Upgrade Extractor - NLP analysis of swarm conversations to identify upgrade suggestions.

This module analyzes chat history to automatically detect when the swarm or users
suggest improvements, new features, or upgrades. These are then prioritized and
fed into the evolution loop as tasks.

Patterns detected:
- "we should add/build/implement..."
- "would be nice to have..."
- "let's upgrade/improve/enhance..."
- "need to fix/change..."
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

# Patterns that indicate upgrade/feature suggestions
UPGRADE_PATTERNS = [
    # Direct suggestions
    (r"(?:we should|let's|need to|could we|can we) (?:build|add|implement|create|make) (.+?)(?:\.|!|\?|$)", "suggestion"),
    (r"(?:would be nice|good idea|great idea) to (?:have|add|build|implement) (.+?)(?:\.|!|\?|$)", "wishlist"),
    (r"(?:upgrade|improve|enhance|fix|optimize) (?:the |our )?(.+?)(?:\.|!|\?|$)", "improvement"),

    # Problem statements that imply needed fixes
    (r"(?:the problem is|issue is|bug in|broken) (.+?)(?:\.|!|\?|$)", "bug"),
    (r"(.+?) (?:doesn't work|is broken|needs fixing|has issues)", "bug"),

    # Feature requests
    (r"(?:feature request|i want|we need) (.+?)(?:\.|!|\?|$)", "feature"),
    (r"(?:add support for|integrate with|connect to) (.+?)(?:\.|!|\?|$)", "integration"),

    # Architecture suggestions
    (r"(?:refactor|restructure|redesign|rewrite) (.+?)(?:\.|!|\?|$)", "refactor"),
    (r"(?:modularize|separate|split) (.+?)(?:\.|!|\?|$)", "refactor"),

    # Performance suggestions
    (r"(?:speed up|make faster|optimize|cache) (.+?)(?:\.|!|\?|$)", "performance"),
    (r"(.+?) (?:is slow|takes too long|needs optimization)", "performance"),

    # New capabilities
    (r"(?:give .+ ability to|enable .+ to|allow .+ to) (.+?)(?:\.|!|\?|$)", "capability"),
    (r"(?:add .+ capability|new feature for) (.+?)(?:\.|!|\?|$)", "capability"),
]

# Keywords that boost priority
PRIORITY_KEYWORDS = {
    "critical": 3,
    "important": 2,
    "urgent": 3,
    "asap": 3,
    "priority": 2,
    "must have": 3,
    "essential": 2,
    "security": 3,
    "vulnerability": 3,
    "memory": 1,
    "performance": 1,
    "evolution": 2,
    "consciousness": 2,
}

# Keywords that indicate technical feasibility
FEASIBILITY_KEYWORDS = {
    "simple": 2,
    "easy": 2,
    "quick": 2,
    "straightforward": 2,
    "complex": -1,
    "difficult": -1,
    "hard": -1,
    "challenging": -1,
}


class UpgradeExtractor:
    """Extracts and prioritizes upgrade suggestions from chat history."""

    def __init__(self):
        self.extracted_upgrades: list[dict] = []
        self.pattern_cache = [(re.compile(p, re.IGNORECASE), cat) for p, cat in UPGRADE_PATTERNS]

    def extract_from_message(self, message: str, speaker: str = None) -> list[dict]:
        """
        Extract upgrade suggestions from a single message.

        Args:
            message: The chat message text
            speaker: Who said it (affects priority - Cosmos/Claude suggestions weighted higher)

        Returns:
            list of extracted upgrades with metadata
        """
        upgrades = []
        message_lower = message.lower()

        for pattern, category in self.pattern_cache:
            matches = pattern.findall(message)
            for match in matches:
                # Clean up the match
                suggestion = match.strip() if isinstance(match, str) else match[0].strip()

                # Skip very short or very long matches (likely noise)
                if len(suggestion) < 5 or len(suggestion) > 200:
                    continue

                # Calculate priority score
                priority = self._calculate_priority(message_lower, speaker)

                # Calculate feasibility score
                feasibility = self._calculate_feasibility(message_lower)

                upgrade = {
                    "suggestion": suggestion,
                    "category": category,
                    "priority": priority,
                    "feasibility": feasibility,
                    "speaker": speaker,
                    "timestamp": datetime.now().isoformat(),
                    "original_message": message[:200],
                    "score": priority + feasibility  # Combined score for ranking
                }
                upgrades.append(upgrade)

        return upgrades

    def _calculate_priority(self, message: str, speaker: str = None) -> int:
        """Calculate priority score based on keywords and speaker."""
        score = 5  # Base priority

        # Check for priority keywords
        for keyword, boost in PRIORITY_KEYWORDS.items():
            if keyword in message:
                score += boost

        # Speaker boost (core swarm members have more weight)
        if speaker:
            speaker_lower = speaker.lower()
            if speaker_lower in ("cosmos", "claude", "grok"):
                score += 2
            elif speaker_lower in ("kimi", "deepseek", "gemini"):
                score += 1
            elif speaker_lower not in ("phi", "swarm-mind"):
                # Human users get slight boost
                score += 1

        return min(score, 10)  # Cap at 10

    def _calculate_feasibility(self, message: str) -> int:
        """Calculate feasibility score based on complexity indicators."""
        score = 5  # Base feasibility

        for keyword, modifier in FEASIBILITY_KEYWORDS.items():
            if keyword in message:
                score += modifier

        return max(1, min(score, 10))  # Clamp between 1-10

    def extract_from_history(self, chat_history: list[dict], limit: int = 100) -> list[dict]:
        """
        Extract upgrades from chat history.

        Args:
            chat_history: list of chat messages with 'content', 'bot_name'/'user_name'
            limit: Maximum messages to analyze

        Returns:
            list of extracted upgrades, deduplicated and sorted by score
        """
        all_upgrades = []

        # Process recent messages
        recent = chat_history[-limit:] if len(chat_history) > limit else chat_history

        for msg in recent:
            content = msg.get("content", "")
            speaker = msg.get("bot_name") or msg.get("user_name", "Unknown")

            upgrades = self.extract_from_message(content, speaker)
            all_upgrades.extend(upgrades)

        # Deduplicate similar suggestions
        deduped = self._deduplicate(all_upgrades)

        # Sort by score (highest first)
        deduped.sort(key=lambda x: x["score"], reverse=True)

        return deduped

    def _deduplicate(self, upgrades: list[dict]) -> list[dict]:
        """Remove duplicate or very similar suggestions."""
        if not upgrades:
            return []

        seen = set()
        unique = []

        for upgrade in upgrades:
            # Create a normalized key for comparison
            suggestion = upgrade["suggestion"].lower()
            # Remove common words for comparison
            words = set(suggestion.split()) - {"the", "a", "an", "to", "for", "with", "and", "or"}
            key = frozenset(words)

            # Check if we've seen something similar
            if key not in seen and len(key) >= 2:
                seen.add(key)
                unique.append(upgrade)

        return unique


async def extract_upgrades(chat_history: list[dict], limit: int = 50) -> list[str]:
    """
    Convenience function to extract upgrade suggestions from chat.

    Args:
        chat_history: Recent chat messages
        limit: Max messages to analyze

    Returns:
        list of upgrade suggestion strings (just the text)
    """
    extractor = UpgradeExtractor()
    upgrades = extractor.extract_from_history(chat_history, limit)
    return [u["suggestion"] for u in upgrades[:10]]  # Top 10


async def prioritize_upgrades(upgrades: list[str]) -> list[dict]:
    """
    Prioritize upgrade suggestions using a model assessment.

    For now, uses simple keyword scoring. Could be enhanced with
    model-based feasibility assessment.

    Args:
        upgrades: list of upgrade suggestion strings

    Returns:
        Sorted list of dicts with suggestion, priority, feasibility
    """
    extractor = UpgradeExtractor()
    prioritized = []

    for suggestion in upgrades:
        # Create a pseudo-message for scoring
        priority = extractor._calculate_priority(suggestion.lower())
        feasibility = extractor._calculate_feasibility(suggestion.lower())

        prioritized.append({
            "suggestion": suggestion,
            "priority": priority,
            "feasibility": feasibility,
            "score": priority + feasibility,
            "task_description": f"Implement: {suggestion}"
        })

    # Sort by combined score
    prioritized.sort(key=lambda x: x["score"], reverse=True)
    return prioritized


def integrate_with_evolution_loop(evolution_loop, swarm_manager):
    """
    Integrate upgrade extraction into the evolution loop's discovery phase.

    This patches the evolution loop to extract upgrades from chat.
    """
    original_discovery = evolution_loop._task_discovery_loop

    async def enhanced_discovery():
        """Enhanced discovery that also checks chat for upgrade suggestions."""
        import asyncio
        from Cosmos.core.agent_spawner import get_spawner, TaskType

        await asyncio.sleep(120)  # Initial delay

        while evolution_loop.running:
            try:
                spawner = get_spawner()
                pending = len(spawner.get_pending_tasks())

                if pending < 3:
                    # Original task generation
                    new_tasks = evolution_loop._generate_new_tasks(spawner)

                    # Also extract upgrades from recent chat
                    if swarm_manager and hasattr(swarm_manager, 'chat_history'):
                        history = list(swarm_manager.chat_history)[-100:]
                        upgrades = await extract_upgrades(history, limit=100)
                        prioritized = await prioritize_upgrades(upgrades)

                        # Add top 3 upgrades as tasks
                        for upgrade in prioritized[:3]:
                            # Map category to task type
                            task_type = TaskType.DEVELOPMENT  # Default

                            new_tasks.append({
                                "type": task_type,
                                "desc": upgrade["task_description"],
                                "agent": "DeepSeek",  # Default to DeepSeek for dev tasks
                                "source": "conversation"
                            })

                        logger.info(f"Extracted {len(prioritized)} upgrades from chat, adding top 3")

                    for task_def in new_tasks:
                        spawner.add_task(
                            task_type=task_def["type"],
                            description=task_def["desc"],
                            assigned_to=task_def["agent"],
                            priority=6
                        )

                    logger.info(f"Generated {len(new_tasks)} new evolution tasks")

                await asyncio.sleep(180)

            except Exception as e:
                logger.error(f"Enhanced discovery error: {e}")
                await asyncio.sleep(60)

    # Replace the discovery loop
    evolution_loop._task_discovery_loop = enhanced_discovery
    logger.info("Upgrade extractor integrated with evolution loop")


# Global instance for convenience
_extractor: Optional[UpgradeExtractor] = None


def get_extractor() -> UpgradeExtractor:
    """Get or create the global upgrade extractor."""
    global _extractor
    if _extractor is None:
        _extractor = UpgradeExtractor()
    return _extractor
