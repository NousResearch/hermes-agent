"""
cosmos Learning Co-Pilot.

"I'm learning at a geometric rate!"

This module assists the user in mastering new concepts.
Features:
1. Learning Path Tracking: Tree-based skill progression.
2. Spaced Repetition System (SRS): Algorithms (SM-2) for flashcard scheduling.
3. Content Recommendation: Suggests next steps.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

@dataclass
class Flashcard:
    id: str
    front: str # Question / Concept
    back: str  # Answer / Definition
    
    # SRS Metadata
    interval: int = 0       # Days until next review
    repetition: int = 0     # Number of successful reviews
    efactor: float = 2.5    # Easiness factor (SM-2 algorithm)
    next_review: datetime = field(default_factory=datetime.now)

@dataclass
class SkillNode:
    id: str
    name: str
    description: str
    level: int = 0          # 0=Novice, 5=Master
    prerequisites: list[str] = field(default_factory=list)
    progress: float = 0.0   # 0 to 1.0

class SpacedRepetitionSystem:
    """
    Implements the SuperMemo-2 (SM-2) algorithm for optimum knowledge retention.
    """
    def process_review(self, card: Flashcard, quality: int):
        """
        Update card schedule based on user's recall quality (0-5).
        0=Blackout, 5=Perfect
        """
        if quality >= 3:
            if card.repetition == 0:
                card.interval = 1
            elif card.repetition == 1:
                card.interval = 6
            else:
                card.interval = int(card.interval * card.efactor)
            
            card.repetition += 1
            card.efactor = card.efactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            if card.efactor < 1.3:
                card.efactor = 1.3
        else:
            card.repetition = 0
            card.interval = 1
            
        card.next_review = datetime.now() + timedelta(days=card.interval)
        logger.info(f"SRS: Card '{card.id}' rescheduled for {card.next_review.date()} (Interval: {card.interval}d)")

class LearningPathManager:
    def __init__(self):
        self.skills: dict[str, SkillNode] = {}
        self.cards: list[Flashcard] = []
        self.srs = SpacedRepetitionSystem()

    def add_skill(self, name: str, description: str, prerequisites: list[str] = []) -> SkillNode:
        sid = name.lower().replace(" ", "_")
        node = SkillNode(id=sid, name=name, description=description, prerequisites=prerequisites)
        self.skills[sid] = node
        return node

    def suggest_next_step(self) -> list[SkillNode]:
        """
        Returns skills where prerequisites are met but progress < 100%.
        """
        available = []
        for skill in self.skills.values():
            if skill.progress >= 1.0:
                continue
            
            # Check prereqs
            all_met = True
            for pid in skill.prerequisites:
                if pid not in self.skills or self.skills[pid].progress < 0.8: # Require 80% mastery of prereq
                    all_met = False
                    break
            
            if all_met:
                available.append(skill)
        
        return available

    def get_review_queue(self) -> list[Flashcard]:
        """Get cards due for review today."""
        now = datetime.now()
        return [c for c in self.cards if c.next_review <= now]

# Global Instance
learning_copilot = LearningPathManager()
