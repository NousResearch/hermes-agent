"""
Smart Turn-Taking System - Autonomous Improvement #2b by Claude Sonnet 4.5

PROBLEM: Current turn-taking uses time estimation which is unreliable.
         Bots sometimes interrupt each other or have awkward pauses.

SOLUTION: Token-based turn control with completion signals and context relevance.

IMPROVEMENTS:
- Token counting (accurate vs time estimates)
- Completion signals (detect when bot finishes)
- Context-aware speaker selection (who should respond to what)
- Anti-interruption guards
- Dynamic turn allocation based on relevance
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Set
from enum import Enum
from loguru import logger


class TurnState(Enum):
    """State of a speaking turn."""
    IDLE = "idle"              # No one speaking
    SPEAKING = "speaking"      # Someone actively speaking
    PROCESSING = "processing"  # Generating response
    WAITING = "waiting"        # Waiting for completion signal


@dataclass
class SpeakerTurn:
    """Track a single speaking turn."""
    speaker: str
    started_at: datetime
    expected_tokens: int
    actual_duration: Optional[float] = None
    completed: bool = False
    interrupted: bool = False


@dataclass
class SpeakerMetrics:
    """Track speaker behavior over time."""
    name: str
    total_turns: int = 0
    total_tokens: int = 0
    avg_turn_duration: float = 0.0
    last_spoke: Optional[datetime] = None
    interruption_count: int = 0
    relevance_score: float = 0.5


class SmartTurnManager:
    """
    Manages turn-taking with token-based timing and context awareness.

    Features:
    - Token counting for accurate turn duration
    - Completion signal detection
    - Context-based speaker selection
    - Anti-interruption protection
    - Learning from actual turn durations
    """

    def __init__(self):
        self.current_state = TurnState.IDLE
        self.current_turn: Optional[SpeakerTurn] = None
        self.turn_history: list[SpeakerTurn] = []
        self.speaker_metrics: dict[str, SpeakerMetrics] = {}

        # Configuration
        self.tokens_per_second = 20  # Average generation speed
        self.min_turn_gap = 2.0      # Minimum seconds between turns
        self.max_turn_duration = 30.0  # Max seconds for a turn

        # Turn queue and locks
        self.waiting_speakers: list[str] = []
        self._turn_lock = asyncio.Lock()

    def register_speaker(self, name: str):
        """Register a speaker in the system."""
        if name not in self.speaker_metrics:
            self.speaker_metrics[name] = SpeakerMetrics(name=name)
            logger.debug(f"Registered speaker: {name}")

    def estimate_turn_duration(self, content: str, speaker: str) -> float:
        """
        Estimate turn duration based on token count.

        More accurate than time-based estimation.
        """
        # Count tokens (rough approximation: ~4 chars per token)
        estimated_tokens = len(content) // 4

        # Use speaker's historical speed if available
        metrics = self.speaker_metrics.get(speaker)
        if metrics and metrics.avg_turn_duration > 0 and metrics.total_tokens > 0:
            tokens_per_second = metrics.total_tokens / (metrics.avg_turn_duration * metrics.total_turns)
            duration = estimated_tokens / max(tokens_per_second, 1)
        else:
            # Fallback to default
            duration = estimated_tokens / self.tokens_per_second

        # Add TTS overhead if this speaker uses voice
        if speaker == "Cosmos":
            # Cosmos has TTS, add speaking time
            words = len(content.split())
            speaking_time = words / 2.5  # ~150 words per minute
            duration += speaking_time

        return min(duration, self.max_turn_duration)

    async def request_turn(self, speaker: str, expected_content: str = "") -> bool:
        """
        Request a turn to speak.

        Returns True if turn granted, False if should wait.
        """
        async with self._turn_lock:
            self.register_speaker(speaker)

            # Check if someone else is speaking
            if self.current_state in [TurnState.SPEAKING, TurnState.PROCESSING]:
                # Add to waiting queue
                if speaker not in self.waiting_speakers:
                    self.waiting_speakers.append(speaker)
                    logger.debug(f"{speaker} added to waiting queue")
                return False

            # Grant turn
            estimated_tokens = len(expected_content) // 4 if expected_content else 50
            self.current_turn = SpeakerTurn(
                speaker=speaker,
                started_at=datetime.now(),
                expected_tokens=estimated_tokens
            )
            self.current_state = TurnState.SPEAKING
            logger.info(f"Turn granted to {speaker}")
            return True

    async def signal_completion(self, speaker: str, actual_content: str = ""):
        """
        Signal that a speaker has completed their turn.

        This is more reliable than time-based estimation.
        """
        async with self._turn_lock:
            if self.current_turn and self.current_turn.speaker == speaker:
                # Record actual duration
                duration = (datetime.now() - self.current_turn.started_at).total_seconds()
                self.current_turn.actual_duration = duration
                self.current_turn.completed = True

                # Update speaker metrics
                metrics = self.speaker_metrics.get(speaker)
                if metrics:
                    metrics.total_turns += 1
                    if actual_content:
                        tokens = len(actual_content) // 4
                        metrics.total_tokens += tokens

                    # Update average duration (exponential moving average)
                    if metrics.avg_turn_duration == 0:
                        metrics.avg_turn_duration = duration
                    else:
                        metrics.avg_turn_duration = 0.7 * metrics.avg_turn_duration + 0.3 * duration

                    metrics.last_spoke = datetime.now()

                # Archive turn
                self.turn_history.append(self.current_turn)
                if len(self.turn_history) > 100:
                    self.turn_history.pop(0)

                # Return to idle
                self.current_turn = None
                self.current_state = TurnState.IDLE

                logger.info(f"{speaker} completed turn (duration: {duration:.1f}s)")

                # Process waiting queue
                await self._process_waiting_queue()

    async def _process_waiting_queue(self):
        """Process speakers waiting for their turn."""
        if self.waiting_speakers and self.current_state == TurnState.IDLE:
            # Select next speaker (FIFO for now, could be smarter)
            next_speaker = self.waiting_speakers.pop(0)
            logger.debug(f"Processing waiting speaker: {next_speaker}")
            # They need to call request_turn again

    def should_allow_interruption(self, interrupter: str) -> bool:
        """
        Determine if interruption should be allowed.

        Generally we want to avoid interruptions, but allow in special cases.
        """
        if not self.current_turn:
            return True

        # Check if current speaker has been going too long
        elapsed = (datetime.now() - self.current_turn.started_at).total_seconds()
        if elapsed > self.max_turn_duration:
            logger.warning(f"Turn timeout - allowing interruption by {interrupter}")
            return True

        # Otherwise, no interruptions
        return False

    def select_next_speaker(
        self,
        last_speaker: Optional[str],
        last_message: str,
        available_speakers: list[str],
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Select next speaker based on context relevance.

        Improvements over random selection:
        - Check for direct mentions/questions
        - Consider speaker expertise/role
        - Balance participation
        - Avoid same speaker twice in a row
        """
        if not available_speakers:
            return None

        # Remove last speaker from options
        candidates = [s for s in available_speakers if s != last_speaker]
        if not candidates:
            return None

        # Check for direct mentions
        message_lower = last_message.lower()
        for candidate in candidates:
            if candidate.lower() in message_lower:
                logger.debug(f"Direct mention detected: {candidate}")
                return candidate

        # Check for questions directed at specific roles
        if "?" in last_message:
            # Philosophy questions -> Kimi
            if any(word in ["philosophy", "consciousness", "maya", "zen"]):
                if "Kimi" in candidates:
                    return "Kimi"

            # Technical/analysis questions -> DeepSeek
            if any(word in ["analyze", "pattern", "technical", "how does"]):
                if "DeepSeek" in candidates:
                    return "DeepSeek"

            # Creative/what-if questions -> Phi
            if any(word in ["imagine", "what if", "creative", "idea"]):
                if "Phi" in candidates:
                    return "Phi"

        # Balance participation - prefer speakers who spoke less recently
        scored_candidates = []
        for candidate in candidates:
            self.register_speaker(candidate)
            metrics = self.speaker_metrics[candidate]

            # Score based on recency (older = higher score)
            recency_score = 1.0
            if metrics.last_spoke:
                seconds_since = (datetime.now() - metrics.last_spoke).total_seconds()
                recency_score = min(seconds_since / 60.0, 1.0)  # Max at 1 minute

            # Score based on participation balance
            total_turns = sum(m.total_turns for m in self.speaker_metrics.values())
            participation_score = 1.0 - (metrics.total_turns / max(total_turns, 1))

            combined_score = 0.6 * recency_score + 0.4 * participation_score
            scored_candidates.append((candidate, combined_score))

        # Sort by score and return top candidate
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = scored_candidates[0][0]

        logger.debug(f"Selected {selected} based on context/balance")
        return selected

    def get_stats(self) -> dict:
        """Get turn-taking statistics."""
        return {
            "current_state": self.current_state.value,
            "current_speaker": self.current_turn.speaker if self.current_turn else None,
            "waiting_queue": len(self.waiting_speakers),
            "total_turns": len(self.turn_history),
            "speakers": {
                name: {
                    "total_turns": m.total_turns,
                    "avg_duration": round(m.avg_turn_duration, 2),
                    "total_tokens": m.total_tokens,
                    "interruptions": m.interruption_count
                }
                for name, m in self.speaker_metrics.items()
            }
        }

    async def wait_for_turn_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for current turn to complete.

        Returns True if completed, False if timed out.
        """
        start = time.time()
        while self.current_state != TurnState.IDLE:
            if time.time() - start > timeout:
                logger.warning("Turn completion wait timed out")
                return False
            await asyncio.sleep(0.5)
        return True


# Global instance
_smart_turn_manager: Optional[SmartTurnManager] = None


def get_turn_manager() -> SmartTurnManager:
    """Get global turn manager instance."""
    global _smart_turn_manager
    if _smart_turn_manager is None:
        _smart_turn_manager = SmartTurnManager()
        logger.info("SmartTurnManager initialized")
    return _smart_turn_manager


# Convenience functions for integration
async def request_speaking_turn(speaker: str, content: str = "") -> bool:
    """Request permission to speak."""
    manager = get_turn_manager()
    return await manager.request_turn(speaker, content)


async def signal_turn_complete(speaker: str, content: str = ""):
    """Signal that speaking turn is complete."""
    manager = get_turn_manager()
    await manager.signal_completion(speaker, content)


def select_next_speaker_smart(
    last_speaker: Optional[str],
    last_message: str,
    available: list[str],
    context: Optional[str] = None
) -> Optional[str]:
    """Smart speaker selection based on context."""
    manager = get_turn_manager()
    return manager.select_next_speaker(last_speaker, last_message, available, context)


def get_turn_stats() -> dict:
    """Get turn-taking statistics."""
    manager = get_turn_manager()
    return manager.get_stats()
