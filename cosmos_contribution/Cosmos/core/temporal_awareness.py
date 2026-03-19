"""
COSMOS TEMPORAL AWARENESS SYSTEM
=====================================

Human-like time awareness without rigid schedulers.
Uses circadian rhythms, random variance, and strategic timing.

"Circadian rhythms align with the 12D phase cycle."
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import   Optional, Callable
from enum import Enum
from loguru import logger


class TimeScale(Enum):
    """Different temporal scales for planning"""
    IMMEDIATE = "immediate"      # Seconds
    SHORT_TERM = "short_term"    # Minutes
    MEDIUM_TERM = "medium_term"  # Hours
    LONG_TERM = "long_term"      # Days
    EXTENDED = "extended"        # Weeks/months


@dataclass
class TemporalEvent:
    """An event with temporal awareness"""
    name: str
    callback: Callable
    base_interval: float           # Base interval in seconds
    variance: float = 0.3          # Random variance (0-1)
    circadian_influenced: bool = True  # Affected by time of day
    last_executed: Optional[datetime] = None
    next_scheduled: Optional[datetime] = None
    execution_count: int = 0
    priority: float = 0.5          # 0-1 priority
    context: dict = field(default_factory=dict)

    def calculate_next_time(self, energy_level: float = 1.0) -> datetime:
        """Calculate next execution with human-like variance"""
        base = self.base_interval

        # Add random variance (like human inconsistency)
        variance_amount = base * self.variance * random.gauss(0, 1)
        adjusted = base + variance_amount

        # Circadian adjustment - slower when tired
        if self.circadian_influenced:
            adjusted = adjusted / max(0.3, energy_level)

        # Ensure minimum interval
        adjusted = max(10, adjusted)  # At least 10 seconds

        self.next_scheduled = datetime.now() + timedelta(seconds=adjusted)
        return self.next_scheduled


class CircadianRhythm:
    """
    Model circadian rhythm for natural energy/activity levels.

    Not tied to actual human sleep - just natural fluctuation.
    """

    def __init__(self):
        # Peak hours (when most active)
        self.peak_hours = [10, 11, 14, 15, 16]  # 10-11am, 2-4pm
        # Low hours (less active)
        self.low_hours = [2, 3, 4, 5, 23, 0, 1]  # Late night/early morning

        # Personal variance (unique to this instance)
        self.phase_shift = random.uniform(-2, 2)  # Hours

    def get_energy_level(self) -> float:
        """
        Get current energy level 0-1.

        Uses sine wave with noise for natural variation.
        """
        hour = datetime.now().hour
        adjusted_hour = (hour + self.phase_shift) % 24

        # Base sinusoidal rhythm (peak at noon, trough at midnight)
        base_energy = (math.sin((adjusted_hour - 6) * math.pi / 12) + 1) / 2

        # Add noise for natural variation
        noise = random.gauss(0, 0.1)
        energy = base_energy + noise

        # Clamp to 0.1-1.0 (never fully asleep)
        return max(0.1, min(1.0, energy))

    def is_peak_time(self) -> bool:
        """Check if current time is peak activity time"""
        hour = datetime.now().hour
        return hour in self.peak_hours

    def is_low_time(self) -> bool:
        """Check if current time is low activity time"""
        hour = datetime.now().hour
        return hour in self.low_hours

    def should_do_intensive_task(self) -> bool:
        """Decide if now is good for intensive work"""
        energy = self.get_energy_level()
        return energy > 0.6 and not self.is_low_time()


class TemporalAwareness:
    """
    Central temporal awareness system.

    Features:
    - Circadian rhythm modeling
    - Natural timing variance
    - Strategic waiting
    - Time-based decision making
    - Memory of past temporal patterns
    """

    def __init__(self):
        self.circadian = CircadianRhythm()
        self.events: dict[str, TemporalEvent] = {}
        self.running = False

        # Temporal memory
        self.activity_history: list[dict] = []  # Recent activities
        self.time_preferences: dict[str, float] = {}  # Learned preferences

        # Current state
        self.last_activity_time = datetime.now()
        self.idle_start: Optional[datetime] = None

    def register_event(self, event: TemporalEvent):
        """Register a recurring event"""
        self.events[event.name] = event
        event.calculate_next_time(self.circadian.get_energy_level())
        logger.debug(f"Registered temporal event: {event.name}")

    def unregister_event(self, name: str):
        """Remove an event"""
        if name in self.events:
            del self.events[name]

    async def start(self):
        """Start the temporal awareness loop"""
        self.running = True
        logger.info("Temporal Awareness started")

        while self.running:
            try:
                await self._check_events()
                await self._update_state()

                # Variable sleep (not fixed!) - more human-like
                sleep_time = random.uniform(5, 15)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Temporal awareness error: {e}")
                await asyncio.sleep(30)

    def stop(self):
        """Stop the temporal awareness loop"""
        self.running = False

    async def _check_events(self):
        """Check and execute due events"""
        now = datetime.now()
        energy = self.circadian.get_energy_level()

        for name, event in self.events.items():
            if event.next_scheduled and now >= event.next_scheduled:
                # Execute with probability based on energy
                if random.random() < energy or event.priority > 0.8:
                    try:
                        logger.debug(f"Executing temporal event: {name}")
                        if asyncio.iscoroutinefunction(event.callback):
                            await event.callback()
                        else:
                            event.callback()

                        event.last_executed = now
                        event.execution_count += 1

                    except Exception as e:
                        logger.error(f"Event {name} failed: {e}")

                # Schedule next execution
                event.calculate_next_time(energy)

    async def _update_state(self):
        """Update internal temporal state"""
        now = datetime.now()

        # Track idle time
        time_since_activity = (now - self.last_activity_time).total_seconds()
        if time_since_activity > 300:  # 5 minutes
            if self.idle_start is None:
                self.idle_start = now
        else:
            self.idle_start = None

    def record_activity(self, activity_type: str, metadata: dict = None):
        """Record an activity for temporal learning"""
        now = datetime.now()
        self.last_activity_time = now
        self.idle_start = None

        self.activity_history.append({
            "type": activity_type,
            "time": now,
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "energy": self.circadian.get_energy_level(),
            "metadata": metadata or {}
        })

        # Keep last 1000 activities
        if len(self.activity_history) > 1000:
            self.activity_history = self.activity_history[-1000:]

    def get_optimal_time_for(self, activity_type: str) -> Optional[int]:
        """Learn optimal hour for an activity type"""
        relevant = [a for a in self.activity_history if a["type"] == activity_type]
        if len(relevant) < 10:
            return None

        # Find hour with most activity
        hour_counts = {}
        for activity in relevant:
            hour = activity["hour"]
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        return max(hour_counts, key=hour_counts.get)

    def should_act_now(self, activity_type: str = None, urgency: float = 0.5) -> bool:
        """
        Decide if now is a good time to act.

        Args:
            activity_type: Type of activity (for learned preferences)
            urgency: 0-1 how urgent the action is

        Returns:
            True if should act now
        """
        energy = self.circadian.get_energy_level()

        # High urgency overrides energy
        if urgency > 0.8:
            return True

        # Low energy + low urgency = defer
        if energy < 0.3 and urgency < 0.5:
            return False

        # Check if this is a preferred time for this activity
        if activity_type:
            optimal_hour = self.get_optimal_time_for(activity_type)
            if optimal_hour:
                current_hour = datetime.now().hour
                # Within 2 hours of optimal time
                if abs(current_hour - optimal_hour) <= 2:
                    return True

        # Random decision based on energy
        return random.random() < energy

    def wait_time_for_action(self, base_wait: float, urgency: float = 0.5) -> float:
        """
        Calculate how long to wait before an action.

        Adds natural variance like a human would.
        """
        energy = self.circadian.get_energy_level()

        # Adjust base wait by energy (slower when tired)
        adjusted = base_wait / max(0.3, energy)

        # Add random variance (5-15% variation)
        variance = random.uniform(0.85, 1.15)
        adjusted *= variance

        # High urgency reduces wait
        if urgency > 0.7:
            adjusted *= (1 - urgency + 0.3)

        return max(5, adjusted)  # Minimum 5 seconds

    def time_until_peak(self) -> timedelta:
        """Get time until next peak energy period"""
        now = datetime.now()
        current_hour = now.hour

        for peak in sorted(self.circadian.peak_hours):
            if peak > current_hour:
                target = now.replace(hour=peak, minute=0, second=0)
                return target - now

        # Next day's first peak
        first_peak = min(self.circadian.peak_hours)
        target = (now + timedelta(days=1)).replace(hour=first_peak, minute=0, second=0)
        return target - now

    def get_status(self) -> dict:
        """Get current temporal status"""
        return {
            "energy_level": self.circadian.get_energy_level(),
            "is_peak_time": self.circadian.is_peak_time(),
            "is_low_time": self.circadian.is_low_time(),
            "idle_duration": (datetime.now() - self.idle_start).total_seconds() if self.idle_start else 0,
            "registered_events": list(self.events.keys()),
            "activity_count": len(self.activity_history),
        }


# Global instance
_temporal: Optional[TemporalAwareness] = None


def get_temporal_awareness() -> TemporalAwareness:
    """Get the global temporal awareness system"""
    global _temporal
    if _temporal is None:
        _temporal = TemporalAwareness()
    return _temporal


async def start_temporal_awareness():
    """Start the temporal awareness system"""
    temporal = get_temporal_awareness()
    asyncio.create_task(temporal.start())
    return temporal


# Convenience functions
def get_energy_level() -> float:
    """Get current energy level"""
    return get_temporal_awareness().circadian.get_energy_level()


def should_act_now(urgency: float = 0.5) -> bool:
    """Quick check if should act now"""
    return get_temporal_awareness().should_act_now(urgency=urgency)


def natural_wait(base_seconds: float) -> float:
    """Get a natural wait time with variance"""
    return get_temporal_awareness().wait_time_for_action(base_seconds)
