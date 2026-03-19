"""
cosmos Focus Timer - Pomodoro-Style Productivity Timer

"Work smarter, not harder... unless you're a robot."

Features:
- Pomodoro technique (25/5 work/break cycles)
- Custom intervals
- Session tracking
- Break reminders
- Daily statistics
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

from loguru import logger


class TimerState(Enum):
    IDLE = "idle"
    WORKING = "working"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"
    PAUSED = "paused"


@dataclass
class FocusSession:
    """A completed focus session."""
    id: str
    start_time: str
    end_time: str
    duration_minutes: int
    state: str
    task_label: str = ""
    completed: bool = True


@dataclass
class TimerConfig:
    """Timer configuration."""
    work_minutes: int = 25
    short_break_minutes: int = 5
    long_break_minutes: int = 15
    sessions_until_long_break: int = 4
    auto_start_breaks: bool = True
    auto_start_work: bool = False
    sound_enabled: bool = True


class FocusTimer:
    """Pomodoro-style focus timer."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "focus"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.config = TimerConfig()
        self.state = TimerState.IDLE
        self.sessions_completed = 0
        self.current_task = ""

        self._timer_task: Optional[asyncio.Task] = None
        self._remaining_seconds = 0
        self._session_start: Optional[datetime] = None
        self._callbacks: List[Callable] = []

        # Session history
        self.history: List[FocusSession] = []
        self._load_history()

    def _load_history(self):
        """Load session history from disk."""
        history_file = self.data_dir / "focus_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                self.history = [FocusSession(**s) for s in data.get("sessions", [])]
                self.sessions_completed = data.get("sessions_today", 0)
            except Exception as e:
                logger.error(f"Failed to load focus history: {e}")

    def _save_history(self):
        """Save session history to disk."""
        try:
            with open(self.data_dir / "focus_history.json", "w") as f:
                json.dump({
                    "sessions": [asdict(s) for s in self.history[-100:]],  # Keep last 100
                    "sessions_today": self.sessions_completed,
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save focus history: {e}")

    def configure(
        self,
        work_minutes: int = None,
        short_break_minutes: int = None,
        long_break_minutes: int = None,
        sessions_until_long_break: int = None,
    ):
        """Update timer configuration."""
        if work_minutes:
            self.config.work_minutes = work_minutes
        if short_break_minutes:
            self.config.short_break_minutes = short_break_minutes
        if long_break_minutes:
            self.config.long_break_minutes = long_break_minutes
        if sessions_until_long_break:
            self.config.sessions_until_long_break = sessions_until_long_break

    def on_state_change(self, callback: Callable):
        """Register callback for state changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: str, data: Dict = None):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data or {})
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def start_work(self, task_label: str = ""):
        """Start a work session."""
        if self.state == TimerState.WORKING:
            return

        self.current_task = task_label
        self.state = TimerState.WORKING
        self._remaining_seconds = self.config.work_minutes * 60
        self._session_start = datetime.now()

        logger.info(f"Focus Timer: Starting work session ({self.config.work_minutes} min)")
        self._notify_callbacks("work_started", {"task": task_label})

        self._timer_task = asyncio.create_task(self._run_timer())

    async def start_break(self, is_long: bool = False):
        """Start a break session."""
        if is_long:
            self.state = TimerState.LONG_BREAK
            self._remaining_seconds = self.config.long_break_minutes * 60
            logger.info(f"Focus Timer: Starting long break ({self.config.long_break_minutes} min)")
        else:
            self.state = TimerState.SHORT_BREAK
            self._remaining_seconds = self.config.short_break_minutes * 60
            logger.info(f"Focus Timer: Starting short break ({self.config.short_break_minutes} min)")

        self._session_start = datetime.now()
        self._notify_callbacks("break_started", {"is_long": is_long})

        self._timer_task = asyncio.create_task(self._run_timer())

    async def pause(self):
        """Pause the current timer."""
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        self.state = TimerState.PAUSED
        self._notify_callbacks("paused", {})
        logger.info("Focus Timer: Paused")

    async def resume(self):
        """Resume a paused timer."""
        if self.state == TimerState.PAUSED and self._remaining_seconds > 0:
            self.state = TimerState.WORKING  # Assume work if resuming
            self._timer_task = asyncio.create_task(self._run_timer())
            self._notify_callbacks("resumed", {})
            logger.info("Focus Timer: Resumed")

    async def stop(self):
        """Stop the timer completely."""
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        if self._session_start and self.state != TimerState.IDLE:
            # Record partial session
            duration = (datetime.now() - self._session_start).seconds // 60
            if duration > 0:
                session = FocusSession(
                    id=f"{datetime.now().timestamp()}",
                    start_time=self._session_start.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_minutes=duration,
                    state=self.state.value,
                    task_label=self.current_task,
                    completed=False,
                )
                self.history.append(session)
                self._save_history()

        self.state = TimerState.IDLE
        self._remaining_seconds = 0
        self._session_start = None
        self._notify_callbacks("stopped", {})
        logger.info("Focus Timer: Stopped")

    async def _run_timer(self):
        """Run the countdown timer."""
        try:
            while self._remaining_seconds > 0:
                await asyncio.sleep(1)
                self._remaining_seconds -= 1

                # Notify every minute
                if self._remaining_seconds % 60 == 0:
                    self._notify_callbacks("tick", {
                        "remaining": self._remaining_seconds,
                        "state": self.state.value,
                    })

            # Timer complete
            await self._on_timer_complete()

        except asyncio.CancelledError:
            pass

    async def _on_timer_complete(self):
        """Handle timer completion."""
        completed_state = self.state
        duration = 0

        if self._session_start:
            duration = (datetime.now() - self._session_start).seconds // 60

            # Record completed session
            session = FocusSession(
                id=f"{datetime.now().timestamp()}",
                start_time=self._session_start.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_minutes=duration,
                state=completed_state.value,
                task_label=self.current_task,
                completed=True,
            )
            self.history.append(session)

        if completed_state == TimerState.WORKING:
            self.sessions_completed += 1
            self._save_history()

            # Determine next break type
            is_long_break = self.sessions_completed % self.config.sessions_until_long_break == 0

            self._notify_callbacks("work_complete", {
                "sessions_completed": self.sessions_completed,
                "duration": duration,
            })

            logger.info(f"Focus Timer: Work session complete! Session #{self.sessions_completed}")

            if self.config.auto_start_breaks:
                await self.start_break(is_long=is_long_break)

        elif completed_state in (TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
            self._notify_callbacks("break_complete", {})
            logger.info("Focus Timer: Break complete!")

            if self.config.auto_start_work:
                await self.start_work(self.current_task)
            else:
                self.state = TimerState.IDLE

    def get_status(self) -> Dict[str, Any]:
        """Get current timer status."""
        return {
            "state": self.state.value,
            "remaining_seconds": self._remaining_seconds,
            "remaining_display": f"{self._remaining_seconds // 60}:{self._remaining_seconds % 60:02d}",
            "current_task": self.current_task,
            "sessions_completed": self.sessions_completed,
            "config": asdict(self.config),
        }

    def get_today_stats(self) -> Dict[str, Any]:
        """Get today's focus statistics."""
        today = datetime.now().date().isoformat()
        today_sessions = [
            s for s in self.history
            if s.start_time.startswith(today) and s.state == "working" and s.completed
        ]

        return {
            "sessions": len(today_sessions),
            "total_minutes": sum(s.duration_minutes for s in today_sessions),
            "tasks": list(set(s.task_label for s in today_sessions if s.task_label)),
        }

    def get_weekly_stats(self) -> Dict[str, Any]:
        """Get this week's focus statistics."""
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        week_sessions = [
            s for s in self.history
            if s.start_time >= week_ago and s.state == "working" and s.completed
        ]

        return {
            "sessions": len(week_sessions),
            "total_minutes": sum(s.duration_minutes for s in week_sessions),
            "avg_daily_minutes": sum(s.duration_minutes for s in week_sessions) / 7,
        }


# Global instance
focus_timer = FocusTimer()
