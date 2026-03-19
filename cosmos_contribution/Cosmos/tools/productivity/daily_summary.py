"""
cosmos Daily Summary - Automated Activity Digest

"What did you accomplish today? Let me remind you!"

Features:
- Auto-generated daily summaries
- Memory-based activity tracking
- Key accomplishments extraction
- Tomorrow's priorities
- Weekly/monthly rollups
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from loguru import logger


@dataclass
class DailySummary:
    """A daily activity summary."""
    date: str
    key_accomplishments: List[str] = field(default_factory=list)
    tasks_completed: int = 0
    memories_created: int = 0
    focus_minutes: int = 0
    conversations: int = 0
    insights: List[str] = field(default_factory=list)
    mood: str = "neutral"  # positive, neutral, challenging
    tomorrow_priorities: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DailySummaryGenerator:
    """Generate and manage daily summaries."""

    def __init__(self, data_dir: str = "./data", llm_fn: Callable = None):
        self.data_dir = Path(data_dir) / "summaries"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.llm_fn = llm_fn
        self.summaries: Dict[str, DailySummary] = {}
        self._load()

    def _load(self):
        """Load summaries from disk."""
        for file in self.data_dir.glob("summary_*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                summary = DailySummary(**data)
                self.summaries[summary.date] = summary
            except Exception as e:
                logger.debug(f"Failed to load summary {file}: {e}")

    def _save(self, summary: DailySummary):
        """Save a summary to disk."""
        try:
            file = self.data_dir / f"summary_{summary.date}.json"
            with open(file, "w") as f:
                json.dump(asdict(summary), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    async def generate_summary(
        self,
        date: str = None,
        memory_system=None,
        focus_timer=None,
        force: bool = False,
    ) -> DailySummary:
        """
        Generate a daily summary.

        Pulls data from memory system, focus timer, and other sources.
        """
        date = date or datetime.now().date().isoformat()

        if date in self.summaries and not force:
            return self.summaries[date]

        summary = DailySummary(date=date)

        # Gather data from memory system
        if memory_system:
            try:
                # Get today's memories
                today_start = f"{date}T00:00:00"
                today_end = f"{date}T23:59:59"

                # Count memories created today
                stats = memory_system.get_stats()
                summary.memories_created = stats.get("archival_memory", {}).get("total_entries", 0)

                # Try to get recent activities from memory
                recent = await memory_system.recall(
                    f"What did I work on today {date}?",
                    top_k=10
                )
                if recent:
                    summary.key_accomplishments = [
                        r.content[:100] for r in recent[:5]
                    ]

            except Exception as e:
                logger.debug(f"Memory integration failed: {e}")

        # Gather data from focus timer
        if focus_timer:
            try:
                today_stats = focus_timer.get_today_stats()
                summary.focus_minutes = today_stats.get("total_minutes", 0)
                summary.tasks_completed = today_stats.get("sessions", 0)
            except Exception:
                pass

        # Generate insights using LLM if available
        if self.llm_fn and summary.key_accomplishments:
            summary.insights = await self._generate_insights(summary)
            summary.tomorrow_priorities = await self._generate_priorities(summary)

        # Determine mood based on productivity
        if summary.focus_minutes > 120:
            summary.mood = "productive"
        elif summary.focus_minutes > 60:
            summary.mood = "moderate"
        elif summary.focus_minutes < 30:
            summary.mood = "light"

        self.summaries[date] = summary
        self._save(summary)

        logger.info(f"DailySummary: Generated summary for {date}")
        return summary

    async def _generate_insights(self, summary: DailySummary) -> List[str]:
        """Use LLM to generate insights from accomplishments."""
        if not self.llm_fn:
            return []

        try:
            accomplishments = "\n".join(f"- {a}" for a in summary.key_accomplishments)
            prompt = f"""Based on these accomplishments from today, provide 2-3 brief insights or patterns:

Accomplishments:
{accomplishments}

Insights (bullet points, max 2-3):"""

            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Parse bullet points
            insights = [
                line.strip("- ").strip()
                for line in response.split("\n")
                if line.strip().startswith("-") or line.strip().startswith("•")
            ]
            return insights[:3]

        except Exception as e:
            logger.debug(f"Insight generation failed: {e}")
            return []

    async def _generate_priorities(self, summary: DailySummary) -> List[str]:
        """Generate tomorrow's priorities based on today's work."""
        if not self.llm_fn:
            return []

        try:
            context = "\n".join(f"- {a}" for a in summary.key_accomplishments)
            prompt = f"""Based on today's work, suggest 2-3 priorities for tomorrow:

Today's work:
{context}

Tomorrow's priorities (bullet points, max 3):"""

            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            priorities = [
                line.strip("- ").strip()
                for line in response.split("\n")
                if line.strip().startswith("-") or line.strip().startswith("•")
            ]
            return priorities[:3]

        except Exception:
            return []

    def get_summary(self, date: str = None) -> Optional[DailySummary]:
        """Get a specific day's summary."""
        date = date or datetime.now().date().isoformat()
        return self.summaries.get(date)

    def get_week_summary(self) -> Dict[str, Any]:
        """Get a weekly summary rollup."""
        today = datetime.now().date()
        week_dates = [(today - timedelta(days=i)).isoformat() for i in range(7)]

        summaries = [self.summaries.get(d) for d in week_dates if d in self.summaries]

        if not summaries:
            return {"message": "No summaries available for this week"}

        return {
            "period": f"{week_dates[-1]} to {week_dates[0]}",
            "days_tracked": len(summaries),
            "total_focus_minutes": sum(s.focus_minutes for s in summaries),
            "total_tasks": sum(s.tasks_completed for s in summaries),
            "avg_daily_focus": sum(s.focus_minutes for s in summaries) / len(summaries),
            "total_memories": sum(s.memories_created for s in summaries),
            "top_accomplishments": [
                a for s in summaries for a in s.key_accomplishments
            ][:10],
        }

    def format_summary(self, summary: DailySummary) -> str:
        """Format a summary for display."""
        lines = [
            f"# Daily Summary - {summary.date}",
            "",
            f"**Mood:** {summary.mood}",
            f"**Focus Time:** {summary.focus_minutes} minutes",
            f"**Tasks Completed:** {summary.tasks_completed}",
            "",
        ]

        if summary.key_accomplishments:
            lines.append("## Key Accomplishments")
            for a in summary.key_accomplishments:
                lines.append(f"- {a}")
            lines.append("")

        if summary.insights:
            lines.append("## Insights")
            for i in summary.insights:
                lines.append(f"- {i}")
            lines.append("")

        if summary.tomorrow_priorities:
            lines.append("## Tomorrow's Priorities")
            for p in summary.tomorrow_priorities:
                lines.append(f"- {p}")

        return "\n".join(lines)


# Global instance
daily_summary = DailySummaryGenerator()
