"""
cosmos Productivity Tools

A collection of productivity utilities:
- Focus Mode (Cone of Silence) - Distraction blocking
- Boomerang - Task resurfacing
- Quick Notes - Fast note-taking
- Snippet Manager - Code snippet storage
- Focus Timer - Pomodoro-style timer
- Daily Summary - Activity digest
"""

from .focus_mode import ConeOfSilence
from .boomerang import Boomerang
from .quick_notes import QuickNotes, quick_notes
from .snippet_manager import SnippetManager, snippet_manager
from .focus_timer import FocusTimer, focus_timer
from .daily_summary import DailySummaryGenerator, daily_summary

__all__ = [
    "ConeOfSilence",
    "Boomerang",
    "QuickNotes",
    "quick_notes",
    "SnippetManager",
    "snippet_manager",
    "FocusTimer",
    "focus_timer",
    "DailySummaryGenerator",
    "daily_summary",
]
