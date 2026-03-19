"""
cosmos Meeting Assistant.

"Good news! I've prepared a briefing so you can pretend to listen!"

This module provides intelligent meeting support:
1. Context Recall: Fetches relevant memories/tasks for a meeting topic.
2. Briefing Generation: Creates a 1-page summary.
3. Action Item Extraction: Parses notes for tasks.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from Cosmos.memory.project_tracking import ProjectTracker, Task
from Cosmos.core.nexus import nexus, Signal, SignalType

@dataclass
class MeetingContext:
    topic: str
    participants: list[str]
    time: datetime
    relevant_memories: list[str] = field(default_factory=list)
    relevant_tasks: list[Task] = field(default_factory=list)

class MeetingAssistant:
    def __init__(self, tracker: ProjectTracker):
        self.tracker = tracker

    async def prepare_briefing(self, topic: str, participants: list[str]) -> str:
        """
        Generate a meeting briefing.
        """
        # 1. Recall Memory (Mock implementation of semantic search)
        # In real system: memories = await memory_system.search(topic)
        memories = [
            f"Discussed {topic} architecture on Monday.",
            f"User {participants[0]} mentioned concern about latency."
        ]
        
        # 2. Find Relevant Tasks
        # In real system: tasks = await tracker.search_tasks(topic)
        tasks = [] # Mock
        
        # 3. Generate Briefing
        briefing = f"""# 📅 Meeting Briefing: {topic}
**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Participants**: {', '.join(participants)}

## 🧠 Context Recall
"""
        for m in memories:
            briefing += f"- {m}\n"
            
        briefing += "\n## 📋 Relevant Tasks\n"
        if tasks:
            for t in tasks:
                 briefing += f"- [ ] {t.title} ({t.status})\n"
        else:
            briefing += "- *No active tasks found linked to this topic.*\n"
            
        briefing += "\n## 🎯 Recommended Objectives\n"
        briefing += "- [ ] Agree on timeline\n- [ ] Assign blockers\n"
        
        return briefing

    async def extract_action_items(self, meeting_notes: str) -> list[dict]:
        """
        Parse meeting notes for action items (heuristics/LLM).
        """
        actions = []
        lines = meeting_notes.split('\n')
        for line in lines:
            if line.strip().lower().startswith("action:") or line.strip().startswith("[]") or line.strip().startswith("- [ ]"):
                clean = line.replace("Action:", "").replace("[]", "").replace("- [ ]", "").strip()
                actions.append({
                    "title": clean,
                    "status": "pending",
                    "source": "meeting_notes"
                })
                
                # Signal task creation
                await nexus.emit(SignalType.TASK_CREATED, {"title": clean, "origin": "meeting"}, "meeting_assistant")
                
        return actions

# Global instance requires tracker, so we leave it factory-based or inject later
def create_meeting_assistant(tracker):
    return MeetingAssistant(tracker)
