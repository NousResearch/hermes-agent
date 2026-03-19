"""
cosmos Synergy Engine.

"The whole is greater than the sum of its parts, especially when the parts are explosive!"

This module implements the "Global Learning Loop". It correlates events across 
disconnected domains (e.g., External Integrations <-> Internal Projects <-> Neuromorphic Weights).

Functionality:
1. Event Correlation: Links external triggers (GitHub PR) to internal state (Project Task).
2. Cross-Domain Reinforcement: Success in one domain (Coding) boosts confidence in related domains (Debugging).
3. Knowledge Graph Densification: Adds edges between seemingly unrelated entities based on temporal co-occurrence.
"""

import asyncio
from loguru import logger

from Cosmos.core.nexus import nexus, Signal, SignalType
from Cosmos.core.neuromorphic.engine import neuro_engine
from Cosmos.core.learning.continual import continual_learner

from Cosmos.memory.project_tracking import ProjectTracker
from Cosmos.core.learning.paths import learning_copilot

class SynergyEngine:
    def __init__(self, project_tracker: ProjectTracker):
        self.project_tracker = project_tracker
        nexus.subscribe(SignalType.EXTERNAL_ALERT, self._handle_external_event)
        nexus.subscribe(SignalType.TASK_COMPLETED, self._handle_internal_success)
        nexus.subscribe(SignalType.USER_MESSAGE, self._handle_user_context)
        
    async def _handle_external_event(self, signal: Signal):
        """Correlate External Events -> Internal Projects."""
        data = signal.payload
        event_type = data.get("type")
        title = data.get("title", "")
        
        if event_type in ["pr_merge", "issue_closed", "email"]:
            # 1. Update Project Tracker (Semantic check)
            # Find tasks matching the external event title
            projects = await self.project_tracker.list_projects()
            for p in projects:
                for t in p.tasks:
                    if t.status == "pending" and (title.lower() in t.title.lower() or t.title.lower() in title.lower()):
                        await self.project_tracker.complete_task(t.id)
                        logger.success(f"Synergy: Auto-completed task '{t.title}' based on external {event_type}")

            # 2. Trigger Neuromorphic Reward
            await neuro_engine._update_weight("external_sync_success", 0.1)

    async def _handle_internal_success(self, signal: Signal):
        """Correlate Internal Success -> Skill Tree."""
        task_title = signal.payload.get("title", "")
        
        # Heuristic Skill Identification
        if any(w in ["fix", "bug", "issue", "error"]):
            learning_copilot.add_skill("Debugging", "Resolving code anomalies")
            # In real system: update progress
        elif any(w in ["implement", "add", "feature"]):
            learning_copilot.add_skill("Software Engineering", "Building complex systems")

    async def _handle_user_context(self, signal: Signal):
        """Correlate User Message -> Affective Resonance."""
        # This is handled primarily by ToM, but Synergy can trigger Memory Dream
        pass

# Global instance initialization happens in server.py
def create_synergy_engine(tracker):
    return SynergyEngine(tracker)
