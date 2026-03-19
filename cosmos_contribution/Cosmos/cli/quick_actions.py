"""
cosmos Quick Actions

One-liner commands for common tasks, perfect for power users.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, date
import json


class QuickActions:
    """
    Quick action shortcuts for common cosmos operations.

    Provides simple, single-command interfaces for:
    - Health tracking
    - Memory operations
    - Agent queries
    - Workflow triggers
    """

    def __init__(self, data_dir: str = "./data"):
        """Initialize quick actions."""
        self.data_dir = data_dir
        self._memory = None
        self._health = None

    async def _ensure_memory(self):
        """Ensure memory system is initialized."""
        if not self._memory:
            try:
                from Cosmos.memory.memory_system import MemorySystem
                self._memory = MemorySystem(data_dir=self.data_dir)
                await self._memory.initialize()
            except Exception:
                pass
        return self._memory

    async def _ensure_health(self):
        """Ensure health system is initialized."""
        if not self._health:
            try:
                from Cosmos.health.providers import HealthProviderManager
                self._health = HealthProviderManager()
            except Exception:
                pass
        return self._health

    # ========== Memory Quick Actions ==========

    async def remember(self, content: str, tags: Optional[List[str]] = None) -> str:
        """
        Quick save a memory.

        Usage: await quick.remember("Important meeting notes")
        """
        memory = await self._ensure_memory()
        if memory:
            mem_id = await memory.remember(content, metadata={"tags": tags or []})
            return f"Saved: {mem_id[:8]}"
        return "Memory system unavailable"

    async def recall(self, query: str, top_k: int = 3) -> List[str]:
        """
        Quick search memories.

        Usage: await quick.recall("meeting notes")
        """
        memory = await self._ensure_memory()
        if memory:
            results = await memory.recall(query, top_k=top_k)
            return [r.content[:200] for r in results]
        return []

    async def forget(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Usage: await quick.forget("abc123")
        """
        memory = await self._ensure_memory()
        if memory:
            try:
                await memory.delete(memory_id)
                return True
            except Exception:
                pass
        return False

    # ========== Health Quick Actions ==========

    async def steps_today(self) -> Optional[int]:
        """
        Get today's step count.

        Usage: steps = await quick.steps_today()
        """
        health = await self._ensure_health()
        if health:
            summaries = await health.get_daily_summaries(date.today())
            if summaries:
                return summaries[0].total_steps
        return None

    async def sleep_last_night(self) -> Optional[float]:
        """
        Get last night's sleep duration in hours.

        Usage: hours = await quick.sleep_last_night()
        """
        health = await self._ensure_health()
        if health:
            summaries = await health.get_daily_summaries(date.today())
            if summaries:
                return summaries[0].sleep_duration_hours
        return None

    async def log_food(self, food_name: str, calories: int, meal_type: str = "snack") -> bool:
        """
        Quick log a food item.

        Usage: await quick.log_food("Apple", 95, "snack")
        """
        try:
            from Cosmos.health.nutrition import NutritionManager

            nutrition = NutritionManager(f"{self.data_dir}/nutrition")
            nutrition.log_meal(
                meal_type=meal_type,
                foods=[{
                    "name": food_name,
                    "calories": calories,
                    "servings": 1.0,
                }],
            )
            return True
        except Exception:
            return False

    async def health_summary(self) -> Dict[str, Any]:
        """
        Get quick health summary.

        Usage: summary = await quick.health_summary()
        """
        health = await self._ensure_health()
        if health:
            summaries = await health.get_daily_summaries(date.today())
            if summaries:
                s = summaries[0]
                return {
                    "steps": s.total_steps,
                    "calories": s.total_calories_burned,
                    "sleep_hours": s.sleep_duration_hours,
                    "resting_hr": s.resting_heart_rate,
                    "recovery": s.recovery_score,
                }
        return {}

    # ========== Agent Quick Actions ==========

    async def ask(self, question: str, expert: str = "general") -> str:
        """
        Quick ask an AI agent.

        Usage: answer = await quick.ask("What is Python?")
        """
        try:
            from Cosmos.agents.swarm_orchestrator import SwarmOrchestrator

            orchestrator = SwarmOrchestrator()
            result = await orchestrator.process(query=question, task_type=expert)
            return result.get("response", "No response")
        except Exception as e:
            return f"Error: {e}"

    async def research(self, topic: str) -> str:
        """
        Quick research a topic.

        Usage: info = await quick.research("quantum computing")
        """
        return await self.ask(f"Research and explain: {topic}", expert="research")

    async def code(self, task: str) -> str:
        """
        Quick coding assistance.

        Usage: code = await quick.code("Python function to reverse a string")
        """
        return await self.ask(task, expert="code")

    # ========== Workflow Quick Actions ==========

    async def trigger_workflow(self, workflow_name: str, data: Optional[Dict] = None) -> bool:
        """
        Trigger a named workflow.

        Usage: await quick.trigger_workflow("daily_backup")
        """
        try:
            from Cosmos.automation.workflow_builder import WorkflowBuilder

            builder = WorkflowBuilder(f"{self.data_dir}/workflows")
            workflows = builder.list_workflows()

            for wf in workflows:
                if wf.get("name") == workflow_name:
                    await builder.execute(wf["id"], data or {})
                    return True
            return False
        except Exception:
            return False

    async def trigger_n8n(self, workflow_id: str, data: Optional[Dict] = None) -> Dict:
        """
        Trigger an n8n workflow.

        Usage: result = await quick.trigger_n8n("123", {"key": "value"})
        """
        try:
            from Cosmos.automation.n8n_enhanced import EnhancedN8nIntegration
            import os

            n8n = EnhancedN8nIntegration(
                api_key=os.getenv("N8N_API_KEY", ""),
                base_url=os.getenv("N8N_URL", "http://localhost:5678"),
            )

            if await n8n.connect():
                return await n8n.trigger_workflow(workflow_id, data or {})
            return {"error": "Failed to connect to n8n"}
        except Exception as e:
            return {"error": str(e)}

    # ========== Utility Quick Actions ==========

    async def backup(self) -> bool:
        """
        Quick create a backup.

        Usage: await quick.backup()
        """
        try:
            from Cosmos.core.resilience import BackupManager

            backup_mgr = BackupManager(
                data_dir=self.data_dir,
                backup_dir=f"{self.data_dir}/../backups",
            )
            await backup_mgr.create_backup()
            return True
        except Exception:
            return False

    def now(self) -> str:
        """
        Get current timestamp.

        Usage: timestamp = quick.now()
        """
        return datetime.now().isoformat()

    async def status(self) -> Dict[str, bool]:
        """
        Quick system status check.

        Usage: status = await quick.status()
        """
        return {
            "memory": (await self._ensure_memory()) is not None,
            "health": (await self._ensure_health()) is not None,
        }


# Global instance for convenience
quick = QuickActions()
