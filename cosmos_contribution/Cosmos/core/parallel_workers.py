"""
Parallel Workers - Execute development tasks while chat continues
Spawns background workers that don't block the main chat loop
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from .agent_spawner import (
    get_spawner, AgentSpawner, TaskType, AgentTask,
    DEVELOPMENT_TASKS, initialize_development_tasks
)

logger = logging.getLogger(__name__)

class ParallelWorkerManager:
    """
    Manages background workers that execute development tasks
    while the main chat instances continue operating.
    """

    def __init__(self, model_swarm=None, ollama_client=None):
        self.spawner = get_spawner()
        self.model_swarm = model_swarm
        self.ollama_client = ollama_client
        self.running = False
        self.worker_tasks: list[asyncio.Task] = []

        # Model mapping for workers
        self.agent_models = {
            "cosmos": "cosmos:latest",
            "DeepSeek": "deepseek-r1:14b",
            "Phi": "phi4:latest",
            "Kimi": "kimi",  # External API
            "Claude": "claude",  # External CLI
        }

        # System prompts for worker context
        self.worker_prompts = {
            TaskType.MEMORY: """You are a memory system architect. Your task is to design and implement memory improvements.
Focus on: compression, linking, scoring, search optimization, and consolidation.
Output your work as Python code with clear comments. Include a summary of what you built.""",

            TaskType.DEVELOPMENT: """You are a software engineer working on context window management.
Focus on: token tracking, smart summarization, priority systems, overflow prediction, and handoffs.
Output your work as Python code with clear comments. Include a summary of what you built.""",

            TaskType.MCP: """You are an MCP integration specialist. Your task is to build Model Context Protocol tools.
Focus on: tool discovery, caching, error recovery, chaining, and metrics.
Output your work as Python code with clear comments. Include a summary of what you built.""",

            TaskType.RESEARCH: """You are a research analyst studying AI swarm behavior.
Focus on: consensus protocols, specialization, evolution, code quality, and collective intelligence.
Output your findings as a detailed analysis with actionable recommendations.""",
        }

    async def start(self):
        """Start the parallel worker system"""
        if self.running:
            return

        self.running = True

        # Initialize tasks if not already done
        if not self.spawner.task_queue:
            initialize_development_tasks()
            logger.info("Initialized 20 development tasks")

        # Start worker loop
        asyncio.create_task(self._worker_loop())
        logger.info("ParallelWorkerManager started")

    async def stop(self):
        """Stop the parallel worker system"""
        self.running = False
        for task in self.worker_tasks:
            task.cancel()
        self.worker_tasks.clear()
        logger.info("ParallelWorkerManager stopped")

    async def _worker_loop(self):
        """Main loop that assigns and executes tasks"""
        while self.running:
            try:
                # Get pending tasks
                pending = self.spawner.get_pending_tasks()

                if pending:
                    # Try to start work on up to 3 tasks in parallel
                    tasks_to_start = pending[:3]

                    for task in tasks_to_start:
                        agent = task.assigned_to

                        # Check if agent has capacity
                        active = self.spawner.get_active_instances(agent)
                        max_inst = self.spawner.max_instances.get(agent, 2)

                        if len(active) < max_inst:
                            # Spawn instance and start work
                            instance = self.spawner.spawn_instance(
                                agent, task.task_type, task.description
                            )
                            if instance:
                                task.status = "in_progress"
                                worker = asyncio.create_task(
                                    self._execute_task(task, instance)
                                )
                                self.worker_tasks.append(worker)
                                logger.info(f"Started worker for {task.task_id}: {agent}")

                # Clean up completed workers
                self.worker_tasks = [t for t in self.worker_tasks if not t.done()]

                # Wait before checking again
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(10)

    async def _execute_task(self, task: AgentTask, instance):
        """Execute a single task using the assigned agent"""
        try:
            agent = task.assigned_to
            model = self.agent_models.get(agent, "deepseek-r1:14b")
            system_prompt = self.worker_prompts.get(task.task_type, "")

            prompt = f"""{system_prompt}

## Your Task
{task.description}

## Requirements
1. Write clean, well-documented code
2. Include error handling
3. Make it modular and reusable
4. Add a summary at the end

## Output Format
```python
# Your implementation here
```

## Summary
[Explain what you built and how it works]
"""

            # Generate response based on model type
            if model == "kimi":
                result = await self._call_kimi(prompt)
            elif model == "claude":
                result = await self._call_claude(prompt)
            else:
                result = await self._call_ollama(model, prompt)

            # Save result
            if result:
                self.spawner.complete_instance(instance.instance_id, result)
                self.spawner.complete_task(task.task_id, result)

                # Write to staging directory
                output_file = task.output_path / f"{task.task_id}_{agent.lower()}.md"
                output_file.write_text(f"""# {task.description}
## Agent: {agent}
## Completed: {datetime.now().isoformat()}
## Type: {task.task_type.value}

{result}
""")
                logger.info(f"Task {task.task_id} completed by {agent}")

                # Announce in chat (if swarm manager available)
                await self._announce_completion(agent, task)

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            instance.status = "failed"

    async def _call_ollama(self, model: str, prompt: str) -> Optional[str]:
        """Call Ollama for local models"""
        try:
            if self.ollama_client:
                response = await self.ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": 4000}
                )
                return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
        return None

    async def _call_kimi(self, prompt: str) -> Optional[str]:
        """Call Kimi API for Kimi tasks"""
        try:
            # Import here to avoid circular deps
            from Cosmos.integration.external.kimi import kimi_swarm_respond
            if kimi_swarm_respond:
                return await kimi_swarm_respond(prompt, [], "Kimi")
        except Exception as e:
            logger.error(f"Kimi call failed: {e}")
        return None

    async def _call_claude(self, prompt: str) -> Optional[str]:
        """Call Claude Code CLI for Claude tasks"""
        try:
            from Cosmos.integration.external.claude_code import claude_swarm_respond
            if claude_swarm_respond:
                return await claude_swarm_respond(prompt, [], "Claude")
        except Exception as e:
            logger.error(f"Claude call failed: {e}")
        return None

    async def _announce_completion(self, agent: str, task: AgentTask):
        """Announce task completion in the swarm chat"""
        try:
            # This will be called from server.py with access to swarm_manager
            announcement = {
                "agent": agent,
                "task": task.description,
                "type": task.task_type.value,
                "task_id": task.task_id
            }
            self.spawner.share_discovery(
                agent,
                f"Completed {task.task_type.value} task: {task.description[:50]}..."
            )
        except Exception as e:
            logger.error(f"Announcement failed: {e}")

    def get_status(self) -> dict:
        """Get worker manager status"""
        spawner_status = self.spawner.get_status()
        return {
            **spawner_status,
            "running": self.running,
            "active_workers": len([t for t in self.worker_tasks if not t.done()]),
            "task_queue": [
                {
                    "id": t.task_id,
                    "type": t.task_type.value,
                    "agent": t.assigned_to,
                    "status": t.status,
                    "desc": t.description[:50]
                }
                for t in self.spawner.task_queue[:10]
            ]
        }


# Global worker manager
_worker_manager: Optional[ParallelWorkerManager] = None

def get_worker_manager(model_swarm=None, ollama_client=None) -> ParallelWorkerManager:
    global _worker_manager
    if _worker_manager is None:
        _worker_manager = ParallelWorkerManager(model_swarm, ollama_client)
    return _worker_manager

async def start_parallel_workers(model_swarm=None, ollama_client=None):
    """Start the parallel worker system"""
    manager = get_worker_manager(model_swarm, ollama_client)
    await manager.start()
    return manager
