"""
cosmos Cognitive Projection (FCP) Engine.

"Good news, everyone! I've invented a device that projects my internal state directly into markdown!"

UPDATES:
- Resilience integration (Circuit Breaker)
- "Deep Focus" Debouncing (prevents I/O floods)
- "Lensing" (Summarization of Focus)
"""

import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from loguru import logger

from Cosmos.core.nexus import nexus, Signal, SignalType
from Cosmos.core.resilience import projector_breaker
from Cosmos.memory.project_tracking import ProjectTracker, Project, TaskStatus

class FCPEngine:
    """
    The holographic projector engine.
    Now with Debouncing and Stress Protection.
    """
    
    def __init__(self, project_root: str, tracker: ProjectTracker):
        self.project_root = Path(project_root)
        self.tracker = tracker
        
        # Holograms
        self.vision_file = self.project_root / "VISION.md"
        self.focus_file = self.project_root / "FOCUS.md"
        self.horizon_file = self.project_root / "HORIZON.md"
        
        # Debouncing State
        self._last_focus_update: dict[str, datetime] = {}
        self._update_queue: set[str] = set()
        self._debounce_interval = timedelta(seconds=2.0)
        self._is_projecting = False
        
        # Connect to the Nexus
        nexus.subscribe(SignalType.TASK_UPDATED, self._on_state_change)
        nexus.subscribe(SignalType.TASK_COMPLETED, self._on_state_change)
        nexus.subscribe(SignalType.DECISION_REACHED, self._on_state_change)
        
        logger.info("FCP Engine online.")

    async def _on_state_change(self, signal: Signal):
        """
        Queue a projection update.
        We don't project immediately to avoid hitting the disk 50 times/sec during rapid thought.
        """
        project_id = signal.payload.get("project_id")
        if project_id:
            self._update_queue.add(project_id)
            # Create a task to process the queue if not already running
            if not self._is_projecting:
                asyncio.create_task(self._process_projection_queue())

    async def _process_projection_queue(self):
        """Debounced processor for the projection queue."""
        self._is_projecting = True
        try:
            while self._update_queue:
                # Wait briefly to let more events collapse
                await asyncio.sleep(0.5)
                
                # Get batch
                project_ids = list(self._update_queue)
                self._update_queue.clear()
                
                for pid in project_ids:
                    # Check throttle
                    last = self._last_focus_update.get(pid)
                    if last and (datetime.now() - last) < self._debounce_interval:
                        # Re-queue if too soon
                        self._update_queue.add(pid)
                        continue
                        
                    await self.project_hologram(pid)
                    self._last_focus_update[pid] = datetime.now()
        finally:
            self._is_projecting = False

    @projector_breaker # Protects against cascading failures
    async def project_hologram(self, project_id: str):
        """
        Project the internal high-dimensional state into the markdown artifacts.
        """
        project = self.tracker.projects.get(project_id)
        if not project:
            return

        # Parallel emission
        await asyncio.gather(
            self._project_vision(project),
            self._project_focus(project),
            self._project_horizon(project)
        )
        logger.debug(f"FCP: Hologram refreshed for {project.name}")

    async def get_resonant_context(self, query: str) -> str:
        """
        Constructs the "Thought Vector" (Context String) for the LLM.
        NOTE: Future versions will perform semantic search here.
        """
        context_blocks = []
        
        # 1. Vision (The Axioms)
        if self.vision_file.exists():
            content = self.vision_file.read_text(encoding="utf-8")
            # Deep Lensing: If vision is huge, we might summarize it here
            context_blocks.append(f"<fcp_vision>\n{content}\n</fcp_vision>")
            
        # 2. Focus (The Working Memory) - The most important part
        if self.focus_file.exists():
            content = self.focus_file.read_text(encoding="utf-8")
            context_blocks.append(f"<fcp_focus>\n{content}\n</fcp_focus>")
            
        return "\n\n".join(context_blocks)

    async def _project_vision(self, project: Project):
        """Updates VISION.md - The crystallized intent."""
        content = f"""# 👁️ Vision: {project.name}
> "The architectural axioms and crystallized intent."

## Core Purpose
{project.description}
## DNA
{', '.join([f'`{t}`' for t in project.tags])}
"""
        await self._write_artifact(self.vision_file, content)

    async def _project_focus(self, project: Project):
        """Updates FOCUS.md - The dynamic working memory."""
        tasks = await self.tracker.list_tasks(project.id)
        active = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
        pending = [t for t in tasks if t.status == TaskStatus.PENDING]
        blocked = [t for t in tasks if t.status == TaskStatus.BLOCKED]
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        content = f"""# 🎯 Focus: {project.name}
**Updated**: {timestamp} | **Swarm Status**: ACTIVE

## ⚡ Active Synapses (In Progress)
"""
        if active:
            for t in active:
                content += f"- **{t.title}** (Priority {t.priority})\n  - {t.description}\n"
        else:
            content += "- *Network Idle - Awaiting Signal*\n"
            
        content += "\n## ⏳ Immediate Queue\n"
        for t in pending[:5]:
            content += f"- {t.title}\n"
            
        content += "\n## 🚧 Neural Blockers\n"
        if blocked:
            for t in blocked:
                content += f"- 🛑 {t.title} (Blocked by dependencies)\n"
        else:
            content += "- *Flow Optimal*\n"
            
        await self._write_artifact(self.focus_file, content)

    async def _project_horizon(self, project: Project):
        """Updates HORIZON.md"""
        content = f"# 🌅 Horizon: {project.name}\n> Future milestones..."
        await self._write_artifact(self.horizon_file, content)

    async def _write_artifact(self, path: Path, content: str):
        # Async file write simulation
        path.write_text(content, encoding="utf-8")
