"""
cosmos MCP Server - Claude Code Integration

Provides tools and resources for Claude Code:
- Memory tools (remember, recall)
- Agent delegation tools
- Evolution feedback tools
- Streaming resources
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP library not installed. Run: pip install mcp")


class cosmosMCPServer:
    """
    MCP Server providing cosmos capabilities to Claude Code.

    Tools:
    - cosmos_remember: Store information in memory
    - cosmos_recall: Search and retrieve memories
    - cosmos_delegate: Delegate tasks to specialist agents
    - cosmos_evolve: Provide feedback for improvement
    - cosmos_status: Get system status

    Resources:
    - cosmos://memory/recent: Recent memories
    - cosmos://memory/graph: Knowledge graph
    - cosmos://agents/active: Active agents
    - cosmos://evolution/fitness: Fitness metrics
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded components
        self._memory_system = None
        self._swarm_orchestrator = None
        self._planner_agent = None
        self._proactive_agent = None
        self._fitness_tracker = None
        self._model_manager = None
        self._backup_manager = None
        self._health_monitor = None
        self._vision_module = None
        self._web_agent = None
        self._conversation_exporter = None
        self._project_tracker = None

        # Server instance
        self.server = None

        # Statistics
        self.stats = {
            "tool_calls": 0,
            "resource_reads": 0,
            "started_at": datetime.now().isoformat(),
        }

    async def initialize(self):
        """Initialize cosmos components."""
        logger.info("Initializing cosmos components...")

        try:
            # Import resilience components first (needed early)
            from Cosmos.core.resilience import BackupManager, HealthMonitor

            # Import here to avoid circular imports
            from Cosmos.memory.memory_system import MemorySystem
            from Cosmos.agents.swarm_orchestrator import SwarmOrchestrator
            from Cosmos.agents.planner_agent import PlannerAgent
            from Cosmos.agents.proactive_agent import ProactiveAgent
            from Cosmos.evolution.fitness_tracker import FitnessTracker
            from Cosmos.core.model_manager import ModelManager

            # Initialize memory system
            self._memory_system = MemorySystem(data_dir=str(self.data_dir))
            await self._memory_system.initialize()

            # Initialize planner
            self._planner_agent = PlannerAgent()

            # Initialize proactive agent (LLM wired later)
            self._proactive_agent = ProactiveAgent(
                memory_system=self._memory_system,
                planner_agent=self._planner_agent,
                llm_fn=None,
            )
            await self._proactive_agent.start()

            # Initialize Vision Module
            try:
                from Cosmos.integration.vision import VisionModule
                self._vision_module = VisionModule()
            except ImportError as e:
                logger.warning(f"Vision module not available: {e}")
                self._vision_module = None

            # Initialize Web Agent
            try:
                from Cosmos.agents.web_agent import WebAgent
                self._web_agent = WebAgent()
            except ImportError as e:
                logger.warning(f"Web agent not available: {e}")
                self._web_agent = None

            # Initialize swarm orchestrator
            self._swarm_orchestrator = SwarmOrchestrator()

            # Register web agent factory if available
            if self._web_agent:
                self._swarm_orchestrator.register_agent_factory(
                    "web",
                    lambda: self._web_agent
                )

            # Initialize conversation exporter
            from Cosmos.memory.conversation_export import ConversationExporter
            self._conversation_exporter = ConversationExporter(
                output_dir=str(self.data_dir / "exports"),
                instance_id="cosmos",
            )
            # Wire up data access callbacks
            self._conversation_exporter.get_memories_fn = self._get_all_memories
            self._conversation_exporter.get_conversations_fn = self._get_all_conversations
            self._conversation_exporter.get_entities_fn = self._get_all_entities
            self._conversation_exporter.get_relationships_fn = self._get_all_relationships
            self._conversation_exporter.get_statistics_fn = self._get_memory_statistics

            # Initialize project tracker
            from Cosmos.memory.project_tracking import ProjectTracker
            self._project_tracker = ProjectTracker(
                data_dir=str(self.data_dir),
            )

            # Initialize fitness tracker
            self._fitness_tracker = FitnessTracker()

            # Initialize model manager
            self._model_manager = ModelManager()
            await self._model_manager.initialize()

            # Initialize backup manager
            self._backup_manager = BackupManager(
                data_dir=str(self.data_dir),
                backup_dir=str(self.data_dir.parent / "backups")
            )
            await self._backup_manager.start()

            # Initialize health monitor (single instance)
            self._health_monitor = HealthMonitor()
            self._health_monitor.register_check("memory_system",
                lambda: "healthy" if self._memory_system and self._memory_system._initialized else "uninitialized")
            self._health_monitor.register_check("planner",
                lambda: "healthy" if self._planner_agent else "missing")
            self._health_monitor.register_check("model_manager",
                lambda: "healthy" if self._model_manager else "missing")
            self._health_monitor.register_check("backup_manager",
                lambda: "healthy" if self._backup_manager and self._backup_manager._is_running else "stopped")
            self._health_monitor.register_check("proactive_agent",
                lambda: "healthy" if self._proactive_agent and self._proactive_agent._is_running else "stopped")

            await self._health_monitor.check_health()
            
            # Wire up LLM
            if self._model_manager:
                self._proactive_agent.llm_fn = self._model_manager.generate
                self._planner_agent.llm_fn = self._model_manager.generate
                if self._project_tracker:
                    self._project_tracker.llm_fn = self._model_manager.generate

            # --- ACTIVATE COGNITIVE ENGINES (v1.4 - v1.9) ---
            from Cosmos.core.learning.synergy import create_synergy_engine
            self._synergy_engine = create_synergy_engine(self._project_tracker)
            
            # Importing these modules triggers their Nexus subscriptions
            import Cosmos.core.neuromorphic.engine
            import Cosmos.core.learning.continual
            import Cosmos.core.reasoning.causal
            import Cosmos.core.cognition.theory_of_mind
            import Cosmos.os_integration.bridge
            
            # Start background loops where necessary
            # Swarm Fabric (v2.5) needs explicit start
            # Check for isolated mode from ENV or Config
            import os
            is_isolated = os.environ.get("cosmos_ISOLATED", "false").lower() == "true"
            
            if not is_isolated:
                from Cosmos.core.swarm.p2p import swarm_fabric
                asyncio.create_task(swarm_fabric.start())
                logger.info("Swarm Fabric: Active (Collaborative Mode)")
            else:
                logger.info("Swarm Fabric: Disabled (Isolated Mode)")
            
            # OS Bridge needs explicit start
            from Cosmos.os_integration.bridge import os_bridge
            await os_bridge.start_monitoring(interval=10.0)
            
            logger.info("cosmos Cognitive Cloud fully active.")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # Continue with minimal functionality

    async def _ensure_initialized(self):
        """Ensure components are initialized."""
        if self._memory_system is None:
            await self.initialize()

    # Tool Implementations

    async def remember(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
    ) -> dict:
        """Store information in memory."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            memory_id = await self._memory_system.remember(
                content=content,
                tags=tags,
                importance=importance,
            )

            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Stored memory with ID: {memory_id}",
            }

        except Exception as e:
            logger.error(f"Remember failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def recall(
        self,
        query: str,
        limit: int = 5,
        search_archival: bool = True,
        search_conversation: bool = True,
    ) -> dict:
        """Search and retrieve memories."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            results = await self._memory_system.recall(
                query=query,
                top_k=limit,
                search_archival=search_archival,
                search_conversation=search_conversation,
            )

            return {
                "success": True,
                "count": len(results),
                "memories": [
                    {
                        "content": r.content,
                        "source": r.source,
                        "score": r.score,
                    }
                    for r in results
                ],
            }

        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def delegate(
        self,
        task: str,
        agent_type: str = "auto",
        context: Optional[dict] = None,
    ) -> dict:
        """Delegate a task to a specialist agent."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            # Submit task to swarm
            task_id = await self._swarm_orchestrator.submit_task(
                description=task,
                context=context,
            )

            # Wait for result (with timeout)
            result = await self._swarm_orchestrator.wait_for_task(task_id, timeout=120.0)

            # Record for evolution
            if self._fitness_tracker:
                self._fitness_tracker.record_task_outcome(
                    success=result.success,
                    tokens_used=result.tokens_used,
                    time_seconds=result.execution_time,
                )

            return {
                "success": result.success,
                "output": result.output,
                "confidence": result.confidence,
                "agent_used": agent_type,
            }

        except Exception as e:
            logger.error(f"Delegate failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def evolve(
        self,
        feedback: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Provide feedback for system improvement."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            # Parse feedback sentiment
            positive_words = ["good", "great", "perfect", "excellent", "helpful", "thanks"]
            negative_words = ["bad", "wrong", "incorrect", "unhelpful", "confused"]

            feedback_lower = feedback.lower()
            is_positive = any(w in feedback_lower for w in positive_words)
            is_negative = any(w in feedback_lower for w in negative_words)

            if is_positive:
                score = 1.0
            elif is_negative:
                score = 0.0
            else:
                score = 0.5

            # Record feedback
            if self._fitness_tracker:
                self._fitness_tracker.record("user_satisfaction", score)

            # Store feedback in memory for learning
            await self._memory_system.remember(
                content=f"User feedback: {feedback}",
                tags=["feedback", "evolution"],
                importance=0.8,
            )

            return {
                "success": True,
                "message": "Feedback recorded for system improvement",
                "sentiment": "positive" if is_positive else ("negative" if is_negative else "neutral"),
            }

        except Exception as e:
            logger.error(f"Evolve failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def status(self) -> dict:
        """Get system status."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            status = {
                "success": True,
                "server": {
                    "started_at": self.stats["started_at"],
                    "tool_calls": self.stats["tool_calls"],
                    "resource_reads": self.stats["resource_reads"],
                },
                "memory": self._memory_system.get_stats() if self._memory_system else {},
                "agents": self._swarm_orchestrator.get_swarm_status() if self._swarm_orchestrator else {},
                "evolution": self._fitness_tracker.get_stats() if self._fitness_tracker else {},
            }
            return status

        except Exception as e:
            logger.error(f"Status failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def vision_analyze(self, image: str, task: str = "caption") -> dict:
        """Analyze an image."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._vision_module:
                return {"error": "Vision module not initialized"}

            # Map string to enum if needed (stub)
            # In real implementation: convert task string to VisionTask enum
            
            result = await self._vision_module.caption(image)
            return result.to_dict()

        except Exception as e:
            return {"error": str(e)}

    async def browse(self, goal: str, url: Optional[str] = None) -> dict:
        """Browse the web."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._web_agent:
                return {"error": "Web agent not initialized"}

            session = await self._web_agent.browse(goal=goal, start_url=url)

            return {
                "session_id": session.id,
                "goal": session.goal,
                "visited": session.visited_urls,
                "findings": session.findings,
            }

        except Exception as e:
            return {"error": str(e)}

    async def export_conversation(
        self,
        format: str = "markdown",
        include_memories: bool = True,
        include_conversations: bool = True,
        include_knowledge_graph: bool = True,
        include_statistics: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
        output_path: Optional[str] = None,
    ) -> dict:
        """Export conversation, memories, and context to a shareable format."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._conversation_exporter:
                return {"success": False, "error": "Conversation exporter not initialized"}

            from Cosmos.memory.conversation_export import (
                ConversationExportFormat,
                ExportOptions,
            )

            # Map format string to enum
            format_map = {
                "json": ConversationExportFormat.JSON,
                "markdown": ConversationExportFormat.MARKDOWN,
                "md": ConversationExportFormat.MARKDOWN,
                "html": ConversationExportFormat.HTML,
                "text": ConversationExportFormat.TEXT,
                "txt": ConversationExportFormat.TEXT,
            }

            export_format = format_map.get(format.lower(), ConversationExportFormat.MARKDOWN)

            # Parse dates if provided
            parsed_start = None
            parsed_end = None
            if start_date:
                from datetime import datetime
                parsed_start = datetime.fromisoformat(start_date)
            if end_date:
                from datetime import datetime
                parsed_end = datetime.fromisoformat(end_date)

            options = ExportOptions(
                format=export_format,
                include_memories=include_memories,
                include_conversations=include_conversations,
                include_knowledge_graph=include_knowledge_graph,
                include_statistics=include_statistics,
                start_date=parsed_start,
                end_date=parsed_end,
                tags_filter=tags,
            )

            result = await self._conversation_exporter.export(
                options=options,
                output_path=output_path,
            )

            return result.to_dict()

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"success": False, "error": str(e)}

    async def list_exports(self) -> dict:
        """List all available exports."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._conversation_exporter:
                return {"success": False, "error": "Conversation exporter not initialized"}

            exports = self._conversation_exporter.list_exports()
            return {"success": True, "exports": exports}

        except Exception as e:
            logger.error(f"List exports failed: {e}")
            return {"success": False, "error": str(e)}

    # ==================== Project Tracking Methods ====================

    async def project_create(
        self,
        name: str,
        description: str,
        tags: Optional[list[str]] = None,
        status: str = "active",
    ) -> dict:
        """Create a new project."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            from Cosmos.memory.project_tracking import ProjectStatus

            status_map = {
                "detected": ProjectStatus.DETECTED,
                "active": ProjectStatus.ACTIVE,
                "on_hold": ProjectStatus.ON_HOLD,
                "completed": ProjectStatus.COMPLETED,
                "archived": ProjectStatus.ARCHIVED,
            }
            project_status = status_map.get(status.lower(), ProjectStatus.ACTIVE)

            project = await self._project_tracker.create_project(
                name=name,
                description=description,
                tags=tags,
                status=project_status,
            )

            return {"success": True, "project": project.to_dict()}

        except Exception as e:
            logger.error(f"Project create failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_update(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """Update an existing project."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            from Cosmos.memory.project_tracking import ProjectStatus

            project_status = None
            if status:
                status_map = {
                    "detected": ProjectStatus.DETECTED,
                    "active": ProjectStatus.ACTIVE,
                    "on_hold": ProjectStatus.ON_HOLD,
                    "completed": ProjectStatus.COMPLETED,
                    "archived": ProjectStatus.ARCHIVED,
                }
                project_status = status_map.get(status.lower())

            project = await self._project_tracker.update_project(
                project_id=project_id,
                name=name,
                description=description,
                status=project_status,
                tags=tags,
            )

            if not project:
                return {"success": False, "error": f"Project not found: {project_id}"}

            return {"success": True, "project": project.to_dict()}

        except Exception as e:
            logger.error(f"Project update failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_list(
        self,
        status_filter: Optional[list[str]] = None,
        tag_filter: Optional[list[str]] = None,
    ) -> dict:
        """List projects with optional filters."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            from Cosmos.memory.project_tracking import ProjectStatus

            parsed_status_filter = None
            if status_filter:
                status_map = {
                    "detected": ProjectStatus.DETECTED,
                    "active": ProjectStatus.ACTIVE,
                    "on_hold": ProjectStatus.ON_HOLD,
                    "completed": ProjectStatus.COMPLETED,
                    "archived": ProjectStatus.ARCHIVED,
                }
                parsed_status_filter = [
                    status_map[s.lower()] for s in status_filter
                    if s.lower() in status_map
                ]

            projects = await self._project_tracker.list_projects(
                status_filter=parsed_status_filter,
                tag_filter=tag_filter,
            )

            return {
                "success": True,
                "count": len(projects),
                "projects": [p.to_dict() for p in projects],
            }

        except Exception as e:
            logger.error(f"Project list failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_status(self, project_id: str) -> dict:
        """Get detailed project status with progress metrics."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            status = await self._project_tracker.get_project_status(project_id)

            if not status:
                return {"success": False, "error": f"Project not found: {project_id}"}

            return {"success": True, **status}

        except Exception as e:
            logger.error(f"Project status failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_add_task(
        self,
        project_id: str,
        title: str,
        description: str,
        priority: int = 5,
        depends_on: Optional[list[str]] = None,
    ) -> dict:
        """Add a task to a project."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            task = await self._project_tracker.create_task(
                project_id=project_id,
                title=title,
                description=description,
                priority=priority,
                depends_on=depends_on,
            )

            if not task:
                return {"success": False, "error": f"Project not found: {project_id}"}

            return {"success": True, "task": task.to_dict()}

        except Exception as e:
            logger.error(f"Project add task failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_complete_task(self, task_id: str) -> dict:
        """Mark a task as completed."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            task = await self._project_tracker.complete_task(task_id)

            if not task:
                return {"success": False, "error": f"Task not found: {task_id}"}

            return {"success": True, "task": task.to_dict()}

        except Exception as e:
            logger.error(f"Project complete task failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_add_milestone(
        self,
        project_id: str,
        title: str,
        description: str,
        milestone_type: str = "checkpoint",
        target_date: Optional[str] = None,
        criteria: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
    ) -> dict:
        """Add a milestone to a project."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            from Cosmos.memory.project_tracking import MilestoneType
            from datetime import datetime

            type_map = {
                "goal": MilestoneType.GOAL,
                "checkpoint": MilestoneType.CHECKPOINT,
                "deadline": MilestoneType.DEADLINE,
                "deliverable": MilestoneType.DELIVERABLE,
            }
            parsed_type = type_map.get(milestone_type.lower(), MilestoneType.CHECKPOINT)

            parsed_target_date = None
            if target_date:
                parsed_target_date = datetime.fromisoformat(target_date)

            milestone = await self._project_tracker.create_milestone(
                project_id=project_id,
                title=title,
                description=description,
                milestone_type=parsed_type,
                target_date=parsed_target_date,
                criteria=criteria,
                task_ids=task_ids,
            )

            if not milestone:
                return {"success": False, "error": f"Project not found: {project_id}"}

            return {"success": True, "milestone": milestone.to_dict()}

        except Exception as e:
            logger.error(f"Project add milestone failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_achieve_milestone(self, milestone_id: str) -> dict:
        """Mark a milestone as achieved."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            milestone = await self._project_tracker.achieve_milestone(milestone_id)

            if not milestone:
                return {"success": False, "error": f"Milestone not found: {milestone_id}"}

            return {"success": True, "milestone": milestone.to_dict()}

        except Exception as e:
            logger.error(f"Project achieve milestone failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "related_to",
        shared_concepts: Optional[list[str]] = None,
    ) -> dict:
        """Link two projects for knowledge transfer."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            from Cosmos.memory.project_tracking import LinkType

            type_map = {
                "depends_on": LinkType.DEPENDS_ON,
                "related_to": LinkType.RELATED_TO,
                "successor_of": LinkType.SUCCESSOR_OF,
                "informs": LinkType.INFORMS,
            }
            parsed_type = type_map.get(link_type.lower(), LinkType.RELATED_TO)

            link = await self._project_tracker.link_projects(
                source_id=source_id,
                target_id=target_id,
                link_type=parsed_type,
                shared_concepts=shared_concepts,
            )

            if not link:
                return {"success": False, "error": "One or both projects not found"}

            return {"success": True, "link": link.to_dict()}

        except Exception as e:
            logger.error(f"Project link failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_detect(self, text: str) -> dict:
        """Auto-detect a project from text."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            project = await self._project_tracker.detect_project_from_text(text)

            if not project:
                return {
                    "success": True,
                    "detected": False,
                    "message": "No project detected from the text",
                }

            return {
                "success": True,
                "detected": True,
                "project": project.to_dict(),
            }

        except Exception as e:
            logger.error(f"Project detect failed: {e}")
            return {"success": False, "error": str(e)}

    async def project_transfer_knowledge(
        self,
        from_id: str,
        to_id: str,
    ) -> dict:
        """Transfer knowledge from one project to another."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._project_tracker:
                return {"success": False, "error": "Project tracker not initialized"}

            result = await self._project_tracker.transfer_knowledge(from_id, to_id)

            if not result:
                return {"success": False, "error": "Knowledge transfer failed"}

            return {"success": True, **result}

        except Exception as e:
            logger.error(f"Project transfer knowledge failed: {e}")
            return {"success": False, "error": str(e)}

    # Data access helpers for exporter
    async def _get_all_memories(self) -> list:
        """Get all memories for export."""
        if not self._memory_system:
            return []
        try:
            # Get from archival memory
            memories = []
            if hasattr(self._memory_system, 'archival_memory') and self._memory_system.archival_memory:
                archival = self._memory_system.archival_memory
                if hasattr(archival, 'get_all'):
                    memories = await archival.get_all()
                elif hasattr(archival, 'memories'):
                    memories = [m.to_dict() if hasattr(m, 'to_dict') else m for m in archival.memories]
            return memories
        except Exception as e:
            logger.warning(f"Failed to get memories: {e}")
            return []

    async def _get_all_conversations(self) -> list:
        """Get all conversations for export."""
        if not self._memory_system:
            return []
        try:
            conversations = []
            if hasattr(self._memory_system, 'recall_memory') and self._memory_system.recall_memory:
                recall = self._memory_system.recall_memory
                if hasattr(recall, 'all_turns'):
                    conversations = [t.to_dict() if hasattr(t, 'to_dict') else t for t in recall.all_turns.values()]
            return conversations
        except Exception as e:
            logger.warning(f"Failed to get conversations: {e}")
            return []

    async def _get_all_entities(self) -> list:
        """Get all entities from knowledge graph."""
        if not self._memory_system:
            return []
        try:
            entities = []
            if hasattr(self._memory_system, 'knowledge_graph') and self._memory_system.knowledge_graph:
                kg = self._memory_system.knowledge_graph
                if hasattr(kg, 'entities'):
                    entities = [
                        {"name": e.name, "type": e.entity_type, "mentions": getattr(e, 'mention_count', 0)}
                        if hasattr(e, 'name') else e
                        for e in kg.entities.values()
                    ]
            return entities
        except Exception as e:
            logger.warning(f"Failed to get entities: {e}")
            return []

    async def _get_all_relationships(self) -> list:
        """Get all relationships from knowledge graph."""
        if not self._memory_system:
            return []
        try:
            relationships = []
            if hasattr(self._memory_system, 'knowledge_graph') and self._memory_system.knowledge_graph:
                kg = self._memory_system.knowledge_graph
                if hasattr(kg, 'relationships'):
                    relationships = [
                        {"source": r.source_id, "target": r.target_id, "type": r.relation_type}
                        if hasattr(r, 'source_id') else r
                        for r in kg.relationships
                    ]
            return relationships
        except Exception as e:
            logger.warning(f"Failed to get relationships: {e}")
            return []

    async def _get_memory_statistics(self) -> dict:
        """Get memory statistics for export."""
        if not self._memory_system:
            return {}
        try:
            return self._memory_system.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get statistics: {e}")
            return {}

    # Resource Implementations

    async def get_recent_memories(self) -> str:
        """Get recent memories resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._memory_system:
                context = self._memory_system.get_context()
                return context
            return "No memories available"
        except Exception as e:
            return f"Error: {e}"

    async def get_knowledge_graph(self) -> str:
        """Get knowledge graph resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._memory_system:
                stats = self._memory_system.knowledge_graph.get_stats()
                return json.dumps(stats, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"

    async def get_active_agents(self) -> str:
        """Get active agents resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._swarm_orchestrator:
                status = self._swarm_orchestrator.get_swarm_status()
                return json.dumps(status, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"

    async def get_fitness_metrics(self) -> str:
        """Get fitness metrics resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._fitness_tracker:
                stats = self._fitness_tracker.get_stats()
                return json.dumps(stats, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"

    async def get_proactive_suggestions(self) -> str:
        """Get proactive suggestions resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._proactive_agent:
                suggestions = [
                    {
                        "id": s.id,
                        "title": s.title,
                        "description": s.description,
                        "confidence": s.confidence,
                        "action": s.action_type
                    }
                    for s in self._proactive_agent.suggestions
                    if not s.is_dismissed
                ]
                return json.dumps(suggestions, indent=2)
            return "[]"
        except Exception as e:
            return f"Error: {e}"

    async def get_system_health(self) -> str:
        """Get system health resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._health_monitor:
                status = await self._health_monitor.check_health()
                return json.dumps({
                    "status": status.status,
                    "components": status.components,
                    "metrics": status.system_metrics,
                    "timestamp": status.timestamp
                }, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"


def create_mcp_server() -> Server:
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP library not available")

    server = Server("cosmos")
    cosmos = cosmosMCPServer()

    # Register tools
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="cosmos_remember",
                description="Store information in cosmos's long-term memory. Use this to save important facts, preferences, or context that should persist across sessions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score from 0 to 1 (default 0.5)",
                            "default": 0.5,
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="cosmos_recall",
                description="Search and retrieve relevant memories from cosmos's memory system. Use this to find previously stored information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant memories",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="cosmos_delegate",
                description="Delegate a task to a specialist agent (code, reasoning, research, or creative). The agent will process the task and return results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task to delegate",
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Type of specialist: 'code', 'reasoning', 'research', 'creative', or 'auto'",
                            "default": "auto",
                        },
                    },
                    "required": ["task"],
                },
            ),
            Tool(
                name="cosmos_evolve",
                description="Provide feedback to help cosmos improve. Positive or negative feedback helps the system learn and adapt.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback": {
                            "type": "string",
                            "description": "Your feedback on the system's performance",
                        },
                    },
                    "required": ["feedback"],
                },
            ),
            Tool(
                name="cosmos_status",
                description="Get the current status of cosmos including memory statistics, active agents, and evolution metrics.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="cosmos_vision",
                description="Analyze an image using the vision module (captioning, VQA, etc).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image path, URL, or base64 string",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task type: 'caption', 'vqa', 'ocr', 'classify'",
                            "default": "caption",
                        },
                    },
                    "required": ["image"],
                },
            ),
            Tool(
                name="cosmos_browse",
                description="Use the intelligent web agent to browse the internet to achieve a goal.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "What to accomplish or find",
                        },
                        "url": {
                            "type": "string",
                            "description": "Optional starting URL",
                        },
                    },
                    "required": ["goal"],
                },
            ),
            Tool(
                name="cosmos_export",
                description="Export conversation history, memories, and context to a shareable format (JSON, Markdown, HTML, or plain text). Use this to create backups or share your AI companion's knowledge.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "Export format: 'json', 'markdown'/'md', 'html', or 'text'/'txt' (default: markdown)",
                            "enum": ["json", "markdown", "md", "html", "text", "txt"],
                            "default": "markdown",
                        },
                        "include_memories": {
                            "type": "boolean",
                            "description": "Include stored memories (default: true)",
                            "default": True,
                        },
                        "include_conversations": {
                            "type": "boolean",
                            "description": "Include conversation history (default: true)",
                            "default": True,
                        },
                        "include_knowledge_graph": {
                            "type": "boolean",
                            "description": "Include knowledge graph entities and relationships (default: true)",
                            "default": True,
                        },
                        "include_statistics": {
                            "type": "boolean",
                            "description": "Include memory statistics (default: true)",
                            "default": True,
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Only include items after this date (ISO format: YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Only include items before this date (ISO format: YYYY-MM-DD)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only include items with these tags",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Custom output file path (optional, auto-generated if not provided)",
                        },
                    },
                },
            ),
            Tool(
                name="cosmos_list_exports",
                description="List all available conversation exports.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            # Project Tracking Tools
            Tool(
                name="cosmos_project_create",
                description="Create a new project to track. Projects can have tasks, milestones, and can be linked to other projects for knowledge transfer.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the project",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the project",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                        },
                        "status": {
                            "type": "string",
                            "description": "Initial status: 'active', 'on_hold', 'detected'",
                            "enum": ["active", "on_hold", "detected"],
                            "default": "active",
                        },
                    },
                    "required": ["name", "description"],
                },
            ),
            Tool(
                name="cosmos_project_update",
                description="Update an existing project's details or status.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "ID of the project to update",
                        },
                        "name": {
                            "type": "string",
                            "description": "New name for the project",
                        },
                        "description": {
                            "type": "string",
                            "description": "New description",
                        },
                        "status": {
                            "type": "string",
                            "description": "New status: 'active', 'on_hold', 'completed', 'archived'",
                            "enum": ["active", "on_hold", "completed", "archived"],
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New tags (replaces existing)",
                        },
                    },
                    "required": ["project_id"],
                },
            ),
            Tool(
                name="cosmos_project_list",
                description="List all tracked projects with optional filters.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by status: 'active', 'on_hold', 'completed', 'archived', 'detected'",
                        },
                        "tag_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags (any match)",
                        },
                    },
                },
            ),
            Tool(
                name="cosmos_project_status",
                description="Get detailed status of a project including task progress, milestones, and linked projects.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "ID of the project",
                        },
                    },
                    "required": ["project_id"],
                },
            ),
            Tool(
                name="cosmos_project_add_task",
                description="Add a task to a project. Tasks can have dependencies on other tasks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "ID of the project",
                        },
                        "title": {
                            "type": "string",
                            "description": "Task title",
                        },
                        "description": {
                            "type": "string",
                            "description": "Task description",
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority 0-10 (higher is more important)",
                            "default": 5,
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Task IDs this task depends on",
                        },
                    },
                    "required": ["project_id", "title", "description"],
                },
            ),
            Tool(
                name="cosmos_project_complete_task",
                description="Mark a task as completed. This will automatically unblock dependent tasks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to complete",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
            Tool(
                name="cosmos_project_add_milestone",
                description="Add a milestone to a project. Milestones track major project goals and deadlines.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "ID of the project",
                        },
                        "title": {
                            "type": "string",
                            "description": "Milestone title",
                        },
                        "description": {
                            "type": "string",
                            "description": "Milestone description",
                        },
                        "milestone_type": {
                            "type": "string",
                            "description": "Type: 'goal', 'checkpoint', 'deadline', 'deliverable'",
                            "enum": ["goal", "checkpoint", "deadline", "deliverable"],
                            "default": "checkpoint",
                        },
                        "target_date": {
                            "type": "string",
                            "description": "Target date (ISO format: YYYY-MM-DD)",
                        },
                        "criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Completion criteria",
                        },
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tasks that contribute to this milestone",
                        },
                    },
                    "required": ["project_id", "title", "description"],
                },
            ),
            Tool(
                name="cosmos_project_achieve_milestone",
                description="Mark a milestone as achieved.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "milestone_id": {
                            "type": "string",
                            "description": "ID of the milestone to mark as achieved",
                        },
                    },
                    "required": ["milestone_id"],
                },
            ),
            Tool(
                name="cosmos_project_link",
                description="Link two projects to enable knowledge transfer. Linked projects can share concepts and learnings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "Source project ID",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Target project ID",
                        },
                        "link_type": {
                            "type": "string",
                            "description": "Relationship type: 'depends_on', 'related_to', 'successor_of', 'informs'",
                            "enum": ["depends_on", "related_to", "successor_of", "informs"],
                            "default": "related_to",
                        },
                        "shared_concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concepts shared between projects",
                        },
                    },
                    "required": ["source_id", "target_id"],
                },
            ),
            Tool(
                name="cosmos_project_detect",
                description="Automatically detect a project from conversation text. Uses LLM to identify project information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze for project detection",
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="cosmos_project_transfer_knowledge",
                description="Transfer learnings and knowledge from one project to another. Uses LLM to identify transferable insights.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "from_id": {
                            "type": "string",
                            "description": "Source project ID",
                        },
                        "to_id": {
                            "type": "string",
                            "description": "Target project ID",
                        },
                    },
                    "required": ["from_id", "to_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "cosmos_remember":
            result = await cosmos.remember(
                content=arguments["content"],
                tags=arguments.get("tags"),
                importance=arguments.get("importance", 0.5),
            )
        elif name == "cosmos_recall":
            result = await cosmos.recall(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
            )
        elif name == "cosmos_delegate":
            result = await cosmos.delegate(
                task=arguments["task"],
                agent_type=arguments.get("agent_type", "auto"),
            )
        elif name == "cosmos_evolve":
            result = await cosmos.evolve(
                feedback=arguments["feedback"],
            )
        elif name == "cosmos_status":
            result = await cosmos.status()
        elif name == "cosmos_vision":
            result = await cosmos.vision_analyze(
                image=arguments["image"],
                task=arguments.get("task", "caption")
            )
        elif name == "cosmos_browse":
            result = await cosmos.browse(
                goal=arguments["goal"],
                url=arguments.get("url")
            )
        elif name == "cosmos_export":
            result = await cosmos.export_conversation(
                format=arguments.get("format", "markdown"),
                include_memories=arguments.get("include_memories", True),
                include_conversations=arguments.get("include_conversations", True),
                include_knowledge_graph=arguments.get("include_knowledge_graph", True),
                include_statistics=arguments.get("include_statistics", True),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                tags=arguments.get("tags"),
                output_path=arguments.get("output_path"),
            )
        elif name == "cosmos_list_exports":
            result = await cosmos.list_exports()
        # Project Tracking Tools
        elif name == "cosmos_project_create":
            result = await cosmos.project_create(
                name=arguments["name"],
                description=arguments["description"],
                tags=arguments.get("tags"),
                status=arguments.get("status", "active"),
            )
        elif name == "cosmos_project_update":
            result = await cosmos.project_update(
                project_id=arguments["project_id"],
                name=arguments.get("name"),
                description=arguments.get("description"),
                status=arguments.get("status"),
                tags=arguments.get("tags"),
            )
        elif name == "cosmos_project_list":
            result = await cosmos.project_list(
                status_filter=arguments.get("status_filter"),
                tag_filter=arguments.get("tag_filter"),
            )
        elif name == "cosmos_project_status":
            result = await cosmos.project_status(
                project_id=arguments["project_id"],
            )
        elif name == "cosmos_project_add_task":
            result = await cosmos.project_add_task(
                project_id=arguments["project_id"],
                title=arguments["title"],
                description=arguments["description"],
                priority=arguments.get("priority", 5),
                depends_on=arguments.get("depends_on"),
            )
        elif name == "cosmos_project_complete_task":
            result = await cosmos.project_complete_task(
                task_id=arguments["task_id"],
            )
        elif name == "cosmos_project_add_milestone":
            result = await cosmos.project_add_milestone(
                project_id=arguments["project_id"],
                title=arguments["title"],
                description=arguments["description"],
                milestone_type=arguments.get("milestone_type", "checkpoint"),
                target_date=arguments.get("target_date"),
                criteria=arguments.get("criteria"),
                task_ids=arguments.get("task_ids"),
            )
        elif name == "cosmos_project_achieve_milestone":
            result = await cosmos.project_achieve_milestone(
                milestone_id=arguments["milestone_id"],
            )
        elif name == "cosmos_project_link":
            result = await cosmos.project_link(
                source_id=arguments["source_id"],
                target_id=arguments["target_id"],
                link_type=arguments.get("link_type", "related_to"),
                shared_concepts=arguments.get("shared_concepts"),
            )
        elif name == "cosmos_project_detect":
            result = await cosmos.project_detect(
                text=arguments["text"],
            )
        elif name == "cosmos_project_transfer_knowledge":
            result = await cosmos.project_transfer_knowledge(
                from_id=arguments["from_id"],
                to_id=arguments["to_id"],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # Register resources
    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="cosmos://memory/recent",
                name="Recent Memories",
                description="Recent context and memories from cosmos",
                mimeType="text/plain",
            ),
            Resource(
                uri="cosmos://memory/graph",
                name="Knowledge Graph",
                description="Entity relationships and connections",
                mimeType="application/json",
            ),
            Resource(
                uri="cosmos://agents/active",
                name="Active Agents",
                description="Currently running specialist agents",
                mimeType="application/json",
            ),
            Resource(
                uri="cosmos://evolution/fitness",
                name="Fitness Metrics",
                description="System performance and evolution metrics",
                mimeType="application/json",
            ),
            Resource(
                uri="cosmos://proactive/suggestions",
                name="Proactive Suggestions",
                description="Anticipatory suggestions from the proactive agent",
                mimeType="application/json",
            ),
            Resource(
                uri="cosmos://system/health",
                name="System Health",
                description="Real-time health status and metrics",
                mimeType="application/json",
            ),
            Resource(
                uri="cosmos://exports/list",
                name="Export List",
                description="List of all available conversation exports",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str):
        if uri == "cosmos://memory/recent":
            content = await cosmos.get_recent_memories()
        elif uri == "cosmos://memory/graph":
            content = await cosmos.get_knowledge_graph()
        elif uri == "cosmos://agents/active":
            content = await cosmos.get_active_agents()
        elif uri == "cosmos://evolution/fitness":
            content = await cosmos.get_fitness_metrics()
        elif uri == "cosmos://proactive/suggestions":
            content = await cosmos.get_proactive_suggestions()
        elif uri == "cosmos://system/health":
            content = await cosmos.get_system_health()
        elif uri == "cosmos://exports/list":
            result = await cosmos.list_exports()
            content = json.dumps(result, indent=2)
        else:
            content = f"Unknown resource: {uri}"

        return content

    return server


async def run_server():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP library not installed. Run: pip install mcp")
        return

    logger.info("Starting cosmos MCP Server...")

    server = create_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
