"""
Cosmos LangGraph Workflows - Hybrid Nexus + Stateful Workflows.

AGI v1.8 Feature: Combines Nexus event-driven architecture with
LangGraph-style stateful workflows for complex multi-step operations.

Features:
- WorkflowState: TypedDict for workflow state management
- WorkflowCheckpoint: Persistent checkpoint storage
- LangGraphNexusHybrid: Bridges Nexus signals with workflow nodes
- DeliberationWorkflow: PROPOSE -> CRITIQUE -> REFINE -> VOTE
- TaskExecutionWorkflow: PLAN -> ASSIGN -> EXECUTE -> VERIFY
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import  Callable, Optional, TypedDict, Set, Awaitable

from loguru import logger


# =============================================================================
# WORKFLOW STATE DEFINITIONS
# =============================================================================

class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Type of workflow node."""
    START = "start"
    END = "end"
    ACTION = "action"
    DECISION = "decision"
    PARALLEL = "parallel"
    WAIT = "wait"


class WorkflowState(TypedDict, total=False):
    """TypedDict for workflow state - compatible with LangGraph patterns."""
    workflow_id: str
    current_node: str
    status: str
    context: dict
    history: list[dict]
    inputs: dict
    outputs: dict
    metadata: dict
    errors: list[str]
    created_at: str
    updated_at: str


@dataclass
class WorkflowNode:
    """Definition of a workflow node."""
    id: str
    name: str
    node_type: NodeType
    handler: Optional[Callable[[WorkflowState], Awaitable[WorkflowState]]] = None
    transitions: dict[str, str] = field(default_factory=dict)  # condition -> next_node
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow persistence and resumption."""
    checkpoint_id: str
    workflow_id: str
    state: WorkflowState
    node_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "state": dict(self.state),
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            state=data["state"],
            node_id=data["node_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkflowDefinition:
    """Complete workflow definition with nodes and transitions."""
    id: str
    name: str
    description: str
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    start_node: str = "start"
    end_nodes: Set[str] = field(default_factory=lambda: {"end"})
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# LANGGRAPH NEXUS HYBRID
# =============================================================================

class LangGraphNexusHybrid:
    """
    Bridges Nexus event bus with LangGraph-style stateful workflows.

    This hybrid system allows:
    - Workflows that respond to Nexus signals
    - Workflows that emit Nexus signals at key transitions
    - Checkpoint persistence for long-running workflows
    - Parallel workflow execution with coordination
    """

    def __init__(self, data_dir: str = "./data/workflows"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Workflow registries
        self._definitions: dict[str, WorkflowDefinition] = {}
        self._active_workflows: dict[str, WorkflowState] = {}
        self._checkpoints: dict[str, WorkflowCheckpoint] = {}

        # Nexus integration
        self._nexus = None
        self._signal_handlers: dict[str, Callable] = {}

        # Metrics
        self._workflow_metrics: dict[str[str]] = {}

        logger.info("LangGraphNexusHybrid initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to the Nexus event bus."""
        self._nexus = nexus
        logger.info("LangGraphNexusHybrid connected to Nexus")

    # =========================================================================
    # WORKFLOW DEFINITION
    # =========================================================================

    def register_workflow(self, definition: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._definitions[definition.id] = definition
        logger.info(f"Registered workflow: {definition.name} ({definition.id})")

    def create_workflow(
        self,
        name: str,
        description: str,
        nodes: list[WorkflowNode],
        start_node: str = "start",
        end_nodes: Optional[Set[str]] = None,
    ) -> WorkflowDefinition:
        """Create and register a new workflow definition."""
        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"

        definition = WorkflowDefinition(
            id=workflow_id,
            name=name,
            description=description,
            nodes={n.id: n for n in nodes},
            start_node=start_node,
            end_nodes=end_nodes or {"end"},
        )

        self.register_workflow(definition)
        return definition

    # =========================================================================
    # WORKFLOW EXECUTION
    # =========================================================================

    async def start_workflow(
        self,
        definition_id: str,
        inputs: dict,
        context: Optional[dict] = None,
    ) -> str:
        """Start a new workflow instance."""
        if definition_id not in self._definitions:
            raise ValueError(f"Unknown workflow: {definition_id}")

        definition = self._definitions[definition_id]
        workflow_id = f"run_{uuid.uuid4().hex[:12]}"

        state: WorkflowState = {
            "workflow_id": workflow_id,
            "current_node": definition.start_node,
            "status": WorkflowStatus.RUNNING.value,
            "context": context or {},
            "history": [],
            "inputs": inputs,
            "outputs": {},
            "metadata": {
                "definition_id": definition_id,
                "definition_name": definition.name,
            },
            "errors": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        self._active_workflows[workflow_id] = state

        # Emit workflow started signal
        await self._emit_signal("WORKFLOW_STARTED", {
            "workflow_id": workflow_id,
            "definition_id": definition_id,
            "inputs": inputs,
        })

        # Start execution
        asyncio.create_task(self._execute_workflow(workflow_id, definition))

        logger.info(f"Started workflow {workflow_id} ({definition.name})")
        return workflow_id

    async def _execute_workflow(
        self,
        workflow_id: str,
        definition: WorkflowDefinition,
    ) -> None:
        """Execute workflow nodes until completion or pause."""
        state = self._active_workflows.get(workflow_id)
        if not state:
            return

        while state["status"] == WorkflowStatus.RUNNING.value:
            current_node_id = state["current_node"]
            node = definition.nodes.get(current_node_id)

            if not node:
                state["status"] = WorkflowStatus.FAILED.value
                state["errors"].append(f"Unknown node: {current_node_id}")
                break

            # Check for end node
            if current_node_id in definition.end_nodes:
                state["status"] = WorkflowStatus.COMPLETED.value
                await self._emit_signal("WORKFLOW_COMPLETED", {
                    "workflow_id": workflow_id,
                    "outputs": state["outputs"],
                })
                break

            # Emit node entered signal
            await self._emit_signal("WORKFLOW_NODE_ENTERED", {
                "workflow_id": workflow_id,
                "node_id": current_node_id,
                "node_name": node.name,
            })

            # Execute node handler
            try:
                if node.handler:
                    state = await asyncio.wait_for(
                        node.handler(state),
                        timeout=node.timeout_seconds,
                    )
                    self._active_workflows[workflow_id] = state

                # Record in history
                state["history"].append({
                    "node_id": current_node_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                })

            except asyncio.TimeoutError:
                node.retry_count += 1
                if node.retry_count >= node.max_retries:
                    state["status"] = WorkflowStatus.FAILED.value
                    state["errors"].append(f"Node {current_node_id} timed out")
                    await self._emit_signal("WORKFLOW_FAILED", {
                        "workflow_id": workflow_id,
                        "error": f"Timeout at node {current_node_id}",
                    })
                    break
                continue

            except Exception as e:
                state["status"] = WorkflowStatus.FAILED.value
                state["errors"].append(str(e))
                await self._emit_signal("WORKFLOW_FAILED", {
                    "workflow_id": workflow_id,
                    "error": str(e),
                })
                break

            # Emit node exited signal
            await self._emit_signal("WORKFLOW_NODE_EXITED", {
                "workflow_id": workflow_id,
                "node_id": current_node_id,
            })

            # Determine next node
            next_node = self._get_next_node(node, state)
            if next_node:
                state["current_node"] = next_node
            else:
                state["status"] = WorkflowStatus.FAILED.value
                state["errors"].append(f"No valid transition from {current_node_id}")
                break

            state["updated_at"] = datetime.now().isoformat()

        logger.info(f"Workflow {workflow_id} finished with status: {state['status']}")

    def _get_next_node(self, node: WorkflowNode, state: WorkflowState) -> Optional[str]:
        """Determine the next node based on transitions and state."""
        if not node.transitions:
            return None

        # Check conditional transitions
        for condition, next_node in node.transitions.items():
            if condition == "default":
                continue
            if self._evaluate_condition(condition, state):
                return next_node

        # Fall back to default transition
        return node.transitions.get("default")

    def _evaluate_condition(self, condition: str, state: WorkflowState) -> bool:
        """Evaluate a transition condition against state (sandboxed)."""
        from Cosmos.core.safe_eval import safe_eval

        context = state.get("context", {})
        outputs = state.get("outputs", {})

        try:
            namespace = {"context": context, "outputs": outputs, "state": state}
            return bool(safe_eval(condition, namespace))
        except Exception:
            return False

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================

    async def checkpoint_workflow(self, workflow_id: str) -> str:
        """Create a checkpoint for a workflow."""
        state = self._active_workflows.get(workflow_id)
        if not state:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            state=state.copy(),
            node_id=state["current_node"],
        )

        self._checkpoints[checkpoint_id] = checkpoint

        # Persist to disk
        checkpoint_path = self.data_dir / f"{checkpoint_id}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        await self._emit_signal("WORKFLOW_CHECKPOINT", {
            "workflow_id": workflow_id,
            "checkpoint_id": checkpoint_id,
            "node_id": state["current_node"],
        })

        logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_id}")
        return checkpoint_id

    async def resume_workflow(self, checkpoint_id: str) -> str:
        """Resume a workflow from a checkpoint."""
        # Load from memory or disk
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            checkpoint_path = self.data_dir / f"{checkpoint_id}.json"
            if checkpoint_path.exists():
                with open(checkpoint_path) as f:
                    checkpoint = WorkflowCheckpoint.from_dict(json.load(f))
            else:
                raise ValueError(f"Unknown checkpoint: {checkpoint_id}")

        # Get definition
        definition_id = checkpoint.state.get("metadata", {}).get("definition_id")
        if not definition_id or definition_id not in self._definitions:
            raise ValueError(f"Unknown workflow definition: {definition_id}")

        definition = self._definitions[definition_id]

        # Create new workflow instance from checkpoint state
        new_workflow_id = f"run_{uuid.uuid4().hex[:12]}"
        state = checkpoint.state.copy()
        state["workflow_id"] = new_workflow_id
        state["status"] = WorkflowStatus.RUNNING.value
        state["updated_at"] = datetime.now().isoformat()
        state["metadata"]["resumed_from"] = checkpoint_id

        self._active_workflows[new_workflow_id] = state

        await self._emit_signal("WORKFLOW_RESUMED", {
            "workflow_id": new_workflow_id,
            "checkpoint_id": checkpoint_id,
            "node_id": state["current_node"],
        })

        # Continue execution
        asyncio.create_task(self._execute_workflow(new_workflow_id, definition))

        logger.info(f"Resumed workflow {new_workflow_id} from checkpoint {checkpoint_id}")
        return new_workflow_id

    # =========================================================================
    # NEXUS INTEGRATION
    # =========================================================================

    async def _emit_signal(self, signal_type: str, payload: dict) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            # Import here to avoid circular dependency
            from Cosmos.core.nexus import SignalType

            # Map string to SignalType enum
            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source="langgraph_hybrid",
                    urgency=0.6,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    def on_nexus_signal(
        self,
        signal_type: str,
        handler: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Register a handler for Nexus signals."""
        self._signal_handlers[signal_type] = handler

    async def emit_with_workflow_state(
        self,
        workflow_id: str,
        signal_type: str,
        additional_payload: Optional[dict] = None,
    ) -> None:
        """Emit a Nexus signal with workflow state context."""
        state = self._active_workflows.get(workflow_id)
        if not state:
            return

        payload = {
            "workflow_id": workflow_id,
            "current_node": state["current_node"],
            "status": state["status"],
            **(additional_payload or {}),
        }

        await self._emit_signal(signal_type, payload)

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_workflow_status(self, workflow_id: str) -> Optional[dict]:
        """Get the current status of a workflow."""
        state = self._active_workflows.get(workflow_id)
        if not state:
            return None

        return {
            "workflow_id": workflow_id,
            "status": state["status"],
            "current_node": state["current_node"],
            "history_length": len(state["history"]),
            "errors": state["errors"],
            "created_at": state["created_at"],
            "updated_at": state["updated_at"],
        }

    def get_stats(self) -> dict:
        """Get workflow system statistics."""
        active_count = sum(
            1 for s in self._active_workflows.values()
            if s["status"] == WorkflowStatus.RUNNING.value
        )
        return {
            "registered_workflows": len(self._definitions),
            "active_workflows": active_count,
            "total_runs": len(self._active_workflows),
            "checkpoints": len(self._checkpoints),
        }


# =============================================================================
# DELIBERATION WORKFLOW (PROPOSE -> CRITIQUE -> REFINE -> VOTE)
# =============================================================================

class DeliberationWorkflow:
    """
    Standard workflow for agent deliberation.

    Flow: PROPOSE -> CRITIQUE -> REFINE -> VOTE -> CONSENSUS
    """

    def __init__(self, hybrid: LangGraphNexusHybrid):
        self.hybrid = hybrid
        self._definition: Optional[WorkflowDefinition] = None
        self._setup_workflow()

    def _setup_workflow(self) -> None:
        """Create the deliberation workflow definition."""
        nodes = [
            WorkflowNode(
                id="start",
                name="Start Deliberation",
                node_type=NodeType.START,
                handler=self._handle_start,
                transitions={"default": "propose"},
            ),
            WorkflowNode(
                id="propose",
                name="Propose Solutions",
                node_type=NodeType.ACTION,
                handler=self._handle_propose,
                transitions={"default": "critique"},
            ),
            WorkflowNode(
                id="critique",
                name="Critique Proposals",
                node_type=NodeType.ACTION,
                handler=self._handle_critique,
                transitions={"default": "refine"},
            ),
            WorkflowNode(
                id="refine",
                name="Refine Solutions",
                node_type=NodeType.ACTION,
                handler=self._handle_refine,
                transitions={
                    "outputs.get('needs_more_critique', False)": "critique",
                    "default": "vote",
                },
            ),
            WorkflowNode(
                id="vote",
                name="Vote on Solutions",
                node_type=NodeType.ACTION,
                handler=self._handle_vote,
                transitions={
                    "outputs.get('consensus_reached', False)": "end",
                    "default": "refine",
                },
            ),
            WorkflowNode(
                id="end",
                name="Deliberation Complete",
                node_type=NodeType.END,
                handler=None,
                transitions={},
            ),
        ]

        self._definition = self.hybrid.create_workflow(
            name="Deliberation Workflow",
            description="Multi-agent deliberation with propose, critique, refine, and vote phases",
            nodes=nodes,
            start_node="start",
            end_nodes={"end"},
        )

    async def start(
        self,
        topic: str,
        participants: list[str],
        context: Optional[dict] = None,
    ) -> str:
        """Start a new deliberation."""
        inputs = {
            "topic": topic,
            "participants": participants,
            "proposals": [],
            "critiques": [],
            "refined_proposals": [],
            "votes": {},
        }
        return await self.hybrid.start_workflow(
            self._definition.id,
            inputs=inputs,
            context=context,
        )

    async def _handle_start(self, state: WorkflowState) -> WorkflowState:
        """Initialize deliberation state."""
        state["context"]["deliberation_round"] = 0
        state["context"]["max_rounds"] = 3
        return state

    async def _handle_propose(self, state: WorkflowState) -> WorkflowState:
        """Collect proposals from participants."""
        # Placeholder - in real implementation, would gather from agents
        state["outputs"]["proposals"] = state["inputs"].get("proposals", [])
        state["context"]["deliberation_round"] += 1
        return state

    async def _handle_critique(self, state: WorkflowState) -> WorkflowState:
        """Collect critiques of proposals."""
        state["outputs"]["critiques"] = []
        return state

    async def _handle_refine(self, state: WorkflowState) -> WorkflowState:
        """Refine proposals based on critiques."""
        state["outputs"]["refined_proposals"] = state["outputs"].get("proposals", [])
        state["outputs"]["needs_more_critique"] = False
        return state

    async def _handle_vote(self, state: WorkflowState) -> WorkflowState:
        """Vote on refined proposals."""
        state["outputs"]["votes"] = {}
        state["outputs"]["consensus_reached"] = True
        state["outputs"]["winning_proposal"] = None
        return state


# =============================================================================
# TASK EXECUTION WORKFLOW (PLAN -> ASSIGN -> EXECUTE -> VERIFY)
# =============================================================================

class TaskExecutionWorkflow:
    """
    Standard workflow for task execution.

    Flow: PLAN -> ASSIGN -> EXECUTE -> VERIFY -> COMPLETE
    """

    def __init__(self, hybrid: LangGraphNexusHybrid):
        self.hybrid = hybrid
        self._definition: Optional[WorkflowDefinition] = None
        self._setup_workflow()

    def _setup_workflow(self) -> None:
        """Create the task execution workflow definition."""
        nodes = [
            WorkflowNode(
                id="start",
                name="Start Task",
                node_type=NodeType.START,
                handler=self._handle_start,
                transitions={"default": "plan"},
            ),
            WorkflowNode(
                id="plan",
                name="Plan Execution",
                node_type=NodeType.ACTION,
                handler=self._handle_plan,
                transitions={"default": "assign"},
            ),
            WorkflowNode(
                id="assign",
                name="Assign Agents",
                node_type=NodeType.ACTION,
                handler=self._handle_assign,
                transitions={"default": "execute"},
            ),
            WorkflowNode(
                id="execute",
                name="Execute Task",
                node_type=NodeType.ACTION,
                handler=self._handle_execute,
                transitions={"default": "verify"},
                timeout_seconds=600.0,
            ),
            WorkflowNode(
                id="verify",
                name="Verify Results",
                node_type=NodeType.ACTION,
                handler=self._handle_verify,
                transitions={
                    "outputs.get('verified', False)": "end",
                    "outputs.get('retry', False)": "execute",
                    "default": "end",
                },
            ),
            WorkflowNode(
                id="end",
                name="Task Complete",
                node_type=NodeType.END,
                handler=None,
                transitions={},
            ),
        ]

        self._definition = self.hybrid.create_workflow(
            name="Task Execution Workflow",
            description="Plan, assign, execute, and verify task workflow",
            nodes=nodes,
            start_node="start",
            end_nodes={"end"},
        )

    async def start(
        self,
        task_description: str,
        required_capabilities: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ) -> str:
        """Start a new task execution."""
        inputs = {
            "task_description": task_description,
            "required_capabilities": required_capabilities or [],
            "subtasks": [],
            "assigned_agents": {},
            "results": [],
        }
        return await self.hybrid.start_workflow(
            self._definition.id,
            inputs=inputs,
            context=context,
        )

    async def _handle_start(self, state: WorkflowState) -> WorkflowState:
        """Initialize task execution state."""
        state["context"]["execution_attempts"] = 0
        state["context"]["max_attempts"] = 3
        return state

    async def _handle_plan(self, state: WorkflowState) -> WorkflowState:
        """Plan task execution - break into subtasks."""
        # Placeholder - in real implementation, would use planner agent
        task_desc = state["inputs"]["task_description"]
        state["outputs"]["subtasks"] = [
            {"id": "sub_1", "description": task_desc, "status": "pending"}
        ]
        state["outputs"]["execution_plan"] = {
            "strategy": "sequential",
            "estimated_steps": 1,
        }
        return state

    async def _handle_assign(self, state: WorkflowState) -> WorkflowState:
        """Assign agents to subtasks."""
        subtasks = state["outputs"].get("subtasks", [])
        capabilities = state["inputs"].get("required_capabilities", [])

        assignments = {}
        for subtask in subtasks:
            assignments[subtask["id"]] = {
                "agent_id": None,  # Would be assigned by orchestrator
                "capabilities": capabilities,
            }

        state["outputs"]["assigned_agents"] = assignments
        return state

    async def _handle_execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the task through assigned agents."""
        state["context"]["execution_attempts"] += 1
        state["outputs"]["results"] = []
        state["outputs"]["execution_status"] = "completed"
        return state

    async def _handle_verify(self, state: WorkflowState) -> WorkflowState:
        """Verify task execution results."""
        results = state["outputs"].get("results", [])
        attempts = state["context"].get("execution_attempts", 0)
        max_attempts = state["context"].get("max_attempts", 3)

        # Simple verification - in real implementation, would use critic agent
        state["outputs"]["verified"] = True
        state["outputs"]["retry"] = False

        if not state["outputs"]["verified"] and attempts < max_attempts:
            state["outputs"]["retry"] = True

        return state


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_langgraph_hybrid(data_dir: str = "./data/workflows") -> LangGraphNexusHybrid:
    """Factory function to create a LangGraphNexusHybrid instance."""
    return LangGraphNexusHybrid(data_dir=data_dir)


def create_deliberation_workflow(hybrid: LangGraphNexusHybrid) -> DeliberationWorkflow:
    """Factory function to create a DeliberationWorkflow."""
    return DeliberationWorkflow(hybrid)


def create_task_workflow(hybrid: LangGraphNexusHybrid) -> TaskExecutionWorkflow:
    """Factory function to create a TaskExecutionWorkflow."""
    return TaskExecutionWorkflow(hybrid)
