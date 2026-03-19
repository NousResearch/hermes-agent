"""
cosmos Visual Workflow Builder

"I've invented a machine that builds machines that build workflows!"

Create, edit, and execute automation workflows with a visual node-based system.
"""

import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class NodeType(Enum):
    """Types of workflow nodes."""
    TRIGGER = "trigger"
    ACTION = "action"
    CONDITION = "condition"
    LOOP = "loop"
    TRANSFORM = "transform"
    HTTP_REQUEST = "http_request"
    CODE = "code"
    WAIT = "wait"
    SPLIT = "split"
    MERGE = "merge"
    ERROR_HANDLER = "error_handler"
    SUBWORKFLOW = "subworkflow"


class TriggerType(Enum):
    """Types of workflow triggers."""
    MANUAL = "manual"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EVENT = "event"
    FILE_CHANGE = "file_change"
    EMAIL = "email"
    API_CALL = "api_call"


class ActionType(Enum):
    """Types of workflow actions."""
    # Communication
    SEND_EMAIL = "send_email"
    SEND_SLACK = "send_slack"
    SEND_DISCORD = "send_discord"
    SEND_TELEGRAM = "send_telegram"
    SEND_WEBHOOK = "send_webhook"

    # Cloud Operations
    AWS_EC2 = "aws_ec2"
    AWS_S3 = "aws_s3"
    AWS_LAMBDA = "aws_lambda"
    AZURE_VM = "azure_vm"
    AZURE_STORAGE = "azure_storage"
    AZURE_FUNCTION = "azure_function"
    GCP_COMPUTE = "gcp_compute"
    GCP_STORAGE = "gcp_storage"
    GCP_FUNCTION = "gcp_function"

    # DevOps
    DOCKER_RUN = "docker_run"
    KUBERNETES_DEPLOY = "kubernetes_deploy"
    TERRAFORM_APPLY = "terraform_apply"
    ANSIBLE_PLAYBOOK = "ansible_playbook"

    # Data
    DATABASE_QUERY = "database_query"
    HTTP_REQUEST = "http_request"
    FILE_OPERATION = "file_operation"
    TRANSFORM_DATA = "transform_data"

    # Security
    VULN_SCAN = "vuln_scan"
    LOG_ANALYSIS = "log_analysis"
    INCIDENT_CREATE = "incident_create"

    # AI
    LLM_PROMPT = "llm_prompt"
    cosmos_TOOL = "cosmos_tool"

    # System
    SHELL_COMMAND = "shell_command"
    PYTHON_CODE = "python_code"
    WAIT = "wait"


@dataclass
class WorkflowNode:
    """A node in the workflow graph."""
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    inputs: List[str] = field(default_factory=list)  # Node IDs this receives from
    outputs: List[str] = field(default_factory=list)  # Node IDs this sends to
    enabled: bool = True
    retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay": 1000,
        "retry_on_error": True
    })
    timeout_ms: int = 60000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "config": self.config,
            "position": self.position,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "enabled": self.enabled,
            "retry_config": self.retry_config,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            name=data["name"],
            config=data.get("config", {}),
            position=data.get("position", {"x": 0, "y": 0}),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            enabled=data.get("enabled", True),
            retry_config=data.get("retry_config", {}),
            timeout_ms=data.get("timeout_ms", 60000),
        )


@dataclass
class WorkflowTrigger:
    """Trigger configuration for a workflow."""
    type: TriggerType
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "config": self.config,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowTrigger":
        return cls(
            type=TriggerType(data["type"]),
            config=data.get("config", {}),
            enabled=data.get("enabled", True),
        )


@dataclass
class WorkflowExecution:
    """Record of a workflow execution."""
    id: str
    workflow_id: str
    status: str  # pending, running, completed, failed, cancelled
    started_at: datetime
    completed_at: Optional[datetime] = None
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    node_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "trigger_data": self.trigger_data,
            "node_results": self.node_results,
            "error": self.error,
        }


@dataclass
class Workflow:
    """Complete workflow definition."""
    id: str
    name: str
    description: str = ""
    nodes: List[WorkflowNode] = field(default_factory=list)
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "timezone": "UTC",
        "error_workflow": None,
        "max_concurrent": 1,
    })
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "triggers": [t.to_dict() for t in self.triggers],
            "variables": self.variables,
            "settings": self.settings,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "enabled": self.enabled,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            nodes=[WorkflowNode.from_dict(n) for n in data.get("nodes", [])],
            triggers=[WorkflowTrigger.from_dict(t) for t in data.get("triggers", [])],
            variables=data.get("variables", {}),
            settings=data.get("settings", {}),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            enabled=data.get("enabled", True),
            version=data.get("version", 1),
        )

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_start_nodes(self) -> List[WorkflowNode]:
        """Get nodes with no inputs (start of workflow)."""
        return [n for n in self.nodes if not n.inputs and n.type != NodeType.TRIGGER]

    def get_trigger_nodes(self) -> List[WorkflowNode]:
        """Get trigger nodes."""
        return [n for n in self.nodes if n.type == NodeType.TRIGGER]

    def validate(self) -> List[str]:
        """Validate workflow configuration. Returns list of errors."""
        errors = []

        # Check for orphan nodes
        node_ids = {n.id for n in self.nodes}
        for node in self.nodes:
            for input_id in node.inputs:
                if input_id not in node_ids:
                    errors.append(f"Node {node.name} references non-existent input {input_id}")
            for output_id in node.outputs:
                if output_id not in node_ids:
                    errors.append(f"Node {node.name} references non-existent output {output_id}")

        # Check for cycles (basic check)
        # TODO: Implement full cycle detection

        # Check trigger configuration
        if not self.triggers and not any(n.type == NodeType.TRIGGER for n in self.nodes):
            errors.append("Workflow has no triggers defined")

        return errors


class WorkflowAction:
    """Base class for workflow action executors."""

    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    async def execute(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action. Override in subclasses."""
        raise NotImplementedError


class WorkflowBuilder:
    """
    Visual workflow builder and executor.

    Creates, manages, and executes automation workflows with a node-based
    architecture similar to n8n.
    """

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("./data/workflows")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self._register_default_handlers()
        self._load_workflows()

    def _load_workflows(self):
        """Load workflows from storage."""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    workflow = Workflow.from_dict(data)
                    self.workflows[workflow.id] = workflow
                    logger.debug(f"Loaded workflow: {workflow.name}")
            except Exception as e:
                logger.error(f"Failed to load workflow {file}: {e}")

    def _save_workflow(self, workflow: Workflow):
        """Save workflow to storage."""
        file_path = self.storage_path / f"{workflow.id}.json"
        with open(file_path, "w") as f:
            json.dump(workflow.to_dict(), f, indent=2)

    def _register_default_handlers(self):
        """Register default action handlers."""
        # HTTP Request
        self.action_handlers[ActionType.HTTP_REQUEST] = self._handle_http_request

        # Shell Command
        self.action_handlers[ActionType.SHELL_COMMAND] = self._handle_shell_command

        # Python Code
        self.action_handlers[ActionType.PYTHON_CODE] = self._handle_python_code

        # Wait
        self.action_handlers[ActionType.WAIT] = self._handle_wait

        # Transform Data
        self.action_handlers[ActionType.TRANSFORM_DATA] = self._handle_transform

        # Send Webhook
        self.action_handlers[ActionType.SEND_WEBHOOK] = self._handle_send_webhook

        # LLM Prompt
        self.action_handlers[ActionType.LLM_PROMPT] = self._handle_llm_prompt

    # =========================================================================
    # WORKFLOW CRUD
    # =========================================================================

    def create_workflow(
        self,
        name: str,
        description: str = "",
        tags: List[str] = None
    ) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            tags=tags or [],
        )
        self.workflows[workflow.id] = workflow
        self._save_workflow(workflow)
        logger.info(f"Created workflow: {name}")
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)

    def list_workflows(self, tags: List[str] = None) -> List[Workflow]:
        """List all workflows, optionally filtered by tags."""
        workflows = list(self.workflows.values())
        if tags:
            workflows = [w for w in workflows if any(t in w.tags for t in tags)]
        return workflows

    def update_workflow(self, workflow: Workflow) -> Workflow:
        """Update an existing workflow."""
        workflow.updated_at = datetime.utcnow()
        workflow.version += 1
        self.workflows[workflow.id] = workflow
        self._save_workflow(workflow)
        logger.info(f"Updated workflow: {workflow.name} (v{workflow.version})")
        return workflow

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        if workflow_id in self.workflows:
            workflow = self.workflows.pop(workflow_id)
            file_path = self.storage_path / f"{workflow_id}.json"
            if file_path.exists():
                file_path.unlink()
            logger.info(f"Deleted workflow: {workflow.name}")
            return True
        return False

    def clone_workflow(self, workflow_id: str, new_name: str) -> Optional[Workflow]:
        """Clone an existing workflow."""
        original = self.get_workflow(workflow_id)
        if not original:
            return None

        # Deep copy
        data = original.to_dict()
        data["id"] = str(uuid.uuid4())
        data["name"] = new_name
        data["created_at"] = datetime.utcnow().isoformat()
        data["updated_at"] = datetime.utcnow().isoformat()
        data["version"] = 1

        # Generate new node IDs
        id_map = {}
        for node in data["nodes"]:
            old_id = node["id"]
            new_id = str(uuid.uuid4())
            id_map[old_id] = new_id
            node["id"] = new_id

        # Update references
        for node in data["nodes"]:
            node["inputs"] = [id_map.get(i, i) for i in node["inputs"]]
            node["outputs"] = [id_map.get(o, o) for o in node["outputs"]]

        cloned = Workflow.from_dict(data)
        self.workflows[cloned.id] = cloned
        self._save_workflow(cloned)
        logger.info(f"Cloned workflow: {original.name} -> {new_name}")
        return cloned

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    def add_node(
        self,
        workflow_id: str,
        node_type: NodeType,
        name: str,
        config: Dict[str, Any] = None,
        position: Dict[str, int] = None
    ) -> Optional[WorkflowNode]:
        """Add a node to a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None

        node = WorkflowNode(
            id=str(uuid.uuid4()),
            type=node_type,
            name=name,
            config=config or {},
            position=position or {"x": 0, "y": 0},
        )
        workflow.nodes.append(node)
        self.update_workflow(workflow)
        return node

    def connect_nodes(
        self,
        workflow_id: str,
        from_node_id: str,
        to_node_id: str
    ) -> bool:
        """Connect two nodes in a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False

        from_node = workflow.get_node(from_node_id)
        to_node = workflow.get_node(to_node_id)

        if not from_node or not to_node:
            return False

        if to_node_id not in from_node.outputs:
            from_node.outputs.append(to_node_id)
        if from_node_id not in to_node.inputs:
            to_node.inputs.append(from_node_id)

        self.update_workflow(workflow)
        return True

    def disconnect_nodes(
        self,
        workflow_id: str,
        from_node_id: str,
        to_node_id: str
    ) -> bool:
        """Disconnect two nodes in a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False

        from_node = workflow.get_node(from_node_id)
        to_node = workflow.get_node(to_node_id)

        if not from_node or not to_node:
            return False

        if to_node_id in from_node.outputs:
            from_node.outputs.remove(to_node_id)
        if from_node_id in to_node.inputs:
            to_node.inputs.remove(from_node_id)

        self.update_workflow(workflow)
        return True

    def remove_node(self, workflow_id: str, node_id: str) -> bool:
        """Remove a node from a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return False

        node = workflow.get_node(node_id)
        if not node:
            return False

        # Remove connections
        for other_node in workflow.nodes:
            if node_id in other_node.inputs:
                other_node.inputs.remove(node_id)
            if node_id in other_node.outputs:
                other_node.outputs.remove(node_id)

        workflow.nodes.remove(node)
        self.update_workflow(workflow)
        return True

    # =========================================================================
    # TRIGGERS
    # =========================================================================

    def add_trigger(
        self,
        workflow_id: str,
        trigger_type: TriggerType,
        config: Dict[str, Any] = None
    ) -> Optional[WorkflowTrigger]:
        """Add a trigger to a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None

        trigger = WorkflowTrigger(
            type=trigger_type,
            config=config or {},
        )
        workflow.triggers.append(trigger)
        self.update_workflow(workflow)
        return trigger

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        if not workflow.enabled:
            raise ValueError(f"Workflow is disabled: {workflow.name}")

        # Validate
        errors = workflow.validate()
        if errors:
            raise ValueError(f"Workflow validation failed: {errors}")

        # Create execution record
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.utcnow(),
            trigger_data=trigger_data or {},
        )
        self.executions[execution.id] = execution

        logger.info(f"Starting workflow execution: {workflow.name} ({execution.id})")

        try:
            # Build execution context
            context = {
                "workflow": workflow.to_dict(),
                "execution_id": execution.id,
                "trigger_data": trigger_data or {},
                "variables": workflow.variables.copy(),
                "node_outputs": {},
            }

            # Execute nodes in topological order
            executed = set()
            start_nodes = workflow.get_start_nodes()

            if not start_nodes:
                # If no explicit start nodes, use nodes connected to triggers
                trigger_nodes = workflow.get_trigger_nodes()
                for trigger_node in trigger_nodes:
                    start_nodes.extend([
                        workflow.get_node(out_id)
                        for out_id in trigger_node.outputs
                    ])

            await self._execute_nodes(workflow, start_nodes, context, executed, execution)

            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            logger.info(f"Workflow completed: {workflow.name} ({execution.id})")

        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow failed: {workflow.name} ({execution.id}): {e}")

        return execution

    async def _execute_nodes(
        self,
        workflow: Workflow,
        nodes: List[WorkflowNode],
        context: Dict[str, Any],
        executed: set,
        execution: WorkflowExecution
    ):
        """Execute a list of nodes and their descendants."""
        for node in nodes:
            if node.id in executed or not node.enabled:
                continue

            # Check if all inputs are ready
            if not all(inp in executed for inp in node.inputs):
                continue

            # Execute node
            try:
                result = await self._execute_node(node, context)
                context["node_outputs"][node.id] = result
                execution.node_results[node.id] = {
                    "status": "success",
                    "output": result,
                }
                executed.add(node.id)

                # Execute downstream nodes
                next_nodes = [workflow.get_node(out_id) for out_id in node.outputs]
                next_nodes = [n for n in next_nodes if n is not None]
                await self._execute_nodes(workflow, next_nodes, context, executed, execution)

            except Exception as e:
                execution.node_results[node.id] = {
                    "status": "error",
                    "error": str(e),
                }

                if node.retry_config.get("retry_on_error", True):
                    # Retry logic
                    max_retries = node.retry_config.get("max_retries", 3)
                    retry_delay = node.retry_config.get("retry_delay", 1000) / 1000

                    for attempt in range(max_retries):
                        await asyncio.sleep(retry_delay)
                        try:
                            result = await self._execute_node(node, context)
                            context["node_outputs"][node.id] = result
                            execution.node_results[node.id] = {
                                "status": "success",
                                "output": result,
                                "retries": attempt + 1,
                            }
                            executed.add(node.id)
                            break
                        except Exception as retry_e:
                            if attempt == max_retries - 1:
                                raise retry_e
                else:
                    raise

    async def _execute_node(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single node."""
        logger.debug(f"Executing node: {node.name} ({node.type.value})")

        # Resolve variables in config
        config = self._resolve_variables(node.config, context)

        if node.type == NodeType.ACTION:
            action_type = ActionType(config.get("action_type", "shell_command"))
            handler = self.action_handlers.get(action_type)
            if handler:
                return await handler(config, context)
            else:
                raise ValueError(f"No handler for action type: {action_type}")

        elif node.type == NodeType.CONDITION:
            return await self._handle_condition(config, context)

        elif node.type == NodeType.LOOP:
            return await self._handle_loop(config, context)

        elif node.type == NodeType.CODE:
            return await self._handle_python_code(config, context)

        elif node.type == NodeType.HTTP_REQUEST:
            return await self._handle_http_request(config, context)

        elif node.type == NodeType.TRANSFORM:
            return await self._handle_transform(config, context)

        elif node.type == NodeType.WAIT:
            return await self._handle_wait(config, context)

        else:
            return {"status": "skipped", "type": node.type.value}

    def _resolve_variables(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve variables in configuration."""
        import re

        def resolve_value(value):
            if isinstance(value, str):
                # Match {{ variable }} patterns
                pattern = r'\{\{\s*([^}]+)\s*\}\}'
                matches = re.findall(pattern, value)
                for match in matches:
                    parts = match.strip().split(".")
                    resolved = context
                    for part in parts:
                        if isinstance(resolved, dict):
                            resolved = resolved.get(part, "")
                        else:
                            resolved = ""
                            break
                    value = value.replace(f"{{{{{match}}}}}", str(resolved))
                return value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            return value

        return resolve_value(config)

    # =========================================================================
    # ACTION HANDLERS
    # =========================================================================

    async def _handle_http_request(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle HTTP request action."""
        import aiohttp

        method = config.get("method", "GET").upper()
        url = config.get("url")
        headers = config.get("headers", {})
        body = config.get("body")
        timeout = config.get("timeout", 30)

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=headers,
                json=body if isinstance(body, dict) else None,
                data=body if isinstance(body, str) else None,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                try:
                    response_body = await response.json()
                except:
                    response_body = await response.text()

                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "body": response_body,
                }

    async def _handle_shell_command(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle shell command action."""
        import subprocess

        command = config.get("command")
        cwd = config.get("cwd")
        timeout = config.get("timeout", 60)
        shell = config.get("shell", True)

        result = subprocess.run(
            command,
            shell=shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    async def _handle_python_code(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Python code execution."""
        code = config.get("code", "")

        # Create safe execution environment
        local_vars = {
            "context": context,
            "result": None,
        }

        exec(code, {"__builtins__": __builtins__}, local_vars)

        return {"result": local_vars.get("result")}

    async def _handle_wait(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle wait action."""
        duration_ms = config.get("duration_ms", 1000)
        await asyncio.sleep(duration_ms / 1000)
        return {"waited_ms": duration_ms}

    async def _handle_transform(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle data transformation."""
        operation = config.get("operation", "passthrough")
        input_data = config.get("input", context.get("node_outputs", {}))

        if operation == "passthrough":
            return {"data": input_data}
        elif operation == "filter":
            # Filter array based on condition
            condition = config.get("condition", "true")
            if isinstance(input_data, list):
                filtered = [
                    item for item in input_data
                    if eval(condition, {"item": item})
                ]
                return {"data": filtered}
        elif operation == "map":
            # Map transformation
            expression = config.get("expression", "item")
            if isinstance(input_data, list):
                mapped = [
                    eval(expression, {"item": item})
                    for item in input_data
                ]
                return {"data": mapped}
        elif operation == "reduce":
            # Reduce to single value
            expression = config.get("expression", "acc + item")
            initial = config.get("initial", 0)
            if isinstance(input_data, list):
                acc = initial
                for item in input_data:
                    acc = eval(expression, {"acc": acc, "item": item})
                return {"data": acc}
        elif operation == "json_parse":
            import json
            return {"data": json.loads(input_data)}
        elif operation == "json_stringify":
            import json
            return {"data": json.dumps(input_data)}

        return {"data": input_data}

    async def _handle_condition(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle condition node."""
        condition = config.get("condition", "true")
        result = eval(condition, {"context": context})
        return {"result": bool(result), "branch": "true" if result else "false"}

    async def _handle_loop(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle loop node."""
        items = config.get("items", [])
        # Loop execution is handled in the main executor
        return {"items": items, "count": len(items)}

    async def _handle_send_webhook(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle webhook sending."""
        return await self._handle_http_request({
            "method": "POST",
            "url": config.get("url"),
            "headers": config.get("headers", {"Content-Type": "application/json"}),
            "body": config.get("body", context.get("trigger_data")),
        }, context)

    async def _handle_llm_prompt(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle LLM prompt action."""
        # This would integrate with cosmos's LLM system
        prompt = config.get("prompt", "")
        model = config.get("model", "default")

        # Placeholder - integrate with actual LLM system
        return {
            "prompt": prompt,
            "model": model,
            "response": f"[LLM Response to: {prompt[:50]}...]"
        }

    # =========================================================================
    # TEMPLATES
    # =========================================================================

    def create_from_template(self, template_name: str, name: str) -> Optional[Workflow]:
        """Create a workflow from a predefined template."""
        templates = {
            "webhook_to_slack": self._template_webhook_to_slack,
            "scheduled_backup": self._template_scheduled_backup,
            "incident_response": self._template_incident_response,
            "data_pipeline": self._template_data_pipeline,
            "user_onboarding": self._template_user_onboarding,
        }

        template_func = templates.get(template_name)
        if template_func:
            return template_func(name)
        return None

    def _template_webhook_to_slack(self, name: str) -> Workflow:
        """Template: Webhook trigger to Slack notification."""
        workflow = self.create_workflow(
            name=name,
            description="Receives webhook and sends to Slack",
            tags=["notification", "slack", "webhook"]
        )

        # Add webhook trigger
        self.add_trigger(workflow.id, TriggerType.WEBHOOK, {
            "path": f"/webhook/{workflow.id}",
            "method": "POST"
        })

        # Add transform node
        transform = self.add_node(workflow.id, NodeType.TRANSFORM, "Format Message", {
            "operation": "passthrough"
        }, {"x": 100, "y": 100})

        # Add Slack action
        slack = self.add_node(workflow.id, NodeType.ACTION, "Send to Slack", {
            "action_type": "send_webhook",
            "url": "{{ variables.slack_webhook_url }}",
            "body": {
                "text": "{{ trigger_data.message }}"
            }
        }, {"x": 300, "y": 100})

        self.connect_nodes(workflow.id, transform.id, slack.id)

        return self.get_workflow(workflow.id)

    def _template_scheduled_backup(self, name: str) -> Workflow:
        """Template: Scheduled database backup."""
        workflow = self.create_workflow(
            name=name,
            description="Scheduled backup with notification",
            tags=["backup", "scheduled", "database"]
        )

        self.add_trigger(workflow.id, TriggerType.SCHEDULE, {
            "cron": "0 2 * * *"  # 2 AM daily
        })

        backup = self.add_node(workflow.id, NodeType.ACTION, "Run Backup", {
            "action_type": "shell_command",
            "command": "{{ variables.backup_command }}"
        }, {"x": 100, "y": 100})

        notify = self.add_node(workflow.id, NodeType.ACTION, "Notify", {
            "action_type": "send_webhook",
            "url": "{{ variables.notification_url }}",
            "body": {"message": "Backup completed"}
        }, {"x": 300, "y": 100})

        self.connect_nodes(workflow.id, backup.id, notify.id)

        return self.get_workflow(workflow.id)

    def _template_incident_response(self, name: str) -> Workflow:
        """Template: Incident response workflow."""
        workflow = self.create_workflow(
            name=name,
            description="Automated incident response",
            tags=["incident", "security", "automation"]
        )

        self.add_trigger(workflow.id, TriggerType.EVENT, {
            "event_type": "security.alert"
        })

        analyze = self.add_node(workflow.id, NodeType.ACTION, "Analyze Alert", {
            "action_type": "llm_prompt",
            "prompt": "Analyze this security alert: {{ trigger_data }}"
        }, {"x": 100, "y": 100})

        condition = self.add_node(workflow.id, NodeType.CONDITION, "Is Critical?", {
            "condition": "context['node_outputs']['" + analyze.id + "']['severity'] == 'critical'"
        }, {"x": 300, "y": 100})

        escalate = self.add_node(workflow.id, NodeType.ACTION, "Escalate", {
            "action_type": "send_webhook",
            "url": "{{ variables.pagerduty_url }}"
        }, {"x": 500, "y": 50})

        log = self.add_node(workflow.id, NodeType.ACTION, "Log", {
            "action_type": "shell_command",
            "command": "echo '{{ trigger_data }}' >> /var/log/incidents.log"
        }, {"x": 500, "y": 150})

        self.connect_nodes(workflow.id, analyze.id, condition.id)
        self.connect_nodes(workflow.id, condition.id, escalate.id)
        self.connect_nodes(workflow.id, condition.id, log.id)

        return self.get_workflow(workflow.id)

    def _template_data_pipeline(self, name: str) -> Workflow:
        """Template: Data ETL pipeline."""
        workflow = self.create_workflow(
            name=name,
            description="Extract, Transform, Load pipeline",
            tags=["data", "etl", "pipeline"]
        )

        self.add_trigger(workflow.id, TriggerType.SCHEDULE, {
            "cron": "0 */6 * * *"  # Every 6 hours
        })

        extract = self.add_node(workflow.id, NodeType.ACTION, "Extract", {
            "action_type": "http_request",
            "method": "GET",
            "url": "{{ variables.source_api }}"
        }, {"x": 100, "y": 100})

        transform = self.add_node(workflow.id, NodeType.TRANSFORM, "Transform", {
            "operation": "map",
            "expression": "{'id': item['id'], 'value': item['data']}"
        }, {"x": 300, "y": 100})

        load = self.add_node(workflow.id, NodeType.ACTION, "Load", {
            "action_type": "http_request",
            "method": "POST",
            "url": "{{ variables.destination_api }}",
            "body": "{{ node_outputs['" + transform.id + "']['data'] }}"
        }, {"x": 500, "y": 100})

        self.connect_nodes(workflow.id, extract.id, transform.id)
        self.connect_nodes(workflow.id, transform.id, load.id)

        return self.get_workflow(workflow.id)

    def _template_user_onboarding(self, name: str) -> Workflow:
        """Template: User onboarding workflow."""
        workflow = self.create_workflow(
            name=name,
            description="Automated user onboarding",
            tags=["onboarding", "user", "automation"]
        )

        self.add_trigger(workflow.id, TriggerType.WEBHOOK, {
            "path": f"/onboard/{workflow.id}",
            "method": "POST"
        })

        create_account = self.add_node(workflow.id, NodeType.ACTION, "Create Account", {
            "action_type": "http_request",
            "method": "POST",
            "url": "{{ variables.user_api }}/users",
            "body": "{{ trigger_data }}"
        }, {"x": 100, "y": 100})

        send_welcome = self.add_node(workflow.id, NodeType.ACTION, "Send Welcome Email", {
            "action_type": "send_email",
            "to": "{{ trigger_data.email }}",
            "subject": "Welcome!",
            "body": "Welcome to our platform!"
        }, {"x": 300, "y": 50})

        add_to_slack = self.add_node(workflow.id, NodeType.ACTION, "Add to Slack", {
            "action_type": "http_request",
            "method": "POST",
            "url": "{{ variables.slack_api }}/invite",
            "body": {"email": "{{ trigger_data.email }}"}
        }, {"x": 300, "y": 150})

        self.connect_nodes(workflow.id, create_account.id, send_welcome.id)
        self.connect_nodes(workflow.id, create_account.id, add_to_slack.id)

        return self.get_workflow(workflow.id)

    # =========================================================================
    # EXPORT / IMPORT
    # =========================================================================

    def export_workflow(self, workflow_id: str, format: str = "json") -> str:
        """Export a workflow to a string."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return ""

        if format == "json":
            return json.dumps(workflow.to_dict(), indent=2)
        elif format == "n8n":
            return self._export_to_n8n(workflow)

        return ""

    def import_workflow(self, data: str, format: str = "json") -> Optional[Workflow]:
        """Import a workflow from a string."""
        if format == "json":
            workflow_data = json.loads(data)
            workflow_data["id"] = str(uuid.uuid4())  # New ID
            workflow = Workflow.from_dict(workflow_data)
            self.workflows[workflow.id] = workflow
            self._save_workflow(workflow)
            return workflow
        elif format == "n8n":
            return self._import_from_n8n(data)

        return None

    def _export_to_n8n(self, workflow: Workflow) -> str:
        """Export workflow to n8n format."""
        # Convert to n8n JSON format
        n8n_workflow = {
            "name": workflow.name,
            "nodes": [],
            "connections": {},
        }

        for node in workflow.nodes:
            n8n_node = {
                "id": node.id,
                "name": node.name,
                "type": self._map_node_type_to_n8n(node.type),
                "position": [node.position["x"], node.position["y"]],
                "parameters": node.config,
            }
            n8n_workflow["nodes"].append(n8n_node)

            # Build connections
            if node.outputs:
                n8n_workflow["connections"][node.name] = {
                    "main": [[{"node": workflow.get_node(out_id).name, "type": "main", "index": 0}
                             for out_id in node.outputs if workflow.get_node(out_id)]]
                }

        return json.dumps(n8n_workflow, indent=2)

    def _import_from_n8n(self, data: str) -> Optional[Workflow]:
        """Import workflow from n8n format."""
        n8n_data = json.loads(data)

        workflow = self.create_workflow(
            name=n8n_data.get("name", "Imported Workflow"),
            description="Imported from n8n",
            tags=["imported", "n8n"]
        )

        node_name_to_id = {}

        # Create nodes
        for n8n_node in n8n_data.get("nodes", []):
            node = self.add_node(
                workflow.id,
                self._map_n8n_type_to_node(n8n_node.get("type", "")),
                n8n_node.get("name", "Node"),
                n8n_node.get("parameters", {}),
                {"x": n8n_node.get("position", [0, 0])[0],
                 "y": n8n_node.get("position", [0, 0])[1]}
            )
            node_name_to_id[n8n_node.get("name")] = node.id

        # Create connections
        for source_name, connections in n8n_data.get("connections", {}).items():
            source_id = node_name_to_id.get(source_name)
            if source_id:
                for main_outputs in connections.get("main", []):
                    for conn in main_outputs:
                        target_id = node_name_to_id.get(conn.get("node"))
                        if target_id:
                            self.connect_nodes(workflow.id, source_id, target_id)

        return self.get_workflow(workflow.id)

    def _map_node_type_to_n8n(self, node_type: NodeType) -> str:
        """Map internal node type to n8n type."""
        mapping = {
            NodeType.TRIGGER: "n8n-nodes-base.webhook",
            NodeType.ACTION: "n8n-nodes-base.httpRequest",
            NodeType.CONDITION: "n8n-nodes-base.if",
            NodeType.CODE: "n8n-nodes-base.code",
            NodeType.HTTP_REQUEST: "n8n-nodes-base.httpRequest",
        }
        return mapping.get(node_type, "n8n-nodes-base.noOp")

    def _map_n8n_type_to_node(self, n8n_type: str) -> NodeType:
        """Map n8n type to internal node type."""
        if "webhook" in n8n_type.lower():
            return NodeType.TRIGGER
        elif "if" in n8n_type.lower():
            return NodeType.CONDITION
        elif "code" in n8n_type.lower():
            return NodeType.CODE
        elif "http" in n8n_type.lower():
            return NodeType.HTTP_REQUEST
        else:
            return NodeType.ACTION


# Singleton instance
workflow_builder = WorkflowBuilder()
