"""
cosmos Runbook Executor

"Just follow the instructions... what could possibly go wrong?"

Automated runbook execution for incident response.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import yaml
from loguru import logger


class StepType(Enum):
    """Types of runbook steps."""
    MANUAL = "manual"  # Requires human action
    COMMAND = "command"  # Shell command
    SCRIPT = "script"  # Script execution
    API_CALL = "api_call"  # HTTP API call
    NOTIFICATION = "notification"  # Send notification
    APPROVAL = "approval"  # Wait for approval
    CONDITIONAL = "conditional"  # Branch based on condition
    PARALLEL = "parallel"  # Execute steps in parallel
    WAIT = "wait"  # Wait for duration or condition
    ROLLBACK = "rollback"  # Trigger rollback


class StepStatus(Enum):
    """Execution status of a step."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class RunbookStep:
    """A single step in a runbook."""
    id: str
    name: str
    description: str
    step_type: StepType
    order: int

    # Execution details
    command: str = ""
    script: str = ""
    api_endpoint: str = ""
    api_method: str = "GET"
    api_body: Dict = field(default_factory=dict)
    notification_channel: str = ""
    notification_message: str = ""

    # Control flow
    condition: str = ""  # Condition to evaluate
    timeout_seconds: int = 300
    retry_count: int = 0
    retry_delay_seconds: int = 30
    continue_on_failure: bool = False
    rollback_step_id: Optional[str] = None

    # Parallel execution
    parallel_steps: List[str] = field(default_factory=list)

    # Wait configuration
    wait_duration_seconds: int = 0
    wait_condition: str = ""

    # Approvers for approval steps
    required_approvers: List[str] = field(default_factory=list)
    minimum_approvals: int = 1

    # Execution state
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: str = ""
    error_message: str = ""
    approved_by: List[str] = field(default_factory=list)


@dataclass
class Runbook:
    """A runbook definition."""
    id: str
    name: str
    description: str
    version: str
    steps: List[RunbookStep] = field(default_factory=list)

    # Metadata
    owner: str = ""
    teams: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    incident_types: List[str] = field(default_factory=list)  # Auto-trigger for incident types
    severity_threshold: str = ""  # Minimum severity to auto-trigger

    # Settings
    auto_execute: bool = False  # Auto-start on incident creation
    require_confirmation: bool = True
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    max_execution_time: int = 3600  # Maximum total execution time

    # Audit
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    last_executed: Optional[datetime] = None
    execution_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "owner": self.owner,
            "teams": self.teams,
            "tags": self.tags,
            "step_count": len(self.steps),
            "auto_execute": self.auto_execute,
            "created_at": self.created_at.isoformat(),
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "execution_count": self.execution_count,
        }

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "Runbook":
        """Create a runbook from YAML definition."""
        data = yaml.safe_load(yaml_content)

        steps = []
        for i, step_data in enumerate(data.get("steps", [])):
            step = RunbookStep(
                id=step_data.get("id", f"step-{i+1}"),
                name=step_data["name"],
                description=step_data.get("description", ""),
                step_type=StepType(step_data.get("type", "manual")),
                order=i + 1,
                command=step_data.get("command", ""),
                script=step_data.get("script", ""),
                api_endpoint=step_data.get("api_endpoint", ""),
                api_method=step_data.get("api_method", "GET"),
                api_body=step_data.get("api_body", {}),
                notification_channel=step_data.get("notification_channel", ""),
                notification_message=step_data.get("notification_message", ""),
                condition=step_data.get("condition", ""),
                timeout_seconds=step_data.get("timeout", 300),
                retry_count=step_data.get("retries", 0),
                continue_on_failure=step_data.get("continue_on_failure", False),
                rollback_step_id=step_data.get("rollback_step"),
                parallel_steps=step_data.get("parallel_steps", []),
                wait_duration_seconds=step_data.get("wait_duration", 0),
                required_approvers=step_data.get("approvers", []),
                minimum_approvals=step_data.get("min_approvals", 1),
            )
            steps.append(step)

        return cls(
            id=data.get("id", ""),
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            steps=steps,
            owner=data.get("owner", ""),
            teams=data.get("teams", []),
            tags=data.get("tags", []),
            incident_types=data.get("incident_types", []),
            severity_threshold=data.get("severity_threshold", ""),
            auto_execute=data.get("auto_execute", False),
            require_confirmation=data.get("require_confirmation", True),
            max_execution_time=data.get("max_execution_time", 3600),
        )


@dataclass
class RunbookExecution:
    """An execution instance of a runbook."""
    id: str
    runbook_id: str
    runbook_name: str
    incident_id: Optional[str] = None

    # Execution state
    status: str = "pending"  # pending, running, paused, completed, failed, cancelled
    current_step: int = 0
    steps: List[RunbookStep] = field(default_factory=list)

    # Context
    variables: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None

    # Metadata
    triggered_by: str = ""
    trigger_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "runbook_id": self.runbook_id,
            "runbook_name": self.runbook_name,
            "incident_id": self.incident_id,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "triggered_by": self.triggered_by,
        }


class RunbookExecutor:
    """
    Automated runbook execution engine.

    Features:
    - Step-by-step execution
    - Approval workflows
    - Parallel step execution
    - Rollback support
    - Variable interpolation
    - Execution history
    """

    def __init__(
        self,
        storage_path: Path = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/runbooks")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.runbooks: Dict[str, Runbook] = {}
        self.executions: Dict[str, RunbookExecution] = {}
        self.step_handlers: Dict[StepType, Callable] = {}
        self.notification_handler: Optional[Callable] = None

        self._register_default_handlers()
        self._load_runbooks()

    def _register_default_handlers(self):
        """Register default step handlers."""
        self.step_handlers[StepType.COMMAND] = self._execute_command
        self.step_handlers[StepType.SCRIPT] = self._execute_script
        self.step_handlers[StepType.API_CALL] = self._execute_api_call
        self.step_handlers[StepType.NOTIFICATION] = self._execute_notification
        self.step_handlers[StepType.WAIT] = self._execute_wait
        self.step_handlers[StepType.MANUAL] = self._execute_manual

    def _load_runbooks(self):
        """Load runbooks from storage."""
        runbooks_dir = self.storage_path / "definitions"
        if not runbooks_dir.exists():
            runbooks_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_runbooks(runbooks_dir)
            return

        for runbook_file in runbooks_dir.glob("*.yaml"):
            try:
                with open(runbook_file) as f:
                    runbook = Runbook.from_yaml(f.read())
                    if not runbook.id:
                        runbook.id = runbook_file.stem
                    self.runbooks[runbook.id] = runbook
                    logger.debug(f"Loaded runbook: {runbook.name}")
            except Exception as e:
                logger.error(f"Failed to load runbook {runbook_file}: {e}")

    def _create_default_runbooks(self, runbooks_dir: Path):
        """Create default runbook templates."""
        # Service outage runbook
        outage_runbook = """
id: service-outage
name: Service Outage Response
description: Standard procedure for handling service outages
version: "1.0"
owner: platform-team
teams: [sre, platform]
tags: [outage, critical]
incident_types: [outage, degradation]
severity_threshold: sev2
auto_execute: false
require_confirmation: true

steps:
  - id: assess
    name: Assess Impact
    description: Determine the scope and impact of the outage
    type: manual

  - id: notify-stakeholders
    name: Notify Stakeholders
    description: Send initial notification to stakeholders
    type: notification
    notification_channel: "#incidents"
    notification_message: "Service outage detected. Investigation in progress."

  - id: check-metrics
    name: Check System Metrics
    description: Review monitoring dashboards and metrics
    type: command
    command: "curl -s http://localhost:9090/api/v1/query?query=up | jq '.data.result'"
    timeout: 60
    continue_on_failure: true

  - id: check-logs
    name: Review Recent Logs
    description: Check application and system logs for errors
    type: command
    command: "tail -n 100 /var/log/app/error.log 2>/dev/null || echo 'Log file not found'"
    timeout: 60

  - id: identify-cause
    name: Identify Root Cause
    description: Determine the root cause of the outage
    type: manual

  - id: apply-fix
    name: Apply Fix
    description: Implement the fix for the issue
    type: approval
    approvers: [incident-commander]
    min_approvals: 1

  - id: verify-recovery
    name: Verify Recovery
    description: Confirm the service has recovered
    type: command
    command: "curl -sf http://localhost:8080/health && echo 'Service healthy'"
    timeout: 120
    retries: 3

  - id: notify-resolved
    name: Notify Resolution
    description: Inform stakeholders the incident is resolved
    type: notification
    notification_channel: "#incidents"
    notification_message: "Service outage has been resolved. Normal operations resumed."
"""

        # Deployment rollback runbook
        rollback_runbook = """
id: deployment-rollback
name: Deployment Rollback
description: Rollback to previous deployment version
version: "1.0"
owner: platform-team
teams: [sre, devops]
tags: [deployment, rollback]

steps:
  - id: confirm-rollback
    name: Confirm Rollback
    description: Verify rollback is needed
    type: approval
    approvers: [tech-lead, sre-lead]
    min_approvals: 1

  - id: stop-traffic
    name: Stop Incoming Traffic
    description: Divert traffic away from affected service
    type: command
    command: "kubectl scale deployment app --replicas=0"
    rollback_step: restore-traffic

  - id: rollback-deployment
    name: Rollback Deployment
    description: Revert to previous version
    type: command
    command: "kubectl rollout undo deployment/app"
    timeout: 300

  - id: wait-rollback
    name: Wait for Rollback
    description: Wait for pods to be ready
    type: wait
    wait_duration: 30

  - id: restore-traffic
    name: Restore Traffic
    description: Restore normal traffic routing
    type: command
    command: "kubectl scale deployment app --replicas=3"

  - id: verify-health
    name: Verify Service Health
    description: Check service is responding correctly
    type: command
    command: "kubectl rollout status deployment/app && curl -sf http://app-service/health"
    timeout: 180
    retries: 5
"""

        # Database incident runbook
        db_runbook = """
id: database-incident
name: Database Incident Response
description: Handle database-related incidents
version: "1.0"
owner: dba-team
teams: [dba, sre]
tags: [database, data]

steps:
  - id: check-connections
    name: Check Active Connections
    description: Review current database connections
    type: command
    command: "psql -c 'SELECT count(*) FROM pg_stat_activity;'"
    timeout: 30

  - id: check-replication
    name: Check Replication Status
    description: Verify replication is healthy
    type: command
    command: "psql -c 'SELECT * FROM pg_stat_replication;'"
    timeout: 30
    continue_on_failure: true

  - id: check-locks
    name: Check for Blocking Locks
    description: Identify any blocking queries
    type: command
    command: "psql -c 'SELECT * FROM pg_locks WHERE NOT granted;'"
    timeout: 30

  - id: assess-situation
    name: Assess Situation
    description: Review findings and determine next steps
    type: manual

  - id: notify-team
    name: Notify DBA Team
    description: Alert the DBA team
    type: notification
    notification_channel: "#dba-alerts"
    notification_message: "Database incident requires attention"
"""

        with open(runbooks_dir / "service-outage.yaml", "w") as f:
            f.write(outage_runbook)
        with open(runbooks_dir / "deployment-rollback.yaml", "w") as f:
            f.write(rollback_runbook)
        with open(runbooks_dir / "database-incident.yaml", "w") as f:
            f.write(db_runbook)

        # Reload
        self._load_runbooks()

    # =========================================================================
    # RUNBOOK CRUD
    # =========================================================================

    def create_runbook(
        self,
        name: str,
        description: str,
        steps: List[RunbookStep],
        **kwargs,
    ) -> Runbook:
        """Create a new runbook."""
        import uuid

        runbook = Runbook(
            id=kwargs.get("id", str(uuid.uuid4())[:8]),
            name=name,
            description=description,
            version=kwargs.get("version", "1.0"),
            steps=steps,
            owner=kwargs.get("owner", ""),
            teams=kwargs.get("teams", []),
            tags=kwargs.get("tags", []),
            created_by=kwargs.get("created_by", ""),
        )

        self.runbooks[runbook.id] = runbook
        self._save_runbook(runbook)

        logger.info(f"Created runbook: {runbook.name}")
        return runbook

    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get a runbook by ID."""
        return self.runbooks.get(runbook_id)

    def list_runbooks(
        self,
        tags: List[str] = None,
        incident_type: str = None,
    ) -> List[Runbook]:
        """List runbooks with optional filters."""
        runbooks = list(self.runbooks.values())

        if tags:
            runbooks = [r for r in runbooks if any(t in r.tags for t in tags)]
        if incident_type:
            runbooks = [r for r in runbooks if incident_type in r.incident_types]

        return runbooks

    def _save_runbook(self, runbook: Runbook):
        """Save a runbook to storage."""
        runbooks_dir = self.storage_path / "definitions"
        runbooks_dir.mkdir(parents=True, exist_ok=True)

        # Convert to YAML format
        data = {
            "id": runbook.id,
            "name": runbook.name,
            "description": runbook.description,
            "version": runbook.version,
            "owner": runbook.owner,
            "teams": runbook.teams,
            "tags": runbook.tags,
            "incident_types": runbook.incident_types,
            "auto_execute": runbook.auto_execute,
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "type": s.step_type.value,
                    "command": s.command,
                    "timeout": s.timeout_seconds,
                    "retries": s.retry_count,
                    "continue_on_failure": s.continue_on_failure,
                }
                for s in runbook.steps
            ],
        }

        with open(runbooks_dir / f"{runbook.id}.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def execute(
        self,
        runbook_id: str,
        incident_id: str = None,
        triggered_by: str = "system",
        variables: Dict[str, Any] = None,
    ) -> Optional[RunbookExecution]:
        """Execute a runbook."""
        runbook = self.runbooks.get(runbook_id)
        if not runbook:
            logger.error(f"Runbook not found: {runbook_id}")
            return None

        import uuid
        execution = RunbookExecution(
            id=str(uuid.uuid4())[:8],
            runbook_id=runbook_id,
            runbook_name=runbook.name,
            incident_id=incident_id,
            steps=[self._copy_step(s) for s in runbook.steps],
            variables=variables or {},
            triggered_by=triggered_by,
            started_at=datetime.utcnow(),
            status="running",
        )

        self.executions[execution.id] = execution

        # Update runbook stats
        runbook.last_executed = datetime.utcnow()
        runbook.execution_count += 1

        logger.info(f"Starting runbook execution: {runbook.name} ({execution.id})")

        # Execute steps
        try:
            await self._execute_steps(execution)
        except Exception as e:
            logger.error(f"Runbook execution failed: {e}")
            execution.status = "failed"

        return execution

    def _copy_step(self, step: RunbookStep) -> RunbookStep:
        """Create a copy of a step for execution."""
        return RunbookStep(
            id=step.id,
            name=step.name,
            description=step.description,
            step_type=step.step_type,
            order=step.order,
            command=step.command,
            script=step.script,
            api_endpoint=step.api_endpoint,
            api_method=step.api_method,
            api_body=step.api_body.copy(),
            notification_channel=step.notification_channel,
            notification_message=step.notification_message,
            condition=step.condition,
            timeout_seconds=step.timeout_seconds,
            retry_count=step.retry_count,
            retry_delay_seconds=step.retry_delay_seconds,
            continue_on_failure=step.continue_on_failure,
            rollback_step_id=step.rollback_step_id,
            parallel_steps=step.parallel_steps.copy(),
            wait_duration_seconds=step.wait_duration_seconds,
            required_approvers=step.required_approvers.copy(),
            minimum_approvals=step.minimum_approvals,
        )

    async def _execute_steps(self, execution: RunbookExecution):
        """Execute all steps in a runbook."""
        for i, step in enumerate(execution.steps):
            execution.current_step = i + 1

            # Check for pause
            if execution.status == "paused":
                logger.info(f"Execution paused at step {i + 1}")
                return

            # Skip if condition not met
            if step.condition and not self._evaluate_condition(step.condition, execution):
                step.status = StepStatus.SKIPPED
                logger.info(f"Skipped step {step.name}: condition not met")
                continue

            # Execute step
            success = await self._execute_step(step, execution)

            if not success and not step.continue_on_failure:
                execution.status = "failed"

                # Trigger rollback if configured
                if step.rollback_step_id:
                    await self._execute_rollback(step.rollback_step_id, execution)

                return

        execution.status = "completed"
        execution.completed_at = datetime.utcnow()
        logger.info(f"Runbook execution completed: {execution.id}")

    async def _execute_step(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Execute a single step."""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()

        logger.info(f"Executing step: {step.name}")

        handler = self.step_handlers.get(step.step_type)
        if not handler:
            logger.error(f"No handler for step type: {step.step_type}")
            step.status = StepStatus.FAILED
            step.error_message = f"No handler for step type: {step.step_type}"
            return False

        # Retry logic
        attempts = 0
        max_attempts = step.retry_count + 1

        while attempts < max_attempts:
            try:
                result = await asyncio.wait_for(
                    handler(step, execution),
                    timeout=step.timeout_seconds,
                )

                if result:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.utcnow()
                    return True

            except asyncio.TimeoutError:
                step.error_message = f"Step timed out after {step.timeout_seconds}s"
            except Exception as e:
                step.error_message = str(e)

            attempts += 1
            if attempts < max_attempts:
                logger.warning(f"Step failed, retrying ({attempts}/{max_attempts})")
                await asyncio.sleep(step.retry_delay_seconds)

        step.status = StepStatus.FAILED
        step.completed_at = datetime.utcnow()
        return False

    async def _execute_rollback(
        self,
        rollback_step_id: str,
        execution: RunbookExecution,
    ):
        """Execute rollback step."""
        for step in execution.steps:
            if step.id == rollback_step_id:
                logger.warning(f"Executing rollback step: {step.name}")
                await self._execute_step(step, execution)
                return

    def _evaluate_condition(
        self,
        condition: str,
        execution: RunbookExecution,
    ) -> bool:
        """Evaluate a step condition."""
        # Simple variable substitution and evaluation
        try:
            for key, value in execution.variables.items():
                condition = condition.replace(f"${{{key}}}", str(value))
            for key, value in execution.outputs.items():
                condition = condition.replace(f"${{outputs.{key}}}", str(value))

            return eval(condition)  # In production, use a safe expression evaluator
        except Exception as e:
            logger.error(f"Failed to evaluate condition: {e}")
            return False

    # =========================================================================
    # STEP HANDLERS
    # =========================================================================

    async def _execute_command(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Execute a shell command."""
        command = self._interpolate(step.command, execution)

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        step.output = stdout.decode() + stderr.decode()
        execution.outputs[step.id] = step.output

        return process.returncode == 0

    async def _execute_script(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Execute a script."""
        script = self._interpolate(step.script, execution)

        # Write script to temp file and execute
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                "bash", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            step.output = stdout.decode() + stderr.decode()
            return process.returncode == 0
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _execute_api_call(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Execute an API call."""
        try:
            import httpx

            url = self._interpolate(step.api_endpoint, execution)
            body = json.loads(self._interpolate(json.dumps(step.api_body), execution))

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    step.api_method,
                    url,
                    json=body if body else None,
                    timeout=step.timeout_seconds,
                )

                step.output = response.text
                execution.outputs[step.id] = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text

                return response.is_success

        except Exception as e:
            step.error_message = str(e)
            return False

    async def _execute_notification(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Send a notification."""
        message = self._interpolate(step.notification_message, execution)

        if self.notification_handler:
            try:
                await self.notification_handler(step.notification_channel, message)
                return True
            except Exception as e:
                step.error_message = str(e)
                return False

        # Default: just log
        logger.info(f"[{step.notification_channel}] {message}")
        return True

    async def _execute_wait(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Wait for duration or condition."""
        if step.wait_duration_seconds:
            await asyncio.sleep(step.wait_duration_seconds)
            return True

        if step.wait_condition:
            # Poll condition until true or timeout
            start = datetime.utcnow()
            while (datetime.utcnow() - start).total_seconds() < step.timeout_seconds:
                if self._evaluate_condition(step.wait_condition, execution):
                    return True
                await asyncio.sleep(5)
            return False

        return True

    async def _execute_manual(
        self,
        step: RunbookStep,
        execution: RunbookExecution,
    ) -> bool:
        """Mark step as requiring manual action."""
        step.status = StepStatus.WAITING_APPROVAL
        logger.info(f"Manual step required: {step.name}")
        # In a real implementation, this would pause and wait for user input
        return True

    def _interpolate(self, text: str, execution: RunbookExecution) -> str:
        """Interpolate variables in text."""
        for key, value in execution.variables.items():
            text = text.replace(f"${{{key}}}", str(value))
        for key, value in execution.outputs.items():
            text = text.replace(f"${{outputs.{key}}}", str(value))
        return text

    # =========================================================================
    # EXECUTION CONTROL
    # =========================================================================

    def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == "running":
            execution.status = "paused"
            execution.paused_at = datetime.utcnow()
            logger.info(f"Paused execution: {execution_id}")
            return True
        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status == "paused":
            execution.status = "running"
            logger.info(f"Resuming execution: {execution_id}")
            await self._execute_steps(execution)
            return True
        return False

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an execution."""
        execution = self.executions.get(execution_id)
        if execution and execution.status in ["running", "paused"]:
            execution.status = "cancelled"
            execution.completed_at = datetime.utcnow()
            logger.info(f"Cancelled execution: {execution_id}")
            return True
        return False

    def approve_step(
        self,
        execution_id: str,
        step_id: str,
        approver: str,
    ) -> bool:
        """Approve an approval step."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False

        for step in execution.steps:
            if step.id == step_id and step.status == StepStatus.WAITING_APPROVAL:
                if approver not in step.approved_by:
                    step.approved_by.append(approver)

                if len(step.approved_by) >= step.minimum_approvals:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.utcnow()
                    logger.info(f"Step {step_id} approved by {approver}")
                    return True

        return False

    # =========================================================================
    # HANDLERS
    # =========================================================================

    def set_notification_handler(self, handler: Callable):
        """Set the notification handler."""
        self.notification_handler = handler

    def register_step_handler(self, step_type: StepType, handler: Callable):
        """Register a custom step handler."""
        self.step_handlers[step_type] = handler

    # =========================================================================
    # EXECUTION HISTORY
    # =========================================================================

    def get_execution(self, execution_id: str) -> Optional[RunbookExecution]:
        """Get an execution by ID."""
        return self.executions.get(execution_id)

    def list_executions(
        self,
        runbook_id: str = None,
        incident_id: str = None,
        status: str = None,
        limit: int = 50,
    ) -> List[RunbookExecution]:
        """List executions with optional filters."""
        executions = list(self.executions.values())

        if runbook_id:
            executions = [e for e in executions if e.runbook_id == runbook_id]
        if incident_id:
            executions = [e for e in executions if e.incident_id == incident_id]
        if status:
            executions = [e for e in executions if e.status == status]

        executions.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        return executions[:limit]


# Singleton instance
runbook_executor = RunbookExecutor()
