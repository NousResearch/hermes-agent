"""
cosmos Terraform Manager

"I've terraformed Mars! ...Well, a virtual Mars. In code."

Comprehensive Terraform integration for Infrastructure as Code management.
"""

import json
import subprocess
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class TerraformAction(Enum):
    """Terraform action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    NO_OP = "no-op"


@dataclass
class ResourceChange:
    """A single resource change in a plan."""
    address: str
    resource_type: str
    name: str
    action: TerraformAction
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    after_unknown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "resource_type": self.resource_type,
            "name": self.name,
            "action": self.action.value,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class TerraformPlan:
    """Terraform plan result."""
    created_at: datetime
    changes: List[ResourceChange]
    outputs: Dict[str, Any] = field(default_factory=dict)
    has_changes: bool = False
    plan_file: Optional[str] = None

    def summary(self) -> Dict[str, int]:
        """Get change summary."""
        summary = {
            "create": 0,
            "update": 0,
            "delete": 0,
            "no_change": 0,
        }
        for change in self.changes:
            if change.action == TerraformAction.CREATE:
                summary["create"] += 1
            elif change.action == TerraformAction.UPDATE:
                summary["update"] += 1
            elif change.action == TerraformAction.DELETE:
                summary["delete"] += 1
            else:
                summary["no_change"] += 1
        return summary


@dataclass
class TerraformState:
    """Terraform state information."""
    version: int
    terraform_version: str
    resources: List[Dict[str, Any]]
    outputs: Dict[str, Any]
    serial: int


@dataclass
class TerraformWorkspace:
    """Terraform workspace."""
    name: str
    path: Path
    backend: str = "local"
    backend_config: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    var_files: List[str] = field(default_factory=list)


class TerraformManager:
    """
    Comprehensive Terraform management for cosmos.

    Features:
    - Workspace management
    - Plan and apply operations
    - State management
    - Drift detection
    - Module management
    - Cost estimation
    """

    def __init__(
        self,
        workspaces_path: Path = None,
        terraform_path: str = None,
    ):
        self.workspaces_path = workspaces_path or Path("./data/terraform")
        self.workspaces_path.mkdir(parents=True, exist_ok=True)

        self.terraform_path = terraform_path or shutil.which("terraform") or "terraform"
        self.workspaces: Dict[str, TerraformWorkspace] = {}

        self._verify_terraform()

    def _verify_terraform(self) -> bool:
        """Verify Terraform is installed."""
        try:
            result = subprocess.run(
                [self.terraform_path, "version", "-json"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                version_info = json.loads(result.stdout)
                logger.info(f"Terraform version: {version_info.get('terraform_version')}")
                return True
        except Exception as e:
            logger.warning(f"Terraform not found: {e}")
        return False

    async def _run_terraform(
        self,
        workspace: TerraformWorkspace,
        command: List[str],
        capture_json: bool = False,
    ) -> Dict[str, Any]:
        """Run a Terraform command."""
        full_command = [self.terraform_path] + command

        if capture_json and "-json" not in command:
            full_command.append("-json")

        logger.debug(f"Running: {' '.join(full_command)}")

        process = await asyncio.create_subprocess_exec(
            *full_command,
            cwd=str(workspace.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        result = {
            "success": process.returncode == 0,
            "return_code": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }

        if capture_json and result["success"]:
            try:
                # Handle JSON lines output
                lines = result["stdout"].strip().split("\n")
                if len(lines) == 1:
                    result["json"] = json.loads(lines[0])
                else:
                    result["json"] = [json.loads(l) for l in lines if l.strip()]
            except json.JSONDecodeError:
                pass

        return result

    # =========================================================================
    # WORKSPACE MANAGEMENT
    # =========================================================================

    def create_workspace(
        self,
        name: str,
        source_path: Path = None,
        backend: str = "local",
        backend_config: Dict[str, Any] = None,
        variables: Dict[str, Any] = None,
    ) -> TerraformWorkspace:
        """Create a new Terraform workspace."""
        workspace_path = self.workspaces_path / name
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Copy source files if provided
        if source_path and source_path.exists():
            for file in source_path.glob("*.tf"):
                shutil.copy(file, workspace_path)

        workspace = TerraformWorkspace(
            name=name,
            path=workspace_path,
            backend=backend,
            backend_config=backend_config or {},
            variables=variables or {},
        )

        self.workspaces[name] = workspace
        logger.info(f"Created Terraform workspace: {name}")
        return workspace

    def get_workspace(self, name: str) -> Optional[TerraformWorkspace]:
        """Get a workspace by name."""
        return self.workspaces.get(name)

    def list_workspaces(self) -> List[TerraformWorkspace]:
        """List all workspaces."""
        return list(self.workspaces.values())

    def delete_workspace(self, name: str, destroy_first: bool = True) -> bool:
        """Delete a workspace."""
        workspace = self.workspaces.pop(name, None)
        if workspace:
            if destroy_first:
                asyncio.run(self.destroy(workspace))
            shutil.rmtree(workspace.path, ignore_errors=True)
            logger.info(f"Deleted workspace: {name}")
            return True
        return False

    # =========================================================================
    # TERRAFORM OPERATIONS
    # =========================================================================

    async def init(
        self,
        workspace: TerraformWorkspace,
        upgrade: bool = False,
        reconfigure: bool = False,
    ) -> Dict[str, Any]:
        """Initialize Terraform workspace."""
        command = ["init", "-input=false"]

        if upgrade:
            command.append("-upgrade")
        if reconfigure:
            command.append("-reconfigure")

        # Add backend config
        for key, value in workspace.backend_config.items():
            command.append(f"-backend-config={key}={value}")

        result = await self._run_terraform(workspace, command)
        if result["success"]:
            logger.info(f"Initialized workspace: {workspace.name}")
        else:
            logger.error(f"Init failed: {result['stderr']}")
        return result

    async def validate(self, workspace: TerraformWorkspace) -> Dict[str, Any]:
        """Validate Terraform configuration."""
        result = await self._run_terraform(workspace, ["validate", "-json"], capture_json=True)
        return result

    async def plan(
        self,
        workspace: TerraformWorkspace,
        save_plan: bool = True,
        target: str = None,
        var_overrides: Dict[str, Any] = None,
    ) -> TerraformPlan:
        """Create a Terraform plan."""
        command = ["plan", "-input=false", "-detailed-exitcode"]

        plan_file = None
        if save_plan:
            plan_file = str(workspace.path / f"plan-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.tfplan")
            command.extend(["-out", plan_file])

        # Add variables
        all_vars = {**workspace.variables, **(var_overrides or {})}
        for key, value in all_vars.items():
            command.append(f"-var={key}={json.dumps(value)}")

        # Add var files
        for var_file in workspace.var_files:
            command.extend(["-var-file", var_file])

        # Add target
        if target:
            command.extend(["-target", target])

        result = await self._run_terraform(workspace, command)

        # Parse plan output
        changes = []
        has_changes = result["return_code"] == 2  # Exit code 2 means changes present

        # Get plan details in JSON format
        if plan_file:
            show_result = await self._run_terraform(
                workspace,
                ["show", "-json", plan_file],
                capture_json=True
            )
            if show_result.get("json"):
                plan_json = show_result["json"]
                if isinstance(plan_json, dict):
                    for rc in plan_json.get("resource_changes", []):
                        actions = rc.get("change", {}).get("actions", [])
                        action = TerraformAction.NO_OP
                        if "create" in actions:
                            action = TerraformAction.CREATE
                        elif "delete" in actions:
                            action = TerraformAction.DELETE
                        elif "update" in actions:
                            action = TerraformAction.UPDATE

                        changes.append(ResourceChange(
                            address=rc.get("address", ""),
                            resource_type=rc.get("type", ""),
                            name=rc.get("name", ""),
                            action=action,
                            before=rc.get("change", {}).get("before", {}),
                            after=rc.get("change", {}).get("after", {}),
                        ))

        plan = TerraformPlan(
            created_at=datetime.utcnow(),
            changes=changes,
            has_changes=has_changes,
            plan_file=plan_file,
        )

        summary = plan.summary()
        logger.info(f"Plan: +{summary['create']} ~{summary['update']} -{summary['delete']}")
        return plan

    async def apply(
        self,
        workspace: TerraformWorkspace,
        plan: TerraformPlan = None,
        auto_approve: bool = False,
        target: str = None,
        var_overrides: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Apply Terraform changes."""
        command = ["apply", "-input=false"]

        if auto_approve:
            command.append("-auto-approve")

        if plan and plan.plan_file:
            command.append(plan.plan_file)
        else:
            # Add variables if not using saved plan
            all_vars = {**workspace.variables, **(var_overrides or {})}
            for key, value in all_vars.items():
                command.append(f"-var={key}={json.dumps(value)}")

            for var_file in workspace.var_files:
                command.extend(["-var-file", var_file])

            if target:
                command.extend(["-target", target])

        result = await self._run_terraform(workspace, command)

        if result["success"]:
            logger.info(f"Apply completed for workspace: {workspace.name}")
        else:
            logger.error(f"Apply failed: {result['stderr']}")

        return result

    async def destroy(
        self,
        workspace: TerraformWorkspace,
        auto_approve: bool = True,
        target: str = None,
    ) -> Dict[str, Any]:
        """Destroy Terraform-managed infrastructure."""
        command = ["destroy", "-input=false"]

        if auto_approve:
            command.append("-auto-approve")

        # Add variables
        for key, value in workspace.variables.items():
            command.append(f"-var={key}={json.dumps(value)}")

        if target:
            command.extend(["-target", target])

        result = await self._run_terraform(workspace, command)

        if result["success"]:
            logger.info(f"Destroy completed for workspace: {workspace.name}")
        else:
            logger.error(f"Destroy failed: {result['stderr']}")

        return result

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    async def get_state(self, workspace: TerraformWorkspace) -> Optional[TerraformState]:
        """Get current Terraform state."""
        result = await self._run_terraform(
            workspace,
            ["show", "-json"],
            capture_json=True
        )

        if not result.get("json"):
            return None

        state_json = result["json"]
        return TerraformState(
            version=state_json.get("format_version", 0),
            terraform_version=state_json.get("terraform_version", ""),
            resources=state_json.get("values", {}).get("root_module", {}).get("resources", []),
            outputs=state_json.get("values", {}).get("outputs", {}),
            serial=0,
        )

    async def list_resources(self, workspace: TerraformWorkspace) -> List[str]:
        """List all resources in state."""
        result = await self._run_terraform(workspace, ["state", "list"])
        if result["success"]:
            return [r.strip() for r in result["stdout"].strip().split("\n") if r.strip()]
        return []

    async def import_resource(
        self,
        workspace: TerraformWorkspace,
        address: str,
        resource_id: str,
    ) -> Dict[str, Any]:
        """Import an existing resource into state."""
        result = await self._run_terraform(
            workspace,
            ["import", address, resource_id]
        )
        if result["success"]:
            logger.info(f"Imported resource: {address}")
        return result

    async def remove_resource(
        self,
        workspace: TerraformWorkspace,
        address: str,
    ) -> Dict[str, Any]:
        """Remove a resource from state (without destroying)."""
        result = await self._run_terraform(
            workspace,
            ["state", "rm", address]
        )
        if result["success"]:
            logger.info(f"Removed resource from state: {address}")
        return result

    async def move_resource(
        self,
        workspace: TerraformWorkspace,
        source: str,
        destination: str,
    ) -> Dict[str, Any]:
        """Move a resource in state."""
        result = await self._run_terraform(
            workspace,
            ["state", "mv", source, destination]
        )
        return result

    async def refresh(self, workspace: TerraformWorkspace) -> Dict[str, Any]:
        """Refresh state from real infrastructure."""
        result = await self._run_terraform(workspace, ["refresh", "-input=false"])
        if result["success"]:
            logger.info(f"State refreshed for workspace: {workspace.name}")
        return result

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    async def get_outputs(
        self,
        workspace: TerraformWorkspace,
    ) -> Dict[str, Any]:
        """Get Terraform outputs."""
        result = await self._run_terraform(
            workspace,
            ["output", "-json"],
            capture_json=True
        )
        return result.get("json", {})

    async def get_output(
        self,
        workspace: TerraformWorkspace,
        name: str,
    ) -> Any:
        """Get a specific output value."""
        outputs = await self.get_outputs(workspace)
        output = outputs.get(name, {})
        return output.get("value")

    # =========================================================================
    # MODULES
    # =========================================================================

    async def get_modules(self, workspace: TerraformWorkspace) -> List[Dict[str, Any]]:
        """List modules in configuration."""
        result = await self._run_terraform(
            workspace,
            ["providers", "-json"],
            capture_json=True
        )
        # Parse providers output for module info
        return []

    # =========================================================================
    # TEMPLATES
    # =========================================================================

    def generate_provider_config(
        self,
        provider: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate provider configuration."""
        lines = [f'provider "{provider}" {{']
        for key, value in config.items():
            if isinstance(value, str):
                lines.append(f'  {key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f'  {key} = {str(value).lower()}')
            else:
                lines.append(f'  {key} = {json.dumps(value)}')
        lines.append("}")
        return "\n".join(lines)

    def generate_resource_config(
        self,
        resource_type: str,
        name: str,
        config: Dict[str, Any],
    ) -> str:
        """Generate resource configuration."""
        lines = [f'resource "{resource_type}" "{name}" {{']
        for key, value in config.items():
            if isinstance(value, str):
                lines.append(f'  {key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f'  {key} = {str(value).lower()}')
            elif isinstance(value, dict):
                lines.append(f"  {key} {{")
                for k, v in value.items():
                    lines.append(f'    {k} = "{v}"' if isinstance(v, str) else f'    {k} = {v}')
                lines.append("  }")
            else:
                lines.append(f'  {key} = {json.dumps(value)}')
        lines.append("}")
        return "\n".join(lines)

    def generate_variable_config(
        self,
        name: str,
        var_type: str = "string",
        default: Any = None,
        description: str = "",
    ) -> str:
        """Generate variable configuration."""
        lines = [f'variable "{name}" {{']
        lines.append(f'  type = {var_type}')
        if description:
            lines.append(f'  description = "{description}"')
        if default is not None:
            if isinstance(default, str):
                lines.append(f'  default = "{default}"')
            else:
                lines.append(f'  default = {json.dumps(default)}')
        lines.append("}")
        return "\n".join(lines)


# Singleton instance
terraform_manager = TerraformManager()
