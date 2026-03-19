"""
cosmos Enhanced n8n Integration

"I've enhanced n8n with my patented cosmos improvements!"

Full n8n API integration for workflow creation, management, and execution.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from loguru import logger


class N8nNodeType(Enum):
    """Common n8n node types."""
    # Triggers
    WEBHOOK = "n8n-nodes-base.webhook"
    CRON = "n8n-nodes-base.cron"
    MANUAL = "n8n-nodes-base.manualTrigger"
    EMAIL_TRIGGER = "n8n-nodes-base.emailTrigger"

    # Core
    HTTP_REQUEST = "n8n-nodes-base.httpRequest"
    CODE = "n8n-nodes-base.code"
    FUNCTION = "n8n-nodes-base.function"
    SET = "n8n-nodes-base.set"
    IF = "n8n-nodes-base.if"
    SWITCH = "n8n-nodes-base.switch"
    MERGE = "n8n-nodes-base.merge"
    SPLIT_IN_BATCHES = "n8n-nodes-base.splitInBatches"
    WAIT = "n8n-nodes-base.wait"
    NO_OP = "n8n-nodes-base.noOp"

    # Data
    SPREADSHEET = "n8n-nodes-base.spreadsheetFile"
    XML = "n8n-nodes-base.xml"
    HTML = "n8n-nodes-base.html"
    MARKDOWN = "n8n-nodes-base.markdown"
    CRYPTO = "n8n-nodes-base.crypto"
    COMPRESSION = "n8n-nodes-base.compression"

    # Integrations
    SLACK = "n8n-nodes-base.slack"
    DISCORD = "n8n-nodes-base.discord"
    TELEGRAM = "n8n-nodes-base.telegram"
    EMAIL = "n8n-nodes-base.emailSend"
    GITHUB = "n8n-nodes-base.github"
    GITLAB = "n8n-nodes-base.gitlab"
    JIRA = "n8n-nodes-base.jira"
    AWS_S3 = "n8n-nodes-base.awsS3"
    AWS_LAMBDA = "n8n-nodes-base.awsLambda"
    GOOGLE_SHEETS = "n8n-nodes-base.googleSheets"
    GOOGLE_DRIVE = "n8n-nodes-base.googleDrive"
    MYSQL = "n8n-nodes-base.mySql"
    POSTGRES = "n8n-nodes-base.postgres"
    MONGODB = "n8n-nodes-base.mongoDb"
    REDIS = "n8n-nodes-base.redis"
    OPENAI = "n8n-nodes-base.openAi"
    SSH = "n8n-nodes-base.ssh"
    FTP = "n8n-nodes-base.ftp"


@dataclass
class N8nNode:
    """n8n workflow node."""
    id: str
    name: str
    type: str
    position: List[int] = field(default_factory=lambda: [250, 300])
    parameters: Dict[str, Any] = field(default_factory=dict)
    credentials: Dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    notes: str = ""
    retry_on_fail: bool = False
    max_tries: int = 3
    wait_between_tries: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "position": self.position,
            "parameters": self.parameters,
            "credentials": self.credentials if self.credentials else {},
            "disabled": self.disabled,
            "notes": self.notes,
            "retryOnFail": self.retry_on_fail,
            "maxTries": self.max_tries,
            "waitBetweenTries": self.wait_between_tries,
            "typeVersion": 1,
        }


@dataclass
class N8nConnection:
    """Connection between n8n nodes."""
    from_node: str
    to_node: str
    from_output: int = 0
    to_input: int = 0


@dataclass
class N8nWorkflow:
    """n8n workflow definition."""
    name: str
    nodes: List[N8nNode] = field(default_factory=list)
    connections: List[N8nConnection] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "executionOrder": "v1",
        "saveManualExecutions": True,
        "callerPolicy": "workflowsFromSameOwner",
    })
    static_data: Dict[str, Any] = field(default_factory=dict)
    active: bool = False

    def add_node(self, node: N8nNode):
        """Add a node to the workflow."""
        self.nodes.append(node)

    def connect(
        self,
        from_node_name: str,
        to_node_name: str,
        from_output: int = 0,
        to_input: int = 0
    ):
        """Connect two nodes by name."""
        self.connections.append(N8nConnection(
            from_node=from_node_name,
            to_node=to_node_name,
            from_output=from_output,
            to_input=to_input,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to n8n API format."""
        # Build connections dict in n8n format
        conn_dict = {}
        for conn in self.connections:
            if conn.from_node not in conn_dict:
                conn_dict[conn.from_node] = {"main": [[]]}

            # Ensure enough output arrays
            while len(conn_dict[conn.from_node]["main"]) <= conn.from_output:
                conn_dict[conn.from_node]["main"].append([])

            conn_dict[conn.from_node]["main"][conn.from_output].append({
                "node": conn.to_node,
                "type": "main",
                "index": conn.to_input,
            })

        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": conn_dict,
            "settings": self.settings,
            "staticData": self.static_data,
            "tags": self.tags,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "N8nWorkflow":
        """Create workflow from n8n API response."""
        workflow = cls(
            name=data.get("name", "Unnamed"),
            tags=data.get("tags", []),
            settings=data.get("settings", {}),
            static_data=data.get("staticData", {}),
            active=data.get("active", False),
        )

        # Parse nodes
        for node_data in data.get("nodes", []):
            node = N8nNode(
                id=node_data.get("id", ""),
                name=node_data.get("name", ""),
                type=node_data.get("type", ""),
                position=node_data.get("position", [250, 300]),
                parameters=node_data.get("parameters", {}),
                credentials=node_data.get("credentials", {}),
                disabled=node_data.get("disabled", False),
            )
            workflow.nodes.append(node)

        # Parse connections
        for from_node, outputs in data.get("connections", {}).items():
            for output_idx, connections in enumerate(outputs.get("main", [])):
                for conn in connections:
                    workflow.connections.append(N8nConnection(
                        from_node=from_node,
                        to_node=conn.get("node", ""),
                        from_output=output_idx,
                        to_input=conn.get("index", 0),
                    ))

        return workflow


class EnhancedN8nIntegration:
    """
    Enhanced n8n integration for cosmos.

    Features:
    - Full CRUD for workflows
    - Credential management
    - Execution monitoring
    - Workflow templates
    - Auto-deployment
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
    ):
        import os
        self.base_url = base_url or os.getenv("N8N_BASE_URL", "http://localhost:5678")
        self.api_key = api_key or os.getenv("N8N_API_KEY")
        self.base_url = self.base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["X-N8N-API-KEY"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Any = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        """Make API request."""
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/{endpoint.lstrip('/')}"

        try:
            async with session.request(
                method,
                url,
                json=data,
                params=params,
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"n8n API error: {response.status} - {error_text}")
                    return {"error": error_text, "status": response.status}
                return await response.json()
        except Exception as e:
            logger.error(f"n8n request failed: {e}")
            return {"error": str(e)}

    # =========================================================================
    # WORKFLOW MANAGEMENT
    # =========================================================================

    async def list_workflows(
        self,
        active: Optional[bool] = None,
        tags: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all workflows."""
        params = {}
        if active is not None:
            params["active"] = str(active).lower()
        if tags:
            params["tags"] = ",".join(tags)

        result = await self._request("GET", "/workflows", params=params)
        return result.get("data", [])

    async def get_workflow(self, workflow_id: str) -> Optional[N8nWorkflow]:
        """Get a workflow by ID."""
        result = await self._request("GET", f"/workflows/{workflow_id}")
        if "error" in result:
            return None
        return N8nWorkflow.from_dict(result)

    async def create_workflow(self, workflow: N8nWorkflow) -> Optional[str]:
        """Create a new workflow. Returns workflow ID."""
        result = await self._request("POST", "/workflows", data=workflow.to_dict())
        if "error" in result:
            logger.error(f"Failed to create workflow: {result['error']}")
            return None
        workflow_id = result.get("id")
        logger.info(f"Created n8n workflow: {workflow.name} (ID: {workflow_id})")
        return workflow_id

    async def update_workflow(
        self,
        workflow_id: str,
        workflow: N8nWorkflow,
    ) -> bool:
        """Update an existing workflow."""
        result = await self._request(
            "PATCH",
            f"/workflows/{workflow_id}",
            data=workflow.to_dict()
        )
        if "error" in result:
            logger.error(f"Failed to update workflow: {result['error']}")
            return False
        logger.info(f"Updated n8n workflow: {workflow.name}")
        return True

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        result = await self._request("DELETE", f"/workflows/{workflow_id}")
        if "error" in result:
            logger.error(f"Failed to delete workflow: {result['error']}")
            return False
        logger.info(f"Deleted n8n workflow: {workflow_id}")
        return True

    async def activate_workflow(self, workflow_id: str) -> bool:
        """Activate a workflow."""
        result = await self._request(
            "PATCH",
            f"/workflows/{workflow_id}",
            data={"active": True}
        )
        if "error" in result:
            return False
        logger.info(f"Activated n8n workflow: {workflow_id}")
        return True

    async def deactivate_workflow(self, workflow_id: str) -> bool:
        """Deactivate a workflow."""
        result = await self._request(
            "PATCH",
            f"/workflows/{workflow_id}",
            data={"active": False}
        )
        if "error" in result:
            return False
        logger.info(f"Deactivated n8n workflow: {workflow_id}")
        return True

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def execute_workflow(
        self,
        workflow_id: str,
        data: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Execute a workflow manually. Returns execution ID."""
        result = await self._request(
            "POST",
            f"/workflows/{workflow_id}/run",
            data={"data": data or {}}
        )
        if "error" in result:
            logger.error(f"Failed to execute workflow: {result['error']}")
            return None
        execution_id = result.get("data", {}).get("executionId")
        logger.info(f"Started workflow execution: {execution_id}")
        return execution_id

    async def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get execution details."""
        return await self._request("GET", f"/executions/{execution_id}")

    async def list_executions(
        self,
        workflow_id: str = None,
        status: str = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List workflow executions."""
        params = {"limit": limit}
        if workflow_id:
            params["workflowId"] = workflow_id
        if status:
            params["status"] = status

        result = await self._request("GET", "/executions", params=params)
        return result.get("data", [])

    async def stop_execution(self, execution_id: str) -> bool:
        """Stop a running execution."""
        result = await self._request("POST", f"/executions/{execution_id}/stop")
        return "error" not in result

    async def retry_execution(self, execution_id: str) -> Optional[str]:
        """Retry a failed execution."""
        result = await self._request("POST", f"/executions/{execution_id}/retry")
        return result.get("data", {}).get("executionId")

    # =========================================================================
    # CREDENTIALS
    # =========================================================================

    async def list_credentials(self) -> List[Dict[str, Any]]:
        """List all credentials."""
        result = await self._request("GET", "/credentials")
        return result.get("data", [])

    async def create_credential(
        self,
        name: str,
        credential_type: str,
        data: Dict[str, Any],
    ) -> Optional[str]:
        """Create a new credential. Returns credential ID."""
        result = await self._request("POST", "/credentials", data={
            "name": name,
            "type": credential_type,
            "data": data,
        })
        if "error" in result:
            logger.error(f"Failed to create credential: {result['error']}")
            return None
        return result.get("id")

    async def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        result = await self._request("DELETE", f"/credentials/{credential_id}")
        return "error" not in result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if n8n is running and accessible."""
        try:
            result = await self._request("GET", "/workflows", params={"limit": 1})
            return "error" not in result
        except Exception:
            return False

    async def import_workflow_json(self, json_str: str) -> Optional[str]:
        """Import a workflow from JSON string."""
        try:
            data = json.loads(json_str)
            workflow = N8nWorkflow.from_dict(data)
            return await self.create_workflow(workflow)
        except Exception as e:
            logger.error(f"Failed to import workflow: {e}")
            return None

    async def export_workflow_json(self, workflow_id: str) -> Optional[str]:
        """Export a workflow to JSON string."""
        workflow = await self.get_workflow(workflow_id)
        if workflow:
            return json.dumps(workflow.to_dict(), indent=2)
        return None

    async def clone_workflow(
        self,
        workflow_id: str,
        new_name: str,
    ) -> Optional[str]:
        """Clone an existing workflow."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None

        workflow.name = new_name
        workflow.active = False
        return await self.create_workflow(workflow)

    # =========================================================================
    # WORKFLOW BUILDERS
    # =========================================================================

    def create_webhook_workflow(
        self,
        name: str,
        webhook_path: str,
        handler_code: str,
    ) -> N8nWorkflow:
        """Create a simple webhook workflow."""
        workflow = N8nWorkflow(name=name, tags=["cosmos", "webhook"])

        # Webhook trigger
        webhook = N8nNode(
            id="webhook_1",
            name="Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": webhook_path,
                "httpMethod": "POST",
                "responseMode": "onReceived",
            },
        )
        workflow.add_node(webhook)

        # Handler code
        handler = N8nNode(
            id="code_1",
            name="Handler",
            type=N8nNodeType.CODE.value,
            position=[450, 300],
            parameters={"jsCode": handler_code},
        )
        workflow.add_node(handler)

        workflow.connect("Webhook", "Handler")

        return workflow

    def create_scheduled_workflow(
        self,
        name: str,
        cron_expression: str,
        task_code: str,
    ) -> N8nWorkflow:
        """Create a scheduled workflow."""
        workflow = N8nWorkflow(name=name, tags=["cosmos", "scheduled"])

        # Cron trigger
        cron = N8nNode(
            id="cron_1",
            name="Schedule",
            type=N8nNodeType.CRON.value,
            position=[250, 300],
            parameters={
                "rule": {
                    "interval": [{
                        "field": "cronExpression",
                        "expression": cron_expression
                    }]
                }
            },
        )
        workflow.add_node(cron)

        # Task code
        task = N8nNode(
            id="code_1",
            name="Task",
            type=N8nNodeType.CODE.value,
            position=[450, 300],
            parameters={"jsCode": task_code},
        )
        workflow.add_node(task)

        workflow.connect("Schedule", "Task")

        return workflow

    def create_api_workflow(
        self,
        name: str,
        webhook_path: str,
        api_url: str,
        method: str = "GET",
    ) -> N8nWorkflow:
        """Create a webhook-to-API workflow."""
        workflow = N8nWorkflow(name=name, tags=["cosmos", "api"])

        # Webhook trigger
        webhook = N8nNode(
            id="webhook_1",
            name="Webhook",
            type=N8nNodeType.WEBHOOK.value,
            position=[250, 300],
            parameters={
                "path": webhook_path,
                "httpMethod": "POST",
            },
        )
        workflow.add_node(webhook)

        # API call
        api = N8nNode(
            id="http_1",
            name="API Call",
            type=N8nNodeType.HTTP_REQUEST.value,
            position=[450, 300],
            parameters={
                "url": api_url,
                "method": method,
            },
        )
        workflow.add_node(api)

        workflow.connect("Webhook", "API Call")

        return workflow


# Singleton instance
n8n_integration = EnhancedN8nIntegration()
