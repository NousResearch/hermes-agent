"""
cosmos GitHub Actions Integration

"Version control? I invented that! I just forgot which version..."

GitHub Actions workflow management via the GitHub API.
"""

import asyncio
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import yaml
from pathlib import Path
from loguru import logger

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from Cosmos.cicd.pipeline_manager import (
    CICDProvider,
    Pipeline,
    PipelineRun,
    PipelineJob,
    PipelineStep,
    PipelineStatus,
    PipelineTrigger,
)


class GitHubActionsManager(CICDProvider):
    """
    GitHub Actions integration for CI/CD management.

    Features:
    - Workflow CRUD operations
    - Trigger workflow runs
    - Monitor workflow execution
    - Access workflow logs
    - Manage workflow secrets
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for GitHub Actions integration")

        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Make an authenticated request to GitHub API."""
        url = f"{self.base_url}{endpoint}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=self.headers,
                    **kwargs,
                )
                response.raise_for_status()

                if response.status_code == 204:
                    return {}
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"GitHub API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"GitHub request failed: {e}")
                return None

    def _parse_owner_repo(self, repository: str) -> tuple:
        """Parse owner/repo from repository string."""
        if "/" in repository:
            return repository.split("/", 1)
        raise ValueError(f"Invalid repository format: {repository}")

    # =========================================================================
    # WORKFLOW MANAGEMENT
    # =========================================================================

    async def list_pipelines(self, repository: str) -> List[Pipeline]:
        """List all workflows in a repository."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request("GET", f"/repos/{owner}/{repo}/actions/workflows")
        if not result:
            return []

        pipelines = []
        for workflow in result.get("workflows", []):
            pipeline = Pipeline(
                id=str(workflow["id"]),
                name=workflow["name"],
                repository=repository,
                description=workflow.get("path", ""),
            )
            pipelines.append(pipeline)

        return pipelines

    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get a workflow by ID. Requires repository context."""
        # This requires knowing the repository, so we store it in memory
        logger.warning("get_pipeline requires repository context - use list_pipelines instead")
        return None

    async def get_workflow(
        self,
        repository: str,
        workflow_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get workflow details."""
        owner, repo = self._parse_owner_repo(repository)

        return await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}",
        )

    async def create_pipeline(self, pipeline: Pipeline) -> bool:
        """Create a new workflow file in the repository."""
        if not pipeline.repository:
            logger.error("Pipeline must have a repository")
            return False

        owner, repo = self._parse_owner_repo(pipeline.repository)
        workflow_path = f".github/workflows/{pipeline.id}.yml"

        # Convert pipeline to GitHub Actions format
        workflow_content = yaml.dump(
            pipeline.to_github_actions(),
            default_flow_style=False,
            sort_keys=False,
        )

        # Create or update file via GitHub API
        # First, try to get existing file for SHA
        existing = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{workflow_path}",
        )

        payload = {
            "message": f"Create workflow: {pipeline.name}",
            "content": base64.b64encode(workflow_content.encode()).decode(),
        }

        if existing:
            payload["sha"] = existing["sha"]
            payload["message"] = f"Update workflow: {pipeline.name}"

        result = await self._request(
            "PUT",
            f"/repos/{owner}/{repo}/contents/{workflow_path}",
            json=payload,
        )

        return result is not None

    async def update_pipeline(self, pipeline: Pipeline) -> bool:
        """Update an existing workflow."""
        return await self.create_pipeline(pipeline)

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a workflow file. Requires repository context."""
        logger.warning("delete_pipeline requires repository context")
        return False

    async def delete_workflow(
        self,
        repository: str,
        workflow_path: str,
    ) -> bool:
        """Delete a workflow file from the repository."""
        owner, repo = self._parse_owner_repo(repository)

        # Get file SHA
        existing = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{workflow_path}",
        )

        if not existing:
            return False

        result = await self._request(
            "DELETE",
            f"/repos/{owner}/{repo}/contents/{workflow_path}",
            json={
                "message": f"Delete workflow: {workflow_path}",
                "sha": existing["sha"],
            },
        )

        return result is not None

    # =========================================================================
    # WORKFLOW RUNS
    # =========================================================================

    async def trigger_pipeline(
        self,
        pipeline_id: str,
        branch: str = "main",
        variables: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a workflow run via workflow_dispatch."""
        logger.warning("trigger_pipeline requires repository context - use trigger_workflow instead")
        return None

    async def trigger_workflow(
        self,
        repository: str,
        workflow_id: str,
        branch: str = "main",
        inputs: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a workflow_dispatch event."""
        owner, repo = self._parse_owner_repo(repository)

        payload = {"ref": branch}
        if inputs:
            payload["inputs"] = inputs

        result = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
            json=payload,
        )

        if result is not None:
            # Get the latest run for this workflow
            await asyncio.sleep(2)  # Wait for run to be created
            runs = await self.list_workflow_runs(repository, workflow_id, limit=1)
            if runs:
                return runs[0]

        return None

    async def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get a workflow run. Requires repository context."""
        logger.warning("get_run requires repository context - use get_workflow_run instead")
        return None

    async def get_workflow_run(
        self,
        repository: str,
        run_id: str,
    ) -> Optional[PipelineRun]:
        """Get details of a specific workflow run."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}",
        )

        if not result:
            return None

        return self._parse_run(result)

    def _parse_run(self, run: Dict) -> PipelineRun:
        """Parse GitHub Actions run to PipelineRun."""
        status_map = {
            "queued": PipelineStatus.QUEUED,
            "in_progress": PipelineStatus.RUNNING,
            "completed": PipelineStatus.SUCCESS,
            "waiting": PipelineStatus.WAITING,
        }

        conclusion_map = {
            "success": PipelineStatus.SUCCESS,
            "failure": PipelineStatus.FAILED,
            "cancelled": PipelineStatus.CANCELLED,
            "skipped": PipelineStatus.SKIPPED,
        }

        status = status_map.get(run["status"], PipelineStatus.PENDING)
        if run["status"] == "completed" and run.get("conclusion"):
            status = conclusion_map.get(run["conclusion"], PipelineStatus.FAILED)

        trigger_map = {
            "push": PipelineTrigger.PUSH,
            "pull_request": PipelineTrigger.PULL_REQUEST,
            "schedule": PipelineTrigger.SCHEDULE,
            "workflow_dispatch": PipelineTrigger.WORKFLOW_DISPATCH,
            "release": PipelineTrigger.RELEASE,
        }

        return PipelineRun(
            id=str(run["id"]),
            pipeline_id=str(run["workflow_id"]),
            pipeline_name=run["name"],
            trigger=trigger_map.get(run["event"], PipelineTrigger.MANUAL),
            triggered_by=run.get("triggering_actor", {}).get("login", "unknown"),
            commit_sha=run["head_sha"],
            branch=run["head_branch"],
            status=status,
            created_at=datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")),
            started_at=datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00")) if run.get("run_started_at") else None,
            completed_at=datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00")) if run["status"] == "completed" else None,
            logs_url=run.get("logs_url", ""),
        )

    async def list_runs(
        self,
        pipeline_id: str,
        limit: int = 20,
    ) -> List[PipelineRun]:
        """List workflow runs. Requires repository context."""
        logger.warning("list_runs requires repository context - use list_workflow_runs instead")
        return []

    async def list_workflow_runs(
        self,
        repository: str,
        workflow_id: str = None,
        limit: int = 20,
        branch: str = None,
        status: str = None,
    ) -> List[PipelineRun]:
        """List workflow runs with optional filters."""
        owner, repo = self._parse_owner_repo(repository)

        params = {"per_page": limit}
        if branch:
            params["branch"] = branch
        if status:
            params["status"] = status

        if workflow_id:
            endpoint = f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
        else:
            endpoint = f"/repos/{owner}/{repo}/actions/runs"

        result = await self._request("GET", endpoint, params=params)

        if not result:
            return []

        return [self._parse_run(run) for run in result.get("workflow_runs", [])]

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a workflow run. Requires repository context."""
        logger.warning("cancel_run requires repository context - use cancel_workflow_run instead")
        return False

    async def cancel_workflow_run(
        self,
        repository: str,
        run_id: str,
    ) -> bool:
        """Cancel a running workflow."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/cancel",
        )

        return result is not None

    async def retry_run(self, run_id: str) -> Optional[PipelineRun]:
        """Retry a workflow run. Requires repository context."""
        logger.warning("retry_run requires repository context - use retry_workflow_run instead")
        return None

    async def retry_workflow_run(
        self,
        repository: str,
        run_id: str,
    ) -> Optional[PipelineRun]:
        """Re-run a failed workflow."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/rerun",
        )

        if result is not None:
            return await self.get_workflow_run(repository, run_id)
        return None

    # =========================================================================
    # JOBS AND LOGS
    # =========================================================================

    async def get_jobs(
        self,
        repository: str,
        run_id: str,
    ) -> List[Dict]:
        """Get jobs for a workflow run."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
        )

        return result.get("jobs", []) if result else []

    async def get_logs(
        self,
        run_id: str,
        job_id: str = None,
    ) -> str:
        """Get logs. Requires repository context."""
        logger.warning("get_logs requires repository context - use download_logs instead")
        return ""

    async def download_logs(
        self,
        repository: str,
        run_id: str,
    ) -> Optional[bytes]:
        """Download logs for a workflow run (as zip file)."""
        owner, repo = self._parse_owner_repo(repository)

        url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.headers,
                    follow_redirects=True,
                )
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.error(f"Failed to download logs: {e}")
                return None

    # =========================================================================
    # SECRETS MANAGEMENT
    # =========================================================================

    async def list_secrets(self, repository: str) -> List[Dict]:
        """List repository secrets (names only, values are not retrievable)."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/secrets",
        )

        return result.get("secrets", []) if result else []

    async def set_secret(
        self,
        repository: str,
        name: str,
        value: str,
    ) -> bool:
        """Create or update a repository secret."""
        owner, repo = self._parse_owner_repo(repository)

        # Get public key for encryption
        key_data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/actions/secrets/public-key",
        )

        if not key_data:
            return False

        # Encrypt the secret using libsodium
        try:
            from nacl import encoding, public

            public_key = public.PublicKey(
                key_data["key"].encode(),
                encoding.Base64Encoder(),
            )
            sealed_box = public.SealedBox(public_key)
            encrypted = sealed_box.encrypt(value.encode())
            encrypted_value = base64.b64encode(encrypted).decode()

        except ImportError:
            logger.error("pynacl is required for secret encryption")
            return False

        result = await self._request(
            "PUT",
            f"/repos/{owner}/{repo}/actions/secrets/{name}",
            json={
                "encrypted_value": encrypted_value,
                "key_id": key_data["key_id"],
            },
        )

        return result is not None

    async def delete_secret(
        self,
        repository: str,
        name: str,
    ) -> bool:
        """Delete a repository secret."""
        owner, repo = self._parse_owner_repo(repository)

        result = await self._request(
            "DELETE",
            f"/repos/{owner}/{repo}/actions/secrets/{name}",
        )

        return result is not None

    # =========================================================================
    # ARTIFACTS
    # =========================================================================

    async def list_artifacts(
        self,
        repository: str,
        run_id: str = None,
    ) -> List[Dict]:
        """List artifacts for a repository or specific run."""
        owner, repo = self._parse_owner_repo(repository)

        if run_id:
            endpoint = f"/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
        else:
            endpoint = f"/repos/{owner}/{repo}/actions/artifacts"

        result = await self._request("GET", endpoint)

        return result.get("artifacts", []) if result else []

    async def download_artifact(
        self,
        repository: str,
        artifact_id: str,
    ) -> Optional[bytes]:
        """Download an artifact."""
        owner, repo = self._parse_owner_repo(repository)

        url = f"{self.base_url}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.headers,
                    follow_redirects=True,
                )
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.error(f"Failed to download artifact: {e}")
                return None
