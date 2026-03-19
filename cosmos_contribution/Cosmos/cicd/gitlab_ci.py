"""
cosmos GitLab CI Integration

"GitLab? Is that where the gits are kept?"

GitLab CI/CD pipeline management via the GitLab API.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml
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


class GitLabCIManager(CICDProvider):
    """
    GitLab CI/CD integration.

    Features:
    - Pipeline management
    - Job control
    - Variable management
    - Merge request pipelines
    - Scheduled pipelines
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://gitlab.com",
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for GitLab CI integration")

        self.token = token
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v4"
        self.headers = {
            "PRIVATE-TOKEN": token,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[Any]:
        """Make an authenticated request to GitLab API."""
        url = f"{self.api_url}{endpoint}"

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
                logger.error(f"GitLab API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"GitLab request failed: {e}")
                return None

    def _encode_project(self, project: str) -> str:
        """URL-encode project path."""
        return project.replace("/", "%2F")

    # =========================================================================
    # PIPELINE MANAGEMENT
    # =========================================================================

    async def list_pipelines(self, repository: str) -> List[Pipeline]:
        """List pipelines for a project."""
        project = self._encode_project(repository)

        result = await self._request(
            "GET",
            f"/projects/{project}/pipelines",
            params={"per_page": 100},
        )

        if not result:
            return []

        # GitLab doesn't have "pipeline definitions" like GitHub workflows
        # Each pipeline is an execution instance
        # We'll return unique refs as "pipelines"
        seen_refs = {}
        for pipeline in result:
            ref = pipeline["ref"]
            if ref not in seen_refs:
                seen_refs[ref] = Pipeline(
                    id=ref,
                    name=f"Pipeline for {ref}",
                    repository=repository,
                    branch=ref,
                )

        return list(seen_refs.values())

    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        # GitLab pipelines are executions, not definitions
        logger.warning("GitLab pipelines are executions - use get_pipeline_run instead")
        return None

    async def create_pipeline(self, pipeline: Pipeline) -> bool:
        """Create a .gitlab-ci.yml file in the repository."""
        project = self._encode_project(pipeline.repository)

        # Convert pipeline to GitLab CI format
        ci_content = yaml.dump(
            pipeline.to_gitlab_ci(),
            default_flow_style=False,
            sort_keys=False,
        )

        # Create or update .gitlab-ci.yml
        # First check if file exists
        existing = await self._request(
            "GET",
            f"/projects/{project}/repository/files/.gitlab-ci.yml",
            params={"ref": pipeline.branch},
        )

        import base64
        content_encoded = base64.b64encode(ci_content.encode()).decode()

        if existing:
            # Update existing file
            result = await self._request(
                "PUT",
                f"/projects/{project}/repository/files/.gitlab-ci.yml",
                json={
                    "branch": pipeline.branch,
                    "content": ci_content,
                    "commit_message": f"Update CI pipeline: {pipeline.name}",
                },
            )
        else:
            # Create new file
            result = await self._request(
                "POST",
                f"/projects/{project}/repository/files/.gitlab-ci.yml",
                json={
                    "branch": pipeline.branch,
                    "content": ci_content,
                    "commit_message": f"Create CI pipeline: {pipeline.name}",
                },
            )

        return result is not None

    async def update_pipeline(self, pipeline: Pipeline) -> bool:
        """Update pipeline definition."""
        return await self.create_pipeline(pipeline)

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline run."""
        logger.warning("Cannot delete pipeline definition via API - delete .gitlab-ci.yml manually")
        return False

    # =========================================================================
    # PIPELINE RUNS
    # =========================================================================

    async def trigger_pipeline(
        self,
        pipeline_id: str,
        branch: str = "main",
        variables: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a pipeline for a project."""
        # pipeline_id is the project path for GitLab
        return await self.create_pipeline_run(pipeline_id, branch, variables)

    async def create_pipeline_run(
        self,
        project: str,
        ref: str = "main",
        variables: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Create and run a new pipeline."""
        project_encoded = self._encode_project(project)

        payload = {"ref": ref}
        if variables:
            payload["variables"] = [
                {"key": k, "value": v}
                for k, v in variables.items()
            ]

        result = await self._request(
            "POST",
            f"/projects/{project_encoded}/pipeline",
            json=payload,
        )

        if result:
            return self._parse_pipeline_run(result, project)
        return None

    def _parse_pipeline_run(self, data: Dict, project: str) -> PipelineRun:
        """Parse GitLab pipeline to PipelineRun."""
        status_map = {
            "created": PipelineStatus.PENDING,
            "waiting_for_resource": PipelineStatus.WAITING,
            "preparing": PipelineStatus.PENDING,
            "pending": PipelineStatus.PENDING,
            "running": PipelineStatus.RUNNING,
            "success": PipelineStatus.SUCCESS,
            "failed": PipelineStatus.FAILED,
            "canceled": PipelineStatus.CANCELLED,
            "skipped": PipelineStatus.SKIPPED,
            "manual": PipelineStatus.WAITING,
            "scheduled": PipelineStatus.QUEUED,
        }

        source_map = {
            "push": PipelineTrigger.PUSH,
            "web": PipelineTrigger.MANUAL,
            "trigger": PipelineTrigger.WEBHOOK,
            "schedule": PipelineTrigger.SCHEDULE,
            "api": PipelineTrigger.MANUAL,
            "merge_request_event": PipelineTrigger.PULL_REQUEST,
        }

        return PipelineRun(
            id=str(data["id"]),
            pipeline_id=project,
            pipeline_name=f"Pipeline #{data['id']}",
            trigger=source_map.get(data.get("source", ""), PipelineTrigger.MANUAL),
            triggered_by=data.get("user", {}).get("username", "unknown"),
            commit_sha=data.get("sha", ""),
            branch=data.get("ref", ""),
            status=status_map.get(data["status"], PipelineStatus.PENDING),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"].replace("Z", "+00:00")) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["finished_at"].replace("Z", "+00:00")) if data.get("finished_at") else None,
            logs_url=data.get("web_url", ""),
        )

    async def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run. Requires project context."""
        logger.warning("get_run requires project context - use get_pipeline_run instead")
        return None

    async def get_pipeline_run(
        self,
        project: str,
        pipeline_id: str,
    ) -> Optional[PipelineRun]:
        """Get details of a pipeline run."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "GET",
            f"/projects/{project_encoded}/pipelines/{pipeline_id}",
        )

        if result:
            return self._parse_pipeline_run(result, project)
        return None

    async def list_runs(
        self,
        pipeline_id: str,
        limit: int = 20,
    ) -> List[PipelineRun]:
        """List pipeline runs for a project."""
        return await self.list_pipeline_runs(pipeline_id, limit=limit)

    async def list_pipeline_runs(
        self,
        project: str,
        ref: str = None,
        status: str = None,
        limit: int = 20,
    ) -> List[PipelineRun]:
        """List pipeline runs with optional filters."""
        project_encoded = self._encode_project(project)

        params = {"per_page": limit}
        if ref:
            params["ref"] = ref
        if status:
            params["status"] = status

        result = await self._request(
            "GET",
            f"/projects/{project_encoded}/pipelines",
            params=params,
        )

        if not result:
            return []

        return [self._parse_pipeline_run(p, project) for p in result]

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a pipeline run. Requires project context."""
        logger.warning("cancel_run requires project context - use cancel_pipeline_run instead")
        return False

    async def cancel_pipeline_run(
        self,
        project: str,
        pipeline_id: str,
    ) -> bool:
        """Cancel a running pipeline."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "POST",
            f"/projects/{project_encoded}/pipelines/{pipeline_id}/cancel",
        )

        return result is not None

    async def retry_run(self, run_id: str) -> Optional[PipelineRun]:
        """Retry a pipeline run. Requires project context."""
        logger.warning("retry_run requires project context - use retry_pipeline_run instead")
        return None

    async def retry_pipeline_run(
        self,
        project: str,
        pipeline_id: str,
    ) -> Optional[PipelineRun]:
        """Retry a failed pipeline."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "POST",
            f"/projects/{project_encoded}/pipelines/{pipeline_id}/retry",
        )

        if result:
            return self._parse_pipeline_run(result, project)
        return None

    # =========================================================================
    # JOBS
    # =========================================================================

    async def list_jobs(
        self,
        project: str,
        pipeline_id: str,
    ) -> List[Dict]:
        """List jobs in a pipeline."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "GET",
            f"/projects/{project_encoded}/pipelines/{pipeline_id}/jobs",
        )

        return result if result else []

    async def get_job(
        self,
        project: str,
        job_id: str,
    ) -> Optional[Dict]:
        """Get job details."""
        project_encoded = self._encode_project(project)

        return await self._request(
            "GET",
            f"/projects/{project_encoded}/jobs/{job_id}",
        )

    async def retry_job(
        self,
        project: str,
        job_id: str,
    ) -> Optional[Dict]:
        """Retry a specific job."""
        project_encoded = self._encode_project(project)

        return await self._request(
            "POST",
            f"/projects/{project_encoded}/jobs/{job_id}/retry",
        )

    async def cancel_job(
        self,
        project: str,
        job_id: str,
    ) -> bool:
        """Cancel a running job."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "POST",
            f"/projects/{project_encoded}/jobs/{job_id}/cancel",
        )

        return result is not None

    async def play_job(
        self,
        project: str,
        job_id: str,
    ) -> Optional[Dict]:
        """Play a manual job."""
        project_encoded = self._encode_project(project)

        return await self._request(
            "POST",
            f"/projects/{project_encoded}/jobs/{job_id}/play",
        )

    async def get_logs(
        self,
        run_id: str,
        job_id: str = None,
    ) -> str:
        """Get job logs. Requires project context."""
        logger.warning("get_logs requires project context - use get_job_log instead")
        return ""

    async def get_job_log(
        self,
        project: str,
        job_id: str,
    ) -> str:
        """Get logs for a job."""
        project_encoded = self._encode_project(project)

        url = f"{self.api_url}/projects/{project_encoded}/jobs/{job_id}/trace"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.error(f"Failed to get job log: {e}")
                return ""

    # =========================================================================
    # VARIABLES
    # =========================================================================

    async def list_variables(self, project: str) -> List[Dict]:
        """List project CI/CD variables."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "GET",
            f"/projects/{project_encoded}/variables",
        )

        return result if result else []

    async def get_variable(
        self,
        project: str,
        key: str,
    ) -> Optional[Dict]:
        """Get a specific variable."""
        project_encoded = self._encode_project(project)

        return await self._request(
            "GET",
            f"/projects/{project_encoded}/variables/{key}",
        )

    async def create_variable(
        self,
        project: str,
        key: str,
        value: str,
        protected: bool = False,
        masked: bool = False,
        environment_scope: str = "*",
    ) -> bool:
        """Create a CI/CD variable."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "POST",
            f"/projects/{project_encoded}/variables",
            json={
                "key": key,
                "value": value,
                "protected": protected,
                "masked": masked,
                "environment_scope": environment_scope,
            },
        )

        return result is not None

    async def update_variable(
        self,
        project: str,
        key: str,
        value: str,
        protected: bool = None,
        masked: bool = None,
    ) -> bool:
        """Update a CI/CD variable."""
        project_encoded = self._encode_project(project)

        payload = {"value": value}
        if protected is not None:
            payload["protected"] = protected
        if masked is not None:
            payload["masked"] = masked

        result = await self._request(
            "PUT",
            f"/projects/{project_encoded}/variables/{key}",
            json=payload,
        )

        return result is not None

    async def delete_variable(
        self,
        project: str,
        key: str,
    ) -> bool:
        """Delete a CI/CD variable."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "DELETE",
            f"/projects/{project_encoded}/variables/{key}",
        )

        return result is not None

    # =========================================================================
    # SCHEDULES
    # =========================================================================

    async def list_schedules(self, project: str) -> List[Dict]:
        """List pipeline schedules."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "GET",
            f"/projects/{project_encoded}/pipeline_schedules",
        )

        return result if result else []

    async def create_schedule(
        self,
        project: str,
        description: str,
        ref: str,
        cron: str,
        active: bool = True,
        variables: Dict[str, str] = None,
    ) -> Optional[Dict]:
        """Create a pipeline schedule."""
        project_encoded = self._encode_project(project)

        payload = {
            "description": description,
            "ref": ref,
            "cron": cron,
            "active": active,
        }

        result = await self._request(
            "POST",
            f"/projects/{project_encoded}/pipeline_schedules",
            json=payload,
        )

        if result and variables:
            schedule_id = result["id"]
            for key, value in variables.items():
                await self._request(
                    "POST",
                    f"/projects/{project_encoded}/pipeline_schedules/{schedule_id}/variables",
                    json={"key": key, "value": value},
                )

        return result

    async def delete_schedule(
        self,
        project: str,
        schedule_id: str,
    ) -> bool:
        """Delete a pipeline schedule."""
        project_encoded = self._encode_project(project)

        result = await self._request(
            "DELETE",
            f"/projects/{project_encoded}/pipeline_schedules/{schedule_id}",
        )

        return result is not None

    async def run_schedule(
        self,
        project: str,
        schedule_id: str,
    ) -> Optional[Dict]:
        """Trigger a scheduled pipeline immediately."""
        project_encoded = self._encode_project(project)

        return await self._request(
            "POST",
            f"/projects/{project_encoded}/pipeline_schedules/{schedule_id}/play",
        )
