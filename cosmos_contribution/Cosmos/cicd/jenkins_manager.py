"""
cosmos Jenkins Integration

"Jenkins? Is he the janitor?"

Jenkins CI/CD management via the Jenkins REST API.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
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
    PipelineStatus,
    PipelineTrigger,
)


class JenkinsManager(CICDProvider):
    """
    Jenkins CI/CD integration.

    Features:
    - Job management
    - Build triggering and monitoring
    - Pipeline execution
    - Credentials management
    - Node management
    """

    def __init__(
        self,
        url: str,
        username: str,
        token: str,
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for Jenkins integration")

        self.url = url.rstrip("/")
        self.username = username
        self.token = token
        self.auth = (username, token)

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[Any]:
        """Make an authenticated request to Jenkins API."""
        url = f"{self.url}{endpoint}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method,
                    url,
                    auth=self.auth,
                    **kwargs,
                )
                response.raise_for_status()

                if response.status_code == 204:
                    return {}

                # Jenkins can return text or JSON
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    return response.json()
                return response.text

            except httpx.HTTPStatusError as e:
                logger.error(f"Jenkins API error: {e.response.status_code}")
                return None
            except Exception as e:
                logger.error(f"Jenkins request failed: {e}")
                return None

    # =========================================================================
    # JOB MANAGEMENT
    # =========================================================================

    async def list_pipelines(self, repository: str = None) -> List[Pipeline]:
        """List all Jenkins jobs."""
        result = await self._request(
            "GET",
            "/api/json?tree=jobs[name,url,buildable,color]",
        )

        if not result:
            return []

        pipelines = []
        for job in result.get("jobs", []):
            pipeline = Pipeline(
                id=job["name"],
                name=job["name"],
                repository=repository or "",
            )
            pipelines.append(pipeline)

        return pipelines

    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get job details."""
        result = await self._request(
            "GET",
            f"/job/{pipeline_id}/api/json",
        )

        if not result:
            return None

        return Pipeline(
            id=result["name"],
            name=result.get("displayName", result["name"]),
            repository="",
            description=result.get("description", ""),
        )

    async def create_pipeline(self, pipeline: Pipeline) -> bool:
        """Create a new Jenkins job."""
        # Create a basic pipeline job config
        config = self._generate_pipeline_config(pipeline)

        result = await self._request(
            "POST",
            f"/createItem?name={pipeline.id}",
            content=config,
            headers={"Content-Type": "application/xml"},
        )

        return result is not None

    def _generate_pipeline_config(self, pipeline: Pipeline) -> str:
        """Generate Jenkins pipeline XML configuration."""
        # Basic pipeline job config
        steps = []
        for job in pipeline.jobs:
            for step in job.steps:
                steps.append(f'                stage("{step.name}") {{\n                    steps {{\n                        sh "{step.command}"\n                    }}\n                }}')

        stages = "\n".join(steps)

        config = f'''<?xml version='1.1' encoding='UTF-8'?>
<flow-definition plugin="workflow-job">
  <description>{pipeline.description}</description>
  <keepDependencies>false</keepDependencies>
  <properties/>
  <definition class="org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition" plugin="workflow-cps">
    <script>
pipeline {{
    agent any
    stages {{
{stages}
    }}
}}
    </script>
    <sandbox>true</sandbox>
  </definition>
  <triggers/>
  <disabled>false</disabled>
</flow-definition>'''

        return config

    async def update_pipeline(self, pipeline: Pipeline) -> bool:
        """Update an existing Jenkins job."""
        config = self._generate_pipeline_config(pipeline)

        result = await self._request(
            "POST",
            f"/job/{pipeline.id}/config.xml",
            content=config,
            headers={"Content-Type": "application/xml"},
        )

        return result is not None

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a Jenkins job."""
        result = await self._request(
            "POST",
            f"/job/{pipeline_id}/doDelete",
        )

        return result is not None

    async def get_job_config(self, job_name: str) -> Optional[str]:
        """Get the XML configuration of a job."""
        return await self._request(
            "GET",
            f"/job/{job_name}/config.xml",
        )

    async def enable_job(self, job_name: str) -> bool:
        """Enable a disabled job."""
        result = await self._request(
            "POST",
            f"/job/{job_name}/enable",
        )
        return result is not None

    async def disable_job(self, job_name: str) -> bool:
        """Disable a job."""
        result = await self._request(
            "POST",
            f"/job/{job_name}/disable",
        )
        return result is not None

    # =========================================================================
    # BUILD MANAGEMENT
    # =========================================================================

    async def trigger_pipeline(
        self,
        pipeline_id: str,
        branch: str = "main",
        variables: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a build."""
        return await self.trigger_build(pipeline_id, variables)

    async def trigger_build(
        self,
        job_name: str,
        parameters: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a new build."""
        if parameters:
            # Parameterized build
            params = "&".join(f"{k}={v}" for k, v in parameters.items())
            endpoint = f"/job/{job_name}/buildWithParameters?{params}"
        else:
            endpoint = f"/job/{job_name}/build"

        result = await self._request("POST", endpoint)

        if result is not None:
            # Get the queued item and wait for it to start
            await asyncio.sleep(2)
            builds = await self.list_runs(job_name, limit=1)
            if builds:
                return builds[0]

        return None

    async def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get build details. Requires job name context."""
        logger.warning("get_run requires job name - use get_build instead")
        return None

    async def get_build(
        self,
        job_name: str,
        build_number: int,
    ) -> Optional[PipelineRun]:
        """Get build details."""
        result = await self._request(
            "GET",
            f"/job/{job_name}/{build_number}/api/json",
        )

        if not result:
            return None

        return self._parse_build(result, job_name)

    def _parse_build(self, build: Dict, job_name: str) -> PipelineRun:
        """Parse Jenkins build to PipelineRun."""
        status_map = {
            "SUCCESS": PipelineStatus.SUCCESS,
            "FAILURE": PipelineStatus.FAILED,
            "UNSTABLE": PipelineStatus.FAILED,
            "ABORTED": PipelineStatus.CANCELLED,
            "NOT_BUILT": PipelineStatus.SKIPPED,
        }

        if build.get("building"):
            status = PipelineStatus.RUNNING
        elif build.get("result"):
            status = status_map.get(build["result"], PipelineStatus.FAILED)
        else:
            status = PipelineStatus.PENDING

        # Determine trigger from cause
        trigger = PipelineTrigger.MANUAL
        for cause in build.get("actions", []):
            if isinstance(cause, dict):
                causes = cause.get("causes", [])
                for c in causes:
                    if "SCMTrigger" in c.get("_class", ""):
                        trigger = PipelineTrigger.PUSH
                    elif "TimerTrigger" in c.get("_class", ""):
                        trigger = PipelineTrigger.SCHEDULE
                    elif "RemoteCause" in c.get("_class", ""):
                        trigger = PipelineTrigger.WEBHOOK

        # Get user who triggered
        triggered_by = "unknown"
        for action in build.get("actions", []):
            if isinstance(action, dict):
                causes = action.get("causes", [])
                for cause in causes:
                    if "userName" in cause:
                        triggered_by = cause["userName"]
                        break

        started_at = datetime.fromtimestamp(build["timestamp"] / 1000) if build.get("timestamp") else None
        duration_ms = build.get("duration", 0)
        completed_at = datetime.fromtimestamp((build["timestamp"] + duration_ms) / 1000) if started_at and duration_ms else None

        return PipelineRun(
            id=str(build["number"]),
            pipeline_id=job_name,
            pipeline_name=job_name,
            trigger=trigger,
            triggered_by=triggered_by,
            commit_sha="",
            branch="",
            status=status,
            started_at=started_at,
            completed_at=completed_at if not build.get("building") else None,
            logs_url=build.get("url", ""),
        )

    async def list_runs(
        self,
        pipeline_id: str,
        limit: int = 20,
    ) -> List[PipelineRun]:
        """List recent builds for a job."""
        result = await self._request(
            "GET",
            f"/job/{pipeline_id}/api/json?tree=builds[number,url,result,building,timestamp,duration,actions[causes[userName,shortDescription]]]",
        )

        if not result:
            return []

        builds = result.get("builds", [])[:limit]
        return [self._parse_build(b, pipeline_id) for b in builds]

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a build. Requires job name."""
        logger.warning("cancel_run requires job name - use stop_build instead")
        return False

    async def stop_build(
        self,
        job_name: str,
        build_number: int,
    ) -> bool:
        """Stop a running build."""
        result = await self._request(
            "POST",
            f"/job/{job_name}/{build_number}/stop",
        )
        return result is not None

    async def retry_run(self, run_id: str) -> Optional[PipelineRun]:
        """Retry by triggering a new build."""
        logger.warning("retry_run requires job name - use trigger_build instead")
        return None

    async def get_logs(
        self,
        run_id: str,
        job_id: str = None,
    ) -> str:
        """Get build console output. Requires job name."""
        logger.warning("get_logs requires job name - use get_console_output instead")
        return ""

    async def get_console_output(
        self,
        job_name: str,
        build_number: int,
    ) -> str:
        """Get console output for a build."""
        result = await self._request(
            "GET",
            f"/job/{job_name}/{build_number}/consoleText",
        )
        return result if result else ""

    async def get_progressive_console(
        self,
        job_name: str,
        build_number: int,
        start: int = 0,
    ) -> Dict[str, Any]:
        """Get progressive console output."""
        url = f"{self.url}/job/{job_name}/{build_number}/logText/progressiveText?start={start}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, auth=self.auth)
                response.raise_for_status()

                return {
                    "text": response.text,
                    "size": int(response.headers.get("X-Text-Size", 0)),
                    "more_data": response.headers.get("X-More-Data") == "true",
                }
            except Exception as e:
                logger.error(f"Failed to get progressive console: {e}")
                return {"text": "", "size": 0, "more_data": False}

    # =========================================================================
    # ARTIFACTS
    # =========================================================================

    async def list_artifacts(
        self,
        job_name: str,
        build_number: int,
    ) -> List[Dict]:
        """List artifacts for a build."""
        result = await self._request(
            "GET",
            f"/job/{job_name}/{build_number}/api/json?tree=artifacts[*]",
        )

        return result.get("artifacts", []) if result else []

    async def download_artifact(
        self,
        job_name: str,
        build_number: int,
        artifact_path: str,
    ) -> Optional[bytes]:
        """Download an artifact."""
        url = f"{self.url}/job/{job_name}/{build_number}/artifact/{artifact_path}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, auth=self.auth)
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.error(f"Failed to download artifact: {e}")
                return None

    # =========================================================================
    # QUEUE MANAGEMENT
    # =========================================================================

    async def get_queue(self) -> List[Dict]:
        """Get the build queue."""
        result = await self._request(
            "GET",
            "/queue/api/json",
        )

        return result.get("items", []) if result else []

    async def cancel_queue_item(self, queue_id: int) -> bool:
        """Cancel a queued build."""
        result = await self._request(
            "POST",
            f"/queue/cancelItem?id={queue_id}",
        )
        return result is not None

    # =========================================================================
    # NODE MANAGEMENT
    # =========================================================================

    async def list_nodes(self) -> List[Dict]:
        """List all Jenkins nodes."""
        result = await self._request(
            "GET",
            "/computer/api/json?tree=computer[displayName,description,idle,offline,temporarilyOffline]",
        )

        return result.get("computer", []) if result else []

    async def get_node(self, node_name: str) -> Optional[Dict]:
        """Get node details."""
        return await self._request(
            "GET",
            f"/computer/{node_name}/api/json",
        )

    async def toggle_node_offline(
        self,
        node_name: str,
        offline: bool,
        message: str = "",
    ) -> bool:
        """Set node online/offline status."""
        endpoint = f"/computer/{node_name}/toggleOffline"
        if message:
            endpoint += f"?offlineMessage={message}"

        result = await self._request("POST", endpoint)
        return result is not None

    # =========================================================================
    # CREDENTIALS MANAGEMENT
    # =========================================================================

    async def list_credentials(
        self,
        domain: str = "_",
    ) -> List[Dict]:
        """List credentials in a domain."""
        result = await self._request(
            "GET",
            f"/credentials/store/system/domain/{domain}/api/json",
        )

        return result.get("credentials", []) if result else []

    async def create_credential(
        self,
        credential_id: str,
        username: str,
        password: str,
        description: str = "",
        domain: str = "_",
    ) -> bool:
        """Create a username/password credential."""
        xml_config = f'''<com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl>
  <scope>GLOBAL</scope>
  <id>{credential_id}</id>
  <description>{description}</description>
  <username>{username}</username>
  <password>{password}</password>
</com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl>'''

        result = await self._request(
            "POST",
            f"/credentials/store/system/domain/{domain}/createCredentials",
            data={"json": f'{{"credentials": {{"scope": "GLOBAL", "id": "{credential_id}", "username": "{username}", "password": "{password}", "description": "{description}", "$class": "com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl"}}}}'},
        )

        return result is not None

    async def delete_credential(
        self,
        credential_id: str,
        domain: str = "_",
    ) -> bool:
        """Delete a credential."""
        result = await self._request(
            "POST",
            f"/credentials/store/system/domain/{domain}/credential/{credential_id}/doDelete",
        )
        return result is not None

    # =========================================================================
    # SYSTEM MANAGEMENT
    # =========================================================================

    async def get_system_info(self) -> Optional[Dict]:
        """Get Jenkins system information."""
        result = await self._request(
            "GET",
            "/api/json",
        )

        if not result:
            return None

        return {
            "version": result.get("hudson", {}).get("version"),
            "mode": result.get("mode"),
            "node_description": result.get("nodeDescription"),
            "node_name": result.get("nodeName"),
            "num_executors": result.get("numExecutors"),
            "description": result.get("description"),
            "url": result.get("url"),
        }

    async def quiet_down(self) -> bool:
        """Prepare Jenkins for shutdown (no new builds)."""
        result = await self._request("POST", "/quietDown")
        return result is not None

    async def cancel_quiet_down(self) -> bool:
        """Cancel shutdown preparation."""
        result = await self._request("POST", "/cancelQuietDown")
        return result is not None

    async def restart(self, safe: bool = True) -> bool:
        """Restart Jenkins."""
        endpoint = "/safeRestart" if safe else "/restart"
        result = await self._request("POST", endpoint)
        return result is not None
