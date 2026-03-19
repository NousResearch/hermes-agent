"""
cosmos CI/CD Pipeline Manager

"I've automated the automation! It's automation all the way down!"

Unified pipeline management across multiple CI/CD platforms.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import json
from loguru import logger


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    WAITING = "waiting"


class PipelineTrigger(Enum):
    """Pipeline trigger types."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    WEBHOOK = "webhook"
    TAG = "tag"
    RELEASE = "release"
    WORKFLOW_DISPATCH = "workflow_dispatch"


@dataclass
class PipelineStep:
    """Individual step in a pipeline job."""
    name: str
    command: str
    working_directory: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_minutes: int = 30
    continue_on_error: bool = False
    condition: Optional[str] = None

    # Execution results
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: str = ""
    exit_code: Optional[int] = None


@dataclass
class PipelineJob:
    """Job containing multiple steps."""
    id: str
    name: str
    steps: List[PipelineStep] = field(default_factory=list)
    runs_on: str = "ubuntu-latest"
    needs: List[str] = field(default_factory=list)  # Job dependencies
    environment: Dict[str, str] = field(default_factory=dict)
    services: Dict[str, Dict] = field(default_factory=dict)
    timeout_minutes: int = 60
    continue_on_error: bool = False
    matrix: Optional[Dict[str, List]] = None
    condition: Optional[str] = None

    # Execution state
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Pipeline:
    """Pipeline definition."""
    id: str
    name: str
    repository: str
    branch: str = "main"
    triggers: List[PipelineTrigger] = field(default_factory=list)
    jobs: List[PipelineJob] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)  # Required secrets
    schedule: Optional[str] = None  # Cron expression
    concurrency: Optional[Dict[str, Any]] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    description: str = ""

    def to_github_actions(self) -> Dict[str, Any]:
        """Convert to GitHub Actions workflow format."""
        workflow = {
            "name": self.name,
            "on": {},
            "env": self.environment,
            "jobs": {},
        }

        # Configure triggers
        for trigger in self.triggers:
            if trigger == PipelineTrigger.PUSH:
                workflow["on"]["push"] = {"branches": [self.branch]}
            elif trigger == PipelineTrigger.PULL_REQUEST:
                workflow["on"]["pull_request"] = {"branches": [self.branch]}
            elif trigger == PipelineTrigger.SCHEDULE and self.schedule:
                workflow["on"]["schedule"] = [{"cron": self.schedule}]
            elif trigger == PipelineTrigger.WORKFLOW_DISPATCH:
                workflow["on"]["workflow_dispatch"] = {}

        # Configure jobs
        for job in self.jobs:
            job_config = {
                "runs-on": job.runs_on,
                "steps": [],
            }

            if job.needs:
                job_config["needs"] = job.needs
            if job.environment:
                job_config["env"] = job.environment
            if job.services:
                job_config["services"] = job.services
            if job.timeout_minutes:
                job_config["timeout-minutes"] = job.timeout_minutes
            if job.matrix:
                job_config["strategy"] = {"matrix": job.matrix}
            if job.condition:
                job_config["if"] = job.condition

            for step in job.steps:
                step_config = {"name": step.name, "run": step.command}
                if step.working_directory != ".":
                    step_config["working-directory"] = step.working_directory
                if step.environment:
                    step_config["env"] = step.environment
                if step.timeout_minutes:
                    step_config["timeout-minutes"] = step.timeout_minutes
                if step.continue_on_error:
                    step_config["continue-on-error"] = True
                if step.condition:
                    step_config["if"] = step.condition
                job_config["steps"].append(step_config)

            workflow["jobs"][job.id] = job_config

        if self.concurrency:
            workflow["concurrency"] = self.concurrency

        return workflow

    def to_gitlab_ci(self) -> Dict[str, Any]:
        """Convert to GitLab CI format."""
        config = {
            "stages": [],
            "variables": self.environment,
        }

        # Add stages from jobs
        seen_stages = set()
        for job in self.jobs:
            stage = job.id.replace("-", "_")
            if stage not in seen_stages:
                config["stages"].append(stage)
                seen_stages.add(stage)

        # Configure jobs
        for job in self.jobs:
            job_config = {
                "stage": job.id.replace("-", "_"),
                "script": [step.command for step in job.steps],
            }

            if job.needs:
                job_config["needs"] = job.needs
            if job.environment:
                job_config["variables"] = job.environment
            if job.services:
                job_config["services"] = list(job.services.keys())
            if job.condition:
                job_config["rules"] = [{"if": job.condition}]

            config[job.name.replace(" ", "_").lower()] = job_config

        return config


@dataclass
class PipelineRun:
    """Pipeline execution instance."""
    id: str
    pipeline_id: str
    pipeline_name: str
    trigger: PipelineTrigger
    triggered_by: str
    commit_sha: str = ""
    branch: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    jobs: List[PipelineJob] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    artifacts: List[str] = field(default_factory=list)
    logs_url: str = ""
    error_message: str = ""

    @property
    def duration(self) -> Optional[timedelta]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "trigger": self.trigger.value,
            "triggered_by": self.triggered_by,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "artifacts": self.artifacts,
            "logs_url": self.logs_url,
        }


class CICDProvider(ABC):
    """Abstract base class for CI/CD providers."""

    @abstractmethod
    async def list_pipelines(self, repository: str) -> List[Pipeline]:
        """List all pipelines for a repository."""
        pass

    @abstractmethod
    async def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get pipeline by ID."""
        pass

    @abstractmethod
    async def create_pipeline(self, pipeline: Pipeline) -> bool:
        """Create a new pipeline."""
        pass

    @abstractmethod
    async def update_pipeline(self, pipeline: Pipeline) -> bool:
        """Update an existing pipeline."""
        pass

    @abstractmethod
    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline."""
        pass

    @abstractmethod
    async def trigger_pipeline(
        self,
        pipeline_id: str,
        branch: str = "main",
        variables: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a pipeline run."""
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run by ID."""
        pass

    @abstractmethod
    async def list_runs(
        self,
        pipeline_id: str,
        limit: int = 20,
    ) -> List[PipelineRun]:
        """List recent pipeline runs."""
        pass

    @abstractmethod
    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        pass

    @abstractmethod
    async def retry_run(self, run_id: str) -> Optional[PipelineRun]:
        """Retry a failed pipeline run."""
        pass

    @abstractmethod
    async def get_logs(self, run_id: str, job_id: str = None) -> str:
        """Get logs for a pipeline run."""
        pass


class PipelineManager:
    """
    Unified CI/CD pipeline management across providers.

    Features:
    - Multi-provider support (GitHub Actions, GitLab CI, Jenkins)
    - Pipeline templates
    - Cross-platform pipeline conversion
    - Unified monitoring and alerting
    - Pipeline analytics
    """

    def __init__(self):
        self.providers: Dict[str, CICDProvider] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.runs: Dict[str, PipelineRun] = {}
        self.templates: Dict[str, Pipeline] = {}
        self.webhooks: Dict[str, Callable] = {}

        self._load_templates()

    def _load_templates(self):
        """Load built-in pipeline templates."""
        # Python package template
        self.templates["python-package"] = Pipeline(
            id="python-package",
            name="Python Package CI",
            repository="",
            triggers=[PipelineTrigger.PUSH, PipelineTrigger.PULL_REQUEST],
            jobs=[
                PipelineJob(
                    id="test",
                    name="Test",
                    runs_on="ubuntu-latest",
                    matrix={"python-version": ["3.9", "3.10", "3.11", "3.12"]},
                    steps=[
                        PipelineStep(
                            name="Checkout",
                            command="git checkout ${{ github.sha }}",
                        ),
                        PipelineStep(
                            name="Set up Python",
                            command="# Uses actions/setup-python@v4",
                        ),
                        PipelineStep(
                            name="Install dependencies",
                            command="pip install -e .[dev]",
                        ),
                        PipelineStep(
                            name="Run tests",
                            command="pytest --cov=. --cov-report=xml",
                        ),
                        PipelineStep(
                            name="Upload coverage",
                            command="# Uses codecov/codecov-action@v3",
                        ),
                    ],
                ),
                PipelineJob(
                    id="lint",
                    name="Lint",
                    runs_on="ubuntu-latest",
                    steps=[
                        PipelineStep(name="Checkout", command="git checkout ${{ github.sha }}"),
                        PipelineStep(name="Run ruff", command="ruff check ."),
                        PipelineStep(name="Run mypy", command="mypy ."),
                    ],
                ),
            ],
            description="Standard Python package CI with testing and linting",
        )

        # Node.js application template
        self.templates["nodejs-app"] = Pipeline(
            id="nodejs-app",
            name="Node.js Application CI",
            repository="",
            triggers=[PipelineTrigger.PUSH, PipelineTrigger.PULL_REQUEST],
            jobs=[
                PipelineJob(
                    id="build",
                    name="Build and Test",
                    runs_on="ubuntu-latest",
                    matrix={"node-version": ["18.x", "20.x"]},
                    steps=[
                        PipelineStep(name="Checkout", command="git checkout"),
                        PipelineStep(name="Install deps", command="npm ci"),
                        PipelineStep(name="Build", command="npm run build"),
                        PipelineStep(name="Test", command="npm test"),
                        PipelineStep(name="Lint", command="npm run lint"),
                    ],
                ),
            ],
            description="Node.js application CI with build, test, and lint",
        )

        # Docker build and push template
        self.templates["docker-build"] = Pipeline(
            id="docker-build",
            name="Docker Build and Push",
            repository="",
            triggers=[PipelineTrigger.PUSH, PipelineTrigger.TAG],
            jobs=[
                PipelineJob(
                    id="build",
                    name="Build and Push",
                    runs_on="ubuntu-latest",
                    steps=[
                        PipelineStep(name="Checkout", command="git checkout"),
                        PipelineStep(
                            name="Set up Docker Buildx",
                            command="# Uses docker/setup-buildx-action@v3",
                        ),
                        PipelineStep(
                            name="Login to Registry",
                            command="docker login -u ${{ secrets.DOCKER_USERNAME }} -p ${{ secrets.DOCKER_PASSWORD }}",
                        ),
                        PipelineStep(
                            name="Build and push",
                            command="docker buildx build --push -t ${{ github.repository }}:${{ github.sha }} .",
                        ),
                    ],
                ),
            ],
            secrets=["DOCKER_USERNAME", "DOCKER_PASSWORD"],
            description="Build Docker image and push to registry",
        )

        # Kubernetes deployment template
        self.templates["k8s-deploy"] = Pipeline(
            id="k8s-deploy",
            name="Kubernetes Deployment",
            repository="",
            triggers=[PipelineTrigger.WORKFLOW_DISPATCH],
            jobs=[
                PipelineJob(
                    id="deploy",
                    name="Deploy to Kubernetes",
                    runs_on="ubuntu-latest",
                    steps=[
                        PipelineStep(name="Checkout", command="git checkout"),
                        PipelineStep(
                            name="Configure kubectl",
                            command="echo ${{ secrets.KUBE_CONFIG }} | base64 -d > ~/.kube/config",
                        ),
                        PipelineStep(
                            name="Deploy",
                            command="kubectl apply -f k8s/",
                        ),
                        PipelineStep(
                            name="Wait for rollout",
                            command="kubectl rollout status deployment/app",
                        ),
                    ],
                ),
            ],
            secrets=["KUBE_CONFIG"],
            description="Deploy application to Kubernetes cluster",
        )

        # Terraform infrastructure template
        self.templates["terraform"] = Pipeline(
            id="terraform",
            name="Terraform Infrastructure",
            repository="",
            triggers=[PipelineTrigger.PULL_REQUEST, PipelineTrigger.PUSH],
            jobs=[
                PipelineJob(
                    id="plan",
                    name="Terraform Plan",
                    runs_on="ubuntu-latest",
                    condition="github.event_name == 'pull_request'",
                    steps=[
                        PipelineStep(name="Checkout", command="git checkout"),
                        PipelineStep(name="Setup Terraform", command="# Uses hashicorp/setup-terraform@v2"),
                        PipelineStep(name="Init", command="terraform init"),
                        PipelineStep(name="Validate", command="terraform validate"),
                        PipelineStep(name="Plan", command="terraform plan -out=tfplan"),
                    ],
                ),
                PipelineJob(
                    id="apply",
                    name="Terraform Apply",
                    runs_on="ubuntu-latest",
                    condition="github.ref == 'refs/heads/main' && github.event_name == 'push'",
                    steps=[
                        PipelineStep(name="Checkout", command="git checkout"),
                        PipelineStep(name="Setup Terraform", command="# Uses hashicorp/setup-terraform@v2"),
                        PipelineStep(name="Init", command="terraform init"),
                        PipelineStep(name="Apply", command="terraform apply -auto-approve"),
                    ],
                ),
            ],
            description="Terraform plan on PR, apply on merge to main",
        )

    # =========================================================================
    # PROVIDER MANAGEMENT
    # =========================================================================

    def register_provider(self, name: str, provider: CICDProvider):
        """Register a CI/CD provider."""
        self.providers[name] = provider
        logger.info(f"Registered CI/CD provider: {name}")

    def get_provider(self, name: str) -> Optional[CICDProvider]:
        """Get a registered provider."""
        return self.providers.get(name)

    # =========================================================================
    # PIPELINE CRUD
    # =========================================================================

    async def create_pipeline(
        self,
        provider: str,
        pipeline: Pipeline,
    ) -> bool:
        """Create a pipeline on a provider."""
        if provider not in self.providers:
            logger.error(f"Provider not found: {provider}")
            return False

        success = await self.providers[provider].create_pipeline(pipeline)
        if success:
            self.pipelines[pipeline.id] = pipeline
            logger.info(f"Created pipeline {pipeline.name} on {provider}")
        return success

    async def create_from_template(
        self,
        provider: str,
        template_id: str,
        repository: str,
        name: str = None,
        **overrides,
    ) -> Optional[Pipeline]:
        """Create a pipeline from a template."""
        if template_id not in self.templates:
            logger.error(f"Template not found: {template_id}")
            return None

        template = self.templates[template_id]

        # Create new pipeline from template
        import uuid
        pipeline = Pipeline(
            id=str(uuid.uuid4())[:8],
            name=name or template.name,
            repository=repository,
            branch=overrides.get("branch", template.branch),
            triggers=overrides.get("triggers", template.triggers),
            jobs=template.jobs.copy(),
            environment={**template.environment, **overrides.get("environment", {})},
            secrets=template.secrets.copy(),
            description=template.description,
        )

        success = await self.create_pipeline(provider, pipeline)
        return pipeline if success else None

    async def update_pipeline(
        self,
        provider: str,
        pipeline: Pipeline,
    ) -> bool:
        """Update a pipeline."""
        if provider not in self.providers:
            return False

        pipeline.updated_at = datetime.utcnow()
        success = await self.providers[provider].update_pipeline(pipeline)
        if success:
            self.pipelines[pipeline.id] = pipeline
        return success

    async def delete_pipeline(
        self,
        provider: str,
        pipeline_id: str,
    ) -> bool:
        """Delete a pipeline."""
        if provider not in self.providers:
            return False

        success = await self.providers[provider].delete_pipeline(pipeline_id)
        if success and pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
        return success

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================

    async def trigger(
        self,
        provider: str,
        pipeline_id: str,
        branch: str = "main",
        variables: Dict[str, str] = None,
    ) -> Optional[PipelineRun]:
        """Trigger a pipeline run."""
        if provider not in self.providers:
            return None

        run = await self.providers[provider].trigger_pipeline(
            pipeline_id, branch, variables
        )
        if run:
            self.runs[run.id] = run
            logger.info(f"Triggered pipeline {pipeline_id} on {provider}, run: {run.id}")
        return run

    async def cancel(self, provider: str, run_id: str) -> bool:
        """Cancel a pipeline run."""
        if provider not in self.providers:
            return False

        success = await self.providers[provider].cancel_run(run_id)
        if success and run_id in self.runs:
            self.runs[run_id].status = PipelineStatus.CANCELLED
        return success

    async def retry(
        self,
        provider: str,
        run_id: str,
    ) -> Optional[PipelineRun]:
        """Retry a failed pipeline run."""
        if provider not in self.providers:
            return None

        run = await self.providers[provider].retry_run(run_id)
        if run:
            self.runs[run.id] = run
        return run

    async def get_status(
        self,
        provider: str,
        run_id: str,
    ) -> Optional[PipelineRun]:
        """Get current status of a pipeline run."""
        if provider not in self.providers:
            return None

        run = await self.providers[provider].get_run(run_id)
        if run:
            self.runs[run.id] = run
        return run

    async def wait_for_completion(
        self,
        provider: str,
        run_id: str,
        timeout: int = 3600,
        poll_interval: int = 30,
    ) -> Optional[PipelineRun]:
        """Wait for a pipeline run to complete."""
        start = datetime.utcnow()

        while (datetime.utcnow() - start).total_seconds() < timeout:
            run = await self.get_status(provider, run_id)
            if not run:
                return None

            if run.status in [
                PipelineStatus.SUCCESS,
                PipelineStatus.FAILED,
                PipelineStatus.CANCELLED,
            ]:
                return run

            await asyncio.sleep(poll_interval)

        logger.warning(f"Timeout waiting for pipeline run {run_id}")
        return await self.get_status(provider, run_id)

    # =========================================================================
    # LOGS AND ARTIFACTS
    # =========================================================================

    async def get_logs(
        self,
        provider: str,
        run_id: str,
        job_id: str = None,
    ) -> str:
        """Get logs for a pipeline run."""
        if provider not in self.providers:
            return ""

        return await self.providers[provider].get_logs(run_id, job_id)

    # =========================================================================
    # PIPELINE CONVERSION
    # =========================================================================

    def convert_pipeline(
        self,
        pipeline: Pipeline,
        target_format: str,
    ) -> str:
        """Convert pipeline to different CI/CD formats."""
        if target_format == "github-actions":
            return yaml.dump(
                pipeline.to_github_actions(),
                default_flow_style=False,
                sort_keys=False,
            )
        elif target_format == "gitlab-ci":
            return yaml.dump(
                pipeline.to_gitlab_ci(),
                default_flow_style=False,
                sort_keys=False,
            )
        else:
            raise ValueError(f"Unsupported format: {target_format}")

    def import_github_actions(self, workflow_yaml: str) -> Pipeline:
        """Import a GitHub Actions workflow as a Pipeline."""
        workflow = yaml.safe_load(workflow_yaml)

        import uuid
        pipeline = Pipeline(
            id=str(uuid.uuid4())[:8],
            name=workflow.get("name", "Imported Workflow"),
            repository="",
            triggers=[],
            jobs=[],
            environment=workflow.get("env", {}),
        )

        # Parse triggers
        on = workflow.get("on", {})
        if isinstance(on, list):
            on = {t: {} for t in on}

        if "push" in on:
            pipeline.triggers.append(PipelineTrigger.PUSH)
        if "pull_request" in on:
            pipeline.triggers.append(PipelineTrigger.PULL_REQUEST)
        if "schedule" in on:
            pipeline.triggers.append(PipelineTrigger.SCHEDULE)
            pipeline.schedule = on["schedule"][0].get("cron") if on["schedule"] else None
        if "workflow_dispatch" in on:
            pipeline.triggers.append(PipelineTrigger.WORKFLOW_DISPATCH)

        # Parse jobs
        for job_id, job_config in workflow.get("jobs", {}).items():
            job = PipelineJob(
                id=job_id,
                name=job_config.get("name", job_id),
                runs_on=job_config.get("runs-on", "ubuntu-latest"),
                needs=job_config.get("needs", []),
                environment=job_config.get("env", {}),
                timeout_minutes=job_config.get("timeout-minutes", 60),
                condition=job_config.get("if"),
            )

            if "strategy" in job_config and "matrix" in job_config["strategy"]:
                job.matrix = job_config["strategy"]["matrix"]

            for step in job_config.get("steps", []):
                if "run" in step:
                    job.steps.append(PipelineStep(
                        name=step.get("name", "Step"),
                        command=step["run"],
                        working_directory=step.get("working-directory", "."),
                        environment=step.get("env", {}),
                        timeout_minutes=step.get("timeout-minutes", 30),
                        continue_on_error=step.get("continue-on-error", False),
                        condition=step.get("if"),
                    ))

            pipeline.jobs.append(job)

        return pipeline

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_pipeline_stats(self, pipeline_id: str) -> Dict[str, Any]:
        """Get statistics for a pipeline."""
        pipeline_runs = [r for r in self.runs.values() if r.pipeline_id == pipeline_id]

        if not pipeline_runs:
            return {"total_runs": 0}

        successful = [r for r in pipeline_runs if r.status == PipelineStatus.SUCCESS]
        failed = [r for r in pipeline_runs if r.status == PipelineStatus.FAILED]

        durations = [
            r.duration.total_seconds()
            for r in pipeline_runs
            if r.duration
        ]

        return {
            "total_runs": len(pipeline_runs),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(pipeline_runs) * 100 if pipeline_runs else 0,
            "average_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "last_run": pipeline_runs[-1].to_dict() if pipeline_runs else None,
        }

    def get_recent_failures(self, limit: int = 10) -> List[PipelineRun]:
        """Get recent failed pipeline runs."""
        failed = [r for r in self.runs.values() if r.status == PipelineStatus.FAILED]
        return sorted(failed, key=lambda r: r.created_at, reverse=True)[:limit]


# Singleton instance
pipeline_manager = PipelineManager()
