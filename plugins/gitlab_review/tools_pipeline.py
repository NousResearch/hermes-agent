"""GitLab CI/CD Pipeline tools for the gitlab-review plugin.

Registers 3 pipeline-related tools:

- gitlab_mr_pipelines  — Check CI/CD pipeline status for an MR
- gitlab_pipeline_jobs — Get job details and logs for a pipeline
- gitlab_pipeline_retry — Retry a failed pipeline
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from plugins.gitlab_review.gitlab_client import (
    GitLabAPIError,
    gitlab_get,
    gitlab_get_paginated,
    gitlab_post,
    is_available,
    project_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool: gitlab_mr_pipelines
# ---------------------------------------------------------------------------

GITLAB_MR_PIPELINES_SCHEMA = {
    "name": "gitlab_mr_pipelines",
    "description": (
        "Check CI/CD pipeline status for a GitLab Merge Request's latest "
        "commit. Returns pipeline IDs, status, and ref for each pipeline "
        "associated with the MR."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project path (e.g. 'group/project') or numeric project ID.",
            },
            "mr_iid": {
                "type": "integer",
                "description": "The MR internal ID (iid).",
            },
        },
        "required": ["project", "mr_iid"],
    },
}


def _handle_mr_pipelines(args: dict, **kw) -> str:
    """Fetch pipeline status for an MR."""
    project = args.get("project", "")
    mr_iid = args.get("mr_iid")
    if not project or mr_iid is None:
        return _error("Missing required parameters: project and mr_iid")

    try:
        path = f"{project_path(project)}/merge_requests/{mr_iid}/pipelines"
        items = gitlab_get_paginated(path, max_pages=3)

        pipelines = []
        for p in items:
            pipelines.append({
                "id": p.get("id"),
                "sha": p.get("sha"),
                "ref": p.get("ref"),
                "status": p.get("status"),
                "source": p.get("source"),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at"),
                "web_url": p.get("web_url"),
            })

        return json.dumps({"result": {"count": len(pipelines), "pipelines": pipelines}})
    except GitLabAPIError as e:
        return _error(f"Failed to fetch pipelines: {e}")
    except Exception as e:
        logger.error("gitlab_mr_pipelines error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Tool: gitlab_pipeline_jobs
# ---------------------------------------------------------------------------

GITLAB_PIPELINE_JOBS_SCHEMA = {
    "name": "gitlab_pipeline_jobs",
    "description": (
        "Get job details and log excerpts for a GitLab CI/CD pipeline. "
        "Returns job names, statuses, stages, and the last 200 lines of "
        "trace (log) for failed jobs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project path (e.g. 'group/project') or numeric project ID.",
            },
            "pipeline_id": {
                "type": "integer",
                "description": "Pipeline ID (get from gitlab_mr_pipelines output).",
            },
            "include_traces": {
                "type": "boolean",
                "description": "Include job trace (log) for failed jobs. Default: true.",
            },
            "trace_lines": {
                "type": "integer",
                "description": "Number of last lines to include from the trace. Default: 200.",
            },
        },
        "required": ["project", "pipeline_id"],
    },
}


def _handle_pipeline_jobs(args: dict, **kw) -> str:
    """Fetch job details and optional traces for a pipeline."""
    project = args.get("project", "")
    pipeline_id = args.get("pipeline_id")
    include_traces = args.get("include_traces", True)
    trace_lines = min(int(args.get("trace_lines", 200)), 2000)

    if not project or pipeline_id is None:
        return _error("Missing required parameters: project and pipeline_id")

    try:
        path = f"{project_path(project)}/pipelines/{pipeline_id}/jobs"
        items = gitlab_get_paginated(path, max_pages=5)

        jobs = []
        for j in items:
            job_entry = {
                "id": j.get("id"),
                "name": j.get("name"),
                "stage": j.get("stage"),
                "status": j.get("status"),
                "allow_failure": j.get("allow_failure", False),
                "created_at": j.get("created_at"),
                "started_at": j.get("started_at"),
                "finished_at": j.get("finished_at"),
                "runner": (j.get("runner") or {}).get("description", ""),
                "web_url": j.get("web_url"),
            }

            # Fetch trace for failed jobs if requested
            if include_traces and j.get("status") in ("failed", "running"):
                try:
                    trace_path = f"{project_path(project)}/jobs/{j['id']}/trace"
                    trace = _fetch_trace(trace_path)
                    if trace:
                        # Trim to last N lines
                        lines = trace.splitlines()
                        if len(lines) > trace_lines:
                            trace = "\n".join(lines[-trace_lines:])
                        job_entry["trace"] = trace
                except Exception as e:
                    logger.debug("Failed to fetch trace for job %s: %s", j.get("id"), e)

            jobs.append(job_entry)

        return json.dumps({"result": {"count": len(jobs), "jobs": jobs}})
    except GitLabAPIError as e:
        return _error(f"Failed to fetch pipeline jobs: {e}")
    except Exception as e:
        logger.error("gitlab_pipeline_jobs error: %s", e)
        return _error(f"Unexpected error: {e}")


def _fetch_trace(trace_path: str) -> str:
    """Fetch a job trace (log) from the GitLab API.

    The trace endpoint returns plain text, not JSON, so we use a raw
    httpx request instead of the JSON-parsing client.
    """
    import httpx
    import os

    from plugins.gitlab_review.gitlab_client import api_url, get_config

    base_url, token = get_config()
    url = api_url(base_url, trace_path)
    headers = {"PRIVATE-TOKEN": token}

    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        return ""


# ---------------------------------------------------------------------------
# Tool: gitlab_pipeline_retry
# ---------------------------------------------------------------------------

GITLAB_PIPELINE_RETRY_SCHEMA = {
    "name": "gitlab_pipeline_retry",
    "description": (
        "Retry a failed GitLab CI/CD pipeline. All failed jobs in the "
        "pipeline will be retried."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project path (e.g. 'group/project') or numeric project ID.",
            },
            "pipeline_id": {
                "type": "integer",
                "description": "Pipeline ID to retry (get from gitlab_mr_pipelines).",
            },
        },
        "required": ["project", "pipeline_id"],
    },
}


def _handle_pipeline_retry(args: dict, **kw) -> str:
    """Retry a failed pipeline."""
    project = args.get("project", "")
    pipeline_id = args.get("pipeline_id")
    if not project or pipeline_id is None:
        return _error("Missing required parameters: project and pipeline_id")

    try:
        path = f"{project_path(project)}/pipelines/{pipeline_id}/retry"
        result = gitlab_post(path)
        return json.dumps({
            "result": {
                "id": result.get("id"),
                "status": result.get("status"),
                "sha": result.get("sha"),
                "ref": result.get("ref"),
                "web_url": result.get("web_url"),
            },
        })
    except GitLabAPIError as e:
        return _error(f"Failed to retry pipeline: {e}")
    except Exception as e:
        logger.error("gitlab_pipeline_retry error: %s", e)
        return _error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg: str) -> str:
    """Return a JSON error result."""
    return json.dumps({"error": msg})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ALL_PIPELINE_SCHEMAS = [
    (GITLAB_MR_PIPELINES_SCHEMA, _handle_mr_pipelines, "🔄"),
    (GITLAB_PIPELINE_JOBS_SCHEMA, _handle_pipeline_jobs, "⚙️"),
    (GITLAB_PIPELINE_RETRY_SCHEMA, _handle_pipeline_retry, "🔁"),
]


def register_pipeline_tools(ctx) -> None:
    """Register all pipeline-related tools with the plugin context."""
    for schema, handler, emoji in ALL_PIPELINE_SCHEMAS:
        ctx.register_tool(
            name=schema["name"],
            toolset="gitlab_review",
            schema=schema,
            handler=handler,
            check_fn=is_available,
            requires_env=["GITLAB_TOKEN"],
            emoji=emoji,
        )
