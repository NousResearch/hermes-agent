"""Roadmaps dashboard plugin — backend REST API routes.

Mounted at /api/plugins/roadmaps/ by the dashboard plugin system, exactly like
the Kanban plugin (``plugins/kanban/dashboard/plugin_api.py``). This layer is
intentionally thin: every handler is a small wrapper around
``hermes_cli.roadmaps_service.RoadmapsService`` (reads) and
``hermes_cli.roadmaps_writer.RoadmapsWriter`` (versioned writes), so the REST
surface, the gateway RPC surface (``tui_gateway/methods_roadmaps.py``) and the
agent toolset (``tools/roadmaps_tools.py``) cannot drift — they all call the
same service/writer methods.

Security note
-------------
Plugin HTTP routes go through the dashboard's session-token auth middleware
(``web_server.auth_middleware``) just like core API routes, so every
``/api/plugins/...`` request must present the session bearer token (or the
session cookie). We do not re-implement auth here; we rely on the dashboard.

Error mapping (mirrors the RPC structured codes, mapped to clean HTTP):
  5063 validation/scope  -> 422
  5064 stale version      -> 409
  5065 not found          -> 404
  5066 invalid transition -> 422
  5067 conflict           -> 409
  5061 unavailable        -> 503
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hermes_cli import profiles as _profiles
from hermes_cli.roadmaps_service import RoadmapsService, RoadmapsUnavailable
from hermes_cli.roadmaps_writer import (
    InvalidRoadmapPlanTransitionError,
    InvalidRoadmapTodoTransitionError,
    InvalidRoadmapTransitionError,
    RoadmapExistsError,
    RoadmapNodeNotFoundError,
    RoadmapNotFoundError,
    RoadmapProjectNotFoundError,
    RoadmapTodoNotFoundError,
    RoadmapVersionExistsError,
    RoadmapVersionNotFoundError,
    RoadmapsWriteError,
    RoadmapsWriter,
    StaleRoadmapVersionError,
)

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Profile -> projects.db resolution (mirrors tui_gateway/server.py:_profile_home)
# ---------------------------------------------------------------------------

def _profile_db_path(profile: str) -> Path:
    """Resolve a validated profile name to its ``projects.db`` path.

    ``default`` (or the launch profile) resolves to the process HERMES_HOME;
    a named profile resolves to ``profiles/<name>/projects.db``. A profile
    whose home does not exist is rejected (never silently seeds a fresh DB
    there — the WRITE path would otherwise create one under a traversal path).
    """
    _profiles.validate_profile_name(profile)
    home = Path(_profiles.get_profile_dir(profile))
    launch = Path(_profiles._get_default_hermes_home())
    if home.resolve() == launch.resolve():
        return launch / "projects.db"
    if (home / "state.db").exists() or home.exists():
        return home / "projects.db"
    raise ValueError(f"profile scope unavailable: {profile!r}")


def _service(profile: str) -> RoadmapsService:
    return RoadmapsService(_profile_db_path(profile))


def _writer(profile: str) -> RoadmapsWriter:
    return RoadmapsWriter(_profile_db_path(profile))


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

def _http_error(exc: Exception, operation: str) -> HTTPException:
    """Map a writer/service exception to a clean HTTP error.

    The raw backend message is never surfaced; callers get a stable,
    designed message plus the structured code. Unexpected exceptions are
    logged server-side and returned as a generic 503.
    """
    from hermes_cli.roadmaps_writer import RoadmapsWriteError  # noqa: F401 (re-import for clarity)

    if isinstance(exc, StaleRoadmapVersionError):
        return HTTPException(status_code=409, detail={"code": 5064, "message": "roadmap version is stale"})
    if isinstance(exc, (
        RoadmapNotFoundError,
        RoadmapProjectNotFoundError,
        RoadmapNodeNotFoundError,
        RoadmapTodoNotFoundError,
        RoadmapVersionNotFoundError,
    )):
        return HTTPException(status_code=404, detail={"code": 5065, "message": "roadmap scope not found"})
    if isinstance(exc, (
        InvalidRoadmapTransitionError,
        InvalidRoadmapTodoTransitionError,
        InvalidRoadmapPlanTransitionError,
    )):
        return HTTPException(status_code=422, detail={"code": 5066, "message": "invalid roadmap transition"})
    if isinstance(exc, (RoadmapExistsError, RoadmapVersionExistsError)):
        return HTTPException(status_code=409, detail={"code": 5067, "message": "roadmap conflict"})
    if isinstance(exc, (RoadmapsWriteError, RoadmapsUnavailable)):
        return HTTPException(status_code=503, detail={"code": 5061, "message": "roadmaps unavailable"})
    log.exception("roadmaps REST %s failed", operation)
    return HTTPException(status_code=503, detail={"code": 5061, "message": "roadmaps unavailable"})


def _validation_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail={"code": 5063, "message": str(exc)})


# ---------------------------------------------------------------------------
# Request helpers (mirror the RPC contract exactly)
# ---------------------------------------------------------------------------

def _required(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} required")
    if len(value) > 128:
        raise ValueError(f"{name} must be at most 128 characters")
    return value


def _expected(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected_version must be an integer")
    return value


def _scope_project(profile: Any, project_id: Any) -> tuple[str, str]:
    """Profile + project, both required."""
    if profile is None:
        raise ValueError("profile_id required")
    p = _required(profile, "profile_id")
    project = _required(project_id, "project_id")
    # Resolve early so an unknown profile fails closed before touching the DB.
    _profile_db_path(p)
    return p, project


def _scope_roadmap(profile: Any, project_id: Any, roadmap_id: Any) -> tuple[str, str, str]:
    """Profile + project + roadmap_id, all required."""
    p, project = _scope_project(profile, project_id)
    rid = _required(roadmap_id, "roadmap_id")
    return p, project, rid


# Pydantic models for the JSON bodies we accept.
class _CreateBody(BaseModel):
    actor: str = Field(...)
    title: str | None = None
    purpose: str | None = None
    roadmap_id: str | None = None


class _UpdateBody(BaseModel):
    actor: str = Field(...)
    expected_version: int | None = None
    title: str | None = None
    purpose: str | None = None


class _ArchiveBody(BaseModel):
    actor: str = Field(...)
    expected_version: int | None = None


class _PlanBody(BaseModel):
    actor: str = Field(...)
    version: int | None = None
    nodes: list | None = None
    relations: list | None = None
    todos: list | None = None
    source: str | None = None
    reason: str | None = None


class _PlanTransitionBody(BaseModel):
    actor: str = Field(...)
    expected_version: int | None = None


class _NodeBody(BaseModel):
    actor: str = Field(...)
    expected_version: int | None = None
    progress: int | None = None
    reason: str | None = None


class _TodoBody(BaseModel):
    actor: str = Field(...)
    expected_version: int | None = None
    state: str = Field(...)


class _AttachBody(BaseModel):
    actor: str = Field(...)
    stored_session_id: str = Field(...)
    expected_version: int | None = None
    kind: str = "vision"
    plan_version: int | None = None


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------

@router.get("/roadmaps")
def list_roadmaps(profile: str, project_id: str):
    try:
        p, project = _scope_project(profile, project_id)
        return _service(p).list(p, project)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "list") from exc


@router.get("/roadmaps/{roadmap_id}")
def get_roadmap(roadmap_id: str, profile: str, project_id: str):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _service(p).get(p, project, rid)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "get") from exc


@router.get("/roadmaps/{roadmap_id}/snapshot")
def snapshot(roadmap_id: str, profile: str, project_id: str):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _service(p).snapshot(p, project, rid)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "snapshot") from exc


@router.get("/roadmaps/{roadmap_id}/plans")
def list_plans(roadmap_id: str, profile: str, project_id: str):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _service(p).list_plans(p, project, rid)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "plans.list") from exc


@router.get("/roadmaps/{roadmap_id}/plans/{version}")
def get_plan(roadmap_id: str, version: int, profile: str, project_id: str):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("version must be an integer")
        return _service(p).get_plan(p, project, rid, version)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "plans.get") from exc


@router.get("/roadmaps/{roadmap_id}/sessions")
def list_sessions(roadmap_id: str, profile: str, project_id: str):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _service(p).list_sessions(p, project, rid)
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "sessions") from exc


@router.get("/planning-rules")
def planning_rules(version: str | None = None):
    try:
        if version is not None and not isinstance(version, str):
            raise ValueError("version must be a string")
        from hermes_cli.roadmaps_planning_rules import get_planning_rules

        rules = get_planning_rules(version)
        return {"version": rules["version"], "rules": rules}
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "planning_rules") from exc


# ---------------------------------------------------------------------------
# CRUD + plan governance mutations
# ---------------------------------------------------------------------------

@router.post("/roadmaps")
def create_roadmap(profile: str, project_id: str, body: _CreateBody):
    try:
        p, project = _scope_project(profile, project_id)
        return _writer(p).create_roadmap(
            p, project, body.title, _required(body.actor, "actor"),
            roadmap_id=body.roadmap_id, purpose=body.purpose,
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "roadmaps.create") from exc


@router.patch("/roadmaps/{roadmap_id}")
def update_roadmap(roadmap_id: str, profile: str, project_id: str, body: _UpdateBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _writer(p).update_roadmap(
            p, project, rid, _required(body.actor, "actor"), _expected(body.expected_version),
            title=body.title, purpose=body.purpose,
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "roadmaps.update") from exc


@router.post("/roadmaps/{roadmap_id}/archive")
def archive_roadmap(roadmap_id: str, profile: str, project_id: str, body: _ArchiveBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _writer(p).archive_roadmap(
            p, project, rid, _required(body.actor, "actor"), _expected(body.expected_version),
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "roadmaps.archive") from exc


@router.post("/roadmaps/{roadmap_id}/plans")
def create_plan(roadmap_id: str, profile: str, project_id: str, body: _PlanBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _writer(p).create_plan(
            p, project, rid, _required(body.actor, "actor"),
            version=body.version, nodes=body.nodes, relations=body.relations,
            todos=body.todos, source=body.source, reason=body.reason,
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "plans.create") from exc


@router.post("/roadmaps/{roadmap_id}/plans/{version}/validate")
def validate_plan(roadmap_id: str, version: int, profile: str, project_id: str, body: _PlanTransitionBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _writer(p).validate_plan(
            p, project, rid, version, _required(body.actor, "actor"), _expected(body.expected_version),
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "plans.validate") from exc


@router.post("/roadmaps/{roadmap_id}/plans/{version}/activate")
def activate_plan(roadmap_id: str, version: int, profile: str, project_id: str, body: _PlanTransitionBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        return _writer(p).activate_plan(
            p, project, rid, version, _required(body.actor, "actor"), _expected(body.expected_version),
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "plans.activate") from exc


# ---------------------------------------------------------------------------
# Node execution mutations
# ---------------------------------------------------------------------------

@router.post("/roadmaps/{roadmap_id}/nodes/{node_id}/claim")
def claim_node(roadmap_id: str, node_id: str, profile: str, project_id: str, body: _NodeBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        nid = _required(node_id, "node_id")
        return _writer(p).claim_node(p, project, rid, nid, _required(body.actor, "actor"), _expected(body.expected_version))
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "claim_node") from exc


@router.post("/roadmaps/{roadmap_id}/nodes/{node_id}/advance")
def advance_node(roadmap_id: str, node_id: str, profile: str, project_id: str, body: _NodeBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        nid = _required(node_id, "node_id")
        return _writer(p).advance_node(p, project, rid, nid, _required(body.actor, "actor"), _expected(body.expected_version))
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "advance_node") from exc


@router.post("/roadmaps/{roadmap_id}/nodes/{node_id}/progress")
def update_progress(roadmap_id: str, node_id: str, profile: str, project_id: str, body: _NodeBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        nid = _required(node_id, "node_id")
        progress = body.progress
        if isinstance(progress, bool) or not isinstance(progress, int):
            raise ValueError("progress must be an integer")
        return _writer(p).update_progress(
            p, project, rid, nid, _required(body.actor, "actor"), progress, _expected(body.expected_version),
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "update_progress") from exc


@router.post("/roadmaps/{roadmap_id}/nodes/{node_id}/complete")
def complete_node(roadmap_id: str, node_id: str, profile: str, project_id: str, body: _NodeBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        nid = _required(node_id, "node_id")
        return _writer(p).complete_node(p, project, rid, nid, _required(body.actor, "actor"), _expected(body.expected_version))
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "complete_node") from exc


@router.post("/roadmaps/{roadmap_id}/nodes/{node_id}/block")
def block_node(roadmap_id: str, node_id: str, profile: str, project_id: str, body: _NodeBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        nid = _required(node_id, "node_id")
        reason = body.reason
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason required")
        return _writer(p).block_node(
            p, project, rid, nid, _required(body.actor, "actor"), reason.strip(), _expected(body.expected_version),
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "block_node") from exc


@router.post("/roadmaps/{roadmap_id}/nodes/{node_id}/unblock")
def unblock_node(roadmap_id: str, node_id: str, profile: str, project_id: str, body: _NodeBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        nid = _required(node_id, "node_id")
        return _writer(p).unblock_node(p, project, rid, nid, _required(body.actor, "actor"), _expected(body.expected_version))
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "unblock_node") from exc


@router.post("/roadmaps/{roadmap_id}/todos/{todo_id}")
def update_todo(roadmap_id: str, todo_id: str, profile: str, project_id: str, body: _TodoBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        tid = _required(todo_id, "todo_id")
        state = _required(body.state, "state")
        return _writer(p).update_todo(
            p, project, rid, tid, _required(body.actor, "actor"), state, _expected(body.expected_version),
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "update_todo") from exc


@router.post("/roadmaps/{roadmap_id}/sessions")
def attach_session(roadmap_id: str, profile: str, project_id: str, body: _AttachBody):
    try:
        p, project, rid = _scope_roadmap(profile, project_id, roadmap_id)
        if "runtime_session_id" in body.model_fields_set and body.model_dump().get("runtime_session_id") is not None:
            raise ValueError("runtime_session_id is not accepted")
        return _writer(p).attach_session(
            p, project, rid,
            _required(body.stored_session_id, "stored_session_id"),
            _required(body.actor, "actor"),
            _expected(body.expected_version),
            kind=body.kind,
            plan_version=body.plan_version,
        )
    except ValueError as exc:
        raise _validation_error(exc) from exc
    except Exception as exc:
        raise _http_error(exc, "attach_session") from exc
