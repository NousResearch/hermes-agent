"""FastAPI adapter for the transport-independent StudyOS application module.

The dashboard plugin loader mounts this router at
``/api/plugins/study_os``.  Its bundled-only compatibility alias keeps the
existing ``/api/study`` interface available while clients migrate.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from plugins.study_os.application import (
    StudyApplicationError,
    StudyCommand,
    StudyOSApplication,
    StudyQuery,
)


router = APIRouter()
_application = StudyOSApplication()


def _query(operation: StudyQuery, **params: Any) -> dict[str, Any]:
    try:
        return _application.query(operation, **params)
    except StudyApplicationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _execute(operation: StudyCommand, **params: Any) -> dict[str, Any]:
    try:
        return _application.execute(operation, **params)
    except StudyApplicationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/projects")
async def list_study_projects():
    return _query(StudyQuery.PROJECTS)


class StudySettingsUpdate(BaseModel):
    vault_path: str


@router.get("/settings")
async def get_study_settings():
    return _query(StudyQuery.SETTINGS)


@router.put("/settings")
async def put_study_settings(body: StudySettingsUpdate):
    return _execute(StudyCommand.UPDATE_SETTINGS, vault_path=body.vault_path)


@router.put("/projects/{project_id}/active")
async def put_study_active_project(project_id: str):
    return _execute(StudyCommand.SELECT_PROJECT, project_id=project_id)


@router.get("/overview")
async def get_study_overview(
    project_id: Optional[str] = None,
    as_of: Optional[str] = None,
    review_limit: int = 10,
    intervention_limit: int = 5,
):
    return _query(
        StudyQuery.OVERVIEW,
        project_id=project_id,
        as_of=as_of,
        review_limit=review_limit,
        intervention_limit=intervention_limit,
    )


class StudyPlanProposalDecision(BaseModel):
    action: str
    decision_note: Optional[str] = None


@router.put("/projects/{project_id}/plan-proposals/{proposal_id}")
async def put_study_plan_proposal_decision(
    project_id: str,
    proposal_id: str,
    body: StudyPlanProposalDecision,
):
    return _execute(
        StudyCommand.DECIDE_PLAN_PROPOSAL,
        project_id=project_id,
        proposal_id=proposal_id,
        action=body.action,
        decision_note=body.decision_note,
    )


@router.get("/projects/{project_id}")
async def get_study_project(project_id: str):
    return _query(StudyQuery.PROJECT, project_id=project_id)


@router.get("/projects/{project_id}/schedules")
async def list_study_schedules(project_id: str):
    return _query(StudyQuery.SCHEDULES, project_id=project_id)


@router.get("/projects/{project_id}/schedules/{schedule_id}")
async def get_study_schedule(project_id: str, schedule_id: str):
    return _query(
        StudyQuery.SCHEDULE,
        project_id=project_id,
        schedule_id=schedule_id,
    )


@router.get("/review/due")
async def get_study_due_reviews(
    subject: str = "",
    level: Optional[int] = None,
    limit: int = 20,
):
    return _query(
        StudyQuery.REVIEW_DUE,
        subject=subject,
        level=level,
        limit=limit,
    )


class StudyReviewDetailRequest(BaseModel):
    note: str


class StudyReviewSubmissionRequest(BaseModel):
    project_id: str
    note: str
    response: str
    result: str
    duration_seconds: int
    self_confidence: int
    transfer_level: str = "execution"
    diagnoses: Optional[list[dict[str, Any]]] = None
    evaluator: Optional[dict[str, Any]] = None
    assistance: Optional[dict[str, Any]] = None
    detail: Optional[str] = None
    session_id: Optional[str] = None


@router.post("/review/detail")
async def post_study_review_detail(body: StudyReviewDetailRequest):
    return _query(StudyQuery.REVIEW_DETAIL, note=body.note)


@router.post("/review/attempt")
async def post_study_review_attempt(body: StudyReviewSubmissionRequest):
    return _execute(
        StudyCommand.SUBMIT_REVIEW,
        **body.model_dump(exclude_none=True),
    )


@router.get("/review/stats")
async def get_study_review_stats(rebuild: bool = False):
    return _query(StudyQuery.REVIEW_STATS, rebuild=rebuild)


@router.get("/review/queue")
async def get_study_review_queue(state: str = "", limit: int = 30):
    return _query(StudyQuery.REVIEW_QUEUE, state=state, limit=limit)


@router.get("/review/concepts")
async def get_study_concept_tree():
    return _query(StudyQuery.REVIEW_CONCEPTS)


@router.get("/profile")
async def get_study_profile():
    return _query(StudyQuery.PROFILE)


class StudyProfileUpdate(BaseModel):
    daily_review_limit: Optional[int] = None
    review_level_filter: Optional[int] = None
    subject_filter: Optional[str] = None


@router.put("/profile")
async def put_study_profile(body: StudyProfileUpdate):
    return _execute(
        StudyCommand.UPDATE_PROFILE,
        **body.model_dump(exclude_none=True),
    )
