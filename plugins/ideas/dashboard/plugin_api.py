"""Ideas dashboard plugin API — HTTP layer over :mod:`hermes_cli.ideas_db`."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from hermes_cli import ideas_db as db

router = APIRouter()


class IdeaBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = ""
    summary: Optional[str] = None
    status: str = db.DEFAULT_STATUS
    tags: list[str] = Field(default_factory=list)


class IdeaUpdateBody(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    body: Optional[str] = None
    summary: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[list[str]] = None
    task_id: Optional[str] = None


class ConvertBody(BaseModel):
    assignee: Optional[str] = None
    priority: int = 0
    triage: bool = True
    tenant: Optional[str] = None


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, db.IdeaNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, db.IdeasError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@router.get("/boards")
def list_boards():
    return db.list_boards()


@router.get("/ideas")
def list_ideas(
    board: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    include_archived: bool = Query(False),
):
    try:
        return db.list_ideas(
            board=board,
            status=status,
            q=q,
            tag=tag,
            include_archived=include_archived,
        )
    except db.IdeasError as exc:
        raise _http_error(exc)


@router.post("/ideas")
def create_idea(payload: IdeaBody, board: Optional[str] = Query(None)):
    try:
        idea = db.create_idea(
            title=payload.title,
            body=payload.body,
            summary=payload.summary,
            status=payload.status,
            tags=payload.tags,
            board=board,
        )
    except db.IdeasError as exc:
        raise _http_error(exc)
    return {"idea": idea}


@router.get("/ideas/{idea_id}")
def get_idea(idea_id: str):
    try:
        return {"idea": db.get_idea(idea_id)}
    except db.IdeaNotFoundError as exc:
        raise _http_error(exc)


@router.put("/ideas/{idea_id}")
def update_idea(idea_id: str, payload: IdeaUpdateBody):
    try:
        idea = db.update_idea(
            idea_id,
            title=payload.title,
            body=payload.body,
            summary=payload.summary,
            status=payload.status,
            tags=payload.tags,
            task_id=payload.task_id,
        )
    except db.IdeasError as exc:
        raise _http_error(exc)
    return {"idea": idea}


@router.delete("/ideas/{idea_id}")
def delete_idea(idea_id: str, delete_file: bool = Query(True)):
    try:
        db.delete_idea(idea_id, delete_file=delete_file)
    except db.IdeaNotFoundError as exc:
        raise _http_error(exc)
    return {"ok": True}


@router.post("/ideas/{idea_id}/duplicate")
def duplicate_idea(idea_id: str):
    try:
        idea = db.duplicate_idea(idea_id)
    except db.IdeasError as exc:
        raise _http_error(exc)
    return {"idea": idea}


@router.post("/ideas/{idea_id}/task")
def convert_to_task(idea_id: str, payload: ConvertBody):
    try:
        return db.convert_to_task(
            idea_id,
            assignee=payload.assignee,
            priority=payload.priority,
            triage=payload.triage,
            tenant=payload.tenant,
        )
    except db.IdeasError as exc:
        raise _http_error(exc)
