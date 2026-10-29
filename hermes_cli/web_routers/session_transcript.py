"""Search and exact match reads for one stored conversation."""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from hermes_state_transcript_search import get_transcript_match_window, search_transcript

router = APIRouter()


@router.get("/api/sessions/{session_id}/messages/search")
async def search_session_transcript(
    session_id: str, q: str = "", profile: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0),
):
    from hermes_cli.web_routers.sessions import _serving_profile, _timeline_session_id, _with_db

    owner = _serving_profile(profile)

    def read(db):
        sid = _timeline_session_id(db, session_id, owner)
        return {"session_id": sid, "profile": owner, **search_transcript(db, sid, q, limit=limit, offset=offset)}

    return await asyncio.to_thread(_with_db, profile, read, read_only=True)


@router.get("/api/sessions/{session_id}/messages/match")
async def get_session_match_window(
    session_id: str, row_id: int = Query(..., ge=1), profile: Optional[str] = None,
    limit: int = Query(120, ge=1, le=120),
):
    from hermes_cli.web_routers.sessions import (
        _history_profile_home, _project_for_display, _serving_profile, _timeline_session_id, _with_db,
    )

    owner = _serving_profile(profile)

    def read(db):
        sid = _timeline_session_id(db, session_id, owner)
        page = get_transcript_match_window(db, sid, row_id, limit=limit)
        if page is None:
            raise HTTPException(status_code=404, detail="Message not found")
        return {"session_id": sid, "profile": owner, **page}

    result = await asyncio.to_thread(_with_db, profile, read, read_only=True)
    result["messages"] = await asyncio.to_thread(
        _project_for_display, result["messages"], home=_history_profile_home(profile))
    return result
