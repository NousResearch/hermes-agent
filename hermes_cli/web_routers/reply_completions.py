"""Profile-scoped replay of committed replies and report deliveries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response

from gateway.response_filters import is_intentional_silence_agent_result
from hermes_cli.dashboard_auth import TokenPrincipal
from hermes_cli.dashboard_auth.token_auth import register_token_route
from hermes_cli.web_routers.sessions import _with_db
from persistence.core.errors import AuthorityRefusal

_REPLY_FEED_PATH = "/api/sessions/reply-completions"
_PROFILE_PATTERN = r"[a-z0-9][a-z0-9_-]{0,63}"
_TOKEN_PATTERN = r"[A-Za-z0-9_-]{43}"
_GRANTS_FILE_ENV = "HERMES_REPLY_FEED_GRANTS_FILE"
_MAX_GRANTS_BYTES = 64 * 1024


def _verify_reply_token(token: str) -> TokenPrincipal | None:
    """Read the operator-provisioned grants each request so deletion or rotation revokes access."""
    filename = os.environ.get(_GRANTS_FILE_ENV, "").strip()
    if not filename or not re.fullmatch(_TOKEN_PATTERN, token):
        return None
    path = Path(filename)
    if not path.is_absolute() or not path.is_file():
        return None
    try:
        with path.open("rb") as file:
            raw = file.read(_MAX_GRANTS_BYTES + 1)
        if len(raw) > _MAX_GRANTS_BYTES:
            return None
        grants = json.loads(raw)
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(grants, dict):
        return None
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    grant = grants.get(digest)
    if not isinstance(grant, dict):
        return None
    profile = grant.get("profile")
    expires_at = grant.get("expires_at")
    if not isinstance(profile, str) or not re.fullmatch(_PROFILE_PATTERN, profile):
        return None
    if type(expires_at) is not int or expires_at <= time.time():
        return None
    return TokenPrincipal(principal="reply-feed", provider="reply-feed-file", scopes=(profile,))


def register_auth() -> None:
    register_token_route(_REPLY_FEED_PATH, verifier=_verify_reply_token)

router = APIRouter()


def _visible_reply(store, event):
    if event.kind == "note":
        return True
    try:
        saved = store.admissions.settled_result(admission_id=event.admission_id)
    except AuthorityRefusal as exc:
        if exc.reason == "not_found":  # The admission was retired while this page was being read.
            return False
        raise
    if saved is None or not saved.result:
        return False
    result = saved.result.get("result", {})
    response = result.get("final_response") if isinstance(result, dict) else None
    return bool(isinstance(response, str) and response.strip()
                and not is_intentional_silence_agent_result(result, response))


def _events_page(store, after, limit):
    events = store.admissions.reply_completions(after=after, limit=limit)
    replies = []
    for event in events:
        target = store.lineage.resolve_resume_target(event.session_id)
        session = store.sessions.get(target)
        if (session is None or session.hidden or session.archived
                or (session.last_read_at or 0) >= event.completed_at or not _visible_reply(store, event)):
            continue
        lineage = store.lineage.resume_lineage_ids(target)
        replies.append({
            "sequence": event.sequence, "session_id": target, "kind": event.kind,
            "conversation_id": lineage[0] if lineage else target,
            "completed_at": event.completed_at,
        })
    # Suppressed or already-read events still advance the cursor.
    return {
        "events": replies,
        "next_cursor": events[-1].sequence if events else after,
        "has_more": len(events) == limit,
    }


@router.get(_REPLY_FEED_PATH)
async def get_reply_completions(
    request: Request,
    response: Response,
    profile: str = Query(..., pattern=f"^{_PROFILE_PATTERN}$", max_length=64),
    after: int = Query(0, ge=0, lt=2**63),
    limit: int = Query(50, ge=1, le=100),
):
    """Read the profile authorized by this bearer grant."""
    principal = getattr(request.state, "token_principal", None)
    if (principal is None or principal.provider != "reply-feed-file"
            or profile not in principal.scopes
            or request.query_params.getlist("profile") != [profile]):
        raise HTTPException(status_code=403, detail="Profile outside reply feed grant")
    page = await asyncio.to_thread(
        _with_db, profile, lambda store: _events_page(store, after, limit), read_only=True,
    )
    response.headers["Cache-Control"] = "no-store"
    return {"profile": profile, **page}
