"""Which session the dashboard's ``/chat`` PTY is on (``GET /api/chat/active-session``).

The chat tab drives the Ink TUI through a PTY, so a FRESH chat's session id never reaches the
SPA: the TUI writes it to the per-channel marker file (``HERMES_TUI_ACTIVE_SESSION_FILE``) and
only a resume replay announces it over the socket (``chat_ws.pty_ws``). Actions that have to read
the transcript without typing into the composer — "copy last response" — ask here instead of
synthesising keystrokes into a composer that may hold an unsent draft (#149751).

Read-only: an unknown channel answers ``null`` rather than minting the marker. The PTY websocket
owns that file's lifetime (``_active_session_file_for_channel``), and a GET has no business
creating state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.web_deps import late
from hermes_cli.web_routers.chat_ws import _VALID_CHANNEL_RE, _read_active_session_file

router = APIRouter()

_active_session_files = late("_get_pty_active_session_files", "hermes_cli.web_server")


@router.get("/api/chat/active-session")
async def get_active_chat_session(request: Request, channel: str = ""):
    """The session id this chat channel is on, or ``null`` before its first turn."""
    chan = (channel or "").strip()
    if not _VALID_CHANNEL_RE.match(chan):
        raise HTTPException(status_code=400, detail="invalid channel")
    path: Optional[Path] = _active_session_files(request.app).get(chan)
    return {"session_id": _read_active_session_file(path) if path is not None else None}
