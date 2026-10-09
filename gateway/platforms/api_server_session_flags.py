"""``PATCH /api/sessions/{session_id}`` — client-safe session metadata updates.

Split out of api_server.py (god-file cap). The cascade confirmation gate lives here:
archiving one session flips the whole compression lineage (#70185), so a multi-row
cascade is refused with 409 + the preview payload unless the caller passes
``confirm_cascade``. api_server-internal helpers are reached via the ``_api_server``
module object handed in by the delegator (origin imports siblings: lazy lookup keeps
``patch("...api_server.X")`` effective).
"""

import asyncio
from typing import Any

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - mirrors api_server's optional import
    web = None  # type: ignore[assignment]


async def _handle_patch_session(self, request: "web.Request", *, _api_server) -> "web.Response":
    """PATCH /api/sessions/{session_id} — update client-safe session metadata."""
    _error_response = _api_server._error_response
    session_id = request.match_info["session_id"]
    session, err = await self._get_existing_session_or_404(session_id)
    if err:
        return err
    body, err = await self._read_json_body(request)
    if err:
        return err
    # pinned/archived/unread are durable desktop-sidebar flags.
    unknown = sorted(set(body) - {
        "title", "end_reason", "pinned", "archived", "hidden", "unread", "confirm_cascade"})
    if unknown:
        return _error_response(
            f"Unsupported session fields: {', '.join(unknown)}", 400, code="unsupported_session_field")
    for flag in ("pinned", "archived", "hidden", "unread", "confirm_cascade"):
        if flag in body and not isinstance(body[flag], bool):
            return _error_response(f"'{flag}' must be a boolean", 400, code="invalid_session_field")
    db = await self._ensure_session_db_async()
    if db is None:
        return self._session_db_unavailable()
    if "title" in body:
        try:
            await asyncio.to_thread(
                db.set_session_title, session_id, "" if body["title"] is None else str(body["title"]))
        except ValueError as exc:
            return _error_response(str(exc), 400, code="invalid_title")
    # Archive cascades through the whole compression lineage (#70185): surface the blast
    # radius and demand an explicit confirm instead of hiding N rows silently.
    if body.get("archived") is True and not body.get("confirm_cascade"):
        preview = await asyncio.to_thread(db.preview_session_archive_lineage, session_id, True)
        if preview["cascade_count"] > 1:
            return web.json_response({
                "error": {
                    "message": "Archiving this session also archives its compression lineage "
                               f"({preview['cascade_count']} sessions); pass confirm_cascade=true",
                    "type": "invalid_request_error", "param": "archived",
                    "code": "archive_cascade_requires_confirmation",
                    "preview": preview,
                }}, status=409)
    # Pinned last: set_session_pinned clears hidden, so a pin in the same request
    # wins over an explicit hidden (same order as the dashboard's _RENAME_FLAG_SETTERS).
    for flag, setter in (("archived", db.set_session_archived), ("hidden", db.set_session_hidden),
                         ("pinned", db.set_session_pinned)):
        if flag in body:
            await asyncio.to_thread(setter, session_id, body[flag])
    if "unread" in body:
        await asyncio.to_thread(db.set_session_read, session_id, read=not body["unread"])
    if body.get("end_reason"):
        await asyncio.to_thread(db.end_session, session_id, str(body["end_reason"]))
    session = await asyncio.to_thread(db.get_session, session_id) or session
    return web.json_response({"object": "hermes.session", "session": self._session_response(session)})
