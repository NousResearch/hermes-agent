"""RoomLink dispatch validation and hidden member-session ownership."""

import asyncio
import hashlib
import hmac
import time
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.platforms.api_server_room_grants import _json_error


async def _ensure_hosted_member_session(self, dispatch: Any) -> str:
    """Create or verify the target's canonical hidden group session. The ``Group: <room_id>``
    namespace is reused on purpose (Desktop-assisted -> hosted keeps one transcript); a
    conflicting title under another session id fails closed rather than merging."""
    db = await self._ensure_session_db_async()
    if db is None:
        raise RuntimeError("session database unavailable")
    title = f"Group: {dispatch.room_id}"
    seed = (
        f"{dispatch.home_install_id}\0{dispatch.room_id}\0"
        f"{dispatch.member_id}\0{dispatch.target_profile}")
    session_id = f"room_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

    def atomic(conn):
        row = conn.execute("SELECT id, title, source FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is not None:
            if row["title"] != title or row["source"] != "bot_room":
                raise RuntimeError("room session identity conflicts with existing data")
            return session_id
        clean_title = db.sanitize_title(title)
        conflict = conn.execute(
            "SELECT id FROM sessions WHERE title=? AND id!=?", (clean_title, session_id)).fetchone()
        if conflict:
            raise RuntimeError(
                "Another group already uses this room title on the target gateway. "
                "Rename or migrate that group before retrying.")
        conn.execute(
            "INSERT INTO sessions(id, source, title, hidden, started_at) VALUES(?, 'bot_room', ?, 1, ?)",
            (session_id, clean_title, time.time()))
        return session_id

    return await asyncio.to_thread(db._execute_write, atomic)


def _room_dispatch_error(exc: Exception, *, _openai_error) -> "web.Response":
    message, code = str(exc), "invalid_room_dispatch"
    lowered = message.lower()
    if "execution policy" in lowered or "remote room execution requires" in lowered:
        message = "Room execution policy changed; reauthorization is required."
        code = "room_execution_policy_changed"
    elif "capability catalog changed" in lowered:
        message = "Room capability catalog changed; reauthorization is required."
        code = "room_capability_catalog_changed"
    return _json_error(_openai_error, message, code=code, status=403)


async def _normalize_room_dispatch(
    self, request: "web.Request", body: Any, *, _api_server) -> tuple[Any, "web.Response | None"]:
    """Validate and normalize a scoped RoomLink dispatch request."""
    _openai_error, room_token = _api_server._openai_error, self._room_grant_token(request)
    if not room_token:
        return body, None
    if not isinstance(body, dict) or set(body) - {"input", "hosted_room_dispatch"}:
        return body, _json_error(
            _openai_error, "Room dispatch accepts only input and hosted_room_dispatch.",
            code="invalid_room_dispatch", status=400)
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import GatewayRoomCatalog, HostedMemberDispatch, verify_room_grant
        from gateway.hosted_room_execution_policy import RoomExecutionPolicy
        from gateway.platforms.api_server_room_grants import _local_room_catalog
        dispatch = HostedMemberDispatch.from_mapping(body.get("hosted_room_dispatch"))
        verify_room_grant(self._room_grant_secret(), room_token, dispatch, permission="dispatch")
        active_profile = _api_server._api_request_profile.get() or "default"
        local_install = hosted_rooms.local_authority_gateway_id()
        if dispatch.target_profile != active_profile or dispatch.target_install_id != local_install:
            raise ValueError("room dispatch target does not match this profile")
        _, catalog_map = _local_room_catalog(self, active_profile, local_install)
        catalog = GatewayRoomCatalog.from_mapping(catalog_map)
        policy = RoomExecutionPolicy.from_mapping(catalog.execution_policy.as_mapping())
        if not hmac.compare_digest(policy.policy_digest, dispatch.execution_policy_digest):
            raise ValueError("room execution policy changed")
        if not hmac.compare_digest(catalog.catalog_digest, dispatch.capability_digest):
            raise ValueError("room capability catalog changed")
        if body.get("input") not in {None, dispatch.prompt}:
            raise ValueError("room dispatch input does not match its prompt")
        expected_key = f"room:{dispatch.task_id}:{dispatch.execution_generation}"
        if request.headers.get("Idempotency-Key", "").strip() != expected_key:
            raise ValueError("room dispatch idempotency key is invalid")
        session_id = await self._ensure_hosted_member_session(dispatch)
        return {
            "input": dispatch.prompt,
            "session_id": session_id,
            "hosted_room_dispatch": dispatch.as_mapping(),
            "_room_execution_policy": policy.as_mapping(),
        }, None
    except Exception as exc:
        return body, _room_dispatch_error(exc, _openai_error=_openai_error)
