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


async def _ensure_hosted_member_session(self, dispatch: Any) -> str:
    """Create or verify the target's canonical hidden group session.

    Reusing the ``Group: <room_id>`` namespace is intentional: a room that
    moves from Desktop-assisted to hosted execution keeps one transcript.
    A conflicting title with a different session id fails closed instead
    of merging unrelated conversations.
    """
    db = await self._ensure_session_db_async()
    if db is None:
        raise RuntimeError("session database unavailable")
    title = f"Group: {dispatch.room_id}"
    seed = (
        f"{dispatch.home_install_id}\0{dispatch.room_id}\0"
        f"{dispatch.member_id}\0{dispatch.target_profile}"
    )
    session_id = f"room_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

    def ensure() -> str:
        def atomic(conn):
            row = conn.execute(
                "SELECT id, title, source FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            if row is not None:
                if row["title"] != title or row["source"] != "bot_room":
                    raise RuntimeError("room session identity conflicts with existing data")
                return session_id
            clean_title = db.sanitize_title(title)
            conflict = conn.execute(
                "SELECT id FROM sessions WHERE title=? AND id!=?",
                (clean_title, session_id),
            ).fetchone()
            if conflict:
                raise RuntimeError(
                    "Another group already uses this room title on the target gateway. "
                    "Rename or migrate that group before retrying."
                )
            conn.execute(
                "INSERT INTO sessions(id, source, title, hidden, started_at) "
                "VALUES(?, 'bot_room', ?, 1, ?)",
                (session_id, clean_title, time.time()),
            )
            return session_id

        return db._execute_write(atomic)

    return await asyncio.to_thread(ensure)


async def _normalize_room_dispatch(
    self,
    request: "web.Request",
    body: Any,
    *,
    _api_server,
) -> tuple[Any, "web.Response | None"]:
    """Validate and normalize a scoped RoomLink dispatch request."""
    _api_request_profile = _api_server._api_request_profile
    _openai_error = _api_server._openai_error

    room_token = self._room_grant_token(request)
    if not room_token:
        return body, None

    allowed_room_fields = {"input", "hosted_room_dispatch"}
    if not isinstance(body, dict) or set(body) - allowed_room_fields:
        return body, web.json_response(
            _openai_error(
                "Room dispatch accepts only input and hosted_room_dispatch.",
                code="invalid_room_dispatch",
            ),
            status=400,
        )
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import (
            GatewayRoomCatalog,
            HostedMemberDispatch,
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            catalog_mapping,
            verify_room_grant,
        )
        from gateway.hosted_room_execution_policy import (
            RoomExecutionPolicy,
            execution_policy_mapping,
        )

        dispatch = HostedMemberDispatch.from_mapping(
            body.get("hosted_room_dispatch")
        )
        verify_room_grant(
            self._room_grant_secret(),
            room_token,
            dispatch,
            permission="dispatch",
        )
        active_profile = _api_request_profile.get() or "default"
        local_install = hosted_rooms.local_authority_gateway_id()
        if (
            dispatch.target_profile != active_profile
            or dispatch.target_install_id != local_install
        ):
            raise ValueError("room dispatch target does not match this profile")
        with self._profile_scope(active_profile):
            execution_policy = execution_policy_mapping(
                target_profile=active_profile
            )
        catalog = GatewayRoomCatalog.from_mapping(
            catalog_mapping(
                installation_id=local_install,
                protocol_versions=(ROOM_LINK_PROTOCOL_VERSION,),
                link_modes=("direct",),
                persistent_process=True,
                text=True,
                attachments=False,
                target_profile=active_profile,
                execution_policy=execution_policy,
            )
        )
        policy = RoomExecutionPolicy.from_mapping(
            catalog.execution_policy.as_mapping()
        )
        if not hmac.compare_digest(
            policy.policy_digest,
            dispatch.execution_policy_digest,
        ):
            raise ValueError("room execution policy changed")
        if not hmac.compare_digest(
            catalog.catalog_digest,
            dispatch.capability_digest,
        ):
            raise ValueError("room capability catalog changed")
        supplied_input = body.get("input")
        if supplied_input not in {None, dispatch.prompt}:
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
        message = str(exc)
        lowered = message.lower()
        policy_changed = (
            "execution policy" in lowered
            or "remote room execution requires" in lowered
        )
        return body, web.json_response(
            _openai_error(
                (
                    "Room execution policy changed; reauthorization is required."
                    if policy_changed
                    else "Room capability catalog changed; reauthorization is required."
                    if "capability catalog changed" in lowered
                    else message
                ),
                code=(
                    "room_execution_policy_changed"
                    if policy_changed
                    else "room_capability_catalog_changed"
                    if "capability catalog changed" in lowered
                    else "invalid_room_dispatch"
                ),
            ),
            status=403,
        )
