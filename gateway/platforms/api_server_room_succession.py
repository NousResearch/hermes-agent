"""Group Chat succession endpoints between a group's computers, authenticated by room identity keys.

``POST /v1/room-members/succession/fence`` fences the group's old epoch here and promises the next
one to the computer continuing it (``gateway/hosted_room_succession_backup.py``). ``.../learn``
takes a successor's verified transition, ``.../query`` tells any configured computer whom this copy
follows (backups use it as the host's heartbeat), ``.../report`` keeps a returning host's evidence
at the successor, and ``.../decision`` carries the owner's choice after ``continued_on_two``.
Requests are signed with the caller's pinned room identity key; fence and learn replies are sealed
to the caller, because they carry run evidence and continuation grants. No room grant is involved:
the old host's grants may be gone, and a successor never had one here. Calls use the default profile.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

MAX_REQUEST_BYTES = {"fence": 64 * 1024, "learn": 2 * 1024 * 1024, "query": 16 * 1024, "report": 512 * 1024,
                     "decision": 1024 * 1024, "handover": 64 * 1024}


def continuation_minter(adapter, custody_db, *, replace_same_epoch: bool):
    """Mint the grants this installation's operator consented to, for a successor at its epoch.

    ``replace_same_epoch`` is for a verified authority only: it releases another computer's
    reservation at that epoch so the verified one's grant becomes current.
    """
    from gateway import hosted_rooms as rooms
    from gateway import hosted_room_succession as succession
    from gateway.platforms.api_server_room_grants import _issue_invitation

    def mint(room_id: str, successor: str, epoch: int) -> list[dict]:
        with rooms._transaction(custody_db, immediate=True) as conn:
            consents = succession.consents_locked(conn, room_id)
            if replace_same_epoch:
                for consent in consents:
                    conn.execute("""DELETE FROM hosted_room_peer_reservations WHERE room_id=? AND member_id=?
                        AND target_profile=? AND authority_epoch=? AND authority_gateway_id!=?""",
                        (room_id, consent["member_id"], consent["target_profile"], epoch, successor))
        grants = []
        for consent in consents:
            options = consent["options"]
            invitation = _issue_invitation(adapter, {
                "room_id": room_id, "home_install_id": successor, "authority_gateway_id": successor,
                "authority_epoch": epoch, "member_id": consent["member_id"],
                "grant_id": f"grant-succession-{uuid.uuid4().hex}",
                **{key: options[key] for key in ("replication", "work_records", "passive_only", "successor",
                                                 "ttl_seconds", "status_ttl_seconds") if key in options}},
                consent["target_profile"])
            grants.append({"member_id": consent["member_id"], "target_profile": invitation["target_profile"],
                           "grant": invitation["grant"], "catalog": invitation["catalog"]})
        return grants

    return mint


def fence_check(adapter):
    """``check(room_id, epoch)`` raising ``RoomAuthorityFenced`` for an epoch fenced here; a gateway
    without a durable Runs store has fenced nothing."""
    from gateway import hosted_room_fence as fence
    store = getattr(adapter, "_run_idempotency_store", None)
    path = getattr(store, "path", None) if getattr(store, "durable", False) is True else None

    def check(room_id: str, epoch: int) -> None:
        if path is not None and epoch <= fence.room_fence_state(path, room_id)["fenced_epoch"]:
            raise fence.RoomAuthorityFenced()
    return check


def _service(adapter):
    from gateway.session_authorities import all_authorities
    from gateway.session_authorities import served_profile_name
    for authority in all_authorities(getattr(adapter, "gateway_runner", None)):
        if served_profile_name(Path(authority.profile_id)) == "default":
            return getattr(authority, "hosted_room_service", None)
    return None


def backup_context(adapter):
    from gateway.hosted_room_succession_backup import BackupContext
    from gateway.platforms.api_server_room_grants import _grant_db
    custody_db = Path(_grant_db(adapter))
    return BackupContext(
        custody_db=custody_db, runs_store=adapter._run_idempotency_store,
        mint_grants=continuation_minter(adapter, custody_db, replace_same_epoch=False),
        replace_grants=continuation_minter(adapter, custody_db, replace_same_epoch=True), service=_service(adapter))


def http_routes(adapter):
    from gateway import hosted_room_succession as succession
    from gateway import hosted_room_succession_backup as backup

    def failure(code, status, detail=None):
        return web.json_response({"error": {"code": code, **({"detail": detail} if detail else {})}}, status=status)

    def operation(name):
        async def handle(request):
            from gateway.platforms import api_server
            if api_server._api_request_profile.get() not in (None, "default"):
                return failure("default_profile_required", 403)
            store = getattr(adapter, "_run_idempotency_store", None)
            if store is None or store.durable is not True:
                return failure("group_stop_storage_unavailable", 503)
            body, denied = await adapter._read_json_body(request.clone(client_max_size=MAX_REQUEST_BYTES[name]))
            if denied is not None:
                return denied
            context = backup_context(adapter)
            try:
                if name == "fence":
                    reply = await asyncio.to_thread(backup.answer_fence, context, body)
                elif name == "learn":
                    reply = await asyncio.to_thread(backup.answer_learn, context, body)
                elif name == "query":
                    reply = await asyncio.to_thread(backup.answer_query, context, body)
                elif name == "decision":
                    reply = await asyncio.to_thread(backup.answer_decision, context, body)
                elif name == "handover":
                    from gateway.hosted_room_succession_handover import answer_handover
                    reply = await asyncio.to_thread(answer_handover, context, body)
                else:
                    reply = await asyncio.to_thread(backup.answer_report, context.custody_db, body)
            except succession.ProofInvalid as exc:
                return failure(exc.reason, 422)
            except succession.SuccessionError as exc:
                status = 403 if exc.reason in {"not_owner", "invalid_succession_request"} else 409
                return failure(exc.reason, status, exc.detail)
            except (ValueError, TypeError, KeyError):
                return failure("invalid_succession_request", 400)
            return web.json_response(reply)
        return handle

    return [("POST", f"/v1/room-members/succession/{name}", operation(name)) for name in MAX_REQUEST_BYTES]
