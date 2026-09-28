"""Bounded native poll operations through a live Matrix adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from plugins.platforms.matrix.effective_event import _decrypt, event_content
from plugins.platforms.matrix.polls import MatrixPoll, UNSTABLE, poll_results, subtype
from plugins.platforms.matrix.read_context import Method, _raw_event


@dataclass(frozen=True)
class _PollSession:
    adapter: Any
    client: Any
    room_id: str
    requester: str
    actor: str

    def check_client(self) -> None:
        if self.adapter._client is not self.client or self.adapter._user_id != self.actor:
            raise ValueError("Matrix client changed during the poll operation")
        if self.client is None or getattr(self.adapter, "_closing", False):
            raise ValueError("Matrix client is disconnected")

    async def check_access(self) -> None:
        self.check_client()
        chat_type = "dm" if await self.adapter._is_dm_room(self.room_id) else "group"
        allowed = await self.adapter._is_allowed_matrix_room_event(self.room_id)
        self.check_client()
        if self.room_id not in self.adapter._joined_rooms or not allowed:
            raise ValueError("Matrix room is not allowed or joined")
        if self.adapter._is_sender_authorized(self.requester, chat_type=chat_type, chat_id=self.room_id) is not True:
            raise ValueError("Matrix requester is not authorized for this room")


@dataclass(frozen=True)
class PollSnapshot:
    poll: MatrixPoll
    events: list[dict[str, Any]]
    moderators: set[str]
    incomplete_reasons: list[str]

    def results(self) -> dict[str, Any]:
        return poll_results(self.poll, self.events, moderators=self.moderators, incomplete_reasons=self.incomplete_reasons)


async def _native_event(session: _PollSession, raw: dict[str, Any], room_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if raw.get("room_id", room_id) != room_id:
        return None, "relation belongs to another room"
    raw = {**raw, "room_id": room_id}
    unsigned = raw.get("unsigned")
    if ((isinstance(unsigned, dict) and unsigned.get("redacted_because"))
            or session.adapter._event_context_cache.is_redacted(room_id, raw.get("event_id"))):
        return None, None
    if raw.get("type") != "m.room.encrypted":
        if raw.get("type") == "m.poll.end" and subtype(event_content(raw), "end") is None:
            raw = {**raw, "content": {**event_content(raw), "m.poll.end": {}}}
        return raw, None
    decrypted, error = await _decrypt(session.client, raw)
    if error:
        return None, error["error"]
    clear = _raw_event(decrypted)
    if clear.get("room_id", room_id) != room_id:
        return None, "decrypted relation belongs to another room"
    content = event_content(decrypted)
    if clear.get("type") == "m.poll.end" and subtype(content, "end") is None:
        content = {**content, "m.poll.end": {}}
    return {**raw, "type": clear.get("type"), "content": content}, None


async def fetch_poll(session: _PollSession, event_id: str, limit: int) -> PollSnapshot:
    room_id = session.room_id
    path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
    raw = _raw_event(await asyncio.wait_for(session.client.api.request(Method.GET, path), timeout=10.0))
    if raw.get("event_id") != event_id:
        raise ValueError("The server returned a different poll event")
    start, error = await _native_event(session, raw, room_id)
    if start is None:
        raise ValueError(error or "The poll was redacted")
    poll = MatrixPoll.from_event(start, room_id)
    path = f"/_matrix/client/v1/rooms/{quote(room_id, safe='')}/relations/{quote(event_id, safe='')}/m.reference"
    events = []
    reasons = []
    token = None
    tokens = set()
    seen = set()
    examined = 0
    while examined < limit:
        query = {"limit": str(min(50, limit - examined)), "dir": "f"}
        if token:
            query["from"] = token
        response = await asyncio.wait_for(session.client.api.request(Method.GET, path, query_params=query), timeout=10.0)
        chunk = response.get("chunk") if isinstance(response, dict) else None
        if not isinstance(chunk, list):
            reasons.append("relations unavailable")
            break
        budget = limit - examined
        for relation in chunk[:budget]:
            examined += 1
            if not isinstance(relation, dict) or not isinstance(relation.get("event_id"), str):
                reasons.append("invalid relation event")
                continue
            if relation["event_id"] in seen:
                continue
            seen.add(relation["event_id"])
            clear, error = await _native_event(session, relation, room_id)
            if error:
                reasons.append(error)
            if clear:
                events.append(clear)
        token = response.get("next_batch")
        if len(chunk) > budget:
            reasons.append("relation limit reached")
            break
        if not token:
            break
        if not isinstance(token, str) or token in tokens or not chunk:
            reasons.append("relation pagination incomplete")
            break
        tokens.add(token)
    if token and examined >= limit:
        reasons.append("relation limit reached")
    moderators = set()
    other_ends = [raw for raw in events if raw.get("type") in {"m.poll.end", f"{UNSTABLE}end"}
                  and raw.get("sender") != poll.creator and isinstance(subtype(raw.get("content", {}), "end"), dict)]
    if other_ends:
        try:
            levels = await _power_levels(session)
            users = levels.get("users", {})
            default = levels.get("users_default", 0)
            required = levels.get("redact", 50)
            if not isinstance(users, dict) or not _integer(default) or not _integer(required):
                raise ValueError("Invalid room power levels")
            for raw in other_ends:
                level = users.get(raw.get("sender"), default)
                if _integer(level) and level >= required:
                    moderators.add(raw["sender"])
        except Exception:
            reasons.append("closing authority unavailable")
    events = [raw for raw in events if not session.adapter._event_context_cache.is_redacted(room_id, raw.get("event_id"))]
    if session.adapter._event_context_cache.is_redacted(room_id, event_id):
        raise ValueError("The poll was redacted")
    return PollSnapshot(poll, events, moderators, reasons)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


async def _power_levels(session: _PollSession) -> dict[str, Any]:
    from mautrix.types import EventType, RoomID

    content = await session.client.get_state_event(RoomID(session.room_id), EventType.ROOM_POWER_LEVELS)
    return content if isinstance(content, dict) else content.serialize()


async def _check_closing_authority(session: _PollSession) -> None:
    levels = await _power_levels(session)
    users = levels.get("users", {})
    level = users.get(session.actor, levels.get("users_default", 0)) if isinstance(users, dict) else None
    required = levels.get("redact", 50)
    if not _integer(level) or not _integer(required) or level < required:
        raise ValueError("The Matrix bot cannot close this poll")


async def _send_native(session: _PollSession, event_type: str, content: dict[str, Any], *, creator: str | None = None) -> str:
    room_id = session.room_id
    from mautrix.errors import MNotFound
    from mautrix.types import EventType, RoomID

    try:
        await session.client.get_state_event(RoomID(room_id), EventType.ROOM_ENCRYPTION)
        encrypted = True
    except MNotFound:
        encrypted = False
    await session.check_access()
    if creator is not None and creator != session.actor:
        await _check_closing_authority(session)
        await session.check_access()
    if encrypted and getattr(session.client, "crypto", None) is None:
        raise ValueError("The encrypted room has no available crypto session")
    event_id = await session.client.send_message_event(
        RoomID(room_id), EventType.find(event_type, EventType.Class.MESSAGE), content,
    )
    return str(event_id)


async def matrix_poll_action(adapter: Any, room_id: str, requester: str, action: str, args: dict[str, Any]) -> dict[str, Any]:
    session = _PollSession(adapter, adapter._client, room_id, requester, adapter._user_id)
    try:
        await session.check_access()
        if action == "create":
            question, labels = args.get("question"), args.get("answers")
            maximum = args.get("max_selections", 1)
            kind = args.get("kind", "disclosed")
            if not isinstance(question, str) or not question.strip() or len(question) > 1200:
                raise ValueError("question must contain between 1 and 1200 characters")
            if (not isinstance(labels, list) or not 2 <= len(labels) <= 20
                    or any(not isinstance(label, str) or not label.strip() or len(label) > 300 for label in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError("answers must contain 2 to 20 distinct non-empty texts, each up to 300 characters")
            if not _integer(maximum) or not 1 <= maximum <= len(labels):
                raise ValueError("max_selections must be between 1 and the number of answers")
            if kind not in {"disclosed", "undisclosed"}:
                raise ValueError("kind must be disclosed or undisclosed")
            answers = [{"id": uuid4().hex, "org.matrix.msc1767.text": label} for label in labels]
            content = {f"{UNSTABLE}start": {
                "question": {"org.matrix.msc1767.text": question}, "answers": answers,
                "kind": f"{UNSTABLE}{kind}", "max_selections": maximum,
            }, "org.matrix.msc1767.text": question + "\n" + "\n".join(f"{index + 1}. {label}" for index, label in enumerate(labels))}
            event_id = await _send_native(session, f"{UNSTABLE}start", content)
            return {"poll_id": event_id, "actor": session.actor, "answers": [{"id": answer["id"], "text": answer["org.matrix.msc1767.text"]} for answer in answers]}
        event_id = args.get("poll_id")
        limit = args.get("limit", 100)
        if not isinstance(event_id, str) or not event_id.startswith("$"):
            raise ValueError("poll_id must be a Matrix event ID")
        if not _integer(limit) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200 relations")
        snapshot = await fetch_poll(session, event_id, limit)
        await session.check_access()
        results = snapshot.results()
        if action == "results":
            return results
        if not results["complete"]:
            raise ValueError("Poll state is incomplete; read complete results before voting or closing")
        if results["closed"]:
            raise ValueError("The poll is already closed")
        poll = snapshot.poll
        content = {"m.relates_to": {"rel_type": "m.reference", "event_id": event_id}}
        if action == "vote":
            selection = args.get("answers")
            if not poll.valid_selection(selection):
                raise ValueError("answers must contain distinct known answer IDs within the selection limit")
            content[f"{UNSTABLE}response"] = {"answers": selection}
            event_type = f"{UNSTABLE}response"
        elif action == "close":
            content[f"{UNSTABLE}end"] = {}
            content["org.matrix.msc1767.text"] = "Poll closed."
            event_type = f"{UNSTABLE}end"
        else:
            raise ValueError("Unknown Matrix poll action")
        sent_id = await _send_native(session, event_type, content, creator=poll.creator if action == "close" else None)
        return {"poll_id": event_id, "event_id": sent_id, "actor": session.actor, "action": action}
    except Exception as exc:
        return {"error": str(exc)}
