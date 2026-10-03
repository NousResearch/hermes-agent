"""Independent persisted matrix-nio checks for approval card delivery and edits."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from nio import (
    AsyncClient, JoinedMembersResponse, KeysQueryResponse, KeysUploadResponse,
    RoomMessageText, RoomRedactResponse, RoomSendResponse, SyncResponse,
)

from client import open_encrypted_client


async def exercise(room_id: str, bot_device: str, other_login: dict[str, str], decision: str, encrypted: bool) -> dict:
    owner = open_encrypted_client()
    other = AsyncClient(owner.homeserver, other_login["user_id"])
    other.restore_login(other_login["user_id"], other_login["device_id"], other_login["access_token"])
    events: dict[str, RoomMessageText] = {}
    wire_types: dict[str, str] = {}
    observed: dict[str, dict] = {}
    anchors: dict[str, dict[str, str]] = {}

    async def receive_until(stage: str, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 30
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"Missing {stage}; observed events: {observed}")
            try:
                response = await asyncio.wait_for(owner.sync(timeout=250), timeout=remaining)
            except TimeoutError as exc:
                raise AssertionError(f"Missing {stage}; observed events: {observed}") from exc
            assert isinstance(response, SyncResponse), response
            joined = response.rooms.join.get(room_id)
            if joined:
                for event in joined.timeline.events:
                    observed[event.event_id] = {
                        "type": event.source.get("type"), "class": type(event).__name__,
                        "sender": event.sender, "content": event.source.get("content"),
                    }
                    if isinstance(event, RoomMessageText) and event.sender == "@hermes:matrix.test":
                        assert event.decrypted is encrypted, event.source
                        events[event.event_id] = event

    async def send(client, content, kind="m.room.message") -> str:
        response = await client.room_send(room_id, kind, content, ignore_unverified_devices=True)
        assert isinstance(response, RoomSendResponse), response
        return response.event_id

    def cards() -> list[RoomMessageText]:
        return [event for event in events.values() if "Hermes wants to run a command that needs your OK" in event.body and "m.new_content" not in event.source["content"]]

    def replacements(card_id: str) -> list[RoomMessageText]:
        return [event for event in events.values() if event.source["content"].get("m.relates_to") == {"rel_type": "m.replace", "event_id": card_id}]

    async def open_thread(marker: str) -> None:
        root = await send(owner, {"msgtype": "m.notice", "body": f"Approval thread root [in:approval-{marker}]"})
        body = f"Run approval {marker} [in:approval-{marker}]"
        opening = await send(owner, {
            "msgtype": "m.text", "body": body,
            "m.relates_to": {"rel_type": "m.thread", "event_id": root, "is_falling_back": True, "m.in_reply_to": {"event_id": root}},
        })
        anchors[marker] = {"root": root, "opening": opening, "body": body}

    try:
        response = await owner.sync(timeout=0, full_state=True)
        assert isinstance(response, SyncResponse), response
        assert owner.rooms[room_id].encrypted is encrypted
        if encrypted:
            uploaded = await owner.keys_upload()
            assert isinstance(uploaded, KeysUploadResponse), uploaded
            members = await owner.joined_members(room_id)
            assert isinstance(members, JoinedMembersResponse), members
            queried = await owner.keys_query()
            assert isinstance(queried, KeysQueryResponse), queried
            assert bot_device in owner.device_store["@hermes:matrix.test"], queried
            device = owner.device_store["@hermes:matrix.test"][bot_device]
            owner.verify_device(device)
            assert device.verified
        await open_thread("first")
        await receive_until("first initial approval card", lambda: len(cards()) == 1)
        expected_cards = 2 if decision == "once" else 1
        if expected_cards == 2:
            await open_thread("second")
            await receive_until("both concurrent approval cards", lambda: len(cards()) == expected_cards)
        pending = cards()
        for card in pending:
            marker = "first" if "approval-first-ran" in card.body else "second"
            content = card.source["content"]
            assert "<pre>" in content["formatted_body"]
            assert content["m.relates_to"] == {
                "rel_type": "m.thread", "event_id": anchors[marker]["root"],
                "is_falling_back": True, "m.in_reply_to": {"event_id": anchors[marker]["opening"]},
            }, {"actual": content["m.relates_to"], "anchors": anchors}
            assert "will NOT run" in card.body

        first = next(card for card in pending if "approval-first-ran" in card.body)
        if decision != "expired":
            await send(other, {"m.relates_to": {"rel_type": "m.annotation", "event_id": first.event_id, "key": "✅"}}, "m.reaction")
            await receive_until("requester-only refusal", lambda: any("Only the user who requested" in event.body for event in events.values()))
            refusal = next(event for event in events.values() if "Only the user who requested" in event.body)
            expected_feedback_relation = {
                "rel_type": "m.thread", "event_id": anchors["first"]["root"],
                "is_falling_back": True, "m.in_reply_to": {"event_id": first.event_id},
            }
            assert refusal.source["content"]["m.relates_to"] == expected_feedback_relation, refusal.source
            await send(owner, {"m.relates_to": {"rel_type": "m.annotation", "event_id": first.event_id, "key": "👍"}}, "m.reaction")
            await receive_until("invalid reaction feedback", lambda: any("That reaction is not valid" in event.body for event in events.values()))
            invalid = next(event for event in events.values() if "That reaction is not valid" in event.body)
            assert invalid.source["content"]["m.relates_to"] == expected_feedback_relation, invalid.source
            assert not any(
                label in event.body
                for event in replacements(first.event_id)
                for label in ("Approved once", "Denied", "Expired", "Cancelled")
            )
            if decision == "summarized":
                await receive_until("advisory replacement", lambda: any("Advisory interpretation" in event.body for event in replacements(first.event_id)))
                advisory = next(event for event in replacements(first.event_id) if "Advisory interpretation" in event.body)
                advisory_html = advisory.source["content"]["m.new_content"]["formatted_body"]
                assert "<pre>" in advisory_html and "<details>" not in advisory_html, advisory_html
            if expected_cards == 2:
                second = next(card for card in pending if "approval-second-ran" in card.body)
                await send(owner, {"m.relates_to": {"rel_type": "m.annotation", "event_id": second.event_id, "key": "✅"}}, "m.reaction")
                await receive_until("second card approval", lambda: any("Approved once" in event.body for event in replacements(second.event_id)))
                await receive_until("second thread final reply", lambda: any(
                    event.body == "Matrix approval live reply"
                    and event.source["content"].get("m.relates_to", {}).get("event_id") == anchors["second"]["root"]
                    for event in events.values()
                ))
                assert not replacements(first.event_id)
            key = "❌" if decision in {"once", "deny"} else "✅"
            choice_id = await send(owner, {"m.relates_to": {"rel_type": "m.annotation", "event_id": first.event_id, "key": key}}, "m.reaction")
        label = "Expired" if decision == "expired" else "Approved once" if decision == "summarized" else "Denied"
        await receive_until(f"first card terminal outcome {label}", lambda: any(label in event.body for event in replacements(first.event_id)))
        terminal = [event for event in replacements(first.event_id) if label in event.body]
        assert len(terminal) == 1
        assert "<details>" in terminal[0].source["content"]["m.new_content"]["formatted_body"]
        assert "approval-first-ran" in terminal[0].source["content"]["m.new_content"]["body"]
        def replies() -> list[RoomMessageText]:
            return [event for event in events.values() if event.body == "Matrix approval live reply"]

        await receive_until("all final gateway replies", lambda: len(replies()) == expected_cards)
        assert [event.source["content"]["m.relates_to"] for event in replies()] == [
            {"rel_type": "m.thread", "event_id": anchor["root"], "is_falling_back": False,
             "m.in_reply_to": {"event_id": anchor["opening"]}}
            for anchor in (list(anchors.values())[::-1] if decision == "once" else anchors.values())
        ], {"actual": [event.source["content"]["m.relates_to"] for event in replies()], "anchors": anchors}
        if decision == "summarized":
            redacted = await owner.room_redact(room_id, choice_id)
            assert isinstance(redacted, RoomRedactResponse), redacted
        repeated = await send(owner, {"m.relates_to": {"rel_type": "m.annotation", "event_id": first.event_id, "key": "✅"}}, "m.reaction")
        await receive_until("repeated terminal choice", lambda: repeated in observed)
        assert [event for event in replacements(first.event_id) if label in event.body] == terminal
        for event_id in [card.event_id for card in pending] + [
            event_id for anchor in anchors.values() for event_id in (anchor["root"], anchor["opening"])
        ]:
            raw = await owner.room_get_event(room_id, event_id)
            source = await raw.transport_response.json()
            wire_types[event_id] = source["type"]
        assert set(wire_types.values()) == {"m.room.encrypted" if encrypted else "m.room.message"}
        return {"cards": len(pending), "terminal": label, "anchors": anchors}
    finally:
        await owner.close()
        await other.close()
