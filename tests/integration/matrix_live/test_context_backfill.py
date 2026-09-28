"""A separate Matrix client verifies room and thread catch-up in model context."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import aiohttp
import pytest
from nio import JoinResponse, RoomInviteResponse, RoomMessageText, RoomRedactResponse, RoomSendResponse, UploadResponse

from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom
from tests.integration.matrix_live.context_client import _send, _wait_for_final
from tests.integration.matrix_live.context_client import group_gateway as group_gateway
from tests.integration.matrix_live.context_client import group_member as group_member










def test_room_mention_recovers_unaddressed_messages(
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    stage = "initial reply"

    async def exchange() -> None:
        nonlocal stage
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} establish",
                        mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "Matrix live reply")

            target = await _send(client, live_room.room_id, "Room decision alpha")
            await _send(client, live_room.room_id, "Room decision beta")
            reacted = await client.room_send(live_room.room_id, "m.reaction", {
                "m.relates_to": {"rel_type": "m.annotation", "event_id": target, "key": "👍"},
            })
            assert isinstance(reacted, RoomSendResponse), reacted
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} catch up",
                        mention=live_room.bot.user_id)
            stage = "catch-up reply"
            await _wait_for_final(client, live_room, seen, "ok")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            prompt = json.dumps(requests[1]["messages"], ensure_ascii=False)
            assert "[Recent room messages]" in prompt
            assert "Room decision alpha" in prompt
            assert "Room decision beta" in prompt
            assert f"[reaction by {live_room.observer.user_id} to {target}] 👍" in prompt
            assert "[New message]" in prompt
        finally:
            await client.close()

    started = time.monotonic()
    try:
        try:
            asyncio.run(asyncio.wait_for(exchange(), timeout=15))
        except asyncio.TimeoutError:
            pytest.fail(
                f"Matrix room catch-up exceeded 15 seconds during {stage}. "
                f"Model requests: {len(group_gateway.model.main_requests())}. Gateway logs:\n"
                + group_gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_thread_mention_recovers_only_its_earlier_messages(
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await client.sync(timeout=0)
            root_a = await _send(client, live_room.room_id, "Thread A root")
            root_b = await _send(client, live_room.room_id, "Thread B root")
            await _send(client, live_room.room_id, "Thread A earlier", root=root_a)
            await _send(client, live_room.room_id, "Thread B earlier", root=root_b)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} thread question",
                        root=root_a, mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, set(), "Matrix live reply")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 1
            prompt = json.dumps(requests[0]["messages"])
            assert "Thread A root" in prompt
            assert "Thread A earlier" in prompt
            assert "Thread B earlier" not in prompt
            assert "Thread B root" not in prompt
        finally:
            await client.close()

    started = time.monotonic()
    try:
        try:
            asyncio.run(asyncio.wait_for(exchange(), timeout=15))
        except asyncio.TimeoutError:
            pytest.fail(
                "Matrix thread catch-up exceeded 15 seconds. Gateway logs:\n"
                + group_gateway.container.get_wrapped_container().logs().decode(errors="replace")[-6000:]
            )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


@pytest.mark.parametrize("gateway", ["pause-context"], indirect=True)
def test_room_catch_up_shows_edits_and_redactions_to_model(
    tmp_path: Path,
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} establish",
                        mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "Matrix live reply")

            edited_target = await _send(client, live_room.room_id, "Draft room decision")
            replacement = await client.room_send(live_room.room_id, "m.room.message", {
                "msgtype": "m.text", "body": "* Final room decision",
                "m.new_content": {"msgtype": "m.text", "body": "Final room decision"},
                "m.relates_to": {"rel_type": "m.replace", "event_id": edited_target},
            })
            assert isinstance(replacement, RoomSendResponse), replacement
            withdrawn_edit = await client.room_send(live_room.room_id, "m.room.message", {
                "msgtype": "m.text", "body": "* Withdrawn edited room decision",
                "m.new_content": {"msgtype": "m.text", "body": "Withdrawn edited room decision"},
                "m.relates_to": {"rel_type": "m.replace", "event_id": edited_target},
            })
            assert isinstance(withdrawn_edit, RoomSendResponse), withdrawn_edit
            edit_redaction = await client.room_redact(live_room.room_id, withdrawn_edit.event_id)
            assert isinstance(edit_redaction, RoomRedactResponse), edit_redaction

            redacted_target = await _send(client, live_room.room_id, "Withdrawn room decision")
            redaction = await client.room_redact(live_room.room_id, redacted_target)
            assert isinstance(redaction, RoomRedactResponse), redaction

            late_target = await _send(client, live_room.room_id, "Withdrawn during enrichment")
            await _send(
                client, live_room.room_id,
                f"> <{live_room.observer.user_id}> Final room decision\n\n"
                f"{live_room.bot.user_id} catch up @matrix-live:pause",
                mention=live_room.bot.user_id, reply=edited_target,
            )
            while not (tmp_path / "hermes" / "context-started").exists():
                await asyncio.sleep(0.01)
            late_edit = await client.room_send(live_room.room_id, "m.room.message", {
                "msgtype": "m.text", "body": "* Revised decision during enrichment",
                "m.new_content": {"msgtype": "m.text", "body": "Revised decision during enrichment"},
                "m.relates_to": {"rel_type": "m.replace", "event_id": edited_target},
            })
            assert isinstance(late_edit, RoomSendResponse), late_edit
            late_redaction = await client.room_redact(live_room.room_id, late_target)
            assert isinstance(late_redaction, RoomRedactResponse), late_redaction
            (tmp_path / "hermes" / "context-release").write_text("release", encoding="utf-8")
            await _wait_for_final(client, live_room, seen, "ok")

            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            prompt = json.dumps(requests[1]["messages"])
            assert "[Recent room messages]" in prompt
            assert "Revised decision during enrichment" in prompt
            assert "Final room decision" not in prompt
            assert "Withdrawn during enrichment" not in prompt
            assert "Live enrichment completed" in prompt
            assert requests[0]["messages"][0] == requests[1]["messages"][0]
            assert "[redacted]" in prompt
            assert "Draft room decision" not in prompt
            assert "Withdrawn room decision" not in prompt
            assert "Withdrawn edited room decision" not in prompt
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


def test_redacted_child_is_removed_from_thread_relations(live_room: LiveRoom) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            root = await _send(client, live_room.room_id, "Thread root")
            child = await _send(client, live_room.room_id, "Thread child", root=root)
            base = f"{live_room.homeserver}/_matrix/client/v1/rooms/{quote(live_room.room_id, safe='')}"
            relations_url = f"{base}/relations/{quote(root, safe='')}/m.thread"
            event_url = f"{live_room.homeserver}/_matrix/client/v3/rooms/{quote(live_room.room_id, safe='')}/event/{quote(child, safe='')}"
            headers = {"Authorization": f"Bearer {live_room.observer.access_token}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(relations_url) as response:
                    assert response.status == 200
                    before = await response.json()

                redaction = await client.room_redact(live_room.room_id, child)
                assert isinstance(redaction, RoomRedactResponse), redaction

                async with session.get(relations_url) as response:
                    assert response.status == 200
                    after = await response.json()
                async with session.get(event_url) as response:
                    assert response.status == 200
                    event = await response.json()

            assert [item["event_id"] for item in before["chunk"]] == [child]
            assert after["chunk"] == []
            assert event["content"] == {}
            assert event["unsigned"]["redacted_because"]["redacts"] == child
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(exchange(), timeout=15))


@pytest.mark.parametrize("gateway", ["pause-image-context", "pause-image-conversion"], indirect=True)
@pytest.mark.parametrize("change", [
    "unchanged", "replacement", "redaction", "redaction-eviction",
    "replacement-redaction-eviction", "sender", "missing-new-content",
])
def test_quoted_image_catch_up_keeps_only_current_model_attachment(
    tmp_path: Path, group_gateway: LiveGateway, live_room: LiveRoom,
    group_member: MatrixAccount, record_property: Callable[[str, object], None], change: str,
    request: pytest.FixtureRequest,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        other = group_member.client(live_room.homeserver)
        home = tmp_path / "hermes"
        seen: set[str] = set()
        pixels = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
        try:
            await client.sync(timeout=0)
            await _send(client, live_room.room_id, f"{live_room.bot.user_id} establish",
                        mention=live_room.bot.user_id)
            await _wait_for_final(client, live_room, seen, "Matrix live reply")
            uploaded, _ = await client.upload(
                io.BytesIO(pixels), content_type="image/png", filename="quoted.png",
                filesize=len(pixels),
            )
            assert isinstance(uploaded, UploadResponse), uploaded
            image = {"msgtype": "m.image", "body": "quoted.png", "url": uploaded.content_uri,
                     "info": {"mimetype": "image/png", "size": len(pixels), "w": 1, "h": 1}}
            target = await client.room_send(live_room.room_id, "m.room.message", image)
            assert isinstance(target, RoomSendResponse), target
            latest = None
            if change == "replacement-redaction-eviction":
                latest = await client.room_send(live_room.room_id, "m.room.message", {
                    **image, "m.new_content": image,
                    "m.relates_to": {"rel_type": "m.replace", "event_id": target.event_id},
                })
                assert isinstance(latest, RoomSendResponse), latest
            await _send(client, live_room.room_id,
                        f"{live_room.bot.user_id} inspect quoted media @matrix-live:pause",
                        mention=live_room.bot.user_id, reply=target.event_id)
            while not (home / "context-started").exists():
                await asyncio.sleep(0.01)
            if change != "unchanged":
                (home / "expected-media-change").write_text(target.event_id, encoding="utf-8")
                if change.endswith("-eviction"):
                    (home / "evict-media-state").write_text("evict", encoding="utf-8")
                if change in {"replacement", "sender", "missing-new-content"}:
                    edit_content = {
                        "msgtype": "m.text", "body": "* Replaced image with text",
                        "m.new_content": {"msgtype": "m.text", "body": "Replaced image with text"},
                        "m.relates_to": {"rel_type": "m.replace", "event_id": target.event_id},
                    }
                    if change == "missing-new-content":
                        edit_content.pop("m.new_content")
                    editor = other if change == "sender" else client
                    replacement = await editor.room_send(live_room.room_id, "m.room.message", edit_content)
                    assert isinstance(replacement, RoomSendResponse), replacement
                else:
                    redacted = await client.room_redact(live_room.room_id, latest.event_id if latest else target.event_id)
                    assert isinstance(redacted, RoomRedactResponse), redacted
                while not (home / "media-change-observed").exists():
                    await asyncio.sleep(0.01)
                assert (home / "media-change-observed").read_text(encoding="utf-8") == target.event_id
            (home / "context-release").write_text("release", encoding="utf-8")
            await _wait_for_final(client, live_room, seen, "ok")
            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            assert requests[0]["messages"][0] == requests[1]["messages"][0]
            assert requests[1]["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]
            current = requests[1]["messages"][-1]["content"]
            parts = current if isinstance(current, list) else [{"type": "text", "text": current}]
            attachments = [part for part in parts if part.get("type") == "image_url"]
            unchanged = change in {"unchanged", "sender", "missing-new-content"}
            assert len(attachments) == (1 if unchanged else 0), {
                "model_input": current,
                "media_logs": [
                    line
                    for path in (home / "logs").glob("gateway.log*")
                    for line in path.read_text(errors="replace").splitlines()
                    if "image" in line.lower() or "media" in line.lower()
                ][-30:],
            }
            text = "\n".join(part["text"] for part in parts if part.get("type") == "text")
            assert "[Recent room messages]" in text
            assert "inspect quoted media" in text
            assert "Live enrichment completed" in text
            if unchanged:
                assert attachments[0]["image_url"]["url"].startswith("data:image/")
                assert "[image]" in text
            elif change == "replacement":
                assert "Replaced image with text" in text
            elif change == "replacement-redaction-eviction":
                expected = ("[event content unavailable]" if request.node.callspec.params["gateway"] == "pause-image-conversion"
                            else "[image]")
                assert expected in text
            else:
                assert "[redacted]" in text
                assert "Replying to" not in text
        finally:
            await client.close()
            await other.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
