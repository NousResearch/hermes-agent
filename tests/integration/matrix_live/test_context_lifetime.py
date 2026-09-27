"""A separate Matrix client verifies pending context across withdrawal and eviction."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from nio import RoomRedactResponse, RoomSendResponse

from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom
from tests.integration.matrix_live.context_client import _send, _wait_for_final
from tests.integration.matrix_live.context_client import group_gateway as group_gateway
from tests.integration.matrix_live.context_client import group_member as group_member


@pytest.mark.parametrize("gateway", ["pause-queued-context"], indirect=True)
@pytest.mark.parametrize("withdrawn", [False, True])
def test_queued_reply_retains_parent_from_intake_until_model_dispatch(
    tmp_path: Path,
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
    withdrawn: bool,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        home = tmp_path / "hermes"
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            await _send(
                client,
                live_room.room_id,
                f"{live_room.bot.user_id} establish",
                mention=live_room.bot.user_id,
            )
            await _wait_for_final(client, live_room, seen, "Matrix live reply")
            await _send(
                client,
                live_room.room_id,
                f"{live_room.bot.user_id} active question @matrix-live:pause",
                mention=live_room.bot.user_id,
            )
            while not (home / "context-started").exists():
                await asyncio.sleep(0.01)
            target = await _send(client, live_room.room_id, "Withdrawn queued parent")
            await _send(
                client,
                live_room.room_id,
                f"> <{live_room.observer.user_id}> Withdrawn queued parent\n\n{live_room.bot.user_id} queued question",
                mention=live_room.bot.user_id,
                reply=target,
            )
            while not (home / "reply-queued").exists():
                await asyncio.sleep(0.01)
            if withdrawn:
                (home / "expected-media-change").write_text(target, encoding="utf-8")
                (home / "evict-media-state").write_text("evict", encoding="utf-8")
                redacted = await client.room_redact(live_room.room_id, target)
                assert isinstance(redacted, RoomRedactResponse), redacted
                while not (home / "media-change-observed").exists():
                    await asyncio.sleep(0.01)
                assert (home / "media-change-observed").read_text(
                    encoding="utf-8"
                ) == target
            (home / "context-release").write_text("release", encoding="utf-8")
            await _wait_for_final(client, live_room, seen, "ok", count=2)
            requests = group_gateway.model.main_requests()
            assert len(requests) == 3
            previous = requests[1]["messages"]
            assert requests[2]["messages"][: len(previous)] == previous
            current = requests[2]["messages"][-1]["content"]
            assert "queued question" in current
            assert ("Withdrawn queued parent" in current) is (not withdrawn)
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


@pytest.mark.parametrize("gateway", ["pause-context"], indirect=True)
@pytest.mark.parametrize("scope", ["room", "thread"])
@pytest.mark.parametrize("withdrawn", [False, True])
def test_active_context_keeps_reaction_redaction_after_cache_eviction(
    tmp_path: Path,
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    record_property: Callable[[str, object], None],
    scope: str,
    withdrawn: bool,
) -> None:
    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        home = tmp_path / "hermes"
        seen: set[str] = set()
        try:
            await client.sync(timeout=0)
            root = (
                await _send(client, live_room.room_id, "Reaction thread root")
                if scope == "thread"
                else None
            )
            await _send(
                client,
                live_room.room_id,
                f"{live_room.bot.user_id} establish",
                root=root,
                mention=live_room.bot.user_id,
            )
            await _wait_for_final(client, live_room, seen, "Matrix live reply")
            target = await _send(
                client,
                live_room.room_id,
                "Reaction snapshot target",
                root=root,
            )
            reaction = await client.room_send(
                live_room.room_id,
                "m.reaction",
                {
                    "m.relates_to": {
                        "rel_type": "m.annotation",
                        "event_id": target,
                        "key": "👍",
                    },
                },
            )
            assert isinstance(reaction, RoomSendResponse), reaction
            await _send(
                client,
                live_room.room_id,
                f"{live_room.bot.user_id} reaction question @matrix-live:pause",
                root=root,
                mention=live_room.bot.user_id,
            )
            while not (home / "context-started").exists():
                await asyncio.sleep(0.01)
            if withdrawn:
                (home / "expected-media-change").write_text(
                    reaction.event_id, encoding="utf-8"
                )
                (home / "evict-media-state").write_text("evict", encoding="utf-8")
                redacted = await client.room_redact(
                    live_room.room_id, reaction.event_id
                )
                assert isinstance(redacted, RoomRedactResponse), redacted
                while not (home / "media-change-observed").exists():
                    await asyncio.sleep(0.01)
                assert (home / "media-change-observed").read_text(
                    encoding="utf-8"
                ) == reaction.event_id
            (home / "context-release").write_text("release", encoding="utf-8")
            await _wait_for_final(client, live_room, seen, "ok")
            requests = group_gateway.model.main_requests()
            assert len(requests) == 2
            previous = requests[0]["messages"]
            assert requests[1]["messages"][: len(previous)] == previous
            current = requests[1]["messages"][-1]["content"]
            assert "Reaction snapshot target" in current
            reaction_text = f"[reaction by {live_room.observer.user_id} to {target}] 👍"
            assert (reaction_text in current) is (not withdrawn)
        finally:
            await client.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=20))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
