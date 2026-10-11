"""The model discovers the requester's joined rooms, Spaces and users without joining a public child."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import dedent

import aiohttp
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from nio import (
    AsyncClient,
    AsyncClientConfig,
    JoinResponse,
    JoinedRoomsResponse,
    ProfileSetDisplayNameResponse,
    RoomCreateResponse,
    RoomMessageText,
    RoomSendResponse,
    RoomVisibility,
)

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom, MatrixAccount, _wait_for


@dataclass(frozen=True)
class DiscoveryRoom(LiveRoom):
    ordinary_room: str
    space: str
    unjoined_room: str
    bot_only_room: str


def _client(account: MatrixAccount, homeserver: str, *, request_timeout: float = 5) -> AsyncClient:
    client = AsyncClient(
        homeserver,
        account.user_id,
        config=AsyncClientConfig(
            encryption_enabled=False,
            request_timeout=request_timeout,
            max_timeouts=0,
            max_limit_exceeded=0,
        ),
    )
    client.restore_login(account.user_id, account.device_id, account.access_token)
    return client


async def _wait_for_directory_entry(
    account: MatrixAccount, homeserver: str, user_id: str, display_name: str,
) -> None:
    """Search the user directory as *account* until it lists *user_id* with *display_name*.

    Synapse updates the directory in the background after profile and membership changes, so a
    search made straight after the set-up can return no rows.
    """
    url = f"{homeserver}/_matrix/client/v3/user_directory/search"
    headers = {"Authorization": f"Bearer {account.access_token}"}
    expected = {"user_id": user_id, "display_name": display_name}
    async with aiohttp.ClientSession(headers=headers) as http:
        while True:
            async with http.post(url, json={"search_term": display_name, "limit": 5}) as response:
                assert response.status == 200, await response.text()
                results = (await response.json())["results"]
            if any({key: row.get(key) for key in expected} == expected for row in results):
                return
            await asyncio.sleep(0.25)


_SETUP_SECONDS = 30


def _directory_state(container: DockerContainer) -> dict[str, object]:
    code = dedent("""\
        import json
        import sqlite3

        with sqlite3.connect("file:/data/homeserver.db?mode=ro", uri=True) as db:
            state = {
                "position": db.execute("SELECT stream_id FROM user_directory_stream_pos").fetchall(),
                "pending": db.execute(
                    "SELECT update_name FROM background_updates "
                    "WHERE update_name LIKE 'populate_user_directory_%'"
                ).fetchall(),
                "entries": db.execute("SELECT user_id,display_name FROM user_directory").fetchall(),
            }
        print(json.dumps(state))
        """)
    result = container.exec(["python", "-c", code])
    assert result.exit_code == 0, result.output.decode(errors="replace")
    return json.loads(result.output)


@pytest.fixture
def live_room(
    live_room: LiveRoom, synapse: tuple[DockerContainer, str, Network],
) -> DiscoveryRoom:
    _wait_for(
        lambda: not _directory_state(synapse[0])["pending"],
        "Synapse user-directory bootstrap",
        timeout=120,
        details=lambda: str(_directory_state(synapse[0])),
    )

    async def create_rooms(alice: AsyncClient, bot: AsyncClient) -> tuple[str, str, str, str]:
        public = await alice.room_create(
            name="Unjoined public child", visibility=RoomVisibility.public
        )
        assert isinstance(public, RoomCreateResponse), public
        ordinary, space, bot_only = await asyncio.gather(
            alice.room_create(name="Joined discovery room", invite=[live_room.bot.user_id]),
            alice.room_create(
                name="Joined discovery Space",
                room_type="m.space",
                invite=[live_room.bot.user_id],
                initial_state=[{
                    "type": "m.space.child",
                    "state_key": public.room_id,
                    "content": {"via": ["matrix.test"]},
                }],
            ),
            bot.room_create(name="Bot-only discovery room"),
        )
        for response in (ordinary, space, bot_only):
            assert isinstance(response, RoomCreateResponse), response
        joins = await asyncio.gather(bot.join(ordinary.room_id), bot.join(space.room_id))
        for joined in joins:
            assert isinstance(joined, JoinResponse), joined
        return ordinary.room_id, space.room_id, public.room_id, bot_only.room_id

    async def prepare() -> DiscoveryRoom:
        # One deadline bounds the whole set-up. createRoom is not idempotent, so
        # the clients never retry a request that timed out.
        alice = _client(live_room.observer, live_room.homeserver, request_timeout=_SETUP_SECONDS)
        bot = _client(live_room.bot, live_room.homeserver, request_timeout=_SETUP_SECONDS)
        try:
            profile = await alice.set_displayname("Alice Discovery")
            assert isinstance(profile, ProfileSetDisplayNameResponse), profile
            rooms, _ = await asyncio.gather(
                create_rooms(alice, bot),
                _wait_for_directory_entry(
                    live_room.bot, live_room.homeserver, live_room.observer.user_id, "Alice Discovery",
                ),
            )
            return DiscoveryRoom(
                live_room.homeserver, live_room.room_id, live_room.bot, live_room.observer, *rooms,
            )
        finally:
            await alice.close()
            await bot.close()

    return asyncio.run(asyncio.wait_for(prepare(), timeout=_SETUP_SECONDS))


@pytest.mark.parametrize("gateway", ["discovery"], indirect=True)
def test_model_discovers_joined_rooms_spaces_and_users_without_autojoin(
    gateway: LiveGateway,
    live_room: DiscoveryRoom,
    synapse: tuple[DockerContainer, str, Network],
    record_property: Callable[[str, object], None],
) -> None:
    before = _directory_state(synapse[0])
    record_property("synapse_directory_before", before)
    async def exchange() -> None:
        alice = _client(live_room.observer, live_room.homeserver)
        bot = _client(live_room.bot, live_room.homeserver)
        try:
            await alice.sync(timeout=0)

            async def send_and_wait(body: str, expected: str) -> None:
                sent = await alice.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": body},
                )
                assert isinstance(sent, RoomSendResponse), sent
                while True:
                    response = await alice.sync(timeout=250)
                    joined = response.rooms.join.get(live_room.room_id)
                    if joined and any(
                        isinstance(event, RoomMessageText)
                        and event.sender == live_room.bot.user_id
                        and event.body == expected
                        for event in joined.timeline.events
                    ):
                        return

            await asyncio.wait_for(
                send_and_wait("Start discovery", "Matrix live reply"), timeout=15
            )
            before = await bot.joined_rooms()
            assert isinstance(before, JoinedRoomsResponse), before
            assert set(before.rooms) == {
                live_room.room_id,
                live_room.ordinary_room,
                live_room.space,
                live_room.bot_only_room,
            }
            gateway.model.push(
                ToolCall(
                    "tool_search",
                    {
                        "queries": [
                            "Matrix joined rooms Spaces user directory discovery"
                        ]
                    },
                ),
                ToolCall("tool_describe", {"names": ["matrix_read"]}),
                ToolCall(
                    "tool_call",
                    {
                        "calls": [
                            {
                                "name": "matrix_read",
                                "arguments": {"kind": "joined_rooms"},
                            }
                        ]
                    },
                    parallel=[
                        (
                            "tool_call",
                            {
                                "calls": [
                                    {
                                        "name": "matrix_read",
                                        "arguments": {"kind": "joined_spaces"},
                                    }
                                ]
                            },
                        ),
                        (
                            "tool_call",
                            {
                                "calls": [
                                    {
                                        "name": "matrix_read",
                                        "arguments": {
                                            "kind": "users",
                                            "search_term": "alice",
                                            "limit": 5,
                                        },
                                    }
                                ]
                            },
                        ),
                    ],
                ),
                Text("Discovery complete"),
            )
            await asyncio.wait_for(
                send_and_wait(
                    "Discover our joined rooms, Spaces and Alice", "Discovery complete"
                ),
                timeout=20,
            )
            requests = gateway.model.main_requests()
            search = [
                json.loads(message["content"])
                for message in requests[2]["messages"]
                if message["role"] == "tool"
            ]
            assert "matrix_read" in search[0]["tools"], search
            described = [
                json.loads(message["content"])
                for message in requests[3]["messages"]
                if message["role"] == "tool"
            ]
            schema = described[1]["tools"]["matrix_read"]["parameters"]
            assert {"joined_rooms", "joined_spaces", "users"} <= set(
                schema["properties"]["kind"]["enum"]
            )
            tools = [
                json.loads(message["content"])
                for message in requests[-1]["messages"]
                if message["role"] == "tool"
            ]
            rooms, spaces, users = tools[2:]

            def summary(room: str, room_type: str | None, name: str) -> dict:
                return {
                    "room_id": room,
                    "room_type": room_type,
                    "name": name,
                    "topic": None,
                    "canonical_alias": None,
                }

            assert rooms == {
                "rooms": sorted(
                    [
                        summary(live_room.room_id, None, "Matrix live test"),
                        summary(live_room.ordinary_room, None, "Joined discovery room"),
                    ],
                    key=lambda room: room["room_id"],
                ),
                "truncated": False,
                "errors": [],
            }
            assert spaces == {
                "rooms": [
                    summary(live_room.space, "m.space", "Joined discovery Space")
                ],
                "truncated": False,
                "errors": [],
            }
            assert users == {
                "users": [
                    {
                        "user_id": live_room.observer.user_id,
                        "display_name": "Alice Discovery",
                        "avatar_url": None,
                    }
                ],
                "truncated": False,
                "errors": [],
            }
            after = await bot.joined_rooms()
            assert isinstance(after, JoinedRoomsResponse), after
            assert set(after.rooms) == set(before.rooms)
            assert live_room.unjoined_room not in after.rooms
        finally:
            await alice.close()
            await bot.close()

    started = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(exchange(), timeout=45))
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
        trace = gateway.home / "discovery-trace.jsonl"
        if trace.exists():
            record_property("discovery_trace", trace.read_text(encoding="utf-8"))
        after = _directory_state(synapse[0])
        record_property("synapse_directory_after", after)
        print("Synapse directory bootstrap:", before, after)
