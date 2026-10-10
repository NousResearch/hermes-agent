"""A separate client checks cron continuation and room identity changes."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
from nio import (
    JoinResponse,
    RoomCreateResponse,
    RoomMessageText,
    RoomPreset,
    RoomSendResponse,
)
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network

from tests.integration.matrix_live.conftest import (
    LiveRoom,
    _create_live_room,
    _host_route,
    _host_user,
    _register as _register_account,
    _synapse_server,
    _wait_for,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home


_register = partial(_register_account, unique=True)


@pytest.fixture(scope="module")
def synapse(docker_engine: None) -> Iterator[tuple[DockerContainer, str, Network]]:
    with _synapse_server(
        extra_config=(
            "rc_registration:\n  per_second: 100\n  burst_count: 100\n"
            "rc_login:\n  address:\n    per_second: 100\n    burst_count: 100\n"
        )
    ) as server:
        yield server


@pytest.fixture
def live_room(synapse: tuple[DockerContainer, str, Network]) -> LiveRoom:
    return _create_live_room(synapse, unique_accounts=True)


_CONTINUE = """
import asyncio
import json
import signal
from pathlib import Path
from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, load_gateway_config
from gateway.run import GatewayRunner, _profile_runtime_scope
from contextlib import nullcontext
from gateway.session import SessionSource
from gateway.session_context import set_session_vars, clear_session_vars
from tools.send_message_tool import send_message_tool

async def main():
    config = load_gateway_config()
    config.multiplex_profiles = POLICY_DISAGREEMENT
    runner = GatewayRunner(config)
    assert await runner.start(), 'Gateway start failed'
    try:
        loop = asyncio.get_running_loop()
        stopped = asyncio.Event()
        loop.add_signal_handler(signal.SIGTERM, stopped.set)
        scope = _profile_runtime_scope(Path('/opt/data/profiles/secondary'), {}) if POLICY_DISAGREEMENT else nullcontext()
        with scope:
            adapter = runner.adapters[Platform.MATRIX]
            if POLICY_DISAGREEMENT:
                assert runner.session_store.config.thread_sessions_per_user
                assert not load_gateway_config().thread_sessions_per_user
            source = SessionSource(
                platform=Platform.MATRIX, chat_id=ROOM, thread_id=THREAD,
                chat_type='dm' if MEMBERSHIP_CHANGE else 'group', user_id=PARTICIPANT,
                profile='secondary' if POLICY_DISAGREEMENT else None,
            )
            if MEMBERSHIP_CHANGE:
                await adapter._resolve_send_target(TARGET.removeprefix('matrix:').split('/')[0])
                adapter._sync_task.cancel()
                await asyncio.gather(adapter._sync_task, return_exceptions=True)
                from mautrix.types import RoomID
                await adapter._client.get_joined_members(RoomID(ROOM))
                assert await adapter._client.state_store.has_full_member_list(RoomID(ROOM))
                assert (await adapter._resolve_room_identity(ROOM)).chat_type == 'dm'
                original_send = adapter._client.send_message_event
                first_send = True
                async def paused_send(*args, **kwargs):
                    nonlocal first_send
                    if first_send:
                        first_send = False
                        print(json.dumps({'send_paused': True}), flush=True)
                        async with asyncio.timeout(10):
                            while not Path('/opt/data/release-send').exists():
                                await asyncio.sleep(0.01)
                    return await original_send(*args, **kwargs)
                adapter._client.send_message_event = paused_send

            if ABSENT_PARTICIPANT:
                bob = runner.session_store.get_or_create_session(SessionSource(
                    platform=Platform.MATRIX, chat_id=ROOM, thread_id=THREAD,
                    chat_type='group', user_id=PEER,
                ))
                runner.session_store.append_to_transcript(bob.session_id, {
                    'role': 'assistant', 'content': "Bob's isolated context",
                })
                bob_before = runner.session_store.load_transcript(bob.session_id)
            elif PER_USER:
                entry = runner.session_store.get_or_create_session(source)
                runner.session_store.append_to_transcript(entry.session_id, {
                    'role': 'user', 'content': 'Remember the existing participant context.',
                })
                runner.session_store.append_to_transcript(entry.session_id, {
                    'role': 'assistant', 'content': 'Existing participant context',
                })
            if PER_USER and not ABSENT_PARTICIPANT:
                before = runner.session_store.load_transcript(entry.session_id)
                runner.session_store._db.update_system_prompt(entry.session_id, 'Existing cached prefix')
            if TOOL_SEND:
                tokens = set_session_vars(platform='matrix', user_id=PARTICIPANT)
                try:
                    result = json.loads(await asyncio.to_thread(
                        send_message_tool, {'target': TARGET, 'message': BRIEF},
                    ))
                finally:
                    clear_session_vars(tokens)
                assert result.get('success'), result
                assert bool(result.get('mirrored')) == (not ABSENT_PARTICIPANT and not MEMBERSHIP_CHANGE), result
                assert (result['chat_id'], result['thread_id']) == (ROOM, THREAD), result
            else:
                http_calls = []
                if HTTP_FALLBACK:
                    await adapter._resolve_send_target(TARGET.removeprefix('matrix:').split('/')[0])
                    from tools import send_message_senders
                    from plugins.platforms.matrix.adapter import MatrixAdapter
                    from plugins.platforms.matrix.standalone import _HTTPDelivery
                    original_http = _HTTPDelivery.request
                    async def observe_http(self, method, path, **kwargs):
                        result = await original_http(self, method, path, **kwargs)
                        http_calls.append((method, path))
                        return result
                    async def unavailable_native(self):
                        return False
                    send_message_senders._live_adapter = lambda *args, **kwargs: (None, None)
                    MatrixAdapter.connect = unavailable_native
                    _HTTPDelivery.request = observe_http
                if TRANSIENT_RESOLUTION:
                    adapter = runner.adapters[Platform.MATRIX]
                    original = adapter._client.resolve_room_alias
                    first = True
                    async def transient(alias):
                        nonlocal first
                        if first:
                            first = False
                            raise ValueError('transient native alias lookup failure')
                        return await original(alias)
                    adapter._client.resolve_room_alias = transient
                error = await asyncio.to_thread(
                    _deliver_result,
                    {
                        'id': 'alias-continuation', 'deliver': TARGET, 'attach_to_session': True,
                        'origin': source.to_dict() if PER_USER else None,
                    },
                    BRIEF, runner.adapters, loop,
                )
                assert error is None, error
                if HTTP_FALLBACK:
                    assert len([path for method, path in http_calls if method == 'PUT' and '/send/' in path]) == 1, http_calls
                    assert len([path for method, path in http_calls if path.endswith('joined_members')]) == 2, http_calls
            if ABSENT_PARTICIPANT:
                assert list(runner.session_store._entries) == [bob.session_key]
                assert runner.session_store.load_transcript(bob.session_id) == bob_before
            elif PER_USER:
                assert list(runner.session_store._entries) == [entry.session_key]
                assert runner.session_store._db.get_session(entry.session_id)['system_prompt'] == 'Existing cached prefix'
                if MEMBERSHIP_CHANGE:
                    assert runner.session_store.load_transcript(entry.session_id) == before
            if MEMBERSHIP_CHANGE:
                assert adapter._sync_task.done()
                adapter._sync_task = asyncio.create_task(adapter._sync_loop())
            print(json.dumps({'cron_ready': True}), flush=True)
            await stopped.wait()
    finally:
        await runner.stop()

asyncio.run(main())
"""


@pytest.mark.parametrize(
    "delivery",
    [
        "shared_cron",
        "per_user_cron",
        "per_user_tool",
        "per_user_tool_absent",
        "per_user_cron_absent",
        "per_user_fallback_cron",
        "per_user_http_fallback",
        "per_user_http_fallback_absent",
        "policy_tool",
        "policy_fallback",
        "membership_tool",
        "membership_cron",
        "membership_fallback",
    ],
)
def test_alias_thread_reply_receives_seeded_cron_brief(
    gateway_image, synapse, live_room: LiveRoom, tmp_path: Path, delivery: str
):
    alias_localpart = f"cron-continuation-{delivery}"
    membership_change = delivery.startswith("membership_")
    tool_send = delivery.endswith("tool") or delivery.startswith("per_user_tool")
    transient_resolution = delivery in {
        "per_user_fallback_cron",
        "per_user_http_fallback",
        "per_user_http_fallback_absent",
        "policy_fallback",
        "membership_fallback",
    }

    async def create():
        client = live_room.observer.client(live_room.homeserver)
        try:
            room = await client.room_create(
                alias=alias_localpart,
                name="Cron continuation",
                preset=RoomPreset.public_chat,
            )
            assert isinstance(room, RoomCreateResponse), room
            peer = await _register(live_room.homeserver, "cron-peer")
            peer_client = peer.client(live_room.homeserver)
            try:
                if not membership_change:
                    joined = await peer_client.join(room.room_id)
                    assert isinstance(joined, JoinResponse), joined
            finally:
                await peer_client.close()
            root = await client.room_send(
                room.room_id,
                "m.room.message",
                {"msgtype": "m.notice", "body": "External thread root"},
            )
            assert isinstance(root, RoomSendResponse), root
            return room.room_id, root.event_id, peer
        finally:
            await client.close()

    room_id, root_id, peer = asyncio.run(asyncio.wait_for(create(), timeout=15))
    peer_id = peer.user_id
    alias = f"#{alias_localpart}:matrix.test"
    brief = "The cron-only launch code is basil-otter-47."
    home = tmp_path / "continuation-home"
    _, _, network = synapse
    route = _host_route(network)
    with FakeLLMServer(
        [Text("Continuation reply")], bind_host=route.bind_host
    ) as model:
        write_hermes_home(
            home,
            f"http://host.docker.internal:{model.port}/v1",
            extra_config=(
                "cron:\n  wrap_response: false\n"
                f"thread_sessions_per_user: {str(delivery != 'shared_cron').lower()}\n"
                "updates:\n  check: false\n"
                "platforms:\n  matrix:\n    enabled: true\n"
                "    extra:\n      e2ee_mode: 'off'\n      require_mention: false\n"
                "      auto_thread: false\n      reactions: false\n"
                + (
                    "multiplex_profiles: true\nprofile_routes:\n"
                    "  - name: continuation-secondary\n    platform: matrix\n"
                    f"    chat_id: '{room_id}'\n    profile: secondary\n"
                    if delivery.startswith("policy_")
                    else ""
                )
            ),
        )
        if delivery.startswith("policy_"):
            write_hermes_home(
                home / "profiles" / "secondary",
                f"http://host.docker.internal:{model.port}/v1",
                extra_config=(
                    "thread_sessions_per_user: false\ncron:\n  wrap_response: false\n"
                    "updates:\n  check: false\nplatforms:\n  matrix:\n    enabled: true\n"
                    "    extra:\n      e2ee_mode: 'off'\n"
                ),
            )
        with (home / ".env").open("a", encoding="utf-8") as stream:
            stream.write(
                "MATRIX_HOMESERVER=http://synapse:8008\n"
                f"MATRIX_ACCESS_TOKEN={live_room.bot.access_token}\n"
                f"MATRIX_ALLOWED_USERS={live_room.observer.user_id}\n"
            )
        code = (
            f"TARGET = {f'matrix:{alias}/{root_id}'!r}\nBRIEF = {brief!r}\n"
            f"ROOM = {room_id!r}\nTHREAD = {root_id!r}\n"
            f"PARTICIPANT = {live_room.observer.user_id!r}\n"
            f"PER_USER = {delivery != 'shared_cron'!r}\n"
            f"TOOL_SEND = {tool_send!r}\n"
            f"ABSENT_PARTICIPANT = {delivery.endswith('_absent')!r}\n"
            f"TRANSIENT_RESOLUTION = {transient_resolution!r}\n"
            f"HTTP_FALLBACK = {'http_fallback' in delivery!r}\n"
            f"MEMBERSHIP_CHANGE = {membership_change!r}\n"
            f"POLICY_DISAGREEMENT = {delivery.startswith('policy_')!r}\n"
            f"PEER = {peer_id!r}\n" + _CONTINUE
        )
        with (
            DockerContainer(
                gateway_image,
                network=network,
                entrypoint="/opt/hermes/.venv/bin/python",
                user=_host_user(),
                working_dir="/opt/hermes",
                extra_hosts={"host.docker.internal": route.container_address},
            )
            .with_command(["-c", code])
            .with_volume_mapping(home, "/opt/data", "rw")
            .with_env("HOME", "/opt/data") as sender
        ):

            def logs():
                return sender.get_wrapped_container().logs().decode(errors="replace")

            if membership_change:
                _wait_for(
                    lambda: '{"send_paused": true}' in logs(),
                    "paused native send",
                    timeout=120,
                    details=logs,
                )

                async def join_peer():
                    client = peer.client(live_room.homeserver)
                    try:
                        joined = await client.join(room_id)
                        assert isinstance(joined, JoinResponse), joined
                    finally:
                        await client.close()

                asyncio.run(asyncio.wait_for(join_peer(), timeout=5))
                (home / "release-send").touch()
            _wait_for(
                lambda: '{"cron_ready": true}' in logs(),
                "cron delivery and seed",
                timeout=120,
                details=lambda: logs()[-6000:],
            )

            async def exchange():
                client = live_room.observer.client(live_room.homeserver)
                try:
                    response = await client.sync(timeout=0)
                    reports = [
                        event
                        for event in response.rooms.join[room_id].timeline.events
                        if isinstance(event, RoomMessageText) and event.body == brief
                    ]
                    assert len(reports) == 1
                    report = reports[0]
                    assert (
                        report.sender,
                        report.source["content"]["m.relates_to"],
                    ) == (
                        live_room.bot.user_id,
                        {
                            "rel_type": "m.thread",
                            "event_id": root_id,
                            "is_falling_back": True,
                            "m.in_reply_to": {"event_id": root_id},
                        },
                    )
                    question = "What was the launch code in your scheduled brief?"
                    sent = await client.room_send(
                        room_id,
                        "m.room.message",
                        {
                            "msgtype": "m.text",
                            "body": question,
                            "m.relates_to": {
                                "rel_type": "m.thread",
                                "event_id": root_id,
                                "is_falling_back": False,
                                "m.in_reply_to": {"event_id": report.event_id},
                            },
                        },
                    )
                    assert isinstance(sent, RoomSendResponse), sent
                    while True:
                        response = await client.sync(timeout=250)
                        joined = response.rooms.join.get(room_id)
                        if not joined:
                            continue
                        for event in joined.timeline.events:
                            if (
                                isinstance(event, RoomMessageText)
                                and event.sender == live_room.bot.user_id
                                and event.body == "Continuation reply"
                            ):
                                assert event.source["content"]["m.relates_to"] == {
                                    "rel_type": "m.thread",
                                    "event_id": root_id,
                                    "m.in_reply_to": {"event_id": sent.event_id},
                                    "is_falling_back": False,
                                }
                                requests = model.main_requests()
                                assert len(requests) == 1
                                messages = requests[0]["messages"]
                                brief_role = "assistant" if tool_send else "user"
                                mirrored = any(
                                    message.get("role") == brief_role
                                    and brief in str(message.get("content"))
                                    and (
                                        tool_send
                                        or "[Cron delivery: alias-continuation]"
                                        in str(message.get("content"))
                                    )
                                    for message in messages
                                )
                                assert mirrored == (
                                    not delivery.endswith("_absent")
                                    and not membership_change
                                ), messages
                                assert "Bob's isolated context" not in json.dumps(
                                    messages
                                )
                                if (
                                    not tool_send
                                    and not membership_change
                                    and not delivery.endswith("_absent")
                                ):
                                    assert (
                                        "[Cron delivery: alias-continuation]"
                                        in json.dumps(messages)
                                    )
                                if (
                                    delivery != "shared_cron"
                                    and not delivery.endswith("_absent")
                                    and not membership_change
                                ):
                                    assert "Existing participant context" in json.dumps(
                                        messages
                                    )
                                assert question in json.dumps(messages)
                                if membership_change:
                                    current = messages[-1]
                                    assert current["role"] == "user"
                                    assert question in str(current["content"])
                                    assert brief in str(current["content"])
                                    assert (
                                        "[Cron delivery: alias-continuation]"
                                        not in str(current["content"])
                                    )
                                return
                finally:
                    await client.close()

            try:
                asyncio.run(asyncio.wait_for(exchange(), timeout=15))
            except asyncio.TimeoutError:
                pytest.fail(
                    "No continuation reply after 15 seconds. Gateway logs:\n"
                    + logs()[-6000:]
                )
    # Fails the test if the container left files that this user cannot delete.
    shutil.rmtree(home)
