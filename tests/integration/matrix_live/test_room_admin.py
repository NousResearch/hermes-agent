"""A separate plain/encrypted client checks model administration and its gates."""

from __future__ import annotations

import asyncio
import json
from textwrap import dedent

import pytest
from nio import JoinResponse, RoomGetStateEventResponse, RoomInviteResponse, RoomLeaveResponse

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.admin_client import prepare, request, send, state
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver, LiveGateway, LiveRoom, MatrixAccount, _register, _wait_for,
)


@pytest.fixture
def gateway_extra_config(request: pytest.FixtureRequest) -> str:
    opt_in = request.node.callspec.params["enabled"]
    return (
        "auxiliary:\n  background_review:\n    enabled: false\n"
        "  title_generation:\n    enabled: false\n    model_upgrade_enabled: false\n"
        + ("platform_toolsets:\n  matrix: [hermes-matrix, matrix_admin]\n"
           "matrix:\n  require_mention: false\n" if opt_in else "")
    )


@pytest.fixture
def invitee(live_room: LiveRoom) -> MatrixAccount:
    return asyncio.run(_register(live_room.homeserver, "invitee"))


@pytest.mark.parametrize("enabled", [True], ids=["opt-in"])
def test_accepted_invite_with_invalid_json_reports_unknown_outcome(
    gateway: LiveGateway, live_room: LiveRoom, invitee: MatrixAccount, enabled: bool,
) -> None:
    from tests.integration.matrix_live.admin_client import power

    async def promote() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            await power(client, live_room.room_id, live_room.observer.user_id, live_room.bot.user_id)
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(promote(), timeout=25))
    code = dedent(f"""\
        import asyncio, importlib, json, tempfile, weakref
        from pathlib import Path
        from aiohttp import ClientSession, web
        from mautrix.client import ClientAPI
        from mautrix.types import DeviceID, UserID
        from gateway.config import PlatformConfig
        from gateway.session_context import clear_session_vars, set_session_vars
        from gateway.session_identity import RoutingIdentity
        from hermes_cli.config import atomic_config_write
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.platforms.matrix.adapter import MatrixAdapter
        from tools.registry import registry

        async def exchange():
            writes = []
            async with ClientSession() as upstream:
                async def forward(request):
                    headers = {{key: value for key, value in request.headers.items() if key.lower() != 'host'}}
                    async with upstream.request(request.method, 'http://synapse:8008' + request.raw_path,
                                                headers=headers, data=await request.read()) as response:
                        body = await response.read()
                        if request.method == 'POST' and request.path.endswith('/invite') and 200 <= response.status < 300:
                            writes.append(response.status)
                            return web.Response(status=response.status, text='<html>', content_type='application/json')
                        return web.Response(status=response.status, body=body,
                                            headers={{'Content-Type': response.headers.get('Content-Type', 'application/json')}})

                app = web.Application()
                app.router.add_route('*', '/{{path:.*}}', forward)
                runner = web.AppRunner(app)
                await runner.setup()
                await web.TCPSite(runner, '127.0.0.1', 0).start()
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        home = Path(directory)
                        atomic_config_write(home / 'config.yaml', {{'platform_toolsets': {{'matrix': ['matrix_admin']}}}})
                        token = set_hermes_home_override(home)
                        client = ClientAPI(UserID({live_room.bot.user_id!r}), DeviceID({live_room.bot.device_id!r}),
                                           base_url=f'http://127.0.0.1:{{runner.addresses[0][1]}}',
                                           token={live_room.bot.access_token!r})
                        adapter = MatrixAdapter(PlatformConfig(enabled=True, token={live_room.bot.access_token!r},
                            extra={{'homeserver': 'http://synapse:8008', 'user_id': {live_room.bot.user_id!r}, 'e2ee_mode': 'off'}}))
                        adapter._client = client
                        adapter._joined_rooms.add({live_room.room_id!r})
                        adapter._allowed_room_ids.clear()
                        adapter.set_authorization_check(lambda user, chat_type, chat_id: user == {live_room.observer.user_id!r})
                        identity = RoutingIdentity('default', 'default', home, home, multiplexed=False,
                                                   transport=weakref.ref(adapter))
                        tokens = set_session_vars(platform='matrix', chat_id={live_room.room_id!r}, chat_type='group',
                            user_id={live_room.observer.user_id!r}, session_key='native-admin-outcome',
                            transport_adapter=adapter, transport_loop=asyncio.get_running_loop(), routing_identity=identity)
                        try:
                            importlib.import_module('tools.matrix_room_tool')
                            result = await asyncio.to_thread(registry.dispatch, 'matrix_room_admin',
                                                             {{'action': 'invite', 'user_id': {invitee.user_id!r}}})
                            print('ADMIN_PROXY_RESULT=' + json.dumps({{'result': json.loads(result), 'writes': writes}}))
                        finally:
                            clear_session_vars(tokens)
                            reset_hermes_home_override(token)
                            await client.api.session.close()
                finally:
                    await runner.cleanup()
        asyncio.run(asyncio.wait_for(exchange(), timeout=30))
    """)
    execution = gateway.container.exec(["/opt/hermes/.venv/bin/python", "-c", code])
    output = execution.output.decode("utf-8", errors="replace")
    assert execution.exit_code == 0, output
    observed = json.loads(output.split("ADMIN_PROXY_RESULT=", 1)[1].splitlines()[0])

    async def membership() -> dict:
        client = live_room.observer.client(live_room.homeserver)
        try:
            return await state(client, live_room.room_id, "m.room.member", invitee.user_id)
        finally:
            await client.close()

    member = asyncio.run(asyncio.wait_for(membership(), timeout=25))
    assert (observed, member["membership"]) == ({
        "result": {
            "error": "Matrix administration failed after the change was sent to the homeserver",
            "outcome": "unknown",
            "next_step": "Ask the user whether the invite arrived before retrying",
        },
        "writes": [200],
    }, "invite")


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
@pytest.mark.parametrize("enabled", [False, True], ids=["default", "opt-in"])
def test_model_admin_uses_owning_client_and_current_room_permissions(
    gateway: LiveGateway, live_room: LiveRoom, linux_nio_observer: LinuxNioObserver,
    invitee: MatrixAccount, encrypted: bool, enabled: bool,
) -> None:
    room, bot, actor = live_room.room_id, live_room.bot.user_id, live_room.observer.user_id

    def call(expression: str):
        if encrypted:
            code = dedent(f"""\
                import asyncio, json
                from client import open_encrypted_client
                from admin_client import prepare, request, send, state, power
                async def exchange():
                    client = open_encrypted_client()
                    try:
                        result = await {expression}
                        print('ADMIN_RESULT=' + json.dumps(result))
                    finally:
                        await client.close()
                asyncio.run(asyncio.wait_for(exchange(), timeout=25))
            """)
            output = linux_nio_observer.run_python(code)
            return json.loads(output.split("ADMIN_RESULT=", 1)[1].splitlines()[0])
        async def exchange():
            client = live_room.observer.client(live_room.homeserver)
            try:
                from tests.integration.matrix_live.admin_client import power
                return await eval(expression, {}, {"client": client, "prepare": prepare, "request": request,
                                                    "send": send, "state": state, "power": power})
            finally:
                await client.close()
        return asyncio.run(asyncio.wait_for(exchange(), timeout=25))

    def action(args: dict, answer: str, *, tool: str = "matrix_room_admin") -> dict:
        gateway.model.push(
            ToolCall("tool_call", {"calls": [{"name": tool, "arguments": args}]}), Text(answer),
        )
        call(f"request(client, {room!r}, {bot!r}, {'Admin '+answer!r}, {answer!r}, encrypted={encrypted!r})")
        return json.loads(gateway.model.main_requests()[-1]["messages"][-1]["content"])

    def tools() -> list[dict]:
        return [json.loads(message["content"]) for message in gateway.model.main_requests()[-1]["messages"]
                if message["role"] == "tool"]

    def check_context() -> None:
        previous = gateway.model.main_requests()[0]["messages"]
        first_system = [message for message in previous if message["role"] == "system"]
        for model_request in gateway.model.main_requests()[1:]:
            messages = model_request["messages"]
            if any(text in json.dumps(messages) for text in ("New private room exchange", "Admin Leave denied")):
                continue
            assert [message for message in messages if message["role"] == "system"] == first_system
            assert messages[:len(previous)] == previous
            roles = [message["role"] for message in messages if message["role"] != "system"]
            assert all(left != right for left, right in zip(roles, roles[1:]))
            previous = messages

    try:
        call(f"prepare(client, {room!r}, {bot!r}, {live_room.bot.device_id!r}, encrypted={encrypted!r})")
        first_reply = call(f"request(client, {room!r}, {bot!r}, 'Begin administration', 'Matrix live reply', encrypted={encrypted!r})")
        gateway.model.push(
            ToolCall("tool_search", {"queries": ["matrix_room_admin", "matrix_pin", "matrix_read"], "limit": 10}),
            Text("Admin catalog"),
        )
        call(f"request(client, {room!r}, {bot!r}, 'Discover administration tools', 'Admin catalog', encrypted={encrypted!r})")
        catalog = tools()[-1]["tools"]
        assert "matrix_read" in catalog, catalog
        assert ("matrix_room_admin" in catalog, "matrix_pin" in catalog) == (enabled, enabled), catalog
        if not enabled:
            rejected = action({"action": "create", "name": "Unavailable room"}, "Admin disabled")
            assert "error" in rejected, rejected
            assert (len(gateway.model.main_requests()), len(gateway.model.aux_requests()), len(gateway.model.requests)) == (5, 0, 5)
            check_context()
            return

        call(f"power(client, {room!r}, {actor!r}, {bot!r}, bot_level=0)")
        denied = action({"action": "invite", "user_id": invitee.user_id}, "Server denied invite")
        server_message = denied.pop("message", "")
        assert (denied, bool(server_message)) == (
            {"error": "Matrix administration failed: MForbidden", "errcode": "M_FORBIDDEN"}, True,
        )
        call(f"power(client, {room!r}, {actor!r}, {bot!r})")
        async def bot_power(actor_level: int):
            from tests.integration.matrix_live.admin_client import power
            client = live_room.bot.client(live_room.homeserver)
            try:
                await power(client, room, actor, bot, actor_level=actor_level)
            finally:
                await client.close()
        call(f"power(client, {room!r}, {actor!r}, {bot!r}, actor_level=0)")
        actor_denied = action({"action": "invite", "user_id": invitee.user_id}, "Actor denied invite")
        assert actor_denied == {"error": "Matrix requester lacks permission to invite users", "required": 50, "level": 0}
        asyncio.run(bot_power(100))
        invited = action({"action": "invite", "user_id": invitee.user_id}, "Invite complete")
        assert invited == {"action": "invite", "room_id": room, "user_id": invitee.user_id}
        assert call(f"state(client, {room!r}, 'm.room.member', {invitee.user_id!r})")["membership"] == "invite"
        async def invitee_membership(join: bool):
            client = invitee.client(live_room.homeserver)
            try:
                response = await (client.join(room) if join else client.room_leave(room))
                assert isinstance(response, JoinResponse if join else RoomLeaveResponse), response
            finally:
                await client.close()
        asyncio.run(invitee_membership(True))
        call(f"power(client, {room!r}, {actor!r}, {bot!r}, actor_level=0)")
        kept = action({"action": "leave"}, "Leave denied")
        assert kept == {"error": "Matrix requester lacks permission to remove the bot from this room",
                        "required": 50, "level": 0}
        asyncio.run(invitee_membership(False))
        asyncio.run(bot_power(100))
        pinned = action({"action": "pin", "event_id": first_reply}, "Pin complete", tool="matrix_pin")
        assert pinned == {"pinned": [first_reply], "state_event_id": pinned["state_event_id"]}
        unpinned = action({"action": "unpin", "event_id": first_reply}, "Unpin complete", tool="matrix_pin")
        assert unpinned == {"pinned": [], "state_event_id": unpinned["state_event_id"]}
        assert call(f"state(client, {room!r}, 'm.room.pinned_events')") == {"pinned": []}
        cross_room = action({"action": "invite", "room_id": "!foreign:matrix.test", "user_id": invitee.user_id}, "Scope denied")
        assert cross_room == {"error": "Matrix administration is limited to the current room"}

        created = action({"action": "create", "name": "Owned private room", "topic": "Admin fixture",
                          "invite": [actor], "encrypted": encrypted}, "Create complete")
        created_room = created["room_id"]
        assert created == {"action": "create", "room_id": created_room, "encrypted": encrypted}
        async def inspect_created():
            client = live_room.observer.client(live_room.homeserver)
            try:
                joined = await client.join(created_room)
                assert isinstance(joined, JoinResponse), joined
                encryption = await client.room_get_state_event(created_room, "m.room.encryption")
                return (await state(client, created_room, "m.room.join_rules"),
                        await state(client, created_room, "m.room.topic"),
                        encryption.content if isinstance(encryption, RoomGetStateEventResponse) else None)
            finally:
                await client.close()
        assert asyncio.run(inspect_created()) == (
            {"join_rule": "invite"}, {"topic": "Admin fixture", "m.topic": {"m.text": [{"body": "Admin fixture"}]}},
            {"algorithm": "m.megolm.v1.aes-sha2"} if encrypted else None,
        )
        call(f"prepare(client, {created_room!r}, {bot!r}, {live_room.bot.device_id!r}, encrypted={encrypted!r})")
        gateway.model.push(Text("New room reply"))
        call(f"request(client, {created_room!r}, {bot!r}, 'New private room exchange', 'New room reply', encrypted={encrypted!r})")

        target = call(f"send(client, {room!r}, 'Redaction target', notice=True)")
        redacted = action({"action": "redact", "event_id": target, "reason": "fixture"}, "Redact complete")
        assert redacted == {"action": "redact", "room_id": room, "event_id": target,
                            "redaction_event_id": redacted["redaction_event_id"]}
        async def read_redacted():
            client = live_room.observer.client(live_room.homeserver)
            try:
                result = await client.room_get_event(room, target)
                return result.event.source["content"]
            finally:
                await client.close()
        assert asyncio.run(read_redacted()) == {}

        before = len(gateway.model.main_requests())
        gateway.model.push(
            ToolCall("tool_call", {"calls": [{"name": "matrix_room_admin", "arguments": {"action": "leave"}}]}),
            ToolCall("tool_call", {"calls": [{"name": "matrix_room_admin", "arguments": {"action": "forget"}}]}),
            Text("Left and forgotten"),
        )
        call(f"send(client, {room!r}, 'Leave and forget this room')")
        _wait_for(lambda: len(gateway.model.main_requests()) == before + 3, "same-turn leave and forget results", timeout=25)
        assert tools()[-2:] == [{"action": "leave", "room_id": room}, {"action": "forget", "room_id": room}]
        async def left():
            client = live_room.bot.client(live_room.homeserver)
            try:
                response = await client.sync(timeout=0)
                return room in response.rooms.join or room in response.rooms.leave
            finally:
                await client.close()
        assert asyncio.run(left()) is False

        async def reinvite():
            client = live_room.observer.client(live_room.homeserver)
            try:
                assert isinstance(await client.room_invite(room, bot), RoomInviteResponse)
                while (await state(client, room, "m.room.member", bot))["membership"] != "join":
                    await asyncio.sleep(0.25)
            finally:
                await client.close()
        asyncio.run(asyncio.wait_for(reinvite(), timeout=25))
        gateway.model.push(Text("Rejoined reply"))
        call(f"prepare(client, {room!r}, {bot!r}, {live_room.bot.device_id!r}, encrypted={encrypted!r})")
        call(f"request(client, {room!r}, {bot!r}, 'Exchange after rejoining', 'Rejoined reply', encrypted={encrypted!r})")
        assert (len(gateway.model.main_requests()), len(gateway.model.aux_requests()), len(gateway.model.requests)) == (26, 0, 26)
        check_context()
    except Exception as exc:
        logs = gateway.container.exec([
            "/opt/hermes/.venv/bin/python", "-c",
            "from pathlib import Path; p = Path('/opt/data/logs/gateway.log'); "
            "print('\\n'.join(p.read_text(errors='replace').splitlines()[-100:]) if p.exists() else 'No gateway log')",
        ])
        output = logs.output.decode(errors="replace")
        pytest.fail(f"{exc}\nModel requests:\n{json.dumps(gateway.model.requests, indent=2)}\nGateway log:\n{output}")
