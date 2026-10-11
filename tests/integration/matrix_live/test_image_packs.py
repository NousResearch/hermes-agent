"""Separate clients verify standard and legacy packs and selected native stickers."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from textwrap import dedent

import pytest

from tests.fakes.fake_llm_provider import Text, ToolCall
from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
)


@pytest.mark.parametrize("gateway", ["image-packs"], indirect=True)
@pytest.mark.parametrize("transport", ["plain", "encrypted", "transition"])
def test_current_bot_packs_send_native_sticker_with_thread_and_reply_context(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    transport: str,
    record_property: Callable[[str, object], None],
) -> None:
    started = time.monotonic()
    encrypted = transport == "encrypted"
    prefix = (
        "import asyncio, json\nfrom image_packs_client import prepare, ask, reply_to_sticker, enable_encryption\n"
        f"ROOM={live_room.room_id!r}\nBOT={live_room.bot.user_id!r}\nDEVICE={live_room.bot.device_id!r}\n"
        f"BOT_TOKEN={live_room.bot.access_token!r}\nENCRYPTED={encrypted!r}\n"
    )
    try:
        output = linux_nio_observer.run_python(
            prefix
            + (
                "result = asyncio.run(asyncio.wait_for(prepare(ROOM, BOT, DEVICE, BOT_TOKEN, encrypted=ENCRYPTED), timeout=20))\n"
                "print(json.dumps(result))\n"
            )
        )
        prepared = json.loads(output.strip().splitlines()[-1])
        prefix += f"ROOT={prepared['root']!r}\nURL={prepared['url']!r}\n"
        gateway.model.push(
            ToolCall("tool_search", {"queries": ["Matrix image pack stickers"]}),
            ToolCall(
                "tool_call",
                {
                    "calls": [
                        {"name": "matrix_image_packs", "arguments": {"action": "list"}}
                    ]
                },
            ),
            Text("Packs listed"),
        )
        linux_nio_observer.run_python(
            prefix
            + (
                "asyncio.run(asyncio.wait_for(ask(ROOM, BOT, ROOT, 'List sticker packs', 'Packs listed', encrypted=ENCRYPTED), timeout=15))\n"
            )
        )
        requests = gateway.model.main_requests()
        assert len(requests) == 4
        results = [
            json.loads(message["content"])
            for message in requests[-1]["messages"]
            if message["role"] == "tool"
        ]
        catalog = results[-1]
        assert {key: value for key, value in catalog.items() if key != "packs"} == {
            "errors": [],
            "truncated": False,
            "untrusted_data": True,
            "account_user_id": live_room.bot.user_id,
        }
        assert sorted(
            (
                pack["source"],
                pack["event_type"],
                pack["state_key"],
                pack["account_user_id"],
                pack["items"][0]["body"],
            )
            for pack in catalog["packs"]
        ) == sorted([
            ("room", "m.room.image_pack", "", None, "Room fox.png"),
            ("room", "m.room.image_pack", "named", None, "Referenced fox.png"),
            ("room", "im.ponies.room_emotes", "legacy", None, "Legacy fox.png"),
            (
                "account_reference",
                "m.room.image_pack",
                "named",
                live_room.bot.user_id,
                "Referenced fox.png",
            ),
            (
                "account_reference",
                "im.ponies.room_emotes",
                "legacy",
                live_room.bot.user_id,
                "Legacy fox.png",
            ),
            (
                "bot_account",
                "im.ponies.user_emotes",
                None,
                live_room.bot.user_id,
                "Bot private fox.png",
            ),
        ])
        selected = next(
            pack["items"][0]
            for pack in catalog["packs"]
            if pack["source"] == "account_reference"
            and pack["event_type"] == "m.room.image_pack"
        )
        if transport == "transition":
            linux_nio_observer.run_python(
                prefix
                + "asyncio.run(asyncio.wait_for(enable_encryption(ROOM, BOT, DEVICE), timeout=15))\n"
            )
            prefix += "ENCRYPTED=True\n"
        gateway.model.push(
            ToolCall(
                "tool_call",
                {
                    "calls": [
                        {
                            "name": "matrix_image_packs",
                            "arguments": {
                                "action": "send",
                                "selection_id": selected["selection_id"],
                            },
                        }
                    ]
                },
            ),
            Text("Sticker sent"),
        )
        output = linux_nio_observer.run_python(
            prefix
            + (
                "result = asyncio.run(asyncio.wait_for(ask(ROOM, BOT, ROOT, 'Send the selected sticker', 'Sticker sent', "
                "encrypted=ENCRYPTED, sticker_body='Referenced fox.png', url=URL), timeout=15))\nprint(json.dumps(result))\n"
            )
        )
        sent = json.loads(output.strip().splitlines()[-1])
        requests = gateway.model.main_requests()
        assert len(requests) == 6
        tools = [
            json.loads(message["content"])
            for message in requests[-1]["messages"]
            if message["role"] == "tool"
        ]
        assert tools[-1] == {"success": True, "event_id": sent["sticker"]}
        gateway.model.push(Text("Sticker context checked"))
        linux_nio_observer.run_python(
            prefix
            + (
                f"STICKER={sent['sticker']!r}\n"
                "asyncio.run(asyncio.wait_for(reply_to_sticker(ROOM, BOT, ROOT, STICKER, encrypted=ENCRYPTED), timeout=15))\n"
            )
        )
        requests = gateway.model.main_requests()
        assert (
            len(requests),
            len(gateway.model.aux_requests()),
            len(gateway.model.requests),
        ) == (7, 0, 7)
        inbound = requests[-1]["messages"][-1]["content"]
        assert "Referenced fox.png" in json.dumps(inbound)
        assert "Requester private sentinel" not in json.dumps(requests)
    except Exception as exc:
        requests = gateway.model.main_requests()
        latest_tools = [
            message
            for message in (requests[-1]["messages"] if requests else [])
            if message["role"] == "tool"
        ]
        gateway_output = (
            gateway.container
            .get_wrapped_container()
            .logs()
            .decode(errors="replace")[-6000:]
        )
        gateway_log = gateway.container.exec([
            "tail",
            "-c",
            "6000",
            "/opt/data/logs/gateway.log",
        ]).output.decode(errors="replace")
        raise AssertionError(
            f"{exc}\nLatest model tool outputs: {latest_tools!r}\n"
            f"Gateway log tail:\n{gateway_log}\nGateway output tail:\n{gateway_output}"
        ) from exc
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))


@pytest.mark.parametrize("gateway", ["image-packs"], indirect=True)
def test_accepted_sticker_with_invalid_json_reports_unknown_outcome(
    gateway: LiveGateway, live_room: LiveRoom,
) -> None:
    from nio import RoomGetEventResponse, RoomPutStateResponse

    async def prepare() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            response = await client.room_put_state(live_room.room_id, "m.room.image_pack", {
                "pack": {"display_name": "Outcome pack", "usage": ["sticker"]},
                "images": {"outcome": {
                    "url": "mxc://matrix.test/outcome", "body": "Outcome sticker",
                    "info": {"mimetype": "image/png", "size": 1},
                }},
            })
            assert isinstance(response, RoomPutStateResponse), response
        finally:
            await client.close()

    asyncio.run(asyncio.wait_for(prepare(), timeout=25))
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
                        if request.method == 'PUT' and '/send/m.sticker/' in request.path and 200 <= response.status < 300:
                            writes.append({{'status': response.status, 'event_id': json.loads(body)['event_id']}})
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
                        atomic_config_write(home / 'config.yaml', {{'platform_toolsets': {{'matrix': ['matrix_image_packs']}}}})
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
                            user_id={live_room.observer.user_id!r}, session_key='native-image-pack-outcome',
                            session_id='native-image-pack-conversation', transport_adapter=adapter,
                            transport_loop=asyncio.get_running_loop(), routing_identity=identity)
                        try:
                            importlib.import_module('tools.matrix_image_packs_tool')
                            catalog = json.loads(await asyncio.to_thread(registry.dispatch, 'matrix_image_packs', {{'action': 'list'}}))
                            selection = next(pack['items'][0]['selection_id'] for pack in catalog['packs']
                                             if pack['source'] == 'room' and pack['event_type'] == 'm.room.image_pack'
                                             and pack['state_key'] == '')
                            result = await asyncio.to_thread(registry.dispatch, 'matrix_image_packs',
                                                             {{'action': 'send', 'selection_id': selection}})
                            print('PACK_PROXY_RESULT=' + json.dumps({{'result': json.loads(result), 'writes': writes}}))
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
    observed = json.loads(output.split("PACK_PROXY_RESULT=", 1)[1].splitlines()[0])
    assert len(observed["writes"]) == 1, observed
    event_id = observed["writes"][0]["event_id"]

    async def confirmed_event() -> dict:
        client = live_room.observer.client(live_room.homeserver)
        try:
            response = await client.room_get_event(live_room.room_id, event_id)
            assert isinstance(response, RoomGetEventResponse), response
            return response.event.source
        finally:
            await client.close()

    confirmed = asyncio.run(asyncio.wait_for(confirmed_event(), timeout=25))
    assert (observed, confirmed["type"], confirmed["sender"], confirmed["content"]["body"]) == ({
        "result": {
            "error": "Matrix sticker send failed: JSONDecodeError",
            "outcome": "unknown",
            "next_step": "Check the room before retrying the sticker send",
        },
        "writes": [{"status": 200, "event_id": event_id}],
    }, "m.sticker", live_room.bot.user_id, "Outcome sticker")
