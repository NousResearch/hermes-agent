"""Independent plain and encrypted clients check turn-boundary Matrix corrections."""

from __future__ import annotations

import asyncio
import copy
import json
from textwrap import dedent

import pytest

from tests.fakes.fake_llm_provider import Text
from tests.integration.matrix_live.conftest import LinuxNioObserver, LiveGateway, LiveRoom, MatrixAccount, _wait_for
from tests.integration.matrix_live.context_client import group_member  # noqa: F401
from tests.integration.matrix_live.edit_client import begin, edits, finish, send


def marker(gateway: LiveGateway, path: str, expected: str | None = None) -> bool:
    result = gateway.container.exec([
        "/opt/hermes/.venv/bin/python", "-c",
        f"from pathlib import Path; p = Path('/opt/data/{path}'); "
        + (f"print(p.exists() and p.read_text(encoding='utf-8').strip() == {expected!r})"
           if expected is not None else "print(p.exists())"),
    ])
    return result.output.decode().strip() == "True"


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
@pytest.mark.parametrize("gateway", ["pause-edit-default", "pause-edit-followups"], indirect=True)
def test_edit_followups_keep_original_thread_and_append_one_new_model_turn(
    gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    group_member: MatrixAccount,
    encrypted: bool,
    request: pytest.FixtureRequest,
) -> None:
    enabled = request.node.callspec.params["gateway"] == "pause-edit-followups"

    async def foreign_message() -> str:
        client = group_member.client(live_room.homeserver)
        try:
            return await send(client, live_room.room_id, {"msgtype": "m.text", "body": "Foreign original"})
        finally:
            await client.close()

    foreign = asyncio.run(asyncio.wait_for(foreign_message(), timeout=15))
    gateway.model.push(Text("Second response"), Text("Correction response"))
    prefix = (
        f"ROOM = {live_room.room_id!r}\nBOT = {live_room.bot.user_id!r}\n"
        f"DEVICE = {live_room.bot.device_id!r}\n"
        + dedent("""
            import asyncio
            import json
            from client import open_encrypted_client
            from edit_client import begin, edits, finish
            async def run(operation):
                client = open_encrypted_client()
                try:
                    return await operation(client)
                finally:
                    await client.close()
        """)
    )

    def encrypted_call(expression: str) -> object:
        output = linux_nio_observer.run_python(
            prefix + f"print(json.dumps(asyncio.run(asyncio.wait_for(run(lambda client: {expression}), timeout=30))))\n"
        )
        return json.loads(output.strip().splitlines()[-1])

    async def plain_begin() -> dict:
        client = live_room.observer.client(live_room.homeserver)
        try:
            return await begin(client, live_room.room_id, live_room.bot.user_id, live_room.bot.device_id, encrypted=False)
        finally:
            await client.close()

    try:
        ids = (encrypted_call("begin(client, ROOM, BOT, DEVICE, encrypted=True)")
               if encrypted else asyncio.run(asyncio.wait_for(plain_begin(), timeout=30)))
        assert isinstance(ids, dict), ids
        _wait_for(lambda: marker(gateway, "context-started"), "blocked second turn")
        first = copy.deepcopy(gateway.model.main_requests())
        assert len(first) == 1

        async def plain_edits() -> list[str]:
            client = live_room.observer.client(live_room.homeserver)
            try:
                return await edits(client, live_room.room_id, live_room.bot.user_id, ids, foreign)
            finally:
                await client.close()

        edit_ids = (encrypted_call(f"edits(client, ROOM, BOT, {ids!r}, {foreign!r})")
                    if encrypted else asyncio.run(asyncio.wait_for(plain_edits(), timeout=15)))
        assert isinstance(edit_ids, list), edit_ids
        _wait_for(
            lambda: marker(gateway, "edits-observed", "\n".join(edit_ids)),
            "all edits admitted or rejected while the turn is blocked",
        )
        assert gateway.model.main_requests() == first
        result = gateway.container.exec([
            "/opt/hermes/.venv/bin/python", "-c",
            "from pathlib import Path; Path('/opt/data/context-release').write_text('release', encoding='utf-8')",
        ])
        assert result.exit_code == 0, result.output

        async def plain_finish() -> list[dict]:
            client = live_room.observer.client(live_room.homeserver)
            try:
                return await finish(client, live_room.room_id, live_room.bot.user_id, ids, encrypted=False, enabled=enabled)
            finally:
                await client.close()

        if encrypted:
            encrypted_call(f"finish(client, ROOM, BOT, {ids!r}, encrypted=True, enabled={enabled!r})")
        else:
            asyncio.run(asyncio.wait_for(plain_finish(), timeout=30))
        requests = gateway.model.main_requests()
        expected_calls = 3 if enabled else 2
        assert (len(requests), len(gateway.model.aux_requests()), len(gateway.model.requests)) == (expected_calls, 0, expected_calls)
        assert requests[:1] == first
        system = [message for message in requests[0]["messages"] if message["role"] == "system"]
        previous_messages = requests[0]["messages"]
        for model_request in requests[1:]:
            messages = model_request["messages"]
            assert [message for message in messages if message["role"] == "system"] == system
            assert messages[:len(previous_messages)] == previous_messages
            roles = [message["role"] for message in messages if message["role"] != "system"]
            assert all(left != right for left, right in zip(roles, roles[1:]))
            previous_messages = messages
        if enabled:
            latest = requests[-1]["messages"][-1]["content"]
            assert "[in:latest]" in latest and "[in:obsolete]" not in latest and "[in:forgery]" not in latest
            assert "Correction to earlier message" in latest
    except (AssertionError, asyncio.TimeoutError) as exc:
        logs = gateway.container.exec([
            "/opt/hermes/.venv/bin/python", "-c",
            "from pathlib import Path; "
            "p = Path('/opt/data/logs/gateway.log'); "
            "print('\\n'.join(p.read_text(errors='replace').splitlines()[-100:]) if p.exists() else 'No gateway log')",
        ])
        pytest.fail(
            f"{exc}\nModel requests:\n{json.dumps(gateway.model.requests, indent=2)}"
            f"\nGateway log tail:\n{logs.output.decode(errors='replace')}",
            pytrace=True,
        )
