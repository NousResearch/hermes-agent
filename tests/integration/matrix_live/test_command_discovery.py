"""A Matrix client discovers commands through the real gateway without a model turn."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import pytest
from nio import RoomMessageText, RoomSendResponse, SyncResponse

from agent.i18n import t
from hermes_cli.commands import gateway_help_lines
from plugins.platforms.matrix.adapter import _normalize_matrix_bang_command
from tests.integration.matrix_live.conftest import LiveGateway, LiveRoom


_SKILLS = {
    f"discovery-{index:02d}": f"Discovery task {index:02d}" for index in range(12)
}
_COMMAND_SPAN = re.compile(r"`([!/][A-Za-z][A-Za-z0-9_-]*)(?: [^`]*)?`")


@pytest.fixture
def gateway_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    home.mkdir(mode=0o777)
    (home / ".no-bundled-skills").touch()
    for command, description in _SKILLS.items():
        directory = home / "skills" / command
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {command}\ndescription: {description}\n---\n\nResearch.\n",
            encoding="utf-8",
        )
    (home / "skills").chmod(0o777)
    return home


def test_client_discovers_native_commands_without_model_turn(
    gateway: LiveGateway,
    gateway_home: Path,
    live_room: LiveRoom,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.skill_commands import get_skill_commands

    monkeypatch.setenv("HERMES_HOME", str(gateway_home))
    monkeypatch.chdir(gateway_home)
    skills = get_skill_commands()
    assert {
        command: skills[command]["description"]
        for command in (f"/{key}" for key in _SKILLS)
    } == {f"/{command}": description for command, description in _SKILLS.items()}
    skill_tokens = {f"!{command[1:]}" for command in skills}
    builtins = {
        command.replace("/", "!", 1)
        for line in gateway_help_lines()
        for command in _COMMAND_SPAN.findall(line)
    }

    def commands_in(body: str) -> set[str]:
        commands = set(_COMMAND_SPAN.findall(body))
        assert all(command.startswith("!") for command in commands), body
        assert {
            command: _normalize_matrix_bang_command(command) for command in commands
        } == {command: f"/{command[1:]}" for command in commands}
        return commands

    async def exchange() -> None:
        client = live_room.observer.client(live_room.homeserver)
        try:
            response = await client.sync(timeout=0)
            assert isinstance(response, SyncResponse), response

            async def reply_to(command: str) -> str:
                sent = await client.room_send(
                    live_room.room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": command},
                )
                assert isinstance(sent, RoomSendResponse), sent
                deadline = time.monotonic() + 15
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        response = await asyncio.wait_for(
                            client.sync(timeout=250), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        break
                    assert isinstance(response, SyncResponse), response
                    joined = response.rooms.join.get(live_room.room_id)
                    if not joined:
                        continue
                    replies = [
                        event.body
                        for event in joined.timeline.events
                        if isinstance(event, RoomMessageText)
                        and event.sender == live_room.bot.user_id
                    ]
                    if replies:
                        assert len(replies) == 1, replies
                        assert gateway.model.main_requests() == []
                        return replies[0]
                pytest.fail(
                    f"No reply to {command} after 15 seconds. Gateway logs:\n"
                    + gateway.container
                    .get_wrapped_container()
                    .logs()
                    .decode(errors="replace")[-6000:]
                )

            help_body = await reply_to("!help")
            assert commands_in(help_body) == builtins | {
                f"!{command[1:]}" for command in sorted(skills)[:10]
            }

            body = await reply_to("!commands")
            directory = commands_in(body)
            page = 1
            while next_pages := {
                int(value)
                for value in re.findall(r"`!commands (\d+)`", body)
                if int(value) > page
            }:
                page = min(next_pages)
                assert page <= len(builtins) + len(skills)
                body = await reply_to(f"!commands {page}")
                directory.update(commands_in(body))
            assert directory == builtins | skill_tokens

            skills_body = await reply_to("!help skills")
            assert commands_in(skills_body) == skill_tokens
            assert skills_body == "\n".join([
                t("gateway.help.skill_header", count=len(skills)),
                *[
                    f"`!{command[1:]}` — {info.get('description', '').strip()}"
                    for command, info in sorted(skills.items())
                ],
            ]).strip()
            assert gateway.model.main_requests() == []
        finally:
            await client.close()

    asyncio.run(exchange())
