"""A mid-turn message in a shared chat carries its sender, as a turn-opening message does.

In a group or thread several people share one session, so a message that opens a turn is labelled
``[Name] text``. One that lands while a turn runs (busy ``steer`` / ``interrupt`` redirect, an explicit
``/steer``) reached the agent as bare text, so it could not tell who spoke and read a second person's
note (or a bot's post) as an instruction from the person it was helping.
"""

import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Receiver:
    _supports_active_turn_redirect = True
    payload = None

    def steer(self, text):
        self.payload = text
        return True

    redirect = steer


async def _inject(route, source, tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig(group_sessions_per_user=False))
    event = MessageEvent(text="/steer check the logs" if route == "explicit" else "check the logs",
                         source=source, message_id="m2")
    receiver = _Receiver()
    runner._session_state("key").turn.agent = receiver
    if route == "explicit":
        await runner._busy_steer_command(event, "key", source)
    elif route == "priority":
        runner._hm_busy_steer(event, receiver, "key")
    elif route == "priority_redirect":
        await runner._hm_busy_interrupt(event, source, receiver, "key")
    else:
        await runner._resolve_busy_steer_or_redirect(
            event, "key", "interrupt" if route == "redirect" else "steer", receiver)
    return receiver.payload


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["explicit", "priority", "normal", "redirect", "priority_redirect"])
async def test_mid_turn_message_in_a_shared_chat_names_its_sender(route, tmp_path, monkeypatch):
    bot = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U2",
                        user_name="Ticket\nBot", is_bot=True)
    payload = await _inject(route, bot, tmp_path, monkeypatch)
    assert payload.endswith("\n\n[Ticket Bot | Slack user <@U2>] check the logs"), payload
    assert json.loads(payload.splitlines()[1])["from_bot"] is True


@pytest.mark.asyncio
async def test_mid_turn_message_in_a_private_chat_stays_unlabelled(tmp_path, monkeypatch):
    dm = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42", user_name="Ana")
    payload = await _inject("normal", dm, tmp_path, monkeypatch)
    assert payload.endswith("\n\ncheck the logs") and "[Ana]" not in payload
    assert "from_bot" not in json.loads(payload.splitlines()[1])
