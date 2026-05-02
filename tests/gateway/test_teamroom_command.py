from types import SimpleNamespace
import asyncio
import re

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


class FakeFeishuAdapter:
    def __init__(self):
        self.observers = []
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(("send", chat_id, content, metadata))
        return SimpleNamespace(success=True, message_id="m-start")

    async def send_text_with_mentions(self, chat_id, content, mentions=None, metadata=None):
        self.sent.append(("mention", chat_id, content, mentions, metadata))
        room_match = re.search(r"\[TEAMROOM:([^\]]+)\]", content)
        role_match = re.search(r"\[TEAMROOM_TO:([^\]]+)\]", content)
        assert room_match is not None
        assert role_match is not None
        room_id = room_match.group(1)
        role = role_match.group(1)
        payload = {
            "chat_id": chat_id,
            "sender_type": "app",
            "sender_open_id": f"ou_{role}",
            "sender_user_id": "",
            "text": f"[TEAMROOM_REPLY:{room_id}:{role}] {role} answer",
        }
        for observer in list(self.observers):
            observer(payload)
        return SimpleNamespace(success=True, message_id=f"m-{role}")

    def add_inbound_observer(self, callback):
        self.observers.append(callback)

        def remove():
            self.observers.remove(callback)

        return remove


def _runner_with_adapter(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.FEISHU: PlatformConfig(enabled=True)},
    )
    runner.adapters = {Platform.FEISHU: adapter}
    return runner


def _teamroom_event(text="/teamroom ship the feature"):
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_teamroom",
            chat_type="group",
            user_id="ou_user",
            user_name="Tester",
        ),
    )


def test_teamroom_rejects_non_group_chat():
    adapter = FakeFeishuAdapter()
    runner = _runner_with_adapter(adapter)
    event = _teamroom_event()
    event.source.chat_type = "dm"

    result = asyncio.run(runner._handle_teamroom_command(event))

    assert "must be run inside" in result
    assert adapter.sent == []


def test_teamroom_command_registered_for_gateway():
    from hermes_cli.commands import gateway_help_lines, is_gateway_known_command, resolve_command

    command = resolve_command("teamroom")
    assert command is not None
    assert command.gateway_only is True
    assert is_gateway_known_command("teamroom") is True
    assert any("/teamroom" in line for line in gateway_help_lines())


def test_teamroom_collects_three_role_replies(monkeypatch):
    monkeypatch.setenv("FEISHU_TEAMROOM_REPLY_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("FEISHU_TEAMROOM_RESEARCHER_OPEN_ID", "ou_researcher")
    monkeypatch.setenv("FEISHU_TEAMROOM_BUILDER_OPEN_ID", "ou_builder")
    monkeypatch.setenv("FEISHU_TEAMROOM_REVIEWER_OPEN_ID", "ou_reviewer")
    adapter = FakeFeishuAdapter()
    runner = _runner_with_adapter(adapter)

    result = asyncio.run(runner._handle_teamroom_command(_teamroom_event()))

    assert "Teamroom tr-" in result
    assert "## Researcher" in result
    assert "researcher answer" in result
    assert "## Builder" in result
    assert "builder answer" in result
    assert "## Reviewer" in result
    assert "reviewer answer" in result
    assert adapter.observers == []
    mention_messages = [item for item in adapter.sent if item[0] == "mention"]
    assert len(mention_messages) == 3
    assert "[TEAMROOM_TO:researcher]" in mention_messages[0][2]
    assert "[TEAMROOM_TO:builder]" in mention_messages[1][2]
    assert "[TEAMROOM_TO:reviewer]" in mention_messages[2][2]


def test_teamroom_role_config_and_response_helpers(monkeypatch):
    monkeypatch.setenv("FEISHU_TEAMROOM_RESEARCHER_OPEN_ID", "ou_researcher")
    runner = _runner_with_adapter(FakeFeishuAdapter())

    roles = runner._teamroom_roles()

    assert roles[0]["role"] == "researcher"
    assert roles[0]["open_id"] == "ou_researcher"
    assert runner._teamroom_payload_matches(
        {
            "chat_id": "oc_teamroom",
            "sender_type": "app",
            "text": "[TEAMROOM_REPLY:tr-demo:researcher] done",
        },
        chat_id="oc_teamroom",
        room_id="tr-demo",
        role={"role": "researcher"},
    )
    assert runner._teamroom_payload_matches(
        {
            "chat_id": "oc_teamroom",
            "sender_type": "app",
            "text": "[TEAMROOM:tr-demo] researcher done",
        },
        chat_id="oc_teamroom",
        room_id="tr-demo",
        role={"role": "researcher"},
    )
    assert not runner._teamroom_payload_matches(
        {
            "chat_id": "oc_teamroom",
            "sender_type": "app",
            "text": "[TEAMROOM:tr-demo] [TEAMROOM_TO:researcher] reply marker: [TEAMROOM_REPLY:tr-demo:researcher]",
        },
        chat_id="oc_teamroom",
        room_id="tr-demo",
        role={"role": "researcher"},
    )
    assert not runner._teamroom_payload_matches(
        {
            "chat_id": "oc_teamroom",
            "sender_type": "user",
            "text": "[TEAMROOM_REPLY:tr-demo:researcher] done",
        },
        chat_id="oc_teamroom",
        room_id="tr-demo",
        role={"role": "researcher"},
    )
    assert not runner._teamroom_payload_matches(
        {
            "chat_id": "oc_teamroom",
            "sender_type": "app",
            "sender_open_id": "ou_intruder",
            "text": "[TEAMROOM_REPLY:tr-demo:researcher] spoofed",
        },
        chat_id="oc_teamroom",
        room_id="tr-demo",
        role={"role": "researcher", "open_id": "ou_researcher"},
    )
    assert runner._teamroom_payload_matches(
        {
            "chat_id": "oc_teamroom",
            "sender_type": "app",
            "sender_open_id": "ou_researcher",
            "text": "[TEAMROOM_REPLY:tr-demo:researcher] verified",
        },
        chat_id="oc_teamroom",
        room_id="tr-demo",
        role={"role": "researcher", "open_id": "ou_researcher"},
    )
    assert runner._clean_teamroom_response(
        "[TEAMROOM_REPLY:tr-demo:researcher] [TEAMROOM:tr-demo] done"
    ) == "done"
