from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "angel-action-packets" / "__init__.py"


def load_plugin():
    module_name = "angel_action_packets_plugin_under_test"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_discord_destination_channel_only():
    plugin = load_plugin()
    dest = plugin.parse_discord_destination("discord:1500166555625984030")
    assert dest.channel_id == "1500166555625984030"
    assert dest.thread_id is None
    assert dest.target_id == "1500166555625984030"


def test_parse_discord_destination_thread_target():
    plugin = load_plugin()
    dest = plugin.parse_discord_destination("discord:1500166555625984030:1503207350637559838")
    assert dest.channel_id == "1500166555625984030"
    assert dest.thread_id == "1503207350637559838"
    assert dest.target_id == "1503207350637559838"


@pytest.mark.parametrize(
    "value",
    [
        "telegram:1",
        "discord:abc",
        "discord:1:bad",
        "discord:1:2:3",
        "discord:",
        "discord:1:",
    ],
)
def test_parse_discord_destination_rejects_invalid(value):
    plugin = load_plugin()
    with pytest.raises(ValueError):
        plugin.parse_discord_destination(value)


class FakeGovernor:
    def __init__(self, has_posted_card: bool = False):
        self.has_posted_card = has_posted_card
        self.sent = []
        self.audits = []
        self.packet = {
            "id": "ACT-test",
            "status": "pending",
            "approval_destination": "discord:111:222",
            "expires_at": "2999-01-01T00:00:00+00:00",
        }

    def list_action_packets(self, status=None):
        assert status == "pending"
        return [{"id": self.packet["id"]}]

    def show_action_packet(self, action_id):
        assert action_id == self.packet["id"]
        return dict(self.packet)


async def _run_poll_once(plugin, governor):
    posted = []

    def fake_has_posted(_governor, action_id):
        assert action_id == "ACT-test"
        return governor.has_posted_card

    async def fake_send(adapter, _governor, packet, destination):
        posted.append((adapter, packet["id"], destination.channel_id, destination.thread_id, destination.target_id))
        return True

    plugin._packet_has_posted_card = fake_has_posted
    plugin.send_action_packet_card_to_destination = fake_send
    return await plugin.poll_once(SimpleNamespace(name="discord"), governor), posted


def test_poll_once_posts_thread_destination_when_no_card_exists():
    plugin = load_plugin()
    count, posted = asyncio.run(_run_poll_once(plugin, FakeGovernor(False)))
    assert count == 1
    assert posted == [(SimpleNamespace(name="discord"), "ACT-test", "111", "222", "222")]


def test_poll_once_skips_packet_when_card_message_already_recorded():
    plugin = load_plugin()
    count, posted = asyncio.run(_run_poll_once(plugin, FakeGovernor(True)))
    assert count == 0
    assert posted == []
