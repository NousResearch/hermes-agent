"""Chats warned that the gateway is going down are told when it comes back.

Before this, only an in-chat ``/restart`` requester (``.restart_notify.json``) or a configured
home channel learned the gateway was back. A restart from outside the process — ``systemctl
restart``, an update, a reboot — left the active chats that DID get the "going down" notice
silent, so the user had to ask "are you back?".
"""

import json
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
import gateway.shutdown_notice as rs
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def test_marker_round_trip(tmp_path):
    targets = [
        {"platform": "telegram", "chat_id": "8607228256", "thread_id": None},
        {"platform": "whatsapp", "chat_id": "30094903001245@lid", "thread_id": "7"},
    ]
    rs.mark_shutdown_notified(tmp_path, targets)
    assert rs.read_shutdown_notice_marker(tmp_path) == targets
    rs.clear_shutdown_notice_marker(tmp_path)
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_missing_marker_reads_empty(tmp_path):
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_corrupt_marker_reads_empty(tmp_path):
    rs._shutdown_notice_marker(tmp_path).write_text("{not json", encoding="utf-8")
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_marker_without_targets_list_reads_empty(tmp_path):
    rs._shutdown_notice_marker(tmp_path).write_text(json.dumps({"targets": "nope"}), encoding="utf-8")
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_non_dict_entries_are_dropped(tmp_path):
    rs._shutdown_notice_marker(tmp_path).write_text(
        json.dumps({"targets": [{"platform": "telegram", "chat_id": "1"}, "junk", 3]}), encoding="utf-8"
    )
    assert rs.read_shutdown_notice_marker(tmp_path) == [{"platform": "telegram", "chat_id": "1"}]


def test_empty_list_does_not_create_a_marker(tmp_path):
    rs.mark_shutdown_notified(tmp_path, [])
    assert not rs._shutdown_notice_marker(tmp_path).exists()


def test_clear_is_idempotent(tmp_path):
    rs.clear_shutdown_notice_marker(tmp_path)  # no marker yet: must not raise
    rs.mark_shutdown_notified(tmp_path, [{"platform": "telegram", "chat_id": "1"}])
    rs.clear_shutdown_notice_marker(tmp_path)
    rs.clear_shutdown_notice_marker(tmp_path)
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_record_keeps_the_profile_and_unprefixes_the_delivery_key(tmp_path):
    # ``delivery_target_key`` folds a secondary profile into a Telegram PRIVATE chat's platform.
    key = rs.delivery_target_key("telegram", "42", None, profile="work")
    assert key[0] == "work:telegram"
    shared = rs.delivery_target_key("telegram", "-100123", None, profile="work")  # group: profile-free
    rs.record_shutdown_notice(tmp_path, {key: "work", shared: "work", ("discord", "9", "3"): None}, None)
    assert rs.read_shutdown_notice_marker(tmp_path) == [
        {"platform": "telegram", "chat_id": "42", "thread_id": None, "profile": "work"},
        {"platform": "telegram", "chat_id": "-100123", "thread_id": None, "profile": "work"},
        {"platform": "discord", "chat_id": "9", "thread_id": "3"},
    ]


def _runner_with_bots():
    class _Bot:
        def __init__(self):
            self.sent = []

        async def send(self, chat_id, content=None, metadata=None, **kw):
            self.sent.append(chat_id)
            return SimpleNamespace(success=True, message_id="m", error=None)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    runner.adapters = {Platform.TELEGRAM: _Bot()}
    runner._profile_adapters = {"work": {Platform.TELEGRAM: _Bot()}}
    runner._primary_profile_name = "default"
    runner.session_store = None
    runner._session_sources = None
    runner._running_agents = {}
    runner._restart_requested = False
    runner._restart_command_source = None
    return runner


@pytest.mark.asyncio
async def test_secondary_profile_private_chat_gets_the_return_notice_from_its_own_bot(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = _runner_with_bots()
    runner._running_agents["agent:work:telegram:dm:42"] = object()
    runner._snapshot_running_agents = lambda: list(runner._running_agents)

    await runner._notify_active_sessions_of_shutdown()
    work_bot = runner._profile_adapters["work"][Platform.TELEGRAM]
    default_bot = runner.adapters[Platform.TELEGRAM]
    assert work_bot.sent == ["42"]  # the "going down" notice
    work_bot.sent.clear()

    await runner._send_return_notice_to_shutdown_chats()

    assert work_bot.sent == ["42"]
    assert default_bot.sent == []
    assert rs.read_shutdown_notice_marker(tmp_path) == []


@pytest.mark.asyncio
async def test_old_format_entry_without_profile_resolves_to_the_default_bot(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = _runner_with_bots()
    rs.mark_shutdown_notified(tmp_path, [{"platform": "telegram", "chat_id": "42", "thread_id": None}])

    await runner._send_return_notice_to_shutdown_chats()

    assert runner.adapters[Platform.TELEGRAM].sent == ["42"]
    assert runner._profile_adapters["work"][Platform.TELEGRAM].sent == []
