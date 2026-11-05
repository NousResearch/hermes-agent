"""``gateway.home: bot-chat`` routes an origin-less cron job to the profile's Bot Chat.

The Bot Chat replaces the platform home channels as the primary home: ``deliver=origin`` with no
captured origin lands in it, ``all`` includes it, while an explicit platform target and a job created
in a messaging chat keep going where they say. Read from the profile's real config.yaml.
"""
from unittest import mock

import pytest

from cron import scheduler_delivery as sched_delivery
from cron.scheduler import _resolve_delivery_targets
from cron.scheduler_delivery import BOT_CHAT_PLATFORM

_BOT_CHAT = {"platform": BOT_CHAT_PLATFORM, "chat_id": "", "thread_id": None}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100123")

    def write(home_value):
        gateway = "" if home_value is None else f"gateway:\n  home: {home_value}\n"
        (tmp_path / "config.yaml").write_text(gateway, encoding="utf-8")
    return write


def _targets(deliver, origin=None):
    return [{k: t.get(k) for k in ("platform", "chat_id", "thread_id")}
            for t in _resolve_delivery_targets({"id": "j1", "deliver": deliver, "origin": origin})]


def test_an_origin_less_job_goes_to_the_bot_chat_home_instead_of_telegram(home):
    home("bot-chat")
    assert _targets("origin") == [_BOT_CHAT]


def test_without_the_setting_it_still_falls_back_to_the_telegram_home(home):
    home(None)
    assert [t["platform"] for t in _targets("origin")] == ["telegram"]


def test_all_reaches_the_bot_chat_home_and_the_platform_homes(home):
    home("bot-chat")
    assert [t["platform"] for t in _targets("all")] == [BOT_CHAT_PLATFORM, "telegram"]


def test_explicit_targets_and_messaging_origins_are_unchanged(home):
    home("bot-chat")
    assert _targets("telegram") == [{"platform": "telegram", "chat_id": "-100123", "thread_id": None}]
    origin = {"platform": "telegram", "chat_id": "42", "chat_type": "dm"}
    assert [t["chat_id"] for t in _targets("origin", origin)] == ["42"]


def test_an_unknown_home_value_is_ignored(home):
    home("somewhere")
    assert [t["platform"] for t in _targets("origin")] == ["telegram"]


def test_the_home_routed_result_is_a_bot_chat_turn_not_a_telegram_send(home):
    home("bot-chat")
    job = {"id": "j1", "name": "brief", "deliver": "origin", "origin": None}
    with mock.patch.object(sched_delivery, "_deliver_to_bot_chat", return_value=None) as bot_chat, \
         mock.patch.object(sched_delivery, "_deliver_standalone") as standalone, \
         mock.patch.object(sched_delivery, "_deliver_via_live_adapter") as live:
        assert sched_delivery._deliver_result(job, "the brief") is None
    bot_chat.assert_called_once()
    assert bot_chat.call_args.args[2] == ""
    standalone.assert_not_called()
    live.assert_not_called()
