"""Bot Chat cron delivery: deliver='bot-chat[:<profile>]' injects job output
into a local profile's canonical Bot Chat session as a real inbound turn.

Covers token parsing, target resolution (own profile / named / missing),
preflight exemption, create-time validation, the canonical admission lane,
and the delivery-targets listing used by UI pickers.
"""

from unittest import mock

import pytest

from cron import scheduler_delivery as sched_delivery
from cron.scheduler import _resolve_delivery_targets
from cron.scheduler_delivery import (
    BOT_CHAT_PLATFORM,
    _deliver_to_bot_chat,
    _resolve_bot_chat_target,
    parse_bot_chat_deliver_token,
)
from cron.scheduler_preflight import _preflight_check_delivery


# ── token parsing ────────────────────────────────────────────────────────────

def test_bare_token_targets_own_profile():
    assert parse_bot_chat_deliver_token("bot-chat") == ""
    assert parse_bot_chat_deliver_token("  Bot-Chat  ") == ""


def test_named_token_returns_profile():
    assert parse_bot_chat_deliver_token("bot-chat:research") == "research"
    assert parse_bot_chat_deliver_token("BOT-CHAT:Research") == "Research"


def test_non_bot_chat_tokens_pass_through():
    assert parse_bot_chat_deliver_token("telegram:-100:17") is None
    assert parse_bot_chat_deliver_token("origin") is None
    assert parse_bot_chat_deliver_token("local") is None
    assert parse_bot_chat_deliver_token("all") is None
    # A platform whose name merely CONTAINS bot-chat must not match.
    assert parse_bot_chat_deliver_token("bot-chatter") is None


# ── target resolution ────────────────────────────────────────────────────────

def test_own_profile_resolves_without_name():
    target = _resolve_bot_chat_target({"id": "j1"}, "")
    assert target == {"platform": BOT_CHAT_PLATFORM, "chat_id": "", "thread_id": None}


def test_named_profile_resolves_when_exists():
    with mock.patch("hermes_cli.profiles.profile_exists", return_value=True):
        target = _resolve_bot_chat_target({"id": "j1"}, "research")
    assert target is not None
    assert target["platform"] == BOT_CHAT_PLATFORM
    assert target["chat_id"] == "research"


def test_unknown_profile_resolves_to_none():
    with mock.patch("hermes_cli.profiles.profile_exists", return_value=False):
        assert _resolve_bot_chat_target({"id": "j1"}, "ghost") is None


def test_resolve_delivery_targets_combines_with_platform_targets():
    """bot-chat rides the same comma-separated deliver string as platforms."""
    job = {"id": "j1", "deliver": "bot-chat,telegram"}
    with mock.patch.object(sched_delivery, "_get_home_target_chat_id", return_value="-100123"), \
         mock.patch.object(sched_delivery, "_get_home_target_thread_id", return_value=None), \
         mock.patch.object(sched_delivery, "_is_known_delivery_platform", return_value=True), \
         mock.patch.object(sched_delivery, "_resolve_origin", return_value=None):
        targets = _resolve_delivery_targets(job)
    platforms = {t["platform"] for t in targets}
    assert BOT_CHAT_PLATFORM in platforms
    assert "telegram" in platforms


# ── preflight ────────────────────────────────────────────────────────────────

def test_preflight_ignores_bot_chat_targets():
    """bot-chat needs no gateway credentials — preflight must not block it."""
    assert _preflight_check_delivery({"id": "j1", "deliver": "bot-chat"}) is None
    assert _preflight_check_delivery({"id": "j1", "deliver": "bot-chat:research"}) is None


def test_preflight_still_blocks_unknown_platforms():
    with mock.patch.object(sched_delivery, "_is_known_delivery_platform", return_value=False):
        err = _preflight_check_delivery({"id": "j1", "deliver": "nonexistent-platform"})
    assert err is not None and "not a known" in err


# ── create-time validation ───────────────────────────────────────────────────

def test_create_validation_rejects_unknown_profile():
    from tools.cronjob_tools import _validate_bot_chat_deliver

    with mock.patch("hermes_cli.profiles.profile_exists", return_value=False):
        err = _validate_bot_chat_deliver("bot-chat:ghost")
    assert err is not None
    assert "machine-local" in err


def test_create_validation_accepts_bare_and_existing():
    from tools.cronjob_tools import _validate_bot_chat_deliver

    assert _validate_bot_chat_deliver("bot-chat") is None
    assert _validate_bot_chat_deliver(None) is None
    assert _validate_bot_chat_deliver("telegram:-100") is None
    with mock.patch("hermes_cli.profiles.profile_exists", return_value=True):
        assert _validate_bot_chat_deliver("bot-chat:research") is None


# ── delivery lane ────────────────────────────────────────────────────────────

def test_deliver_message_carries_cron_attribution(tmp_path, monkeypatch):
    """The admitted turn self-identifies as scheduled output, not the user."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    captured = {}

    def fake_deliver(home, owner, message, *, delivery_id):
        captured["message"] = message
        return {"status": "settled", "message": message, "delivery_id": delivery_id}

    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: {"session_id": "local-bot"})
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", fake_deliver)
    assert _deliver_to_bot_chat({"id": "j1", "name": "Daily digest", "execution_id": "r1"}, "the payload", "") is None
    assert 'Cronjob "Daily digest" output' in captured["message"]
    assert "not the user" in captured["message"]
    assert "the payload" in captured["message"]


def test_deliver_without_authority_is_unverified_not_a_second_writer(tmp_path, monkeypatch):
    """No running authority → explicit unverified status; never a local CLI turn."""
    import subprocess
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", mock.Mock(side_effect=AssertionError("no local inference fallback")))
    monkeypatch.setattr(mailbox, "find_canonical_live_owner",
                        mock.Mock(side_effect=ValueError("profile authority is not ready")))
    err = _deliver_to_bot_chat({"id": "j1", "name": "n", "execution_id": "r1"}, "out", "")
    assert err is not None and "unverified" in err and "not ready" in err


# ── delivery-targets listing (UI pickers) ────────────────────────────────────

def test_delivery_targets_include_local_profiles():
    with mock.patch("hermes_cli.profiles.list_profile_names",
                    return_value=["default", "research"]):
        targets = sched_delivery.cron_delivery_targets()
    ids = [t["id"] for t in targets]
    assert f"{BOT_CHAT_PLATFORM}:default" in ids
    assert f"{BOT_CHAT_PLATFORM}:research" in ids
    bot_chat_entries = [t for t in targets if t["id"].startswith(BOT_CHAT_PLATFORM)]
    # No gateway home channel needed for bot-chat targets.
    assert all(t["home_target_set"] for t in bot_chat_entries)
