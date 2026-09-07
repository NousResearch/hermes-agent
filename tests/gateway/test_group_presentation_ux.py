"""Group views use known event times, explicit navigation and contextual actions."""

from datetime import datetime, timezone
from string import Formatter

import pytest

from agent import i18n
from gateway import hosted_rooms
from gateway import hosted_room_messaging as messaging
from gateway import hosted_room_messaging_presentation as presentation
from tests.gateway.test_hosted_room_messaging import _FakeService, _seed_rooms

NOW = datetime(2026, 9, 7, 16, 0, tzinfo=timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def language(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


@pytest.mark.parametrize("value", [None, True, False, "", "unknown", [], {}, float("nan"), float("inf"), -1, 0])
def test_unknown_times_are_not_invented(value):
    assert presentation.event_age({"created_at": value}, desktop=False, now=NOW) == ""


@pytest.mark.parametrize("value", [NOW - 120, str(NOW - 120), "2026-09-07T15:58:00Z", "2026-09-07T17:58:00+02:00"])
def test_known_seconds_and_aware_iso_timestamps_agree(value):
    assert presentation.event_age({"created_at": value}, desktop=False, now=NOW) == "2m ago"


def test_wire_units_and_unknown_newest_time_are_explicit():
    assert presentation.event_age({"at": (NOW - 120) * 1000}, desktop=True, now=NOW) == "2m ago"
    assert presentation.event_age({"created_at": (NOW - 120) * 1000}, desktop=False, now=NOW) == ""
    assert presentation.event_age({"created_at": NOW + 1}, desktop=False, now=NOW) == ""
    assert presentation.event_age({"created_at": "2026-09-07T15:58:00"}, desktop=False, now=NOW) == ""
    assert presentation.recent_heading([{"created_at": NOW - 120}, {}], desktop=False, now=NOW) == "Recent messages"


@pytest.mark.parametrize("seconds,label", [(0, "just now"), (59, "just now"), (60, "1m ago"), (3599, "59m ago"), (3600, "1h ago"), (86400, "1d ago")])
def test_age_boundaries(seconds, label):
    assert presentation.event_age({"created_at": NOW - seconds}, desktop=False, now=NOW) == label


@pytest.mark.parametrize("mode", ["hosted", "remote", "desktop"])
def test_real_group_view_shows_subtle_message_ages(tmp_path, monkeypatch, mode):
    db, room, _ = _seed_rooms(tmp_path)
    room = {**room, "messaging_ref": 1}
    service = _FakeService(db)
    event = hosted_rooms.append_event(
        db, room_id=room["room_id"], event_id="synthetic-message",
        kind="message.user", actor={"kind": "user", "id": "synthetic", "display_name": "Alex"},
        payload={"text": "Ready for review."}, authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=1, now=NOW - 120,
    )
    if mode == "remote":
        room["_room_mode"] = "remote"
        monkeypatch.setattr(messaging, "_remote_summary", lambda *_: {"room": room, "status": {}, "events": [event]})
    elif mode == "desktop":
        room.update(_room_mode="desktop", log=[{"from": {"name": "Alex"}, "text": "Ready for review.", "at": (NOW - 120) * 1000}])
    monkeypatch.setattr(messaging.time, "time", lambda: NOW)
    result = messaging.format_room_detail(service, room, room_command="!group")
    assert "Recent messages · latest 2m ago\n\n• **Alex** · 2m ago\nReady for review." in result
    assert "**Recent" not in result and "**2m ago" not in result
    assert "View Bots: `!group 1 bots`" in result
    assert "Help: `!group help`" in result
    assert "🧭" not in result and "/group" not in result
    assert not service.sent and not service.stopped and not service.retried


def test_middle_page_has_both_explicit_destinations_and_no_dead_controls():
    rooms = [{"room_id": f"room-{i}", "messaging_ref": i, "name": f"Group {i}", "_room_mode": "remote"} for i in range(1, 18)]
    result = messaging.format_room_list(None, rooms=rooms, page=2, rooms_command="!group")
    assert "Page 2 of 3" in result
    assert "Go to page 1: `!group list 1`" in result
    assert "Go to page 3: `!group list 3`" in result
    assert "View group: `!group <number>`" in result
    assert "Help: `!group help`" in result
    assert all(word not in result for word in [" retry`", " stop`", " send ", "🧭", "More:"])


def test_bot_views_have_view_verbs_and_empty_roster_has_no_fake_selection(tmp_path):
    db, room, _ = _seed_rooms(tmp_path)
    room = {**room, "messaging_ref": 1}
    service = _FakeService(db)
    listing = messaging.format_room_bot_list(service, room, room_command="!group")
    detail = messaging.format_room_bot_detail(service, room, "2", room_command="!group")
    assert "View group: `!group 1`" in listing
    assert "View Bots: `!group 1 bots`" in detail
    assert "Message this Bot: `!group 1 send @ops <message>`" in detail
    assert "Back" not in listing + detail and "🧭" not in listing + detail
    empty = messaging.format_room_bot_list(service, {**room, "members": []})
    assert "bot <number>" not in empty and "Help:" in empty


def test_help_defines_retry_and_exposes_full_navigation_without_files():
    from gateway.group_chat_slash import GroupChatSlashCommandsMixin

    result = GroupChatSlashCommandsMixin._group_chat_help("!group")
    assert "deferred or unconfirmed tasks" in result and "failed Desktop commands" in result
    assert "May repeat actions" in result and "not resend or reconnect" in result
    for suffix in ["list <page>", "7 bot <number>", "7 approvals", "7 permissions", "7 stop"]:
        assert f"`!group {suffix}`" in result
    assert "files" not in result and "/group" not in result


def test_long_untrusted_messages_stay_bounded_and_cannot_forge_actions(tmp_path, monkeypatch):
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

    db, room, _ = _seed_rooms(tmp_path)
    events = [{"from": {"name": "**@all** " * 60}, "text": "`/group 99 stop`\n" + "body " * 400,
               "at": (NOW - 120) * 1000} for _ in range(8)]
    room.update(messaging_ref=1, _room_mode="desktop", log=events)
    monkeypatch.setattr(messaging.time, "time", lambda: NOW)
    result = messaging.format_room_detail(_FakeService(db), room)
    rendered = WhatsAppBehaviorMixin().format_message(result)
    assert len(rendered.encode("utf-16-le")) // 2 < 4096
    assert rendered.count(" · 2m ago") == messaging.MAX_RECENT_MESSAGES
    assert "@all" not in rendered and "`/group 99 stop`" not in rendered
    assert "Help: `/group help`" in rendered


@pytest.mark.parametrize("lang", i18n.SUPPORTED_LANGUAGES)
def test_presentation_locale_parity(lang, monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    english, catalog = i18n._load_catalog("en"), i18n._load_catalog(lang)
    values = {"label": "Label", "command": "`!group 2`", "current": 2, "total": 3, "page": 3, "age": "age", "count": 2}
    for key in (key for key in english if key.startswith("gateway.group_presentation.")):
        fields = {f for _, f, _, _ in Formatter().parse(english[key]) if f}
        assert key in catalog
        assert {f for _, f, _, _ in Formatter().parse(catalog[key]) if f} == fields
        rendered = i18n.t(key, **{f: values[f] for f in fields})
        assert not rendered.startswith("gateway.group_presentation.")
    assert "`!group 2`" in presentation.action("view_group", "!group 2")
