from datetime import datetime, timezone

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.important_contacts import is_important_sender
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.unread_important_messages import (
    ImportantUnreadNudgeScheduler,
    evaluate_unread_important_message,
)


def _source(
    *,
    platform: Platform = Platform.SMS,
    chat_id: str = "sms-important-user",
    user_id: str | None = "sms-important-user",
    user_name: str | None = "Fred",
    chat_id_alt: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
        user_name=user_name,
        chat_id_alt=chat_id_alt,
    )


def test_gateway_config_loads_top_level_important_contacts(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "important_contacts:\n"
        "  contacts:\n"
        "    - platform: sms\n"
        "      user_ids: ['sms-important-user']\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config.important_contacts == {
        "contacts": [{"platform": "sms", "user_ids": ["sms-important-user"]}]
    }


def test_gateway_config_loads_nested_important_contacts(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "gateway:\n"
        "  important_contacts:\n"
        "    contacts:\n"
        "      - platform: bluebubbles\n"
        "        user_ids: ['fred@example.com']\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config.important_contacts == {
        "contacts": [{"platform": "bluebubbles", "user_ids": ["fred@example.com"]}]
    }


def test_important_sender_matches_explicit_platform_user_id():
    config = {
        "contacts": [
            {"platform": "sms", "user_ids": ["sms-important-user"]},
        ]
    }

    assert is_important_sender(_source(), config)


def test_important_sender_fails_closed_for_ambiguous_identity():
    config = {
        "contacts": [
            {"platform": "sms", "user_names": ["Fred"]},
        ]
    }

    source = _source(user_id=None, chat_id="", user_name="Fred")

    assert not is_important_sender(source, config)


def test_important_sender_does_not_match_spoofable_display_name_only():
    config = {
        "contacts": [
            {"platform": "sms", "user_names": ["Fred"]},
        ]
    }

    source = _source(user_id="sms-other-user", chat_id="sms-other-user", user_name="Fred")

    assert not is_important_sender(source, config)


def test_unread_text_from_important_contact_returns_internal_nudge_event():
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "sms", "user_ids": ["sms-important-user"]},
                ]
            }
        }
    )
    event = MessageEvent(
        text="please call me",
        message_type=MessageType.TEXT,
        source=_source(),
        raw_message={"MessageSid": "SMabc123"},
        message_id="SMabc123",
        timestamp=datetime(2026, 5, 22, tzinfo=timezone.utc),
    )

    result = evaluate_unread_important_message(event, gateway_config.important_contacts)

    assert result.eligible is True
    assert result.reason == "eligible"
    assert result.internal_event is not None
    assert result.internal_event.stable_message_id == "SMabc123"
    assert result.internal_event.dedupe_key == "sms:SMabc123"
    assert result.internal_event.platform == "sms"
    assert result.internal_event.sender_id == "sms-important-user"


def test_non_important_contact_does_not_enter_nudge_path():
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "sms", "user_ids": ["sms-other-user"]},
                ]
            }
        }
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="SMignored",
    )

    result = evaluate_unread_important_message(event, gateway_config.important_contacts)

    assert result.eligible is False
    assert result.reason == "not_important_sender"
    assert result.internal_event is None


def test_non_contact_urgent_actionable_message_can_enter_nudge_path_when_scoring_enabled():
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "sms", "user_ids": ["sms-other-user"]},
                ],
                "importance_scoring": {"enabled": True, "threshold": 70},
            }
        }
    )
    event = MessageEvent(
        text="Urgent — can you call me ASAP? I need help with pickup.",
        message_type=MessageType.TEXT,
        source=_source(user_id="sms-low-volume-user", chat_id="sms-low-volume-user"),
        message_id="SMurgent",
    )

    result = evaluate_unread_important_message(event, gateway_config.important_contacts)

    assert result.eligible is True
    assert result.reason == "scored_important"
    assert result.internal_event is not None
    assert result.internal_event.dedupe_key == "sms:SMurgent"


def test_non_contact_chitchat_stays_out_of_nudge_path_when_scoring_enabled():
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "sms", "user_ids": ["sms-other-user"]},
                ],
                "importance_scoring": {"enabled": True, "threshold": 70},
            }
        }
    )
    event = MessageEvent(
        text="lol that was funny",
        message_type=MessageType.TEXT,
        source=_source(user_id="sms-low-volume-user", chat_id="sms-low-volume-user"),
        message_id="SMcasual",
    )

    result = evaluate_unread_important_message(event, gateway_config.important_contacts)

    assert result.eligible is False
    assert result.reason == "not_important_sender"
    assert result.internal_event is None


def test_read_bluebubbles_message_is_not_eligible_for_nudging():
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "bluebubbles", "user_ids": ["fred@example.com"]},
                ]
            }
        }
    )
    event = MessageEvent(
        text="already read",
        message_type=MessageType.TEXT,
        source=_source(
            platform=Platform.BLUEBUBBLES,
            chat_id="iMessage;-;fred@example.com",
            user_id="fred@example.com",
        ),
        raw_message={"data": {"guid": "bb-guid", "dateRead": 1779480000}},
        message_id="bb-guid",
    )

    result = evaluate_unread_important_message(event, gateway_config.important_contacts)

    assert result.eligible is False
    assert result.reason == "already_read"
    assert result.internal_event is None


def test_missing_message_id_gets_safe_unstable_dedupe_key():
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "sms", "user_ids": [_source().user_id]},
                ]
            }
        }
    )
    event = MessageEvent(
        text="no sid yet",
        message_type=MessageType.TEXT,
        source=_source(),
        timestamp=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
    )

    result = evaluate_unread_important_message(event, gateway_config.important_contacts)

    assert result.eligible is True
    assert result.internal_event is not None
    assert result.internal_event.stable_message_id is None
    assert result.internal_event.dedupe_key.startswith("sms:unstable:")


def _important_unread_event(message_id: str = "SMabc123"):
    gateway_config = GatewayConfig.from_dict(
        {
            "important_contacts": {
                "contacts": [
                    {"platform": "sms", "user_ids": [_source().user_id]},
                ]
            }
        }
    )
    event = MessageEvent(
        text="please call me",
        message_type=MessageType.TEXT,
        source=_source(),
        raw_message={"MessageSid": message_id, "unread": True},
        message_id=message_id,
        timestamp=datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc),
    )
    result = evaluate_unread_important_message(event, gateway_config.important_contacts)
    assert result.internal_event is not None
    return result.internal_event


def test_nudge_scheduler_generates_once_then_waits_for_cadence():
    unread_event = _important_unread_event()
    scheduler = ImportantUnreadNudgeScheduler(cadence_seconds=(60, 300))
    now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)

    first = scheduler.detect_due_nudge(unread_event, is_unread=True, now=now)
    duplicate = scheduler.detect_due_nudge(unread_event, is_unread=True, now=now)
    too_soon = scheduler.detect_due_nudge(
        unread_event,
        is_unread=True,
        now=datetime(2026, 5, 22, 12, 0, 59, tzinfo=timezone.utc),
    )
    due_again = scheduler.detect_due_nudge(
        unread_event,
        is_unread=True,
        now=datetime(2026, 5, 22, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert first is unread_event
    assert duplicate is None
    assert too_soon is None
    assert due_again is unread_event


def test_nudge_scheduler_suppresses_read_messages_and_clears_state():
    unread_event = _important_unread_event()
    scheduler = ImportantUnreadNudgeScheduler(cadence_seconds=(60,))
    now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)

    assert scheduler.detect_due_nudge(unread_event, is_unread=True, now=now) is unread_event
    assert scheduler.detect_due_nudge(
        unread_event,
        is_unread=False,
        now=datetime(2026, 5, 22, 12, 1, 0, tzinfo=timezone.utc),
    ) is None
    assert scheduler.detect_due_nudge(
        unread_event,
        is_unread=True,
        now=datetime(2026, 5, 22, 12, 2, 0, tzinfo=timezone.utc),
    ) is None


def test_nudge_scheduler_allows_new_generation_after_explicit_new_message():
    first_event = _important_unread_event("SMabc123")
    second_event = _important_unread_event("SMdef456")
    scheduler = ImportantUnreadNudgeScheduler(cadence_seconds=(60,))
    now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)

    assert scheduler.detect_due_nudge(first_event, is_unread=True, now=now) is first_event
    scheduler.mark_read(first_event.dedupe_key)

    assert scheduler.detect_due_nudge(first_event, is_unread=True, now=now) is None
    assert scheduler.detect_due_nudge(second_event, is_unread=True, now=now) is second_event
