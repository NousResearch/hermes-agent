import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from gateway.life_inbox_store import (
    BUSINESS_PAYLOAD_PROBE_LANE,
    BUSINESS_PAYLOAD_PROBE_SCENARIOS,
    LifeInboxStore,
    detect_candidate_reasons,
    resolve_life_inbox_db_path,
)


def _write_accounts_registry(life_home: Path) -> None:
    profile_rel = "accounts/telegram-602562/profile.json"
    profile_path = life_home / profile_rel
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("{}\n")
    (life_home / "accounts.json").write_text(
        json.dumps(
            {
                "version": 1,
                "accounts": {
                    "telegram:602562": {
                        "display_name": "Alen",
                        "telegram_user_id": "602562",
                        "life_profile": profile_rel,
                    }
                },
            }
        )
        + "\n"
    )


def test_resolve_life_inbox_db_path_uses_account_scoped_profile_dir(tmp_path):
    life_home = tmp_path / ".hermes-life"
    _write_accounts_registry(life_home)

    db_path = resolve_life_inbox_db_path("602562", life_home=life_home)

    assert db_path == life_home / "accounts/telegram-602562/life_inbox.sqlite"


def test_record_business_connection_upserts_without_raw_message_storage(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")

    store.record_business_connection(
        update_id=101,
        connection_id="conn-1",
        is_enabled=True,
        user_chat_id="602562",
        user_id="602562",
        username="oldman",
        full_name="Alen",
        rights={"can_read_messages": True},
    )
    store.record_business_connection(
        update_id=102,
        connection_id="conn-1",
        is_enabled=False,
        user_chat_id="602562",
        user_id="602562",
        username="oldman",
        full_name="Alen Updated",
        rights={"can_read_messages": False},
    )

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT connection_id, is_enabled, user_chat_id, full_name, rights_json FROM business_connections"
        ).fetchone()

    assert row[0] == "conn-1"
    assert row[1] == 0
    assert row[2] == "602562"
    assert row[3] == "Alen Updated"
    assert json.loads(row[4]) == {"can_read_messages": False}


def test_record_business_message_dedupes_metadata_and_archives_raw_text(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")

    message_pk_1 = store.record_business_message(
        update_id=201,
        update_type="business_message",
        connection_id="conn-1",
        chat_id="1566649385",
        chat_type="private",
        chat_name="BaliRadar",
        message_id="1408996",
        sender_id="1566649385",
        sender_name="BaliRadar",
        text="завтра в 15 созвон по Cockpit",
        message_date="2026-05-19T22:09:49+00:00",
    )
    message_pk_2 = store.record_business_message(
        update_id=202,
        update_type="edited_business_message",
        connection_id="conn-1",
        chat_id="1566649385",
        chat_type="private",
        chat_name="BaliRadar",
        message_id="1408996",
        sender_id="1566649385",
        sender_name="BaliRadar",
        text="завтра в 15 созвон по Cockpit",
        message_date="2026-05-19T22:10:00+00:00",
    )

    assert message_pk_1 == message_pk_2

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            """
            SELECT update_id, update_type, chat_id, chat_name, text_len, text_sha256,
                   text_preview, raw_text_stored, candidate_reasons_json
            FROM business_messages
            """
        ).fetchone()
        chat_rule = conn.execute(
            "SELECT platform, chat_id, rule_mode FROM chat_rules"
        ).fetchone()
        text_row = conn.execute(
            "SELECT text FROM business_message_text WHERE business_message_id = ?",
            (message_pk_1,),
        ).fetchone()

    assert row[0] == 202
    assert row[1] == "edited_business_message"
    assert row[2] == "1566649385"
    assert row[3] == "BaliRadar"
    assert row[4] == len("завтра в 15 созвон по Cockpit")
    assert len(row[5]) == 64
    assert row[6] is None
    assert row[7] == 1
    assert set(json.loads(row[8])) >= {"meeting", "time_reference"}
    assert chat_rule == ("telegram_business", "1566649385", "full_rag_selected")
    assert text_row == ("завтра в 15 созвон по Cockpit",)

    if os.name != "nt":
        assert stat.S_IMODE(store.db_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.db_path.stat().st_mode) == 0o600


def test_record_business_message_archives_raw_text_and_business_users(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")

    message_pk = store.record_business_message(
        update_id=301,
        update_type="business_message",
        connection_id="conn-archive",
        chat_id="1566649385",
        chat_type="private",
        chat_name="BaliRadar",
        chat_username="RadarAdmin",
        message_id="1409001",
        sender_id="1566649385",
        sender_name="BaliRadar",
        sender_username="RadarAdmin",
        sender_is_bot=False,
        sender_language_code="en",
        text="private archive text for analyzer",
        message_date="2026-05-20T09:10:00+00:00",
    )

    # Outgoing messages should still retain the private-chat counterpart as a user,
    # not only Alen as the sender.
    store.record_business_message(
        update_id=302,
        update_type="business_message",
        connection_id="conn-archive",
        chat_id="1566649385",
        chat_type="private",
        chat_name="BaliRadar",
        chat_username="RadarAdmin",
        message_id="1409002",
        sender_id="602562",
        sender_name="Alen",
        sender_username="oldman",
        sender_is_bot=False,
        sender_language_code="ru",
        text="alen outgoing archive text",
        message_date="2026-05-20T09:11:00+00:00",
    )

    with sqlite3.connect(store.db_path) as conn:
        message_row = conn.execute(
            """
            SELECT raw_text_stored, text_len
            FROM business_messages
            WHERE id = ?
            """,
            (message_pk,),
        ).fetchone()
        text_row = conn.execute(
            """
            SELECT text
            FROM business_message_text
            WHERE business_message_id = ?
            """,
            (message_pk,),
        ).fetchone()
        users = conn.execute(
            """
            SELECT user_id, username, full_name, is_bot, language_code
            FROM business_users
            ORDER BY user_id
            """
        ).fetchall()
        participants = conn.execute(
            """
            SELECT chat_id, user_id
            FROM chat_participants
            ORDER BY user_id
            """
        ).fetchall()

    assert message_row == (1, len("private archive text for analyzer"))
    assert text_row == ("private archive text for analyzer",)
    assert users == [
        ("1566649385", "RadarAdmin", "BaliRadar", 0, "en"),
        ("602562", "oldman", "Alen", 0, "ru"),
    ]
    assert participants == [
        ("1566649385", "1566649385"),
        ("1566649385", "602562"),
    ]


def test_record_business_message_rejects_missing_identity_fields(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")

    with pytest.raises(ValueError):
        store.record_business_message(
            update_id=201,
            update_type="business_message",
            connection_id="conn-1",
            chat_id=None,
            chat_type="private",
            chat_name="BaliRadar",
            message_id="1408996",
            sender_id="1566649385",
            sender_name="BaliRadar",
            text="test",
            message_date="2026-05-19T22:09:49+00:00",
        )

    with pytest.raises(ValueError):
        store.record_business_message(
            update_id=201,
            update_type="business_message",
            connection_id="conn-1",
            chat_id="1566649385",
            chat_type="private",
            chat_name="BaliRadar",
            message_id=None,
            sender_id="1566649385",
            sender_name="BaliRadar",
            text="test",
            message_date="2026-05-19T22:09:49+00:00",
        )


def test_detect_candidate_reasons_covers_life_inbox_keywords():
    assert set(detect_candidate_reasons("завтра в 15 созвон, напомни follow up")) >= {
        "meeting",
        "time_reference",
        "reminder",
        "follow_up",
    }
    assert detect_candidate_reasons("просто болтаем ни о чём") == []


def test_prepare_payload_probe_scenarios_stores_hashes_not_probe_text(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")
    probe_text = "TBP-20260520-S1-contact-inbound"

    store.prepare_business_payload_probe_scenarios(
        [
            {
                "scenario_id": "S1_contact_inbound",
                "alias": "CONTACT_1",
                "expected_direction": "incoming_to_owner",
                "probe_text": probe_text,
            }
        ]
    )

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            """
            SELECT source_lane, scenario_id, alias, expected_direction,
                   probe_text_len, probe_text_sha256, status
            FROM business_payload_probe_scenarios
            """
        ).fetchone()

    assert row[0] == BUSINESS_PAYLOAD_PROBE_LANE
    assert row[1] == "S1_contact_inbound"
    assert row[2] == "CONTACT_1"
    assert row[3] == "incoming_to_owner"
    assert row[4] == len(probe_text)
    assert len(row[5]) == 64
    assert row[6] == "pending"
    assert probe_text not in store.db_path.read_bytes().decode("utf-8", errors="ignore")


def test_record_payload_probe_event_matches_scenario_and_keeps_shape_only(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")
    probe_text = "TBP-20260520-S2-owner-outbound"
    store.prepare_business_payload_probe_scenarios(
        [
            {
                "scenario_id": "S2_contact_alen_manual_outbound",
                "alias": "CONTACT_1",
                "expected_direction": "outgoing_from_owner",
                "probe_text": probe_text,
            }
        ]
    )

    event_id = store.record_business_payload_probe_event(
        update_id=701,
        update_type="business_message",
        connection_id="conn-1",
        owner_user_chat_id="602562",
        chat_id="1566649385",
        message_id="1409010",
        sender_id="602562",
        text=probe_text,
        message_date="2026-05-20T08:40:00+00:00",
        field_availability={"message": {"has_text": True, "has_reply_to_message": False}},
        payload_shape={"message": {"text": {"type": "str"}, "from": {"id": {"type": "int"}}}},
    )

    assert event_id is not None
    with sqlite3.connect(store.db_path) as conn:
        event_row = conn.execute(
            """
            SELECT source_lane, scenario_id, direction, text_len, raw_text_stored,
                   field_availability_json, payload_shape_json
            FROM business_payload_probe_events
            """
        ).fetchone()
        scenario_row = conn.execute(
            """
            SELECT status, matched_event_id, matched_at
            FROM business_payload_probe_scenarios
            WHERE scenario_id = 'S2_contact_alen_manual_outbound'
            """
        ).fetchone()

    assert event_row[0] == BUSINESS_PAYLOAD_PROBE_LANE
    assert event_row[1] == "S2_contact_alen_manual_outbound"
    assert event_row[2] == "outgoing_from_owner"
    assert event_row[3] == len(probe_text)
    assert event_row[4] == 0
    assert json.loads(event_row[5])["message"]["has_text"] is True
    assert json.loads(event_row[6])["message"]["text"]["type"] == "str"
    assert scenario_row[0] == "matched"
    assert scenario_row[1] == event_id
    assert scenario_row[2] is not None
    assert probe_text not in store.db_path.read_bytes().decode("utf-8", errors="ignore")


def test_payload_probe_event_can_match_followup_edit_by_message_identity(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")
    probe_text = "TBP-20260520-S1-edit-followup"
    store.prepare_business_payload_probe_scenarios(
        [
            {
                "scenario_id": "S1_contact_inbound",
                "alias": "CONTACT_1",
                "expected_direction": "incoming_to_owner",
                "probe_text": probe_text,
            }
        ]
    )
    first_event_id = store.record_business_payload_probe_event(
        update_id=801,
        update_type="business_message",
        connection_id="conn-1",
        owner_user_chat_id="602562",
        chat_id="1566649385",
        message_id="1409011",
        sender_id="1566649385",
        text=probe_text,
        message_date="2026-05-20T08:45:00+00:00",
        field_availability={},
        payload_shape={},
    )

    edited_event_id = store.record_business_payload_probe_event(
        update_id=802,
        update_type="edited_business_message",
        connection_id="conn-1",
        owner_user_chat_id="602562",
        chat_id="1566649385",
        message_id="1409011",
        sender_id="1566649385",
        text="edited private text not registered as a probe code",
        message_date="2026-05-20T08:46:00+00:00",
        field_availability={"message": {"has_edit_date": True}},
        payload_shape={"message": {"edit_date": {"type": "datetime"}}},
    )

    assert first_event_id is not None
    assert edited_event_id is not None
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT update_type, scenario_id FROM business_payload_probe_events ORDER BY id"
        ).fetchall()

    assert rows == [
        ("business_message", "S1_contact_inbound"),
        ("edited_business_message", "S1_contact_inbound"),
    ]
    assert "private text" not in store.db_path.read_bytes().decode("utf-8", errors="ignore")


def test_payload_probe_event_without_scenario_is_skipped_unless_capture_all(tmp_path):
    store = LifeInboxStore(tmp_path / "life_inbox.sqlite")

    skipped = store.record_business_payload_probe_event(
        update_id=901,
        update_type="business_message",
        connection_id="conn-1",
        owner_user_chat_id="602562",
        chat_id="1566649385",
        message_id="1409012",
        sender_id="1566649385",
        text="private unmatched text",
        message_date="2026-05-20T08:50:00+00:00",
        field_availability={},
        payload_shape={},
    )
    captured = store.record_business_payload_probe_event(
        update_id=902,
        update_type="business_message",
        connection_id="conn-1",
        owner_user_chat_id="602562",
        chat_id="1566649385",
        message_id="1409013",
        sender_id="1566649385",
        text="private unmatched text",
        message_date="2026-05-20T08:50:10+00:00",
        field_availability={"message": {"has_photo": True}},
        payload_shape={"message": {"photo": {"type": "list", "length": 1}}},
        capture_all=True,
    )

    assert skipped is None
    assert captured is not None
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT scenario_id, text_len FROM business_payload_probe_events").fetchall()

    assert rows == [(None, len("private unmatched text"))]
    assert "private unmatched text" not in store.db_path.read_bytes().decode("utf-8", errors="ignore")
    assert len(BUSINESS_PAYLOAD_PROBE_SCENARIOS) == 6
