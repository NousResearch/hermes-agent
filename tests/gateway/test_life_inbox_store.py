import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from gateway.life_inbox_store import (
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


def test_record_business_message_dedupes_metadata_and_never_stores_raw_text(tmp_path):
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

    assert row[0] == 202
    assert row[1] == "edited_business_message"
    assert row[2] == "1566649385"
    assert row[3] == "BaliRadar"
    assert row[4] == len("завтра в 15 созвон по Cockpit")
    assert len(row[5]) == 64
    assert row[6] is None
    assert row[7] == 0
    assert set(json.loads(row[8])) >= {"meeting", "time_reference"}
    assert chat_rule == ("telegram_business", "1566649385", "metadata_only")

    assert "завтра" not in store.db_path.read_bytes().decode("utf-8", errors="ignore")
    assert "Cockpit" not in store.db_path.read_bytes().decode("utf-8", errors="ignore")

    if os.name != "nt":
        assert stat.S_IMODE(store.db_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.db_path.stat().st_mode) == 0o600


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
