import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


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


@pytest.mark.asyncio
async def test_business_update_handler_persists_metadata_without_auto_reply(tmp_path, monkeypatch, caplog):
    life_home = tmp_path / ".hermes-life"
    _write_accounts_registry(life_home)
    monkeypatch.setenv("HERMES_LIFE_HOME", str(life_home))
    caplog.set_level(logging.WARNING, logger="gateway.platforms.telegram")

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    handled_messages = []

    async def fake_handle_message(event):
        handled_messages.append(event)

    adapter.handle_message = fake_handle_message

    rights = SimpleNamespace(to_dict=lambda: {"can_read_messages": True})
    user = SimpleNamespace(id=602562, username="oldman", full_name="Alen")
    business_connection = SimpleNamespace(
        id="conn-1",
        is_enabled=True,
        user_chat_id=602562,
        user=user,
        rights=rights,
    )
    await adapter._handle_business_update(
        SimpleNamespace(
            update_id=301,
            business_connection=business_connection,
            business_message=None,
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    chat = SimpleNamespace(id=1566649385, type="private", title=None, full_name="BaliRadar")
    sender = SimpleNamespace(id=1566649385, full_name="BaliRadar")
    business_message = SimpleNamespace(
        business_connection_id="conn-1",
        chat=chat,
        from_user=sender,
        text="завтра в 15 созвон",
        caption=None,
        message_id=1408996,
        date=datetime(2026, 5, 19, 22, 9, 49, tzinfo=timezone.utc),
    )
    await adapter._handle_business_update(
        SimpleNamespace(
            update_id=302,
            business_connection=None,
            business_message=business_message,
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    assert handled_messages == []

    db_path = life_home / "accounts/telegram-602562/life_inbox.sqlite"
    with sqlite3.connect(db_path) as conn:
        connection_row = conn.execute(
            "SELECT connection_id, user_chat_id, username FROM business_connections"
        ).fetchone()
        message_row = conn.execute(
            """
            SELECT chat_id, message_id, text_len, text_preview, raw_text_stored, candidate_reasons_json
            FROM business_messages
            """
        ).fetchone()

    assert connection_row == ("conn-1", "602562", "oldman")
    assert message_row[0] == "1566649385"
    assert message_row[1] == "1408996"
    assert message_row[2] == len("завтра в 15 созвон")
    assert message_row[3] is None
    assert message_row[4] == 0
    assert set(json.loads(message_row[5])) >= {"meeting", "time_reference"}

    assert "созвон" not in db_path.read_bytes().decode("utf-8", errors="ignore")

    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "chat_id=1566649385" not in warning_text
    assert "text_sha256" not in warning_text
    assert "BaliRadar" not in warning_text


@pytest.mark.asyncio
async def test_business_message_storage_survives_adapter_restart(tmp_path, monkeypatch):
    life_home = tmp_path / ".hermes-life"
    _write_accounts_registry(life_home)
    monkeypatch.setenv("HERMES_LIFE_HOME", str(life_home))

    first_adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    rights = SimpleNamespace(to_dict=lambda: {"can_read_messages": True})
    user = SimpleNamespace(id=602562, username="oldman", full_name="Alen")
    await first_adapter._handle_business_update(
        SimpleNamespace(
            update_id=401,
            business_connection=SimpleNamespace(
                id="conn-1",
                is_enabled=True,
                user_chat_id=602562,
                user=user,
                rights=rights,
            ),
            business_message=None,
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    restarted_adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    assert restarted_adapter._business_connection_user_chat_ids == {}
    await restarted_adapter._handle_business_update(
        SimpleNamespace(
            update_id=402,
            business_connection=None,
            business_message=SimpleNamespace(
                business_connection_id="conn-1",
                chat=SimpleNamespace(id=1566649385, type="private", title=None, full_name="BaliRadar"),
                from_user=SimpleNamespace(id=1566649385, full_name="BaliRadar"),
                text="завтра в 15 созвон",
                caption=None,
                message_id=1408997,
                date=datetime(2026, 5, 19, 22, 10, 49, tzinfo=timezone.utc),
            ),
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    db_path = life_home / "accounts/telegram-602562/life_inbox.sqlite"
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM business_messages").fetchone()[0]

    assert count == 1
    assert restarted_adapter._business_connection_user_chat_ids == {"conn-1": "602562"}


@pytest.mark.asyncio
async def test_text_handler_routes_business_connection_messages_to_inbox_not_agent(tmp_path, monkeypatch):
    life_home = tmp_path / ".hermes-life"
    _write_accounts_registry(life_home)
    monkeypatch.setenv("HERMES_LIFE_HOME", str(life_home))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    handled_messages = []
    enqueued_messages = []

    async def fake_handle_message(event):
        handled_messages.append(event)

    def fake_enqueue_event(event):
        enqueued_messages.append(event)

    adapter.handle_message = fake_handle_message
    adapter._enqueue_text_event = fake_enqueue_event

    await adapter._handle_business_update(
        SimpleNamespace(
            update_id=501,
            business_connection=SimpleNamespace(
                id="conn-1",
                is_enabled=True,
                user_chat_id=602562,
                user=SimpleNamespace(id=602562, username="oldman", full_name="Alen"),
                rights=SimpleNamespace(to_dict=lambda: {"can_read_messages": True}),
            ),
            business_message=None,
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    # PTB/Bot API can expose Business/Profile Automation traffic through the
    # regular text MessageHandler path while preserving Message.business_connection_id.
    # Owner-sent replies must be stored passively, not treated as authorized prompts.
    chat = SimpleNamespace(id=1566649385, type="private", title=None, full_name="BaliRadar")
    owner_sender = SimpleNamespace(id=602562, username="oldman", full_name="Alen")
    message = SimpleNamespace(
        business_connection_id="conn-1",
        chat=chat,
        from_user=owner_sender,
        text="testmsgtoAdmin",
        caption=None,
        message_id=1409001,
        date=datetime(2026, 5, 20, 7, 13, 40, tzinfo=timezone.utc),
    )

    await adapter._handle_text_message(
        SimpleNamespace(
            update_id=502,
            message=message,
            effective_message=message,
            business_connection=None,
            business_message=None,
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    assert handled_messages == []
    assert enqueued_messages == []

    db_path = life_home / "accounts/telegram-602562/life_inbox.sqlite"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT chat_id, sender_id, message_id, text_len FROM business_messages WHERE message_id = '1409001'"
        ).fetchone()

    assert row == ("1566649385", "602562", "1409001", len("testmsgtoAdmin"))


@pytest.mark.asyncio
async def test_business_inbox_skips_copies_of_the_hermes_bot_chat(tmp_path, monkeypatch):
    life_home = tmp_path / ".hermes-life"
    _write_accounts_registry(life_home)
    monkeypatch.setenv("HERMES_LIFE_HOME", str(life_home))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = SimpleNamespace(id=796330107, username="alenrbot")

    await adapter._handle_business_update(
        SimpleNamespace(
            update_id=601,
            business_connection=SimpleNamespace(
                id="conn-1",
                is_enabled=True,
                user_chat_id=602562,
                user=SimpleNamespace(id=602562, username="oldman", full_name="Alen"),
                rights=SimpleNamespace(to_dict=lambda: {"can_read_messages": True}),
            ),
            business_message=None,
            edited_business_message=None,
            deleted_business_messages=None,
        ),
        None,
    )

    # Business/Profile Automation mirrors the owner's DM with this same bot as
    # business_message updates. Those should not enter the life inbox.
    bot_chat = SimpleNamespace(id=796330107, type="private", full_name="Птолемей | Ассистент Алена")
    owner_message_to_bot = SimpleNamespace(
        business_connection_id="conn-1",
        chat=bot_chat,
        from_user=SimpleNamespace(id=602562, username="oldman", full_name="Alen"),
        text="проверяй",
        caption=None,
        message_id=1409673,
        date=datetime(2026, 5, 20, 7, 42, 13, tzinfo=timezone.utc),
    )
    bot_reply_copy = SimpleNamespace(
        business_connection_id="conn-1",
        chat=bot_chat,
        from_user=SimpleNamespace(id=796330107, username="alenrbot", full_name="Птолемей | Ассистент Алена"),
        text="assistant response",
        caption=None,
        message_id=1409674,
        date=datetime(2026, 5, 20, 7, 42, 20, tzinfo=timezone.utc),
    )

    for update_id, message in ((602, owner_message_to_bot), (603, bot_reply_copy)):
        assert await adapter._handle_business_update(
            SimpleNamespace(
                update_id=update_id,
                business_connection=None,
                business_message=message,
                edited_business_message=None,
                deleted_business_messages=None,
            ),
            None,
        ) is True

    db_path = life_home / "accounts/telegram-602562/life_inbox.sqlite"
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM business_messages").fetchone()[0]

    assert count == 0


@pytest.mark.asyncio
async def test_send_refuses_own_bot_chat_without_calling_telegram():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    send_calls = []

    async def fake_send_message(**kwargs):
        send_calls.append(kwargs)
        raise AssertionError("send_message must not be called for own bot chat")

    adapter._bot = SimpleNamespace(id=796330107, username="alenrbot", send_message=fake_send_message)

    result = await adapter.send(chat_id="796330107", content="should not send")

    assert result.success is True
    assert result.message_id is None
    assert send_calls == []


def test_should_process_message_ignores_messages_from_own_bot():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = SimpleNamespace(id=796330107, username="alenrbot")
    message = SimpleNamespace(
        chat=SimpleNamespace(id=796330107, type="private"),
        from_user=SimpleNamespace(id=796330107, is_bot=True, username="alenrbot"),
        text="loop bait",
        caption=None,
    )

    assert adapter._should_process_message(message) is False
