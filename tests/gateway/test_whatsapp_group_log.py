"""Opt-in JSON-lines group log for the WhatsApp (Baileys bridge) adapter.

With ``require_mention: true`` the adapter drops every group message that does not address the bot
before anything is stored. ``group_log_chats`` keeps those messages (and the addressed ones) in a
file, so a digest or a review can read the whole conversation while the agent acts only when asked.
"""

import json

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter, _bridge_timestamp, _group_log_file_stem

GROUP = "120363001234567890@g.us"
OTHER = "120363009999999999@g.us"
BOT = "15550009999@s.whatsapp.net"
SENDER = "15550001111@s.whatsapp.net"


def _adapter(tmp_path, **extra):
    base = {
        "session_name": "test", "require_mention": True, "group_policy": "open",
        "group_log_chats": [GROUP], "group_log_dir": str(tmp_path / "group-log"),
    }
    base.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base))


def _msg(text, chat=GROUP, mentioned=(), quoted=None, **more):
    data = {
        "messageId": "ABC123", "chatId": chat, "senderId": SENDER, "senderName": "Brenda",
        "chatName": chat.split("@")[0], "isGroup": chat.endswith("@g.us"), "body": text,
        "hasMedia": False, "mediaType": "", "mentionedIds": list(mentioned), "botIds": [BOT],
        "hasQuotedMessage": bool(quoted), "timestamp": 1760300000,
    }
    if quoted:
        data.update({"quotedMessageId": "Q1", "quotedParticipant": quoted, "quotedText": "earlier text"})
    data.update(more)
    return data


def _records(tmp_path):
    files = sorted((tmp_path / "group-log").glob("*.jsonl"))
    lines = [line for f in files for line in f.read_text(encoding="utf-8").splitlines() if line]
    return files, [json.loads(line) for line in lines]


@pytest.mark.asyncio
async def test_unmentioned_group_message_is_logged_but_not_dispatched(tmp_path):
    adapter = _adapter(tmp_path)
    assert await adapter._build_message_event(_msg("we should repaint the cyc wall")) is None
    files, records = _records(tmp_path)
    assert [f.name for f in files] == [f"{_group_log_file_stem(GROUP)}.jsonl"]
    assert records == [{
        "ts": 1760300000.0, "chat_id": GROUP, "message_id": "ABC123", "sender_id": SENDER,
        "sender_name": "Brenda", "text": "we should repaint the cyc wall", "to_bot": False,
    }]
    assert oct(files[0].stat().st_mode & 0o777) == "0o600"
    assert oct((tmp_path / "group-log").stat().st_mode & 0o777) == "0o700"


@pytest.mark.asyncio
async def test_addressed_message_is_logged_and_dispatched(tmp_path):
    adapter = _adapter(tmp_path)
    event = await adapter._build_message_event(_msg("@15550009999 what is the hourly rate?", mentioned=[BOT]))
    assert event is not None
    assert event.text == "what is the hourly rate?"
    _, records = _records(tmp_path)
    assert len(records) == 1
    assert records[0]["to_bot"] is True
    assert records[0]["text"] == "@15550009999 what is the hourly rate?"


@pytest.mark.asyncio
async def test_reply_to_bot_keeps_quote_fields(tmp_path):
    adapter = _adapter(tmp_path)
    await adapter._build_message_event(_msg("yes please", quoted=BOT))
    _, records = _records(tmp_path)
    assert records[0]["to_bot"] is True
    assert records[0]["reply_to"] == {
        "message_id": "Q1", "author_id": adapter._normalize_whatsapp_id(BOT), "text": "earlier text",
    }


@pytest.mark.asyncio
async def test_two_messages_append_to_one_file(tmp_path):
    adapter = _adapter(tmp_path)
    await adapter._build_message_event(_msg("first"))
    await adapter._build_message_event(_msg("second", messageId="DEF456"))
    files, records = _records(tmp_path)
    assert len(files) == 1
    assert [r["message_id"] for r in records] == ["ABC123", "DEF456"]


@pytest.mark.asyncio
async def test_unlisted_group_dm_and_unadmitted_group_are_not_logged(tmp_path):
    adapter = _adapter(tmp_path)
    await adapter._build_message_event(_msg("hello", chat=OTHER))
    await adapter._build_message_event(_msg("hello", chat=SENDER))
    # Listed for the log but not admitted by the group policy: the list never widens intake.
    strict = _adapter(tmp_path, group_policy="allowlist", group_allow_from=[OTHER])
    await strict._build_message_event(_msg("hello"))
    assert not (tmp_path / "group-log").exists()


def test_log_is_off_by_default(tmp_path):
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    assert adapter._group_log_chats == set()
    assert adapter._append_group_log(_msg("hello")) is False


def test_env_csv_lists_chats_and_record_keeps_media_and_owner_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("WHATSAPP_GROUP_LOG_CHATS", f"{GROUP}, {OTHER}")
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    assert adapter._group_log_chats == {GROUP, OTHER}
    record = adapter._group_log_record(_msg("", mediaType="image", hasMedia=True, fromOwner=True))
    assert record["media_type"] == "image"
    assert record["from_owner"] is True
    assert record["to_bot"] is False
    assert "reply_to" not in record


def test_slash_command_counts_as_addressed(tmp_path):
    adapter = _adapter(tmp_path)
    assert adapter._group_log_record(_msg("/help"))["to_bot"] is True


def test_write_failure_is_reported_not_raised(tmp_path, capsys):
    adapter = _adapter(tmp_path)
    (tmp_path / "group-log").write_text("not a directory")
    assert adapter._append_group_log(_msg("hello")) is False
    assert "Group log write failed" in capsys.readouterr().out


def test_timestamp_and_file_stem_helpers():
    assert _bridge_timestamp(1760300000) == 1760300000.0
    assert _bridge_timestamp("1760300000") == 1760300000.0
    assert _bridge_timestamp({"low": 1760300000, "high": 0, "unsigned": True}) == 1760300000.0
    assert _bridge_timestamp(None) > 1760300000
    assert _bridge_timestamp(0) > 1760300000
    assert _group_log_file_stem(GROUP) == "120363001234567890_g.us"
    assert _group_log_file_stem("") == "group"
