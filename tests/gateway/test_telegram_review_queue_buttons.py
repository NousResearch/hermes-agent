import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_send_review_card_renders_review_buttons(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    adapter = _make_adapter()
    msg = MagicMock()
    msg.message_id = 222
    adapter._bot.send_message = AsyncMock(return_value=msg)

    result = await adapter.send_review_card(
        chat_id="-1003915682412",
        thread_id="3930",
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        person="@cfm_sol",
        url="https://x.com/cfm_sol/status/2066780746610831667",
        source="test",
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["chat_id"] == -1003915682412
    assert kwargs["message_thread_id"] == 3930
    assert "Agentic VCs" in kwargs["text"]
    assert kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_review_queue_callback_accept_updates_card_and_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
    )

    adapter = _make_adapter()
    query = AsyncMock()
    query.data = "rq:a:831667"
    query.message = MagicMock()
    query.message.chat_id = -1003915682412
    query.message.message_thread_id = 3930
    query.message.text = "card text"
    query.message.chat.type = "supergroup"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    card = rq.get_card("831667")
    assert card["status"] == "accepted"
    query.answer.assert_called_once()
    query.edit_message_text.assert_called_once()
    _, kwargs = query.edit_message_text.call_args
    assert kwargs["reply_markup"] is not None  # Add accepted note / No note next step


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["evidence", "expert"])
async def test_review_queue_evidence_and_expert_have_no_skip_and_deny_prompts_for_note(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="831667",
        kind=kind,
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim" if kind == "evidence" else "Person: @source\nRationale: thesis-relevant source",
        target="telegram:-1003915682412:3930",
    )
    card = rq.get_card("831667")
    assert card is not None
    buttons = rq.build_button_rows(card)
    callback_data = [cb for row in buttons for _, cb in row]
    assert not any(cb.startswith("rq:s:") for cb in callback_data)
    assert f"rq:d:831667" in callback_data

    adapter = _make_adapter()
    query = AsyncMock()
    query.data = "rq:d:831667"
    query.message = MagicMock()
    query.message.chat_id = -1003915682412
    query.message.message_thread_id = 3930
    query.message.text = "card text"
    query.message.chat.type = "supergroup"
    query.message.delete = AsyncMock()
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    card = rq.get_card("831667")
    assert card is not None
    assert card["status"] == "denied_pending_note"
    query.answer.assert_called_once()
    query.message.delete.assert_not_called()
    query.edit_message_text.assert_called_once()
    _, kwargs = query.edit_message_text.call_args
    assert kwargs["reply_markup"] is not None
    next_callbacks = [cb for row in rq.build_button_rows(card) for _, cb in row]
    assert f"rq:dn:831667" in next_callbacks
    assert f"rq:nn:831667" in next_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["evidence", "expert"])
async def test_review_queue_evidence_and_expert_deny_no_note_deletes(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="deny-no-note-card",
        kind=kind,
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim" if kind == "evidence" else "Person: @source\nRationale: thesis-relevant source",
        target="telegram:-1003915682412:3930",
    )

    adapter = _make_adapter()

    async def press(callback_data):
        query = AsyncMock()
        query.data = callback_data
        query.message = MagicMock()
        query.message.chat_id = -1003915682412
        query.message.message_thread_id = 3930
        query.message.text = "card text"
        query.message.chat.type = "supergroup"
        query.message.delete = AsyncMock()
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())
        return query

    deny_query = await press("rq:d:deny-no-note-card")
    denied_pending = rq.get_card("deny-no-note-card")
    assert denied_pending is not None
    assert denied_pending["status"] == "denied_pending_note"
    deny_query.message.delete.assert_not_called()
    deny_query.edit_message_text.assert_called_once()

    no_note_query = await press("rq:nn:deny-no-note-card")
    denied_card = rq.get_card("deny-no-note-card")
    assert denied_card is not None
    assert denied_card["status"] == "denied"
    no_note_query.message.delete.assert_awaited_once()
    no_note_query.edit_message_text.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["evidence", "expert"])
async def test_review_queue_evidence_and_expert_accept_no_note_deletes(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="accept-no-note-card",
        kind=kind,
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim" if kind == "evidence" else "Person: @source\nRationale: thesis-relevant source",
        target="telegram:-1003915682412:3930",
    )

    adapter = _make_adapter()

    async def press(callback_data):
        query = AsyncMock()
        query.data = callback_data
        query.message = MagicMock()
        query.message.chat_id = -1003915682412
        query.message.message_thread_id = 3930
        query.message.text = "card text"
        query.message.chat.type = "supergroup"
        query.message.delete = AsyncMock()
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())
        return query

    accept_query = await press("rq:a:accept-no-note-card")
    accepted_card = rq.get_card("accept-no-note-card")
    assert accepted_card is not None
    assert accepted_card["status"] == "accepted"
    accept_query.message.delete.assert_not_called()
    accept_query.edit_message_text.assert_called_once()

    no_note_query = await press("rq:n:accept-no-note-card")
    no_note_card = rq.get_card("accept-no-note-card")
    assert no_note_card is not None
    assert no_note_card["status"] == "accepted_no_note"
    no_note_query.message.delete.assert_awaited_once()
    no_note_query.edit_message_text.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["job", "startup"])
async def test_review_queue_skip_deletes_job_and_application_cards(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id=f"skip-{kind}-card",
        kind=kind,
        thesis="Jobs & Applications",
        body="Company: TestCo\nRole: Operator\nFit: relevant enough to review",
        target="telegram:-1003915682412:1524",
    )
    card = rq.get_card(f"skip-{kind}-card")
    assert card is not None
    callbacks = [cb for row in rq.build_button_rows(card) for _, cb in row]
    assert f"rq:s:skip-{kind}-card" in callbacks

    adapter = _make_adapter()
    query = AsyncMock()
    query.data = f"rq:s:skip-{kind}-card"
    query.message = MagicMock()
    query.message.chat_id = -1003915682412
    query.message.message_thread_id = 1524
    query.message.text = "card text"
    query.message.chat.type = "supergroup"
    query.message.delete = AsyncMock()
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    skipped_card = rq.get_card(f"skip-{kind}-card")
    assert skipped_card is not None
    assert skipped_card["status"] == "skipped"
    query.message.delete.assert_awaited_once()
    query.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_review_queue_callback_denies_unknown_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    adapter = _make_adapter()
    query = AsyncMock()
    query.data = "rq:d:missing"
    query.message = MagicMock()
    query.message.chat_id = -1003915682412
    query.message.message_thread_id = 3930
    query.message.chat.type = "supergroup"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()

    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    assert "not found" in query.answer.call_args[1]["text"].lower()


@pytest.mark.asyncio
async def test_review_queue_elaborate_captures_next_text_note(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    vault = tmp_path / "Antidote" / "Thesis"
    thesis_dir = vault / "Agentic VCs"
    thesis_dir.mkdir(parents=True)
    (thesis_dir / "Review queue.md").write_text("# Review queue — Agentic VCs\n", encoding="utf-8")
    (thesis_dir / "Agentic VCs.md").write_text("# Agentic VCs\n", encoding="utf-8")
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="831667",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
        telegram_message_id="222",
    )

    adapter = _make_adapter()
    assert adapter._bot is not None
    adapter._bot.delete_message = AsyncMock()
    query = AsyncMock()
    query.data = "rq:e:831667"
    query.message = MagicMock()
    query.message.chat_id = -1003915682412
    query.message.message_thread_id = 3930
    query.message.chat.type = "supergroup"
    query.from_user = MagicMock()
    query.from_user.id = "777"
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())

    msg = MagicMock()
    msg.text = "this test works"
    msg.chat.id = -1003915682412
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True
    msg.chat.title = "Axel & Ant1dote"
    msg.message_thread_id = 3930
    msg.is_topic_message = True
    msg.message_id = 999
    msg.date = None
    msg.from_user.id = 777
    msg.from_user.full_name = "Tester"
    msg.from_user.first_name = "Tester"
    msg.reply_to_message = None
    update2 = MagicMock()
    update2.update_id = 1234
    update2.message = msg
    update2.effective_message = msg

    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = MagicMock()
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_text_message(update2, MagicMock())

    card = rq.get_card("831667")
    assert card["status"] == "elaborated"
    assert card["decision_note"] == "this test works"
    adapter._bot.delete_message.assert_awaited_once_with(chat_id=-1003915682412, message_id=222)
    adapter.handle_message.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_review_queue_deny_note_capture_deletes_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_QUEUE_DB", str(tmp_path / "rq.db"))
    vault = tmp_path / "Antidote" / "Thesis"
    thesis_dir = vault / "Agentic VCs"
    thesis_dir.mkdir(parents=True)
    (thesis_dir / "Review queue.md").write_text("# Review queue — Agentic VCs\n", encoding="utf-8")
    monkeypatch.setenv("ANTIDOTE_THESIS_ROOT", str(vault))
    from tools import review_queue_cards as rq

    rq.create_card(
        card_id="deny-note-card",
        kind="evidence",
        thesis="Agentic VCs",
        body="FAIR multi-agent AI venture fund claim",
        target="telegram:-1003915682412:3930",
        telegram_message_id="222",
    )

    adapter = _make_adapter()
    assert adapter._bot is not None
    adapter._bot.delete_message = AsyncMock()

    async def press(callback_data):
        query = AsyncMock()
        query.data = callback_data
        query.message = MagicMock()
        query.message.chat_id = -1003915682412
        query.message.message_thread_id = 3930
        query.message.chat.type = "supergroup"
        query.message.delete = AsyncMock()
        query.from_user = MagicMock()
        query.from_user.id = "777"
        query.from_user.first_name = "Tester"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, MagicMock())
        return query

    deny_query = await press("rq:d:deny-note-card")
    denied_pending_card = rq.get_card("deny-note-card")
    assert denied_pending_card is not None
    assert denied_pending_card["status"] == "denied_pending_note"
    deny_query.message.delete.assert_not_called()
    deny_query.edit_message_text.assert_called_once()

    note_query = await press("rq:dn:deny-note-card")
    note_requested_card = rq.get_card("deny-note-card")
    assert note_requested_card is not None
    assert note_requested_card["status"] == "deny_note_requested"
    note_query.message.delete.assert_not_called()
    note_query.edit_message_text.assert_called_once()

    msg = MagicMock()
    msg.text = "too generic; tighten filter to primary sources only"
    msg.chat.id = -1003915682412
    msg.chat.type = "supergroup"
    msg.chat.is_forum = True
    msg.chat.title = "Axel & Ant1dote"
    msg.message_thread_id = 3930
    msg.is_topic_message = True
    msg.message_id = 999
    msg.date = None
    msg.from_user.id = 777
    msg.from_user.full_name = "Tester"
    msg.from_user.first_name = "Tester"
    msg.reply_to_message = None
    update2 = MagicMock()
    update2.update_id = 1234
    update2.message = msg
    update2.effective_message = msg

    adapter.handle_message = AsyncMock()
    adapter._enqueue_text_event = MagicMock()
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_text_message(update2, MagicMock())

    card = rq.get_card("deny-note-card")
    assert card is not None
    assert card["status"] == "denied_with_note"
    assert card["decision_note"] == "too generic; tighten filter to primary sources only"
    adapter._bot.delete_message.assert_awaited_once_with(chat_id=-1003915682412, message_id=222)
    adapter.handle_message.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()
