"""Tests for Telegram DecisionCard inline keyboard UI."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported without telegram deps
# ---------------------------------------------------------------------------
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


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _decision_card():
    return {
        "decision_id": "dq-20260505-001",
        "title": "Decision Queue 운영 모드",
        "why_now": "첫 운영 기준을 잡아야 함",
        "options": [
            {"key": "A", "label": "모든 decision에 taste signal"},
            {"key": "B", "label": "중요한 decision만"},
            {"key": "C", "label": "approve/reject만"},
            {"key": "D", "label": "defer"},
        ],
        "recommendation": "A",
        "default": "A",
        "taste_signal": "초반 UX 비용보다 judgment replication 학습을 우선하는가?",
    }


class TestTelegramDecisionCardSend:
    @pytest.mark.asyncio
    async def test_sends_inline_keyboard_decision_card_in_thread(self, tmp_path):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 777
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            result = await adapter.send_decision_card(
                chat_id="12345",
                card=_decision_card(),
                metadata={"thread_id": "3220"},
            )

        assert result.success is True
        assert result.message_id == "777"
        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert kwargs["message_thread_id"] == 3220
        assert "Decision Queue 운영 모드" in kwargs["text"]
        assert "*Mina recommends:* A" in kwargs["text"]
        assert kwargs["reply_markup"] is not None
        assert "dq-20260505-001" in adapter._decision_card_state


class TestTelegramDecisionCardCallback:
    @pytest.mark.asyncio
    async def test_callback_records_choice_edits_card_and_removes_buttons(self, tmp_path):
        adapter = _make_adapter()
        adapter._decision_card_state["dq-20260505-001"] = _decision_card()

        query = MagicMock()
        query.data = "dq:dq-20260505-001:A"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="✅ A recorded")
        query.edit_message_text.assert_called_once()
        edit_kwargs = query.edit_message_text.call_args.kwargs
        assert "✅ Decision recorded: *A* by Joohyun" in edit_kwargs["text"]
        assert edit_kwargs["reply_markup"] is None

        records_path = tmp_path / "decision_queue" / "decisions.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["decision_id"] == "dq-20260505-001"
        assert record["choice"] == "A"
        assert record["selected_label"] == "모든 decision에 taste signal"
        assert record["user_id"] == "111"
        assert record["user_name"] == "Joohyun"
        assert record["chat_id"] == "12345"
        assert record["thread_id"] == "3220"
        assert record["title"] == "Decision Queue 운영 모드"
        assert record["recommendation"] == "A"
        assert record["taste_signal"] == "초반 UX 비용보다 judgment replication 학습을 우선하는가?"
        assert record["recorded_at"]
        assert "dq-20260505-001" not in adapter._decision_card_state

    @pytest.mark.asyncio
    async def test_callback_resolves_waiting_clarify_decision_card(self, tmp_path):
        adapter = _make_adapter()
        card = _decision_card()
        card["decision_id"] = "clarify-abc123"
        adapter._decision_card_state["clarify-abc123"] = card

        import threading
        event = threading.Event()
        adapter._decision_card_waiters["clarify-abc123"] = {"event": event}

        query = MagicMock()
        query.data = "dq:clarify-abc123:B"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, MagicMock())

        assert event.is_set()
        waiter = adapter._decision_card_waiters["clarify-abc123"]
        assert waiter["result"] == "B"
        assert waiter["label"] == "중요한 decision만"

    @pytest.mark.asyncio
    async def test_callback_loads_persisted_card_after_restart(self, tmp_path):
        sending_adapter = _make_adapter()
        card = _decision_card()
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            sending_adapter._save_active_decision_card(card["decision_id"], card)

        restarted_adapter = _make_adapter()
        assert restarted_adapter._decision_card_state == {}

        query = MagicMock()
        query.data = "dq:dq-20260505-001:C"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await restarted_adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="✅ C recorded")
        active_path = tmp_path / "decision_queue" / "active_cards.json"
        active = json.loads(active_path.read_text())
        assert "dq-20260505-001" not in active

    @pytest.mark.asyncio
    async def test_callback_rejects_unauthorized_user(self, tmp_path):
        adapter = _make_adapter()
        adapter._decision_card_state["dq-20260505-001"] = _decision_card()

        query = MagicMock()
        query.data = "dq:dq-20260505-001:B"
        query.from_user.id = 222
        query.from_user.first_name = "Other"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
                await adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="⛔ You are not authorized to answer this decision.")
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / "decision_queue" / "decisions.jsonl").exists()

    @pytest.mark.asyncio
    async def test_other_button_arms_free_text_reply_without_resolving_yet(self, tmp_path):
        adapter = _make_adapter()
        card = _decision_card()
        card["options"].append({"key": "__other__", "label": "Other / 직접 입력"})
        adapter._decision_card_state["dq-20260505-001"] = card

        query = MagicMock()
        query.data = "dq:dq-20260505-001:__other__"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, MagicMock())

        query.answer.assert_called_once_with(text="✍️ Reply with your answer")
        assert adapter._decision_card_text_waiters["111:12345:3220"] == "dq-20260505-001"
        assert "dq-20260505-001" in adapter._decision_card_state
        assert not (tmp_path / "decision_queue" / "decisions.jsonl").exists()

    @pytest.mark.asyncio
    async def test_free_text_reply_records_reason_and_resolves_waiter(self, tmp_path):
        adapter = _make_adapter()
        card = _decision_card()
        card["options"].append({"key": "__other__", "label": "Other / 직접 입력"})
        adapter._decision_card_state["dq-20260505-001"] = card
        adapter._decision_card_text_waiters["111:12345:3220"] = "dq-20260505-001"

        import threading
        event = threading.Event()
        adapter._decision_card_waiters["dq-20260505-001"] = {"event": event}

        msg = MagicMock()
        msg.text = "내가 원하는 답은 별도 운영 큐야"
        msg.chat_id = 12345
        msg.message_thread_id = 3220
        msg.from_user.id = 111
        msg.from_user.first_name = "Joohyun"

        update = MagicMock()
        update.message = msg

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            handled = await adapter._maybe_handle_decision_card_free_text(update)

        assert handled is True
        assert event.is_set()
        waiter = adapter._decision_card_waiters["dq-20260505-001"]
        assert waiter["result"] == "__other__"
        assert waiter["label"] == "내가 원하는 답은 별도 운영 큐야"
        assert "111:12345:3220" not in adapter._decision_card_text_waiters
        assert "dq-20260505-001" not in adapter._decision_card_state

        records_path = tmp_path / "decision_queue" / "decisions.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["choice"] == "__other__"
        assert records[0]["selected_label"] == "Other / 직접 입력"
        assert records[0]["free_text_reason"] == "내가 원하는 답은 별도 운영 큐야"

    @pytest.mark.asyncio
    async def test_cancel_button_records_and_resolves_waiter(self, tmp_path):
        adapter = _make_adapter()
        card = _decision_card()
        card["options"].append({"key": "__cancel__", "label": "Cancel"})
        adapter._decision_card_state["dq-20260505-001"] = card

        import threading
        event = threading.Event()
        adapter._decision_card_waiters["dq-20260505-001"] = {"event": event}

        query = MagicMock()
        query.data = "dq:dq-20260505-001:__cancel__"
        query.from_user.id = 111
        query.from_user.first_name = "Joohyun"
        query.message.chat_id = 12345
        query.message.message_thread_id = 3220
        query.message.chat.type = "supergroup"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, MagicMock())

        assert event.is_set()
        assert adapter._decision_card_waiters["dq-20260505-001"]["result"] == "__cancel__"
        query.answer.assert_called_once_with(text="✅ Cancel recorded")
        records = [json.loads(line) for line in (tmp_path / "decision_queue" / "decisions.jsonl").read_text().splitlines()]
        assert records[0]["choice"] == "__cancel__"
        assert records[0]["selected_label"] == "Cancel"
