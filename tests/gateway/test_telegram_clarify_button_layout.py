"""Tests for Telegram clarify button layout - single-row vs per-row.

PR #104769: when choices <= 8, all numeric buttons render on a single row
to avoid vertical stretching. When > 8, each button gets its own row
(Telegram silently drops extras beyond ~12 per row).
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Fake InlineKeyboardButton/InlineKeyboardMarkup that capture row structure
# ---------------------------------------------------------------------------
class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"Btn({self.text})"


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard

    def to_dict(self):
        return {"inline_keyboard": self.inline_keyboard}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fake_telegram(monkeypatch):
    """Replace real telegram classes with fakes so we can inspect row layout."""
    import plugins.platforms.telegram.adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "InlineKeyboardButton", _FakeInlineKeyboardButton)
    monkeypatch.setattr(adapter_mod, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)
    # Also patch the lazy import guard in the adapter module itself
    import plugins.platforms.telegram.adapter as adapter_mod2
    monkeypatch.setattr(adapter_mod2, "InlineKeyboardButton", _FakeInlineKeyboardButton)
    monkeypatch.setattr(adapter_mod2, "InlineKeyboardMarkup", _FakeInlineKeyboardMarkup)


def _make_adapter(extra=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import PlatformConfig
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


# ===========================================================================
# Button layout tests
# ===========================================================================

class TestTelegramClarifyButtonLayout:
    """Verify the single-row vs per-row layout logic from PR #104769."""

    @pytest.mark.asyncio
    async def test_few_choices_render_single_row(self):
        """<=8 choices: all numeric buttons in one row, Other in second row."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 200
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        choices = [f"opt_{i}" for i in range(5)]  # 5 <= 8
        await adapter.send_clarify(
            chat_id="12345",
            question="Pick one",
            choices=choices,
            clarify_id="cid-single",
            session_key="sk-single",
        )

        kwargs = adapter._bot.send_message.call_args[1]
        markup = kwargs["reply_markup"]
        assert markup is not None, "Expected InlineKeyboardMarkup"

        rows = markup.inline_keyboard
        # Row 0: all 5 choice buttons in one row
        assert len(rows) == 2, f"Expected 2 rows (choices + Other), got {len(rows)}"
        assert len(rows[0]) == 5, (
            f"Expected 5 buttons in first row for 5 choices, got {len(rows[0])}"
        )
        # Each button should be a short numeric label
        for i, btn in enumerate(rows[0]):
            assert btn.text == str(i + 1), (
                f"Button {i} label should be '{i + 1}', got '{btn.text}'"
            )
            assert btn.callback_data == f"cl:cid-single:{i}", (
                f"Button {i} callback_data mismatch"
            )
        # Row 1: "Other" button
        assert len(rows[1]) == 1
        assert "Other" in rows[1][0].text

    @pytest.mark.asyncio
    async def test_many_choices_render_per_row(self):
        """>8 choices: each numeric button in its own row, Other in last row."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 201
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        choices = [f"opt_{i}" for i in range(10)]  # 10 > 8
        await adapter.send_clarify(
            chat_id="12345",
            question="Pick one",
            choices=choices,
            clarify_id="cid-multi",
            session_key="sk-multi",
        )

        kwargs = adapter._bot.send_message.call_args[1]
        markup = kwargs["reply_markup"]
        assert markup is not None

        rows = markup.inline_keyboard
        # 10 choice rows + 1 Other row = 11 rows
        assert len(rows) == 11, (
            f"Expected 11 rows (10 per-choice + Other), got {len(rows)}"
        )
        # Each choice row should have exactly 1 button
        for i in range(10):
            assert len(rows[i]) == 1, (
                f"Row {i} should have 1 button, got {len(rows[i])}"
            )
            assert rows[i][0].text == str(i + 1)
        # Last row: Other
        assert "Other" in rows[10][0].text

    @pytest.mark.asyncio
    async def test_exactly_eight_choices_single_row(self):
        """Boundary: exactly 8 choices still uses single-row layout."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 202
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        choices = [f"opt_{i}" for i in range(8)]
        await adapter.send_clarify(
            chat_id="12345",
            question="Pick one",
            choices=choices,
            clarify_id="cid-8",
            session_key="sk-8",
        )

        kwargs = adapter._bot.send_message.call_args[1]
        markup = kwargs["reply_markup"]
        rows = markup.inline_keyboard
        assert len(rows) == 2, f"Expected 2 rows for 8 choices, got {len(rows)}"
        assert len(rows[0]) == 8, (
            f"Expected 8 buttons in first row, got {len(rows[0])}"
        )

    @pytest.mark.asyncio
    async def test_nine_choices_per_row(self):
        """Boundary: 9 choices uses per-row layout."""
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 203
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        choices = [f"opt_{i}" for i in range(9)]
        await adapter.send_clarify(
            chat_id="12345",
            question="Pick one",
            choices=choices,
            clarify_id="cid-9",
            session_key="sk-9",
        )

        kwargs = adapter._bot.send_message.call_args[1]
        markup = kwargs["reply_markup"]
        rows = markup.inline_keyboard
        assert len(rows) == 10, f"Expected 10 rows for 9 choices, got {len(rows)}"
        for i in range(9):
            assert len(rows[i]) == 1, f"Row {i} should have 1 button"
