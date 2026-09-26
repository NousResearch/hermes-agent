"""Tests for Telegram inline keyboard approval buttons."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import unauthorized_action_notice, utf16_len

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    """Create a TelegramAdapter with mocked internals."""
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class _AuthRunner:
    """Minimal runner shim for callback auth tests."""

    def __init__(self, authorized: bool):
        self.authorized = authorized
        self.last_source = None

    async def _handle_message(self, event):
        return None

    def _is_user_authorized(self, source):
        self.last_source = source
        return self.authorized


# ===========================================================================
# send_exec_approval — inline keyboard buttons
# ===========================================================================

class TestTelegramExecApproval:
    """Test the send_exec_approval method sends InlineKeyboard buttons."""

    @pytest.mark.asyncio
    async def test_sends_inline_keyboard(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="rm -rf /important",
            session_key="agent:main:telegram:group:12345:99",
            description="dangerous deletion",
        )

        assert result.success is True
        assert result.message_id == "42"

        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "rm -rf /important" in kwargs["text"]
        assert "dangerous deletion" in kwargs["text"]
        assert kwargs["reply_markup"] is not None  # InlineKeyboardMarkup

    @pytest.mark.asyncio
    @pytest.mark.parametrize("smart_denied", [False, True])
    async def test_oversized_escaped_approval_text_keeps_inline_keyboard(self, smart_denied):
        """The rendered HTML card (escaped command + reason + framing) must fit Telegram's
        4096-char cap, otherwise the API rejects it and the gateway falls back to /approve."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))

        await adapter.send_exec_approval(
            chat_id="12345",
            command="&" * 3700,  # inside the old raw budget; 5x larger once escaped
            session_key="s",
            description="<reason>" * 1000,
            smart_denied=smart_denied,
        )

        kwargs = adapter._bot.send_message.call_args.kwargs
        assert len(kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH
        assert "&amp;&amp;" in kwargs["text"] and "&lt;reason&gt;" in kwargs["text"]
        assert kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_emoji_dense_approval_card_fits_in_utf16_units(self):
        """Telegram counts UTF-16 code units (astral emoji = 2), like the adapter's chunker."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))

        await adapter.send_exec_approval(chat_id="12345", command="😀" * 3000, session_key="s")

        kwargs = adapter._bot.send_message.call_args.kwargs
        assert utf16_len(kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH
        assert kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_slash_confirm_preview_fits_after_markdown_escaping(self):
        """The slash-confirm card is measured after format_message (MarkdownV2 escaping expands
        text), so a 3800-char raw message must still land under the 4096 cap."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))

        await adapter.send_slash_confirm(
            chat_id="12345", title="t", message="." * 3800, session_key="s", confirm_id="c1")

        kwargs = adapter._bot.send_message.call_args.kwargs
        assert utf16_len(kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH
        assert kwargs["reply_markup"] is not None


    @pytest.mark.asyncio
    async def test_non_smart_allow_permanent_false_keeps_session(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        buttons = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: buttons.append(text) or text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", lambda rows: rows
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
            allow_permanent=False,
        )

        assert buttons == ["✅ Allow Once", "✅ Session", "❌ Deny"]



    @pytest.mark.asyncio
    async def test_smart_deny_two_buttons_share_one_row(self, monkeypatch):
        """smart_deny yields 2 buttons — they pair into a single readable row."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        captured_rows = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
            allow_permanent=False, smart_denied=True,
        )

        assert captured_rows == [
            ["✅ Allow Once", "❌ Deny"],
        ]


    @pytest.mark.asyncio
    async def test_send_update_prompt_escapes_dynamic_prompt(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=55)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_update_prompt(
            chat_id="12345",
            prompt="Fix [issue]_1 and verify *markdown*",
            default="alpha_beta",
            metadata={"thread_id": "999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "Fix \\[issue\\]\\_1" in sent["text"]
        assert "alpha\\_beta" in sent["text"]

# _handle_callback_query — approval button clicks
# ===========================================================================

class TestTelegramApprovalCallback:
    """Test the approval callback handling in _handle_callback_query."""


    @pytest.mark.asyncio
    async def test_resume_typing_after_inline_approval(self):
        """Clicking an inline approval button must un-pause the chat's typing.

        Regression for #27853: the text /approve path resumed typing, but the
        ea: callback path did not, so the typing indicator stayed gone for the
        rest of a long-running turn after a button click.
        """
        adapter = _make_adapter()
        adapter._approval_state[5] = ("agent:main:telegram:group:12345:99", "req-5")
        adapter.pause_typing_for_chat("12345")
        assert "12345" in adapter._typing_paused

        query = AsyncMock()
        query.data = "ea:once:5"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Norbert"
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        assert "12345" not in adapter._typing_paused


    @pytest.mark.asyncio
    async def test_approval_callback_escapes_dynamic_user_name(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = ("agent:main:telegram:group:12345:99", "req-3")

        query = AsyncMock()
        query.data = "ea:once:3"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Alice_Bob"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        query.from_user.id = "12345"

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "Alice\\_Bob" in edit_kwargs["text"]


    @pytest.mark.asyncio
    async def test_second_card_resolves_only_second_request(self):
        """Card 2's tap resolves card 2's request, never card 1's queued one."""
        from tools import approval
        from tools.approval_gateway_wait import _ApprovalEntry

        adapter = _make_adapter()
        session = "agent:main:telegram:group:12345:99"
        first = _ApprovalEntry({"command": "rm -rf /tmp/important", "pattern_key": "recursive delete"})
        second = _ApprovalEntry({"command": "git push --force", "pattern_key": "force push"})

        # Deal the cards through the real send path so the mapping under test
        # is the one production builds, not a hand-written stub. The approval
        # id is the adapter's own counter, not the Telegram message_id.
        adapter._bot.send_message = AsyncMock(
            side_effect=[SimpleNamespace(message_id=900), SimpleNamespace(message_id=901)])
        await adapter.send_exec_approval(
            chat_id="12345", command=first.data["command"], session_key=session,
            request_id=first.data["request_id"])
        await adapter.send_exec_approval(
            chat_id="12345", command=second.data["command"], session_key=session,
            request_id=second.data["request_id"])
        second_card = max(adapter._approval_state)

        query = AsyncMock()
        query.data = f"ea:once:{second_card}"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Owner"
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)
        with approval._lock:
            approval._gateway_queues[session] = [first, second]
        try:
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
                await adapter._handle_callback_query(update, MagicMock())
            assert first.result is None
            assert second.result == "once"
        finally:
            with approval._lock:
                approval._gateway_queues.pop(session, None)

    @pytest.mark.asyncio
    async def test_expired_card_does_not_resolve_new_request(self):
        """A tap on a card whose request is gone must NOT authorize the next one up.

        This is the P1 shape: the old card's approval timed out and was dropped,
        a different command is queued, and the user taps the stale card. The
        resolver's no-request_id fallback takes the oldest queued entry, so the
        tap authorizes a command the card never described.
        """
        from tools import approval
        from tools.approval_gateway_wait import _ApprovalEntry

        adapter = _make_adapter()
        session = "agent:main:telegram:group:12345:99"
        expired = _ApprovalEntry({"command": "rm -rf /tmp/old", "pattern_key": "recursive delete"})
        next_request = _ApprovalEntry({"command": "git push --force", "pattern_key": "force push"})

        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=900))
        await adapter.send_exec_approval(
            chat_id="12345", command=expired.data["command"], session_key=session,
            request_id=expired.data["request_id"])
        stale_card = max(adapter._approval_state)

        query = AsyncMock()
        query.data = f"ea:once:{stale_card}"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Owner"
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        # Only the NEWER request is queued; the card's own request is long gone.
        with approval._lock:
            approval._gateway_queues[session] = [next_request]
        try:
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
                await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
            assert next_request.result is None
            assert "expired" in query.edit_message_text.call_args.kwargs["text"].lower()
        finally:
            with approval._lock:
                approval._gateway_queues.pop(session, None)

    @pytest.mark.asyncio
    async def test_update_prompt_callback_not_affected(self, tmp_path):
        """Ensure update prompt callbacks still work."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                # Allow the caller — the new fail-closed allowlist gate
                # (#24457) rejects empty TELEGRAM_ALLOWED_USERS, but this
                # test isn't exercising that gate; it's verifying the
                # update_prompt callback still writes the response.
                with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}):
                    await adapter._handle_callback_query(update, context)

        # Should NOT have triggered approval resolution
        mock_resolve.assert_not_called()
        assert (tmp_path / ".update_response").read_text() == "y"

    @pytest.mark.asyncio
    async def test_update_prompt_callback_rejects_unauthorized_user(self, tmp_path):
        """Update prompt buttons should honor TELEGRAM_ALLOWED_USERS."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 222
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert query.answer.call_args[1]["text"] == unauthorized_action_notice("telegram")
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()

    @pytest.mark.asyncio
    async def test_update_prompt_callback_rejects_user_blocked_by_global_allowlist(self, tmp_path):
        adapter = _make_adapter()
        runner = _AuthRunner(authorized=False)
        adapter._message_handler = runner._handle_message

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.from_user = MagicMock()
        query.from_user.id = 222
        query.from_user.first_name = "Mallory"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert query.answer.call_args[1]["text"] == unauthorized_action_notice("telegram")
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()
        assert runner.last_source is not None
        assert runner.last_source.platform == Platform.TELEGRAM
        assert runner.last_source.user_id == "222"

