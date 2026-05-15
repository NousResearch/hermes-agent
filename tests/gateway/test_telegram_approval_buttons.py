"""Tests for Telegram inline keyboard approval buttons."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported
# ---------------------------------------------------------------------------
def _ensure_telegram_mock():
    """Wire up the minimal mocks required to import TelegramAdapter."""
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
    # Provide real exception classes so ``except (NetworkError, ...)`` in
    # connect() doesn't blow up under xdist when this mock leaks.
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

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
    async def test_stores_approval_state(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_exec_approval(
            chat_id="12345",
            command="echo test",
            session_key="my-session-key",
        )

        # The approval_id should map to the session_key
        assert len(adapter._approval_state) == 1
        approval_id = list(adapter._approval_state.keys())[0]
        assert adapter._approval_state[approval_id] == "my-session-key"

    @pytest.mark.asyncio
    async def test_sends_in_thread(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_exec_approval(
            chat_id="12345",
            command="ls",
            session_key="s",
            metadata={"thread_id": "999"},
        )

        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs.get("message_thread_id") == 999

    @pytest.mark.asyncio
    async def test_retries_without_thread_when_thread_not_found(self):
        adapter = _make_adapter()
        call_log = []

        class FakeBadRequest(Exception):
            pass

        async def mock_send_message(**kwargs):
            call_log.append(dict(kwargs))
            if kwargs.get("message_thread_id") is not None:
                raise FakeBadRequest("Message thread not found")
            return SimpleNamespace(message_id=42)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="ls",
            session_key="s",
            metadata={"thread_id": "999"},
        )

        assert result.success is True
        assert len(call_log) == 2
        assert call_log[0]["message_thread_id"] == 999
        assert "message_thread_id" not in call_log[1] or call_log[1]["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._bot = None
        result = await adapter.send_exec_approval(
            chat_id="12345", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_disable_link_previews_sets_preview_kwargs(self):
        adapter = _make_adapter(extra={"disable_link_previews": True})
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_exec_approval(
            chat_id="12345", command="ls", session_key="s"
        )

        kwargs = adapter._bot.send_message.call_args[1]
        assert (
            kwargs.get("disable_web_page_preview") is True
            or kwargs.get("link_preview_options") is not None
        )

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

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        long_cmd = "x" * 5000
        await adapter.send_exec_approval(
            chat_id="12345", command=long_cmd, session_key="s"
        )

        kwargs = adapter._bot.send_message.call_args[1]
        assert "..." in kwargs["text"]
        assert len(kwargs["text"]) < 5000
# _handle_callback_query — approval button clicks
# ===========================================================================

class TestTelegramApprovalCallback:
    """Test the approval callback handling in _handle_callback_query."""

    @pytest.mark.asyncio
    async def test_resolves_approval_on_click(self):
        adapter = _make_adapter()
        # Set up approval state
        adapter._approval_state[1] = "agent:main:telegram:group:12345:99"

        # Mock callback query
        query = AsyncMock()
        query.data = "ea:once:1"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Norbert"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        query.from_user.id = "12345"

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                await adapter._handle_callback_query(update, context)

        mock_resolve.assert_called_once_with("agent:main:telegram:group:12345:99", "once")
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()

        # State should be cleaned up
        assert 1 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_resume_typing_after_inline_approval(self):
        """Clicking an inline approval button must un-pause the chat's typing.

        Regression for #27853: the text /approve path resumed typing, but the
        ea: callback path did not, so the typing indicator stayed gone for the
        rest of a long-running turn after a button click.
        """
        adapter = _make_adapter()
        adapter._approval_state[5] = "agent:main:telegram:group:12345:99"
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
    async def test_typing_stays_paused_when_resolve_returns_zero(self):
        """If resolve_gateway_approval reports 0 resolves, the agent thread
        was never unblocked, so typing should NOT be force-resumed."""
        adapter = _make_adapter()
        adapter._approval_state[6] = "agent:main:telegram:group:12345:99"
        adapter.pause_typing_for_chat("12345")

        query = AsyncMock()
        query.data = "ea:once:6"
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
            with patch("tools.approval.resolve_gateway_approval", return_value=0):
                await adapter._handle_callback_query(update, context)

        assert "12345" in adapter._typing_paused

    @pytest.mark.asyncio
    async def test_approval_callback_escapes_dynamic_user_name(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = "agent:main:telegram:group:12345:99"

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
        assert "Approved once" in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_deny_button(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = "some-session"

        query = AsyncMock()
        query.data = "ea:deny:2"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Alice"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        query.from_user.id = "12345"

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                await adapter._handle_callback_query(update, context)

        mock_resolve.assert_called_once_with("some-session", "deny")
        edit_kwargs = query.edit_message_text.call_args[1]
        assert "Denied" in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_approval_callback_rejects_user_blocked_by_global_allowlist(self):
        adapter = _make_adapter()
        adapter._approval_state[7] = "agent:main:telegram:group:12345:99"
        runner = _AuthRunner(authorized=False)
        adapter._message_handler = runner._handle_message

        query = AsyncMock()
        query.data = "ea:once:7"
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

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_callback_query(update, context)

        mock_resolve.assert_not_called()
        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        assert adapter._approval_state[7] == "agent:main:telegram:group:12345:99"
        assert runner.last_source is not None
        assert runner.last_source.platform == Platform.TELEGRAM
        assert runner.last_source.user_id == "222"
        assert runner.last_source.chat_id == "12345"

    @pytest.mark.asyncio
    async def test_already_resolved(self):
        adapter = _make_adapter()
        # No state for approval_id 99 — already resolved

        query = AsyncMock()
        query.data = "ea:once:99"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Bob"
        query.answer = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        query.from_user.id = "12345"

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
                await adapter._handle_callback_query(update, context)

        # Should NOT resolve — already handled
        mock_resolve.assert_not_called()
        # Should still ack with "already resolved" message
        query.answer.assert_called_once()
        assert "already been resolved" in query.answer.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_model_picker_callback_not_affected(self):
        """Ensure model picker callbacks still route correctly."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "mp:some_provider"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        # Model picker callback should be handled (not crash)
        # We just verify it doesn't try to resolve an approval
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            with patch.object(adapter, "_handle_model_picker_callback", new_callable=AsyncMock):
                await adapter._handle_callback_query(update, context)

        mock_resolve.assert_not_called()

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
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
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
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()
        assert runner.last_source is not None
        assert runner.last_source.platform == Platform.TELEGRAM
        assert runner.last_source.user_id == "222"

    @pytest.mark.asyncio
    async def test_update_prompt_callback_allows_authorized_user(self, tmp_path):
        """Allowed Telegram users can still answer update prompt buttons."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:n"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 111
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        assert (tmp_path / ".update_response").read_text() == "n"


# ===========================================================================
# Issue #25693 — "Request Changes" button
# ===========================================================================

class TestTelegramRequestChangesButton:
    """Telegram-side wiring for the new \"Request Changes\" approval option.

    Three things to lock in:
      1. ``send_exec_approval`` renders the 5th button with the
         documented ``callback_data=\"ea:changes:<id>\"`` schema.
      2. Clicking the button registers a ``clarify_gateway`` entry in
         text-capture mode so the existing intercept resolves the user's
         next typed reply.
      3. The bridge eventually calls ``resolve_gateway_approval(...,
         choice=\"changes\", feedback=...)`` so the agent thread unblocks
         with the user's feedback.
    """

    @pytest.mark.asyncio
    async def test_inline_keyboard_includes_request_changes_button(self):
        """The 5-button layout must include the Request Changes button
        with the right callback_data shape so the callback parser can
        route it correctly.

        NOTE: this module mocks the ``telegram`` SDK (see
        ``_ensure_telegram_mock``), so ``InlineKeyboardMarkup`` /
        ``InlineKeyboardButton`` are MagicMocks whose ``.inline_keyboard``
        attribute is itself a mock, not a real nested list. We therefore
        capture the ``InlineKeyboardButton`` constructor calls directly
        rather than introspecting the assembled markup object — that works
        whether the SDK is mocked (local) or real (CI with
        python-telegram-bot installed)."""
        import gateway.platforms.telegram as tg

        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 99
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        # Capture every InlineKeyboardButton(...) call's callback_data.
        captured_callback_data = []

        def _spy_button(text, callback_data=None, **kwargs):
            captured_callback_data.append(callback_data)
            return SimpleNamespace(text=text, callback_data=callback_data)

        with patch.object(tg, "InlineKeyboardButton", side_effect=_spy_button):
            await adapter.send_exec_approval(
                chat_id="12345",
                command="DELETE FROM strava_activities",
                session_key="agent:main:telegram:direct:7675116888",
                description="SQL DELETE without WHERE",
            )

        assert any(
            cb is not None and cb.startswith("ea:changes:")
            for cb in captured_callback_data
        ), f"Request Changes button missing from layout: {captured_callback_data}"
        # All five required choices must be present.
        for choice in ("once", "session", "always", "changes", "deny"):
            assert any(
                cb is not None and cb.startswith(f"ea:{choice}:")
                for cb in captured_callback_data
            ), f"Missing 'ea:{choice}:*' button (captured: {captured_callback_data})"

    @pytest.mark.asyncio
    async def test_changes_click_starts_capture_and_keeps_approval_pending(self, tmp_path):
        """Clicking Request Changes must NOT pop _approval_state (the
        agent thread is still blocked) and must register a clarify entry
        in text-capture mode so the user's next reply resolves it."""
        from tools import clarify_gateway as cg

        adapter = _make_adapter()
        session_key = "agent:main:telegram:direct:7675116888"
        approval_id = 7
        adapter._approval_state[approval_id] = session_key

        query = AsyncMock()
        query.data = f"ea:changes:{approval_id}"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 88
        query.message.text = "⚠️ Command Approval Required"
        query.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = 7675116888
        query.from_user.first_name = "Daniel"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        # Authorize the caller.
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "7675116888"}):
            await adapter._handle_callback_query(update, context)

        # Approval entry stays pending — agent thread still blocked.
        assert approval_id in adapter._approval_state, (
            "Request Changes click prematurely popped _approval_state — "
            "the agent thread is no longer reachable!"
        )

        # The clarify entry was registered in text-capture mode.
        pending = cg.get_pending_for_session(session_key)
        assert pending is not None, "clarify entry not registered"
        assert pending.awaiting_text is True, (
            "clarify entry must be in awaiting_text mode for the gateway "
            "text-intercept to pick up the user's reply"
        )

        # UI was acknowledged + edited to show the awaiting state.
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()

        # Cleanup so the next test doesn't see this entry.
        cg.clear_session(session_key)

    @pytest.mark.asyncio
    async def test_changes_bridge_resolves_with_feedback(self):
        """Once the clarify entry resolves (simulating the user's typed
        reply being intercepted by the gateway), the bridge forwards the
        feedback into resolve_gateway_approval so the agent unblocks
        with the feedback intact."""
        from tools import approval as approval_module
        from tools import clarify_gateway as cg

        adapter = _make_adapter()
        session_key = "agent:main:telegram:direct:bridge-test"
        approval_id = 42
        adapter._approval_state[approval_id] = session_key

        # Pre-register the approval entry so resolve_gateway_approval has
        # something to set.
        approval_module._gateway_queues.clear()
        ap_entry = approval_module._ApprovalEntry({"command": "DELETE FROM x"})
        with approval_module._lock:
            approval_module._gateway_queues.setdefault(session_key, []).append(ap_entry)

        query = AsyncMock()
        query.data = f"ea:changes:{approval_id}"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 88
        query.message.text = "⚠️ Command Approval Required"
        query.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = 7675116888
        query.from_user.first_name = "Daniel"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        # Short-circuit the clarify timeout so the test doesn't hang if
        # we forget to resolve. 5s is plenty for asyncio scheduling.
        with patch.object(cg, "get_clarify_timeout", return_value=5):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "7675116888"}):
                await adapter._handle_callback_query(update, context)

            # Simulate the gateway text-intercept resolving the clarify
            # with the user's typed feedback.
            clarify_id = f"approval-changes-{approval_id}"
            await asyncio.sleep(0.05)  # let bridge task start
            resolved = cg.resolve_gateway_clarify(
                clarify_id, "Use INSERT OR REPLACE instead of DELETE",
            )
            assert resolved is True

            # Wait for the bridge to forward the feedback. Bounded so a
            # bug can't hang the suite.
            for _ in range(50):
                await asyncio.sleep(0.05)
                if ap_entry.event.is_set():
                    break

        assert ap_entry.event.is_set(), "bridge did not resolve the approval entry"
        assert isinstance(ap_entry.result, dict), (
            f"expected dict result for 'changes', got {type(ap_entry.result).__name__}: "
            f"{ap_entry.result!r}"
        )
        assert ap_entry.result["choice"] == "changes"
        assert "INSERT OR REPLACE" in ap_entry.result["feedback"]
        # Cleanup
        assert approval_id not in adapter._approval_state, (
            "approval state should be popped after the bridge resolves"
        )

    @pytest.mark.asyncio
    async def test_changes_bridge_cancel_degrades_to_deny(self):
        """If the user types the literal word ``cancel`` (case-insensitive)
        the bridge treats it as a deny rather than feedback."""
        from tools import approval as approval_module
        from tools import clarify_gateway as cg

        adapter = _make_adapter()
        session_key = "agent:main:telegram:direct:cancel-test"
        approval_id = 77
        adapter._approval_state[approval_id] = session_key

        approval_module._gateway_queues.clear()
        ap_entry = approval_module._ApprovalEntry({"command": "rm -rf /tmp"})
        with approval_module._lock:
            approval_module._gateway_queues.setdefault(session_key, []).append(ap_entry)

        query = AsyncMock()
        query.data = f"ea:changes:{approval_id}"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 88
        query.message.text = "⚠️ Command Approval Required"
        query.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = 7675116888
        query.from_user.first_name = "Daniel"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.object(cg, "get_clarify_timeout", return_value=5):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "7675116888"}):
                await adapter._handle_callback_query(update, context)
            clarify_id = f"approval-changes-{approval_id}"
            await asyncio.sleep(0.05)
            cg.resolve_gateway_clarify(clarify_id, "CANCEL")
            for _ in range(50):
                await asyncio.sleep(0.05)
                if ap_entry.event.is_set():
                    break

        assert ap_entry.event.is_set()
        # "CANCEL" is the documented escape word.
        assert ap_entry.result == "deny", (
            f"cancel should degrade to deny, got {ap_entry.result!r}"
        )
