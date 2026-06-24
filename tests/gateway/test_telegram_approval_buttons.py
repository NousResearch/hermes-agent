"""Tests for Telegram inline keyboard approval buttons."""

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
        assert adapter._approval_message_state == {"12345:42": approval_id}

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
    async def test_long_command_is_truncated_after_html_escape(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        command = "printf '<danger>&' " * 600
        result = await adapter.send_exec_approval(
            chat_id="12345",
            command=command,
            session_key="s",
            description="Security scan — pipe to interpreter",
        )

        assert result.success is True
        text = adapter._bot.send_message.call_args[1]["text"]
        assert len(text) <= 4096
        assert "<truncated>" in text
        assert adapter._bot.send_message.call_args[1]["reply_markup"] is not None

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
        assert "MARKDOWN_V2" in repr(sent["parse_mode"]) or sent["parse_mode"] == "MarkdownV2"
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
    async def test_callback_resolution_cleans_message_state(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = "agent:main:telegram:group:12345:99"
        adapter._approval_message_state["12345:42"] = 1

        query = AsyncMock()
        query.data = "ea:once:1"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Norbert"
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, MagicMock())

        assert adapter._approval_state == {}
        assert adapter._approval_message_state == {}

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
    async def test_late_approval_click_starts_followup_when_resolve_returns_zero(self):
        """If the original approval wait timed out, a later button click should
        start a fresh turn with the original approval context instead of only
        editing the stale prompt."""
        adapter = _make_adapter()
        adapter._approval_state[6] = "agent:main:telegram:group:12345:99"
        adapter._approval_context_state[6] = {
            "prompt": "Run dangerous deploy?",
            "default": "deploy --force",
        }
        adapter.pause_typing_for_chat("12345")
        adapter.handle_message = AsyncMock()

        query = AsyncMock()
        query.data = "ea:once:6"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.message_id = 77
        query.message.chat = MagicMock()
        query.message.chat.id = 12345
        query.message.chat.type = "private"
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
        adapter.handle_message.assert_awaited_once()
        late_event = adapter.handle_message.await_args.args[0]
        assert "Approval button pressed after" in late_event.text
        assert "Run dangerous deploy?" in late_event.text
        assert "deploy --force" in late_event.text
        assert late_event.source.platform == Platform.TELEGRAM
        assert late_event.source.chat_id == "12345"

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
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"]) or edit_kwargs["parse_mode"] == "MarkdownV2"
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


def _reaction_update(
    *,
    chat_id=12345,
    message_id=42,
    emoji="👍",
    user_id=12345,
    first_name="Norbert",
    username="norbert",
    chat_type="private",
):
    reaction = SimpleNamespace(emoji=emoji) if emoji is not None else None
    new_reaction = [] if reaction is None else [reaction]
    return SimpleNamespace(
        message_reaction=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type=chat_type),
            message_id=message_id,
            user=SimpleNamespace(id=user_id, first_name=first_name, username=username),
            new_reaction=new_reaction,
        )
    )


class TestTelegramApprovalReactions:
    """Test resolving approval prompts from inbound Telegram reactions."""

    @pytest.mark.asyncio
    async def test_thumbs_up_reaction_approves_once(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = "some-session"
        adapter._approval_message_state["12345:42"] = 1

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                await adapter._handle_message_reaction(_reaction_update(), MagicMock())

        mock_resolve.assert_called_once_with("some-session", "once")
        adapter._bot.edit_message_text.assert_awaited_once()
        edit_kwargs = adapter._bot.edit_message_text.call_args[1]
        assert edit_kwargs["chat_id"] == 12345
        assert edit_kwargs["message_id"] == 42
        assert "Approved once" in edit_kwargs["text"]
        assert adapter._approval_state == {}
        assert adapter._approval_message_state == {}

    @pytest.mark.asyncio
    async def test_checkmark_string_reaction_approves_once(self):
        adapter = _make_adapter()
        adapter._approval_state[1] = "some-session"
        adapter._approval_message_state["12345:42"] = 1

        update = _reaction_update(emoji=None)
        update.message_reaction.new_reaction = ["✅"]

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                await adapter._handle_message_reaction(update, MagicMock())

        mock_resolve.assert_called_once_with("some-session", "once")

    @pytest.mark.asyncio
    async def test_thumbs_down_reaction_denies(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = "some-session"
        adapter._approval_message_state["12345:42"] = 2

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                await adapter._handle_message_reaction(
                    _reaction_update(emoji="👎"), MagicMock()
                )

        mock_resolve.assert_called_once_with("some-session", "deny")
        assert "Denied" in adapter._bot.edit_message_text.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_cross_reaction_denies(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = "some-session"
        adapter._approval_message_state["12345:42"] = 3

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                await adapter._handle_message_reaction(
                    _reaction_update(emoji="❌"), MagicMock()
                )

        mock_resolve.assert_called_once_with("some-session", "deny")

    @pytest.mark.asyncio
    async def test_unauthorized_reaction_is_ignored_and_state_remains(self):
        adapter = _make_adapter()
        adapter._approval_state[4] = "some-session"
        adapter._approval_message_state["12345:42"] = 4
        runner = _AuthRunner(authorized=False)
        adapter._message_handler = runner._handle_message

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_message_reaction(
                _reaction_update(user_id=222, first_name="Mallory"), MagicMock()
            )

        mock_resolve.assert_not_called()
        adapter._bot.edit_message_text.assert_not_awaited()
        assert adapter._approval_state[4] == "some-session"
        assert adapter._approval_message_state["12345:42"] == 4
        assert runner.last_source is not None
        assert runner.last_source.user_id == "222"

    @pytest.mark.asyncio
    async def test_reaction_to_non_approval_message_is_ignored(self):
        adapter = _make_adapter()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_message_reaction(_reaction_update(), MagicMock())

        mock_resolve.assert_not_called()
        adapter._bot.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unrelated_reaction_on_approval_message_is_ignored(self):
        adapter = _make_adapter()
        adapter._approval_state[5] = "some-session"
        adapter._approval_message_state["12345:42"] = 5

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
                await adapter._handle_message_reaction(
                    _reaction_update(emoji="🔥"), MagicMock()
                )

        mock_resolve.assert_not_called()
        adapter._bot.edit_message_text.assert_not_awaited()
        assert adapter._approval_state[5] == "some-session"
        assert adapter._approval_message_state["12345:42"] == 5
