"""Tests for Telegram inline keyboard approval buttons."""

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.i18n import t
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

        assert buttons == [t(f"platform.telegram.approval.action_{c}") for c in ("once", "session", "deny")]



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
            [t("platform.telegram.approval.action_once"), t("platform.telegram.approval.action_deny")],
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


class TestTelegramApprovalResolutionKeepsPrompt:
    """Resolving an approval appends the decision to the original prompt text
    instead of replacing it, so the chat keeps audit context (#128982)."""

    def _query(self, data, user="Norbert"):
        query = AsyncMock()
        query.data = data
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = user
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        return query, update, MagicMock()

    @pytest.mark.asyncio
    async def test_resolved_edit_keeps_command_text(self):
        adapter = _make_adapter()
        prompt_html = "⚠️ <b>Header</b>\n\n<pre>rm -rf /important</pre>\n\ndeadline line"
        adapter._approval_state[7] = {
            "session_key": "agent:main:telegram:group:12345:99", "text": prompt_html}
        query, update, context = self._query("ea:once:7")

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "HTML" in repr(edit_kwargs["parse_mode"])
        assert edit_kwargs["reply_markup"] is None
        assert "rm -rf /important" in edit_kwargs["text"]
        assert "Approved once" in edit_kwargs["text"]
        assert "Norbert" in edit_kwargs["text"]
        assert utf16_len(edit_kwargs["text"]) <= adapter.MAX_MESSAGE_LENGTH

    @pytest.mark.asyncio
    async def test_expired_edit_keeps_command_text(self):
        adapter = _make_adapter()
        prompt_html = "⚠️ <b>Header</b>\n\n<pre>rm -rf /important</pre>"
        adapter._approval_state[8] = {
            "session_key": "agent:main:telegram:group:12345:99", "text": prompt_html}
        query, update, context = self._query("ea:once:8")

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=0):
                await adapter._handle_callback_query(update, context)

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "rm -rf /important" in edit_kwargs["text"]
        assert "expired" in edit_kwargs["text"].lower()

    @pytest.mark.asyncio
    async def test_legacy_plain_state_falls_back_to_short_edit(self):
        """Pre-fix in-memory shape (bare session key) keeps the old short edit."""
        adapter = _make_adapter()
        adapter._approval_state[9] = "agent:main:telegram:group:12345:99"
        query, update, context = self._query("ea:once:9")

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "rm -rf" not in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_stores_prompt_text(self):
        """The send path keeps the exact HTML shown to the user."""
        from types import SimpleNamespace

        adapter = _make_adapter()
        adapter._send_control_message = AsyncMock(
            return_value=SimpleNamespace(message_id=77))
        prompt = SimpleNamespace(
            text="⚠️ <b>H</b>\n\n<pre>ls</pre>",
            session_key="agent:main:telegram:dm:1:2",
            actions=[("Allow Once", "once", None), ("Deny", "deny", None)],
            chat_id="12345", metadata={})

        result = await adapter._send_exec_approval_prompt(prompt)

        assert result.success is True
        approval_id = next(iter(adapter._approval_state))
        stored = adapter._approval_state[approval_id]
        assert stored["session_key"] == "agent:main:telegram:dm:1:2"
        assert stored["text"] == "⚠️ <b>H</b>\n\n<pre>ls</pre>"


class TestApprovalResolutionHtml:
    """Unit contract for the prompt + decision composition."""

    def test_empty_prompt_returns_none(self):
        assert _make_adapter()._approval_resolution_html("", "x") is None

    def test_short_prompt_appends_escaped_decision(self):
        out = _make_adapter()._approval_resolution_html(
            "<pre>ls</pre>", "Approved once by Alice_Bob")
        assert out.startswith("<pre>ls</pre>")
        assert out.endswith("— Approved once by Alice_Bob")

    def test_over_budget_cut_keeps_cap_and_valid_html(self):
        adapter = _make_adapter()
        body = "<pre>" + "x" * 4050 + "&amp;" + "y" * 60 + "</pre>\n\ntail"
        out = adapter._approval_resolution_html(body, "Approved once by Norbert")
        assert utf16_len(out) <= adapter.MAX_MESSAGE_LENGTH
        assert out.endswith("— Approved once by Norbert")
        assert "…" in out
        assert out.count("<pre>") == out.count("</pre>")
        assert "&am…" not in out and "&a…" not in out

    def test_cut_sweep_never_splits_tag_or_entity(self):
        """Pin the #129161 cut-boundary defect: over prompt lengths x
        first_name lengths, no cut may land inside a tag (``</b…``) or an
        entity, and the ``b``/``pre`` tags stay balanced and nested."""
        adapter = _make_adapter()
        prompts = set()
        for reps in range(150, 260):
            prompts.add(adapter._format_exec_approval(
                "python -c " + "print('x'); " * reps, "Looks destructive", True))
        cut, checked, failures = 0, 0, []
        for prompt in sorted(prompts):
            for fn_len in range(1, 65):
                out = adapter._approval_resolution_html(
                    prompt, "Approved once by " + "x" * fn_len)
                assert out is not None
                checked += 1
                assert utf16_len(out) <= adapter.MAX_MESSAGE_LENGTH
                if "…" not in out:
                    continue
                cut += 1
                problems = []
                tag_frag = re.search(r"<[^>…]*…", out)
                if tag_frag:
                    problems.append(f"cut-in-tag:{tag_frag.group(0)[-12:]}")
                ent_frag = re.search(r"&[^;\s]*…", out)
                if ent_frag:
                    problems.append(f"cut-in-entity:{ent_frag.group(0)}")
                stack = []
                for m in re.finditer(r"<", out):
                    tok = re.match(r"</?(?:b|pre)>", out[m.start():])
                    if tok is None:
                        problems.append(f"partial-tag:{out[m.start():m.start() + 12]}")
                        break
                    if tok.group(0).startswith("</"):
                        if not stack or f"<{tok.group(0)[2:-1]}>" != stack[-1]:
                            problems.append(f"mismatch:{tok.group(0)}")
                            break
                        stack.pop()
                    else:
                        stack.append(tok.group(0))
                if stack:
                    problems.append(f"unclosed:{stack}")
                for m in re.finditer(r"&", out):
                    if not re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[A-Za-z]+);").match(out, m.start()):
                        problems.append(f"bad-entity:{out[m.start():m.start() + 12]}")
                        break
                if problems and len(failures) < 5:
                    idx = out.find("…")
                    failures.append((fn_len, problems, out[max(0, idx - 40):idx + 20]))
        assert cut > 0, "sweep produced no over-budget cuts"
        assert not failures, f"{len(failures)} malformed payloads of {cut} cuts ({checked} checked)"
