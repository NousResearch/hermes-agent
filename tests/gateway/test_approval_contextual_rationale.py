"""Test contextual_reason parameter in send_exec_approval across adapters.

Verifies that the new ``contextual_reason`` kwarg is accepted by all adapter
``send_exec_approval`` methods and renders the rationale text in the approval
prompt when provided.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ===========================================================================
# Helpers
# ===========================================================================

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
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()


def _make_telegram_adapter():
    from gateway.platforms.telegram import TelegramAdapter
    from gateway.config import PlatformConfig
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


# ===========================================================================
# Telegram
# ===========================================================================

class TestTelegramContextualRationale:

    @pytest.mark.asyncio
    async def test_rationale_appears_in_text(self):
        adapter = _make_telegram_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="rm -rf /tmp/cache",
            session_key="s:key",
            description="recursive delete",
            contextual_reason="I need to free up disk space before the build.",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        text = kwargs["text"]
        assert "I need to free up disk space" in text
        assert "recursive delete" in text

    @pytest.mark.asyncio
    async def test_no_rationale_omits_block(self):
        adapter = _make_telegram_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="rm -rf /tmp",
            session_key="s:key",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert "Command Approval Required" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_rationale_html_escaped(self):
        adapter = _make_telegram_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="echo test",
            session_key="s:key",
            contextual_reason="This has <script>alert(1)</script> tags",
        )

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        text = kwargs["text"]
        assert "&lt;script&gt;" in text
        assert "<script>" not in text


# ===========================================================================
# Feishu
# ===========================================================================

class TestFeishuContextualRationale:

    @pytest.mark.asyncio
    async def test_rationale_in_card_markdown(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._client = MagicMock()
        adapter._approval_counter = iter(range(1, 100))
        adapter._approval_state = {}
        adapter._feishu_send_with_retry = AsyncMock(
            return_value=MagicMock(data={"data": {"message_id": "om_xxx"}})
        )
        adapter._finalize_send_result = MagicMock(
            return_value=MagicMock(success=True, message_id="om_xxx")
        )

        result = await adapter.send_exec_approval(
            chat_id="oc_xxx",
            command="rm -rf /tmp",
            session_key="s:key",
            description="dangerous command",
            contextual_reason="Cleaning up temp files for the new build.",
        )

        assert result.success is True
        call_args = adapter._feishu_send_with_retry.call_args
        payload = call_args.kwargs.get("payload") or call_args[1].get("payload")
        card = json.loads(payload)
        md_content = card["elements"][0]["content"]
        assert "Cleaning up temp files" in md_content

    @pytest.mark.asyncio
    async def test_no_rationale_clean(self):
        from gateway.platforms.feishu import FeishuAdapter
        adapter = object.__new__(FeishuAdapter)
        adapter._client = MagicMock()
        adapter._approval_counter = iter(range(1, 100))
        adapter._approval_state = {}
        adapter._feishu_send_with_retry = AsyncMock(
            return_value=MagicMock(data={"data": {"message_id": "om_xxx"}})
        )
        adapter._finalize_send_result = MagicMock(
            return_value=MagicMock(success=True, message_id="om_xxx")
        )

        result = await adapter.send_exec_approval(
            chat_id="oc_xxx",
            command="rm -rf /tmp",
            session_key="s:key",
        )

        assert result.success is True


# ===========================================================================
# Matrix
# ===========================================================================

class TestMatrixContextualRationale:

    @pytest.mark.asyncio
    async def test_rationale_in_text(self):
        from gateway.platforms.matrix import MatrixAdapter
        from gateway.config import PlatformConfig
        config = PlatformConfig(enabled=True)
        adapter = MatrixAdapter(config)
        adapter._client = MagicMock()
        adapter._approval_prompt_by_session = {}
        adapter._approval_prompts_by_event = {}

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message_id = "evt_1"
        adapter.send = AsyncMock(return_value=mock_result)
        adapter._send_reaction = AsyncMock(return_value="evt_react")

        result = await adapter.send_exec_approval(
            chat_id="!room:server",
            command="rm -rf /tmp",
            session_key="s:key",
            description="dangerous command",
            contextual_reason="Removing stale cache files.",
        )

        assert result.success is True
        send_text = adapter.send.call_args[0][1]
        assert "Removing stale cache files" in send_text


# ===========================================================================
# Signature acceptance tests (parameter accepted without error)
# ===========================================================================

class TestAdapterSignatureAcceptsContextualReason:
    """Ensure all adapters accept the new contextual_reason kwarg."""

    @pytest.mark.asyncio
    async def test_telegram_accepts_contextual_reason(self):
        adapter = _make_telegram_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)
        # Should not raise TypeError
        result = await adapter.send_exec_approval(
            chat_id="1", command="echo hi", session_key="s",
            contextual_reason="Test rationale",
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_matrix_accepts_contextual_reason(self):
        from gateway.platforms.matrix import MatrixAdapter
        from gateway.config import PlatformConfig
        adapter = MatrixAdapter(PlatformConfig(enabled=True))
        adapter._client = MagicMock()
        adapter._approval_prompt_by_session = {}
        adapter._approval_prompts_by_event = {}
        mock_result = MagicMock(success=True, message_id="e1")
        adapter.send = AsyncMock(return_value=mock_result)
        adapter._send_reaction = AsyncMock(return_value="r1")

        result = await adapter.send_exec_approval(
            chat_id="!r:s", command="echo", session_key="s",
            contextual_reason="Rationale here",
        )
        assert result.success is True
