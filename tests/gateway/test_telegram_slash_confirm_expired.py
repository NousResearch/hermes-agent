"""A slash-confirm card tapped after its confirmation timed out must not say "Approved".

``tools.slash_confirm.resolve`` refuses a confirm older than ``DEFAULT_TIMEOUT_SECONDS`` and runs
nothing, but the Telegram tap handler rendered the "Approved once" toast and card edit before it
asked: the user read "approved" while the command never ran.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.i18n import t
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import slash_confirm
from tools.approval_context import prompt_authority

_SESSION = "agent:main:telegram:dm:12345"


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    adapter._app = MagicMock()
    return adapter


async def _card(adapter: TelegramAdapter, confirm_id: str, handler) -> str:
    """Register and send a confirm the way ``GatewayRunner._request_slash_confirm`` does."""
    slash_confirm.register(_SESSION, confirm_id, "reload-mcp", handler,
                           authority=prompt_authority("12345", "Ana", "dm"))
    await adapter.send_slash_confirm(chat_id="12345", title="/reload-mcp", message="Reload MCP servers?",
                                     session_key=_SESSION, confirm_id=confirm_id)
    return f"sc:once:{confirm_id}"


async def _tap(adapter: TelegramAdapter, data: str) -> AsyncMock:
    query = AsyncMock(data=data)
    query.message = MagicMock(chat_id=12345, text_html="<b>/reload-mcp</b>")
    query.from_user = MagicMock(first_name="Ana", id="12345")
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
    return query


def _said(query: AsyncMock) -> str:
    """Everything the user saw: the toast and the card edit."""
    calls = query.answer.call_args_list + query.edit_message_text.call_args_list
    return " ".join(str(c.kwargs.get("text") or (c.args[0] if c.args else "")) for c in calls)


@pytest.mark.asyncio
async def test_tap_after_the_confirm_timed_out_runs_nothing_and_says_expired():
    adapter, handler = _adapter(), AsyncMock(return_value="reloaded")
    card = await _card(adapter, "7", handler)
    slash_confirm._pending[_SESSION]["created_at"] -= slash_confirm.DEFAULT_TIMEOUT_SECONDS + 1

    said = _said(await _tap(adapter, card))

    handler.assert_not_called()
    assert t("platform.telegram.slash_confirm.resolved_once") not in said, said
    assert t("platform.telegram.approval.expired") in said, said


@pytest.mark.asyncio
async def test_tap_in_time_still_runs_the_command_and_says_approved():
    adapter, handler = _adapter(), AsyncMock(return_value="reloaded")
    card = await _card(adapter, "8", handler)

    said = _said(await _tap(adapter, card))

    handler.assert_awaited_once_with("once")
    assert t("platform.telegram.slash_confirm.resolved_once") in said, said
