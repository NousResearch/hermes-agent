import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageSendContext
from hermes_cli.plugins import VALID_HOOKS


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack = MagicMock()
    slack.async_app.AsyncApp = MagicMock
    slack.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack.web.async_client.AsyncWebClient = MagicMock
    for name in (
        "slack_bolt",
        "slack_bolt.async_app",
        "slack_bolt.adapter",
        "slack_bolt.adapter.socket_mode",
        "slack_bolt.adapter.socket_mode.async_handler",
        "slack_sdk",
        "slack_sdk.web",
        "slack_sdk.web.async_client",
        "slack_sdk.errors",
    ):
        sys.modules.setdefault(name, slack)


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def adapter():
    value = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    value._app = MagicMock()
    value._pop_slash_context = MagicMock(return_value=None)
    value._resolve_thread_ts = MagicMock(return_value=None)
    value.stop_typing = AsyncMock()
    client = MagicMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.222"})
    client.chat_update = AsyncMock(return_value={"ok": True})
    value._get_client = MagicMock(return_value=client)
    return value


def test_hook_contract_is_public():
    assert "pre_message_send" in VALID_HOOKS
    ctx = MessageSendContext(platform="slack", chat_id="C1", content="hello")
    assert ctx.metadata == {}
    assert ctx.cancel is False


def test_hook_rewrite_reaches_send(adapter):
    def rewrite(_name, *, ctx, **_kwargs):
        ctx.content = ctx.content.replace("Recovered", "Recovery unverified")

    with patch("hermes_cli.plugins.invoke_hook", side_effect=rewrite):
        result = _run(adapter.send("C1", "**Status:** Recovered"))

    assert result.success
    posted = adapter._get_client().chat_postMessage.await_args.kwargs["text"]
    assert "Recovery unverified" in posted
    assert "Recovered" not in posted


def test_hook_cancel_suppresses_send(adapter):
    def cancel(_name, *, ctx, **_kwargs):
        ctx.cancel = True

    with patch("hermes_cli.plugins.invoke_hook", side_effect=cancel):
        result = _run(adapter.send("C1", "progress only"))

    assert result.success
    assert result.message_id is None
    adapter._get_client().chat_postMessage.assert_not_awaited()


def test_only_final_edit_runs_hook(adapter):
    seen = []

    def rewrite(_name, *, ctx, **_kwargs):
        seen.append(ctx.content)
        ctx.content = "guarded final"

    with patch("hermes_cli.plugins.invoke_hook", side_effect=rewrite):
        preview = _run(adapter.edit_message("C1", "111.222", "preview"))
        final = _run(
            adapter.edit_message("C1", "111.222", "unverified", finalize=True)
        )

    assert preview.success and final.success
    assert seen == ["unverified"]
    calls = adapter._get_client().chat_update.await_args_list
    assert calls[0].kwargs["text"] == "preview"
    assert calls[1].kwargs["text"] == "guarded final"
