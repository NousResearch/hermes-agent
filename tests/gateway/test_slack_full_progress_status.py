"""Slack Assistant status stays active through threaded full-progress sends."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter  # SDK capability is initialized by this test module


@pytest.fixture
def slack_adapter():
    instance = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    instance._app = MagicMock()
    instance._app.client = AsyncMock()
    instance._bot_user_id = "U_BOT"
    instance._running = True
    return instance


@pytest.mark.asyncio
async def test_full_progress_preserves_status_until_final(slack_adapter):
    adapter = slack_adapter
    client = adapter._app.client
    client.chat_postMessage.return_value = {"ok": True, "ts": "progress_ts"}
    client.assistant_threads_setStatus = AsyncMock()
    key = ("", "C123", "parent_ts")
    adapter._active_status_threads[key] = {"thread_ts": "parent_ts", "team_id": ""}
    metadata = {"thread_id": "parent_ts", "_interim_send": True, "_literal_text": True}

    result = await adapter.send("C123", "tool arguments", metadata=metadata)
    assert result.success
    assert key in adapter._active_status_threads
    client.assistant_threads_setStatus.assert_not_awaited()
    assert client.chat_postMessage.await_args.kwargs["thread_ts"] == "parent_ts"
    assert "_interim_send" in metadata

    await adapter.send("C123", "final answer", metadata={"thread_id": "parent_ts"})
    assert key not in adapter._active_status_threads
    client.assistant_threads_setStatus.assert_awaited_once()
