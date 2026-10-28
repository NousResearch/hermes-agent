"""The Block Kit approval prompt must leave the composer usable.

Slack auto-clears the assistant status server-side when the approval
message posts into the thread, but the adapter's tracked entry survived,
so a delayed "is thinking" write re-created the status and disabled the
compose box for the whole approval wait. The approval send now runs the
client-side clear, dropping the tracked entry.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


class RecordingClient:
    def __init__(self):
        self.calls = []

    async def assistant_threads_setStatus(self, channel_id, thread_ts, status):
        self.calls.append((thread_ts, status))

    async def chat_postMessage(self, **kwargs):
        self.calls.append((kwargs.get("thread_ts"), "post"))
        return {"ts": "999.000"}


def make_adapter(client):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._app = MagicMock()
    adapter._get_client = lambda chat_id, team_id=None: client
    adapter._channel_team = {"C123": "T1"}
    return adapter


META = {"thread_id": "100.000", "team_id": "T1"}


@pytest.mark.asyncio
async def test_exec_approval_clears_tracked_status_after_posting():
    client = RecordingClient()
    adapter = make_adapter(client)

    await adapter.send_typing("C123", dict(META))
    assert ("T1", "C123", "100.000") in adapter._active_status_threads

    result = await adapter.send_exec_approval(
        "C123",
        command="rm -rf /tmp/x",
        session_key="s1",
        metadata=dict(META),
    )

    assert result.success
    assert client.calls[-1] == ("100.000", "")  # explicit client-side clear
    assert ("T1", "C123", "100.000") not in adapter._active_status_threads


@pytest.mark.asyncio
async def test_exec_approval_without_thread_context_skips_clear():
    client = RecordingClient()
    adapter = make_adapter(client)

    result = await adapter.send_exec_approval(
        "C123",
        command="ls",
        session_key="s1",
        metadata=None,
    )

    assert result.success
    assert [s for _, s in client.calls if s == ""] == []  # nothing to clear
