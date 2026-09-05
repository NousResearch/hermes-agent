"""Characterize Slack workspace scope in channel-directory entries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.channel_directory import _build_slack


def _workspace_client(channel_id: str, name: str):
    client = MagicMock()
    client.users_conversations = AsyncMock(return_value={
        "ok": True,
        "channels": [{
            "id": channel_id,
            "name": name,
            "is_private": False,
        }],
        "response_metadata": {},
    })
    return client


def test_multi_workspace_directory_entries_are_unscoped():
    """Directory discovery records Slack conversation IDs, not workspace IDs.

    This is a characterization test for the current directory format. It does
    not assert that an unscoped entry is a guaranteed outbound delivery target.
    """
    alpha = _workspace_client("C_ALPHA", "alpha-engineering")
    beta = _workspace_client("C_BETA", "beta-engineering")
    adapter = SimpleNamespace(_team_clients={"T_ALPHA": alpha, "T_BETA": beta})

    entries = asyncio.run(_build_slack(adapter))

    assert entries == [
        {"id": "C_ALPHA", "name": "alpha-engineering", "type": "channel"},
        {"id": "C_BETA", "name": "beta-engineering", "type": "channel"},
    ]
    alpha.users_conversations.assert_awaited_once()
    beta.users_conversations.assert_awaited_once()
