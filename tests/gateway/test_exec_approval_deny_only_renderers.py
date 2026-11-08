"""A shortened approval preview that cannot be backed by the full command leaves Deny as the ONLY answer on
every native renderer, and the backend refuses any other choice for that request even when a tap is forged."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult

FENCED = "echo '```' " + "A" * 4000 + " && rm -rf ~ # TAIL"  # cannot be fenced verbatim -> Deny only


@pytest.fixture
def pending(monkeypatch):
    """A real queued approval for ``sk`` / ``r1`` in ``tools.approval``'s queue."""
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    entry = _ApprovalEntry({"request_id": "r1", "command": FENCED})
    monkeypatch.setitem(approval._gateway_queues, "sk", [entry])
    return entry


def _discord_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = []

    async def fake_send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(id=len(sent))

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock())
    return adapter, sent


@pytest.mark.asyncio
async def test_discord_command_past_the_chunk_cap_gets_a_deny_only_card(pending, monkeypatch):
    """Discord's send() caps a split message at MAX_SPLIT_MESSAGES and still reports success, so the full
    command it would post is clipped: the card must not keep Allow Once."""
    from plugins.platforms.discord.adapter import ExecApprovalView

    removed = []  # the test double for discord.ui.View has no remove_item; record what the view drops
    monkeypatch.setattr(ExecApprovalView, "remove_item", lambda self, item: removed.append(item.__name__),
                        raising=False)
    adapter, sent = _discord_adapter()
    command = "echo " + "A" * 20000 + " && rm -rf /tmp/example"
    await adapter.send_exec_approval(chat_id="555", command=command, session_key="sk", request_id="r1")
    assert sorted(removed) == ["allow_always", "allow_once", "allow_session"]
    assert "rm -rf /tmp/example" not in "".join(s.get("content", "") for s in sent)
