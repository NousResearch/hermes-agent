"""Pending clarify button prompts survive a Discord gateway reconnect.

A WebSocket drop makes ``connect()`` build a brand-new ``commands.Bot``. Views
are dispatched by the client that sent them, so buttons sent on the old client
are orphaned: every click answers "This interaction failed / application did
not respond" while the clarify entry keeps waiting until ``clarify_timeout``.

After the fix, ``send_clarify`` records button prompts and ``on_ready`` calls
``_repost_live_clarify_prompts()``: still-pending prompts lose their dead
buttons and are re-sent with the SAME clarify_id (so the new buttons resolve
the same waiting entry). Answered/expired prompts and prompts whose client did
not change are left alone.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.discord.adapter import DiscordAdapter  # conftest installs the discord mock
from gateway.config import PlatformConfig


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.fixture(autouse=True)
def _isolated_clarify_state():
    _clear_clarify_state()
    yield
    _clear_clarify_state()


def _make_adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()

    channel = MagicMock()
    sent_ids = iter(range(1000, 2000))

    async def _send(**kwargs):
        msg = MagicMock()
        msg.id = next(sent_ids)
        channel.sent.append(kwargs)
        return msg

    channel.sent = []
    channel.send = AsyncMock(side_effect=_send)
    partial = MagicMock()
    partial.edit = AsyncMock()
    channel.get_partial_message = MagicMock(return_value=partial)
    adapter._resolve_channel = AsyncMock(return_value=channel)
    return adapter, channel, partial


def _register(clarify_id, choices=("A", "B")):
    from tools import clarify_gateway as cm
    cm.register(clarify_id, "sk", "Pick one?", list(choices) if choices else None)


@pytest.mark.asyncio
async def test_pending_prompt_is_reposted_with_same_id_after_reconnect():
    adapter, channel, partial = _make_adapter()
    _register("cid1")
    first = await adapter.send_clarify("123", "Pick one?", ["A", "B"], "cid1", "sk")
    assert first.success and len(channel.sent) == 1

    adapter._client = MagicMock()  # reconnect: connect() built a new commands.Bot
    await adapter._repost_live_clarify_prompts()

    channel.get_partial_message.assert_called_once_with(int(first.message_id))
    partial.edit.assert_awaited_once_with(view=None)  # dead buttons stripped
    assert len(channel.sent) == 2  # re-posted
    reposted_view = channel.sent[1]["view"]
    assert reposted_view.clarify_id == "cid1"
    rec = adapter._live_clarify_prompts["cid1"]
    assert rec["client_id"] == id(adapter._client)  # tracked under the new client


@pytest.mark.asyncio
async def test_answered_prompt_is_not_reposted():
    from tools import clarify_gateway as cm
    adapter, channel, partial = _make_adapter()
    _register("cid2")
    await adapter.send_clarify("123", "Pick one?", ["A", "B"], "cid2", "sk")
    cm.resolve_gateway_clarify("cid2", "A")

    adapter._client = MagicMock()
    await adapter._repost_live_clarify_prompts()

    assert len(channel.sent) == 1
    partial.edit.assert_not_awaited()
    assert "cid2" not in adapter._live_clarify_prompts


@pytest.mark.asyncio
async def test_same_client_on_ready_refire_leaves_prompt_alone():
    adapter, channel, partial = _make_adapter()
    _register("cid3")
    await adapter.send_clarify("123", "Pick one?", ["A", "B"], "cid3", "sk")

    await adapter._repost_live_clarify_prompts()  # client unchanged: its view still listens

    assert len(channel.sent) == 1
    partial.edit.assert_not_awaited()
    assert "cid3" in adapter._live_clarify_prompts


@pytest.mark.asyncio
async def test_text_only_prompt_is_not_tracked():
    adapter, _channel, _ = _make_adapter()
    _register("cid4", choices=None)
    await adapter.send_clarify("123", "Free text?", None, "cid4", "sk")

    assert not getattr(adapter, "_live_clarify_prompts", None)


@pytest.mark.asyncio
async def test_strip_failure_still_reposts():
    adapter, channel, partial = _make_adapter()
    partial.edit.side_effect = RuntimeError("Unknown Message")
    _register("cid5")
    await adapter.send_clarify("123", "Pick one?", ["A", "B"], "cid5", "sk")

    adapter._client = MagicMock()
    await adapter._repost_live_clarify_prompts()

    assert len(channel.sent) == 2
