"""Channel-scoped NIP-10 ancestry: replies join the real root through seeds, sends and restarts."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_buzz_adapter import CHANNEL, OTHER_PUBKEY, SELF_PUBKEY, _event, _make_adapter


def nested(event_id, parent, *, channel=CHANNEL, marked=True, pubkey=OTHER_PUBKEY, content="@Chip /whoami"):
    event = _event(event_id, pubkey=pubkey, content=content)
    event["tags"] = [["h", channel], ["e", parent, "", "reply"] if marked else ["e", parent]]
    return event


@pytest.fixture
def adapter():
    a = _make_adapter()
    a._dispatched = []

    async def capture(**kwargs):
        a._dispatched.append(kwargs)

    a._dispatch_message = capture
    a._message_handler = AsyncMock()
    a._run_cli = AsyncMock(return_value=(0, "[]", ""))
    a._channel_state[CHANNEL] = a._new_channel_state("group")
    return a


def _last_thread(a):
    return a._dispatched[-1]["thread_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("marked", [True, False])
async def test_newest_first_seed_routes_dispatch_and_send_without_echo(adapter, marked):
    a = adapter
    a._run_cli = AsyncMock(return_value=(0, json.dumps([
        nested("leaf", "middle", marked=marked), nested("middle", "original", marked=marked),
        _event("original")]), ""))
    await a._seed_channel(CHANNEL, "group")
    assert a._dispatched == []
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], nested("new-child", "leaf", marked=marked))
    assert _last_thread(a) == "original"
    # No relay echo of our own send is injected; the receipt alone must anchor the follow-up.
    a._run_cli = AsyncMock(return_value=(0, json.dumps({"accepted": True, "event_id": "outgoing"}), ""))
    result = await a.send(CHANNEL, "answer", reply_to="leaf")
    assert result.success
    args = a._run_cli.call_args.args[0]
    assert args[args.index("--reply-to") + 1] == "original"
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], nested("after-send", "outgoing"))
    assert _last_thread(a) == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "file"])
async def test_attachment_receipt_preserves_noecho_root(adapter, tmp_path, kind):
    a = adapter
    path = tmp_path / "image.png"
    path.write_bytes(b"not-a-real-image")
    a._run_cli = AsyncMock(return_value=(0, json.dumps({"accepted": True, "event_id": "media-out"}), ""))
    if kind == "image":
        result = await a.send_image_file(CHANNEL, str(path), reply_to="media-root")
    else:
        result = await a.send_document(CHANNEL, str(path), reply_to="media-root")
    assert result.success
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], nested("media-followup", "media-out"))
    assert _last_thread(a) == "media-root"


@pytest.mark.asyncio
async def test_seed_ancestry_is_scoped_to_channel(adapter):
    a = adapter
    for channel, root in [(CHANNEL, "our-root"), ("second-channel", "their-root")]:
        a._run_cli = AsyncMock(return_value=(0, json.dumps([nested("same-id", root, channel=channel)]), ""))
        await a._seed_channel(channel, "group")
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], nested("ours", "same-id"))
    assert _last_thread(a) == "our-root"
    await a._handle_event("second-channel", a._channel_state["second-channel"],
                          nested("theirs", "same-id", channel="second-channel"))
    assert _last_thread(a) == "their-root"


@pytest.mark.asyncio
async def test_self_echo_ancestry_is_routing_not_mention_authority(adapter):
    a = adapter
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL],
                          nested("self-child", "echo-root", pubkey=SELF_PUBKEY, content="prompt"))
    assert a._dispatched == []
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], nested("mentioned", "self-child"))
    assert _last_thread(a) == "echo-root"


@pytest.mark.asyncio
async def test_dm_root_and_cyclic_seed_are_bounded(adapter):
    a = adapter
    a._channel_state[CHANNEL]["chat_type"] = "dm"
    a._channel_meta[CHANNEL] = {"name": "DM", "description": "DM"}
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], _event("dm-top", content="@Chip /whoami"))
    assert _last_thread(a) is None
    await a._handle_event(CHANNEL, a._channel_state[CHANNEL], nested("dm-next", "dm-top"))
    assert _last_thread(a) == "dm-top"
    a._run_cli = AsyncMock(return_value=(0, json.dumps([
        nested("cycle-a", "cycle-b"), nested("cycle-b", "cycle-a")]), ""))
    await a._seed_channel(CHANNEL, "group")
    # Neither corrupt/cyclic history nor an unknown parent can hang publication.
    a._run_cli = AsyncMock(return_value=(0, json.dumps({"accepted": True, "event_id": "cycle-out"}), ""))
    for anchor in ("cycle-a", "unknown"):
        result = await asyncio.wait_for(a.send(CHANNEL, "answer", reply_to=anchor), 2)
        assert result.success
        args = a._run_cli.call_args.args[0]
        assert args[args.index("--reply-to") + 1] == anchor


def test_legacy_anchor_resolution_without_channel_needs_unambiguous_ancestry():
    a = _make_adapter()
    a._record_thread_root("child", {"tags": [["e", "root", "", "root"]]}, CHANNEL)
    assert a._resolve_reply_anchor("child") == "root"
    a._record_thread_root("child", {"tags": [["e", "other-root", "", "root"]]}, "second-channel")
    assert a._resolve_reply_anchor("child") == "child"
    assert a._resolve_reply_anchor("child", CHANNEL) == "root"
