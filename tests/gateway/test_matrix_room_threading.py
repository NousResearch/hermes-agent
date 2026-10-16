"""Per-room Matrix threading policy, loaded from YAML (regression for #136742)."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from hermes_cli.plugins import discover_plugins
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["matrix", "platforms", "gateway"])
@pytest.mark.parametrize("msgtype", ["m.text", "m.image"])
@pytest.mark.parametrize("mentioned", [False, True])
@pytest.mark.parametrize("is_dm", [False, True])
@pytest.mark.parametrize("scope,auto,dm_auto,dm_mentions", [
    ("auto", False, False, False),
    ("auto", True, False, False),
    ("room", True, True, True),
    ("thread", False, False, False),
    ("auto", False, True, False),
    ("auto", False, False, True),
])
async def test_room_overrides_control_creation_but_preserve_real_threads(
        tmp_path, monkeypatch, section, msgtype, mentioned, is_dm,
        scope, auto, dm_auto, dm_mentions):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key, value in {
        "MATRIX_SESSION_SCOPE": scope,
        "MATRIX_AUTO_THREAD": str(auto).lower(),
        "MATRIX_DM_AUTO_THREAD": str(dm_auto).lower(),
        "MATRIX_DM_MENTION_THREADS": str(dm_mentions).lower(),
    }.items():
        monkeypatch.setenv(key, value)
    settings = {
        "enabled": True,
        "user_id": "@hermes:example.org",
        "require_mention": False,
        "session_scope": scope,
        "auto_thread": auto,
        "dm_mention_threads": dm_mentions,
        "room_overrides": {
            "!continuous:example.org": {"auto_thread": False},
            "!tasks:example.org": {"auto_thread": True},
            "!inherit:example.org": {},
        },
    }
    config = {"matrix": settings}
    if section == "platforms":
        config = {"platforms": config}
    elif section == "gateway":
        config = {"gateway": config}
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    discover_plugins()
    adapter = MatrixAdapter(load_gateway_config().platforms[Platform.MATRIX])
    adapter._text_batch_delay_seconds = 0
    adapter._background_read_receipt = MagicMock()
    adapter.handle_message = AsyncMock()
    adapter._download_and_cache_media = AsyncMock(return_value=str(tmp_path / "image.png"))
    adapter._client = MagicMock()
    adapter._client.state_store.get_members = AsyncMock(
        return_value=["@hermes:example.org", "@alice:example.org"] + (
            [] if is_dm else ["@bob:example.org"]))
    adapter._client.get_state_event = AsyncMock(return_value={})
    adapter._client.send_message_event = AsyncMock(return_value="$sent")

    default = ((dm_auto or (dm_mentions and mentioned)) if is_dm else
               (scope == "thread" or (scope != "room" and auto)))
    for room, synthetic in (
        ("continuous", False), ("tasks", True), ("inherit", default), ("unset", default),
    ):
        room_id = f"!{room}:example.org"
        for existing_root in (None, "$existing"):
            event_id = f"${room}-{existing_root or 'new'}"
            content = {
                "msgtype": msgtype,
                "body": "Please help",
                "m.mentions": {"user_ids": ["@hermes:example.org"] if mentioned else []},
            }
            if msgtype == "m.image":
                content.update(url="mxc://example.org/image", info={"mimetype": "image/png"})
            if existing_root:
                content["m.relates_to"] = {"rel_type": "m.thread", "event_id": existing_root}
            event = SimpleNamespace(
                room_id=room_id, sender="@alice:example.org", event_id=event_id,
                timestamp=int(time.time() * 1000), content=content)
            adapter.handle_message.reset_mock()
            await adapter._on_room_message(event)
            adapter.handle_message.assert_awaited_once()
            source = adapter.handle_message.await_args.args[0].source
            expected_root = existing_root or (event_id if synthetic else None)
            assert source.thread_id == expected_root
            assert source.chat_type == ("dm" if is_dm else "group")
            assert source.parent_chat_id == (room_id if expected_root else None)

            result = await adapter.send(
                source.chat_id, "Reply", reply_to=event_id,
                metadata={"thread_id": source.thread_id})
            assert result.success
            sent = adapter._client.send_message_event.await_args.args[2]
            relation = sent.get("m.relates_to", {})
            if expected_root:
                assert relation["rel_type"] == "m.thread"
                assert relation["event_id"] == expected_root
            else:
                assert relation.get("rel_type") != "m.thread"


@pytest.mark.parametrize("overrides", [
    None,
    [],
    {"#alias:example.org": {"auto_thread": True}},
    {"!room:example.org": False},
    {"!room:example.org": {"auto_thread": "false"}},
    {"!room:example.org": {"auto_thread": 0}},
    {"!room:example.org": {"auto_thread": None}},
    {"!room:example.org": {"auto_threads": False}},
])
def test_invalid_room_threading_overrides_fail_explicitly(overrides):
    with pytest.raises(ValueError, match=r"matrix\.room_overrides"):
        MatrixAdapter(PlatformConfig(extra={"room_overrides": overrides}))
