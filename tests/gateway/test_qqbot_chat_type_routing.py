"""QQ Bot chat-type routing for sends to a chat this process never received from.

Regression: ``QQAdapter._chat_type_map`` is only populated by inbound traffic and lives in
memory, so the first send to a group after a gateway restart guessed ``c2c`` and posted to
``/v2/users/<group_openid>``; QQ answered 400 "请求的资源不存在(用户/群已注销)" and the reply was
lost. The channel directory persists the same chat types across restarts, so it backs the map.

Coroutines are driven with ``asyncio.run`` (no pytest-asyncio dependency).
"""

import asyncio
import json
from unittest import mock

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from hermes_cli.config import get_hermes_home

GROUP_OPENID = "6FEFBC76E87133E0BA89620EA8F10D90"
USER_OPENID = "4A20B9D409B9A73AD64B6B7BD1820BF5"


def _adapter(**extra):
    from gateway.platforms.qqbot import QQAdapter
    return QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b", **extra}))


def _persist_directory(entries):
    path = get_hermes_home() / "channel_directory.json"
    path.write_text(json.dumps({"updated_at": None, "platforms": {"qqbot": entries}}), encoding="utf-8")
    return path


def _sent_path(adapter, chat_id, content="hi"):
    """Path of the one POST ``_send_chunk`` issues for *chat_id*."""
    post = mock.AsyncMock(return_value=SendResult(success=True, message_id="m1"))
    with mock.patch.object(adapter, "_post_message", post):
        result = asyncio.run(adapter._send_chunk(chat_id, content))
    assert result.success, result.error
    return post.await_args.args[0]


class TestChatTypeAfterRestart:
    def test_group_from_directory_targets_group_endpoint(self):
        _persist_directory([{"id": GROUP_OPENID, "name": GROUP_OPENID, "type": "group", "thread_id": None}])
        adapter = _adapter()
        assert _sent_path(adapter, GROUP_OPENID) == f"/v2/groups/{GROUP_OPENID}/messages"

    def test_dm_from_directory_targets_users_endpoint(self):
        _persist_directory([{"id": USER_OPENID, "name": USER_OPENID, "type": "dm", "thread_id": None}])
        adapter = _adapter()
        assert _sent_path(adapter, USER_OPENID) == f"/v2/users/{USER_OPENID}/messages"

    def test_unknown_chat_still_defaults_to_c2c(self):
        _persist_directory([])
        adapter = _adapter()
        assert _sent_path(adapter, GROUP_OPENID) == f"/v2/users/{GROUP_OPENID}/messages"

    def test_inbound_learned_scene_wins_over_directory(self):
        """The live scene is authoritative: a guild channel id recorded as "group" in the
        directory must not override what this process learned from an inbound event."""
        _persist_directory([{"id": GROUP_OPENID, "name": GROUP_OPENID, "type": "group", "thread_id": None}])
        adapter = _adapter()
        adapter._chat_type_map[GROUP_OPENID] = "guild"
        assert _sent_path(adapter, GROUP_OPENID) == f"/channels/{GROUP_OPENID}/messages"
