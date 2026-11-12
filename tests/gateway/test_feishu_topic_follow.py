"""Feishu topic follow: a group @mention is answered in a new topic, and inside a topic the bot
owns, follow-ups need no @mention.

Behaviour contracts:
- ``thread_reply`` opens the topic before the turn runs, so the first turn's source already
  carries the topic's ``thread_id`` (one session for the whole topic).
- ``thread_follow`` admits an un-mentioned topic message only when this bot is the one the topic
  belongs to: in ``last_active`` mode (the default, ``true``) the bot triggered last, in ``owner``
  mode the bot that replied first. A message @mentioning someone else, a sender the group policy
  rejects, and a bot sender stay rejected.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.feishu_helpers import install_dedup_state, make_adapter_skeleton, make_sender

APP_ID = "cli_me"
PEER_APP_ID = "cli_peer"


class _FakeMessageApi:
    """Stands in for ``client.im.v1.message``: records reply/list requests."""

    def __init__(self, *, thread_id="omt_new", reply_ok=True, topic_senders=(), topic_messages=None):
        self.replies = []
        self.lists = []
        self._thread_id = thread_id
        self._reply_ok = reply_ok
        self._topic_senders = list(topic_senders)
        self._topic_messages = topic_messages

    def reply(self, request):
        self.replies.append(request)
        data = SimpleNamespace(thread_id=self._thread_id, message_id="om_ack")
        return SimpleNamespace(success=lambda: self._reply_ok, data=data, code=0 if self._reply_ok else 230001,
                               msg="ok" if self._reply_ok else "denied")

    def list(self, request):
        self.lists.append(request)
        if self._topic_messages is not None:
            items = list(self._topic_messages)
            if request.sort_type == "ByCreateTimeDesc":
                items.reverse()
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=items))
        items = [SimpleNamespace(message_id=f"om_{i}", sender=SimpleNamespace(sender_type=kind, id=sid))
                 for i, (kind, sid) in enumerate(self._topic_senders)]
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=items))


def _wire_client(adapter, api):
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=api)))

    async def _run_blocking(func, *args):
        return func(*args)

    adapter._run_blocking = _run_blocking


# --- Opening a topic for a group reply ----------------------------------------------------------


def _inbound_adapter(extra, api):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter.config = PlatformConfig(extra=extra)
    adapter._app_id = APP_ID
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_me", "", "Hermes"
    adapter._init_topic_follow_state()
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter.get_chat_info = AsyncMock(return_value={"name": "Team"})
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_id": "u1", "user_name": "Neo", "user_id_alt": None})
    adapter._resolve_source_chat_type = Mock(return_value="group")
    adapter.build_source = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    adapter._dispatch_inbound_event = AsyncMock()
    _wire_client(adapter, api)
    return adapter


def _run_inbound(adapter, *, chat_type="group", chat_id="oc_team", thread_id=None):
    message = SimpleNamespace(
        content=json.dumps({"text": "look at this"}), message_type="text", message_id="om_user",
        mentions=[], chat_id=chat_id, parent_id=None, upper_message_id=None, root_id=None, thread_id=thread_id,
    )
    asyncio.run(adapter._process_inbound_message(
        data=message, message=message, sender_id=None, chat_type=chat_type, message_id="om_user",
    ))
    return adapter.build_source.call_args.kwargs["thread_id"]


def test_thread_reply_opens_a_topic_before_the_turn_and_routes_the_turn_into_it():
    api = _FakeMessageApi(thread_id="omt_new")
    adapter = _inbound_adapter({"thread_reply": True}, api)

    assert _run_inbound(adapter) == "omt_new"
    (request,) = api.replies
    assert request.message_id == "om_user"
    assert request.request_body.reply_in_thread is True
    assert json.loads(request.request_body.content)["text"]
    # The topic this bot opened is its own without asking Feishu.
    assert asyncio.run(adapter._topic_owned_by_self("omt_new")) is True
    assert api.lists == []


@pytest.mark.parametrize("extra", [
    {},
    {"thread_reply": False},
    {"thread_reply": True, "group_rules": {"oc_team": {"policy": "open", "thread_reply": False}}},
])
def test_no_topic_when_thread_reply_is_off_for_the_chat(extra):
    api = _FakeMessageApi()
    adapter = _inbound_adapter(extra, api)

    assert _run_inbound(adapter) is None
    assert api.replies == []


def test_group_rule_turns_thread_reply_on_for_one_chat():
    api = _FakeMessageApi(thread_id="omt_rule")
    adapter = _inbound_adapter({"group_rules": {"oc_team": {"policy": "open", "thread_reply": "true"}}}, api)

    assert _run_inbound(adapter, chat_id="oc_team") == "omt_rule"
    assert _run_inbound(adapter, chat_id="oc_other") is None


@pytest.mark.parametrize("kwargs", [{"chat_type": "p2p"}, {"thread_id": "omt_existing"}])
def test_dm_and_existing_topic_messages_never_open_a_topic(kwargs):
    api = _FakeMessageApi()
    adapter = _inbound_adapter({"thread_reply": True}, api)

    assert _run_inbound(adapter, **kwargs) == kwargs.get("thread_id")
    assert api.replies == []


def test_refused_topic_falls_back_to_an_ordinary_group_reply():
    api = _FakeMessageApi(reply_ok=False)
    adapter = _inbound_adapter({"thread_reply": True}, api)

    assert _run_inbound(adapter) is None
    adapter._dispatch_inbound_event.assert_awaited_once()


# --- Following up inside an owned topic without @mention ----------------------------------------


def _follow_adapter(api, *, extra=None, group_policy="open"):
    adapter = make_adapter_skeleton(group_policy=group_policy)
    adapter._app_id = APP_ID
    adapter.config = PlatformConfig(extra={"thread_follow": True} if extra is None else extra)
    adapter._init_topic_follow_state()
    install_dedup_state(adapter)
    _wire_client(adapter, api)
    adapter._process_inbound_message = AsyncMock()
    return adapter


_n = iter(range(10_000))


def _deliver(adapter, *, thread_id: str | None = "omt_1", mentions=None, sender=None):
    message = SimpleNamespace(
        message_id=f"om_in_{next(_n)}", chat_type="group", chat_id="oc_team", thread_id=thread_id,
        mentions=mentions, content="", message_type="text",
    )
    adapter._process_inbound_message.reset_mock()
    asyncio.run(adapter._handle_message_event_data(
        SimpleNamespace(event=SimpleNamespace(sender=sender or make_sender(open_id="ou_neo"), message=message))
    ))
    return adapter._process_inbound_message.await_count == 1


_OWNER = {"thread_follow": "owner"}


def test_owner_mode_admits_unmentioned_message_in_own_topic_and_caches_ownership():
    api = _FakeMessageApi(topic_senders=[("user", "ou_neo"), ("app", APP_ID)])
    adapter = _follow_adapter(api, extra=_OWNER)

    assert _deliver(adapter) is True
    assert _deliver(adapter) is True
    assert len(api.lists) == 1


def test_owner_mode_topic_stays_with_its_first_bot_after_this_bot_was_mentioned_once():
    # Y answered first; X was @mentioned later and replied once. The topic is still Y's.
    api = _FakeMessageApi(topic_senders=[("user", "ou_neo"), ("app", PEER_APP_ID), ("app", APP_ID)])
    adapter = _follow_adapter(api, extra=_OWNER)

    assert _deliver(adapter) is False


def test_message_mentioning_someone_else_is_left_to_them():
    api = _FakeMessageApi(topic_senders=[("app", APP_ID)])
    adapter = _follow_adapter(api)
    colleague = SimpleNamespace(key="@_user_1", id=SimpleNamespace(open_id="ou_colleague", user_id=None), name="Ann")

    assert _deliver(adapter, mentions=[colleague]) is False
    assert api.lists == []


def test_owner_mode_topic_no_bot_has_answered_yet_is_not_followed_and_is_asked_again():
    api = _FakeMessageApi(topic_senders=[("user", "ou_neo")])
    adapter = _follow_adapter(api, extra=_OWNER)

    assert _deliver(adapter) is False
    assert _deliver(adapter) is False
    assert len(api.lists) == 2


@pytest.mark.parametrize("extra", [
    {},
    {"thread_follow": True, "group_rules": {"oc_team": {"policy": "open", "thread_follow": False}}},
    {"thread_follow": "nonsense"},
])
def test_follow_off_for_the_chat_keeps_the_mention_requirement(extra):
    api = _FakeMessageApi(topic_senders=[("app", APP_ID)])
    adapter = _follow_adapter(api, extra=extra)

    assert _deliver(adapter) is False


def test_group_policy_still_gates_follow_ups():
    api = _FakeMessageApi(topic_senders=[("app", APP_ID)])
    adapter = _follow_adapter(api, group_policy="allowlist")  # empty allowlist: nobody may trigger

    assert _deliver(adapter) is False
    assert api.lists == []


def test_bot_senders_and_plain_group_messages_are_not_followed():
    api = _FakeMessageApi(topic_senders=[("app", APP_ID)])
    adapter = _follow_adapter(api)
    adapter._allow_bots = "all"
    adapter._bot_open_id = "ou_me"

    assert _deliver(adapter, sender=make_sender(sender_type="bot", open_id="ou_peer")) is False
    assert _deliver(adapter, thread_id=None) is False


# --- last_active mode: the bot triggered last answers -------------------------------------------


def _msg(sender_type, sender_id, *mentioned, message_id=None):
    mentions = [SimpleNamespace(key=f"@_user_{i}", id=oid, id_type="open_id", name=oid)
                for i, oid in enumerate(mentioned, 1)] or None
    return SimpleNamespace(message_id=message_id or f"om_h_{next(_n)}", mentions=mentions,
                           sender=SimpleNamespace(sender_type=sender_type, id=sender_id))


def _last_active_adapter(*history, extra=None):
    """History oldest first; this bot is APP_ID / ou_me, the peer bot is PEER_APP_ID / ou_peer."""
    api = _FakeMessageApi(topic_messages=list(history))
    adapter = _follow_adapter(api, extra=extra)
    adapter._bot_open_id, adapter._bot_user_id, adapter._bot_name = "ou_me", "", "Hermes"
    return adapter, api


@pytest.mark.parametrize("extra", [None, {"thread_follow": "last_active"}, {"thread_follow": "LAST_ACTIVE"}])
def test_last_active_follows_the_bot_that_replied_last(extra):
    me, _ = _last_active_adapter(_msg("user", "ou_neo", "ou_me"), _msg("app", APP_ID), _msg("user", "ou_neo"),
                                 extra=extra)
    assert _deliver(me) is True


def test_last_active_moves_to_a_bot_mentioned_mid_topic_after_it_replied():
    # This bot opened the topic; the peer was @mentioned and replied, so the topic is now the peer's.
    me, api = _last_active_adapter(
        _msg("user", "ou_neo", "ou_me"), _msg("app", APP_ID), _msg("user", "ou_neo", "ou_peer"), _msg("app", PEER_APP_ID),
    )
    assert _deliver(me) is False
    # Re-read every time: nothing is cached in this mode.
    assert _deliver(me) is False
    assert len(api.lists) == 2


def test_last_active_mentioned_bot_takes_over_before_it_has_replied():
    # Peer opened the topic; the user @mentions this bot and follows up before the reply lands.
    me, _ = _last_active_adapter(_msg("user", "ou_neo", "ou_peer"), _msg("app", PEER_APP_ID),
                                 _msg("user", "ou_neo", "ou_me"))
    assert _deliver(me) is True
    me, _ = _last_active_adapter(_msg("app", APP_ID), _msg("user", "ou_neo", "ou_peer"))
    assert _deliver(me) is False


def test_last_active_bot_mentioning_another_bot_hands_the_topic_over():
    # Peer's message @mentions this bot: the topic is this bot's.
    me, _ = _last_active_adapter(_msg("app", PEER_APP_ID), _msg("app", PEER_APP_ID, "ou_me"))
    assert _deliver(me) is True
    # This bot's own message @mentions the peer: no longer this bot's.
    me, _ = _last_active_adapter(_msg("app", APP_ID), _msg("app", APP_ID, "ou_peer"))
    assert _deliver(me) is False


def test_last_active_skips_plain_chatter_the_current_message_and_mention_all():
    current = _msg("user", "ou_neo", message_id="om_current")
    everyone = _msg("user", "ou_neo")
    everyone.mentions = [SimpleNamespace(key="@_all", id="all", id_type="", name="所有人")]
    me, _ = _last_active_adapter(_msg("app", APP_ID), _msg("user", "ou_ann"), everyone, current)
    message = SimpleNamespace(thread_id="omt_1", message_id="om_current")
    assert asyncio.run(me._topic_last_triggered_self(message.thread_id, message.message_id)) is True


def test_last_active_topic_without_any_trigger_or_unreadable_is_not_followed():
    me, _ = _last_active_adapter(_msg("user", "ou_neo"))
    assert _deliver(me) is False
    me, api = _last_active_adapter(_msg("app", APP_ID))
    api.list = lambda request: SimpleNamespace(success=lambda: False, code=99991672, msg="no scope")
    assert _deliver(me) is False
