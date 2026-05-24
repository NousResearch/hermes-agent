"""Behavior-contract tests for ``FeishuAdapter``.

These tests treat the adapter as a black box: feed events / call public
methods, assert observable outputs. The boundary is stable across SDK
upgrades.
"""

import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("lark_oapi.channel")

from .conftest import dispatch_inbound_event, _drain_adapter_tasks


def _build_text_event(*, chat_id: str, chat_type: str, sender_open_id: str,
                     text: str = "hi", message_id: str = "om_test") -> dict:
    return {
        "header": {"event_id": f"evt_{message_id}", "event_type": "im.message.receive_v1",
                   "create_time": "1714200000000", "token": "test_token", "app_id": "cli_test_app"},
        "event": {
            "sender": {"sender_id": {"open_id": sender_open_id, "user_id": sender_open_id.replace("ou_", "u_")},
                       "sender_type": "user"},
            "message": {
                "message_id": message_id, "chat_id": chat_id, "chat_type": chat_type,
                "message_type": "text",
                "content": '{"text":"' + text + '"}',
                "create_time": "1714200000000",
                "mentions": [{"key": "@_user_1",
                              "id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"},
                              "name": "HermesBot"}] if chat_type == "group" else [],
            }
        }
    }


# NOTE: FeishuAdapterSettings is a frozen dataclass, so we use
# dataclasses.replace(...) rather than __class__(**{**__dict__, ...}).
# The latter works for simple frozen dataclasses but replace is the
# canonical API and survives field additions.
#
# Bot identity is stored on FeishuAdapterSettings as a manual fallback, then
# SDK hydration can refresh the adapter fields after connect().


def _replace_settings(adapter, **overrides):
    """Replace adapter._settings; bot identity dynamic attrs survive untouched."""
    adapter._settings = dataclasses.replace(adapter._settings, **overrides)
    adapter._apply_settings(adapter._settings)
    # _apply_settings sets _default_group_policy = settings.default_group_policy
    # or settings.group_policy. The conftest forces group_policy/default to "open"
    # post-init via private attrs; preserve that here unless the test explicitly
    # set default_group_policy.
    if "default_group_policy" not in overrides:
        adapter._default_group_policy = "open"
        adapter._group_policy = "open"


def test_sdk_channel_requests_all_reaction_notifications(adapter_harness):
    channel = adapter_harness.adapter._build_sdk_channel(register_handlers=False)

    assert channel._config.inbound.reaction_notifications == "all"


def test_package_reexports_legacy_availability_constants():
    import gateway.platforms.feishu as feishu

    assert hasattr(feishu, "FEISHU_WEBHOOK_AVAILABLE")
    assert hasattr(feishu, "FEISHU_WEBSOCKET_AVAILABLE")


def test_package_preserves_legacy_normalize_message_import_surface():
    from gateway.platforms.feishu import normalize_feishu_message

    normalized = normalize_feishu_message(
        message_type="text",
        raw_content='{"text":"hello"}',
    )

    assert normalized.raw_type == "text"
    assert normalized.text_content == "hello"


def test_normalize_feishu_message_preserves_mention_rendering():
    from gateway.platforms.feishu import normalize_feishu_message

    mention = SimpleNamespace(
        key="@_user_1",
        id=SimpleNamespace(open_id="ou_alice", user_id="u_alice"),
        name="Alice",
    )
    normalized = normalize_feishu_message(
        message_type="text",
        raw_content='{"text":"@_user_1 hi @_all"}',
        mentions=[mention],
    )

    assert normalized.text_content == "@Alice hi @all"
    assert [m.name for m in normalized.mentions if not m.is_all] == ["Alice"]
    assert any(m.is_all for m in normalized.mentions)


def test_normalize_feishu_message_preserves_merge_forward_summary():
    from gateway.platforms.feishu import normalize_feishu_message

    normalized = normalize_feishu_message(
        message_type="merge_forward",
        raw_content=json.dumps(
            {
                "title": "Sprint recap",
                "messages": [
                    {"sender_name": "Alice", "text": "Please review PR-128"},
                    {
                        "sender_name": "Bob",
                        "message_type": "post",
                        "content": {
                            "en_us": {
                                "content": [[{"tag": "text", "text": "Ship it"}]],
                            }
                        },
                    },
                ],
            }
        ),
    )

    assert normalized.relation_kind == "merge_forward"
    assert normalized.text_content == "Sprint recap\n- Alice: Please review PR-128\n- Bob: Ship it"
    assert normalized.metadata["entry_count"] == 2


def test_normalize_feishu_message_preserves_share_chat_summary_and_metadata():
    from gateway.platforms.feishu import normalize_feishu_message

    normalized = normalize_feishu_message(
        message_type="share_chat",
        raw_content=json.dumps({"chat_id": "oc_chat_shared", "chat_name": "Backend Guild"}),
    )

    assert normalized.relation_kind == "share_chat"
    assert normalized.text_content == "Shared chat: Backend Guild\nChat ID: oc_chat_shared"
    assert normalized.metadata["chat_id"] == "oc_chat_shared"
    assert normalized.metadata["chat_name"] == "Backend Guild"


def test_normalize_feishu_message_preserves_interactive_card_text_and_actions():
    from gateway.platforms.feishu import normalize_feishu_message

    normalized = normalize_feishu_message(
        message_type="interactive",
        raw_content=json.dumps(
            {
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": "Build Failed"}},
                    "elements": [
                        {"tag": "div", "text": {"tag": "lark_md", "content": "Service: payments-api"}},
                        {"tag": "div", "text": {"tag": "plain_text", "content": "Branch: main"}},
                        {
                            "tag": "action",
                            "actions": [
                                {"tag": "button", "text": {"tag": "plain_text", "content": "View Logs"}},
                                {"tag": "button", "text": {"tag": "plain_text", "content": "Retry"}},
                            ],
                        },
                    ],
                }
            }
        ),
    )

    assert normalized.relation_kind == "interactive"
    assert normalized.text_content == (
        "Build Failed\n"
        "Service: payments-api\n"
        "Branch: main\n"
        "View Logs\n"
        "Retry\n"
        "Actions: View Logs, Retry"
    )


class TestGroupPolicyAdminBypass:
    """Global admins must pass through any group gate."""

    @pytest.fixture
    def harness(self, adapter_harness):
        # Configure: default_group_policy=disabled, but ou_admin1 in admins
        _replace_settings(
            adapter_harness.adapter,
            default_group_policy="disabled",
            admins=frozenset({"ou_admin1"}),
        )
        return adapter_harness

    def test_admin_passes_through_disabled_group(self, harness):
        event = _build_text_event(chat_id="oc_locked", chat_type="group",
                                  sender_open_id="ou_admin1", message_id="om_admin")
        asyncio.run(dispatch_inbound_event(harness, event))
        assert len(harness.captured_inbound) == 1, "Admin should bypass disabled-group gate"

    def test_non_admin_blocked_in_disabled_group(self, harness):
        event = _build_text_event(chat_id="oc_locked", chat_type="group",
                                  sender_open_id="ou_random", message_id="om_random")
        asyncio.run(dispatch_inbound_event(harness, event))
        assert len(harness.captured_inbound) == 0, "Non-admin must be blocked"


class TestGroupBlocklist:
    """A group with policy=blocklist should reject senders on the blocklist."""

    @pytest.fixture
    def harness(self, adapter_harness):
        from gateway.platforms.feishu import FeishuGroupRule
        _replace_settings(
            adapter_harness.adapter,
            group_rules={"oc_team": FeishuGroupRule(policy="blacklist", blacklist={"ou_banned"})},
        )
        return adapter_harness

    def test_blocked_sender_rejected(self, harness):
        event = _build_text_event(chat_id="oc_team", chat_type="group",
                                  sender_open_id="ou_banned", message_id="om_b")
        asyncio.run(dispatch_inbound_event(harness, event))
        assert len(harness.captured_inbound) == 0

    def test_allowed_sender_passes(self, harness):
        event = _build_text_event(chat_id="oc_team", chat_type="group",
                                  sender_open_id="ou_normal", message_id="om_n")
        asyncio.run(dispatch_inbound_event(harness, event))
        assert len(harness.captured_inbound) == 1


class TestGroupAdminOnly:
    """policy=admin_only — only admins from the global set may pass."""

    @pytest.fixture
    def harness(self, adapter_harness):
        from gateway.platforms.feishu import FeishuGroupRule
        _replace_settings(
            adapter_harness.adapter,
            admins=frozenset({"ou_admin1"}),
            group_rules={"oc_admin_only": FeishuGroupRule(policy="admin_only")},
        )
        return adapter_harness

    def test_admin_passes(self, harness):
        event = _build_text_event(chat_id="oc_admin_only", chat_type="group",
                                  sender_open_id="ou_admin1", message_id="om_aok")
        asyncio.run(dispatch_inbound_event(harness, event))
        assert len(harness.captured_inbound) == 1

    def test_non_admin_rejected(self, harness):
        event = _build_text_event(chat_id="oc_admin_only", chat_type="group",
                                  sender_open_id="ou_random", message_id="om_anr")
        asyncio.run(dispatch_inbound_event(harness, event))
        assert len(harness.captured_inbound) == 0


class TestMentionHandling:
    """Leading/trailing self-mentions are stripped, mid-text is preserved,
    other mentions get a hint prepended."""

    def test_leading_self_mention_stripped(self, adapter_harness):
        event = {
            "header": {"event_id": "evt_lead", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"}, "sender_type": "user"},
                "message": {
                    "message_id": "om_lead", "chat_id": "oc_team", "chat_type": "group",
                    "message_type": "text",
                    "content": '{"text":"@_user_1 hello"}',
                    "create_time": "1714200000000",
                    "mentions": [{"key": "@_user_1",
                                  "id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"}, "name": "HermesBot"}],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 1
        assert adapter_harness.captured_inbound[0].text == "hello"

    def test_only_self_mention_is_dropped(self, adapter_harness):
        event = {
            "header": {"event_id": "evt_only_mention", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"}, "sender_type": "user"},
                "message": {
                    "message_id": "om_only_mention", "chat_id": "oc_team", "chat_type": "group",
                    "message_type": "text",
                    "content": '{"text":"@_user_1"}',
                    "create_time": "1714200000000",
                    "mentions": [{"key": "@_user_1",
                                  "id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"}, "name": "HermesBot"}],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 0

    def test_mid_text_self_mention_preserved(self, adapter_harness):
        event = {
            "header": {"event_id": "evt_mid", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"}, "sender_type": "user"},
                "message": {
                    "message_id": "om_mid", "chat_id": "oc_team", "chat_type": "group",
                    "message_type": "text",
                    "content": '{"text":"hey @_user_1 do you know?"}',
                    "create_time": "1714200000000",
                    "mentions": [{"key": "@_user_1",
                                  "id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"}, "name": "HermesBot"}],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 1
        # Mid-text @bot is preserved (rendered through _normalize_feishu_text)
        text = adapter_harness.captured_inbound[0].text
        assert "@HermesBot" in text or "@_user_1" in text  # exact rendering depends on normalize impl
        assert text.startswith("hey "), f"Expected text to start with 'hey ', got {text!r}"
        assert text.endswith("do you know?"), f"Expected text to end with 'do you know?', got {text!r}"

    def test_peer_bot_mention_all_counts_for_mentions_mode(self, adapter_harness):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.types import TextContent

        adapter_harness.adapter._allow_bots = "mentions"
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter_harness.adapter.handle_message = _capture
        msg = InboundMessage(
            id="om_peer_bot_all",
            create_time=1,
            conversation=Conversation(chat_id="oc_team", chat_type="group"),
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
            mentions=[],
            mentioned_all=True,
            content=TextContent(raw={}, text="@_all hello"),
            raw={},
            content_text="@all hello",
            resources=[],
        )

        asyncio.run(adapter_harness.adapter._on_sdk_message(msg))

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_other_bot_mention_is_not_treated_as_self(self, adapter_harness):
        from lark_oapi.channel import Conversation, Identity, InboundMessage, Mention
        from lark_oapi.channel.types import TextContent

        adapter_harness.adapter._require_mention = False

        msg = InboundMessage(
            id="om_other_bot_mention",
            create_time=1,
            conversation=Conversation(chat_id="p2p_alice", chat_type="p2p"),
            sender=Identity(open_id="ou_alice", user_id="u_alice", is_bot=False),
            mentions=[
                Mention(
                    key="@_user_2",
                    open_id="ou_other_bot",
                    user_id="u_other_bot",
                    name="OtherBot",
                    is_bot=True,
                )
            ],
            mentioned_all=False,
            content=TextContent(raw={}, text="@_user_2 hello"),
            raw={},
            content_text="@OtherBot hello",
            resources=[],
        )

        await adapter_harness.adapter._on_sdk_message(msg)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1
        text = adapter_harness.captured_inbound[0].text
        assert "OtherBot" in text
        assert text.endswith("@OtherBot hello")

    def test_other_mention_injects_hint(self, adapter_harness):
        event = {
            "header": {"event_id": "evt_other", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"}, "sender_type": "user"},
                "message": {
                    "message_id": "om_other", "chat_id": "oc_team", "chat_type": "group",
                    "message_type": "text",
                    "content": '{"text":"@_user_1 ping @_user_2 to confirm"}',
                    "create_time": "1714200000000",
                    "mentions": [
                        {"key": "@_user_1", "id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"}, "name": "HermesBot"},
                        {"key": "@_user_2", "id": {"open_id": "ou_bob", "user_id": "u_bob"}, "name": "Bob"},
                    ],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 1
        text = adapter_harness.captured_inbound[0].text
        # Hint should mention Bob (the non-self mention)
        assert "Bob" in text, f"Expected mention hint to include Bob, got: {text!r}"


class TestReactionSynthesis:
    """SDK ``ReactionEvent`` becomes a synthetic TEXT MessageEvent when the
    target message was authored by this bot. A bot's own reaction is filtered.

    Tests drive the SDK-side handler ``_on_sdk_reaction`` directly.
    """

    def _build_sdk_reaction(self, *, reactor_open_id: str, emoji: str = "THUMBSUP",
                            action: str = "added", chat_id: str | None = "p2p_alice",
                            raw: dict | None = None) -> "ReactionEvent":
        from lark_oapi.channel.types import ReactionEvent, EventOperator
        return ReactionEvent(
            message_id="om_botmsg",
            operator=EventOperator(
                open_id=reactor_open_id,
                user_id=reactor_open_id.replace("ou_", "u_"),
            ),
            emoji_type=emoji,
            action=action,
            chat_id=chat_id,
            chat_type="p2p",
            raw=raw or {},
        )

    @pytest.mark.asyncio
    async def test_user_reaction_routes_as_text_event(self, adapter_harness):
        """User reaction on a bot message must route as synthetic TEXT."""
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice")
        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()
        assert len(adapter_harness.captured_inbound) == 1
        event = adapter_harness.captured_inbound[0]
        text = event.text.lower()
        assert event.source.platform.value == "feishu"
        assert any(marker in text for marker in (
            "thumbsup", "added", "reaction", ":+1:")), (
            f"Reaction text must indicate the reaction; got {text!r}"
        )

    @pytest.mark.asyncio
    async def test_user_reaction_removal_routes_as_text_event(self, adapter_harness):
        """User reaction removal must preserve the removed action in synthetic TEXT."""
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice", action="removed")

        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1
        assert adapter_harness.captured_inbound[0].text == "reaction:removed:THUMBSUP"

    def test_reaction_without_sdk_chat_id_is_dropped(self, adapter_harness):
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice", chat_id=None)
        evt.chat_type = None

        asyncio.run(adapter_harness.adapter._on_sdk_reaction(evt))

        assert adapter_harness.captured_inbound == []

    def test_bot_origin_reaction_filtered(self, adapter_harness):
        evt = self._build_sdk_reaction(reactor_open_id="ou_hermes_bot")
        asyncio.run(adapter_harness.adapter._on_sdk_reaction(evt))
        # Bot's own reaction must never produce a synthetic event.
        assert len(adapter_harness.captured_inbound) == 0

    def test_peer_bot_reaction_dropped_when_bots_disabled(self, adapter_harness):
        adapter_harness.adapter._allow_bots = "none"
        evt = self._build_sdk_reaction(
            reactor_open_id="ou_peer_bot",
            raw={
                "event": {
                    "user_id": {"open_id": "ou_peer_bot", "user_id": "u_peer_bot"},
                    "user_type": "app",
                }
            },
        )

        asyncio.run(adapter_harness.adapter._on_sdk_reaction(evt))

        assert adapter_harness.captured_inbound == []

    @pytest.mark.asyncio
    async def test_peer_bot_reaction_routes_when_bots_allowed(self, adapter_harness):
        adapter_harness.adapter._allow_bots = "all"
        evt = self._build_sdk_reaction(
            reactor_open_id="ou_peer_bot",
            raw={
                "event": {
                    "user_id": {"open_id": "ou_peer_bot", "user_id": "u_peer_bot"},
                    "user_type": "app",
                }
            },
        )

        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1
        assert adapter_harness.captured_inbound[0].source.is_bot is True

    @pytest.mark.asyncio
    async def test_reaction_targeting_bot_message_routes_after_sent_cache_clear(
        self, adapter_harness
    ):
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice")
        adapter_harness.adapter._channel._sent_messages = {}

        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1

    @pytest.mark.asyncio
    async def test_reaction_targeting_bot_message_from_openapi_dict_routes(
        self, adapter_harness
    ):
        async def fetch_bot_message(message_id):
            return {
                "data": {
                    "items": [
                        {
                            "message_id": message_id,
                            "sender": {
                                "id": "ou_hermes_bot",
                                "id_type": "open_id",
                                "sender_type": "app",
                            },
                        },
                    ],
                },
            }

        adapter_harness.adapter._channel.fetch_message = fetch_bot_message
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice")

        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1

    @pytest.mark.asyncio
    async def test_reaction_targeting_bot_message_from_openapi_app_sender_routes(
        self, adapter_harness
    ):
        async def fetch_bot_message(message_id):
            return {
                "data": {
                    "items": [
                        {
                            "message_id": message_id,
                            "sender": {
                                "id": "cli_test_app",
                                "id_type": "app_id",
                                "sender_type": "app",
                            },
                        },
                    ],
                },
            }

        adapter_harness.adapter._channel.fetch_message = fetch_bot_message
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice")

        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1

    @pytest.mark.asyncio
    async def test_reaction_without_sdk_chat_id_uses_fetched_target_chat(
        self, adapter_harness
    ):
        async def fetch_bot_message(message_id):
            return {
                "data": {
                    "items": [
                        {
                            "message_id": message_id,
                            "chat_id": "p2p_alice",
                            "sender": {
                                "id": "cli_test_app",
                                "id_type": "app_id",
                                "sender_type": "app",
                            },
                        },
                    ],
                },
            }

        adapter_harness.adapter._channel.fetch_message = fetch_bot_message
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice", chat_id=None)
        evt.chat_type = None

        await adapter_harness.adapter._on_sdk_reaction(evt)
        await _drain_adapter_tasks()

        assert len(adapter_harness.captured_inbound) == 1
        assert adapter_harness.captured_inbound[0].source.chat_id == "p2p_alice"

    def test_reaction_targeting_human_message_is_dropped(self, adapter_harness):
        async def fetch_human_message(message_id):
            return SimpleNamespace(
                message_id=message_id,
                sender=SimpleNamespace(open_id="ou_human_author"),
            )

        adapter_harness.adapter._channel.fetch_message = fetch_human_message
        evt = self._build_sdk_reaction(reactor_open_id="ou_alice")

        asyncio.run(adapter_harness.adapter._on_sdk_reaction(evt))

        assert adapter_harness.captured_inbound == []


class TestPeerBotSource:
    """Peer-bot messages admitted by the adapter must remain marked as bots."""

    def test_peer_bot_message_preserves_source_is_bot(self, adapter_harness):
        event = _build_text_event(
            chat_id="oc_team",
            chat_type="group",
            sender_open_id="ou_peer_bot",
            text="@_user_1 ping",
            message_id="om_peer_bot",
        )
        event["event"]["sender"]["sender_type"] = "app"

        asyncio.run(dispatch_inbound_event(adapter_harness, event))

        assert len(adapter_harness.captured_inbound) == 1
        assert adapter_harness.captured_inbound[0].source.is_bot is True


class TestBotLifecycle:
    """Bot lifecycle SDK events should invalidate per-chat cached state."""

    def test_bot_added_invalidates_chat_info_cache(self, adapter_harness):
        adapter = adapter_harness.adapter
        adapter._chat_info_cache["oc_new"] = {"name": "stale"}
        evt = SimpleNamespace(
            chat_id="oc_new",
            operator=SimpleNamespace(open_id="ou_operator"),
        )

        asyncio.run(adapter._on_sdk_bot_added(evt))

        assert "oc_new" not in adapter._chat_info_cache

    def test_bot_leave_invalidates_chat_info_cache(self, adapter_harness):
        adapter = adapter_harness.adapter
        adapter._chat_info_cache["oc_old"] = {"name": "stale"}
        evt = SimpleNamespace(
            chat_id="oc_old",
            operator=SimpleNamespace(open_id="ou_operator"),
        )

        asyncio.run(adapter._on_sdk_bot_leave(evt))

        assert "oc_old" not in adapter._chat_info_cache

    def test_first_mention_in_new_chat_routes_normally(self, adapter_harness):
        event = _build_text_event(
            chat_id="oc_new_chat",
            chat_type="group",
            sender_open_id="ou_alice",
            text="@_user_1 first ping",
            message_id="om_first_mention",
        )

        asyncio.run(dispatch_inbound_event(adapter_harness, event))

        assert len(adapter_harness.captured_inbound) == 1
        captured = adapter_harness.captured_inbound[0]
        assert captured.source.chat_id == "oc_new_chat"
        assert captured.text == "first ping"


class TestConnectFailureCleanup:
    """Startup failure paths must release the scoped Feishu app lock."""

    @pytest.mark.asyncio
    async def test_websocket_start_supports_connect_wait_ready_sdk(
        self, adapter_harness
    ):
        calls = []

        class ConnectWaitReadyOnlyChannel:
            async def connect(self):
                calls.append("connect")

            async def wait_ready(self, *, timeout=None):
                calls.append(("wait_ready", timeout))

            async def disconnect(self):
                calls.append("disconnect")

        adapter_harness.adapter._channel = ConnectWaitReadyOnlyChannel()

        await adapter_harness.adapter._connect_websocket_once()

        assert calls == ["connect", ("wait_ready", 30.0)]

    @pytest.mark.asyncio
    async def test_websocket_start_retries_transient_failure(
        self, monkeypatch, tmp_path,
    ):
        import gateway.platforms.feishu.adapter as adapter_module
        import lark_oapi.channel as sdk_channel
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        attempts = []

        def fake_acquire(scope, identity, metadata=None):
            return True, {}

        def fake_release(scope, identity):
            return None

        class FlakyChannel:
            def __init__(self, *args, **kwargs):
                self._ws_client = None

            def on(self, *args, **kwargs):
                return None

            def start(self):
                raise AssertionError("adapter should use SDK connect_until_ready")

            async def connect_until_ready(self, *, timeout=None):
                attempts.append(("connect_until_ready", timeout))
                if len(attempts) == 1:
                    raise RuntimeError("temporary websocket failure")
                self._ws_client = SimpleNamespace(_conn=object())

            async def disconnect(self):
                return None

        monkeypatch.setattr(adapter_module, "acquire_scoped_lock", fake_acquire)
        monkeypatch.setattr(adapter_module, "release_scoped_lock", fake_release)
        monkeypatch.setattr(sdk_channel, "FeishuChannel", FlakyChannel)

        adapter = FeishuAdapter(PlatformConfig(
            enabled=True,
            extra={
                "app_id": "cli_retry_test",
                "app_secret": "secret",
                "connection_mode": "websocket",
            },
        ))
        adapter._dedup_state_path = tmp_path / "dedup.json"

        try:
            ok = await adapter.connect()
            assert ok is True
            assert attempts == [
                ("connect_until_ready", 30.0),
                ("connect_until_ready", 30.0),
            ]
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_websocket_start_isolates_sdk_loop_captured_from_running_loop(
        self, monkeypatch, tmp_path,
    ):
        import gateway.platforms.feishu.adapter as adapter_module
        import lark_oapi.channel as sdk_channel
        import lark_oapi.ws.client as ws_client
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        running_loop = asyncio.get_running_loop()
        monkeypatch.setattr(ws_client, "loop", running_loop)

        def fake_acquire(scope, identity, metadata=None):
            return True, {}

        def fake_release(scope, identity):
            return None

        class LoopCheckingChannel:
            def __init__(self, *args, **kwargs):
                self._ws_client = None

            def on(self, *args, **kwargs):
                return None

            async def connect_until_ready(self, *, timeout=None):
                if ws_client.loop is running_loop or ws_client.loop.is_running():
                    raise RuntimeError("SDK websocket loop was not isolated")
                self._ws_client = SimpleNamespace(_conn=object())

            async def disconnect(self):
                return None

        monkeypatch.setattr(adapter_module, "acquire_scoped_lock", fake_acquire)
        monkeypatch.setattr(adapter_module, "release_scoped_lock", fake_release)
        monkeypatch.setattr(adapter_module, "_FEISHU_CONNECT_ATTEMPTS", 1)
        monkeypatch.setattr(sdk_channel, "FeishuChannel", LoopCheckingChannel)

        adapter = FeishuAdapter(PlatformConfig(
            enabled=True,
            extra={
                "app_id": "cli_loop_test",
                "app_secret": "secret",
                "connection_mode": "websocket",
            },
        ))
        adapter._dedup_state_path = tmp_path / "dedup.json"

        try:
            ok = await adapter.connect()
            assert ok is True
        finally:
            replacement_loop = ws_client.loop
            await adapter.disconnect()
            if replacement_loop is not running_loop and not replacement_loop.is_closed():
                replacement_loop.close()

    @pytest.mark.asyncio
    async def test_websocket_start_failure_releases_app_lock(
        self, monkeypatch, tmp_path,
    ):
        import gateway.platforms.feishu.adapter as adapter_module
        import lark_oapi.channel as sdk_channel
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        released = []

        def fake_acquire(scope, identity, metadata=None):
            return True, {}

        def fake_release(scope, identity):
            released.append((scope, identity))

        class FailingChannel:
            def __init__(self, *args, **kwargs):
                self._ws_client = None

            def on(self, *args, **kwargs):
                return None

            def start(self):
                raise RuntimeError("sdk start failed")

        monkeypatch.setattr(adapter_module, "acquire_scoped_lock", fake_acquire)
        monkeypatch.setattr(adapter_module, "release_scoped_lock", fake_release)
        monkeypatch.setattr(sdk_channel, "FeishuChannel", FailingChannel)

        adapter = FeishuAdapter(PlatformConfig(
            enabled=True,
            extra={
                "app_id": "cli_lock_test",
                "app_secret": "secret",
                "connection_mode": "websocket",
            },
        ))
        adapter._dedup_state_path = tmp_path / "dedup.json"

        ok = await adapter.connect()

        assert ok is False
        assert released == [
            (adapter_module._FEISHU_APP_LOCK_SCOPE, "cli_lock_test"),
        ]
        assert adapter._app_lock_identity is None
        assert adapter._channel is None
        assert adapter._dedup_store is None

    @pytest.mark.asyncio
    async def test_webhook_start_failure_disconnects_started_sdk_channel(
        self, monkeypatch, tmp_path,
    ):
        import gateway.platforms.feishu.adapter as adapter_module
        import gateway.platforms.feishu.webhook_guard as webhook_guard
        import lark_oapi.channel as sdk_channel
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        order = []

        def fake_acquire(scope, identity, metadata=None):
            return True, {}

        def fake_release(scope, identity):
            order.append("lock_release")

        class StartedWebhookChannel:
            def __init__(self, *args, **kwargs):
                pass

            def on(self, *args, **kwargs):
                return None

            async def connect(self):
                order.append("channel_connect")

            async def disconnect(self):
                order.append("channel_disconnect")

            async def handle_webhook_request(self, request):
                return None

        async def fail_start_webhook_server(**kwargs):
            order.append("webhook_bind_failed")
            raise OSError("address already in use")

        monkeypatch.setattr(adapter_module, "acquire_scoped_lock", fake_acquire)
        monkeypatch.setattr(adapter_module, "release_scoped_lock", fake_release)
        monkeypatch.setattr(adapter_module, "_FEISHU_CONNECT_ATTEMPTS", 1)
        monkeypatch.setattr(sdk_channel, "FeishuChannel", StartedWebhookChannel)
        monkeypatch.setattr(webhook_guard, "WEBHOOK_AVAILABLE", True)
        monkeypatch.setattr(webhook_guard, "start_webhook_server", fail_start_webhook_server)

        adapter = FeishuAdapter(PlatformConfig(
            enabled=True,
            extra={
                "app_id": "cli_webhook_cleanup",
                "app_secret": "secret",
                "connection_mode": "webhook",
                "webhook_port": 8765,
            },
        ))
        adapter._dedup_state_path = tmp_path / "dedup.json"

        ok = await adapter.connect()

        assert ok is False
        assert order == [
            "channel_connect",
            "webhook_bind_failed",
            "channel_disconnect",
            "lock_release",
        ]
        assert adapter._channel is None
        assert adapter._app_lock_identity is None


class TestSdkDisconnectCleanup:
    """Disconnect delegates websocket lifecycle cleanup to the SDK channel."""

    @pytest.mark.asyncio
    async def test_disconnect_releases_app_lock_after_channel_shutdown(
        self, adapter_harness, monkeypatch
    ):
        import gateway.platforms.feishu.adapter as adapter_module

        adapter = adapter_harness.adapter
        order = []

        class FakeChannel:
            async def disconnect(self):
                order.append("channel_disconnect")

        class FakeDedupStore:
            def flush(self):
                order.append("dedup_flush")

        def fake_release(scope, identity):
            order.append("lock_release")

        adapter._app_lock_identity = "cli_lock_test"
        adapter._channel = FakeChannel()
        adapter._dedup_store = FakeDedupStore()
        monkeypatch.setattr(adapter_module, "release_scoped_lock", fake_release)

        await adapter.disconnect()

        assert order == [
            "channel_disconnect",
            "dedup_flush",
            "lock_release",
        ]
        assert adapter._app_lock_identity is None


class TestMediaBatching:
    """Media batching boundary contract."""

    @pytest.fixture
    def batch_harness(self, adapter_harness):
        # Set short batch delay for fast test; >0 to enable batching
        import dataclasses
        adapter_harness.adapter._settings = dataclasses.replace(
            adapter_harness.adapter._settings,
            media_batch_delay_seconds=0.05,
        )
        adapter_harness.adapter._apply_settings(adapter_harness.adapter._settings)
        # Re-poke bot identity (apply_settings clobbers it)
        adapter_harness.adapter._bot_open_id = "ou_hermes_bot"
        adapter_harness.adapter._bot_user_id = "u_hermes_bot"
        adapter_harness.adapter._bot_name = "HermesBot"
        adapter_harness.adapter._group_policy = "open"
        adapter_harness.adapter._default_group_policy = "open"
        adapter_harness.adapter._require_mention = False
        return adapter_harness

    def _build_image_event(self, *, message_id: str, chat_id: str = "p2p_alice",
                           parent_id: str = None) -> dict:
        # Non-p2p chat_ids are treated as group chats, so set chat_type
        # accordingly so the group-policy gate (with policy="open") sees
        # them rather than the p2p path.
        chat_type = "p2p" if chat_id.startswith("p2p") else "group"
        msg = {
            "message_id": message_id, "chat_id": chat_id, "chat_type": chat_type,
            "message_type": "image",
            "content": '{"image_key":"img_' + message_id + '"}',
            "create_time": "1714200000000", "mentions": [],
        }
        if parent_id:
            msg["parent_id"] = parent_id
        return {
            "header": {"event_id": f"evt_{message_id}", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_alice", "user_id": "u_alice"}, "sender_type": "user"},
                "message": msg,
            }
        }

    def test_consecutive_images_same_chat_merge(self, batch_harness):
        async def _run():
            for i in range(3):
                await dispatch_inbound_event(
                    batch_harness,
                    self._build_image_event(message_id=f"om_img_{i}"),
                    drain=False,
                )
            await asyncio.sleep(0.2)  # Wait for flush
        asyncio.run(_run())
        events = batch_harness.captured_inbound
        assert len(events) == 1
        assert all(mtype.startswith("image/") for mtype in events[0].media_types)
        assert [Path(url).name for url in events[0].media_urls] == [
            "img_om_img_0.jpg",
            "img_om_img_1.jpg",
            "img_om_img_2.jpg",
        ]

    def test_image_download_failure_keeps_dispatching_without_media(self, batch_harness):
        async def fail_download(*args, **kwargs):
            raise RuntimeError("download failed")

        batch_harness.adapter._channel.download_resource_to_file = fail_download

        async def _run():
            await dispatch_inbound_event(
                batch_harness,
                self._build_image_event(message_id="om_img_fail"),
                drain=False,
            )
            await asyncio.sleep(0.2)
        asyncio.run(_run())

        assert len(batch_harness.captured_inbound) == 1
        event = batch_harness.captured_inbound[0]
        assert event.media_urls == []
        assert event.media_types == []

    def test_images_with_different_reply_to_do_not_merge(self, batch_harness):
        async def _run():
            await dispatch_inbound_event(batch_harness,
                self._build_image_event(message_id="om_a", parent_id="om_parent_1"))
            await dispatch_inbound_event(batch_harness,
                self._build_image_event(message_id="om_b", parent_id="om_parent_2"))
            await asyncio.sleep(0.2)
        asyncio.run(_run())
        assert len(batch_harness.captured_inbound) == 2, (
            "Different reply_to targets must not merge"
        )

    def test_images_across_chats_do_not_merge(self, batch_harness):
        async def _run():
            await dispatch_inbound_event(batch_harness,
                self._build_image_event(message_id="om_x", chat_id="oc_chat1"))
            await dispatch_inbound_event(batch_harness,
                self._build_image_event(message_id="om_y", chat_id="oc_chat2"))
            await asyncio.sleep(0.2)
        asyncio.run(_run())
        # SDK owns mention/policy gating now. With group_policy="open"
        # (set by the harness) both images flow through to
        # ``_on_sdk_message`` independently, so the cross-chat batching
        # boundary is exercised directly: each chat owns its own batch
        # key, so two images across two chats produce two events.
        assert len(batch_harness.captured_inbound) == 2, (
            f"Different chat_ids must not merge into a single event; "
            f"got {len(batch_harness.captured_inbound)}"
        )
        chats = {e.source.chat_id for e in batch_harness.captured_inbound}
        assert chats == {"oc_chat1", "oc_chat2"}


class TestSelfSentFilter:
    """Events whose sender is the bot itself must be dropped to prevent
    feedback loops."""

    def test_app_sender_with_bot_open_id_is_filtered(self, adapter_harness):
        event = {
            "header": {"event_id": "evt_self", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"},
                           "sender_type": "app"},
                "message": {
                    "message_id": "om_self", "chat_id": "p2p_alice", "chat_type": "p2p",
                    "message_type": "text", "content": '{"text":"echo"}',
                    "create_time": "1714200000000", "mentions": [],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 0, (
            "Self-sent (sender=app+bot_open_id) must be filtered"
        )

    def test_user_sender_with_same_open_id_is_not_filtered(self, adapter_harness):
        # Edge case: sender_type=user but open_id collides with bot.
        # The current judgment requires sender_type=='app' too, so this should pass through.
        event = {
            "header": {"event_id": "evt_user_collide", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_hermes_bot", "user_id": "u_hermes_bot"},
                           "sender_type": "user"},
                "message": {
                    "message_id": "om_user_collide", "chat_id": "p2p_alice", "chat_type": "p2p",
                    "message_type": "text", "content": '{"text":"hello"}',
                    "create_time": "1714200000000", "mentions": [],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 1, (
            "sender_type=user must NOT be filtered even with bot open_id"
        )

    def test_other_app_sender_is_not_filtered(self, adapter_harness):
        # Another bot in the same chat should still be heard.
        event = {
            "header": {"event_id": "evt_other_app", "event_type": "im.message.receive_v1",
                       "create_time": "1714200000000", "token": "t", "app_id": "cli_test_app"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_other_bot", "user_id": "u_other_bot"},
                           "sender_type": "app"},
                "message": {
                    "message_id": "om_other_app", "chat_id": "p2p_alice", "chat_type": "p2p",
                    "message_type": "text", "content": '{"text":"hi from another bot"}',
                    "create_time": "1714200000000", "mentions": [],
                }
            }
        }
        asyncio.run(dispatch_inbound_event(adapter_harness, event))
        assert len(adapter_harness.captured_inbound) == 1


class TestSdkRejectMetric:
    """``_on_sdk_reject`` must surface SDK reject reasons through Hermes'
    metrics hook (``_record_inbound_drop``) verbatim, so downstream
    dashboards keep working when the SDK renames a reason string.
    """

    def test_reject_with_reason_emits_metric(self, adapter_harness):
        adapter = adapter_harness.adapter
        captured = []

        def _recorder(*, reason, chat_id, sender_id):
            captured.append({"reason": reason, "chat_id": chat_id, "sender_id": sender_id})

        # _on_sdk_reject is best-effort: emits to ``_record_inbound_drop`` if
        # the attribute exists. We attach a lambda recorder for assertion.
        adapter._record_inbound_drop = _recorder

        from types import SimpleNamespace
        evt = SimpleNamespace(
            reason="policy_group_not_in_allowlist",
            message_id="om_reject_1",
            chat_id="oc_locked",
            sender_id="ou_random",
        )
        asyncio.run(adapter._on_sdk_reject(evt))

        assert len(captured) == 1, "Reject event must trigger one metric call"
        assert captured[0]["reason"] == "policy_group_not_in_allowlist"
        assert captured[0]["chat_id"] == "oc_locked"
        assert captured[0]["sender_id"] == "ou_random"

    def test_reject_without_recorder_does_not_raise(self, adapter_harness):
        # Recorder absent (the default); handler must tolerate it silently.
        adapter = adapter_harness.adapter
        # Ensure no recorder attribute lingers from other tests.
        if hasattr(adapter, "_record_inbound_drop"):
            delattr(adapter, "_record_inbound_drop")

        from types import SimpleNamespace
        evt = SimpleNamespace(
            reason="duplicate", message_id="om_x", chat_id="oc_x", sender_id="ou_x",
        )
        # Should not raise.
        asyncio.run(adapter._on_sdk_reject(evt))


class TestSdkCommentDispatch:
    """``_on_sdk_comment`` must convert the SDK ``CommentEvent`` into the
    legacy drive-comment dict shape and dispatch to
    ``handle_drive_comment_event``.
    """

    def test_comment_event_dispatches_to_drive_handler(self, adapter_harness, monkeypatch):
        adapter = adapter_harness.adapter
        called = {}

        async def _fake_handler(client, data, *, self_open_id):
            called["client"] = client
            called["data"] = data
            called["self_open_id"] = self_open_id

        # _on_sdk_comment lazily imports ``handle_drive_comment_event`` from
        # ``gateway.platforms.feishu.comments``; patch the source module so
        # the lazy import picks up our fake.
        import gateway.platforms.feishu.comments as comments_mod
        monkeypatch.setattr(
            comments_mod, "handle_drive_comment_event", _fake_handler, raising=True,
        )

        from types import SimpleNamespace
        # CommentEvent.raw is the SDK-normalized inner event dict (see
        # lark_oapi/channel/normalize/comment.py); _sdk_comment_to_legacy_dict
        # wraps it back into a legacy SimpleNamespace envelope.
        evt = SimpleNamespace(
            file_token="doxc_test_token",
            file_type="docx",
            comment_id="cm_1",
            reply_id=None,
            content="hello @bot",
            timestamp=1714200000,
            event_id="evt_comment_1",
            tenant_key="tk_test",
            raw={
                "file_token": "doxc_test_token",
                "comment_id": "cm_1",
                "user_id": {"open_id": "ou_alice"},
            },
        )

        asyncio.run(adapter._on_sdk_comment(evt))

        # The legacy envelope is a SimpleNamespace with ``.event`` (inner
        # dict-as-namespace), ``.header``, and ``.ts``; it always carries
        # file_token + comment_id for downstream parse_drive_comment_event.
        assert called, "Drive-comment handler must be called"
        envelope = called["data"]
        inner = envelope.event
        assert getattr(inner, "file_token", None) == "doxc_test_token"
        assert getattr(inner, "comment_id", None) == "cm_1"
        assert called["self_open_id"] == "ou_hermes_bot"

    def test_comment_action_dedup_key_includes_reply_id(self, adapter_harness):
        """Multiple replies in one comment thread must not share one action key."""
        adapter = adapter_harness.adapter
        channel = adapter._build_sdk_channel(transport_kind="webhook", register_handlers=True)
        captured = []

        async def fake_action_safety(*, event_id, queue_scope, handler):
            captured.append((event_id, queue_scope))

        channel._through_action_safety = fake_action_safety

        def payload(reply_id):
            return SimpleNamespace(
                event={
                    "comment_id": "cm_1",
                    "reply_id": reply_id,
                    "is_mentioned": True,
                    "notice_meta": {
                        "file_token": "doxc_test_token",
                        "file_type": "docx",
                        "from_user_id": {"open_id": "ou_alice"},
                        "to_user_id": {"open_id": "ou_hermes_bot"},
                        "notice_type": "comment_add",
                    },
                },
                header=SimpleNamespace(create_time="1714200000"),
            )

        asyncio.run(channel._handle_comment_event(payload("rp_1")))
        asyncio.run(channel._handle_comment_event(payload("rp_2")))

        event_ids = [event_id for event_id, _ in captured]
        assert event_ids == [
            "comment:doxc_test_token:cm_1:rp_1",
            "comment:doxc_test_token:cm_1:rp_2",
        ]


class TestRemoteDocumentDownloadFinalization:
    """Content-Type and body must be read while the ``httpx.AsyncClient``
    context is still active so pooled connections fully release on exit.
    Reading them after ``__aexit__`` works today only because httpx
    eagerly buffers non-streaming responses; a future move to
    ``.stream()`` would silently read-after-close."""

    @pytest.mark.asyncio
    async def test_response_read_inside_client_context(self, monkeypatch, tmp_path):
        from gateway.platforms.feishu.adapter import FeishuAdapter

        observations: dict = {}

        # _FakeClient drives _open=True only between __aenter__ and __aexit__.
        # _FakeResp.headers and .content are properties that record the
        # client's _open state at access time, so the test fails if the
        # production code reads them after the `async with` block exits.
        class _FakeResp:
            def __init__(self, client):
                self._client = client
                self._headers = {"Content-Type": "application/pdf"}
                self._content = b"fake-pdf-bytes"

            @property
            def headers(self):
                observations["context_open_during_headers"] = self._client._open
                return self._headers

            @property
            def content(self):
                observations["context_open_during_content"] = self._client._open
                return self._content

            def raise_for_status(self):
                pass

        class _FakeClient:
            def __init__(self, *a, **kw):
                self._open = False

            async def __aenter__(self):
                self._open = True
                return self

            async def __aexit__(self, *exc):
                self._open = False
                return False

            async def get(self, url, headers=None):
                observations["context_open_during_get"] = self._open
                return _FakeResp(self)

        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", _FakeClient)
        monkeypatch.setattr(
            "gateway.platforms.feishu.adapter.cache_document_from_bytes",
            lambda body, name: str(tmp_path / name),
        )
        # Bypass SSRF guard for the synthetic file URL.
        import tools.url_safety as _url_safety
        monkeypatch.setattr(_url_safety, "is_safe_url", lambda url: True)

        # Minimal adapter shell — bypass __init__ to avoid loading settings.
        adapter = FeishuAdapter.__new__(FeishuAdapter)
        cached_path, filename = await adapter._download_remote_document(
            "https://example.com/doc.pdf",
            default_ext=".pdf",
            preferred_name="doc.pdf",
        )
        assert cached_path.endswith("doc.pdf")
        assert filename == "doc.pdf"
        # The bug we're guarding against: headers/content were read AFTER
        # `async with` exited, leaving pooled connections half-released.
        assert observations["context_open_during_get"] is True
        assert observations["context_open_during_headers"] is True, (
            "response.headers was read AFTER the httpx.AsyncClient context "
            "exited; snapshot Content-Type inside the `async with` block."
        )
        assert observations["context_open_during_content"] is True, (
            "response.content was read AFTER the httpx.AsyncClient context "
            "exited; snapshot the body inside the `async with` block."
        )


class TestReceiveIdTypeRouting:
    """Sending to a user open_id (``ou_`` prefix) must route as a DM
    (``receive_id_type=open_id``); sending to a chat_id (``oc_`` prefix)
    must route as a chat.

    SDK ``lark_oapi.channel.outbound.routing.infer_receive_id_type``
    maps ``ou_`` → ``open_id`` and ``oc_`` → ``chat_id``, matching the
    public Feishu / Lark Open API conventions. ``FeishuChannel.send``
    invokes it for any caller that doesn't supply ``receive_id_type``
    explicitly via opts, so the adapter doesn't need to detect the
    prefix itself.
    """

    @pytest.mark.asyncio
    async def test_send_to_user_open_id_uses_open_id_receive_type(
        self, adapter_harness
    ):
        result = await adapter_harness.adapter.send("ou_user_dm_123", "hello")
        assert result.success, f"send to user open_id failed: {result.error!r}"
        captured = adapter_harness.captured_sends[-1]
        # SDK's FeishuChannel.send infers receive_id_type from the prefix
        # via infer_receive_id_type. The harness mock mirrors that and
        # surfaces the resolved value under ``extra``.
        assert captured.extra.get("receive_id_type") == "open_id", (
            f"expected open_id for ou_ prefix, got {captured.extra!r}"
        )
        assert captured.body.get("receive_id") == "ou_user_dm_123"

    @pytest.mark.asyncio
    async def test_send_to_chat_id_uses_chat_id_receive_type(
        self, adapter_harness
    ):
        result = await adapter_harness.adapter.send(
            "oc_chat_group_456", "hello group"
        )
        assert result.success, f"send to chat_id failed: {result.error!r}"
        captured = adapter_harness.captured_sends[-1]
        assert captured.extra.get("receive_id_type") == "chat_id", (
            f"expected chat_id for oc_ prefix, got {captured.extra!r}"
        )
        assert captured.body.get("receive_id") == "oc_chat_group_456"


class TestThreadIdFallbackChain:
    """When the SDK ``InboundMessage.conversation.thread_id`` is absent
    but the message carries ``root_id`` (top-of-thread reply) or
    ``upper_message_id``, ``MessageEvent.source.thread_id`` must surface
    the in-thread anchor so outbound replies stay in-thread.

    The SDK does not expose ``root_id`` / ``upper_message_id`` as
    typed fields on ``InboundMessage`` — they live in ``msg.raw`` (the
    original event payload). The fallback chain in ``events_mapping``
    reads ``raw["root_id"]`` / ``raw["upper_message_id"]`` to recover
    the thread anchor.
    """

    @pytest.mark.asyncio
    async def test_root_id_used_when_thread_id_absent(self, adapter_harness):
        # Build an event WITHOUT thread_id but WITH root_id (the wire
        # shape Feishu sends for top-of-thread replies in topic chats).
        event = _build_text_event(
            chat_id="oc_topic_chat",
            chat_type="group",
            sender_open_id="ou_user_topic",
            text="hello",
            message_id="om_topic_msg_1",
        )
        # Inject thread linkage into the message dict.
        event["event"]["message"]["root_id"] = "om_root_abc"
        # No thread_id key — exercising the fallback path.

        await dispatch_inbound_event(adapter_harness, event)
        assert len(adapter_harness.captured_inbound) == 1
        captured_event = adapter_harness.captured_inbound[0]
        assert captured_event.source.thread_id == "om_root_abc", (
            f"expected fallback to root_id, got "
            f"{captured_event.source.thread_id!r}"
        )

    @pytest.mark.asyncio
    async def test_upper_message_id_used_when_thread_and_root_absent(
        self, adapter_harness
    ):
        event = _build_text_event(
            chat_id="oc_topic_chat",
            chat_type="group",
            sender_open_id="ou_user_topic_2",
            text="reply within thread",
            message_id="om_topic_msg_2",
        )
        # Only upper_message_id — second-tier fallback.
        event["event"]["message"]["upper_message_id"] = "om_upper_xyz"

        await dispatch_inbound_event(adapter_harness, event)
        assert len(adapter_harness.captured_inbound) == 1
        captured_event = adapter_harness.captured_inbound[0]
        assert captured_event.source.thread_id == "om_upper_xyz", (
            f"expected fallback to upper_message_id, got "
            f"{captured_event.source.thread_id!r}"
        )

    @pytest.mark.asyncio
    async def test_conversation_thread_id_takes_priority(self, adapter_harness):
        """When SDK sets conversation.thread_id, fallbacks are not used."""
        event = _build_text_event(
            chat_id="oc_topic_chat",
            chat_type="group",
            sender_open_id="ou_user_topic_3",
            text="canonical thread",
            message_id="om_topic_msg_3",
        )
        event["event"]["message"]["thread_id"] = "om_canonical"
        # Decoy fields that would be picked up by the fallback chain;
        # priority must keep conversation.thread_id.
        event["event"]["message"]["root_id"] = "om_root_decoy"
        event["event"]["message"]["upper_message_id"] = "om_upper_decoy"

        await dispatch_inbound_event(adapter_harness, event)
        captured_event = adapter_harness.captured_inbound[0]
        assert captured_event.source.thread_id == "om_canonical", (
            f"conversation.thread_id should win, got "
            f"{captured_event.source.thread_id!r}"
        )


class TestThreadReplySkipFallback:
    """When an in-thread reply fails with a "target revoked / withdrawn"
    code, the SDK's outbound sender will fall back to a fresh top-level
    send. For in-thread context that is the wrong behavior (it creates
    a new thread), so the adapter must intercept and skip the fallback
    when metadata signals the call is bound to a thread.

    Today the SDK does the auto-fallback in
    ``_send_one_with_fallback`` (see
    ``lark_oapi.channel.outbound.sender``) and our adapter does NOT
    yet pass an in-thread guard. The trim of the legacy adapter
    deleted ``_FEISHU_REPLY_FALLBACK_CODES`` along with its guard.
    This test pins the behavior we want: a single attempt, no
    second top-level send.
    """

    @pytest.mark.asyncio
    async def test_thread_metadata_reply_target_is_used_without_explicit_reply_to(
        self, adapter_harness
    ):
        from types import SimpleNamespace

        sends: list = []

        async def capture_send(chat_id, message, opts=None):
            sends.append(dict(opts) if opts else None)
            return SimpleNamespace(success=True, message_id="om_reply", error=None)

        adapter_harness.adapter._channel.send = capture_send

        result = await adapter_harness.adapter.send(
            "oc_topic_chat",
            "thread reply",
            metadata={
                "thread_id": "omt_thread_a",
                "reply_to_message_id": "om_trigger",
            },
        )

        assert result.success is True
        assert sends == [{
            "reply_to": "om_trigger",
            "reply_in_thread": True,
            "reply_target_gone": "fail",
        }]

    @pytest.mark.asyncio
    async def test_thread_reply_opts_are_preserved_for_all_chunks_with_thread_only_metadata(
        self, adapter_harness
    ):
        from types import SimpleNamespace

        adapter_harness.adapter.MAX_MESSAGE_LENGTH = 20
        sends: list = []

        async def capture_send(chat_id, message, opts=None):
            sends.append(dict(opts) if opts else None)
            return SimpleNamespace(success=True, message_id=f"om_reply_{len(sends)}", error=None)

        adapter_harness.adapter._channel.send = capture_send

        result = await adapter_harness.adapter.send(
            "oc_topic_chat",
            "x" * 60,
            reply_to="om_trigger",
            metadata={"thread_id": "omt_thread_a"},
        )

        assert result.success is True
        assert len(sends) > 1
        assert all(
            opts
            and opts.get("reply_to") == "om_trigger"
            and opts.get("reply_in_thread") is True
            and opts.get("reply_target_gone") == "fail"
            for opts in sends
        ), sends

    @pytest.mark.asyncio
    async def test_captioned_document_keeps_file_inside_thread_with_thread_only_metadata(
        self, adapter_harness, tmp_path
    ):
        from types import SimpleNamespace

        sends: list = []

        async def capture_send(chat_id, message, opts=None):
            sends.append(dict(opts) if opts else None)
            return SimpleNamespace(success=True, message_id=f"om_reply_{len(sends)}", error=None)

        adapter_harness.adapter._channel.send = capture_send
        doc_path = tmp_path / "sample.txt"
        doc_path.write_text("payload", encoding="utf-8")

        result = await adapter_harness.adapter.send_document(
            "oc_topic_chat",
            str(doc_path),
            caption="caption",
            reply_to="om_trigger",
            metadata={"thread_id": "omt_thread_a"},
        )

        assert result.success is True
        assert sends == [
            {"reply_to": "om_trigger", "reply_in_thread": True, "reply_target_gone": "fail"},
            {"reply_to": "om_trigger", "reply_in_thread": True, "reply_target_gone": "fail"},
        ]

    @pytest.mark.asyncio
    async def test_failed_reply_in_thread_does_not_fallback_to_top_level(
        self, adapter_harness, monkeypatch
    ):
        from types import SimpleNamespace

        # Replace _channel.send with a stub that simulates the SDK
        # withdrawn-message fallback semantics: caller asks for a reply,
        # send returns SUCCESS but signals it created a fresh top-level
        # message. (We assert via send-call count + opts that the adapter
        # didn't call channel.send twice for an in-thread reply.)
        sends: list = []

        async def failing_reply(chat_id, message, opts=None):
            sends.append({
                "chat_id": chat_id,
                "message": message,
                "opts": dict(opts) if opts else None,
            })
            # Simulate withdrawn-message error surfacing at the SDK
            # boundary for a reply-in-thread.
            return SimpleNamespace(
                success=False,
                message_id=None,
                error=SimpleNamespace(code=230020, message="target revoked"),
            )

        adapter_harness.adapter._channel.send = failing_reply

        result = await adapter_harness.adapter.send(
            "oc_topic_chat",
            "in-thread reply",
            reply_to="om_msg_x",
            metadata={"thread_id": "om_thread_a", "reply_to": "om_msg_x"},
        )
        # Whether the underlying call succeeded or not, the adapter must
        # NOT issue a second top-level send (which would create a new
        # thread). Pin one invocation, with reply_to set on it.
        assert len(sends) == 1, (
            f"expected exactly one channel.send invocation; got "
            f"{len(sends)}: {sends!r}"
        )
        assert sends[0]["opts"] is not None
        assert sends[0]["opts"].get("reply_to") == "om_msg_x"
        assert sends[0]["opts"].get("reply_in_thread") is True
        assert sends[0]["opts"].get("reply_target_gone") == "fail"
        # Result is whatever the adapter chose to surface from the
        # single attempt (success/failure both acceptable for this
        # contract — what matters is no second send).
        assert result is not None

    @pytest.mark.asyncio
    async def test_plain_reply_keeps_sdk_default_fresh_fallback(
        self, adapter_harness
    ):
        from types import SimpleNamespace

        sends: list = []

        async def capture_send(chat_id, message, opts=None):
            sends.append(dict(opts) if opts else None)
            return SimpleNamespace(success=True, message_id="om_reply", error=None)

        adapter_harness.adapter._channel.send = capture_send

        result = await adapter_harness.adapter.send(
            "oc_plain_chat",
            "plain reply",
            reply_to="om_parent",
            metadata={},
        )

        assert result.success is True
        assert sends == [{"reply_to": "om_parent"}]
