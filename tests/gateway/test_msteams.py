"""Tests for the Microsoft Teams gateway integration."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPlatformEnum(unittest.TestCase):
    def test_msteams_in_platform_enum(self):
        from gateway.config import Platform

        self.assertEqual(Platform.MSTEAMS.value, "msteams")


class TestConfigEnvOverrides(unittest.TestCase):
    @patch.dict(os.environ, {
        "MSTEAMS_APP_ID": "app-id",
        "MSTEAMS_APP_PASSWORD": "secret",
        "MSTEAMS_TENANT_ID": "tenant-id",
    }, clear=False)
    def test_msteams_config_loaded_from_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.MSTEAMS, config.platforms)
        self.assertTrue(config.platforms[Platform.MSTEAMS].enabled)
        self.assertEqual(config.platforms[Platform.MSTEAMS].extra["app_id"], "app-id")
        self.assertEqual(config.platforms[Platform.MSTEAMS].extra["tenant_id"], "tenant-id")

    @patch.dict(os.environ, {
        "MSTEAMS_APP_ID": "app-id",
        "MSTEAMS_APP_PASSWORD": "secret",
        "MSTEAMS_TENANT_ID": "tenant-id",
        "MSTEAMS_HOME_CHANNEL": "19:testthread",
    }, clear=False)
    def test_msteams_home_channel_loaded(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.platforms[Platform.MSTEAMS].home_channel
        self.assertIsNotNone(home)
        self.assertEqual(home.chat_id, "19:testthread")

    @patch.dict(os.environ, {
        "MSTEAMS_APP_ID": "app-id",
        "MSTEAMS_APP_PASSWORD": "secret",
        "MSTEAMS_TENANT_ID": "tenant-id",
        "MSTEAMS_REQUIRE_MENTION": "false",
        "MSTEAMS_REPLY_STYLE": "top-level",
        "MSTEAMS_DM_POLICY": "allowlist",
        "MSTEAMS_GROUP_POLICY": "open",
        "MSTEAMS_CHUNK_MODE": "newline",
        "MSTEAMS_ALLOW_FROM": "aad-1,aad-2",
        "MSTEAMS_GROUP_ALLOW_FROM": "aad-3",
        "MSTEAMS_DANGEROUSLY_ALLOW_NAME_MATCHING": "true",
        "MSTEAMS_TEAMS_JSON": '{"team-1": {"channels": {"channel-1": {"requireMention": false}}}}',
        "MSTEAMS_TEXT_CHUNK_LIMIT": "3500",
        "MSTEAMS_HISTORY_LIMIT": "25",
        "MSTEAMS_DM_HISTORY_LIMIT": "12",
        "MSTEAMS_MAX_BODY_BYTES": "2048",
        "MSTEAMS_IDEMPOTENCY_TTL_SECONDS": "7200",
        "MSTEAMS_AUTH_CACHE_TTL_SECONDS": "1800",
        "MSTEAMS_PENDING_UPLOAD_TTL_SECONDS": "600",
        "MSTEAMS_STATE_PATH": "/tmp/msteams-state.json",
        "MSTEAMS_SHAREPOINT_SITE_ID": "tenant.sharepoint.com,site-guid,drive-guid",
        "MSTEAMS_MEDIA_ALLOW_HOSTS": "graph.microsoft.com,teams.microsoft.com",
        "MSTEAMS_MEDIA_AUTH_ALLOW_HOSTS": "graph.microsoft.com,api.botframework.com",
    }, clear=False)
    def test_msteams_advanced_policy_env_loaded(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        extra = config.platforms[Platform.MSTEAMS].extra
        self.assertFalse(extra["require_mention"])
        self.assertEqual(extra["reply_style"], "top-level")
        self.assertEqual(extra["dm_policy"], "allowlist")
        self.assertEqual(extra["group_policy"], "open")
        self.assertEqual(extra["chunk_mode"], "newline")
        self.assertEqual(extra["text_chunk_limit"], 3500)
        self.assertEqual(extra["history_limit"], 25)
        self.assertEqual(extra["dm_history_limit"], 12)
        self.assertEqual(extra["max_body_bytes"], 2048)
        self.assertEqual(extra["idempotency_ttl_seconds"], 7200)
        self.assertEqual(extra["auth_cache_ttl_seconds"], 1800)
        self.assertEqual(extra["pending_upload_ttl_seconds"], 600)
        self.assertEqual(extra["state_path"], "/tmp/msteams-state.json")
        self.assertEqual(extra["share_point_site_id"], "tenant.sharepoint.com,site-guid,drive-guid")
        self.assertEqual(extra["media_allow_hosts"], ["graph.microsoft.com", "teams.microsoft.com"])
        self.assertEqual(extra["media_auth_allow_hosts"], ["graph.microsoft.com", "api.botframework.com"])
        self.assertEqual(extra["allow_from"], ["aad-1", "aad-2"])
        self.assertEqual(extra["group_allow_from"], ["aad-3"])
        self.assertTrue(extra["dangerously_allow_name_matching"])
        self.assertIn("team-1", extra["teams"])

    @patch.dict(os.environ, {
        "MSTEAMS_APP_ID": "app-id",
        "MSTEAMS_APP_PASSWORD": "secret",
        "MSTEAMS_TENANT_ID": "tenant-id",
        "MSTEAMS_ALLOWED_USERS": "aad-legacy-1,aad-legacy-2",
    }, clear=False)
    def test_msteams_legacy_allowed_users_bridge_into_new_policy_lists(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        extra = config.platforms[Platform.MSTEAMS].extra
        self.assertEqual(extra["allow_from"], ["aad-legacy-1", "aad-legacy-2"])
        self.assertEqual(extra["group_allow_from"], ["aad-legacy-1", "aad-legacy-2"])

    @patch.dict(os.environ, {
        "MSTEAMS_APP_ID": "app-id",
        "MSTEAMS_APP_PASSWORD": "secret",
        "MSTEAMS_TENANT_ID": "tenant-id",
    }, clear=False)
    def test_msteams_in_connected_platforms(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.MSTEAMS, config.get_connected_platforms())


class TestGatewayIntegration(unittest.TestCase):
    def test_msteams_in_adapter_factory(self):
        source = Path("gateway/run.py").read_text(encoding="utf-8")
        self.assertIn("Platform.MSTEAMS", source)
        self.assertIn("MSTeamsAdapter", source)

    def test_msteams_in_authorization_maps(self):
        source = Path("gateway/run.py").read_text(encoding="utf-8")
        self.assertIn("MSTEAMS_ALLOWED_USERS", source)
        self.assertIn("MSTEAMS_ALLOW_ALL_USERS", source)

    def test_msteams_toolset_exists(self):
        from toolsets import TOOLSETS

        self.assertIn("hermes-msteams", TOOLSETS)
        self.assertIn("hermes-msteams", TOOLSETS["hermes-gateway"]["includes"])

    def test_msteams_in_platform_registry(self):
        from hermes_cli.platforms import PLATFORMS

        self.assertIn("msteams", PLATFORMS)
        self.assertEqual(PLATFORMS["msteams"].default_toolset, "hermes-msteams")


class TestConfigYamlHomeChannelPersistence(unittest.TestCase):
    def test_structured_msteams_home_channel_loaded_from_config_yaml(self):
        from gateway.config import Platform, load_gateway_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
platforms:
  msteams:
    home_channel:
      platform: msteams
      chat_id: conv-123
      name: Sean DM
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with patch("gateway.config.get_hermes_home", return_value=Path(tmpdir)), patch.dict(os.environ, {}, clear=True):
                config = load_gateway_config()

            home = config.get_home_channel(Platform.MSTEAMS)
            self.assertIsNotNone(home)
            self.assertEqual(home.chat_id, "conv-123")
            self.assertEqual(home.name, "Sean DM")

    def test_legacy_top_level_msteams_home_channel_in_config_yaml_is_bridged(self):
        from gateway.config import Platform, load_gateway_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
MSTEAMS_HOME_CHANNEL: conv-legacy
MSTEAMS_HOME_CHANNEL_NAME: Legacy DM
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with patch("gateway.config.get_hermes_home", return_value=Path(tmpdir)), patch.dict(os.environ, {}, clear=True):
                config = load_gateway_config()

            home = config.get_home_channel(Platform.MSTEAMS)
            self.assertIsNotNone(home)
            self.assertEqual(home.chat_id, "conv-legacy")
            self.assertEqual(home.name, "Legacy DM")


class TestSetHomeCommandPersistence(unittest.IsolatedAsyncioTestCase):
    async def test_sethome_writes_structured_config_and_updates_runtime_config(self):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.platforms.base import MessageEvent
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.MSTEAMS: PlatformConfig(enabled=True)})

        source = SessionSource(
            platform=Platform.MSTEAMS,
            chat_id="conv-42",
            chat_name="Sean DM",
            chat_type="dm",
        )
        event = MessageEvent(text="/sethome", source=source)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
MSTEAMS_HOME_CHANNEL: old-conv
MSTEAMS_HOME_CHANNEL_NAME: Old Name
platforms:
  msteams:
    enabled: true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with patch("gateway.run._hermes_home", Path(tmpdir)), patch.dict(os.environ, {}, clear=True):
                result = await runner._handle_set_home_command(event)

                self.assertIn("✅ Home channel set", result)
                self.assertEqual(os.environ["MSTEAMS_HOME_CHANNEL"], "conv-42")
                self.assertEqual(os.environ["MSTEAMS_HOME_CHANNEL_NAME"], "Sean DM")
                self.assertTrue(runner._has_home_channel_configured(Platform.MSTEAMS))

                import yaml
                persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertNotIn("MSTEAMS_HOME_CHANNEL", persisted)
        self.assertNotIn("MSTEAMS_HOME_CHANNEL_NAME", persisted)
        self.assertEqual(
            persisted["platforms"]["msteams"]["home_channel"],
            {"platform": "msteams", "chat_id": "conv-42", "name": "Sean DM"},
        )
        self.assertEqual(runner.config.get_home_channel(Platform.MSTEAMS).chat_id, "conv-42")


class TestMSTeamsGatewayAuthorization(unittest.TestCase):
    def _runner(self, extra):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.MSTEAMS: PlatformConfig(enabled=True, extra=extra)})
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved.return_value = False
        return runner

    @patch.dict(os.environ, {
        "MSTEAMS_ALLOWED_USERS": "",
        "MSTEAMS_ALLOW_ALL_USERS": "",
        "GATEWAY_ALLOWED_USERS": "",
        "GATEWAY_ALLOW_ALL_USERS": "",
    }, clear=False)
    def test_open_dm_policy_bypasses_legacy_gateway_allowlists(self):
        from gateway.config import Platform
        from gateway.session import SessionSource

        runner = self._runner({"dm_policy": "open"})
        source = SessionSource(
            platform=Platform.MSTEAMS,
            user_id="aad-open",
            user_name="Sean",
            chat_id="conv-open",
            chat_type="dm",
        )

        self.assertTrue(runner._is_user_authorized(source))

    @patch.dict(os.environ, {
        "MSTEAMS_ALLOWED_USERS": "",
        "MSTEAMS_ALLOW_ALL_USERS": "",
        "GATEWAY_ALLOWED_USERS": "",
        "GATEWAY_ALLOW_ALL_USERS": "",
    }, clear=False)
    def test_pairing_dm_policy_still_requires_pairing_store_approval(self):
        from gateway.config import Platform
        from gateway.session import SessionSource

        runner = self._runner({"dm_policy": "pairing"})
        source = SessionSource(
            platform=Platform.MSTEAMS,
            user_id="aad-pairing",
            user_name="Sean",
            chat_id="conv-pairing",
            chat_type="dm",
        )

        self.assertFalse(runner._is_user_authorized(source))
        runner.pairing_store.is_approved.return_value = True
        self.assertTrue(runner._is_user_authorized(source))


class TestMSTeamsAdapter(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, **extra):
        from gateway.config import PlatformConfig
        from gateway.platforms.msteams import MSTeamsAdapter

        merged = {"app_id": "app", "app_password": "secret", "tenant_id": "tenant"}
        merged.update(extra)
        return MSTeamsAdapter(PlatformConfig(enabled=True, extra=merged))

    def _activity(self, *, conversation_type="personal", conversation_id="conv-1", text="hello", entities=None, reply_to_id=None, sender_id="aad-1", sender_name="Sean", recipient_id="bot-1", recipient_name="Hermes", team_id=None, team_name=None, channel_id=None, channel_name=None, activity_id="activity-1"):
        activity = {
            "type": "message",
            "id": activity_id,
            "text": text,
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "conversation": {"id": conversation_id, "conversationType": conversation_type, "name": "Test Chat"},
            "from": {"id": sender_id, "aadObjectId": sender_id, "name": sender_name},
            "recipient": {"id": recipient_id, "name": recipient_name},
            "channelId": "msteams",
        }
        if entities is not None:
            activity["entities"] = entities
        if reply_to_id:
            activity["replyToId"] = reply_to_id
        if team_id or channel_id or team_name or channel_name:
            activity["channelData"] = {
                "tenant": {"id": "tenant-1"},
                "team": {"id": team_id, "name": team_name},
                "channel": {"id": channel_id, "name": channel_name},
            }
        return activity

    async def test_build_event_normalizes_personal_message(self):
        adapter = self._adapter()
        activity = self._activity(text="<at>Hermes</at> hello there")

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.text, "hello there")
        self.assertEqual(event.source.chat_id, "conv-1")
        self.assertEqual(event.source.chat_type, "dm")
        self.assertEqual(event.source.user_id, "aad-1")
        self.assertIsNone(event.source.thread_id)

    async def test_dm_policy_allowlist_blocks_unknown_sender(self):
        adapter = self._adapter(dm_policy="allowlist", allow_from=["aad-9"])

        event = adapter._build_event(self._activity())

        self.assertIsNone(event)

    async def test_group_policy_blocks_by_default_even_with_mention(self):
        adapter = self._adapter()
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> please help",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )

        event = adapter._build_event(activity)

        self.assertIsNone(event)

    async def test_group_open_policy_requires_mention_by_default(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(conversation_type="groupChat", text="plain group message")

        event = adapter._build_event(activity)

        self.assertIsNone(event)

    async def test_group_open_policy_passes_when_bot_mentioned(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> please help",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.source.chat_type, "group")

    async def test_group_open_policy_preserves_slash_command_after_bot_mention(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> /new",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.text, "/new")
        self.assertEqual(event.get_command(), "new")

    async def test_group_allowlist_allows_known_sender(self):
        adapter = self._adapter(group_policy="allowlist", group_allow_from=["aad-1"])
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> please help",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)

    async def test_group_allowlist_allows_wildcard_sender(self):
        adapter = self._adapter(group_policy="allowlist", group_allow_from=["*"])
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> please help",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)

    async def test_team_channel_override_can_disable_require_mention(self):
        adapter = self._adapter(
            group_policy="allowlist",
            teams={
                "team-1": {
                    "channels": {
                        "channel-1": {"requireMention": False, "replyStyle": "top-level"},
                    }
                }
            },
        )
        activity = self._activity(
            conversation_type="channel",
            conversation_id="conv-channel",
            text="plain channel message",
            team_id="team-1",
            channel_id="channel-1",
            team_name="Team 1",
            channel_name="General",
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.source.chat_type, "channel")
        ref = adapter._conversations.get("conv-channel")
        self.assertEqual(ref.reply_style, "top-level")
        self.assertFalse(ref.require_mention)

    async def test_dangerous_name_matching_allows_named_team_channel(self):
        adapter = self._adapter(
            group_policy="allowlist",
            dangerously_allow_name_matching=True,
            teams={
                "Team Name": {
                    "channels": {
                        "General": {"requireMention": False},
                    }
                }
            },
        )
        activity = self._activity(
            conversation_type="channel",
            conversation_id="conv-channel-2",
            text="plain channel message",
            team_id="unknown-team-id",
            channel_id="unknown-channel-id",
            team_name="Team Name",
            channel_name="General",
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)

    async def test_top_level_group_message_does_not_create_thread_session(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> please help",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )

        event = adapter._build_event(activity)

        self.assertIsNone(event.source.thread_id)

    async def test_untrusted_service_url_is_rejected(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> please help",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["serviceUrl"] = "https://attacker.example.com/capture"

        event = adapter._build_event(activity)

        self.assertIsNone(event)

    async def test_handle_activity_requires_bearer_auth(self):
        from types import SimpleNamespace

        adapter = self._adapter(group_policy="open")
        request = SimpleNamespace(headers={}, content_length=0)

        response = await adapter._check_bearer_auth(request)

        self.assertEqual(response.status, 401)

    async def test_handle_activity_rejects_invalid_verified_token(self):
        class StubValidator:
            async def validate(self, token, service_url=None):
                raise RuntimeError("bad token")

        class StubRequest:
            headers = {"Authorization": "Bearer token"}
            content_length = 0

            async def json(self):
                return {"serviceUrl": "https://smba.trafficmanager.net/amer/"}

        adapter = self._adapter(group_policy="open")
        adapter._auth_validator = StubValidator()

        response = await adapter._check_bearer_auth(StubRequest())

        self.assertEqual(response.status, 401)

    async def test_handle_activity_accepts_valid_verified_token(self):
        class StubValidator:
            async def validate(self, token, service_url=None):
                return {"aud": "app", "serviceurl": service_url}

        class StubRequest:
            headers = {"Authorization": "Bearer token"}
            content_length = 0

            async def json(self):
                return {"serviceUrl": "https://smba.trafficmanager.net/amer/"}

        adapter = self._adapter(group_policy="open")
        adapter._auth_validator = StubValidator()

        response = await adapter._check_bearer_auth(StubRequest())

        self.assertIsNone(response)

    async def test_handle_activity_rejects_oversized_payload_by_content_length(self):
        from types import SimpleNamespace

        adapter = self._adapter(max_body_bytes=100)
        request = SimpleNamespace(headers={"Authorization": "Bearer token"}, content_length=2048)

        response = await adapter._handle_activity(request)

        self.assertEqual(response.status, 413)

    async def test_handle_activity_rejects_invalid_json(self):
        class StubRequest:
            headers = {"Authorization": "Bearer token"}
            content_length = 5

            async def read(self):
                return b"{bad]"

        class StubValidator:
            async def validate(self, token, service_url=None):
                return {"aud": "app"}

        adapter = self._adapter()
        adapter._auth_validator = StubValidator()
        response = await adapter._handle_activity(StubRequest())

        self.assertEqual(response.status, 400)

    async def test_handle_activity_deduplicates_same_activity_id(self):
        class StubRequest:
            def __init__(self, body):
                self.headers = {"Authorization": "Bearer token"}
                self.content_length = len(body)
                self._body = body

            async def read(self):
                return self._body

        class StubValidator:
            async def validate(self, token, service_url=None):
                return {"aud": "app", "serviceurl": service_url}

        adapter = self._adapter(group_policy="open")
        adapter._auth_validator = StubValidator()
        body = b'{"type":"message","id":"dup-1","text":"<at>Hermes</at> hi","serviceUrl":"https://smba.trafficmanager.net/amer/","conversation":{"id":"conv-dup","conversationType":"groupChat","name":"Test Chat"},"from":{"id":"aad-1","aadObjectId":"aad-1","name":"Sean"},"recipient":{"id":"bot-1","name":"Hermes"},"channelId":"msteams","entities":[{"type":"mention","mentioned":{"id":"bot-1","name":"Hermes"}}]}'

        response1 = await adapter._handle_activity(StubRequest(body))
        response2 = await adapter._handle_activity(StubRequest(body))

        self.assertEqual(response1.status, 202)
        self.assertEqual(response2.status, 200)

    async def test_reply_to_bot_message_counts_as_engaged_thread(self):
        from gateway.platforms.msteams_state import ConversationRef

        adapter = self._adapter(group_policy="open")
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-1",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
            )
        )
        adapter._conversations.register_sent_message("conv-1", "bot-sent-1")
        activity = self._activity(conversation_type="groupChat", text="follow-up without mention", reply_to_id="bot-sent-1")

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.source.thread_id, "bot-sent-1")

    async def test_send_respects_top_level_reply_style(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"conversation_id": ref.conversation_id, "content": content, "reply_to": reply_to, "entities": entities, "attachments": attachments})
                return {"id": "sent-1"}

        adapter = self._adapter(reply_style="top-level")
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-5",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="existing-thread-anchor",
                reply_style="top-level",
            )
        )

        result = await adapter.send("conv-5", "hello")

        self.assertTrue(result.success)
        self.assertEqual(adapter._bot.calls[0]["reply_to"], None)

    async def test_send_threads_to_latest_inbound_anchor_without_self_threading(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []
                self.counter = 0

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.counter += 1
                message_id = f"sent-{self.counter}"
                self.calls.append({"conversation_id": ref.conversation_id, "content": content, "reply_to": reply_to, "id": message_id, "entities": entities, "attachments": attachments})
                return {"id": message_id}

        adapter = self._adapter(group_policy="open")
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-6",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-1",
                last_inbound_activity_id="user-anchor-1",
                reply_style="thread",
            )
        )

        result1 = await adapter.send("conv-6", "hello")
        result2 = await adapter.send("conv-6", "hello again")

        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertEqual(adapter._bot.calls[0]["reply_to"], "user-anchor-1")
        self.assertEqual(adapter._bot.calls[1]["reply_to"], "user-anchor-1")
        ref = adapter._conversations.get("conv-6")
        self.assertEqual(ref.activity_id, "user-anchor-1")
        self.assertEqual(ref.last_sent_activity_id, "sent-2")
        self.assertTrue(adapter._conversations.has_sent_activity("conv-6", "sent-1"))
        self.assertTrue(adapter._conversations.has_sent_activity("conv-6", "sent-2"))

    async def test_inbound_reply_to_bot_does_not_override_user_anchor(self):
        from gateway.platforms.msteams_state import ConversationRef

        adapter = self._adapter(group_policy="open")
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-7",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-1",
                last_inbound_activity_id="user-anchor-1",
                reply_style="thread",
                sent_activity_ids={"bot-sent-1"},
            )
        )
        activity = self._activity(
            conversation_type="groupChat",
            conversation_id="conv-7",
            text="follow-up without mention",
            reply_to_id="bot-sent-1",
            activity_id="activity-7",
        )

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        ref = adapter._conversations.get("conv-7")
        self.assertEqual(ref.activity_id, "user-anchor-1")
        self.assertEqual(ref.last_inbound_activity_id, "activity-7")

    async def test_build_event_extracts_reply_context_from_attachment_text(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="follow up",
            reply_to_id="reply-1",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["attachments"] = [{"contentType": "text/html", "text": "Quoted parent message"}]

        event = adapter._build_event(activity)

        self.assertEqual(event.reply_to_text, "Quoted parent message")

    async def test_build_event_extracts_reply_context_from_reply_html_attachment(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text='<blockquote itemscope itemtype="http://schema.skype.com/Reply" itemid="1776239661790"><strong itemprop="mri" itemid="28:49858f21-9a40-4411-9891-e4c3b1c8e1fa">Display Name</strong><span itemprop="time">2026/4/15 15:54</span><p itemprop="copy">已改。 当前状态</p></blockquote><p><at id="0">Captain</at>&nbsp;同意</p>',
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
            activity_id="activity-reply-html",
        )
        activity["attachments"] = [{
            "contentType": "text/html",
            "content": '<blockquote itemscope itemtype="http://schema.skype.com/Reply" itemid="1776239661790"><strong itemprop="mri" itemid="28:49858f21-9a40-4411-9891-e4c3b1c8e1fa">Display Name</strong><span itemprop="time">2026/4/15 15:54</span><p itemprop="copy">已改。 当前状态</p></blockquote>',
        }]

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.text, "同意")
        self.assertEqual(event.reply_to_text, "已改。 当前状态")

    async def test_build_event_uses_message_reference_preview_when_reply_to_id_missing(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> 好",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
            activity_id="activity-message-reference",
        )
        activity["attachments"] = [{
            "contentType": "messageReference",
            "content": json.dumps({
                "messageId": "parent-activity-1",
                "messagePreview": "Quoted parent message",
            }),
        }]

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.text, "好")
        self.assertEqual(event.reply_to_message_id, "parent-activity-1")
        self.assertEqual(event.source.thread_id, "parent-activity-1")
        self.assertEqual(event.reply_to_text, "Quoted parent message")

    async def test_build_event_uses_nested_message_reference_preview(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="follow up",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
            activity_id="activity-nested-message-reference",
        )
        activity["attachments"] = [{
            "contentType": "messageReference",
            "content": json.dumps({
                "messageReference": {
                    "messageId": "parent-activity-2",
                    "messagePreview": "Nested quoted parent message",
                }
            }),
        }]

        event = adapter._build_event(activity)

        self.assertIsNotNone(event)
        self.assertEqual(event.reply_to_message_id, "parent-activity-2")
        self.assertEqual(event.reply_to_text, "Nested quoted parent message")

    async def test_build_event_uses_media_placeholder_when_reply_attachment_has_no_text(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="groupChat",
            text="follow up",
            reply_to_id="reply-2",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["attachments"] = [{"contentType": "image/png", "contentUrl": "https://example.test/image.png"}]

        event = adapter._build_event(activity)

        self.assertEqual(event.reply_to_text, "<media:image>")

    async def test_media_download_skips_hosts_outside_allowlist(self):
        adapter = self._adapter(group_policy="open", media_allow_hosts=["graph.microsoft.com"])
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> see file",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["attachments"] = [{"contentType": "application/pdf", "contentUrl": "https://evil.example.test/file.pdf", "name": "file.pdf"}]

        class StubHTTP:
            def __init__(self):
                self.calls = []

            def get(self, url, headers=None):
                self.calls.append({"url": url, "headers": headers})
                raise AssertionError("HTTP client should not be used for disallowed hosts")

        adapter._http = StubHTTP()
        media_urls, media_types = await adapter._collect_media_from_activity(activity)

        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])
        self.assertEqual(adapter._http.calls, [])

    async def test_graph_media_fallback_recovers_hosted_contents_from_html_stub(self):
        adapter = self._adapter(group_policy="open")
        activity = self._activity(
            conversation_type="channel",
            conversation_id="conv-channel-media",
            text="<at>Hermes</at> see media",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
            reply_to_id="root-1",
            team_id="team-1",
            channel_id="channel-1",
            activity_id="reply-1",
        )
        activity["attachments"] = [{"contentType": "text/html", "content": '<div><img src="cid:image" /></div>'}]

        class StubGraph:
            def build_message_url_candidates(self, payload):
                return ["/teams/team-1/channels/channel-1/messages/root-1/replies/reply-1"]

            async def download_message_media(self, message_path):
                return [{
                    "kind": "hosted",
                    "name": "image/png",
                    "content_type": "image/png",
                    "data": b"\x89PNG\r\n\x1a\n",
                }]

        adapter._graph = StubGraph()

        media_urls, media_types = await adapter._collect_media_from_activity(activity)

        self.assertEqual(len(media_urls), 1)
        self.assertEqual(media_types, ["image/png"])
        self.assertTrue(media_urls[0].endswith('.png'))

    async def test_graph_media_fallback_recovers_reference_attachment_via_graph_share_path(self):
        adapter = self._adapter(group_policy="open", media_allow_hosts=["sharepoint.example.test"])
        activity = self._activity(
            conversation_type="channel",
            conversation_id="conv-channel-doc",
            text="<at>Hermes</at> see doc",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
            reply_to_id="root-2",
            team_id="team-1",
            channel_id="channel-1",
            activity_id="reply-2",
        )
        activity["attachments"] = [{"contentType": "text/html", "content": '<div>stub</div>'}]

        class StubGraph:
            def __init__(self):
                self.bytes_calls = []

            def build_message_url_candidates(self, payload):
                return ["/teams/team-1/channels/channel-1/messages/root-2/replies/reply-2"]

            async def download_message_media(self, message_path):
                return [{
                    "kind": "reference",
                    "name": "file.pdf",
                    "content_type": "application/pdf",
                    "content_url": "https://sharepoint.example.test/sites/test/file.pdf",
                    "graph_path": "/shares/u!abc/driveItem/content",
                }]

            async def _graph_get_bytes(self, graph_path):
                self.bytes_calls.append(graph_path)
                return b"%PDF-1.4"

        adapter._graph = StubGraph()

        with patch("gateway.platforms.msteams.cache_document_from_bytes", return_value="/tmp/from-graph.pdf") as cache_doc:
            media_urls, media_types = await adapter._collect_media_from_activity(activity)

        self.assertEqual(media_urls, ["/tmp/from-graph.pdf"])
        self.assertEqual(media_types, ["application/pdf"])
        self.assertEqual(adapter._graph.bytes_calls, ["/shares/u!abc/driveItem/content"])
        cache_doc.assert_called_once_with(b"%PDF-1.4", "file.pdf")

    async def test_media_download_retries_with_auth_header_for_allowed_host(self):
        adapter = self._adapter(group_policy="open", media_allow_hosts=["graph.microsoft.com"], media_auth_allow_hosts=["graph.microsoft.com"])
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> see file",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["attachments"] = [{"contentType": "application/pdf", "contentUrl": "https://graph.microsoft.com/v1.0/shares/u!/content", "name": "file.pdf"}]

        class StubResponse:
            def __init__(self, status, body=b"", payload=None):
                self.status = status
                self._body = body
                self._payload = payload if payload is not None else {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def read(self):
                return self._body

            async def json(self, content_type=None):
                return self._payload

        class StubHTTP:
            def __init__(self):
                self.calls = []
                self._responses = [
                    StubResponse(401, payload={"error": "unauthorized"}),
                    StubResponse(200, body=b"pdf-bytes"),
                ]

            def get(self, url, headers=None):
                self.calls.append({"url": url, "headers": headers})
                return self._responses.pop(0)

        class StubTokenProvider:
            async def _get_token_for_scope(self, scope):
                return f"token-for:{scope}"

        adapter._http = StubHTTP()
        adapter._graph = StubTokenProvider()
        adapter._bot = StubTokenProvider()

        with patch("gateway.platforms.msteams.cache_document_from_bytes", return_value="/tmp/cached-file.pdf") as cache_doc:
            media_urls, media_types = await adapter._collect_media_from_activity(activity)

        self.assertEqual(media_urls, ["/tmp/cached-file.pdf"])
        self.assertEqual(media_types, ["application/pdf"])
        self.assertEqual(len(adapter._http.calls), 2)
        self.assertIsNone(adapter._http.calls[0]["headers"])
        self.assertIn("Authorization", adapter._http.calls[1]["headers"])
        cache_doc.assert_called_once_with(b"pdf-bytes", "file.pdf")

    async def test_enrich_event_media_downloads_attachments_to_local_cache(self):
        from gateway.platforms.base import MessageType

        adapter = self._adapter(group_policy="open", media_allow_hosts=["example.test"])
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> see image",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["attachments"] = [{"contentType": "image/png", "contentUrl": "https://example.test/image.png", "name": "image.png"}]
        event = adapter._build_event(activity)

        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\x0bIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        async def fake_download(url):
            return png_bytes

        adapter._download_media_bytes = fake_download
        enriched = await adapter._enrich_event_media(event, activity)

        self.assertEqual(len(enriched.media_urls), 1)
        self.assertTrue(enriched.media_urls[0].endswith('.png'))
        self.assertEqual(Path(enriched.media_urls[0]).read_bytes(), png_bytes)
        self.assertEqual(enriched.media_types, ["image/png"])
        self.assertEqual(enriched.message_type, MessageType.PHOTO)

    async def test_collect_media_from_file_download_info_image_attachment(self):
        adapter = self._adapter(group_policy="open", media_allow_hosts=["example.test"])
        activity = self._activity(
            conversation_type="groupChat",
            text="<at>Hermes</at> see image",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        activity["attachments"] = [{
            "contentType": "application/vnd.microsoft.teams.file.download.info",
            "name": "inline-image",
            "content": {
                "downloadUrl": "https://example.test/download/inline-image",
                "fileType": "png",
                "fileName": "inline-image.png",
            },
        }]

        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\x0bIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        async def fake_download(url):
            return png_bytes

        adapter._download_media_bytes = fake_download
        media_urls, media_types = await adapter._collect_media_from_activity(activity)

        self.assertEqual(len(media_urls), 1)
        self.assertTrue(media_urls[0].endswith('.png'))
        self.assertEqual(Path(media_urls[0]).read_bytes(), png_bytes)
        self.assertEqual(media_types, ["image/png"])

    async def test_send_threads_all_chunks_and_tracks_each_chunk_id(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []
                self.counter = 0

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.counter += 1
                message_id = f"sent-{self.counter}"
                self.calls.append({"conversation_id": ref.conversation_id, "content": content, "reply_to": reply_to, "id": message_id, "entities": entities, "attachments": attachments})
                return {"id": message_id}

        adapter = self._adapter(group_policy="open", text_chunk_limit=5)
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-8",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-2",
                reply_style="thread",
            )
        )

        result = await adapter.send("conv-8", "1234567890AB")

        self.assertTrue(result.success)
        self.assertGreaterEqual(len(adapter._bot.calls), 2)
        self.assertTrue(all(call["reply_to"] == "user-anchor-2" for call in adapter._bot.calls))
        self.assertTrue(adapter._conversations.has_sent_activity("conv-8", "sent-1"))
        self.assertTrue(adapter._conversations.has_sent_activity("conv-8", adapter._bot.calls[-1]["id"]))

    async def test_chunk_mode_newline_prefers_paragraph_boundaries(self):
        adapter = self._adapter(chunk_mode="newline", text_chunk_limit=24)

        chunks = adapter._chunk_text_for_send("alpha beta\n\nsecond para\n\nthird para")

        self.assertEqual(chunks, ["alpha beta", "second para", "third para"])

    async def test_text_chunk_limit_is_clamped_to_platform_max(self):
        adapter = self._adapter(text_chunk_limit=99999)

        self.assertEqual(adapter._text_chunk_limit, adapter.MAX_MESSAGE_LENGTH)

    async def test_send_builds_mention_entities_from_metadata(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "entities": entities, "attachments": attachments})
                return {"id": "sent-mention-1"}

        adapter = self._adapter(group_policy="open")
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-mention",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-3",
                reply_style="thread",
            )
        )

        result = await adapter.send(
            "conv-mention",
            "Hello @[Sean](aad-1)",
            metadata={"mentions": [{"id": "aad-1", "name": "Sean"}]},
        )

        self.assertTrue(result.success)
        call = adapter._bot.calls[0]
        self.assertIn("<at>Sean</at>", call["content"])
        self.assertIsNotNone(call["entities"])
        self.assertEqual(call["entities"][0]["type"], "mention")
        self.assertEqual(call["entities"][0]["mentioned"]["id"], "aad-1")
        self.assertEqual(call["entities"][-1]["additionalType"], ["AIGeneratedContent"])
        self.assertIsNone(call["attachments"])

    async def test_send_resolves_mention_without_id_via_graph_lookup(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "entities": entities, "attachments": attachments})
                return {"id": "sent-mention-2"}

        class StubGraph:
            async def search_users(self, query, limit=1):
                return [{"id": "aad-lookup-1", "displayName": "Taylor User"}]

        adapter = self._adapter(group_policy="open")
        adapter._bot = StubBot()
        adapter._graph = StubGraph()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-mention-lookup",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-33",
                reply_style="thread",
            )
        )

        result = await adapter.send(
            "conv-mention-lookup",
            "Hello @[Taylor User](aad-lookup-1)",
            metadata={"mentions": [{"name": "Taylor User", "query": "Taylor User"}]},
        )

        self.assertTrue(result.success)
        call = adapter._bot.calls[0]
        self.assertEqual(call["entities"][0]["mentioned"]["id"], "aad-lookup-1")
        self.assertEqual(call["entities"][0]["mentioned"]["name"], "Taylor User")

    async def test_get_member_info_uses_graph_client(self):
        class StubGraph:
            async def get_member_info(self, user_id):
                return {"user": {"id": user_id, "displayName": "Sean Shu"}}

        adapter = self._adapter()
        adapter._graph = StubGraph()

        result = await adapter.get_member_info("aad-member-1")

        self.assertEqual(result["user"]["id"], "aad-member-1")
        self.assertEqual(result["user"]["displayName"], "Sean Shu")

    async def test_enrich_event_history_prefixes_thread_context(self):
        class StubGraph:
            async def build_thread_context(self, team_id, channel_id, message_id, limit=20):
                return "[Thread context — prior messages in this thread (not yet in conversation history):]\n[thread parent] Parent: earlier\n[End of thread context]\n\n"

        adapter = self._adapter(group_policy="allowlist", teams={"team-1": {"channels": {"channel-1": {"requireMention": False}}}})
        adapter._graph = StubGraph()
        activity = self._activity(
            conversation_type="channel",
            conversation_id="conv-history",
            text="current message",
            reply_to_id="root-1",
            team_id="team-1",
            channel_id="channel-1",
            team_name="Team 1",
            channel_name="General",
        )
        event = adapter._build_event(activity)

        enriched = await adapter._enrich_event_history(event, activity)

        self.assertTrue(enriched.text.startswith("[Thread context — prior messages in this thread"))
        self.assertIn("current message", enriched.text)

    async def test_enrich_new_session_history_prefixes_group_context(self):
        class StubGraph:
            async def build_recent_chat_context(self, chat_id, **kwargs):
                return "[Recent Teams context — prior messages not yet in conversation history:]\nTaylor: earlier group message\n[End of recent Teams context]\n\n"

        adapter = self._adapter(group_policy="open", history_limit=5)
        adapter._graph = StubGraph()
        activity = self._activity(
            conversation_type="groupChat",
            conversation_id="conv-group-history",
            text="latest message",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        event = adapter._build_event(activity)

        enriched = await adapter.enrich_new_session_history(event)

        self.assertTrue(enriched.text.startswith("[Recent Teams context"))
        self.assertIn("latest message", enriched.text)

    async def test_enrich_new_session_history_keeps_wildcard_group_allowlist_unrestricted(self):
        class StubGraph:
            def __init__(self):
                self.kwargs = None

            async def build_recent_chat_context(self, chat_id, **kwargs):
                self.kwargs = kwargs
                return "[Recent Teams context — prior messages not yet in conversation history:]\nTaylor: earlier group message\n[End of recent Teams context]\n\n"

        adapter = self._adapter(group_policy="allowlist", group_allow_from=["*"], history_limit=5)
        adapter._graph = StubGraph()
        activity = self._activity(
            conversation_type="groupChat",
            conversation_id="conv-group-history-wildcard",
            text="latest message",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        event = adapter._build_event(activity)

        enriched = await adapter.enrich_new_session_history(event)

        self.assertTrue(enriched.text.startswith("[Recent Teams context"))
        self.assertIn("latest message", enriched.text)
        self.assertIsNotNone(adapter._graph.kwargs)
        self.assertIsNone(adapter._graph.kwargs.get("allowed_sender_ids"))

    async def test_enrich_new_session_history_prefixes_dm_context(self):
        class StubGraph:
            async def build_recent_chat_context(self, chat_id, **kwargs):
                return "[Recent Teams context — prior messages not yet in conversation history:]\nSean: earlier dm message\n[End of recent Teams context]\n\n"

        adapter = self._adapter(dm_policy="open", dm_history_limit=3)
        adapter._graph = StubGraph()
        activity = self._activity(conversation_type="personal", conversation_id="conv-dm-history", text="new dm turn")
        event = adapter._build_event(activity)

        enriched = await adapter.enrich_new_session_history(event)

        self.assertTrue(enriched.text.startswith("[Recent Teams context"))
        self.assertIn("new dm turn", enriched.text)

    async def test_enrich_new_session_history_skips_commands(self):
        class StubGraph:
            async def build_recent_chat_context(self, chat_id, **kwargs):
                raise AssertionError("history should not be fetched for commands")

        adapter = self._adapter(group_policy="open", history_limit=5)
        adapter._graph = StubGraph()
        activity = self._activity(
            conversation_type="groupChat",
            conversation_id="conv-group-cmd",
            text="<at>Hermes</at> /new",
            entities=[{"type": "mention", "mentioned": {"id": "bot-1", "name": "Hermes"}}],
        )
        event = adapter._build_event(activity)

        enriched = await adapter.enrich_new_session_history(event)

        self.assertEqual(enriched.text, "/new")

    async def test_send_adaptive_card_uses_single_attachment_message(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "entities": entities, "attachments": attachments})
                return {"id": "sent-card-1"}

        adapter = self._adapter(group_policy="open", text_chunk_limit=5)
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-card",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-4",
                reply_style="thread",
            )
        )
        card = {
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [{"type": "TextBlock", "text": "Hello"}],
        }

        result = await adapter.send("conv-card", "ignored for chunking", metadata={"adaptive_card": card})

        self.assertTrue(result.success)
        self.assertEqual(len(adapter._bot.calls), 1)
        call = adapter._bot.calls[0]
        self.assertEqual(call["reply_to"], "user-anchor-4")
        self.assertEqual(call["content"], "")
        self.assertIsNone(call["entities"])
        self.assertEqual(call["attachments"][0]["contentType"], "application/vnd.microsoft.card.adaptive")
        self.assertEqual(call["attachments"][0]["content"], card)

    async def test_send_poll_builds_adaptive_card_payload(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "entities": entities, "attachments": attachments})
                return {"id": "sent-poll-1"}

        adapter = self._adapter(group_policy="open")
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-poll",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="user-anchor-5",
                reply_style="thread",
            )
        )

        result = await adapter.send(
            "conv-poll",
            "",
            metadata={
                "poll": {
                    "question": "Which option?",
                    "options": ["A", "B", "C"],
                    "max_selections": 2,
                    "poll_id": "poll-123",
                }
            },
        )

        self.assertTrue(result.success)
        self.assertEqual(len(adapter._bot.calls), 1)
        card = adapter._bot.calls[0]["attachments"][0]["content"]
        self.assertEqual(card["type"], "AdaptiveCard")
        self.assertEqual(card["body"][1]["id"], "choices")
        self.assertTrue(card["body"][1]["isMultiSelect"])
        self.assertEqual(card["actions"][0]["data"]["pollId"], "poll-123")

    async def test_send_returns_error_for_unknown_conversation(self):
        adapter = self._adapter()

        result = await adapter.send("missing", "hello")

        self.assertFalse(result.success)
        self.assertIn("Unknown Teams conversation", result.error)

    async def test_send_typing_uses_bot_client_when_conversation_known(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_typing(self, ref):
                self.calls.append(ref.conversation_id)
                return {"id": "typing-1"}

        adapter = self._adapter()
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-typing",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
            )
        )

        await adapter.send_typing("conv-typing")

        self.assertEqual(adapter._bot.calls, ["conv-typing"])

    async def test_send_image_file_sends_inline_attachment(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "attachments": attachments})
                return {"id": "img-1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            adapter = self._adapter()
            adapter._bot = StubBot()
            adapter._conversations.remember(
                ConversationRef(
                    conversation_id="conv-image",
                    service_url="https://smba.trafficmanager.net/amer/",
                    conversation_type="groupChat",
                    chat_type="group",
                    activity_id="anchor-image",
                    reply_style="thread",
                )
            )

            result = await adapter.send_image_file("conv-image", str(image_path), caption="see image")

        self.assertTrue(result.success)
        call = adapter._bot.calls[0]
        self.assertEqual(call["reply_to"], "anchor-image")
        self.assertEqual(call["content"], "see image")
        self.assertEqual(call["attachments"][0]["name"], "sample.png")
        self.assertTrue(call["attachments"][0]["contentUrl"].startswith("data:image/png;base64,"))

    async def test_send_image_file_returns_missing_error_for_unknown_file(self):
        adapter = self._adapter()

        result = await adapter.send_image_file("conv-image", "/does/not/exist.png")

        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    async def test_send_document_sends_inline_attachment(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "attachments": attachments})
                return {"id": "doc-1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.pdf"
            file_path.write_bytes(b"%PDF-1.4")
            adapter = self._adapter()
            adapter._bot = StubBot()
            adapter._conversations.remember(
                ConversationRef(
                    conversation_id="conv-doc",
                    service_url="https://smba.trafficmanager.net/amer/",
                    conversation_type="groupChat",
                    chat_type="group",
                    activity_id="anchor-doc",
                    reply_style="thread",
                )
            )

            result = await adapter.send_document("conv-doc", str(file_path), caption="see doc", file_name="renamed.pdf")

        self.assertTrue(result.success)
        call = adapter._bot.calls[0]
        self.assertEqual(call["reply_to"], "anchor-doc")
        self.assertEqual(call["content"], "see doc")
        self.assertEqual(call["attachments"][0]["name"], "renamed.pdf")
        self.assertTrue(call["attachments"][0]["contentUrl"].startswith("data:application/pdf;base64,"))

    async def test_send_document_in_dm_uses_file_consent_card(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "attachments": attachments})
                return {"id": "consent-1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.pdf"
            file_path.write_bytes(b"%PDF-1.4")
            adapter = self._adapter()
            adapter._bot = StubBot()
            adapter._conversations.remember(
                ConversationRef(
                    conversation_id="conv-dm-doc",
                    service_url="https://smba.trafficmanager.net/amer/",
                    conversation_type="personal",
                    chat_type="dm",
                    activity_id="anchor-dm-doc",
                    reply_style="thread",
                )
            )

            result = await adapter.send_document("conv-dm-doc", str(file_path), caption="send file")

        self.assertTrue(result.success)
        self.assertIn("pending_upload_id", result.raw_response)
        call = adapter._bot.calls[0]
        self.assertEqual(call["reply_to"], "anchor-dm-doc")
        self.assertEqual(call["attachments"][0]["contentType"], "application/vnd.microsoft.teams.card.file.consent")
        self.assertEqual(call["attachments"][0]["content"]["sizeInBytes"], len(b"%PDF-1.4"))

    async def test_handle_file_consent_accept_uploads_and_posts_file_info_card(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, status=200, text="ok"):
                self.status = status
                self._text = text

            async def text(self):
                return self._text

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class StubHTTPSession:
            def __init__(self):
                self.calls = []

            def put(self, url, data=None, headers=None):
                self.calls.append({"url": url, "data": data, "headers": headers})
                return StubResponse()

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "attachments": attachments})
                return {"id": "uploaded-1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.pdf"
            file_path.write_bytes(b"%PDF-1.4")
            adapter = self._adapter()
            adapter._http = StubHTTPSession()
            adapter._bot = StubBot()
            adapter._conversations.remember(
                ConversationRef(
                    conversation_id="conv-dm-doc",
                    service_url="https://smba.trafficmanager.net/amer/",
                    conversation_type="personal",
                    chat_type="dm",
                    activity_id="anchor-dm-doc",
                    reply_style="thread",
                )
            )
            result = await adapter.send_document("conv-dm-doc", str(file_path), caption="send file")
            upload_id = result.raw_response["pending_upload_id"]

            activity = self._activity(conversation_type="personal", conversation_id="conv-dm-doc")
            activity["type"] = "invoke"
            activity["name"] = "fileConsent/invoke"
            activity["value"] = {
                "action": "accept",
                "uploadInfo": {
                    "uploadUrl": "https://upload.example.test/file",
                    "contentUrl": "https://files.example.test/sample.pdf",
                    "uniqueId": "unique-1",
                    "fileType": "pdf",
                    "name": "sample.pdf",
                },
                "context": {"uploadId": upload_id},
            }

            response = await adapter._handle_file_consent_invoke(activity)

        self.assertEqual(response.status, 200)
        self.assertEqual(adapter._http.calls[0]["url"], "https://upload.example.test/file")
        self.assertEqual(adapter._http.calls[0]["data"], b"%PDF-1.4")
        attachment = adapter._bot.calls[-1]["attachments"][0]
        self.assertEqual(attachment["contentType"], "application/vnd.microsoft.teams.card.file.info")
        self.assertEqual(attachment["contentUrl"], "https://files.example.test/sample.pdf")
        self.assertNotIn(upload_id, adapter._pending_uploads)

    async def test_send_document_in_group_uses_sharepoint_upload_when_configured(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                self.calls.append({"content": content, "reply_to": reply_to, "attachments": attachments})
                return {"id": "sharepoint-1"}

        class StubGraph:
            def __init__(self):
                self.uploads = []
                self.links = []

            async def upload_file_to_sharepoint(self, site_id, file_name, data, *, content_type="application/octet-stream", folder="HermesShared"):
                self.uploads.append({"site_id": site_id, "file_name": file_name, "data": data, "content_type": content_type, "folder": folder})
                return {"id": "drive-item-1", "name": file_name, "webDavUrl": "https://sharepoint.example.test/file.pdf", "eTag": '"etag,unique-2"'}

            async def create_sharepoint_link(self, site_id, item_id, *, scope="organization", link_type="view"):
                self.links.append({"site_id": site_id, "item_id": item_id, "scope": scope, "link_type": link_type})
                return {"link": {"webUrl": "https://sharepoint.example.test/share/file.pdf"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.pdf"
            file_path.write_bytes(b"%PDF-1.4")
            adapter = self._adapter(share_point_site_id="tenant.sharepoint.com,site-guid,drive-guid")
            adapter._bot = StubBot()
            adapter._graph = StubGraph()
            adapter._conversations.remember(
                ConversationRef(
                    conversation_id="conv-group-doc",
                    service_url="https://smba.trafficmanager.net/amer/",
                    conversation_type="groupChat",
                    chat_type="group",
                    activity_id="anchor-group-doc",
                    reply_style="thread",
                )
            )

            result = await adapter.send_document("conv-group-doc", str(file_path), caption="share file")

        self.assertTrue(result.success)
        self.assertEqual(adapter._graph.uploads[0]["site_id"], "tenant.sharepoint.com,site-guid,drive-guid")
        self.assertEqual(adapter._bot.calls[0]["reply_to"], "anchor-group-doc")
        attachment = adapter._bot.calls[0]["attachments"][0]
        self.assertEqual(attachment["contentType"], "application/vnd.microsoft.teams.card.file.info")
        self.assertEqual(attachment["contentUrl"], "https://sharepoint.example.test/file.pdf")

    async def test_delete_message_uses_bot_client(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def delete_message(self, ref, message_id):
                self.calls.append({"conversation_id": ref.conversation_id, "message_id": message_id})

        adapter = self._adapter()
        adapter._bot = StubBot()
        adapter._conversations.remember(ConversationRef(conversation_id="conv-delete", service_url="https://smba.trafficmanager.net/amer/", conversation_type="groupChat", chat_type="group"))

        result = await adapter.delete_message("conv-delete", "msg-1")

        self.assertTrue(result.success)
        self.assertEqual(adapter._bot.calls[0]["message_id"], "msg-1")

    async def test_get_message_uses_graph_client(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubGraph:
            async def get_message(self, ref, message_id):
                return {"id": message_id, "text": "hello", "createdAt": "2026-01-01T00:00:00Z"}

        adapter = self._adapter()
        adapter._graph = StubGraph()
        adapter._conversations.remember(ConversationRef(conversation_id="conv-read", service_url="https://smba.trafficmanager.net/amer/", conversation_type="groupChat", chat_type="group"))

        result = await adapter.get_message("conv-read", "msg-1")

        self.assertEqual(result["id"], "msg-1")
        self.assertEqual(result["text"], "hello")

    async def test_pin_search_and_reaction_actions_use_graph_client(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubGraph:
            def __init__(self):
                self.calls = []

            async def pin_message(self, ref, message_id):
                self.calls.append(("pin", ref.conversation_id, message_id))
                return {"ok": True, "pinnedMessageId": "pin-1"}

            async def list_pins(self, ref):
                self.calls.append(("list_pins", ref.conversation_id))
                return {"pins": [{"id": "pin-1", "messageId": "msg-1"}]}

            async def unpin_message(self, ref, pinned_message_id):
                self.calls.append(("unpin", ref.conversation_id, pinned_message_id))
                return {"ok": True}

            async def set_reaction(self, ref, message_id, reaction_type):
                self.calls.append(("react", ref.conversation_id, message_id, reaction_type))
                return {"ok": True}

            async def unset_reaction(self, ref, message_id, reaction_type):
                self.calls.append(("unreact", ref.conversation_id, message_id, reaction_type))
                return {"ok": True}

            async def list_reactions(self, ref, message_id):
                self.calls.append(("list_reactions", ref.conversation_id, message_id))
                return {"reactions": [{"reactionType": "like", "count": 1, "users": []}]}

            async def search_messages(self, ref, query, *, from_display_name=None, limit=25):
                self.calls.append(("search", ref.conversation_id, query, from_display_name, limit))
                return {"messages": [{"id": "msg-2"}]}

        adapter = self._adapter()
        adapter._graph = StubGraph()
        adapter._conversations.remember(ConversationRef(conversation_id="conv-actions", service_url="https://smba.trafficmanager.net/amer/", conversation_type="groupChat", chat_type="group"))

        pin_result = await adapter.pin_message("conv-actions", "msg-1")
        pins = await adapter.list_pins("conv-actions")
        unpin_result = await adapter.unpin_message("conv-actions", "pin-1")
        react_result = await adapter.add_reaction("conv-actions", "msg-1", "like")
        unreact_result = await adapter.remove_reaction("conv-actions", "msg-1", "like")
        reactions = await adapter.list_reactions("conv-actions", "msg-1")
        search = await adapter.search_messages("conv-actions", "hello", from_display_name="Sean", limit=10)

        self.assertTrue(pin_result["ok"])
        self.assertEqual(pins["pins"][0]["id"], "pin-1")
        self.assertTrue(unpin_result["ok"])
        self.assertTrue(react_result["ok"])
        self.assertTrue(unreact_result["ok"])
        self.assertEqual(reactions["reactions"][0]["reactionType"], "like")
        self.assertEqual(search["messages"][0]["id"], "msg-2")

    async def test_list_channels_and_get_channel_info_use_graph_client(self):
        class StubGraph:
            async def list_channels(self, team_id):
                return {"channels": [{"id": "channel-1", "displayName": "General"}], "truncated": False}

            async def get_channel_info(self, team_id, channel_id):
                return {"channel": {"id": channel_id, "displayName": "General", "webUrl": "https://teams.example.test/channel"}}

        adapter = self._adapter()
        adapter._graph = StubGraph()

        channels = await adapter.list_channels("team-1")
        channel = await adapter.get_channel_info("team-1", "channel-1")

        self.assertEqual(channels["channels"][0]["displayName"], "General")
        self.assertEqual(channel["channel"]["id"], "channel-1")

    async def test_send_voice_and_video_delegate_to_document_flow(self):
        adapter = self._adapter()

        with patch.object(adapter, "send_document", side_effect=[MagicMock(success=True), MagicMock(success=True)]) as send_document:
            voice_result = await adapter.send_voice("conv-1", "/tmp/sample.wav", caption="voice")
            video_result = await adapter.send_video("conv-1", "/tmp/sample.mp4", caption="video")

        self.assertTrue(voice_result.success)
        self.assertTrue(video_result.success)
        self.assertEqual(send_document.call_count, 2)

    async def test_send_document_returns_missing_error_for_unknown_file(self):
        adapter = self._adapter()

        result = await adapter.send_document("conv-doc", "/does/not/exist.pdf")

        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    async def test_edit_message_updates_existing_message(self):
        from gateway.platforms.msteams_state import ConversationRef

        class StubBot:
            def __init__(self):
                self.calls = []

            async def update_message(self, ref, message_id, content):
                self.calls.append({"conversation_id": ref.conversation_id, "message_id": message_id, "content": content})
                return {"id": message_id}

        adapter = self._adapter()
        adapter._bot = StubBot()
        adapter._conversations.remember(
            ConversationRef(
                conversation_id="conv-edit",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
            )
        )

        result = await adapter.edit_message("conv-edit", "msg-1", "updated content")

        self.assertTrue(result.success)
        self.assertEqual(adapter._bot.calls[0]["message_id"], "msg-1")
        self.assertEqual(adapter._bot.calls[0]["content"], "updated content")

    async def test_process_event_serializes_same_conversation(self):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource

        adapter = self._adapter()
        order = []
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fake_handle_message(event):
            order.append(("start", event.source.chat_id, event.message_id))
            if event.message_id == "m1":
                entered.set()
                await release.wait()
            order.append(("end", event.source.chat_id, event.message_id))

        adapter.handle_message = fake_handle_message
        source = SessionSource(platform=Platform.MSTEAMS, chat_id="conv-serial", chat_type="group")
        event1 = MessageEvent(text="one", message_type=MessageType.TEXT, source=source, message_id="m1")
        event2 = MessageEvent(text="two", message_type=MessageType.TEXT, source=source, message_id="m2")

        task1 = asyncio.create_task(adapter._process_event(event1))
        await entered.wait()
        task2 = asyncio.create_task(adapter._process_event(event2))
        await asyncio.sleep(0.05)
        self.assertEqual(order, [("start", "conv-serial", "m1")])
        release.set()
        await asyncio.gather(task1, task2)
        self.assertEqual(
            order,
            [
                ("start", "conv-serial", "m1"),
                ("end", "conv-serial", "m1"),
                ("start", "conv-serial", "m2"),
                ("end", "conv-serial", "m2"),
            ],
        )

    async def test_process_event_allows_parallel_different_conversations(self):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import SessionSource

        adapter = self._adapter()
        starts = []
        release = asyncio.Event()

        async def fake_handle_message(event):
            starts.append((event.source.chat_id, event.message_id))
            await release.wait()

        adapter.handle_message = fake_handle_message
        source1 = SessionSource(platform=Platform.MSTEAMS, chat_id="conv-a", chat_type="group")
        source2 = SessionSource(platform=Platform.MSTEAMS, chat_id="conv-b", chat_type="group")
        event1 = MessageEvent(text="one", message_type=MessageType.TEXT, source=source1, message_id="m1")
        event2 = MessageEvent(text="two", message_type=MessageType.TEXT, source=source2, message_id="m2")

        task1 = asyncio.create_task(adapter._process_event(event1))
        task2 = asyncio.create_task(adapter._process_event(event2))
        await asyncio.sleep(0.05)
        self.assertCountEqual(starts, [("conv-a", "m1"), ("conv-b", "m2")])
        release.set()
        await asyncio.gather(task1, task2)


class TestMSTeamsStatePersistence(unittest.TestCase):
    def test_conversation_registry_roundtrips_to_disk(self):
        from gateway.platforms.msteams_state import ConversationRef, ConversationRegistry

        registry = ConversationRegistry()
        registry.remember(
            ConversationRef(
                conversation_id="conv-persist",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="anchor-1",
                tenant_id="tenant-1",
                user_id="aad-1",
                chat_name="Persisted Chat",
                sent_activity_ids={"sent-1", "sent-2"},
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "msteams-state.json"
            registry.save_to_path(state_path)
            loaded = ConversationRegistry.load_from_path(state_path)

        ref = loaded.get("conv-persist")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.service_url, "https://smba.trafficmanager.net/amer/")
        self.assertTrue(loaded.has_sent_activity("conv-persist", "sent-1"))
        self.assertEqual(ref.chat_name, "Persisted Chat")


class TestMSTeamsBotClient(unittest.IsolatedAsyncioTestCase):
    async def test_get_token_uses_real_oauth_endpoint_and_caches_token(self):
        from gateway.platforms.msteams_graph import MSTeamsBotClient

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.calls = []

            def post(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "post", "url": url, "data": data, "json": json, "headers": headers})
                if data is not None:
                    return StubResponse({"access_token": "token-1", "expires_in": 3600})
                return StubResponse({"id": "sent-1"})

            def put(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "put", "url": url, "data": data, "json": json, "headers": headers})
                return StubResponse({"id": "updated-1"})

        session = StubSession()
        client = MSTeamsBotClient("app-id", "secret", "tenant-id", session)

        token1 = await client._get_token()
        token2 = await client._get_token()

        self.assertEqual(token1, "token-1")
        self.assertEqual(token2, "token-1")
        self.assertEqual(len([c for c in session.calls if c["data"] is not None]), 1)
        self.assertEqual(
            session.calls[0]["url"],
            "https://login.microsoftonline.com/tenant-id/oauth2/v2.0/token",
        )
        self.assertEqual(session.calls[0]["data"]["client_id"], "app-id")
        self.assertEqual(session.calls[0]["data"]["client_secret"], "secret")

    async def test_send_message_posts_reply_to_reply_endpoint(self):
        from gateway.platforms.msteams_graph import MSTeamsBotClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.calls = []

            def post(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "post", "url": url, "data": data, "json": json, "headers": headers})
                if data is not None:
                    return StubResponse({"access_token": "token-1", "expires_in": 3600})
                return StubResponse({"id": "reply-1"})

            def put(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "put", "url": url, "data": data, "json": json, "headers": headers})
                return StubResponse({"id": "updated-1"})

        session = StubSession()
        client = MSTeamsBotClient("app-id", "secret", "tenant-id", session)
        ref = ConversationRef(
            conversation_id="conv-1",
            service_url="https://smba.trafficmanager.net/amer/",
            conversation_type="groupChat",
            chat_type="group",
        )

        result = await client.send_message(ref, "hello", reply_to="anchor-1")

        self.assertEqual(result["id"], "reply-1")
        self.assertEqual(
            session.calls[-1]["url"],
            "https://smba.trafficmanager.net/amer/v3/conversations/conv-1/activities/anchor-1",
        )
        self.assertEqual(session.calls[-1]["json"]["type"], "message")

    async def test_send_message_preserves_markdown_without_forcing_xml_text_format(self):
        from gateway.platforms.msteams_graph import MSTeamsBotClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.calls = []

            def post(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "post", "url": url, "data": data, "json": json, "headers": headers})
                if data is not None:
                    return StubResponse({"access_token": "***", "expires_in": 3600})
                return StubResponse({"id": "sent-markdown-1"})

            def put(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "put", "url": url, "data": data, "json": json, "headers": headers})
                return StubResponse({"id": "updated-1"})

        session = StubSession()
        client = MSTeamsBotClient("app-id", "secret", "tenant-id", session)
        ref = ConversationRef(
            conversation_id="conv-markdown",
            service_url="https://smba.trafficmanager.net/amer/",
            conversation_type="groupChat",
            chat_type="group",
        )

        markdown_text = "**bold**\n\n`code` and [link](https://example.com)"
        result = await client.send_message(ref, markdown_text)

        self.assertEqual(result["id"], "sent-markdown-1")
        payload = session.calls[-1]["json"]
        self.assertEqual(payload["text"], markdown_text)
        self.assertNotIn("textFormat", payload)

    async def test_send_message_with_mentions_keeps_openclaw_style_payload(self):
        from gateway.platforms.msteams_graph import MSTeamsBotClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.calls = []

            def post(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "post", "url": url, "data": data, "json": json, "headers": headers})
                if data is not None:
                    return StubResponse({"access_token": "***", "expires_in": 3600})
                return StubResponse({"id": "sent-mention-openclaw-1"})

            def put(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "put", "url": url, "data": data, "json": json, "headers": headers})
                return StubResponse({"id": "updated-1"})

        session = StubSession()
        client = MSTeamsBotClient("app-id", "secret", "tenant-id", session)
        ref = ConversationRef(
            conversation_id="conv-mention-openclaw",
            service_url="https://smba.trafficmanager.net/amer/",
            conversation_type="groupChat",
            chat_type="group",
        )

        entities = [{"type": "mention", "text": "<at>Sean</at>", "mentioned": {"id": "aad-1", "name": "Sean"}}]
        result = await client.send_message(ref, "Hello <at>Sean</at> and **team**", entities=entities)

        self.assertEqual(result["id"], "sent-mention-openclaw-1")
        payload = session.calls[-1]["json"]
        self.assertEqual(payload["text"], "Hello <at>Sean</at> and **team**")
        self.assertEqual(payload["entities"], entities)
        self.assertNotIn("textFormat", payload)

    async def test_update_message_uses_put_activity_endpoint(self):
        from gateway.platforms.msteams_graph import MSTeamsBotClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.calls = []

            def post(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "post", "url": url, "data": data, "json": json, "headers": headers})
                if data is not None:
                    return StubResponse({"access_token": "token-1", "expires_in": 3600})
                return StubResponse({"id": "sent-1"})

            def put(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "put", "url": url, "data": data, "json": json, "headers": headers})
                return StubResponse({"id": "updated-1"})

        session = StubSession()
        client = MSTeamsBotClient("app-id", "secret", "tenant-id", session)
        ref = ConversationRef(
            conversation_id="conv-2",
            service_url="https://smba.trafficmanager.net/amer/",
            conversation_type="groupChat",
            chat_type="group",
        )

        result = await client.update_message(ref, "msg-1", "edited")

        self.assertEqual(result["id"], "updated-1")
        self.assertEqual(session.calls[-1]["method"], "put")
        self.assertEqual(session.calls[-1]["url"], "https://smba.trafficmanager.net/amer/v3/conversations/conv-2/activities/msg-1")

    async def test_update_message_preserves_markdown_without_forcing_xml_text_format(self):
        from gateway.platforms.msteams_graph import MSTeamsBotClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.calls = []

            def post(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "post", "url": url, "data": data, "json": json, "headers": headers})
                if data is not None:
                    return StubResponse({"access_token": "***", "expires_in": 3600})
                return StubResponse({"id": "sent-1"})

            def put(self, url, data=None, json=None, headers=None):
                self.calls.append({"method": "put", "url": url, "data": data, "json": json, "headers": headers})
                return StubResponse({"id": "updated-markdown-1"})

        session = StubSession()
        client = MSTeamsBotClient("app-id", "secret", "tenant-id", session)
        ref = ConversationRef(
            conversation_id="conv-update-markdown",
            service_url="https://smba.trafficmanager.net/amer/",
            conversation_type="groupChat",
            chat_type="group",
        )

        markdown_text = "Updated **bold**\n\n`code`"
        result = await client.update_message(ref, "msg-markdown-1", markdown_text)

        self.assertEqual(result["id"], "updated-markdown-1")
        payload = session.calls[-1]["json"]
        self.assertEqual(payload["text"], markdown_text)
        self.assertNotIn("textFormat", payload)


class TestBotFrameworkJWTValidator(unittest.IsolatedAsyncioTestCase):
    async def test_validate_accepts_valid_rs256_token(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        from gateway.platforms.msteams import BotFrameworkJWTValidator

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_numbers = private_key.public_key().public_numbers()

        def _b64url_uint(value: int) -> str:
            import base64

            raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

        jwks = {
            "keys": [{
                "kty": "RSA",
                "use": "sig",
                "kid": "kid-1",
                "alg": "RS256",
                "n": _b64url_uint(public_numbers.n),
                "e": _b64url_uint(public_numbers.e),
            }]
        }
        now = int(time.time())
        token = __import__("jwt").encode(
            {
                "iss": "https://api.botframework.com",
                "aud": "app-id",
                "nbf": now,
                "exp": now + 3600,
                "serviceurl": "https://smba.trafficmanager.net/amer/",
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "kid-1"},
        )

        class StubResponse:
            def __init__(self, payload):
                self._payload = payload
                self.status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.urls = []

            def get(self, url):
                self.urls.append(url)
                if url.endswith("openidconfiguration"):
                    return StubResponse({"jwks_uri": "https://example.test/jwks"})
                if url.endswith("/jwks"):
                    return StubResponse(jwks)
                raise AssertionError(f"Unexpected URL: {url}")

        validator = BotFrameworkJWTValidator("app-id", StubSession())
        payload = await validator.validate(token, service_url="https://smba.trafficmanager.net/amer/")

        self.assertEqual(payload["aud"], "app-id")
        self.assertEqual(payload["iss"], "https://api.botframework.com")


class TestMSTeamsGraphClient(unittest.IsolatedAsyncioTestCase):
    async def test_get_member_info_normalizes_graph_user(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.posts = []
                self.gets = []

            def post(self, url, data=None, json=None, headers=None):
                self.posts.append({"url": url, "data": data})
                return StubResponse({"access_token": "graph-token", "expires_in": 3600})

            def get(self, url, params=None, headers=None):
                self.gets.append({"url": url, "params": params, "headers": headers})
                return StubResponse({
                    "id": "aad-1",
                    "displayName": "Sean Shu",
                    "mail": "sean@example.com",
                    "jobTitle": "Frontend Lead",
                    "userPrincipalName": "sean@example.com",
                    "officeLocation": "Shanghai",
                })

        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", StubSession())
        result = await client.get_member_info("aad-1")

        self.assertEqual(result["user"]["id"], "aad-1")
        self.assertEqual(result["user"]["displayName"], "Sean Shu")
        self.assertEqual(result["user"]["jobTitle"], "Frontend Lead")

    async def test_search_users_uses_consistency_level_for_display_name_queries(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.posts = []
                self.gets = []

            def post(self, url, data=None, json=None, headers=None):
                self.posts.append({"url": url, "data": data})
                return StubResponse({"access_token": "graph-token", "expires_in": 3600})

            def get(self, url, params=None, headers=None):
                self.gets.append({"url": url, "params": params, "headers": headers})
                return StubResponse({"value": [{"id": "aad-2", "displayName": "Taylor", "mail": "", "jobTitle": "", "userPrincipalName": "", "officeLocation": ""}]})

        session = StubSession()
        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", session)
        users = await client.search_users("Taylor", limit=1)

        self.assertEqual(users[0]["id"], "aad-2")
        self.assertEqual(session.gets[-1]["headers"]["ConsistencyLevel"], "eventual")
        self.assertEqual(session.gets[-1]["params"]["$search"], '"Taylor"')

    async def test_build_message_url_candidates_prefers_reply_then_top_level_for_channels(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient

        class StubSession:
            def post(self, *args, **kwargs):
                raise AssertionError("token request should not run")

        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", StubSession())
        activity = {
            "id": "reply-1",
            "replyToId": "root-1",
            "conversation": {"id": "conv-1", "conversationType": "channel"},
            "channelData": {
                "team": {"id": "team-1"},
                "channel": {"id": "channel-1"},
                "messageId": "reply-1",
            },
            "from": {"aadObjectId": "aad-1"},
        }

        urls = client.build_message_url_candidates(activity)

        self.assertEqual(urls[0], "/teams/team-1/channels/channel-1/messages/root-1/replies/reply-1")
        self.assertIn("/teams/team-1/channels/channel-1/messages/reply-1", urls)

    async def test_download_message_media_collects_hosted_and_reference_entries(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient

        class StubResponse:
            def __init__(self, payload=None, status=200, body=b""):
                self._payload = payload if payload is not None else {}
                self.status = status
                self._body = body

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

            async def read(self):
                return self._body

        class StubSession:
            def __init__(self):
                self.posts = []
                self.gets = []

            def post(self, url, data=None, json=None, headers=None):
                self.posts.append({"url": url, "data": data})
                return StubResponse({"access_token": "***", "expires_in": 3600})

            def get(self, url, params=None, headers=None):
                self.gets.append({"url": url, "params": params, "headers": headers})
                if url.endswith('/hostedContents/hosted-1/$value'):
                    return StubResponse(status=200, body=b"hosted-bytes")
                if url.endswith('/hostedContents'):
                    return StubResponse({"value": [{"id": "hosted-1", "contentType": "image/png"}]})
                return StubResponse({
                    "attachments": [{
                        "contentType": "reference",
                        "contentUrl": "https://sharepoint.example.test/file.pdf",
                        "name": "file.pdf",
                    }]
                })

        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", StubSession())
        media = await client.download_message_media('/teams/team-1/channels/channel-1/messages/root-1')

        self.assertEqual(len(media), 2)
        self.assertEqual(media[0]["graph_path"], "/shares/u!aHR0cHM6Ly9zaGFyZXBvaW50LmV4YW1wbGUudGVzdC9maWxlLnBkZg/driveItem/content")
        self.assertEqual(media[1]["data"], b"hosted-bytes")

    async def test_get_message_matches_openclaw_payload_shape(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload=None, status=200):
                self._payload = payload if payload is not None else {}
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.posts = []
                self.gets = []

            def post(self, url, data=None, json=None, headers=None, params=None):
                self.posts.append({"url": url})
                return StubResponse({"access_token": "***", "expires_in": 3600})

            def get(self, url, params=None, headers=None):
                self.gets.append({"url": url, "params": params, "headers": headers})
                return StubResponse({"id": "msg-1", "body": {"content": "hello body"}, "from": {"user": {"displayName": "Sean"}}, "createdDateTime": "2026-01-01T00:00:00Z"})

        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", StubSession())
        ref = ConversationRef(conversation_id="conv-1", service_url="https://smba.trafficmanager.net/amer/", conversation_type="channel", chat_type="channel", team_id="team-1", channel_id="channel-1")

        result = await client.get_message(ref, "msg-1")

        self.assertEqual(result["id"], "msg-1")
        self.assertEqual(result["text"], "hello body")
        self.assertEqual(result["createdAt"], "2026-01-01T00:00:00Z")

    async def test_graph_channel_actions_match_openclaw_payload_shapes(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient
        from gateway.platforms.msteams_state import ConversationRef

        class StubResponse:
            def __init__(self, payload=None, status=200):
                self._payload = payload if payload is not None else {}
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

            async def read(self):
                return b""

        class StubSession:
            def __init__(self):
                self.posts = []
                self.gets = []
                self.deletes = []

            def post(self, url, data=None, json=None, headers=None, params=None):
                self.posts.append({"url": url, "data": data, "json": json, "headers": headers, "params": params})
                if 'oauth2' in url:
                    return StubResponse({"access_token": "***", "expires_in": 3600})
                if url.endswith('/pinnedMessages'):
                    return StubResponse({"id": "pin-1"})
                return StubResponse({})

            def get(self, url, params=None, headers=None):
                self.gets.append({"url": url, "params": params, "headers": headers})
                if url.endswith('/pinnedMessages'):
                    return StubResponse({"value": [{"id": "pin-1", "message": {"id": "msg-1", "body": {"content": "hello"}}}]})
                if '/messages/' in url and not url.endswith('/pinnedMessages') and params is None:
                    return StubResponse({"reactions": [{"reactionType": "like", "user": {"id": "aad-1", "displayName": "Sean"}}]})
                if '/messages' in url and params is not None:
                    return StubResponse({"value": [{"id": "msg-2", "body": {"content": "hello world"}, "from": {"user": {"displayName": "Sean"}}, "createdDateTime": "2026-01-01T00:00:00Z"}]})
                if '/channels' in url and params is not None and params.get('$select') == 'id,displayName,description,membershipType':
                    return StubResponse({"value": [{"id": "channel-1", "displayName": "General", "description": "desc", "membershipType": "standard"}]})
                return StubResponse({"id": "channel-1", "displayName": "General", "description": "desc", "membershipType": "standard", "webUrl": "https://teams.example.test/channel", "createdDateTime": "2026-01-01T00:00:00Z"})

            def delete(self, url, headers=None, params=None):
                self.deletes.append({"url": url, "headers": headers, "params": params})
                return StubResponse({})

        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", StubSession())
        ref = ConversationRef(conversation_id="conv-1", service_url="https://smba.trafficmanager.net/amer/", conversation_type="channel", chat_type="channel", team_id="team-1", channel_id="channel-1")

        pin = await client.pin_message(ref, "msg-1")
        pins = await client.list_pins(ref)
        await client.unpin_message(ref, "pin-1")
        await client.set_reaction(ref, "msg-1", "like")
        await client.unset_reaction(ref, "msg-1", "like")
        reactions = await client.list_reactions(ref, "msg-1")
        search = await client.search_messages(ref, "hello", from_display_name="Sean", limit=10)
        channels = await client.list_channels("team-1")
        channel = await client.get_channel_info("team-1", "channel-1")

        self.assertEqual(pin["pinnedMessageId"], "pin-1")
        self.assertEqual(pins["pins"][0]["messageId"], "msg-1")
        self.assertEqual(reactions["reactions"][0]["count"], 1)
        self.assertEqual(search["messages"][0]["id"], "msg-2")
        self.assertEqual(channels["channels"][0]["id"], "channel-1")
        self.assertEqual(channel["channel"]["webUrl"], "https://teams.example.test/channel")
        self.assertIn('/messages/msg-1/setReaction', client._session.posts[-2]["url"])
        self.assertIn('/messages/msg-1/unsetReaction', client._session.posts[-1]["url"])

    async def test_build_thread_context_formats_parent_and_replies(self):
        from gateway.platforms.msteams_graph import MSTeamsGraphClient, THREAD_CONTEXT_FOOTER, THREAD_CONTEXT_HEADER

        class StubResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                return self._payload

        class StubSession:
            def __init__(self):
                self.posts = []
                self.gets = []

            def post(self, url, data=None, json=None, headers=None):
                self.posts.append({"url": url, "data": data})
                return StubResponse({"access_token": "graph-token", "expires_in": 3600})

            def get(self, url, params=None, headers=None):
                self.gets.append({"url": url, "params": params, "headers": headers})
                if url.endswith("/messages/root-1"):
                    return StubResponse({
                        "id": "root-1",
                        "body": {"content": "<at>Hermes</at> parent body"},
                        "from": {"user": {"displayName": "Parent User"}},
                    })
                return StubResponse({"value": [
                    {"id": "reply-1", "body": {"content": "reply body"}, "from": {"user": {"displayName": "Reply User"}}}
                ]})

        client = MSTeamsGraphClient("app-id", "secret", "tenant-id", StubSession())
        context = await client.build_thread_context("team-1", "channel-1", "root-1")

        self.assertIn(THREAD_CONTEXT_HEADER, context)
        self.assertIn("[thread parent] Parent User: parent body", context)
        self.assertIn("Reply User: reply body", context)
        self.assertIn(THREAD_CONTEXT_FOOTER, context)


class TestSendMessageMSTeams(unittest.IsolatedAsyncioTestCase):
    async def test_send_msteams_reads_persisted_state_file_when_extra_ref_missing(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.msteams_state import ConversationRef, ConversationRegistry
        from tools.send_message_tool import _send_msteams

        registry = ConversationRegistry()
        registry.remember(
            ConversationRef(
                conversation_id="conv-tool",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="anchor-tool",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "teams-state.json"
            registry.save_to_path(state_path)
            pconfig = PlatformConfig(enabled=True, extra={
                "app_id": "app",
                "app_password": "secret",
                "tenant_id": "tenant",
                "state_path": str(state_path),
            })

            class StubBotClient:
                def __init__(self, app_id, app_password, tenant_id, session):
                    self.app_id = app_id

                async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                    return {"id": "sent-tool-1"}

            with patch("gateway.platforms.msteams_graph.MSTeamsBotClient", StubBotClient):
                result = await _send_msteams(pconfig, "conv-tool", "hello from state")

        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], "sent-tool-1")

    async def test_send_to_platform_lets_teams_adapter_own_chunking(self):
        from gateway.config import Platform, PlatformConfig
        from tools.send_message_tool import _send_to_platform

        async def fake_send_msteams(pconfig, chat_id, message, media_files=None, thread_id=None):
            calls.append({"chat_id": chat_id, "message": message, "thread_id": thread_id, "media_files": media_files})
            return {"success": True, "message_id": "sent-direct-1"}

        calls = []
        pconfig = PlatformConfig(enabled=True, extra={"app_id": "app", "app_password": "secret", "tenant_id": "tenant"})
        long_message = "A" * 5000

        with patch("tools.send_message_tool._send_msteams", side_effect=fake_send_msteams):
            result = await _send_to_platform(Platform.MSTEAMS, pconfig, "conv-1", long_message)

        self.assertTrue(result["success"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["message"], long_message)

    async def test_send_to_platform_preserves_teams_image_media_for_native_sender(self):
        from gateway.config import Platform, PlatformConfig
        from tools.send_message_tool import _send_to_platform

        async def fake_send_msteams(pconfig, chat_id, message, media_files=None, thread_id=None):
            calls.append({"chat_id": chat_id, "message": message, "media_files": media_files})
            return {"success": True, "message_id": "sent-direct-2"}

        calls = []
        pconfig = PlatformConfig(enabled=True, extra={"app_id": "app", "app_password": "secret", "tenant_id": "tenant"})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            with patch("tools.send_message_tool._send_msteams", side_effect=fake_send_msteams):
                result = await _send_to_platform(Platform.MSTEAMS, pconfig, "conv-1", "", media_files=[(str(image_path), False)])

        self.assertTrue(result["success"])
        self.assertEqual(calls[0]["media_files"], [(str(image_path), False)])

    async def test_send_msteams_sends_document_media_without_text(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.msteams_state import ConversationRef, ConversationRegistry
        from tools.send_message_tool import _send_msteams

        registry = ConversationRegistry()
        registry.remember(
            ConversationRef(
                conversation_id="conv-doc-tool",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="anchor-doc-tool",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "teams-state.json"
            file_path = Path(tmpdir) / "sample.pdf"
            file_path.write_bytes(b"%PDF-1.4")
            registry.save_to_path(state_path)
            pconfig = PlatformConfig(enabled=True, extra={
                "app_id": "app",
                "app_password": "secret",
                "tenant_id": "tenant",
                "state_path": str(state_path),
            })

            class StubBotClient:
                def __init__(self, app_id, app_password, tenant_id, session):
                    pass

                async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                    return {"id": "doc-tool-1"}

            class StubGraphClient:
                def __init__(self, app_id, app_password, tenant_id, session):
                    pass

            with patch("gateway.platforms.msteams_graph.MSTeamsBotClient", StubBotClient), patch("gateway.platforms.msteams_graph.MSTeamsGraphClient", StubGraphClient):
                result = await _send_msteams(pconfig, "conv-doc-tool", "", media_files=[(str(file_path), False)])

        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], "doc-tool-1")

    async def test_send_msteams_supports_mixed_text_and_document_media(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.msteams_state import ConversationRef, ConversationRegistry
        from tools.send_message_tool import _send_msteams

        registry = ConversationRegistry()
        registry.remember(
            ConversationRef(
                conversation_id="conv-mixed-tool",
                service_url="https://smba.trafficmanager.net/amer/",
                conversation_type="groupChat",
                chat_type="group",
                activity_id="anchor-mixed-tool",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "teams-state.json"
            file_path = Path(tmpdir) / "sample.pdf"
            file_path.write_bytes(b"%PDF-1.4")
            registry.save_to_path(state_path)
            pconfig = PlatformConfig(enabled=True, extra={
                "app_id": "app",
                "app_password": "secret",
                "tenant_id": "tenant",
                "state_path": str(state_path),
            })

            class StubBotClient:
                def __init__(self, app_id, app_password, tenant_id, session):
                    pass

                async def send_message(self, ref, content, *, reply_to=None, entities=None, attachments=None):
                    return {"id": "tool-doc-2"}

            class StubGraphClient:
                def __init__(self, app_id, app_password, tenant_id, session):
                    pass

            with patch("gateway.platforms.msteams_graph.MSTeamsBotClient", StubBotClient), patch("gateway.platforms.msteams_graph.MSTeamsGraphClient", StubGraphClient):
                result = await _send_msteams(pconfig, "conv-mixed-tool", "hello with file", media_files=[(str(file_path), False)])

        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], "tool-doc-2")

    async def test_parse_target_ref_treats_real_teams_conversation_id_as_explicit(self):
        from tools.send_message_tool import _parse_target_ref

        chat_id, thread_id, explicit = _parse_target_ref("msteams", "19:abc@thread.tacv2")

        self.assertEqual(chat_id, "19:abc@thread.tacv2")
        self.assertIsNone(thread_id)
        self.assertTrue(explicit)
