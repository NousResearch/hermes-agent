"""Tests for the Feishu gateway integration."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

try:
    import lark_oapi
    _HAS_LARK_OAPI = True
except ImportError:
    _HAS_LARK_OAPI = False


def _mock_event_dispatcher_builder(mock_handler_class):
    mock_builder = Mock()
    mock_builder.register_p2_im_message_message_read_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_receive_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_reaction_created_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_reaction_deleted_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_card_action_trigger = Mock(return_value=mock_builder)
    mock_builder.build = Mock(return_value=object())
    mock_handler_class.builder = Mock(return_value=mock_builder)
    return mock_builder


class TestPlatformEnum(unittest.TestCase):
    def test_feishu_in_platform_enum(self):
        from gateway.config import Platform

        self.assertEqual(Platform.FEISHU.value, "feishu")


class TestConfigEnvOverrides(unittest.TestCase):
    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
        "FEISHU_CONNECTION_MODE": "websocket",
        "FEISHU_DOMAIN": "feishu",
    }, clear=False)
    def test_feishu_config_loaded_from_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.FEISHU, config.platforms)
        self.assertTrue(config.platforms[Platform.FEISHU].enabled)
        self.assertEqual(config.platforms[Platform.FEISHU].extra["app_id"], "cli_xxx")
        self.assertEqual(config.platforms[Platform.FEISHU].extra["connection_mode"], "websocket")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
        "FEISHU_HOME_CHANNEL": "oc_xxx",
    }, clear=False)
    def test_feishu_home_channel_loaded(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.platforms[Platform.FEISHU].home_channel
        self.assertIsNotNone(home)
        self.assertEqual(home.chat_id, "oc_xxx")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
    }, clear=False)
    def test_feishu_in_connected_platforms(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.FEISHU, config.get_connected_platforms())


class TestGatewayIntegration(unittest.TestCase):
    def test_feishu_in_adapter_factory(self):
        source = Path("gateway/run.py").read_text(encoding="utf-8")
        self.assertIn("Platform.FEISHU", source)
        self.assertIn("FeishuAdapter", source)

    def test_feishu_in_authorization_maps(self):
        source = Path("gateway/run.py").read_text(encoding="utf-8")
        self.assertIn("FEISHU_ALLOWED_USERS", source)
        self.assertIn("FEISHU_ALLOW_ALL_USERS", source)

    def test_feishu_toolset_exists(self):
        from toolsets import TOOLSETS

        self.assertIn("hermes-feishu", TOOLSETS)
        self.assertIn("hermes-feishu", TOOLSETS["hermes-gateway"]["includes"])


class TestFeishuPostParsing(unittest.TestCase):
    def test_parse_post_content_extracts_text_mentions_and_media_refs(self):
        from gateway.platforms.feishu import parse_feishu_post_content

        result = parse_feishu_post_content(
            json.dumps(
                {
                    "en_us": {
                        "title": "Rich message",
                        "content": [
                            [{"tag": "img", "image_key": "img_1", "alt": "diagram"}],
                            [{"tag": "at", "user_name": "Alice", "open_id": "ou_alice"}],
                            [{"tag": "media", "file_key": "file_1", "file_name": "spec.pdf"}],
                        ],
                    }
                }
            )
        )

        self.assertEqual(result.text_content, "Rich message\n[Image: diagram]\n@Alice\n[Attachment: spec.pdf]")
        self.assertEqual(result.image_keys, ["img_1"])
        self.assertEqual(result.mentioned_ids, ["ou_alice"])
        self.assertEqual(len(result.media_refs), 1)
        self.assertEqual(result.media_refs[0].file_key, "file_1")
        self.assertEqual(result.media_refs[0].file_name, "spec.pdf")
        self.assertEqual(result.media_refs[0].resource_type, "file")

    def test_parse_post_content_uses_fallback_when_invalid(self):
        from gateway.platforms.feishu import FALLBACK_POST_TEXT, parse_feishu_post_content

        result = parse_feishu_post_content("not-json")

        self.assertEqual(result.text_content, FALLBACK_POST_TEXT)
        self.assertEqual(result.image_keys, [])
        self.assertEqual(result.media_refs, [])
        self.assertEqual(result.mentioned_ids, [])

    def test_parse_post_content_preserves_rich_text_semantics(self):
        from gateway.platforms.feishu import parse_feishu_post_content

        result = parse_feishu_post_content(
            json.dumps(
                {
                    "en_us": {
                        "title": "Plan *v2*",
                        "content": [
                            [
                                {"tag": "text", "text": "Bold", "style": {"bold": True}},
                                {"tag": "text", "text": " "},
                                {"tag": "text", "text": "Italic", "style": {"italic": True}},
                                {"tag": "text", "text": " "},
                                {"tag": "text", "text": "Code", "style": {"code": True}},
                            ],
                            [{"tag": "text", "text": "line1"}, {"tag": "br"}, {"tag": "text", "text": "line2"}],
                            [{"tag": "hr"}],
                            [{"tag": "code_block", "language": "python", "text": "print('hi')"}],
                        ],
                    }
                }
            )
        )

        self.assertEqual(
            result.text_content,
            "Plan *v2*\n**Bold** *Italic* `Code`\nline1\nline2\n---\n```python\nprint('hi')\n```",
        )


class TestFeishuMessageNormalization(unittest.TestCase):
    def test_normalize_merge_forward_preserves_summary_lines(self):
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

        self.assertEqual(normalized.relation_kind, "merge_forward")
        self.assertEqual(
            normalized.text_content,
            "Sprint recap\n- Alice: Please review PR-128\n- Bob: Ship it",
        )

    def test_normalize_share_chat_exposes_summary_and_metadata(self):
        from gateway.platforms.feishu import normalize_feishu_message

        normalized = normalize_feishu_message(
            message_type="share_chat",
            raw_content=json.dumps(
                {
                    "chat_id": "oc_chat_shared",
                    "chat_name": "Backend Guild",
                }
            ),
        )

        self.assertEqual(normalized.relation_kind, "share_chat")
        self.assertEqual(normalized.text_content, "Shared chat: Backend Guild\nChat ID: oc_chat_shared")
        self.assertEqual(normalized.metadata["chat_id"], "oc_chat_shared")
        self.assertEqual(normalized.metadata["chat_name"], "Backend Guild")

    def test_normalize_interactive_card_preserves_title_body_and_actions(self):
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

        self.assertEqual(normalized.relation_kind, "interactive")
        self.assertEqual(
            normalized.text_content,
            "Build Failed\nService: payments-api\nBranch: main\nView Logs\nRetry\nActions: View Logs, Retry",
        )


class TestFeishuAdapterMessaging(unittest.TestCase):
    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
        "FEISHU_CONNECTION_MODE": "webhook",
        "FEISHU_WEBHOOK_HOST": "127.0.0.1",
        "FEISHU_WEBHOOK_PORT": "9001",
        "FEISHU_WEBHOOK_PATH": "/hook",
    }, clear=True)
    def test_connect_webhook_mode_starts_local_server(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        runner = AsyncMock()
        site = AsyncMock()
        web_module = SimpleNamespace(
            Application=lambda: SimpleNamespace(router=SimpleNamespace(add_post=lambda *_args, **_kwargs: None)),
            AppRunner=lambda _app: runner,
            TCPSite=lambda _runner, host, port: SimpleNamespace(start=site.start, host=host, port=port),
        )

        with (
            patch("gateway.platforms.feishu.FEISHU_AVAILABLE", True),
            patch("gateway.platforms.feishu.FEISHU_WEBHOOK_AVAILABLE", True),
            patch("gateway.platforms.feishu.EventDispatcherHandler") as mock_handler_class,
            patch("gateway.platforms.feishu.acquire_scoped_lock", return_value=(True, None)),
            patch("gateway.platforms.feishu.release_scoped_lock"),
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
            patch("gateway.platforms.feishu.web", web_module),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)
            connected = asyncio.run(adapter.connect())

        self.assertTrue(connected)
        runner.setup.assert_awaited_once()
        site.start.assert_awaited_once()

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_acquires_scoped_lock_and_disconnect_releases_it(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ws_client = SimpleNamespace()

        with (
            patch("gateway.platforms.feishu.FEISHU_AVAILABLE", True),
            patch("gateway.platforms.feishu.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch("gateway.platforms.feishu.lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO", WARNING="WARNING"))),
            patch("gateway.platforms.feishu.EventDispatcherHandler") as mock_handler_class,
            patch("gateway.platforms.feishu.FeishuWSClient", return_value=ws_client),
            patch("gateway.platforms.feishu._run_official_feishu_ws_client"),
            patch("gateway.platforms.feishu.acquire_scoped_lock", return_value=(True, None)) as acquire_lock,
            patch("gateway.platforms.feishu.release_scoped_lock") as release_lock,
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)

            loop = asyncio.new_event_loop()
            future = loop.create_future()
            future.set_result(None)

            class _Loop:
                def run_in_executor(self, *_args, **_kwargs):
                    return future

                def is_closed(self):
                    return False

            try:
                with patch("gateway.platforms.feishu.asyncio.get_running_loop", return_value=_Loop()):
                    connected = asyncio.run(adapter.connect())
                    asyncio.run(adapter.disconnect())
            finally:
                loop.close()

        self.assertTrue(connected)
        self.assertIsNone(adapter._event_handler)
        acquire_lock.assert_called_once_with(
            "feishu-app-id",
            "cli_app",
            metadata={"platform": "feishu"},
        )
        release_lock.assert_called_once_with("feishu-app-id", "cli_app")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_rejects_existing_app_lock(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        with (
            patch("gateway.platforms.feishu.FEISHU_AVAILABLE", True),
            patch("gateway.platforms.feishu.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch(
                "gateway.platforms.feishu.acquire_scoped_lock",
                return_value=(False, {"pid": 4321}),
            ),
        ):
            connected = asyncio.run(adapter.connect())

        self.assertFalse(connected)
        self.assertEqual(adapter.fatal_error_code, "feishu_app_lock")
        self.assertFalse(adapter.fatal_error_retryable)
        self.assertIn("PID 4321", adapter.fatal_error_message)

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_app",
        "FEISHU_APP_SECRET": "secret_app",
    }, clear=True)
    def test_connect_retries_transient_startup_failure(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ws_client = SimpleNamespace()
        sleeps = []

        with (
            patch("gateway.platforms.feishu.FEISHU_AVAILABLE", True),
            patch("gateway.platforms.feishu.FEISHU_WEBSOCKET_AVAILABLE", True),
            patch("gateway.platforms.feishu.lark", SimpleNamespace(LogLevel=SimpleNamespace(INFO="INFO", WARNING="WARNING"))),
            patch("gateway.platforms.feishu.EventDispatcherHandler") as mock_handler_class,
            patch("gateway.platforms.feishu.FeishuWSClient", return_value=ws_client),
            patch("gateway.platforms.feishu.acquire_scoped_lock", return_value=(True, None)),
            patch("gateway.platforms.feishu.release_scoped_lock"),
            patch.object(adapter, "_hydrate_bot_identity", new=AsyncMock()),
            patch("gateway.platforms.feishu.asyncio.sleep", side_effect=lambda delay: sleeps.append(delay)),
            patch.object(adapter, "_build_lark_client", return_value=SimpleNamespace()),
        ):
            _mock_event_dispatcher_builder(mock_handler_class)

            loop = asyncio.new_event_loop()
            future = loop.create_future()
            future.set_result(None)

            class _Loop:
                def __init__(self):
                    self.calls = 0

                def run_in_executor(self, *_args, **_kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        raise OSError("temporary websocket failure")
                    return future

                def is_closed(self):
                    return False

            fake_loop = _Loop()
            try:
                with patch("gateway.platforms.feishu.asyncio.get_running_loop", return_value=fake_loop):
                    connected = asyncio.run(adapter.connect())
            finally:
                loop.close()

        self.assertTrue(connected)
        self.assertEqual(sleeps, [1])
        self.assertEqual(fake_loop.calls, 2)

    @patch.dict(os.environ, {}, clear=True)
    def test_edit_message_updates_existing_feishu_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def update(self, request):
                captured["request"] = request
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="📖 read_file: \"/tmp/image.png\"",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_progress")
        self.assertEqual(captured["request"].message_id, "om_progress")
        self.assertEqual(captured["request"].request_body.msg_type, "text")
        self.assertEqual(
            captured["request"].request_body.content,
            json.dumps({"text": "📖 read_file: \"/tmp/image.png\""}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_edit_message_falls_back_to_text_when_post_update_is_rejected(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"calls": []}

        class _MessageAPI:
            def update(self, request):
                captured["calls"].append(request)
                if len(captured["calls"]) == 1:
                    return SimpleNamespace(success=lambda: False, code=230001, msg="content format of the post type is incorrect")
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.edit_message(
                    chat_id="oc_chat",
                    message_id="om_progress",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["calls"][0].request_body.msg_type, "post")
        self.assertEqual(captured["calls"][1].request_body.msg_type, "text")
        self.assertEqual(
            captured["calls"][1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_get_chat_info_uses_real_feishu_chat_api(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _ChatAPI:
            def get(self, request):
                self.request = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(name="Hermes Group", chat_type="group"),
                )

        chat_api = _ChatAPI()
        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    chat=chat_api,
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            info = asyncio.run(adapter.get_chat_info("oc_chat"))

        self.assertEqual(chat_api.request.chat_id, "oc_chat")
        self.assertEqual(info["chat_id"], "oc_chat")
        self.assertEqual(info["name"], "Hermes Group")
        self.assertEqual(info["type"], "group")

class TestAdapterModule(unittest.TestCase):
    def test_adapter_requirement_helper_exists(self):
        source = Path("gateway/platforms/feishu.py").read_text(encoding="utf-8")
        self.assertIn("def check_feishu_requirements()", source)
        self.assertIn("FEISHU_AVAILABLE", source)

    def test_adapter_declares_websocket_scope(self):
        source = Path("gateway/platforms/feishu.py").read_text(encoding="utf-8")
        self.assertIn("Supported modes: websocket, webhook", source)
        self.assertIn("FEISHU_CONNECTION_MODE", source)

    def test_adapter_registers_message_read_noop_handler(self):
        source = Path("gateway/platforms/feishu.py").read_text(encoding="utf-8")
        self.assertIn("register_p2_im_message_message_read_v1", source)
        self.assertIn("def _on_message_read_event", source)

    def test_adapter_registers_reaction_and_card_handlers_for_websocket(self):
        source = Path("gateway/platforms/feishu.py").read_text(encoding="utf-8")
        self.assertIn("register_p2_im_message_reaction_created_v1", source)
        self.assertIn("register_p2_im_message_reaction_deleted_v1", source)
        self.assertIn("register_p2_card_action_trigger", source)

    def test_load_settings_uses_sdk_defaults_for_invalid_ws_reconnect_values(self):
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_reconnect_nonce": -1,
                "ws_reconnect_interval": "bad",
            }
        )

        self.assertEqual(settings.ws_reconnect_nonce, 30)
        self.assertEqual(settings.ws_reconnect_interval, 120)

    def test_load_settings_accepts_custom_ws_reconnect_values(self):
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_reconnect_nonce": 0,
                "ws_reconnect_interval": 3,
            }
        )

        self.assertEqual(settings.ws_reconnect_nonce, 0)
        self.assertEqual(settings.ws_reconnect_interval, 3)

    def test_load_settings_accepts_custom_ws_ping_values(self):
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_ping_interval": 10,
                "ws_ping_timeout": 8,
            }
        )

        self.assertEqual(settings.ws_ping_interval, 10)
        self.assertEqual(settings.ws_ping_timeout, 8)

    def test_load_settings_ignores_invalid_ws_ping_values(self):
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(
            {
                "ws_ping_interval": 0,
                "ws_ping_timeout": -1,
            }
        )

        self.assertIsNone(settings.ws_ping_interval)
        self.assertIsNone(settings.ws_ping_timeout)

    def test_runtime_ws_overrides_reapply_after_sdk_configure(self):
        import sys
        from types import ModuleType

        class _FakeWSClient:
            def __init__(self):
                self._reconnect_nonce = 30
                self._reconnect_interval = 120
                self._ping_interval = 120
                self.configure_calls = []

            def _configure(self, conf):
                self.configure_calls.append(conf)
                self._reconnect_nonce = conf.ReconnectNonce
                self._reconnect_interval = conf.ReconnectInterval
                self._ping_interval = conf.PingInterval

            def start(self):
                conf = SimpleNamespace(ReconnectNonce=99, ReconnectInterval=88, PingInterval=77)
                self._configure(conf)
                raise RuntimeError("stop test client")

        fake_client = _FakeWSClient()
        fake_adapter = SimpleNamespace(
            _ws_thread_loop=None,
            _ws_reconnect_nonce=2,
            _ws_reconnect_interval=3,
            _ws_ping_interval=4,
            _ws_ping_timeout=5,
        )
        fake_client_module = ModuleType("lark_oapi.ws.client")
        fake_client_module.loop = None
        fake_client_module.websockets = SimpleNamespace(connect=AsyncMock())
        fake_ws_module = ModuleType("lark_oapi.ws")
        fake_ws_module.client = fake_client_module
        fake_root_module = ModuleType("lark_oapi")
        fake_root_module.ws = fake_ws_module

        original_modules = sys.modules.copy()
        sys.modules["lark_oapi"] = fake_root_module
        sys.modules["lark_oapi.ws"] = fake_ws_module
        sys.modules["lark_oapi.ws.client"] = fake_client_module
        try:
            from gateway.platforms.feishu import _run_official_feishu_ws_client

            _run_official_feishu_ws_client(fake_client, fake_adapter)
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

        self.assertEqual(len(fake_client.configure_calls), 1)
        self.assertEqual(fake_client._reconnect_nonce, 2)
        self.assertEqual(fake_client._reconnect_interval, 3)
        self.assertEqual(fake_client._ping_interval, 4)


class TestAdapterBehavior(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_build_event_handler_registers_reaction_and_card_processors(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        calls = []

        class _Builder:
            def register_p2_im_message_message_read_v1(self, _handler):
                calls.append("message_read")
                return self

            def register_p2_im_message_receive_v1(self, _handler):
                calls.append("message_receive")
                return self

            def register_p2_im_message_reaction_created_v1(self, _handler):
                calls.append("reaction_created")
                return self

            def register_p2_im_message_reaction_deleted_v1(self, _handler):
                calls.append("reaction_deleted")
                return self

            def register_p2_card_action_trigger(self, _handler):
                calls.append("card_action")
                return self

            def register_p2_im_chat_member_bot_added_v1(self, _handler):
                calls.append("bot_added")
                return self

            def register_p2_im_chat_member_bot_deleted_v1(self, _handler):
                calls.append("bot_deleted")
                return self

            def build(self):
                calls.append("build")
                return "handler"

        class _Dispatcher:
            @staticmethod
            def builder(_encrypt_key, _verification_token):
                calls.append("builder")
                return _Builder()

        with patch("gateway.platforms.feishu.EventDispatcherHandler", _Dispatcher):
            handler = adapter._build_event_handler()

        self.assertEqual(handler, "handler")
        self.assertEqual(
            calls,
            [
                "builder",
                "message_read",
                "message_receive",
                "reaction_created",
                "reaction_deleted",
                "card_action",
                "bot_added",
                "bot_deleted",
                "build",
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    @unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
    def test_add_ack_reaction_uses_ok_emoji(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ReactionAPI:
            def create(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(reaction_id="r_typing"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message_reaction=_ReactionAPI()))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            reaction_id = asyncio.run(adapter._add_ack_reaction("om_msg"))

        self.assertEqual(reaction_id, "r_typing")
        self.assertEqual(captured["request"].request_body.reaction_type["emoji_type"], "OK")

    @patch.dict(os.environ, {}, clear=True)
    def test_add_ack_reaction_logs_warning_on_failure(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _ReactionAPI:
            def create(self, request):
                raise RuntimeError("boom")

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message_reaction=_ReactionAPI()))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct),
            self.assertLogs("gateway.platforms.feishu", level="WARNING") as logs,
        ):
            reaction_id = asyncio.run(adapter._add_ack_reaction("om_msg"))

        self.assertIsNone(reaction_id)
        self.assertTrue(
            any("Failed to add ack reaction to om_msg" in entry for entry in logs.output),
            logs.output,
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_ack_reaction_events_are_ignored_to_avoid_feedback_loops(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._loop = object()
        event = SimpleNamespace(
            message_id="om_msg",
            operator_type="user",
            reaction_type=SimpleNamespace(emoji_type="OK"),
        )
        data = SimpleNamespace(event=event)

        with patch("gateway.platforms.feishu.asyncio.run_coroutine_threadsafe") as run_threadsafe:
            adapter._on_reaction_event("im.message.reaction.created_v1", data)

        run_threadsafe.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_normalize_inbound_text_strips_feishu_mentions(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        cleaned = adapter._normalize_inbound_text("hi @_user_1  there @_user_2")
        self.assertEqual(cleaned, "hi there")

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_requires_mentions_even_when_policy_open(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(mentions=[])
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        self.assertFalse(adapter._should_accept_group_message(message, sender_id, ""))

        message_with_mention = SimpleNamespace(mentions=[SimpleNamespace(key="@_user_1")])
        self.assertFalse(adapter._should_accept_group_message(message_with_mention, sender_id, ""))

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_with_other_user_mention_is_rejected_when_bot_identity_unknown(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        other_mention = SimpleNamespace(
            name="Other User",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )

        self.assertFalse(adapter._should_accept_group_message(SimpleNamespace(mentions=[other_mention]), sender_id, ""))

    @patch.dict(
        os.environ,
        {
            "FEISHU_GROUP_POLICY": "allowlist",
            "FEISHU_ALLOWED_USERS": "ou_allowed",
            "FEISHU_BOT_NAME": "Hermes Bot",
        },
        clear=True,
    )
    def test_group_message_allowlist_and_mention_both_required(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        mentioned = SimpleNamespace(
            mentions=[
                SimpleNamespace(
                    name="Hermes Bot",
                    id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
                )
            ]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                mentioned,
                SimpleNamespace(open_id="ou_allowed", user_id=None),
                "",
            )
        )
        self.assertFalse(
            adapter._should_accept_group_message(
                mentioned,
                SimpleNamespace(open_id="ou_blocked", user_id=None),
                "",
            )
        )

    def test_per_group_allowlist_policy_gates_by_sender(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig(
            extra={
                "group_rules": {
                    "oc_chat_a": {
                        "policy": "allowlist",
                        "allowlist": ["ou_alice", "ou_bob"],
                    }
                }
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_alice", user_id=None),
                "oc_chat_a",
            )
        )
        self.assertFalse(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_charlie", user_id=None),
                "oc_chat_a",
            )
        )

    def test_per_group_blacklist_policy_blocks_specific_users(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig(
            extra={
                "group_rules": {
                    "oc_chat_b": {
                        "policy": "blacklist",
                        "blacklist": ["ou_blocked"],
                    }
                }
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_alice", user_id=None),
                "oc_chat_b",
            )
        )
        self.assertFalse(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_blocked", user_id=None),
                "oc_chat_b",
            )
        )

    def test_per_group_admin_only_policy_requires_admin(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig(
            extra={
                "admins": ["ou_admin"],
                "group_rules": {
                    "oc_chat_c": {
                        "policy": "admin_only",
                    }
                },
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_admin", user_id=None),
                "oc_chat_c",
            )
        )
        self.assertFalse(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_regular", user_id=None),
                "oc_chat_c",
            )
        )

    def test_per_group_disabled_policy_blocks_all(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig(
            extra={
                "admins": ["ou_admin"],
                "group_rules": {
                    "oc_chat_d": {
                        "policy": "disabled",
                    }
                },
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_admin", user_id=None),
                "oc_chat_d",
            )
        )
        self.assertFalse(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_regular", user_id=None),
                "oc_chat_d",
            )
        )

    def test_global_admins_bypass_all_group_rules(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig(
            extra={
                "admins": ["ou_admin"],
                "group_rules": {
                    "oc_chat_e": {
                        "policy": "allowlist",
                        "allowlist": ["ou_alice"],
                    }
                },
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_admin", user_id=None),
                "oc_chat_e",
            )
        )

    def test_default_group_policy_fallback_for_chats_without_explicit_rule(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig(
            extra={
                "default_group_policy": "open",
            }
        )
        adapter = FeishuAdapter(config)
        adapter._bot_open_id = "ou_bot"

        message = SimpleNamespace(
            mentions=[SimpleNamespace(name="Bot", id=SimpleNamespace(open_id="ou_bot", user_id=None))]
        )

        self.assertTrue(
            adapter._should_accept_group_message(
                message,
                SimpleNamespace(open_id="ou_anyone", user_id=None),
                "oc_chat_unknown",
            )
        )

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_matches_bot_open_id_when_configured(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._bot_open_id = "ou_bot"
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)

        bot_mention = SimpleNamespace(
            name="Hermes",
            id=SimpleNamespace(open_id="ou_bot", user_id="u_bot"),
        )
        other_mention = SimpleNamespace(
            name="Other",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )

        self.assertTrue(adapter._should_accept_group_message(SimpleNamespace(mentions=[bot_mention]), sender_id, ""))
        self.assertFalse(adapter._should_accept_group_message(SimpleNamespace(mentions=[other_mention]), sender_id, ""))

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_message_matches_bot_name_when_only_name_available(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._bot_name = "Hermes Bot"
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)

        named_mention = SimpleNamespace(
            name="Hermes Bot",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )
        different_mention = SimpleNamespace(
            name="Another Bot",
            id=SimpleNamespace(open_id="ou_other", user_id="u_other"),
        )

        self.assertTrue(adapter._should_accept_group_message(SimpleNamespace(mentions=[named_mention]), sender_id, ""))
        self.assertFalse(adapter._should_accept_group_message(SimpleNamespace(mentions=[different_mention]), sender_id, ""))

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_group_post_message_uses_parsed_mentions_when_sdk_mentions_missing(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._bot_open_id = "ou_bot"
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        message = SimpleNamespace(
            message_type="post",
            mentions=[],
            content='{"en_us":{"content":[[{"tag":"at","user_name":"Hermes","open_id":"ou_bot"}]]}}',
        )

        self.assertTrue(adapter._should_accept_group_message(message, sender_id, ""))

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_as_text(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="post",
            content='{"zh_cn":{"title":"Title","content":[[{"tag":"text","text":"hello "}],[{"tag":"a","text":"doc","href":"https://example.com"}]]}}',
            message_id="om_post",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Title\nhello\n[doc](https://example.com)")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_uses_first_available_language_block(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="post",
            content='{"fr_fr":{"title":"Subject","content":[[{"tag":"text","text":"bonjour"}]]}}',
            message_id="om_post_fr",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Subject\nbonjour")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_with_rich_elements_does_not_drop_content(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="post",
            content=(
                '{"en_us":{"title":"Rich message","content":['
                '[{"tag":"img","alt":"diagram"}],'
                '[{"tag":"at","user_name":"Alice"},{"tag":"text","text":" please check the attachment"}],'
                '[{"tag":"media","file_name":"spec.pdf"}],'
                '[{"tag":"emotion","emoji_type":"smile"}]'
                ']}}'
            ),
            message_id="om_post_rich",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Rich message\n[Image: diagram]\n@Alice please check the attachment\n[Attachment: spec.pdf]\n:smile:")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_post_message_downloads_embedded_resources(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_image = AsyncMock(return_value=("/tmp/feishu-image.png", "image/png"))
        adapter._download_feishu_message_resource = AsyncMock(return_value=("/tmp/spec.pdf", "application/pdf"))
        message = SimpleNamespace(
            message_type="post",
            content=(
                '{"en_us":{"title":"Rich message","content":['
                '[{"tag":"img","image_key":"img_123","alt":"diagram"}],'
                '[{"tag":"media","file_key":"file_123","file_name":"spec.pdf"}]'
                ']}}'
            ),
            message_id="om_post_media",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Rich message\n[Image: diagram]\n[Attachment: spec.pdf]")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, ["/tmp/feishu-image.png", "/tmp/spec.pdf"])
        self.assertEqual(media_types, ["image/png", "application/pdf"])
        adapter._download_feishu_image.assert_awaited_once_with(
            message_id="om_post_media",
            image_key="img_123",
        )
        adapter._download_feishu_message_resource.assert_awaited_once_with(
            message_id="om_post_media",
            file_key="file_123",
            resource_type="file",
            fallback_filename="spec.pdf",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_merge_forward_message_as_text_summary(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="merge_forward",
            content=json.dumps(
                {
                    "title": "Forwarded updates",
                    "messages": [
                        {"sender_name": "Alice", "text": "Investigating the incident"},
                        {"sender_name": "Bob", "text": "ETA 10 minutes"},
                    ],
                }
            ),
            message_id="om_merge_forward",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(
            text,
            "Forwarded updates\n- Alice: Investigating the incident\n- Bob: ETA 10 minutes",
        )
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_share_chat_message_as_text_summary(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="share_chat",
            content='{"chat_id":"oc_shared","chat_name":"Platform Ops"}',
            message_id="om_share_chat",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Shared chat: Platform Ops\nChat ID: oc_shared")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_interactive_message_as_text_summary(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            message_type="interactive",
            content=json.dumps(
                {
                    "card": {
                        "header": {"title": {"tag": "plain_text", "content": "Approval Request"}},
                        "elements": [
                            {"tag": "div", "text": {"tag": "plain_text", "content": "Requester: Alice"}},
                            {
                                "tag": "action",
                                "actions": [
                                    {"tag": "button", "text": {"tag": "plain_text", "content": "Approve"}},
                                ],
                            },
                        ],
                    }
                }
            ),
            message_id="om_interactive",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "Approval Request\nRequester: Alice\nApprove\nActions: Approve")
        self.assertEqual(msg_type.value, "text")
        self.assertEqual(media_urls, [])
        self.assertEqual(media_types, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_image_message_downloads_and_caches(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_image = AsyncMock(return_value=("/tmp/feishu-image.png", "image/png"))
        message = SimpleNamespace(
            message_type="image",
            content='{"image_key":"img_123"}',
            message_id="om_image",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "photo")
        self.assertEqual(media_urls, ["/tmp/feishu-image.png"])
        self.assertEqual(media_types, ["image/png"])
        adapter._download_feishu_image.assert_awaited_once_with(
            message_id="om_image",
            image_key="img_123",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_audio_message_downloads_and_caches(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/feishu-audio.ogg", "audio/ogg")
        )
        message = SimpleNamespace(
            message_type="audio",
            content='{"file_key":"file_audio","file_name":"voice.ogg"}',
            message_id="om_audio",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "audio")
        self.assertEqual(media_urls, ["/tmp/feishu-audio.ogg"])
        self.assertEqual(media_types, ["audio/ogg"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_file_message_downloads_and_caches(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/doc_123_report.pdf", "application/pdf")
        )
        message = SimpleNamespace(
            message_type="file",
            content='{"file_key":"file_doc","file_name":"report.pdf"}',
            message_id="om_file",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "document")
        self.assertEqual(media_urls, ["/tmp/doc_123_report.pdf"])
        self.assertEqual(media_types, ["application/pdf"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_media_message_with_image_mime_becomes_photo(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/feishu-media.jpg", "image/jpeg")
        )
        message = SimpleNamespace(
            message_type="media",
            content='{"file_key":"file_media","file_name":"photo.jpg"}',
            message_id="om_media",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "photo")
        self.assertEqual(media_urls, ["/tmp/feishu-media.jpg"])
        self.assertEqual(media_types, ["image/jpeg"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_media_message_with_video_mime_becomes_video(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._download_feishu_message_resource = AsyncMock(
            return_value=("/tmp/feishu-video.mp4", "video/mp4")
        )
        message = SimpleNamespace(
            message_type="media",
            content='{"file_key":"file_video","file_name":"clip.mp4"}',
            message_id="om_video",
        )

        text, msg_type, media_urls, media_types = asyncio.run(adapter._extract_message_content(message))

        self.assertEqual(text, "")
        self.assertEqual(msg_type.value, "video")
        self.assertEqual(media_urls, ["/tmp/feishu-video.mp4"])
        self.assertEqual(media_types, ["video/mp4"])

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_from_raw_content_uses_relation_message_fallbacks(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        shared = adapter._extract_text_from_raw_content(
            msg_type="share_chat",
            raw_content='{"chat_id":"oc_shared","chat_name":"Platform Ops"}',
        )
        attachment = adapter._extract_text_from_raw_content(
            msg_type="file",
            raw_content='{"file_key":"file_1","file_name":"report.pdf"}',
        )

        self.assertEqual(shared, "Shared chat: Platform Ops\nChat ID: oc_shared")
        self.assertEqual(attachment, "[Attachment: report.pdf]")

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_message_starting_with_slash_becomes_command(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id=None,
            parent_id=None,
            upper_message_id=None,
            message_type="text",
            content='{"text":"/help test"}',
            message_id="om_command",
        )

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                chat_type="p2p",
                message_id="om_command",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.message_type.value, "command")
        self.assertEqual(event.text, "/help test")

    @patch.dict(os.environ, {}, clear=True)
    def test_extract_text_file_injects_content(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("hello from feishu")
            path = tmp.name

        try:
            text = asyncio.run(adapter._maybe_extract_text_document(path, "text/plain"))
        finally:
            os.unlink(path)

        self.assertIn("hello from feishu", text)
        self.assertIn("[Content of", text)

    @patch.dict(os.environ, {}, clear=True)
    def test_message_event_submits_to_adapter_loop(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _Loop:
            def is_closed(self):
                return False

        adapter._loop = _Loop()

        message = SimpleNamespace(
            message_id="om_text",
            chat_type="p2p",
            chat_id="oc_chat",
            message_type="text",
            content='{"text":"hello"}',
        )
        sender_id = SimpleNamespace(open_id="ou_user", user_id=None, union_id=None)
        sender = SimpleNamespace(sender_id=sender_id, sender_type="user")
        data = SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))

        future = SimpleNamespace(add_done_callback=lambda *_args, **_kwargs: None)

        def _submit(coro, _loop):
            coro.close()
            return future

        with patch("gateway.platforms.feishu.asyncio.run_coroutine_threadsafe", side_effect=_submit) as submit:
            adapter._on_message_event(data)

        self.assertTrue(submit.called)

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_request_uses_same_message_dispatch_path(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._on_message_event = Mock()

        body = json.dumps({
            "header": {"event_type": "im.message.receive_v1"},
            "event": {"message": {"message_id": "om_test"}},
        }).encode("utf-8")
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            headers={},
            read=AsyncMock(return_value=body),
        )

        response = asyncio.run(adapter._handle_webhook_request(request))

        self.assertEqual(response.status, 200)
        adapter._on_message_event.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_process_inbound_message_uses_event_sender_identity_only(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageType
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        # Sender name now comes from the contact API; mock it to return a known value.
        adapter._resolve_sender_name_from_api = AsyncMock(return_value="张三")
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
        )
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id=None,
            message_type="text",
            content='{"text":"hello"}',
            message_id="om_text",
        )
        sender_id = SimpleNamespace(
            open_id="ou_user",
            user_id="u_user",
            union_id="on_union",
        )
        data = SimpleNamespace(event=SimpleNamespace(message=message, sender=SimpleNamespace(sender_id=sender_id)))

        asyncio.run(
            adapter._process_inbound_message(
                data=data,
                message=message,
                sender_id=sender_id,
                chat_type="p2p",
                message_id="om_text",
            )
        )

        adapter._dispatch_inbound_event.assert_awaited_once()
        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.message_type, MessageType.TEXT)
        self.assertEqual(event.source.user_id, "ou_user")
        self.assertEqual(event.source.user_name, "张三")
        self.assertEqual(event.source.user_id_alt, "on_union")
        self.assertEqual(event.source.chat_name, "Feishu DM")

    @patch.dict(os.environ, {}, clear=True)
    def test_text_batch_merges_rapid_messages_into_single_event(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.platforms.feishu import FeishuAdapter
        from gateway.session import SessionSource

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()
        source = SessionSource(
            platform=adapter.platform,
            chat_id="oc_chat",
            chat_name="Feishu DM",
            chat_type="dm",
            user_id="ou_user",
            user_name="张三",
        )

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("gateway.platforms.feishu.asyncio.sleep", side_effect=_sleep):
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="A", message_type=MessageType.TEXT, source=source, message_id="om_1")
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="B", message_type=MessageType.TEXT, source=source, message_id="om_2")
                )
                pending = list(adapter._pending_text_batch_tasks.values())
                self.assertEqual(len(pending), 1)
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_run())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.text, "A\nB")
        self.assertEqual(event.message_type, MessageType.TEXT)

    @patch.dict(
        os.environ,
        {
            "HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES": "2",
        },
        clear=True,
    )
    def test_text_batch_flushes_when_message_count_limit_is_hit(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.platforms.feishu import FeishuAdapter
        from gateway.session import SessionSource

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()
        source = SessionSource(
            platform=adapter.platform,
            chat_id="oc_chat",
            chat_name="Feishu DM",
            chat_type="dm",
            user_id="ou_user",
            user_name="张三",
        )

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("gateway.platforms.feishu.asyncio.sleep", side_effect=_sleep):
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="A", message_type=MessageType.TEXT, source=source, message_id="om_1")
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="B", message_type=MessageType.TEXT, source=source, message_id="om_2")
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(text="C", message_type=MessageType.TEXT, source=source, message_id="om_3")
                )
                pending = list(adapter._pending_text_batch_tasks.values())
                self.assertEqual(len(pending), 1)
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_run())

        self.assertEqual(adapter.handle_message.await_count, 2)
        first = adapter.handle_message.await_args_list[0].args[0]
        second = adapter.handle_message.await_args_list[1].args[0]
        self.assertEqual(first.text, "A\nB")
        self.assertEqual(second.text, "C")

    @patch.dict(os.environ, {}, clear=True)
    def test_media_batch_merges_rapid_photo_messages(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.platforms.feishu import FeishuAdapter
        from gateway.session import SessionSource

        adapter = FeishuAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()
        source = SessionSource(
            platform=adapter.platform,
            chat_id="oc_chat",
            chat_name="Feishu DM",
            chat_type="dm",
            user_id="ou_user",
            user_name="张三",
        )

        async def _sleep(_delay):
            return None

        async def _run() -> None:
            with patch("gateway.platforms.feishu.asyncio.sleep", side_effect=_sleep):
                await adapter._dispatch_inbound_event(
                    MessageEvent(
                        text="第一张",
                        message_type=MessageType.PHOTO,
                        source=source,
                        message_id="om_p1",
                        media_urls=["/tmp/a.png"],
                        media_types=["image/png"],
                    )
                )
                await adapter._dispatch_inbound_event(
                    MessageEvent(
                        text="第二张",
                        message_type=MessageType.PHOTO,
                        source=source,
                        message_id="om_p2",
                        media_urls=["/tmp/b.png"],
                        media_types=["image/png"],
                    )
                )
                pending = list(adapter._pending_media_batch_tasks.values())
                self.assertEqual(len(pending), 1)
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_run())

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        self.assertEqual(event.media_urls, ["/tmp/a.png", "/tmp/b.png"])
        self.assertIn("第一张", event.text)
        self.assertIn("第二张", event.text)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_image_downloads_then_uses_native_image_send(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter.send_image_file = AsyncMock(return_value=SimpleNamespace(success=True, message_id="om_img"))

        async def _run():
            with patch("gateway.platforms.feishu.cache_image_from_url", new=AsyncMock(return_value="/tmp/cached.png")):
                return await adapter.send_image("oc_chat", "https://example.com/cat.png", caption="cat")

        result = asyncio.run(_run())

        self.assertTrue(result.success)
        adapter.send_image_file.assert_awaited_once()
        self.assertEqual(adapter.send_image_file.await_args.kwargs["image_path"], "/tmp/cached.png")

    @patch.dict(os.environ, {}, clear=True)
    def test_send_animation_degrades_to_document_send(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter.send_document = AsyncMock(return_value=SimpleNamespace(success=True, message_id="om_gif"))

        async def _run():
            with patch.object(
                adapter,
                "_download_remote_document",
                new=AsyncMock(return_value=("/tmp/anim.gif", "anim.gif")),
            ):
                return await adapter.send_animation("oc_chat", "https://example.com/anim.gif", caption="look")

        result = asyncio.run(_run())

        self.assertTrue(result.success)
        adapter.send_document.assert_awaited_once()
        caption = adapter.send_document.await_args.kwargs["caption"]
        self.assertIn("GIF downgraded to file", caption)
        self.assertIn("look", caption)

    def test_dedup_state_persists_across_adapter_restart(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        with tempfile.TemporaryDirectory() as temp_home:
            with patch.dict(os.environ, {"HERMES_HOME": temp_home}, clear=False):
                first = FeishuAdapter(PlatformConfig())
                self.assertFalse(first._is_duplicate("om_same"))
                second = FeishuAdapter(PlatformConfig())
                self.assertTrue(second._is_duplicate("om_same"))

    @patch.dict(os.environ, {}, clear=True)
    def test_process_inbound_group_message_keeps_group_type_when_chat_lookup_falls_back(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_group", "name": "oc_group", "type": "dm"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        message = SimpleNamespace(
            chat_id="oc_group",
            thread_id=None,
            message_type="text",
            content='{"text":"hello group"}',
            message_id="om_group_text",
        )
        sender_id = SimpleNamespace(open_id="ou_user", user_id=None, union_id=None)
        data = SimpleNamespace(event=SimpleNamespace(message=message))

        asyncio.run(
            adapter._process_inbound_message(
                data=data,
                message=message,
                sender_id=sender_id,
                chat_type="group",
                message_id="om_group_text",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.source.chat_type, "group")

    @patch.dict(os.environ, {}, clear=True)
    def test_process_inbound_message_fetches_reply_to_text(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_chat", "name": "Feishu DM", "type": "dm"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "张三", "user_id_alt": None}
        )
        adapter._fetch_message_text = AsyncMock(return_value="父消息内容")
        message = SimpleNamespace(
            chat_id="oc_chat",
            thread_id=None,
            parent_id="om_parent",
            upper_message_id=None,
            message_type="text",
            content='{"text":"reply"}',
            message_id="om_reply",
        )

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                chat_type="p2p",
                message_id="om_reply",
            )
        )

        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.reply_to_message_id, "om_parent")
        self.assertEqual(event.reply_to_text, "父消息内容")

    @patch.dict(os.environ, {}, clear=True)
    def test_send_replies_in_thread_when_thread_metadata_present(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ReplyAPI:
            def reply(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_reply"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_ReplyAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="hello",
                    reply_to="om_parent",
                    metadata={"thread_id": "omt-thread"},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_reply")
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_retries_transient_failure(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"attempts": 0}
        sleeps = []

        class _MessageAPI:
            def create(self, request):
                captured["attempts"] += 1
                captured["request"] = request
                if captured["attempts"] == 1:
                    raise OSError("temporary send failure")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_retry"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def _sleep(delay):
            sleeps.append(delay)

        with (
            patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct),
            patch("gateway.platforms.feishu.asyncio.sleep", side_effect=_sleep),
        ):
            result = asyncio.run(adapter.send(chat_id="oc_chat", content="hello retry"))

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_retry")
        self.assertEqual(captured["attempts"], 2)
        self.assertEqual(sleeps, [1])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_does_not_retry_deterministic_api_failure(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"attempts": 0}
        sleeps = []

        class _MessageAPI:
            def create(self, request):
                captured["attempts"] += 1
                return SimpleNamespace(
                    success=lambda: False,
                    code=400,
                    msg="bad request",
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        async def _sleep(delay):
            sleeps.append(delay)

        with (
            patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct),
            patch("gateway.platforms.feishu.asyncio.sleep", side_effect=_sleep),
        ):
            result = asyncio.run(adapter.send(chat_id="oc_chat", content="bad payload"))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "[400] bad request")
        self.assertEqual(captured["attempts"], 1)
        self.assertEqual(sleeps, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_document_reply_uses_thread_flag(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_123"),
                )

        class _MessageAPI:
            def reply(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_file_reply"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test")
            file_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(
                    adapter.send_document(
                        chat_id="oc_chat",
                        file_path=file_path,
                        reply_to="om_parent",
                        metadata={"thread_id": "omt-thread"},
                    )
                )
        finally:
            os.unlink(file_path)

        self.assertTrue(result.success)
        self.assertTrue(captured["request"].request_body.reply_in_thread)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_document_uploads_file_and_sends_file_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_file_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test")
            file_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_document(chat_id="oc_chat", file_path=file_path))
        finally:
            os.unlink(file_path)

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_file_msg")
        self.assertEqual(captured["upload_request"].request_body.file_type, "pdf")
        self.assertEqual(
            captured["message_request"].request_body.content,
            '{"file_key": "file_123"}',
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_document_with_caption_uses_single_post_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_post_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test")
            file_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(
                    adapter.send_document(chat_id="oc_chat", file_path=file_path, caption="报告请看")
                )
        finally:
            os.unlink(file_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["message_request"].request_body.msg_type, "post")
        self.assertIn('"tag": "media"', captured["message_request"].request_body.content)
        self.assertIn('"file_key": "file_123"', captured["message_request"].request_body.content)
        self.assertIn("报告请看", captured["message_request"].request_body.content)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_image_file_uploads_image_and_sends_image_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ImageAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(image_key="img_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_image_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    image=_ImageAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n")
            image_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_image_file(chat_id="oc_chat", image_path=image_path))
        finally:
            os.unlink(image_path)

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "om_image_msg")
        self.assertEqual(captured["upload_request"].request_body.image_type, "message")
        self.assertEqual(
            captured["message_request"].request_body.content,
            '{"image_key": "img_123"}',
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_image_file_with_caption_uses_single_post_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _ImageAPI:
            def create(self, request):
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(image_key="img_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_post_img"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    image=_ImageAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n")
            image_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(
                    adapter.send_image_file(chat_id="oc_chat", image_path=image_path, caption="截图说明")
                )
        finally:
            os.unlink(image_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["message_request"].request_body.msg_type, "post")
        self.assertIn('"tag": "img"', captured["message_request"].request_body.content)
        self.assertIn('"image_key": "img_123"', captured["message_request"].request_body.content)
        self.assertIn("截图说明", captured["message_request"].request_body.content)

    @patch.dict(os.environ, {}, clear=True)
    def test_send_video_uploads_file_and_sends_media_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_video_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_video_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".mp4", delete=False) as tmp:
            tmp.write(b"\x00\x00\x00\x18ftypmp42")
            video_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_video(chat_id="oc_chat", video_path=video_path))
        finally:
            os.unlink(video_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["upload_request"].request_body.file_type, "mp4")
        self.assertEqual(captured["message_request"].request_body.msg_type, "media")
        self.assertEqual(captured["message_request"].request_body.content, '{"file_key": "file_video_123"}')

    @patch.dict(os.environ, {}, clear=True)
    def test_send_voice_uploads_opus_and_sends_audio_message(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _FileAPI:
            def create(self, request):
                captured["upload_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(file_key="file_audio_123"),
                )

        class _MessageAPI:
            def create(self, request):
                captured["message_request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_audio_msg"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    file=_FileAPI(),
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.NamedTemporaryFile("wb", suffix=".opus", delete=False) as tmp:
            tmp.write(b"opus")
            audio_path = tmp.name

        try:
            with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
                result = asyncio.run(adapter.send_voice(chat_id="oc_chat", audio_path=audio_path))
        finally:
            os.unlink(audio_path)

        self.assertTrue(result.success)
        self.assertEqual(captured["upload_request"].request_body.file_type, "opus")
        self.assertEqual(captured["message_request"].request_body.msg_type, "audio")
        self.assertEqual(captured["message_request"].request_body.content, '{"file_key": "file_audio_123"}')

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_extracts_title_and_links(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(adapter._build_post_payload("# 标题\n访问 [文档](https://example.com)"))

        elements = payload["zh_cn"]["content"][0]
        self.assertEqual(elements, [{"tag": "md", "text": "# 标题\n访问 [文档](https://example.com)"}])

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_wraps_markdown_in_md_tag(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload("支持 **粗体**、*斜体* 和 `代码`")
        )

        elements = payload["zh_cn"]["content"][0]
        self.assertEqual(
            elements,
            [
                {"tag": "md", "text": "支持 **粗体**、*斜体* 和 `代码`"},
            ],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_build_post_payload_keeps_full_markdown_text(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        payload = json.loads(
            adapter._build_post_payload(
                "---\n1. 第一项\n  2. 子项\n- 外层\n  - 内层\n<u>下划线</u> 和 ~~删除线~~"
            )
        )

        rows = payload["zh_cn"]["content"]
        self.assertEqual(
            rows,
            [[{"tag": "md", "text": "---\n1. 第一项\n  2. 子项\n- 外层\n  - 内层\n<u>下划线</u> 和 ~~删除线~~"}]],
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_uses_post_for_inline_markdown(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def create(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_markdown"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "post")
        payload = json.loads(captured["request"].request_body.content)
        elements = payload["zh_cn"]["content"][0]
        self.assertEqual(elements, [{"tag": "md", "text": "可以用 **粗体** 和 *斜体*。"}])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_falls_back_to_text_when_post_payload_is_rejected(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"calls": []}

        class _MessageAPI:
            def create(self, request):
                captured["calls"].append(request)
                if len(captured["calls"]) == 1:
                    raise RuntimeError("content format of the post type is incorrect")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_plain"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["calls"][0].request_body.msg_type, "post")
        self.assertEqual(captured["calls"][1].request_body.msg_type, "text")
        self.assertEqual(
            captured["calls"][1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_falls_back_to_text_when_post_response_is_unsuccessful(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {"calls": []}

        class _MessageAPI:
            def create(self, request):
                captured["calls"].append(request)
                if len(captured["calls"]) == 1:
                    return SimpleNamespace(success=lambda: False, code=230001, msg="content format of the post type is incorrect")
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_plain_response"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="可以用 **粗体** 和 *斜体*。",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["calls"][0].request_body.msg_type, "post")
        self.assertEqual(captured["calls"][1].request_body.msg_type, "text")
        self.assertEqual(
            captured["calls"][1].request_body.content,
            json.dumps({"text": "可以用 粗体 和 斜体。"}, ensure_ascii=False),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_send_uses_post_for_advanced_markdown_lines(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured = {}

        class _MessageAPI:
            def create(self, request):
                captured["request"] = request
                return SimpleNamespace(
                    success=lambda: True,
                    data=SimpleNamespace(message_id="om_markdown_advanced"),
                )

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(
                    message=_MessageAPI(),
                )
            )
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(
                adapter.send(
                    chat_id="oc_chat",
                    content="---\n1. 第一项\n<u>下划线</u>\n~~删除线~~",
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["request"].request_body.msg_type, "post")
        payload = json.loads(captured["request"].request_body.content)
        rows = payload["zh_cn"]["content"]
        self.assertEqual(
            rows,
            [[{"tag": "md", "text": "---\n1. 第一项\n<u>下划线</u>\n~~删除线~~"}]],
        )


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestWebhookSecurity(unittest.TestCase):
    """Tests for webhook signature verification, rate limiting, and body size limits."""

    def _make_adapter(self, encrypt_key: str = "") -> "FeishuAdapter":
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        with patch.dict(os.environ, {"FEISHU_APP_ID": "cli", "FEISHU_APP_SECRET": "sec", "FEISHU_ENCRYPT_KEY": encrypt_key}, clear=True):
            return FeishuAdapter(PlatformConfig())

    def test_signature_valid_passes(self):
        import hashlib
        from gateway.platforms.feishu import FeishuAdapter
        from gateway.config import PlatformConfig

        encrypt_key = "test_secret"
        adapter = self._make_adapter(encrypt_key)
        body = b'{"type":"event"}'
        timestamp = "1700000000"
        nonce = "abc123"
        content = f"{timestamp}{nonce}{encrypt_key}" + body.decode("utf-8")
        sig = hashlib.sha256(content.encode("utf-8")).hexdigest()
        headers = {"x-lark-request-timestamp": timestamp, "x-lark-request-nonce": nonce, "x-lark-signature": sig}
        self.assertTrue(adapter._is_webhook_signature_valid(headers, body))

    def test_signature_invalid_rejected(self):
        adapter = self._make_adapter("test_secret")
        headers = {
            "x-lark-request-timestamp": "1700000000",
            "x-lark-request-nonce": "abc",
            "x-lark-signature": "deadbeef" * 8,
        }
        self.assertFalse(adapter._is_webhook_signature_valid(headers, b'{"type":"event"}'))

    def test_signature_missing_headers_rejected(self):
        adapter = self._make_adapter("test_secret")
        self.assertFalse(adapter._is_webhook_signature_valid({}, b'{}'))

    def test_rate_limit_allows_requests_within_window(self):
        adapter = self._make_adapter()
        for _ in range(5):
            self.assertTrue(adapter._check_webhook_rate_limit("10.0.0.1"))

    def test_rate_limit_blocks_after_exceeding_max(self):
        from gateway.platforms.feishu import _FEISHU_WEBHOOK_RATE_LIMIT_MAX
        adapter = self._make_adapter()
        for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
            adapter._check_webhook_rate_limit("10.0.0.2")
        self.assertFalse(adapter._check_webhook_rate_limit("10.0.0.2"))

    def test_rate_limit_resets_after_window_expires(self):
        from gateway.platforms.feishu import _FEISHU_WEBHOOK_RATE_LIMIT_MAX, _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS
        adapter = self._make_adapter()
        ip = "10.0.0.3"
        for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
            adapter._check_webhook_rate_limit(ip)
        self.assertFalse(adapter._check_webhook_rate_limit(ip))
        # Simulate window expiry by backdating the stored entry.
        count, window_start = adapter._webhook_rate_counts[ip]
        adapter._webhook_rate_counts[ip] = (count, window_start - _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS - 1)
        self.assertTrue(adapter._check_webhook_rate_limit(ip))

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_request_rejects_oversized_body(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter, _FEISHU_WEBHOOK_MAX_BODY_BYTES

        adapter = FeishuAdapter(PlatformConfig())
        # Simulate a request whose Content-Length already signals oversize.
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=_FEISHU_WEBHOOK_MAX_BODY_BYTES + 1,
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 413)

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_request_rejects_invalid_json(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            read=AsyncMock(return_value=b"not-json"),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 400)

    @patch.dict(os.environ, {"FEISHU_ENCRYPT_KEY": "secret"}, clear=True)
    def test_webhook_request_rejects_bad_signature(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        body = json.dumps({"header": {"event_type": "im.message.receive_v1"}}).encode()
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            headers={"x-lark-request-timestamp": "123", "x-lark-request-nonce": "abc", "x-lark-signature": "bad"},
            read=AsyncMock(return_value=body),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 401)

    @patch.dict(os.environ, {}, clear=True)
    def test_webhook_url_verification_challenge_passes_without_signature(self):
        """Challenge requests must succeed even when no encrypt_key is set."""
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        body = json.dumps({"type": "url_verification", "challenge": "test_challenge_token"}).encode()
        request = SimpleNamespace(
            remote="127.0.0.1",
            content_length=None,
            read=AsyncMock(return_value=body),
        )
        response = asyncio.run(adapter._handle_webhook_request(request))
        self.assertEqual(response.status, 200)
        self.assertIn(b"test_challenge_token", response.body)


class TestDedupTTL(unittest.TestCase):
    """Tests for TTL-aware deduplication."""

    @patch.dict(os.environ, {}, clear=True)
    def test_duplicate_within_ttl_is_rejected(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with patch.object(adapter, "_persist_seen_message_ids"):
            adapter._seen_message_ids = {"om_dup": time.time()}
            adapter._seen_message_order = ["om_dup"]
            self.assertTrue(adapter._is_duplicate("om_dup"))

    @patch.dict(os.environ, {}, clear=True)
    def test_expired_entry_is_not_considered_duplicate(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter, _FEISHU_DEDUP_TTL_SECONDS

        adapter = FeishuAdapter(PlatformConfig())
        # Plant an entry that expired well past the TTL.
        stale_ts = time.time() - _FEISHU_DEDUP_TTL_SECONDS - 60
        adapter._seen_message_ids = {"om_old": stale_ts}
        adapter._seen_message_order = ["om_old"]
        with patch.object(adapter, "_persist_seen_message_ids"):
            self.assertFalse(adapter._is_duplicate("om_old"))

    @patch.dict(os.environ, {}, clear=True)
    def test_persist_saves_timestamps_as_dict(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        ts = time.time()
        adapter._seen_message_ids = {"om_ts1": ts}
        adapter._seen_message_order = ["om_ts1"]
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter._dedup_state_path = Path(tmpdir) / "dedup.json"
            adapter._persist_seen_message_ids()
            saved = json.loads(adapter._dedup_state_path.read_text())
        self.assertIsInstance(saved["message_ids"], dict)
        self.assertAlmostEqual(saved["message_ids"]["om_ts1"], ts, places=1)

    @patch.dict(os.environ, {}, clear=True)
    def test_load_backward_compat_list_format(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dedup.json"
            path.write_text(json.dumps({"message_ids": ["om_a", "om_b"]}), encoding="utf-8")
            adapter._dedup_state_path = path
            adapter._load_seen_message_ids()
        self.assertIn("om_a", adapter._seen_message_ids)
        self.assertIn("om_b", adapter._seen_message_ids)


class TestGroupMentionAtAll(unittest.TestCase):
    """Tests for @_all (Feishu @everyone) group mention routing."""

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "open"}, clear=True)
    def test_at_all_in_content_accepts_without_explicit_bot_mention(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(
            content='{"text":"@_all 请注意"}',
            mentions=[],
        )
        sender_id = SimpleNamespace(open_id="ou_any", user_id=None)
        self.assertTrue(adapter._should_accept_group_message(message, sender_id, ""))

    @patch.dict(os.environ, {"FEISHU_GROUP_POLICY": "allowlist", "FEISHU_ALLOWED_USERS": "ou_allowed"}, clear=True)
    def test_at_all_still_requires_policy_gate(self):
        """@_all bypasses mention gating but NOT the allowlist policy."""
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        message = SimpleNamespace(content='{"text":"@_all attention"}', mentions=[])
        # Non-allowlisted user — should be blocked even with @_all.
        blocked_sender = SimpleNamespace(open_id="ou_blocked", user_id=None)
        self.assertFalse(adapter._should_accept_group_message(message, blocked_sender, ""))
        # Allowlisted user — should pass.
        allowed_sender = SimpleNamespace(open_id="ou_allowed", user_id=None)
        self.assertTrue(adapter._should_accept_group_message(message, allowed_sender, ""))


@unittest.skipUnless(_HAS_LARK_OAPI, "lark-oapi not installed")
class TestSenderNameResolution(unittest.TestCase):
    """Tests for _resolve_sender_name_from_api."""

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_client_is_none(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = None
        result = asyncio.run(adapter._resolve_sender_name_from_api("ou_abc"))
        self.assertIsNone(result)

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_cached_name_within_ttl(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace()
        future_expire = time.time() + 600
        adapter._sender_name_cache["ou_cached"] = ("Alice", future_expire)
        result = asyncio.run(adapter._resolve_sender_name_from_api("ou_cached"))
        self.assertEqual(result, "Alice")

    @patch.dict(os.environ, {}, clear=True)
    def test_fetches_and_caches_name_from_api(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        user_obj = SimpleNamespace(name="Bob", display_name=None, nickname=None, en_name=None)
        mock_response = SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(user=user_obj),
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        class _ContactAPI:
            def get(self, request):
                return mock_response

        adapter._client = SimpleNamespace(
            contact=SimpleNamespace(v3=SimpleNamespace(user=_ContactAPI()))
        )

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_bob"))

        self.assertEqual(result, "Bob")
        self.assertIn("ou_bob", adapter._sender_name_cache)

    @patch.dict(os.environ, {}, clear=True)
    def test_expired_cache_triggers_new_api_call(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        # Expired cache entry.
        adapter._sender_name_cache["ou_expired"] = ("OldName", time.time() - 1)

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        user_obj = SimpleNamespace(name="NewName", display_name=None, nickname=None, en_name=None)

        class _ContactAPI:
            def get(self, request):
                return SimpleNamespace(success=lambda: True, data=SimpleNamespace(user=user_obj))

        adapter._client = SimpleNamespace(
            contact=SimpleNamespace(v3=SimpleNamespace(user=_ContactAPI()))
        )

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_expired"))

        self.assertEqual(result, "NewName")

    @patch.dict(os.environ, {}, clear=True)
    def test_api_failure_returns_none_without_raising(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())

        class _BrokenContactAPI:
            def get(self, _request):
                raise RuntimeError("API down")

        adapter._client = SimpleNamespace(
            contact=SimpleNamespace(v3=SimpleNamespace(user=_BrokenContactAPI()))
        )

        async def _direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
            result = asyncio.run(adapter._resolve_sender_name_from_api("ou_broken"))

        self.assertIsNone(result)
