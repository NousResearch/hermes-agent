"""Tests for multi-app Feishu list-config compatibility.

Regression guard: when platforms.feishu is a list, several code paths used to
raise AttributeError: 'list' object has no attribute 'extra'.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import BasePlatformAdapter


class DummyMixin(GatewayAuthorizationMixin):
    """Minimal mixin instance for testing list-config paths."""

    def __init__(self, config):
        self.config = config
        self.adapters = {}


class DummyAdapter(BasePlatformAdapter):
    """Small concrete adapter for runner routing tests."""

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, **kwargs):
        calls = getattr(self, "send_calls", None)
        if calls is not None:
            calls.append((chat_id, content, kwargs))
        return None

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


class TestMultiAppListConfig:
    """Cover the list-config branches introduced in PR #42499."""

    @pytest.fixture
    def multi_config(self):
        return GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        token="app1-token",
                        extra={
                            "dm_policy": "open",
                            "unauthorized_dm_behavior": "ignore",
                            "notice_delivery": "private",
                            "app_id": "cli_app1",
                        },
                    ),
                    PlatformConfig(
                        enabled=True,
                        token="app2-token",
                        extra={
                            "dm_policy": "pairing",
                            "app_id": "cli_app2",
                        },
                    ),
                ],
            }
        )

    @pytest.fixture
    def single_config(self):
        return GatewayConfig(
            platforms={
                Platform.FEISHU: PlatformConfig(
                    enabled=True,
                    token="single-token",
                    extra={
                        "dm_policy": "open",
                        "unauthorized_dm_behavior": "pair",
                        "notice_delivery": "public",
                    },
                ),
            }
        )

    # --- GatewayConfig ---

    def test_get_unauthorized_dm_behavior_list_first_wins(self, multi_config):
        """First config in the list with the key wins."""
        assert multi_config.get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"

    def test_get_unauthorized_dm_behavior_single(self, single_config):
        assert single_config.get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    def test_get_unauthorized_dm_behavior_missing_platform(self, single_config):
        assert single_config.get_unauthorized_dm_behavior(Platform.SLACK) == "pair"

    def test_get_notice_delivery_list_first_wins(self, multi_config):
        assert multi_config.get_notice_delivery(Platform.FEISHU) == "private"

    def test_get_notice_delivery_single(self, single_config):
        assert single_config.get_notice_delivery(Platform.FEISHU) == "public"

    def test_get_notice_delivery_missing_platform(self, single_config):
        assert single_config.get_notice_delivery(Platform.SLACK) == "public"

    # --- GatewayAuthorizationMixin._adapter_dm_policy ---

    def test_adapter_dm_policy_list(self, multi_config):
        mixin = DummyMixin(multi_config)
        assert mixin._adapter_dm_policy(Platform.FEISHU) == "open"

    def test_adapter_dm_policy_single(self, single_config):
        mixin = DummyMixin(single_config)
        assert mixin._adapter_dm_policy(Platform.FEISHU) == "open"

    def test_adapter_dm_policy_no_config(self):
        mixin = DummyMixin(GatewayConfig())
        assert mixin._adapter_dm_policy(Platform.FEISHU) == ""

    # --- GatewayAuthorizationMixin._get_unauthorized_dm_behavior ---

    def test_get_unauthorized_dm_behavior_list_explicit(self, multi_config):
        mixin = DummyMixin(multi_config)
        # First list item has explicit unauthorized_dm_behavior
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"

    def test_get_unauthorized_dm_behavior_single_explicit(self, single_config):
        mixin = DummyMixin(single_config)
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    def test_get_unauthorized_dm_behavior_list_dm_policy_pairing(self):
        """When no explicit unauthorized_dm_behavior but dm_policy == pairing."""
        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        extra={"dm_policy": "pairing"},
                    ),
                ],
            }
        )
        mixin = DummyMixin(config)
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    def test_get_unauthorized_dm_behavior_list_dm_policy_disabled(self):
        """When dm_policy is disabled, default to ignore."""
        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        extra={"dm_policy": "disabled"},
                    ),
                ],
            }
        )
        mixin = DummyMixin(config)
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"

    def test_get_unauthorized_dm_behavior_no_platform(self):
        mixin = DummyMixin(GatewayConfig())
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    # --- GatewayConfig roundtrip with list ---

    def test_to_dict_from_dict_preserves_list(self, multi_config):
        d = multi_config.to_dict()
        restored = GatewayConfig.from_dict(d)
        feishu_cfgs = restored.platforms[Platform.FEISHU]
        assert isinstance(feishu_cfgs, list)
        assert len(feishu_cfgs) == 2
        assert feishu_cfgs[0].extra["app_id"] == "cli_app1"
        assert feishu_cfgs[1].extra["app_id"] == "cli_app2"

    def test_from_dict_preserves_top_level_platform_specific_keys_in_extra(self):
        restored = GatewayConfig.from_dict(
            {
                "platforms": {
                    "feishu": [
                        {"enabled": True, "app_id": "cli_app1", "app_secret": "secret1"},
                        {"enabled": True, "app_id": "cli_app2", "app_secret": "secret2"},
                    ]
                }
            }
        )

        feishu_cfgs = restored.platforms[Platform.FEISHU]

        assert isinstance(feishu_cfgs, list)
        assert feishu_cfgs[0].extra["app_id"] == "cli_app1"
        assert feishu_cfgs[0].extra["app_secret"] == "secret1"
        assert feishu_cfgs[1].extra["app_id"] == "cli_app2"
        assert feishu_cfgs[1].extra["app_secret"] == "secret2"

    def test_env_overrides_do_not_clobber_explicit_multi_app_credentials(self):
        config = GatewayConfig.from_dict(
            {
                "platforms": {
                    "feishu": [
                        {"enabled": True, "app_id": "cli_app1", "app_secret": "secret1"},
                        {"enabled": True, "app_id": "cli_app2", "app_secret": "secret2"},
                    ]
                }
            }
        )

        with patch.dict(
            "os.environ",
            {
                "FEISHU_APP_ID": "cli_env",
                "FEISHU_APP_SECRET": "secret_env",
                "FEISHU_DOMAIN": "lark",
                "FEISHU_CONNECTION_MODE": "websocket",
            },
            clear=False,
        ):
            _apply_env_overrides(config)

        feishu_cfgs = config.platforms[Platform.FEISHU]

        assert isinstance(feishu_cfgs, list)
        assert feishu_cfgs[0].extra["app_id"] == "cli_app1"
        assert feishu_cfgs[0].extra["app_secret"] == "secret1"
        assert feishu_cfgs[1].extra["app_id"] == "cli_app2"
        assert feishu_cfgs[1].extra["app_secret"] == "secret2"
        assert feishu_cfgs[0].extra["domain"] == "lark"
        assert feishu_cfgs[1].extra["domain"] == "lark"

    def test_env_home_channel_targets_matching_feishu_app(self):
        config = GatewayConfig.from_dict(
            {
                "platforms": {
                    "feishu": [
                        {"enabled": True, "app_id": "cli_app1", "app_secret": "secret1"},
                        {"enabled": True, "app_id": "cli_app2", "app_secret": "secret2"},
                    ]
                }
            }
        )

        with patch.dict(
            "os.environ",
            {
                "FEISHU_APP_ID": "cli_app2",
                "FEISHU_APP_SECRET": "secret2",
                "FEISHU_HOME_CHANNEL": "chat-app2",
                "FEISHU_HOME_CHANNEL_NAME": "App 2 Home",
            },
            clear=False,
        ):
            _apply_env_overrides(config)

        first, second = config.platforms[Platform.FEISHU]
        assert first.home_channel is None
        assert second.home_channel is not None
        assert second.home_channel.chat_id == "chat-app2"
        assert second.home_channel.name == "App 2 Home"

    def test_get_home_channel_list(self, multi_config):
        """get_home_channel should iterate list and return first match."""
        # Default home_channel is None; test that it doesn't crash
        assert multi_config.get_home_channel(Platform.FEISHU) is None

    def test_connected_platforms_list(self, multi_config):
        """connected_platforms should include platform if any list item is connected."""
        # No real connection, but ensure it doesn't crash on list
        connected = multi_config.get_connected_platforms()
        assert isinstance(connected, list)

    def test_is_connected_true_with_app_id(self):
        """Feishu's plugin-registry connection checker (#68046 item 0)
        reports connected once app_id is set, regardless of app_secret —
        mirrors the legacy _PLATFORM_CONNECTED_CHECKERS lambda it replaced."""
        from plugins.platforms.feishu.adapter import _is_connected

        assert _is_connected(SimpleNamespace(extra={"app_id": "cli_app1"})) is True

    def test_is_connected_false_with_placeholder_app_id(self):
        """A placeholder profile (empty app_id) must be reported as NOT
        connected so cron delivery routing skips it instead of attempting a
        real Feishu API call with empty credentials."""
        from plugins.platforms.feishu.adapter import _is_connected

        assert _is_connected(SimpleNamespace(extra={"app_id": ""})) is False
        assert _is_connected(SimpleNamespace(extra={})) is False

    def test_build_source_carries_adapter_id(self):
        adapter = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}),
            Platform.FEISHU,
        )
        adapter.adapter_id = "feishu:cli_app1"

        source = adapter.build_source(chat_id="chat-1", user_id="user-1")

        assert source.adapter_id == "feishu:cli_app1"

    def test_build_source_carries_thread_id_for_topic_routing(self):
        """Regression: bot-to-bot 路由（_route_bot_to_bot）构造合成 source 时
        必须把原话题 thread_id（omt_）传给 build_source，否则 peer bot 回复会
        丢失话题上下文，在话题群里每条回复都新建一个话题根。"""
        adapter = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}),
            Platform.FEISHU,
        )
        adapter.adapter_id = "feishu:cli_app1"

        source = adapter.build_source(
            chat_id="chat-1", user_id="user-1", thread_id="omt_topic"
        )

        assert source.thread_id == "omt_topic"

    @pytest.mark.asyncio
    async def test_route_bot_to_bot_carries_thread_id_and_anchor(self):
        """问题1: _route_bot_to_bot 必须把 thread_id 与 sender 的 anchor_message_id
        透传到合成事件，使 peer bot 回复精确 reply 到 sender 的消息（om_）回到原话题，
        而非因丢失上下文在话题群里新建话题根。"""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}
        runner._adapter_profile_map = {}

        sender = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}), Platform.FEISHU
        )
        sender.adapter_id = "feishu:cli_app1"
        sender._bot_name = "Tony"
        sender._bot_open_id = "ou_sender"
        target = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app2"}), Platform.FEISHU
        )
        target.adapter_id = "feishu:cli_app2"
        target._bot_name = "Pete"
        target._resolve_channel_prompt = lambda chat_id: None

        runner._register_connected_adapter(Platform.FEISHU, sender)
        runner._register_connected_adapter(Platform.FEISHU, target)
        runner._adapter_profile_map = {
            "feishu:cli_app1": "tech-lead",
            "feishu:cli_app2": "pm",
        }
        runner.adapter_id_for_profile = lambda profile, platform=None: {
            "pm": "feishu:cli_app2"
        }.get(profile)

        captured: dict = {}

        async def _capture(event):
            captured["event"] = event

        target._handle_message_with_guards = _capture

        await runner._route_bot_to_bot(
            [("pm", "Pete")],
            "hello @Pete",
            "oc_chat",
            sender,
            hop=0,
            thread_id="omt_topic",
            anchor_message_id="om_tony_msg",
        )

        assert captured["event"].source.thread_id == "omt_topic"
        assert captured["event"].reply_to_message_id == "om_tony_msg"

    def test_session_key_isolates_matching_chat_ids_across_feishu_apps(self):
        """Different Feishu apps must not share agent/session state."""
        first = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_same",
            chat_type="dm",
            user_id="ou_same",
            adapter_id="feishu:cli_app1",
        )
        second = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_same",
            chat_type="dm",
            user_id="ou_same",
            adapter_id="feishu:cli_app2",
        )

        first_key = build_session_key(first)
        second_key = build_session_key(second)

        assert first_key == "agent:main:feishu:adapter=feishu%3Acli_app1:dm:oc_same"
        assert second_key == "agent:main:feishu:adapter=feishu%3Acli_app2:dm:oc_same"
        assert first_key != second_key

    def test_runner_routes_source_to_matching_feishu_app_adapter(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}

        app1 = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}),
            Platform.FEISHU,
        )
        app2 = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app2"}),
            Platform.FEISHU,
        )

        runner._register_connected_adapter(Platform.FEISHU, app1)
        runner._register_connected_adapter(Platform.FEISHU, app2)

        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="chat-from-app2",
            adapter_id="feishu:cli_app2",
        )

        assert runner.adapters[Platform.FEISHU] is app1
        assert runner._adapter_for_source(source) is app2

    @pytest.mark.asyncio
    async def test_startup_notifications_use_each_feishu_app_home_channel(self):
        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        extra={"app_id": "cli_app1"},
                        home_channel=HomeChannel(
                            platform=Platform.FEISHU,
                            chat_id="chat-app1",
                            name="App 1",
                        ),
                    ),
                    PlatformConfig(
                        enabled=True,
                        extra={"app_id": "cli_app2"},
                        home_channel=HomeChannel(
                            platform=Platform.FEISHU,
                            chat_id="chat-app2",
                            name="App 2",
                        ),
                    ),
                ],
            }
        )
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = config
        runner.adapters = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}

        app1 = DummyAdapter(config.platforms[Platform.FEISHU][0], Platform.FEISHU)
        app1.send_calls = []
        app2 = DummyAdapter(config.platforms[Platform.FEISHU][1], Platform.FEISHU)
        app2.send_calls = []

        runner._register_connected_adapter(Platform.FEISHU, app1)
        runner._register_connected_adapter(Platform.FEISHU, app2)

        delivered = await runner._send_home_channel_startup_notifications()

        assert ("feishu", "chat-app1", None) in delivered
        assert ("feishu", "chat-app2", None) in delivered
        assert [call[0] for call in app1.send_calls] == ["chat-app1"]
        assert [call[0] for call in app2.send_calls] == ["chat-app2"]

    @pytest.mark.asyncio
    async def test_background_completion_uses_source_feishu_app_adapter(
        self, monkeypatch
    ):
        import tools.process_registry as process_registry_module

        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}),
                    PlatformConfig(enabled=True, extra={"app_id": "cli_app2"}),
                ]
            }
        )
        runner = GatewayRunner(config)
        app1 = DummyAdapter(config.platforms[Platform.FEISHU][0], Platform.FEISHU)
        app2 = DummyAdapter(config.platforms[Platform.FEISHU][1], Platform.FEISHU)
        app1.handle_message = AsyncMock()
        app2.handle_message = AsyncMock()
        runner._register_connected_adapter(Platform.FEISHU, app1)
        runner._register_connected_adapter(Platform.FEISHU, app2)
        runner._load_background_notifications_mode = lambda: "all"

        session = SimpleNamespace(
            output_buffer="done\n",
            exited=True,
            exit_code=0,
            command="echo done",
        )

        class _Registry:
            def get(self, _session_id):
                return session

            def is_completion_consumed(self, _session_id):
                return False

        monkeypatch.setattr(process_registry_module, "process_registry", _Registry())

        async def _instant_sleep(*_args, **_kwargs):
            return None

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        await runner._run_process_watcher(
            {
                "session_id": "proc-app2",
                "check_interval": 0,
                "session_key": (
                    "agent:main:feishu:adapter=feishu%3Acli_app2:dm:oc_same"
                ),
                "platform": "feishu",
                "chat_id": "oc_same",
                "notify_on_complete": True,
            }
        )

        app1.handle_message.assert_not_awaited()
        app2.handle_message.assert_awaited_once()
        event = app2.handle_message.await_args.args[0]
        assert event.source.adapter_id == "feishu:cli_app2"


class TestApplyYamlConfigBridgesPolicy:
    """Regression: _apply_yaml_config must bridge per-app policy fields
    (default_group_policy, require_mention, …) into each app's extra, not
    just the explicit whitelist (app_id/allow_bots/…). Without this, YAML
    ``default_group_policy: open`` was silently dropped and the adapter fell
    back to ``FEISHU_GROUP_POLICY=allowlist``, rejecting all group traffic
    even though the user configured ``open`` (#68046).
    """

    def test_default_group_policy_bridged_to_each_app_extra(self):
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        feishu_cfg = {
            "apps": [
                {
                    "app_id": "cli_app1",
                    "app_secret": "s1",
                    "default_group_policy": "open",
                    "require_mention": False,
                    "profile": "architect",
                },
                {
                    "app_id": "cli_app2",
                    "app_secret": "s2",
                    "default_group_policy": "allowlist",
                },
            ]
        }
        result = _apply_yaml_config({}, feishu_cfg)
        assert result is not None
        apps = result["platforms_list"]
        assert len(apps) == 2

        by_app = {app["extra"]["app_id"]: app["extra"] for app in apps}
        # Core regression: default_group_policy reaches extra (was dropped →
        # adapter fell back to env allowlist → group traffic rejected).
        assert by_app["cli_app1"]["default_group_policy"] == "open"
        assert by_app["cli_app2"]["default_group_policy"] == "allowlist"
        # Generic bridge carries other policy fields too.
        assert by_app["cli_app1"]["require_mention"] is False
        # Explicit fields assigned upstream stay intact (setdefault).
        assert by_app["cli_app1"]["app_secret"] == "s1"
        assert by_app["cli_app1"]["profile"] == "architect"

    def test_gateway_level_routing_keys_bridged_to_each_app(self):
        """顶层 peer_routing_fallback_chat / send_failure_dead_letter_chat 必须
        复制进每个 app 的 extra——各 adapter 只读自己的 config.extra，顶层键
        不桥接就会全部读不到（p2p 重路由、死信转发双双失效）。"""
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        feishu_cfg = {
            "peer_routing_fallback_chat": "oc_fallback",
            "send_failure_dead_letter_chat": "oc_dead",
            "apps": [
                {"app_id": "cli_app1", "app_secret": "s1"},
                {"app_id": "cli_app2", "app_secret": "s2"},
            ],
        }
        result = _apply_yaml_config({}, feishu_cfg)
        apps = result["platforms_list"]
        assert len(apps) == 2
        for app in apps:
            assert app["extra"]["peer_routing_fallback_chat"] == "oc_fallback"
            assert app["extra"]["send_failure_dead_letter_chat"] == "oc_dead"

    def test_gateway_level_routing_keys_absent_when_unconfigured(self):
        """未配置这两个顶层键时，不应写入 extra（空值跳过，setdefault 不写入）。"""
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        feishu_cfg = {"apps": [{"app_id": "cli_app1", "app_secret": "s1"}]}
        result = _apply_yaml_config({}, feishu_cfg)
        extra = result["platforms_list"][0]["extra"]
        assert "peer_routing_fallback_chat" not in extra
        assert "send_failure_dead_letter_chat" not in extra


class TestP2pFallbackReroute:
    """规则 A：私聊(dm)里 @ 不在场的 peer，send hook 把 bot-to-bot 注入重路由
    到配置的 fallback 群，而非原 chat_id（否则 peer 回复 230002 静默丢失）。"""

    def _make_adapter(self, *, fallback_chat="oc_fallback", chat_type="dm"):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import SendResult
        from plugins.platforms.feishu.adapter import FeishuAdapter

        extra = {"app_id": "cli_app1"}
        if fallback_chat:
            extra["peer_routing_fallback_chat"] = fallback_chat
        # object.__new__ 跳过 __init__（同 feishu_helpers.make_adapter_skeleton），
        # 只 mock send() 与 send hook 依赖的方法。
        adapter = object.__new__(FeishuAdapter)
        adapter.config = PlatformConfig(enabled=True, extra=extra)
        adapter._client = SimpleNamespace()  # truthy → 绕过 send() 的 not-connected 早退
        adapter._bot_to_bot_reply_profile = None
        adapter._parse_peer_mention_targets = lambda content: [("tech-leader", "Tony")]
        adapter.get_chat_info = AsyncMock(return_value={"type": chat_type})
        adapter._feishu_send_with_retry = AsyncMock(
            return_value=SimpleNamespace(success=lambda: True)
        )
        adapter._response_succeeded = lambda r: True
        adapter._finalize_send_result = lambda resp, default: SendResult(
            success=True, message_id="om_x"
        )
        adapter.format_message = lambda c: c
        adapter.truncate_message = lambda c, m: [c]
        adapter._build_outbound_payload = lambda chunk, prefer_post=False: ("text", "{}")
        return adapter

    async def _capture_route_chat(self, adapter):
        routed = []

        async def _router(targets, content, chat_id, sender_adapter, **kw):
            routed.append(chat_id)

        adapter._bot_to_bot_router = _router
        await adapter.send(chat_id="oc_dm", content="hello @Tony")
        # send hook 用 asyncio.create_task fire-and-forget；让事件循环跑完它。
        for _ in range(20):
            if routed:
                break
            await asyncio.sleep(0)
        return routed

    @pytest.mark.asyncio
    async def test_p2p_mention_reroutes_to_fallback_chat(self):
        adapter = self._make_adapter(fallback_chat="oc_fallback", chat_type="dm")
        assert await self._capture_route_chat(adapter) == ["oc_fallback"]

    @pytest.mark.asyncio
    async def test_group_chat_keeps_original_chat_id(self):
        adapter = self._make_adapter(fallback_chat="oc_fallback", chat_type="group")
        assert await self._capture_route_chat(adapter) == ["oc_dm"]

    @pytest.mark.asyncio
    async def test_unconfigured_fallback_keeps_original_chat_id(self):
        adapter = self._make_adapter(fallback_chat=None, chat_type="dm")
        assert await self._capture_route_chat(adapter) == ["oc_dm"]


class TestSendDeadLetter:
    """规则 B：发送永久失败 → _send_with_retry 触发 _on_send_dead_letter；
    feishu override 把消息转发到死信群（由 default adapter 推送）。"""

    @pytest.mark.asyncio
    async def test_fallback_failure_triggers_dead_letter_hook(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.base import SendResult

        adapter = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}), Platform.FEISHU
        )
        adapter.adapter_id = "feishu:cli_app1"
        # send 始终失败（含 plain-text fallback 重发）→ 走 base.py fallback-fail 死信钩子。
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="[230002] out of chat"))
        adapter._is_retryable_error = lambda e: False
        called = []
        adapter._on_send_dead_letter = AsyncMock(side_effect=lambda *a, **k: called.append(a))
        result = await adapter._send_with_retry("oc_chat", "hello")
        assert result.success is False
        assert called and called[0][0] == "oc_chat" and called[0][1] == "hello"

    @pytest.mark.asyncio
    async def test_timeout_failure_skips_dead_letter_hook(self):
        """timeout 分支（消息可能已投递）不转死信，避免误导性留底。"""
        from gateway.config import PlatformConfig
        from gateway.platforms.base import SendResult

        adapter = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}), Platform.FEISHU
        )
        adapter.adapter_id = "feishu:cli_app1"
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="timeout"))
        adapter._is_retryable_error = lambda e: False
        adapter._is_timeout_error = lambda e: True
        called = []
        adapter._on_send_dead_letter = AsyncMock(side_effect=lambda *a, **k: called.append(a))
        await adapter._send_with_retry("oc_chat", "hello")
        assert not called

    @pytest.mark.asyncio
    async def test_feishu_override_forwards_to_injected_forwarder(self):
        """FeishuAdapter._on_send_dead_letter 读 extra 死信群并调注入的 forwarder。"""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        ns = SimpleNamespace(
            config=SimpleNamespace(extra={"send_failure_dead_letter_chat": "oc_dead"}),
            _dead_letter_forwarder=AsyncMock(),
        )
        await FeishuAdapter._on_send_dead_letter(
            ns, "oc_src", "原内容", SimpleNamespace(error="[230002] x")
        )
        ns._dead_letter_forwarder.assert_awaited_once_with(ns, "oc_src", "原内容", "[230002] x")

    @pytest.mark.asyncio
    async def test_feishu_override_noop_when_dead_letter_unconfigured(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        ns = SimpleNamespace(
            config=SimpleNamespace(extra={}),
            _dead_letter_forwarder=AsyncMock(),
        )
        await FeishuAdapter._on_send_dead_letter(ns, "oc_src", "x", SimpleNamespace(error="e"))
        ns._dead_letter_forwarder.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_runner_forwarder_uses_default_adapter(self):
        """_forward_feishu_dead_letter 用 default adapter 把格式化死信发到死信群。"""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {}
        runner._adapter_profile_map = {}

        default = DummyAdapter(
            PlatformConfig(
                enabled=True,
                extra={"app_id": "cli_default", "send_failure_dead_letter_chat": "oc_dead"},
            ),
            Platform.FEISHU,
        )
        default.adapter_id = "feishu:cli_default"
        sent = []
        default.send = AsyncMock(side_effect=lambda **kw: sent.append(kw))
        runner.adapters[Platform.FEISHU] = default

        failed = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_x"}), Platform.FEISHU
        )
        failed.adapter_id = "feishu:cli_failed"
        failed._bot_name = "技术总监Tony"
        runner._adapter_profile_map = {"feishu:cli_failed": "tech-leader"}

        await runner._forward_feishu_dead_letter(failed, "oc_target", "原内容正文", "[230002] out of chat")
        assert len(sent) == 1
        assert sent[0]["chat_id"] == "oc_dead"
        assert "tech-leader" in sent[0]["content"]
        assert "oc_target" in sent[0]["content"]
        assert "原内容正文" in sent[0]["content"]

    @pytest.mark.asyncio
    async def test_runner_forwarder_swallows_default_send_error(self):
        """default adapter 自身发送异常时不递归抛错（避免死信→死信循环）。"""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {}
        runner._adapter_profile_map = {}

        default = DummyAdapter(
            PlatformConfig(
                enabled=True,
                extra={"app_id": "cli_default", "send_failure_dead_letter_chat": "oc_dead"},
            ),
            Platform.FEISHU,
        )
        default.send = AsyncMock(side_effect=RuntimeError("boom"))
        runner.adapters[Platform.FEISHU] = default

        failed = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_x"}), Platform.FEISHU
        )
        failed._bot_name = "Tony"
        # 不抛即通过
        await runner._forward_feishu_dead_letter(failed, "oc_t", "x", "err")

    @pytest.mark.asyncio
    async def test_runner_forwarder_suppresses_bot_to_bot_router(self):
        """死信转发期间摘掉 default adapter 的 router，防止死信原文里的 @peer
        触发 bot-to-bot route 形成回环（实测：Tony 从死信投递里"看到"了消息）。"""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {}
        runner._adapter_profile_map = {}

        default = DummyAdapter(
            PlatformConfig(
                enabled=True,
                extra={"app_id": "cli_default", "send_failure_dead_letter_chat": "oc_dead"},
            ),
            Platform.FEISHU,
        )
        default._bot_to_bot_router = lambda *a, **k: None  # 模拟注入的 router（非 None）
        router_during_send = []

        async def _capture_send(**kw):
            router_during_send.append(getattr(default, "_bot_to_bot_router", None))

        default.send = AsyncMock(side_effect=_capture_send)
        runner.adapters[Platform.FEISHU] = default

        failed = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_x"}), Platform.FEISHU
        )
        failed._bot_name = "Tony"

        await runner._forward_feishu_dead_letter(failed, "oc_t", "原文 @技术总监Tony ...", "err")
        # 发送期间 router 被摘除 → 死信不会触发 bot-to-bot route
        assert router_during_send == [None]
        # 发送后 router 恢复，不影响后续正常协作
        assert callable(default._bot_to_bot_router)
