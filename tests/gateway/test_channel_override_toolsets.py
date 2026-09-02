"""Per-channel enabled_toolsets via channel_overrides.

Related: webhook per-route toolsets (test_webhook_route_toolsets.py) and
BlueBubbles chat GUID admission (PR #63990). Together they enable multi-mode
messaging: narrow capture groups vs full-tool private desk chats on one
platform without widening every conversation.
"""

from __future__ import annotations

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _get_channel_override
from hermes_cli.tools_config import _get_platform_tools


class _Src:
    def __init__(self, chat_id, platform=Platform.BLUEBUBBLES, thread_id=None):
        self.chat_id = chat_id
        self.platform = platform
        self.thread_id = thread_id


def _runner_with_config(config: GatewayConfig) -> GatewayRunner:
    gr = object.__new__(GatewayRunner)
    gr.config = config
    gr._adapter_for_source = lambda source: None
    return gr


BASE_BB = {"platform_toolsets": {"bluebubbles": ["skills", "todo", "kanban", "no_mcp"]}}


class TestChannelOverrideEnabledToolsetsRoundtrip:
    def test_from_dict_list(self):
        ov = ChannelOverride.from_dict(
            {"enabled_toolsets": ["terminal", "file", "  ", "web"]}
        )
        assert ov.enabled_toolsets == ["terminal", "file", "web"]
        d = ov.to_dict()
        assert d["enabled_toolsets"] == ["terminal", "file", "web"]

    def test_from_dict_comma_string(self):
        ov = ChannelOverride.from_dict({"enabled_toolsets": "terminal, file, web"})
        assert ov.enabled_toolsets == ["terminal", "file", "web"]

    def test_empty_list_is_no_override(self):
        ov = ChannelOverride.from_dict({"enabled_toolsets": []})
        assert ov.enabled_toolsets is None
        assert "enabled_toolsets" not in ov.to_dict()

    def test_absent_stays_none(self):
        ov = ChannelOverride.from_dict({"model": "gpt-4"})
        assert ov.enabled_toolsets is None


class TestResolveEnabledToolsetsChannelOverride:
    def test_channel_override_replaces_platform_toolsets(self):
        desk = "any;+;desk-guid"
        config = GatewayConfig(
            platforms={
                Platform.BLUEBUBBLES: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        desk: ChannelOverride(
                            enabled_toolsets=["terminal", "file", "web", "skills"]
                        ),
                    },
                ),
            },
        )
        gr = _runner_with_config(config)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_BB, _Src(desk), "bluebubbles"
        )
        assert "terminal" in res
        assert "file" in res
        assert "web" in res
        # platform capture list fully replaced for this chat
        expected = sorted(
            _get_platform_tools(
                {
                    "platform_toolsets": {
                        "bluebubbles": ["terminal", "file", "web", "skills"]
                    }
                },
                "bluebubbles",
            )
        )
        assert res == expected

    def test_other_chat_keeps_platform_defaults(self):
        desk = "any;+;desk-guid"
        capture = "any;+;capture-guid"
        config = GatewayConfig(
            platforms={
                Platform.BLUEBUBBLES: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        desk: ChannelOverride(
                            enabled_toolsets=["terminal", "file", "web"]
                        ),
                    },
                ),
            },
        )
        gr = _runner_with_config(config)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_BB, _Src(capture), "bluebubbles"
        )
        assert res == sorted(_get_platform_tools(BASE_BB, "bluebubbles"))
        assert "terminal" not in res

    def test_adapter_override_wins_over_channel_override(self):
        desk = "any;+;desk-guid"

        class _Adapter:
            def toolsets_for_source(self, source):
                return ["web"]

        config = GatewayConfig(
            platforms={
                Platform.BLUEBUBBLES: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        desk: ChannelOverride(
                            enabled_toolsets=["terminal", "file"]
                        ),
                    },
                ),
            },
        )
        gr = _runner_with_config(config)
        gr._adapter_for_source = lambda source: _Adapter()
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, BASE_BB, _Src(desk), "bluebubbles"
        )
        assert "web" in res
        assert "terminal" not in res

    def test_lookup_via_get_channel_override(self):
        ov = ChannelOverride(enabled_toolsets=["file", "skills"])
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={"-100123": ov},
                ),
            },
        )
        got = _get_channel_override(config, Platform.TELEGRAM, "-100123")
        assert got is not None
        assert got.enabled_toolsets == ["file", "skills"]
