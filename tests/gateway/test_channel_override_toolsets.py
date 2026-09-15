"""Per-channel toolset resolution through channel_overrides."""

from gateway.config import ChannelOverride, GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.tools_config import _get_platform_tools


BASE_CONFIG = {"platform_toolsets": {"bluebubbles": ["web", "vision"]}}


class _Adapter:
    def __init__(self, override=None):
        self.override = override

    def toolsets_for_source(self, source):
        return self.override


def _runner(channel_overrides=None, adapter_override=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig.from_dict({
        "platforms": {
            "bluebubbles": {
                "enabled": True,
                "channel_overrides": channel_overrides or {},
            }
        }
    })
    adapter = _Adapter(adapter_override)
    runner._delivery_adapter_for = lambda source: adapter
    return runner


def _source(chat_id="chat-1", *, thread_id=None, parent_chat_id=None):
    return SessionSource(
        platform=Platform.BLUEBUBBLES,
        chat_id=chat_id,
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
    )


def _resolve(runner, source=None, config=None):
    return GatewayRunner._resolve_enabled_toolsets_for_source(
        runner, config or BASE_CONFIG, source or _source(), "bluebubbles"
    )


class TestChannelOverrideToolsetConfig:
    def test_absent_empty_and_nonempty_are_distinct_and_round_trip(self):
        absent = ChannelOverride.from_dict({"model": "example/model"})
        null = ChannelOverride.from_dict({"enabled_toolsets": None})
        empty = ChannelOverride.from_dict({"enabled_toolsets": []})
        populated = ChannelOverride.from_dict({"enabled_toolsets": [" web ", "", "file"]})

        assert absent.enabled_toolsets is None
        assert "enabled_toolsets" not in absent.to_dict()
        assert null.enabled_toolsets is None
        assert empty.enabled_toolsets == []
        assert empty.to_dict()["enabled_toolsets"] == []
        assert populated.enabled_toolsets == ["web", "file"]


class TestChannelOverrideToolsetResolution:
    def test_channel_override_replaces_platform_defaults(self):
        runner = _runner({"chat-1": {"enabled_toolsets": ["terminal", "file", "no_mcp"]}})

        result = _resolve(runner)

        assert "terminal" in result and "file" in result
        assert "web" not in result and "vision" not in result
        assert "no_mcp" not in result

    def test_explicit_empty_override_exposes_no_tools(self):
        runner = _runner({"chat-1": {"enabled_toolsets": []}})

        assert _resolve(runner) == []

    def test_absent_override_uses_platform_defaults(self):
        runner = _runner({"chat-1": {"system_prompt": "Channel prompt"}})

        assert _resolve(runner) == sorted(_get_platform_tools(BASE_CONFIG, "bluebubbles"))

    def test_parent_channel_override_is_inherited(self):
        runner = _runner({"parent": {"enabled_toolsets": ["file"]}})

        assert _resolve(runner, _source("thread", parent_chat_id="parent")) == ["file"]

    def test_adapter_route_policy_wins_over_channel_override(self):
        runner = _runner(
            {"chat-1": {"enabled_toolsets": ["terminal"]}},
            adapter_override=["web"],
        )

        result = _resolve(runner)
        assert "web" in result
        assert "terminal" not in result

    def test_global_disabled_toolsets_still_win(self):
        runner = _runner({"chat-1": {"enabled_toolsets": ["terminal", "file"]}})
        config = {
            "platform_toolsets": BASE_CONFIG["platform_toolsets"],
            "agent": {"disabled_toolsets": ["terminal"]},
        }

        result = _resolve(runner, config=config)
        assert "terminal" not in result
        assert "file" in result

    def test_platform_restricted_toolset_is_filtered(self):
        override = ["file", "discord_admin"]
        runner = _runner({"chat-1": {"enabled_toolsets": override}})
        expected = sorted(_get_platform_tools(
            {"platform_toolsets": {"bluebubbles": override}}, "bluebubbles"
        ))

        assert _resolve(runner) == expected
        assert "discord_admin" not in expected

    def test_resolution_does_not_mutate_user_config(self):
        config = {"platform_toolsets": {"bluebubbles": ["web"]}}
        runner = _runner({"chat-1": {"enabled_toolsets": ["file"]}})

        _resolve(runner, config=config)

        assert config == {"platform_toolsets": {"bluebubbles": ["web"]}}

    def test_resolved_toolsets_participate_in_agent_cache_signature(self):
        common = ("example/model", {"provider": "test"})
        broad = GatewayRunner._agent_config_signature(*common, ["web", "file"], "")
        narrow = GatewayRunner._agent_config_signature(*common, ["file"], "")

        assert broad != narrow
