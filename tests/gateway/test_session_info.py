"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.run import GatewayRunner


@pytest.fixture()
def runner():
    """Create a bare GatewayRunner without __init__."""
    return GatewayRunner.__new__(GatewayRunner)


def _patch_info(tmp_path, config_yaml, model, runtime):
    """Return a context-manager stack that patches _format_session_info deps."""
    cfg_path = tmp_path / "config.yaml"
    if config_yaml is not None:
        cfg_path.write_text(config_yaml)
    return (
        patch("gateway.run._hermes_home", tmp_path),
        patch("gateway.run._resolve_gateway_model", return_value=model),
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
    )


class TestFormatSessionInfo:



    def test_config_context_length(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  context_length: 32768\n",
                                  "test-model",
                                  {"provider": "custom", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "32K" in info
        assert "config" in info


    def test_local_endpoint_shown(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: qwen3:8b\n  provider: custom\n  base_url: http://localhost:11434/v1\n  context_length: 8192\n",
            "qwen3:8b",
            {"provider": "custom", "base_url": "http://localhost:11434/v1", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "localhost:11434" in info
        assert "8K" in info

    def test_moa_preset_names_the_billed_aggregator(self, runner, tmp_path):
        """#112359: the preset name hides who pays; /model must name the acting aggregator."""
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: review\n  provider: moa\n",
                                  "review", {"provider": "moa", "base_url": "", "api_key": ""})
        moa_cfg = {"moa": {"presets": {"review": {
            "reference_models": [{"provider": "openai", "model": "gpt-5.5"}],
            "aggregator": {"provider": "nous", "model": "claude-opus-4.8"},
        }}}}
        with p1, p2, p3, patch("hermes_cli.config.load_config", return_value=moa_cfg):
            info = runner._format_session_info()
        assert "nous:claude-opus-4.8" in info

    def test_named_custom_provider_keeps_context_pin_without_model_base_url(
        self, runner, tmp_path
    ):
        """Session-reset banner must honor model.context_length for named custom providers.

        Repro: /status shows 262144 from config while the reset banner said
        ``131K tokens (detected)`` because empty model.base_url + runtime URL
        falsely cleared the pin and fell through to the Qwen family default.
        """
        model = "custom-local-agentw/Qwen-AgentWorld-35B-A3B-Q5_K_XL"
        config_yaml = (
            "model:\n"
            f"  default: {model}\n"
            "  provider: custom-local-agentw\n"
            "  context_length: 262144\n"
            "custom_providers:\n"
            "  - name: custom-local-agentw\n"
            "    base_url: http://127.0.0.1:8080/v1\n"
            "    models: {}\n"
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            config_yaml,
            model,
            {
                "provider": "custom-local-agentw",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "",
            },
        )
        with p1, p2, p3, patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[
                {
                    "name": "custom-local-agentw",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "models": {},
                }
            ],
        ), patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda *args, **kwargs: (
                kwargs.get("config_context_length")
                if kwargs.get("config_context_length")
                else 131072
            ),
        ):
            info = runner._format_session_info()
        assert "262K" in info
        assert "config" in info
        assert "131K" not in info


class TestResetNoticeSessionInfo:
    """#59003: the auto-reset banner must report the serving profile's config,
    not the multiplexer's base config."""

    _RUNTIME = {"provider": "", "base_url": "", "api_key": ""}

    def _source(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id="123", user_id="u1",
            profile="planner",
        )

    def _homes(self, tmp_path):
        base = tmp_path / "base"
        profile = tmp_path / "profiles" / "planner"
        profile.mkdir(parents=True)
        base.mkdir()
        base.joinpath("config.yaml").write_text(
            "model:\n  default: base-model\n  provider: custom\n  context_length: 1000\n")
        profile.joinpath("config.yaml").write_text(
            "model:\n  default: profile-model\n  provider: anthropic\n  context_length: 2000\n")
        return base, profile

    def test_multiplex_uses_profile_config(self, runner, tmp_path):
        from types import SimpleNamespace
        base, profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=True)
        with patch("gateway.run._hermes_home", base), \
             patch.object(GatewayRunner, "_resolve_profile_home_for_source", return_value=profile), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME):
            info = runner._reset_notice_session_info(self._source())
        assert "profile-model" in info
        assert "anthropic" in info
        assert "base-model" not in info


class TestSessionRouteForSource:
    """#130709: lane route lookup mirrors the turn's channel_overrides branch."""

    def _runner_with_override(self, runner, overrides):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides=overrides)},
        )
        return runner

    def test_chat_id_match_returns_model_and_route(self, runner):
        from gateway.config import ChannelOverride
        self._runner_with_override(runner, {"lane1": ChannelOverride(model="lane-model", provider="openrouter")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="lane1", user_id="u1")
        rt = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
              "api_key": "k", "model": "lane-model"}
        with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=dict(rt)):
            model, route = runner._session_route_for_source(source)
        assert model == "lane-model"
        assert route["provider"] == "openrouter"
        assert route["base_url"] == "https://openrouter.ai/api/v1"

    def test_thread_id_match_for_topic_lane(self, runner):
        """Telegram DM topic lanes key on the thread id, not the chat id."""
        from gateway.config import ChannelOverride
        self._runner_with_override(runner, {"thread-9": ChannelOverride(model="topic-model", provider="openrouter")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm-chat", user_id="u1", thread_id="thread-9")
        rt = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "k"}
        with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=dict(rt)):
            model, route = runner._session_route_for_source(source)
        assert model == "topic-model"
        assert route["provider"] == "openrouter"

    def test_parent_id_fallback(self, runner):
        from gateway.config import ChannelOverride
        self._runner_with_override(runner, {"parent-1": ChannelOverride(model="parent-model")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="thread-x", user_id="u1",
                               parent_chat_id="parent-1")
        model, route = runner._session_route_for_source(source)
        assert model == "parent-model"
        assert route is None

    def test_no_override_returns_none_none(self, runner):
        from gateway.config import ChannelOverride
        self._runner_with_override(runner, {"other": ChannelOverride(model="x")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="lane-missing", user_id="u1")
        assert runner._session_route_for_source(source) == (None, None)

    def test_system_prompt_only_override_returns_none_none(self, runner):
        from gateway.config import ChannelOverride
        self._runner_with_override(runner, {"lane1": ChannelOverride(system_prompt="Be brief.")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="lane1", user_id="u1")
        assert runner._session_route_for_source(source) == (None, None)

    def test_broken_provider_falls_back_to_global_without_raising(self, runner):
        """Broken channel provider falls back model and route together.

        Keeping the override model with route=None would pair it with the
        global provider in _resolve_gateway_model_context() (e.g.
        claude-sonnet with anthropic) — an unusable route the turn cannot run.
        """
        from gateway.config import ChannelOverride
        self._runner_with_override(
            runner, {"lane1": ChannelOverride(model="lane-model", provider="bogus-provider")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="lane1", user_id="u1")
        with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   side_effect=RuntimeError("no such provider")):
            model, route = runner._session_route_for_source(source)
        assert (model, route) == (None, None)

    def test_no_config_or_source_never_raises(self, runner):
        runner.config = None
        assert runner._session_route_for_source(None) == (None, None)
        from gateway.config import Platform
        from gateway.session import SessionSource
        assert runner._session_route_for_source(
            SessionSource(platform=Platform.TELEGRAM, chat_id="x", user_id="u")) == (None, None)

    def test_provider_only_override_adopts_bundled_model(self, runner):
        from gateway.config import ChannelOverride
        self._runner_with_override(runner, {"lane1": ChannelOverride(provider="openrouter")})
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="lane1", user_id="u1")
        rt = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
              "api_key": "k", "model": "bundled-model"}
        with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=dict(rt)):
            model, route = runner._session_route_for_source(source)
        assert model == "bundled-model"
        assert route["provider"] == "openrouter"


class TestFormatSessionInfoChannelOverrides:
    """#130709: /new banner must report the lane route, not the global default."""

    _GLOBAL_RUNTIME = {"provider": "anthropic", "base_url": "", "api_key": ""}
    _LANE_RUNTIME = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "k"}

    def _source(self, **kwargs):
        from gateway.config import Platform
        from gateway.session import SessionSource
        base = dict(platform=Platform.TELEGRAM, chat_id="lane1", user_id="u1")
        base.update(kwargs)
        return SessionSource(**base)

    def _runner(self, runner, overrides):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides=overrides)},
        )
        return runner

    def test_lane_override_rendered_in_session_info(self, runner, tmp_path):
        from gateway.config import ChannelOverride
        self._runner(runner, {"lane1": ChannelOverride(model="lane-model", provider="openrouter")})
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model:\n  default: global-model\n  provider: anthropic\n")
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(self._GLOBAL_RUNTIME)), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   return_value=dict(self._LANE_RUNTIME)):
            info = runner._format_session_info(self._source())
        assert "lane-model" in info
        assert "openrouter" in info
        assert "global-model" not in info

    def test_no_override_falls_back_to_global(self, runner, tmp_path):
        from gateway.config import ChannelOverride
        self._runner(runner, {"other": ChannelOverride(model="elsewhere")})
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model:\n  default: global-model\n  provider: anthropic\n")
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(self._GLOBAL_RUNTIME)):
            info = runner._format_session_info(self._source(chat_id="plain"))
        assert "global-model" in info
        assert "anthropic" in info

    def test_broken_override_still_renders_global(self, runner, tmp_path):
        from gateway.config import ChannelOverride
        self._runner(runner, {"lane1": ChannelOverride(model="lane-model", provider="bogus")})
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model:\n  default: global-model\n  provider: anthropic\n")
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(self._GLOBAL_RUNTIME)), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   side_effect=RuntimeError("bad provider")):
            info = runner._format_session_info(self._source())
        # Broken lane falls back model and route together: exact global pairing.
        assert "global-model" in info
        assert "anthropic" in info
        assert "lane-model" not in info

    def test_broken_claude_sonnet_override_reports_global_pairing(self, runner, tmp_path):
        """Reviewer repro: claude-sonnet lane + anthropic global must not mix."""
        from gateway.config import ChannelOverride
        self._runner(
            runner, {"lane1": ChannelOverride(model="claude-sonnet", provider="bogus-lane")})
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model:\n  default: global-model\n  provider: anthropic\n")
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(self._GLOBAL_RUNTIME)), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   side_effect=RuntimeError("bad provider")):
            info = runner._format_session_info(self._source())
        assert "global-model" in info
        assert "anthropic" in info
        assert "claude-sonnet" not in info

    def test_reset_notice_forwards_source(self, runner, tmp_path):
        """_reset_notice_session_info passes its source so the /new banner sees the lane."""
        from gateway.config import ChannelOverride
        self._runner(runner, {"lane1": ChannelOverride(model="lane-model", provider="openrouter")})
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model:\n  default: global-model\n  provider: anthropic\n")
        seen = {}

        real_format = runner._format_session_info

        def _spy(source=None):
            seen["source"] = source
            return real_format(source)

        runner._format_session_info = _spy
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(self._GLOBAL_RUNTIME)), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   return_value=dict(self._LANE_RUNTIME)):
            info = runner._reset_notice_session_info(self._source())
        assert seen.get("source") is not None
        assert getattr(seen["source"], "chat_id", "") == "lane1"
        assert "lane-model" in info


class TestFooterPreviewChannelOverrides:
    """#130709: /footer preview must use the lane model, not the global default."""

    def _event(self, source, arg="on"):
        from gateway.platforms.event import MessageEvent
        event = MessageEvent(text=f"/footer {arg}".strip(), source=source, message_id="m1")
        # Handler parses event.message (legacy); mirror text there.
        event.message = f"/footer {arg}".strip()
        return event

    def _runner(self, overrides):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides=overrides)},
        )
        return runner

    def _source(self, **kwargs):
        from gateway.config import Platform
        from gateway.session import SessionSource
        base = dict(platform=Platform.TELEGRAM, chat_id="lane1", user_id="u1")
        base.update(kwargs)
        return SessionSource(**base)

    def test_footer_preview_uses_lane_model(self, tmp_path):
        import asyncio
        from gateway.config import ChannelOverride
        runner = self._runner({"lane1": ChannelOverride(model="lane-model", provider="openrouter")})
        source = self._source()
        event = self._event(source, "on")
        captured = {}
        user_config = {"model": {"default": "global-model"},
                       "display": {"runtime_footer": {"enabled": False}}}

        def _fake_format(*, model, **kwargs):
            captured["model"] = model
            return f"_{model}_"

        with patch("gateway.run._load_gateway_config", return_value=user_config), \
             patch.object(GatewayRunner, "_display_config_target",
                          return_value=(tmp_path / "config.yaml", "telegram")), \
             patch("gateway.slash_commands._write_raw_config_leaf", return_value=None), \
             patch("gateway.runtime_footer.format_runtime_footer", side_effect=_fake_format), \
             patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                   return_value={"provider": "openrouter",
                                 "base_url": "https://openrouter.ai/api/v1", "api_key": "k"}):
            out = asyncio.run(runner._handle_footer_command(event))
        assert captured.get("model") == "lane-model"
        assert "lane-model" in out
        assert "global-model" not in out

    def test_footer_preview_falls_back_to_global_without_override(self, tmp_path):
        import asyncio
        from gateway.config import ChannelOverride
        runner = self._runner({"other": ChannelOverride(model="elsewhere")})
        source = self._source(chat_id="plain")
        event = self._event(source, "on")
        captured = {}
        user_config = {"model": {"default": "global-model"},
                       "display": {"runtime_footer": {"enabled": False}}}

        def _fake_format(*, model, **kwargs):
            captured["model"] = model
            return f"_{model}_"

        with patch("gateway.run._load_gateway_config", return_value=user_config), \
             patch.object(GatewayRunner, "_display_config_target",
                          return_value=(tmp_path / "config.yaml", "telegram")), \
             patch("gateway.slash_commands._write_raw_config_leaf", return_value=None), \
             patch("gateway.runtime_footer.format_runtime_footer", side_effect=_fake_format):
            out = asyncio.run(runner._handle_footer_command(event))
        assert captured.get("model") == "global-model"
        assert "global-model" in out


