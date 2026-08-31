"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import pytest
from unittest.mock import patch

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


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

    def test_includes_model_name(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
                                  "anthropic/claude-opus-4.6",
                                  {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "k"})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "claude-opus-4.6" in info


    def test_config_context_length(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  context_length: 32768\n",
                                  "test-model",
                                  {"provider": "custom", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "32K" in info
        assert "config" in info

    def test_default_fallback_hint(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: unknown-model-xyz\n",
                                  "unknown-model-xyz",
                                  {"provider": "", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "256K" in info
        assert "model.context_length" in info

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

    @pytest.mark.parametrize(
        ("source", "override_key"),
        [
            (
                SessionSource(
                    platform=Platform.DISCORD,
                    chat_id="thread-1",
                    thread_id="thread-1",
                    parent_chat_id="parent-1",
                    user_id="u1",
                ),
                "thread-1",
            ),
            (
                SessionSource(
                    platform=Platform.DISCORD,
                    chat_id="thread-1",
                    thread_id="thread-1",
                    parent_chat_id="parent-1",
                    user_id="u1",
                ),
                "parent-1",
            ),
        ],
        ids=("exact-thread-route", "parent-route-inheritance"),
    )
    def test_reset_notice_uses_one_effective_thread_route(
        self, runner, tmp_path, source, override_key
    ):
        """Every banner field must come from the inherited channel/thread route."""
        tmp_path.joinpath("config.yaml").write_text(
            "model:\n"
            "  default: global-model\n"
            "  provider: anthropic\n"
            "  base_url: https://api.anthropic.com\n"
            "  context_length: 12345\n"
        )
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        override_key: ChannelOverride(
                            model="thread-model",
                            provider="thread-provider",
                        )
                    },
                )
            }
        )
        runner._session_key_for_source = lambda _source: None
        route_runtime = {
            "provider": "thread-provider",
            "requested_provider": "thread-provider",
            "base_url": "http://localhost:9000/v1",
            "api_key": "route-key",
            "api_mode": "chat_completions",
        }

        with (
            patch("gateway.run._hermes_home", tmp_path),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value={
                    "provider": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "api_key": "global-key",
                },
            ),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                return_value=route_runtime,
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                side_effect=lambda *args, **kwargs: (
                    65536
                    if kwargs.get("provider") == "thread-provider"
                    and kwargs.get("base_url") == "http://localhost:9000/v1"
                    else 12345
                ),
            ),
        ):
            info = runner._reset_notice_session_info(source)

        assert "thread-model" in info
        assert "thread-provider" in info
        assert "65K" in info
        assert "http://localhost:9000/v1" in info
        assert "global-model" not in info
        assert "anthropic" not in info
