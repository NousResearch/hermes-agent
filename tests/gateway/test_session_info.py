"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture()
def runner():
    """Create a bare GatewayRunner without __init__."""
    gr = GatewayRunner.__new__(GatewayRunner)
    gr._session_model_overrides = {}
    gr._last_resolved_model = {}
    return gr


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

    def test_includes_provider(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  provider: openrouter\n",
                                  "test-model",
                                  {"provider": "openrouter", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "openrouter" in info

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

    def test_cloud_endpoint_hidden(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  provider: openrouter\n",
                                  "test-model",
                                  {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "k"})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "Endpoint" not in info

    def test_million_context_format(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  context_length: 1000000\n",
                                  "test-model",
                                  {"provider": "", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "1.0M" in info

    def test_missing_config(self, runner, tmp_path):
        """No config.yaml should not crash."""
        p1, p2, p3 = _patch_info(tmp_path, None,  # don't create config
                                  "anthropic/claude-sonnet-4.6",
                                  {"provider": "openrouter", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "Model" in info
        assert "Context" in info

    def test_runtime_resolution_failure_doesnt_crash(self, runner, tmp_path):
        """If runtime resolution raises, should still produce output."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model:\n  default: test-model\n  context_length: 4096\n")
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run._resolve_gateway_model", return_value="test-model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=RuntimeError("no creds")):
            info = runner._format_session_info()
        assert "4K" in info
        assert "config" in info

    def test_channel_override_used_when_source_provided(self, runner, tmp_path):
        """/new-style banner must show the room route, not global model.default."""
        room_id = "!LpLKQQpGcNTCxhdJWd:chat.homelab.one"
        runner.config = GatewayConfig(
            platforms={
                Platform.MATRIX: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        room_id: ChannelOverride(
                            model="grok-4.5",
                            provider="xai-oauth",
                        ),
                    },
                )
            }
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room_id,
            user_id="u1",
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: global-model\n  provider: openai-codex\n  context_length: 128000\n",
            "global-model",
            {"provider": "openai-codex", "base_url": "", "api_key": "k"},
        )
        with p1, p2, p3, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value={
                "provider": "xai-oauth",
                "base_url": "https://api.x.ai/v1",
                "api_key": "xk",
            },
        ):
            info = runner._format_session_info(source)
        assert "grok-4.5" in info
        assert "xai-oauth" in info
        assert "channel override" in info
        assert "global-model" not in info
        assert "openai-codex" not in info

    def test_channel_override_discards_global_context_for_routed_model(
        self, runner, tmp_path
    ):
        room_id = "!routed-context:server"
        runner.config = GatewayConfig(
            platforms={
                Platform.MATRIX: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        room_id: ChannelOverride(
                            model="routed-model",
                            provider="routed-provider",
                        )
                    },
                )
            }
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room_id,
            user_id="u1",
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n"
            "  default: global-model\n"
            "  provider: global-provider\n"
            "  context_length: 128000\n"
            "custom_providers:\n"
            "  - name: routed-provider\n"
            "    model: routed-model\n"
            "    context_length: 65536\n",
            "global-model",
            {"provider": "global-provider", "base_url": "", "api_key": "gk"},
        )
        with (
            p1,
            p2,
            p3,
            patch(
                "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                return_value={
                    "provider": "routed-provider",
                    "base_url": "https://routed.example/v1",
                    "api_key": "rk",
                },
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                side_effect=lambda _model, **kwargs: kwargs.get(
                    "config_context_length"
                )
                or 262144,
            ) as detect_context,
        ):
            info = runner._format_session_info(source)

        assert "routed-model" in info
        assert "65K" in info
        assert "128K" not in info
        assert "config" in info
        assert detect_context.call_args.kwargs["config_context_length"] == 65536

    def test_session_override_takes_priority_over_channel(self, runner, tmp_path):
        room_id = "!room:server"
        runner.config = GatewayConfig(
            platforms={
                Platform.MATRIX: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        room_id: ChannelOverride(
                            model="room-model",
                            provider="room-provider",
                        ),
                    },
                )
            }
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room_id,
            user_id="u1",
        )
        session_key = runner._session_key_for_source(source)
        runner._session_model_overrides[session_key] = {
            "model": "session-model",
            "provider": "session-provider",
            "api_key": "sk",
            "base_url": "https://session.example/v1",
            "api_mode": None,
        }
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: global-model\n  provider: openrouter\n  context_length: 8000\n",
            "global-model",
            {"provider": "openrouter", "base_url": "", "api_key": "k"},
        )
        with p1, p2, p3, patch.object(
            GatewayRunner, "_rehydrate_session_model_override", return_value=None
        ):
            info = runner._format_session_info(source)
        assert "session-model" in info
        assert "session-provider" in info
        assert "session override" in info
        assert "room-model" not in info
        assert "global-model" not in info

    def test_source_without_channel_override_stays_global(self, runner, tmp_path):
        runner.config = GatewayConfig(
            platforms={
                Platform.MATRIX: PlatformConfig(enabled=True, channel_overrides={})
            }
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!no-override:server",
            user_id="u1",
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: global-model\n  provider: openrouter\n  context_length: 16000\n",
            "global-model",
            {"provider": "openrouter", "base_url": "", "api_key": "k"},
        )
        with p1, p2, p3:
            info = runner._format_session_info(source)
        assert "global-model" in info
        assert "openrouter" in info
        assert "Route:" not in info


class TestResetNoticeSessionInfo:
    """#59003: the auto-reset banner must report the serving profile's config,
    not the multiplexer's base config.
    """

    _RUNTIME = {"provider": "", "base_url": "", "api_key": ""}

    def _source(self):
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
        base, profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=True, platforms={})
        with patch("gateway.run._hermes_home", base), \
             patch.object(GatewayRunner, "_resolve_profile_home_for_source", return_value=profile), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME):
            info = runner._reset_notice_session_info(self._source())
        assert "profile-model" in info
        assert "anthropic" in info
        assert "base-model" not in info

    def test_single_profile_uses_base_config(self, runner, tmp_path):
        base, _profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=False, platforms={})
        with patch("gateway.run._hermes_home", base), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME):
            info = runner._reset_notice_session_info(self._source())
        assert "base-model" in info
        assert "profile-model" not in info

    def test_reset_notice_passes_source_for_channel_route(self, runner, tmp_path):
        """_reset_notice_session_info must forward source into the formatter."""
        room_id = "!room:server"
        runner.config = GatewayConfig(
            multiplex_profiles=False,
            platforms={
                Platform.MATRIX: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        room_id: ChannelOverride(
                            model="room-model",
                            provider="room-provider",
                        ),
                    },
                )
            },
        )
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room_id,
            user_id="u1",
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: global-model\n  provider: openrouter\n  context_length: 8000\n",
            "global-model",
            {"provider": "openrouter", "base_url": "", "api_key": "k"},
        )
        with p1, p2, p3, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value={
                "provider": "room-provider",
                "base_url": "",
                "api_key": "rk",
            },
        ):
            info = runner._reset_notice_session_info(source)
        assert "room-model" in info
        assert "room-provider" in info
        assert "channel override" in info
