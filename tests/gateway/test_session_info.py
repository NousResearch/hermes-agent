"""Tests for GatewayRunner._format_session_info — session config surfacing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture()
def runner():
    """Create a bare GatewayRunner without __init__."""
    gr = GatewayRunner.__new__(GatewayRunner)
    gr._sessions = {}
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


def _source(room_id="!room:example.org"):
    return SessionSource(
        platform=Platform.MATRIX,
        chat_id=room_id,
        user_id="u1",
    )


def _matrix_config(room_id=None, override=None):
    overrides = {room_id: override} if room_id and override else {}
    return GatewayConfig(
        platforms={
            Platform.MATRIX: PlatformConfig(
                enabled=True,
                channel_overrides=overrides,
            )
        }
    )


class TestFormatSessionInfo:
    def test_includes_model_name(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
            "anthropic/claude-opus-4.6",
            {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "k",
            },
        )
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "claude-opus-4.6" in info

    def test_config_context_length(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: test-model\n  context_length: 32768\n",
            "test-model",
            {"provider": "custom", "base_url": "", "api_key": ""},
        )
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "32K" in info
        assert "config" in info

    def test_default_fallback_hint(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: unknown-model-xyz\n",
            "unknown-model-xyz",
            {"provider": "", "base_url": "", "api_key": ""},
        )
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "256K" in info
        assert "model.context_length" in info

    def test_local_endpoint_shown(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n"
            "  default: qwen3:8b\n"
            "  provider: custom\n"
            "  base_url: http://localhost:11434/v1\n"
            "  context_length: 8192\n",
            "qwen3:8b",
            {
                "provider": "custom",
                "base_url": "http://localhost:11434/v1",
                "api_key": "",
            },
        )
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "localhost:11434" in info
        assert "8K" in info

    def test_named_custom_provider_keeps_context_pin_without_model_base_url(
        self, runner, tmp_path
    ):
        """Session-reset banner must honor model.context_length for named custom providers."""
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
        with (
            p1,
            p2,
            p3,
            patch(
                "hermes_cli.config.get_compatible_custom_providers",
                return_value=[
                    {
                        "name": "custom-local-agentw",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "models": {},
                    }
                ],
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                side_effect=lambda *args, **kwargs: (
                    kwargs.get("config_context_length") or 131072
                ),
            ),
        ):
            info = runner._format_session_info()
        assert "262K" in info
        assert "config" in info
        assert "131K" not in info

    def test_channel_override_uses_effective_route(self, runner, tmp_path):
        room_id = "!channel-route:example.org"
        runner.config = _matrix_config(
            room_id,
            ChannelOverride(model="routed-model", provider="routed-provider"),
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n"
            "  default: global-model\n"
            "  provider: global-provider\n"
            "  context_length: 128000\n",
            "global-model",
            {
                "provider": "global-provider",
                "base_url": "https://global.example/v1",
                "api_key": "global-key",
            },
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
                    "api_key": "routed-key",
                },
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=65536,
            ),
        ):
            info = runner._format_session_info(_source(room_id))

        assert "routed-model" in info
        assert "routed-provider" in info
        assert "channel override" in info
        assert "global-model" not in info
        assert "global-provider" not in info

    def test_effective_global_route_is_one_atomic_snapshot(self, runner, tmp_path):
        runner.config = _matrix_config()
        first = {
            "model": "model-r1",
            "provider": "provider-r1",
            "base_url": "",
            "api_key": None,
        }
        second = {
            "model": "model-r2",
            "provider": "provider-r2",
            "base_url": "https://second.example/v1",
            "api_key": "key-r2",
        }
        p1, p2, _p3 = _patch_info(
            tmp_path,
            "model:\n  default: configured-model\n",
            "configured-model",
            first,
        )
        with (
            p1,
            p2,
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                side_effect=[first, second],
            ) as resolve_runtime,
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=32768,
            ) as detect_context,
        ):
            info = runner._format_session_info(_source())

        assert "model-r1" in info
        assert resolve_runtime.call_count == 1
        assert detect_context.call_args.kwargs["provider"] == "provider-r1"
        assert detect_context.call_args.kwargs["base_url"] == ""
        assert detect_context.call_args.kwargs["api_key"] == ""

    def test_falsey_routed_fields_do_not_inherit_global_values(
        self, runner, tmp_path
    ):
        room_id = "!routed-empty:example.org"
        runner.config = _matrix_config(
            room_id,
            ChannelOverride(model="routed-model", provider="routed-provider"),
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n"
            "  default: global-model\n"
            "  provider: global-provider\n"
            "  base_url: http://localhost:9999/v1\n",
            "global-model",
            {
                "provider": "global-provider",
                "base_url": "http://localhost:9999/v1",
                "api_key": "global-key",
            },
        )
        with (
            p1,
            p2,
            p3,
            patch(
                "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                return_value={
                    "provider": "routed-provider",
                    "base_url": "",
                    "api_key": None,
                },
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=32768,
            ) as detect_context,
        ):
            info = runner._format_session_info(_source(room_id))

        assert "routed-model" in info
        assert "routed-provider" in info
        assert "localhost:9999" not in info
        assert detect_context.call_args.kwargs["provider"] == "routed-provider"
        assert detect_context.call_args.kwargs["base_url"] == ""
        assert detect_context.call_args.kwargs["api_key"] == ""

    def test_same_model_different_provider_discards_global_context_pin(
        self, runner, tmp_path
    ):
        room_id = "!same-model-route:example.org"
        runner.config = _matrix_config(
            room_id,
            ChannelOverride(model="shared-model", provider="routed-provider"),
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n"
            "  default: shared-model\n"
            "  provider: global-provider\n"
            "  context_length: 128000\n"
            "custom_providers:\n"
            "  - name: routed-provider\n"
            "    base_url: https://routed.example/v1\n"
            "    models:\n"
            "      shared-model:\n"
            "        context_length: 65536\n",
            "shared-model",
            {
                "provider": "global-provider",
                "base_url": "",
                "api_key": "global-key",
            },
        )

        def detected(_model, **kwargs):
            return kwargs.get("config_context_length") or 262144

        with (
            p1,
            p2,
            p3,
            patch(
                "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                return_value={
                    "provider": "routed-provider",
                    "base_url": "https://routed.example/v1",
                    "api_key": "routed-key",
                },
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                side_effect=detected,
            ) as detect_context,
        ):
            info = runner._format_session_info(_source(room_id))

        assert "routed-provider" in info
        assert "65K" in info
        assert "128K" not in info
        assert detect_context.call_args.kwargs["config_context_length"] == 65536

    def test_session_override_takes_priority_over_channel(self, runner, tmp_path):
        room_id = "!session-route:example.org"
        runner.config = _matrix_config(
            room_id,
            ChannelOverride(model="room-model", provider="room-provider"),
        )
        source = _source(room_id)
        session_key = runner._session_key_for_source(source)
        runner._session_state(session_key).conversation.model_override = {
            "model": "session-model",
            "provider": "session-provider",
            "api_key": "session-key",
            "base_url": "https://session.example/v1",
            "api_mode": None,
        }
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: global-model\n  provider: global-provider\n",
            "global-model",
            {
                "provider": "global-provider",
                "base_url": "",
                "api_key": "global-key",
            },
        )
        with (
            p1,
            p2,
            p3,
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=32768,
            ),
        ):
            info = runner._format_session_info(source)

        assert "session-model" in info
        assert "session-provider" in info
        assert "session override" in info
        assert "room-model" not in info
        assert "global-model" not in info

    def test_source_without_override_stays_global(self, runner, tmp_path):
        runner.config = _matrix_config()
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: global-model\n  provider: global-provider\n",
            "global-model",
            {
                "provider": "global-provider",
                "base_url": "",
                "api_key": "global-key",
            },
        )
        with p1, p2, p3:
            info = runner._format_session_info(_source())
        assert "global-model" in info
        assert "global-provider" in info
        assert "Route:" not in info

    def test_source_resolution_failure_uses_global_fallback(self, runner, tmp_path):
        runner.config = _matrix_config()
        p1, p2, _p3 = _patch_info(
            tmp_path,
            "model:\n  default: configured-model\n",
            "configured-model",
            {},
        )
        with (
            p1,
            p2,
            patch.object(
                GatewayRunner,
                "_resolve_session_agent_runtime",
                side_effect=RuntimeError("route unavailable"),
            ),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value={
                    "model": "fallback-model",
                    "provider": "fallback-provider",
                    "base_url": "",
                    "api_key": "fallback-key",
                },
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=32768,
            ),
        ):
            info = runner._format_session_info(_source())
        assert "fallback-model" in info
        assert "fallback-provider" in info


class TestResetNoticeSessionInfo:
    """Auto-reset banners must use both profile and channel/session routing."""

    def _source(self):
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            user_id="u1",
            profile="planner",
        )

    def _homes(self, tmp_path):
        base = tmp_path / "base"
        profile = tmp_path / "profiles" / "planner"
        profile.mkdir(parents=True)
        base.mkdir()
        base.joinpath("config.yaml").write_text(
            "model:\n"
            "  default: base-model\n"
            "  provider: custom\n"
            "  context_length: 1000\n"
        )
        profile.joinpath("config.yaml").write_text(
            "model:\n"
            "  default: profile-model\n"
            "  provider: anthropic\n"
            "  context_length: 2000\n"
        )
        return base, profile

    def test_multiplex_uses_profile_config(self, runner, tmp_path):
        base, profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=True, platforms={})
        with (
            patch("gateway.run._hermes_home", base),
            patch.object(
                GatewayRunner,
                "_resolve_profile_home_for_source",
                return_value=profile,
            ),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value={
                    "provider": "anthropic",
                    "base_url": "",
                    "api_key": "profile-key",
                },
            ),
        ):
            info = runner._reset_notice_session_info(self._source())
        assert "profile-model" in info
        assert "anthropic" in info
        assert "base-model" not in info

    def test_single_profile_uses_base_config(self, runner, tmp_path):
        base, _profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=False, platforms={})
        with (
            patch("gateway.run._hermes_home", base),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value={
                    "provider": "custom",
                    "base_url": "",
                    "api_key": "base-key",
                },
            ),
        ):
            info = runner._reset_notice_session_info(self._source())
        assert "base-model" in info
        assert "custom" in info
        assert "profile-model" not in info

    def test_reset_notice_passes_source_to_formatter(self, runner):
        runner.config = SimpleNamespace(multiplex_profiles=False)
        source = self._source()
        with patch.object(
            GatewayRunner,
            "_format_session_info",
            return_value="info",
        ) as format_info:
            assert runner._reset_notice_session_info(source) == "info"
        format_info.assert_called_once_with(source)
