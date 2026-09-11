"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import pytest
from unittest.mock import patch

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

    def test_channel_override_model_wins_over_global_default(self, runner, tmp_path):
        """The /new (and auto-reset) banner must report the channel override model, not the
        global default — the turn would use the override, so the banner must not lie."""
        from gateway.config import Platform, ChannelOverride, PlatformConfig
        from gateway.session import SessionSource
        from types import SimpleNamespace

        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="14930030", user_id="u1",
            thread_id="624616",
        )
        runner.config = SimpleNamespace(
            platforms={
                Platform.TELEGRAM: SimpleNamespace(
                    channel_overrides={"624616": ChannelOverride(model="deepseek/deepseek-v4-pro", provider="commandcode")},
                )
            }
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: xiaomi/mimo-v2.5\n  provider: commandcode\n",
            "xiaomi/mimo-v2.5",
            {"provider": "commandcode", "base_url": "", "api_key": ""},
        )
        with p1, p2, p3:
            info = runner._format_session_info(source)
        assert "deepseek-v4-pro" in info
        assert "mimo-v2.5" not in info
        assert "commandcode" in info

    def test_no_override_falls_back_to_global(self, runner, tmp_path):
        """No matching channel override → the global default still surfaces."""
        from gateway.config import Platform, PlatformConfig
        from gateway.session import SessionSource
        from types import SimpleNamespace

        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="14930030", user_id="u1",
            thread_id="999999",
        )
        runner.config = SimpleNamespace(
            platforms={Platform.TELEGRAM: SimpleNamespace(channel_overrides={})}
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: xiaomi/mimo-v2.5\n  provider: commandcode\n",
            "xiaomi/mimo-v2.5",
            {"provider": "commandcode", "base_url": "", "api_key": ""},
        )
        with p1, p2, p3:
            info = runner._format_session_info(source)
        assert "mimo-v2.5" in info
        assert "commandcode" in info

    def test_plain_dm_chat_no_thread_id_no_override(self, runner, tmp_path):
        """A normal Telegram DM/group chat has no thread_id and no override: the banner must
        render exactly as before the fix — global model, no crash, no override leakage."""
        from gateway.config import Platform, PlatformConfig
        from gateway.session import SessionSource
        from types import SimpleNamespace

        # Plain private DM: chat_id set, thread_id=None, no channel_overrides at all.
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="14930030", user_id="u1",
        )
        runner.config = SimpleNamespace(
            platforms={Platform.TELEGRAM: SimpleNamespace(channel_overrides={})}
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: xiaomi/mimo-v2.5\n  provider: commandcode\n",
            "xiaomi/mimo-v2.5",
            {"provider": "commandcode", "base_url": "", "api_key": ""},
        )
        with p1, p2, p3:
            info = runner._format_session_info(source)
        assert "mimo-v2.5" in info
        assert "commandcode" in info
        assert "◆ Model:" in info
        assert "◆ Provider:" in info

    def test_plain_dm_without_config_attr_does_not_crash(self, runner, tmp_path):
        """A bare runner with no `.config` attribute (test/double edge case) must still render
        the global model — the override lookup is guarded by getattr."""
        from gateway.config import Platform
        from gateway.session import SessionSource

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="14930030", user_id="u1")
        # Deliberately no runner.config set.
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: xiaomi/mimo-v2.5\n  provider: commandcode\n",
            "xiaomi/mimo-v2.5",
            {"provider": "commandcode", "base_url": "", "api_key": ""},
        )
        with p1, p2, p3:
            info = runner._format_session_info(source)
        assert "mimo-v2.5" in info
        assert "commandcode" in info


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

