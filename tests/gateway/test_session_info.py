"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import pytest
from contextlib import nullcontext
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


class TestChannelOverrideSessionInfo:
    """The /new and auto-reset banners must advertise the channel override's
    model/provider (and that provider's runtime endpoint), not the global default."""

    _GLOBAL_RUNTIME = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "global-key",
    }
    _OVERRIDE_RUNTIME = {
        "provider": "anthropic",
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key": "override-key",
    }

    def _runner_with_override(self, runner, chat_id="123", **override_kwargs):
        from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
        runner.config = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={chat_id: ChannelOverride(**override_kwargs)},
            ),
        })
        return runner

    def _source(self, chat_id="123", thread_id=None):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1",
            thread_id=thread_id,
        )

    def test_override_model_and_provider_shown(self, runner, tmp_path):
        self._runner_with_override(runner, model="override-model", provider="anthropic")
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n  provider: openrouter\n",
            "base-model", self._GLOBAL_RUNTIME)
        with p1, p2, p3, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value=self._OVERRIDE_RUNTIME,
        ):
            info = runner._format_session_info(self._source())
        assert "override-model" in info
        assert "anthropic" in info
        assert "base-model" not in info

    def test_no_source_uses_base_model(self, runner, tmp_path):
        self._runner_with_override(runner, model="override-model")
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n",
            "base-model", self._GLOBAL_RUNTIME)
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "base-model" in info
        assert "override-model" not in info

    def test_unmatched_chat_uses_base_model(self, runner, tmp_path):
        self._runner_with_override(runner, chat_id="999", model="override-model")
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n",
            "base-model", self._GLOBAL_RUNTIME)
        with p1, p2, p3:
            info = runner._format_session_info(self._source(chat_id="123"))
        assert "base-model" in info
        assert "override-model" not in info

    def test_override_provider_survives_runtime_failure(self, runner, tmp_path):
        self._runner_with_override(runner, model="override-model", provider="anthropic")
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n  context_length: 4096\n",
            "base-model", self._GLOBAL_RUNTIME)
        with p1, p2, p3, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            side_effect=RuntimeError("no creds"),
        ):
            info = runner._format_session_info(self._source())
        assert "override-model" in info
        assert "anthropic" in info

    def test_reset_notice_helper_uses_channel_override(self, runner, tmp_path):
        """Production path: _reset_notice_session_info(source) must pass source through."""
        self._runner_with_override(runner, model="override-model", provider="anthropic")
        # Keep profile scoping a no-op so this asserts the override path, not multiplex homes.
        runner.config.multiplex_profiles = False
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n  provider: openrouter\n",
            "base-model", self._GLOBAL_RUNTIME)
        with p1, p2, p3, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value=self._OVERRIDE_RUNTIME,
        ), patch.object(GatewayRunner, "_standalone_launch_scope", return_value=nullcontext()):
            info = runner._reset_notice_session_info(self._source())
        assert "override-model" in info
        assert "anthropic" in info
        assert "base-model" not in info

    def test_override_runtime_endpoint_not_global(self, runner, tmp_path):
        """Context probe / endpoint must come from the override provider, not the global route."""
        self._runner_with_override(runner, model="override-model", provider="anthropic")
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: base-model\n  provider: openrouter\n  context_length: 8192\n",
            "base-model",
            self._GLOBAL_RUNTIME,
        )
        with p1, p2, p3, patch(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            return_value=self._OVERRIDE_RUNTIME,
        ) as for_provider:
            info = runner._format_session_info(self._source())
        for_provider.assert_called_once()
        assert for_provider.call_args.args[0] == "anthropic"
        assert "127.0.0.1:4000" in info
        assert "openrouter.ai" not in info
        assert "anthropic" in info


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


class TestProviderDisplayLabel:
    """A bare ``custom`` banner cell names the saved endpoint serving the session.

    Runtime resolution folds every named custom endpoint into the literal adapter id
    ``custom``, so a user with several saved endpoints read ``custom`` for all of them.
    """

    _PROVIDERS = {
        "workbuddy": {"api": "http://192.0.2.10:7863/v1", "models": {"cn:deepseek-v4.1-flash": {}}},
        "traework": {"api": "http://192.0.2.10:7864/v1", "models": {"cn:minimax-m3.1-flash-max": {}}},
    }

    def _info(self, runner, tmp_path, provider, base_url, model):
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n", model,
            {"provider": provider, "base_url": base_url, "api_key": ""})
        config = {"providers": self._PROVIDERS}
        with p1, p2, p3, patch("hermes_cli.config.load_config", return_value=config):
            return runner._format_session_info()

    def test_named_custom_endpoint_is_named(self, runner, tmp_path):
        info = self._info(runner, tmp_path, "custom", "http://192.0.2.10:7863/v1", "cn:deepseek-v4.1-flash")
        assert "custom:workbuddy" in info

    def test_shared_url_resolves_by_model(self, runner, tmp_path):
        """Two entries on one proxy: the entry that declares the model wins, not the first."""
        shared = {"a-first": {"api": "http://192.0.2.20:9000/v1", "models": {"model-a": {}}},
                  "z-second": {"api": "http://192.0.2.20:9000/v1", "models": {"model-b": {}}}}
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n", "model-b",
            {"provider": "custom", "base_url": "http://192.0.2.20:9000/v1", "api_key": ""})
        with p1, p2, p3, patch("hermes_cli.config.load_config", return_value={"providers": shared}):
            info = runner._format_session_info()
        assert "custom:z-second" in info
        assert "custom:a-first" not in info

    def test_unconfigured_endpoint_keeps_bare_custom(self, runner, tmp_path):
        info = self._info(runner, tmp_path, "custom", "http://192.0.2.99:1234/v1", "unknown-model")
        assert "custom: " not in info and ": custom" in info
        assert "custom:" not in info

    def test_builtin_provider_unchanged(self, runner, tmp_path):
        info = self._info(runner, tmp_path, "openrouter", "https://openrouter.ai/api/v1", "gpt-5.5")
        assert "openrouter" in info

    def test_empty_provider_falls_back_to_openrouter(self, runner, tmp_path):
        info = self._info(runner, tmp_path, "", "", "gpt-5.5")
        assert "openrouter" in info

    def test_unreadable_config_keeps_bare_custom(self, runner, tmp_path):
        """Lookup failure must narrow the banner to what it showed before, never raise."""
        info = self._info(runner, tmp_path, "custom", "http://192.0.2.10:7863/v1", "cn:deepseek-v4.1-flash")
        assert "workbuddy" in info  # sanity: lookup works when config is readable
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: base-model\n", "cn:deepseek-v4.1-flash",
            {"provider": "custom", "base_url": "http://192.0.2.10:7863/v1", "api_key": ""})
        with p1, p2, p3, patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad yaml")):
            info = runner._format_session_info()
        assert "custom" in info
        assert "workbuddy" not in info
