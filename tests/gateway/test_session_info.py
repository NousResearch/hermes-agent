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


class TestFormatSessionInfoReasoning:
    """The ◆ Reasoning: row reflects the effective reasoning effort."""

    def _run(self, runner, tmp_path, reasoning_cfg):
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: test-model\n  provider: openrouter\n",
            "test-model", {"provider": "openrouter", "base_url": "", "api_key": ""})
        with p1, p2, p3, patch.object(
            type(runner), "_load_reasoning_config", return_value=reasoning_cfg
        ):
            return runner._format_session_info()

    def test_explicit_effort_shown(self, runner, tmp_path):
        info = self._run(runner, tmp_path, {"enabled": True, "effort": "xhigh"})
        assert "◆ Reasoning: xhigh" in info

    def test_none_when_disabled(self, runner, tmp_path):
        info = self._run(runner, tmp_path, {"enabled": False})
        assert "◆ Reasoning: none" in info

    def test_default_when_unset(self, runner, tmp_path):
        # parse_reasoning_effort returns None for unset → default (medium)
        info = self._run(runner, tmp_path, None)
        assert "◆ Reasoning: medium (default)" in info

    def test_row_ordered_between_provider_and_context(self, runner, tmp_path):
        info = self._run(runner, tmp_path, {"enabled": True, "effort": "high"})
        lines = info.splitlines()
        prov = next(i for i, l in enumerate(lines) if l.startswith("◆ Provider:"))
        reas = next(i for i, l in enumerate(lines) if l.startswith("◆ Reasoning:"))
        ctx = next(i for i, l in enumerate(lines) if l.startswith("◆ Context:"))
        assert prov < reas < ctx

    def test_reasoning_resolution_failure_omits_row(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path, "model:\n  default: test-model\n  provider: openrouter\n",
            "test-model", {"provider": "openrouter", "base_url": "", "api_key": ""})
        with p1, p2, p3, patch.object(
            type(runner), "_load_reasoning_config", side_effect=RuntimeError("boom")
        ):
            info = runner._format_session_info()
        # Banner still renders; the reasoning row is simply omitted.
        assert "◆ Model:" in info
        assert "◆ Context:" in info
        assert "Reasoning" not in info


class TestReasoningEffortLabel:
    """The shared _reasoning_effort_label helper — single source of truth for
    the reasoning-effort display string used by both the /new reset banner and
    the /model switch confirmation."""

    def test_none_is_medium_default(self):
        assert GatewayRunner._reasoning_effort_label(None) == "medium (default)"

    def test_disabled_is_none(self):
        assert GatewayRunner._reasoning_effort_label({"enabled": False}) == "none"

    def test_explicit_effort(self):
        assert GatewayRunner._reasoning_effort_label(
            {"enabled": True, "effort": "xhigh"}
        ) == "xhigh"

    def test_enabled_without_effort_defaults_medium(self):
        assert GatewayRunner._reasoning_effort_label({"enabled": True}) == "medium"
