"""Tests for MoA slot_runtime api_mode propagation (issue #54379).

Verify that _slot_runtime passes the resolved api_mode through to call_llm,
so reference slots using providers that require a specific API surface
(e.g. Copilot GPT-5.x → codex_responses) get routed correctly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


class TestSlotRuntimeApiMode:
    """_slot_runtime should include api_mode when resolve_runtime_provider returns it."""

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_includes_api_mode(self, mock_resolve):
        """api_mode from resolve_runtime_provider is forwarded in output dict."""
        mock_resolve.return_value = {
            "provider": "copilot",
            "model": "gpt-5.5",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "test-key",
            "api_mode": "codex_responses",
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "copilot", "model": "gpt-5.5"})
        assert result["api_mode"] == "codex_responses"
        assert result["base_url"] == "https://api.githubcopilot.com"
        assert result["api_key"] == "test-key"

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_omits_api_mode_when_absent(self, mock_resolve):
        """When resolve_runtime_provider does not return api_mode, output omits it."""
        mock_resolve.return_value = {
            "provider": "openai",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "openai", "model": "gpt-4o"})
        assert "api_mode" not in result

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_omits_api_mode_when_empty(self, mock_resolve):
        """Empty string api_mode is treated as absent."""
        mock_resolve.return_value = {
            "provider": "copilot",
            "model": "gpt-5.5",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "test-key",
            "api_mode": "",
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "copilot", "model": "gpt-5.5"})
        assert "api_mode" not in result

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_includes_request_override_extra_body(self, mock_resolve):
        """Custom-provider extra_body is forwarded in call_llm's shape."""
        mock_resolve.return_value = {
            "provider": "custom",
            "model": "qwen3.7-max",
            "base_url": "https://dashscope.example/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
            "request_overrides": {
                "extra_body": {
                    "enable_thinking": False,
                    "reasoning": {"effort": "none"},
                }
            },
        }
        from agent.moa_loop import _slot_runtime

        result = _slot_runtime({"provider": "dashscope", "model": "qwen3.7-max"})
        assert result["extra_body"] == {
            "enable_thinking": False,
            "reasoning": {"effort": "none"},
        }


def test_run_reference_passes_slot_extra_body(monkeypatch):
    """Reference advisors should receive custom provider extra_body."""
    from agent import moa_loop

    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("advisor")

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "custom",
            "model": "qwen3.7-max",
            "base_url": "https://dashscope.example/v1",
            "api_key": "test-key",
            "extra_body": {"enable_thinking": False},
        },
    )
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(moa_loop, "_maybe_apply_moa_cache_control", lambda messages, runtime: messages)

    label, text, _usage = moa_loop._run_reference(
        {"provider": "dashscope", "model": "qwen3.7-max"},
        [{"role": "user", "content": "hello"}],
    )

    assert label == "dashscope:qwen3.7-max"
    assert text == "advisor"
    assert captured["extra_body"] == {"enable_thinking": False}


def test_moa_aggregator_merges_slot_extra_body_with_caller_override(tmp_path, monkeypatch):
    """Aggregator calls should merge slot defaults without duplicate kwargs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: closed
  presets:
    closed:
      enabled: true
      reference_models: []
      aggregator:
        provider: dashscope
        model: qwen3.7-max
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent import moa_loop

    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("acted")

    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "custom",
            "model": "qwen3.7-max",
            "base_url": "https://dashscope.example/v1",
            "api_key": "test-key",
            "extra_body": {
                "enable_thinking": False,
                "metadata": {"source": "slot"},
            },
        },
    )
    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    facade = moa_loop.MoAChatCompletions("closed")
    facade.create(
        model="closed",
        messages=[{"role": "user", "content": "hello"}],
        extra_body={
            "metadata": {"source": "caller"},
            "reasoning": {"effort": "none"},
        },
    )

    assert captured["extra_body"] == {
        "enable_thinking": False,
        "metadata": {"source": "caller"},
        "reasoning": {"effort": "none"},
    }


class TestCallLlmApiMode:
    """call_llm should accept and forward api_mode parameter."""

    def test_call_llm_accepts_api_mode_kwarg(self):
        """call_llm signature includes api_mode parameter."""
        import inspect
        from agent.auxiliary_client import call_llm

        sig = inspect.signature(call_llm)
        assert "api_mode" in sig.parameters
        assert sig.parameters["api_mode"].default is None
