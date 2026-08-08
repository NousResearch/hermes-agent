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
    def test_slot_explicit_connection_info_forwarded(self, mock_resolve):
        """Slot base_url/api_key reach resolve_runtime_provider as explicit_*."""
        mock_resolve.return_value = {
            "base_url": "https://mirror.example/v1",
            "api_key": "mirror-key",
        }
        import agent.moa_loop as moa
        moa._runtime_cache.clear()

        moa._slot_runtime({
            "provider": "custom", "model": "some-model",
            "base_url": "https://mirror.example/v1", "api_key": "mirror-key",
        })
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs["explicit_base_url"] == "https://mirror.example/v1"
        assert kwargs["explicit_api_key"] == "mirror-key"

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_api_mode_overrides_autodetected(self, mock_resolve):
        """An explicit slot api_mode wins over the resolver's auto-detected one."""
        mock_resolve.return_value = {
            "base_url": "https://x.example/v1",
            "api_key": "k",
            "api_mode": "chat_completions",
        }
        import agent.moa_loop as moa
        moa._runtime_cache.clear()

        result = moa._slot_runtime({
            "provider": "custom", "model": "m",
            "base_url": "https://x.example/v1",
            "api_mode": "anthropic_messages",
        })
        assert result["api_mode"] == "anthropic_messages"

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    def test_slot_runtime_cache_keyed_by_connection_info(self, mock_resolve):
        """Slots sharing provider+model but differing in base_url/api_key must
        NOT share a cache entry — otherwise the first slot resolved poisons
        the cache and every sibling silently inherits its endpoint and
        credentials.
        """

        def _echo_explicit(**kwargs):
            return {
                "base_url": kwargs.get("explicit_base_url") or "http://catalog",
                "api_key": kwargs.get("explicit_api_key") or "catalog-key",
            }

        mock_resolve.side_effect = _echo_explicit
        import agent.moa_loop as moa
        moa._runtime_cache.clear()

        slot_a = {
            "provider": "custom", "model": "shared-model",
            "base_url": "https://a.example/v1", "api_key": "KEY-A",
        }
        slot_b = {
            "provider": "custom", "model": "shared-model",
            "base_url": "https://b.example/v1", "api_key": "KEY-B",
        }
        result_a = moa._slot_runtime(slot_a)
        result_b = moa._slot_runtime(slot_b)
        assert result_a["base_url"] == "https://a.example/v1"
        assert result_b["base_url"] == "https://b.example/v1"
        assert result_a["api_key"] == "KEY-A"
        assert result_b["api_key"] == "KEY-B"

        # The cache itself still works: an identical slot re-resolves from
        # cache without another resolve_runtime_provider call.
        calls_before = mock_resolve.call_count
        again = moa._slot_runtime(dict(slot_a))
        assert again["base_url"] == "https://a.example/v1"
        assert mock_resolve.call_count == calls_before


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


def test_one_shot_aggregate_moa_context_passes_slot_extra_body(monkeypatch):
    """The one-shot `/moa <prompt>` synthesis call (aggregate_moa_context) is
    the third independent MoA call path — its aggregator call receives the
    slot runtime via **agg_runtime, so custom-provider extra_body must flow
    through it too."""
    from agent import moa_loop

    captured_calls = []

    def fake_call_llm(**kwargs):
        captured_calls.append(kwargs)
        return _response("synthesis")

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
    monkeypatch.setattr(
        moa_loop, "_maybe_apply_moa_cache_control", lambda messages, runtime: messages
    )

    result = moa_loop.aggregate_moa_context(
        user_prompt="hello",
        api_messages=[{"role": "user", "content": "hello"}],
        reference_models=[{"provider": "dashscope", "model": "qwen3.7-max"}],
        aggregator={"provider": "dashscope", "model": "qwen3.7-max"},
    )

    assert "synthesis" in result
    agg_calls = [c for c in captured_calls if c.get("task") == "moa_aggregator"]
    assert len(agg_calls) == 1
    assert agg_calls[0]["extra_body"] == {"enable_thinking": False}


class TestCallLlmApiMode:
    """call_llm should accept and forward api_mode parameter."""

    def test_call_llm_accepts_api_mode_kwarg(self):
        """call_llm signature includes api_mode parameter."""
        import inspect
        from agent.auxiliary_client import call_llm

        sig = inspect.signature(call_llm)
        assert "api_mode" in sig.parameters
        assert sig.parameters["api_mode"].default is None
