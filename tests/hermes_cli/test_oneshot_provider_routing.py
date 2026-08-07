"""Regression tests for issue #74214's sibling gap: the oneshot
(-z/--oneshot) path bypasses HermesCLI entirely, so it needs its own
provider:model / custom:<name>:<model> parsing applied before provider
detection and runtime resolution -- otherwise the compound string reached
resolve_runtime_provider() unsplit, silently sending the prompt to the
default provider instead of the named custom endpoint.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


class _StopAtResolve(Exception):
    """Raised by the mocked resolve_runtime_provider() to short-circuit
    _run_agent() right after the call we want to inspect, before it
    reaches agent construction (which needs much heavier mocking that's
    irrelevant to what this fix changes)."""


def _capture_resolve_call(monkeypatch):
    captured = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        raise _StopAtResolve()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _fake_resolve
    )
    return captured


class TestOneshotProviderModelRouting:
    def test_custom_provider_triple_syntax_supplies_split_requested_and_model(
        self, monkeypatch, tmp_path
    ):
        """The exact case from review: custom:<name>:<model> must supply
        requested='custom:<name>' and the stripped target model to the
        resolver, not the compound string."""
        from hermes_cli.oneshot import _run_agent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"model": {}},
        )
        captured = _capture_resolve_call(monkeypatch)

        with pytest.raises(_StopAtResolve):
            _run_agent(
                "prompt", model="custom:jetson-vllm:nemotron-nano-30b",
            )

        assert captured.get("requested") == "custom:jetson-vllm", captured
        assert captured.get("target_model") == "nemotron-nano-30b", captured

    def test_explicit_provider_flag_wins_over_embedded_provider(
        self, monkeypatch, tmp_path
    ):
        """--provider must take precedence over a provider: prefix in the
        model string, matching the HermesCLI (-m/--model) fix's
        precedence rule."""
        from hermes_cli.oneshot import _run_agent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"model": {}},
        )
        captured = _capture_resolve_call(monkeypatch)

        with pytest.raises(_StopAtResolve):
            _run_agent(
                "prompt",
                model="custom:jetson-vllm:nemotron-nano-30b",
                provider="deepseek",
            )

        assert captured.get("requested") == "deepseek", captured
        # The model string is left unsplit when an explicit --provider wins.
        assert captured.get("target_model") == "custom:jetson-vllm:nemotron-nano-30b", captured

    def test_plain_model_without_provider_prefix_unaffected(
        self, monkeypatch, tmp_path
    ):
        """A normal model name (no colon, or a colon that isn't a known
        provider prefix) must not be touched -- this fix must not regress
        the common case, including the existing detect_provider_for_model
        auto-detection path."""
        from hermes_cli.oneshot import _run_agent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"model": {}},
        )
        monkeypatch.setattr(
            "hermes_cli.models.detect_provider_for_model",
            lambda model, current: None,
        )
        captured = _capture_resolve_call(monkeypatch)

        with pytest.raises(_StopAtResolve):
            _run_agent("prompt", model="claude-sonnet-4.5")

        assert captured.get("target_model") == "claude-sonnet-4.5", captured
