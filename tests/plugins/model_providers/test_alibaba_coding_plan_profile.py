"""Unit tests for the Alibaba Coding Plan provider profile's reasoning wiring.

Bug class (verified against the live Aliyun gateway, 2026-08-08): the generic
``_build_call_kwargs`` fallback emits OpenRouter-shaped
``extra_body.reasoning = {"enabled": ..., "effort": ...}`` for providers whose
profile has no reasoning hook. Aliyun's compatible-mode gateway does not
reject that object — it silently blackholes the request: zero response bytes,
no error, no RST, until the client gives up (observed hangs of 25min–7.8h;
reproduced with byte-identical curl replays on three client stacks). The flat
top-level ``reasoning_effort`` field is honored (200 + reasoning_tokens).

These tests pin the profile's wire-shape contract so the OpenRouter-shaped
object can never leak onto the Aliyun wire again, without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def alibaba_profile():
    """Resolve the registered profile through the real discovery path."""
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("alibaba-coding-plan")
    assert profile is not None, "alibaba-coding-plan provider profile must be registered"
    return profile


class TestAlibabaReasoningWireShape:
    """``build_api_kwargs_extras`` produces Aliyun's accepted wire format."""

    def test_no_preference_omits_reasoning(self, alibaba_profile):
        """No reasoning_config → send nothing; server default applies."""
        extra_body, top_level = alibaba_profile.build_api_kwargs_extras(
            reasoning_config=None, model="qwen3.8-max"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_enabled_sends_flat_reasoning_effort(self, alibaba_profile):
        """Enabled+effort → flat top-level ``reasoning_effort`` — the shape
        Aliyun provably honors — never the OpenRouter-shaped object."""
        extra_body, top_level = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"}, model="qwen3.8-max"
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "medium"}

    def test_enabled_without_effort_defaults_medium(self, alibaba_profile):
        extra_body, top_level = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="qwen3.8-max"
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "medium"}

    def test_disabled_omits_reasoning(self, alibaba_profile):
        """``enabled=False`` → omit the field entirely.

        The compatible-mode wire has no documented thinking-off parameter;
        omitting keeps the server default instead of risking another
        unrecognized-field blackhole.
        """
        extra_body, top_level = alibaba_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="qwen3.8-max"
        )
        assert extra_body == {}
        assert top_level == {}


class TestAlibabaEndToEndCallKwargs:
    """Through ``_build_call_kwargs``: the OpenRouter-shaped fallback object
    must not leak for this provider (the actual 2026-08 blackhole path)."""

    def _kwargs(self, reasoning_config):
        from agent.auxiliary_client import _build_call_kwargs

        return _build_call_kwargs(
            provider="alibaba-coding-plan",
            model="qwen3.8-max",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1800,
            task="moa_reference",
            reasoning_config=reasoning_config,
            base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )

    def test_flat_effort_on_wire_no_reasoning_object(self):
        kw = self._kwargs({"enabled": True, "effort": "high"})
        assert kw.get("reasoning_effort") == "high"
        assert "reasoning" not in (kw.get("extra_body") or {})

    def test_no_reasoning_config_sends_nothing(self):
        kw = self._kwargs(None)
        assert "reasoning_effort" not in kw
        assert "reasoning" not in (kw.get("extra_body") or {})
