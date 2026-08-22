"""Tests for the truncation-boost ceiling.

Regression: the continuation-retry path used to cap the boosted ``max_tokens``
at a hard-coded 32 768 — well below the output limits of modern models
(GLM-4.5-Flash: 98 304, GPT-5.x: 131 072, …).  When a long code block or tool
argument truncated, every retry hit the same artificial wall and forced the
agent into the fallback chain, where downstream models started from scratch.

The ceiling now honours, in order:

1. the ``max_tokens`` the caller already sent (``requested_cap``),
2. the model's declared ``limit.output`` from the models.dev cache,
3. a hard-coded floor of 32 768 only when neither is available.
"""

from __future__ import annotations

from unittest.mock import patch
from types import SimpleNamespace

from agent._truncation_boost_cap import resolve_truncation_boost_cap


class TestResolveTruncationBoostCap:
    def test_requested_cap_is_always_honoured(self):
        """An explicit caller-set max_tokens is the floor — we never shrink it."""
        assert resolve_truncation_boost_cap(
            requested_cap=200000, provider="zai", model="glm-5.2"
        ) == 200000

    def test_requested_cap_below_fallback_still_keeps_floor(self):
        """A tiny requested_cap (e.g. 1024) still returns the 32 768 floor so
        the boost has room to grow on subsequent retries."""
        assert resolve_truncation_boost_cap(
            requested_cap=1024, provider="zai", model="glm-5.2"
        ) == 32768

    def test_model_output_limit_used_when_no_requested_cap(self):
        """When the caller didn't send max_tokens, the model's declared
        limit.output from models.dev is the ceiling — not the hard-coded 32K."""
        fake_caps = SimpleNamespace(max_output_tokens=98304)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert resolve_truncation_boost_cap(
                requested_cap=None, provider="zai", model="glm-4.5-flash"
            ) == 98304

    def test_fallback_floor_when_model_unknown(self):
        """Unknown model / cache miss → the 32 768 floor applies."""
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert resolve_truncation_boost_cap(
                requested_cap=None, provider="acme", model="unknown-model"
            ) == 32768

    def test_fallback_floor_when_lookup_raises(self):
        """A cache lookup failure must never block the continuation path."""
        with patch(
            "agent.models_dev.get_model_capabilities", side_effect=RuntimeError("boom")
        ):
            assert resolve_truncation_boost_cap(
                requested_cap=None, provider="zai", model="glm-5.2"
            ) == 32768

    def test_requested_cap_wins_over_smaller_model_limit(self):
        """If the caller asked for more than the model declares, honour the
        caller — they opted in explicitly."""
        fake_caps = SimpleNamespace(max_output_tokens=8192)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert resolve_truncation_boost_cap(
                requested_cap=65536, provider="zai", model="glm-5.2"
            ) == 65536

    def test_none_provider_and_model_returns_floor(self):
        """When both provider and model are None the floor applies."""
        assert resolve_truncation_boost_cap(
            requested_cap=None, provider=None, model=None
        ) == 32768

    def test_requested_cap_at_exact_floor_boundary(self):
        """requested_cap=32768 (exactly the floor) must not shrink."""
        assert resolve_truncation_boost_cap(
            requested_cap=32768, provider="zai", model="glm-5.2"
        ) == 32768

    def test_requested_cap_just_above_floor(self):
        """requested_cap=32769 must not be clamped to the floor."""
        assert resolve_truncation_boost_cap(
            requested_cap=32769, provider="zai", model="glm-5.2"
        ) == 32769

    def test_model_output_zero_falls_through_to_floor(self):
        """A declared limit of 0 (or falsy) is ignored — the floor applies."""
        fake_caps = SimpleNamespace(max_output_tokens=0)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert resolve_truncation_boost_cap(
                requested_cap=None, provider="zai", model="glm-5.2"
            ) == 32768

    def test_model_output_float_is_coerced_to_int(self):
        """Float output limits (e.g. from a future cache schema) are coerced."""
        fake_caps = SimpleNamespace(max_output_tokens=98304.0)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            result = resolve_truncation_boost_cap(
                requested_cap=None, provider="zai", model="glm-4.5-flash"
            )
        assert result == 98304
        assert isinstance(result, int)

    def test_requested_cap_wins_even_when_model_limit_is_larger(self):
        """requested_cap always wins — the caller opted in explicitly, even
        when the model declares a higher limit."""
        fake_caps = SimpleNamespace(max_output_tokens=131072)
        with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
            assert resolve_truncation_boost_cap(
                requested_cap=65536, provider="zai", model="glm-5.2"
            ) == 65536
