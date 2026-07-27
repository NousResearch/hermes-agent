"""Tests for per-slot max_tokens in MoA reference calls.

Verifies that a ``max_tokens`` field on a reference slot dict takes
precedence over the preset-level ``reference_max_tokens``, and that
slot-level max_tokens=None falls back to the preset-level cap.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRunReferenceSlotMaxTokens:
    """_run_reference should prefer slot-level max_tokens over preset-level."""

    def test_slot_max_tokens_overrides_preset_level(self):
        """When slot has max_tokens, it overrides the preset-level cap."""
        from agent.moa_loop import _run_reference

        captured_kwargs: dict = {}

        def fake_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="advice"))]
            mock_resp.usage = None
            return mock_resp

        slot = {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "max_tokens": 600}

        with patch("agent.moa_loop._slot_runtime", return_value={"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}), \
             patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), \
             patch("agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt: msgs):
            _run_reference(slot, [{"role": "user", "content": "hi"}], max_tokens=2000)

        assert captured_kwargs.get("max_tokens") == 600

    def test_slot_max_tokens_absent_falls_back_to_preset(self):
        """When slot has no max_tokens, the preset-level cap is used."""
        from agent.moa_loop import _run_reference

        captured_kwargs: dict = {}

        def fake_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="advice"))]
            mock_resp.usage = None
            return mock_resp

        slot = {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}

        with patch("agent.moa_loop._slot_runtime", return_value={"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}), \
             patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), \
             patch("agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt: msgs):
            _run_reference(slot, [{"role": "user", "content": "hi"}], max_tokens=2000)

        assert captured_kwargs.get("max_tokens") == 2000

    def test_both_none_means_uncapped(self):
        """When neither slot nor preset has max_tokens, it's None (uncapped)."""
        from agent.moa_loop import _run_reference

        captured_kwargs: dict = {}

        def fake_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="advice"))]
            mock_resp.usage = None
            return mock_resp

        slot = {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}

        with patch("agent.moa_loop._slot_runtime", return_value={"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}), \
             patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), \
             patch("agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt: msgs):
            _run_reference(slot, [{"role": "user", "content": "hi"}], max_tokens=None)

        assert captured_kwargs.get("max_tokens") is None
