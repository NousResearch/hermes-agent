"""Tests for per-provider backoff and throttle helpers in conversation_loop."""

import pytest
from agent.conversation_loop import _provider_backoff_params, _provider_min_request_interval


class TestProviderBackoffParams:
    def test_nvidia_overrides(self):
        base, cap = _provider_backoff_params("nvidia")
        assert base == 30.0
        assert cap == 120.0

    def test_nvidia_ignores_caller_defaults(self):
        base, cap = _provider_backoff_params("nvidia", default_base=2.0, default_max=60.0)
        assert base == 30.0
        assert cap == 120.0

    def test_unknown_uses_function_defaults(self):
        base, cap = _provider_backoff_params("nonexistent-provider-xyz")
        assert base == 5.0
        assert cap == 120.0

    def test_unknown_uses_caller_defaults(self):
        base, cap = _provider_backoff_params(
            "nonexistent-provider-xyz", default_base=2.0, default_max=60.0
        )
        assert base == 2.0
        assert cap == 60.0

    def test_empty_provider_string(self):
        base, cap = _provider_backoff_params("")
        assert base == 5.0
        assert cap == 120.0


class TestProviderMinRequestInterval:
    def test_nvidia_has_throttle(self):
        assert _provider_min_request_interval("nvidia") == 2.0

    def test_unknown_no_throttle(self):
        assert _provider_min_request_interval("nonexistent-provider-xyz") == 0.0

    def test_empty_string_no_throttle(self):
        assert _provider_min_request_interval("") == 0.0
