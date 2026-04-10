"""Tests for proxy credential env passthrough integration."""

from __future__ import annotations

import os

import pytest

from tools.env_passthrough import (
    register_env_passthrough,
    is_env_passthrough,
    clear_env_passthrough,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clean_passthrough():
    """Reset passthrough state before/after each test."""
    clear_env_passthrough()
    reset_config_cache()
    yield
    clear_env_passthrough()
    reset_config_cache()


class TestProxyCredentialPassthrough:
    """Proxy-declared env vars must survive the sanitize step."""

    def test_register_makes_var_passthrough(self):
        register_env_passthrough(["CLOUDFLARE_API_TOKEN"])
        assert is_env_passthrough("CLOUDFLARE_API_TOKEN")

    def test_unregistered_var_not_passthrough(self):
        assert not is_env_passthrough("SOME_RANDOM_VAR")

    def test_multiple_vars_registered(self):
        register_env_passthrough(["VAR_A", "VAR_B", "VAR_C"])
        assert is_env_passthrough("VAR_A")
        assert is_env_passthrough("VAR_B")
        assert is_env_passthrough("VAR_C")

    def test_clear_removes_registered(self):
        register_env_passthrough(["MY_TOKEN"])
        assert is_env_passthrough("MY_TOKEN")
        clear_env_passthrough()
        assert not is_env_passthrough("MY_TOKEN")
