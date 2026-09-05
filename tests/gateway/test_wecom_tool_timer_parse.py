"""Tests for the WeCom adapter's ``tool_timer_enabled`` parser (Issue B).

The tool-timer animation sends progress status over the WeCom transport, so
its opt-in flag must fail *closed*: only the canonical truthy tokens enable
it.  The previous blocklist implementation ("not in false/0/no/off") was
fail-open — typos and unknown strings silently turned the feature on.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import WeComAdapter


def _supports_tool_timer(extra: dict | None) -> bool:
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra=extra))
    return adapter.SUPPORTS_TOOL_TIMER


class TestToolTimerEnabledParser:
    """Strict allowlist parsing for extra.tool_timer_enabled."""

    def test_omitted_is_false(self):
        assert _supports_tool_timer(None) is False
        assert _supports_tool_timer({}) is False

    @pytest.mark.parametrize(
        "value",
        ["true", "True", "TRUE", "1", "yes", "YES", "on", "On"],
    )
    def test_truthy_tokens_enable(self, value):
        assert _supports_tool_timer({"tool_timer_enabled": value}) is True

    @pytest.mark.parametrize(
        "value",
        ["false", "False", "0", "no", "off", "flase", "disabled", "garbage", ""],
    )
    def test_everything_else_disables(self, value):
        assert _supports_tool_timer({"tool_timer_enabled": value}) is False

    def test_native_bool_true(self):
        assert _supports_tool_timer({"tool_timer_enabled": True}) is True

    def test_native_bool_false(self):
        assert _supports_tool_timer({"tool_timer_enabled": False}) is False

    def test_whitespace_padded_truthy(self):
        assert _supports_tool_timer({"tool_timer_enabled": "  true  "}) is True
