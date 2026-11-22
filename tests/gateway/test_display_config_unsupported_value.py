"""#142368: an unsupported ``display.tool_progress`` value must not fail open into the noisiest mode.

``_norm_tristate`` answered every unrecognised string with its *on* choice, so an attempted
``errors`` mode flipped Telegram's quiet ``off`` default to ``all`` — and reported the value as
``explicit=True``, which also suppressed the platform tier it was meant to inherit. Both surfaces
that share the normaliser are covered here: ``tool_progress`` and ``live_status``.
"""

from __future__ import annotations

import logging

import pytest


def _resolve(display: dict, platform: str = "telegram", env: str | None = None) -> tuple[str, bool]:
    from gateway.display_config import resolve_tool_progress

    return resolve_tool_progress({"display": display}, platform, env)


@pytest.fixture(autouse=True)
def _plugin_platform_tiers():
    """Telegram/Slack defaults ship on their ``PlatformEntry``; the resolver reads the registry."""
    from hermes_cli.plugins import discover_plugins

    discover_plugins()


@pytest.fixture(autouse=True)
def _fresh_warning_memory():
    from gateway import display_config

    def clear():
        warned = getattr(display_config, "_UNSUPPORTED_DISPLAY_WARNED", None)
        if warned is not None:  # absent before the fix, so the sabotage run still asserts
            warned.clear()

    clear()
    yield
    clear()


class TestUnsupportedToolProgress:
    def test_global_typo_keeps_the_quiet_platform_default(self):
        """Telegram's tier override is ``off``; a typo must not turn it into ``all``."""
        assert _resolve({"tool_progress": "errors"}) == ("off", False)

    def test_platform_typo_keeps_the_quiet_platform_default(self):
        assert _resolve({"platforms": {"telegram": {"tool_progress": "errors"}}}) == ("off", False)

    def test_typo_is_not_reported_as_an_explicit_choice(self):
        assert _resolve({"tool_progress": "errors"}, platform="api_server") == ("all", False)

    def test_typo_in_the_legacy_env_bridge_is_not_explicit(self):
        assert _resolve({}, platform="api_server", env="errors") == ("all", False)

    def test_invalid_platform_value_falls_through_to_a_valid_global_one(self):
        display = {"tool_progress": "new", "platforms": {"telegram": {"tool_progress": "erors"}}}
        assert _resolve(display) == ("new", True)

    def test_the_invalid_value_is_named_once_in_the_log(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gateway.display_config"):
            for _ in range(3):
                _resolve({"tool_progress": "errors"})
        named = [r.getMessage() for r in caplog.records if "errors" in r.getMessage()]
        assert len(named) == 1
        assert "tool_progress" in named[0] and "telegram" in named[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("off", "off"),
        ("new", "new"),
        ("all", "all"),
        ("verbose", "verbose"),
        ("log", "log"),
        ("OFF", "off"),
        (" New ", "new"),
        ("on", "all"),
        ("true", "all"),
        ("yes", "all"),
        (True, "all"),
        (False, "off"),
    ],
)
def test_documented_values_and_boolean_aliases_still_resolve(value, expected):
    assert _resolve({"tool_progress": value}, platform="api_server") == (expected, True)


class TestUnsupportedLiveStatus:
    """``live_status`` shares the tristate normaliser, so it fails open the same way."""

    def test_unknown_value_falls_back_to_the_resolved_default(self):
        from gateway.display_config import resolve_display_setting

        assert resolve_display_setting({"display": {"live_status": "verbose"}}, "slack", "live_status") == "full"

    def test_supported_choice_still_resolves(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"live_status": "verb"}}
        assert resolve_display_setting(config, "slack", "live_status") == "verb"
