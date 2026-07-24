"""Env flags accept 'on' as truthy consistently (salvage of #2863).

Behavior contract: every env-driven enable flag in gateway config coerces
through the shared TRUTHY_STRINGS set, so "on" behaves like "1"/"true"/"yes".
"""

import os
from unittest.mock import patch

from utils import TRUTHY_STRINGS, env_var_enabled


def test_truthy_strings_include_on():
    assert "on" in TRUTHY_STRINGS


def test_env_var_enabled_accepts_on():
    with patch.dict(os.environ, {"WHATSAPP_ENABLED": "on"}):
        assert env_var_enabled("WHATSAPP_ENABLED") is True


def test_env_var_enabled_default_respected():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SIGNAL_IGNORE_STORIES", None)
        assert env_var_enabled("SIGNAL_IGNORE_STORIES", "true") is True
        assert env_var_enabled("SIGNAL_IGNORE_STORIES") is False


def test_gateway_config_flags_use_shared_helper():
    """Invariant: no env-flag site in gateway/config.py hand-rolls a truthy
    set that omits 'on'."""
    import inspect

    import gateway.config as gc

    src = inspect.getsource(gc)
    for pattern in ('in {"true", "1", "yes"}', 'in ("true", "1", "yes")'):
        assert pattern not in src, f"hand-rolled truthy set without 'on': {pattern}"


def test_desktop_gate_accepts_on():
    from tools.close_terminal_tool import check_close_terminal_requirements
    from tools.read_terminal_tool import check_read_terminal_requirements

    with patch.dict(os.environ, {"HERMES_DESKTOP": "on"}):
        assert check_read_terminal_requirements() is True
        assert check_close_terminal_requirements() is True
    with patch.dict(os.environ, {"HERMES_DESKTOP": "off"}):
        assert check_read_terminal_requirements() is False
