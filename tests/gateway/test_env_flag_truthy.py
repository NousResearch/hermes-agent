"""Env flags accept 'on' as truthy consistently (salvage of #2863).

Behavior contract: every env-driven enable flag in gateway config coerces
through the shared TRUTHY_STRINGS set, so "on" behaves like "1"/"true"/"yes".
"""

import os
from unittest.mock import patch

from utils import TRUTHY_STRINGS, env_var_enabled


def test_truthy_strings_include_on():
    assert "on" in TRUTHY_STRINGS


def test_env_var_enabled_default_respected():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SIGNAL_IGNORE_STORIES", None)
        assert env_var_enabled("SIGNAL_IGNORE_STORIES", "true") is True
        assert env_var_enabled("SIGNAL_IGNORE_STORIES") is False


