"""Regression test for #50120.

A malformed HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS must not crash the Telegram
message handler. The grace read must go through the hardened ``_float_env``
helper, like every other timeout env read in ``gateway/run.py``.
"""

import importlib
import pathlib
import re

from gateway.run import _float_env

VAR = "HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS"


def test_float_env_handles_malformed_value(monkeypatch):
    monkeypatch.setenv(VAR, "abc")
    assert _float_env(VAR, 3.0) == 3.0


def test_float_env_handles_blank_value(monkeypatch):
    monkeypatch.setenv(VAR, "   ")
    assert _float_env(VAR, 3.0) == 3.0


def test_float_env_parses_valid_value(monkeypatch):
    monkeypatch.setenv(VAR, "5.5")
    assert _float_env(VAR, 3.0) == 5.5


def test_grace_read_routes_through_float_env():
    """Pin the call site: the grace read must use _float_env, not bare float()."""
    src = pathlib.Path(importlib.import_module("gateway.run").__file__).read_text()

    assert re.search(
        r"_float_env\(\s*[\"']HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS[\"']", src
    ), "grace read should route through _float_env"

    assert not re.search(
        r"float\(\s*os\.getenv\(\s*[\"']HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS[\"']",
        src,
    ), "bare float(os.getenv(...)) for grace must be removed"
