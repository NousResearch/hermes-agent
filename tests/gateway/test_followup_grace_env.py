"""Regression tests for HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS env parsing.

A misconfigured ``HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS`` (typo, stray
whitespace, empty string) must NOT crash the gateway message handler.
Before the fix the value was read with a bare ``float(os.getenv(...))`` in
``_handle_message``; a non-numeric value raised ``ValueError`` on the hot
path that decides whether a rapid Telegram follow-up interrupts the running
agent -- dropping the message.  Every sibling timeout read in the gateway
(``HERMES_AGENT_TIMEOUT``, ``HERMES_AGENT_NOTIFY_INTERVAL``) already routes
through the hardened ``_float_env`` helper; this read was the lone straggler
that the ``_float_env`` extraction (411f586c6) missed.

These tests assert the helper-backed contract that the call site now uses.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gateway.run import _float_env

ENV_NAME = "HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS"
DEFAULT = 3.0


def _restore_env(prev):
    if prev is None:
        os.environ.pop(ENV_NAME, None)
    else:
        os.environ[ENV_NAME] = prev


def test_followup_grace_valid_value_parsed():
    """A well-formed value is parsed as a float."""
    prev = os.environ.get(ENV_NAME)
    try:
        os.environ[ENV_NAME] = "5.5"
        assert _float_env(ENV_NAME, DEFAULT) == 5.5
    finally:
        _restore_env(prev)


def test_followup_grace_invalid_value_falls_back_without_crash():
    """A non-numeric value falls back to the default instead of raising.

    This is the exact regression: a bare ``float("abc")`` raised ValueError
    on the message-handling hot path and dropped the inbound message.
    """
    prev = os.environ.get(ENV_NAME)
    try:
        os.environ[ENV_NAME] = "abc"
        # Must not raise.
        assert _float_env(ENV_NAME, DEFAULT) == DEFAULT
    finally:
        _restore_env(prev)


def test_followup_grace_empty_string_falls_back():
    """An empty/whitespace value falls back to the default."""
    prev = os.environ.get(ENV_NAME)
    try:
        os.environ[ENV_NAME] = ""
        assert _float_env(ENV_NAME, DEFAULT) == DEFAULT
    finally:
        _restore_env(prev)


def test_followup_grace_unset_uses_default():
    """An unset env var uses the default."""
    prev = os.environ.get(ENV_NAME)
    try:
        os.environ.pop(ENV_NAME, None)
        assert _float_env(ENV_NAME, DEFAULT) == DEFAULT
    finally:
        _restore_env(prev)


def test_followup_grace_zero_disables_grace():
    """An explicit 0 is preserved (the call site treats >0 as 'grace on')."""
    prev = os.environ.get(ENV_NAME)
    try:
        os.environ[ENV_NAME] = "0"
        assert _float_env(ENV_NAME, DEFAULT) == 0.0
    finally:
        _restore_env(prev)


if __name__ == "__main__":
    test_followup_grace_valid_value_parsed()
    test_followup_grace_invalid_value_falls_back_without_crash()
    test_followup_grace_empty_string_falls_back()
    test_followup_grace_unset_uses_default()
    test_followup_grace_zero_disables_grace()
    print("All follow-up grace env tests passed.")
