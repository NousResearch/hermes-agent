"""Regression tests for ``gateway.run._int_env``.

The gateway reads ``HERMES_MAX_ITERATIONS`` from the environment in three
places — the startup budget log, the background-task agent spawn, and the
per-turn agent spawn.  The previous bare ``int(os.getenv(...))`` calls
raised ``ValueError`` if a user's ``.env`` carried a typo such as
``HERMES_MAX_ITERATIONS=abc`` or an empty value left over after a stale
edit, which aborted every gateway turn for that platform.

``_int_env`` mirrors the existing ``_float_env`` helper so a misconfigured
env var falls back to the default instead of crashing the agent loop.
"""

from __future__ import annotations

import pytest

from gateway.run import _int_env


@pytest.mark.parametrize("raw", ["abc", "100k", "1.5", "  ", "1,000", "-"])
def test_int_env_falls_back_on_malformed(monkeypatch, raw):
    """Non-integer values must fall back to the default, not raise."""
    monkeypatch.setenv("HERMES_TEST_INT_ENV", raw)
    assert _int_env("HERMES_TEST_INT_ENV", 90) == 90


def test_int_env_falls_back_on_empty(monkeypatch):
    """Empty string (common after a botched ``.env`` edit) falls back."""
    monkeypatch.setenv("HERMES_TEST_INT_ENV", "")
    assert _int_env("HERMES_TEST_INT_ENV", 90) == 90


def test_int_env_falls_back_when_unset(monkeypatch):
    """Unset env var returns the default."""
    monkeypatch.delenv("HERMES_TEST_INT_ENV", raising=False)
    assert _int_env("HERMES_TEST_INT_ENV", 90) == 90


def test_int_env_parses_valid_value(monkeypatch):
    """A well-formed integer is honored."""
    monkeypatch.setenv("HERMES_TEST_INT_ENV", "500")
    assert _int_env("HERMES_TEST_INT_ENV", 90) == 500


def test_int_env_parses_negative_value(monkeypatch):
    """Negative integers parse — caller is responsible for clamping."""
    monkeypatch.setenv("HERMES_TEST_INT_ENV", "-1")
    assert _int_env("HERMES_TEST_INT_ENV", 90) == -1


def test_int_env_default_is_returned_as_int(monkeypatch):
    """The default is coerced to int so callers can pass numeric literals freely."""
    monkeypatch.setenv("HERMES_TEST_INT_ENV", "abc")
    result = _int_env("HERMES_TEST_INT_ENV", 90)
    assert isinstance(result, int)
    assert result == 90


def test_int_env_rejects_float_string(monkeypatch):
    """``int("1.5")`` raises ValueError — must fall back, not silently truncate."""
    monkeypatch.setenv("HERMES_TEST_INT_ENV", "1.5")
    assert _int_env("HERMES_TEST_INT_ENV", 90) == 90


def test_max_iterations_env_uses_int_env_helper():
    """The three ``HERMES_MAX_ITERATIONS`` reads in gateway/run.py must route
    through ``_int_env`` so a malformed value can never abort an agent turn.

    Guards against regressions that reintroduce ``int(os.getenv(...))``
    against this env var.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    text = src.read_text(encoding="utf-8")

    bare_int_calls = re.findall(
        r'int\(\s*os\.getenv\(\s*["\']HERMES_MAX_ITERATIONS["\']',
        text,
    )
    assert bare_int_calls == [], (
        f"Found {len(bare_int_calls)} bare int(os.getenv('HERMES_MAX_ITERATIONS', ...)) "
        f"call(s) in gateway/run.py. Use _int_env() instead so malformed values fall back "
        f"to the default rather than raising ValueError mid-turn."
    )
