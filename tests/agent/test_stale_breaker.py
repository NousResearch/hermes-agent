"""Tests for the stale-streak circuit-breaker exemption (Bug 3 Path A).

The local-endpoint exemption lives inside :func:`_check_stale_giveup` in
``agent/chat_completion_helpers.py``.  Stub agents carry only ``.base_url``
and ``._consecutive_stale_streams`` — the two attributes the function actually
reads.

Run with::

    .venv/bin/python -m pytest tests/agent/test_stale_breaker.py -q
"""

from __future__ import annotations

import os
import types
from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import _check_stale_giveup


def _make_stub(base_url: str | None, streak: int) -> types.SimpleNamespace:
    """Minimal stub carrying only the attributes _check_stale_giveup reads."""
    return types.SimpleNamespace(
        base_url=base_url,
        _consecutive_stale_streams=streak,
    )


@pytest.fixture(autouse=True)
def _clear_stale_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the default threshold back to 5 so ambient env cannot skew tests."""
    monkeypatch.delenv("HERMES_STREAM_STALE_GIVEUP", raising=False)


# ---------------------------------------------------------------------------
# Local endpoints — exempt from the give-up breaker
# ---------------------------------------------------------------------------

def test_local_http_127_0_0_1_streak_5_no_raise() -> None:
    stub = _make_stub("http://127.0.0.1:8080/v1", 5)
    _check_stale_giveup(stub)  # should be a no-op


def test_local_localhost_11434_streak_10_no_raise() -> None:
    stub = _make_stub("http://localhost:11434/v1", 10)
    _check_stale_giveup(stub)  # should be a no-op


# ---------------------------------------------------------------------------
# Cloud endpoints — subject to the breaker
# ---------------------------------------------------------------------------

def test_cloud_openai_streak_5_raises() -> None:
    stub = _make_stub("https://api.openai.com/v1", 5)
    with pytest.raises(RuntimeError, match="consecutive stale attempts"):
        _check_stale_giveup(stub)


def test_cloud_openai_streak_4_no_raise() -> None:
    stub = _make_stub("https://api.openai.com/v1", 4)
    _check_stale_giveup(stub)  # below the default threshold of 5


# ---------------------------------------------------------------------------
# Missing base_url — treated as cloud (no exemption)
# ---------------------------------------------------------------------------

def test_missing_base_url_streak_5_raises() -> None:
    stub = _make_stub(None, 5)
    # SimpleNamespace with base_url=None is equivalent to no attribute for the
    # ``getattr ... or ""`` chain inside the exemption.
    # Re-create as a real missing-attribute stub:
    stub = MagicMock(spec=["_consecutive_stale_streams"])
    stub._consecutive_stale_streams = 5
    with pytest.raises(RuntimeError, match="consecutive stale attempts"):
        _check_stale_giveup(stub)
