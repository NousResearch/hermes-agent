"""Regression tests for the Mattermost platform plugin's check_fn.

The Mattermost adapter's ``check_mattermost_requirements()`` is registered as
the platform's ``check_fn``.  This function is invoked on every
``load_gateway_config()`` call (dozens of times during normal gateway
operation).  It must therefore be a *silent* predicate — returning True/False
without logging — otherwise every user without Mattermost configured gets
their logs flooded with WARNING messages every few seconds.

Sibling of the raft check_fn fix (PR #49240).
"""

import logging
from unittest.mock import patch

import pytest


@pytest.fixture
def mm_check():
    """Import check_mattermost_requirements."""
    from plugins.platforms.mattermost.adapter import check_mattermost_requirements

    return check_mattermost_requirements


def test_check_returns_false_when_token_missing(mm_check, monkeypatch):
    """check_fn returns False when MATTERMOST_TOKEN is not set."""
    monkeypatch.delenv("MATTERMOST_TOKEN", raising=False)
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    assert mm_check() is False


def test_check_returns_false_when_url_missing(mm_check, monkeypatch):
    """check_fn returns False when MATTERMOST_URL is not set."""
    monkeypatch.setenv("MATTERMOST_TOKEN", "tok-abc")
    monkeypatch.delenv("MATTERMOST_URL", raising=False)
    assert mm_check() is False


def test_check_returns_false_when_aiohttp_missing(mm_check, monkeypatch):
    """check_fn returns False when aiohttp is not installed."""
    monkeypatch.setenv("MATTERMOST_TOKEN", "tok-abc")
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    with patch.dict("sys.modules", {"aiohttp": None}):
        assert mm_check() is False


def test_check_returns_true_when_all_present(mm_check, monkeypatch):
    """check_fn returns True when token, URL, and aiohttp are available."""
    monkeypatch.setenv("MATTERMOST_TOKEN", "tok-abc")
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    import types

    fake_aiohttp = types.ModuleType("aiohttp")
    with patch.dict("sys.modules", {"aiohttp": fake_aiohttp}):
        assert mm_check() is True


def test_check_silent_when_token_missing(mm_check, monkeypatch, caplog):
    """check_fn must NOT log a WARNING when MATTERMOST_TOKEN is missing."""
    monkeypatch.delenv("MATTERMOST_TOKEN", raising=False)
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.mattermost.adapter"):
        mm_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_mattermost_requirements must be silent (no WARNING logs), "
        f"but emitted: {[r.getMessage() for r in warnings]}"
    )


def test_check_silent_when_url_missing(mm_check, monkeypatch, caplog):
    """check_fn must NOT log a WARNING when MATTERMOST_URL is missing."""
    monkeypatch.setenv("MATTERMOST_TOKEN", "tok-abc")
    monkeypatch.delenv("MATTERMOST_URL", raising=False)
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.mattermost.adapter"):
        mm_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_mattermost_requirements must be silent (no WARNING logs), "
        f"but emitted: {[r.getMessage() for r in warnings]}"
    )


def test_check_silent_when_aiohttp_missing(mm_check, monkeypatch, caplog):
    """check_fn must NOT log a WARNING when aiohttp is missing."""
    monkeypatch.setenv("MATTERMOST_TOKEN", "tok-abc")
    monkeypatch.setenv("MATTERMOST_URL", "https://mm.example.com")
    with patch.dict("sys.modules", {"aiohttp": None}):
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.mattermost.adapter"):
            mm_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_mattermost_requirements must be silent (no WARNING logs), "
        f"but emitted: {[r.getMessage() for r in warnings]}"
    )
