"""Regression tests for the raft platform plugin's check_fn.

The raft platform adapter's ``check_raft_requirements()`` is registered as
the platform's ``check_fn``. This function is invoked on every
``load_gateway_config()`` call (dozens of times during normal gateway
operation). It must therefore be a *silent* predicate — returning True/False
without logging — otherwise every user without the ``raft`` CLI installed
gets their logs flooded with WARNING messages every few seconds.

See: https://github.com/NousResearch/hermes-agent/issues/49234
"""

import logging
from unittest.mock import patch

import pytest


@pytest.fixture
def raft_check():
    """Import check_raft_requirements fresh (adapter self-manages sys.path)."""
    from plugins.platforms.raft.adapter import check_raft_requirements

    return check_raft_requirements


def test_check_returns_false_when_raft_cli_missing(raft_check):
    """check_fn returns False when raft CLI is not in PATH."""
    with patch("plugins.platforms.raft.adapter.shutil.which", return_value=None), \
         patch("plugins.platforms.raft.adapter.AIOHTTP_AVAILABLE", True):
        assert raft_check() is False


def test_check_returns_false_when_aiohttp_missing(raft_check):
    """check_fn returns False when aiohttp dependency is unavailable."""
    with patch("plugins.platforms.raft.adapter.AIOHTTP_AVAILABLE", False):
        assert raft_check() is False


def test_check_returns_true_when_all_deps_present(raft_check):
    """check_fn returns True when all dependencies are available."""
    with patch("plugins.platforms.raft.adapter.shutil.which", return_value="/usr/bin/raft"), \
         patch("plugins.platforms.raft.adapter.AIOHTTP_AVAILABLE", True):
        assert raft_check() is True


def test_check_silent_when_raft_cli_missing(raft_check, caplog):
    """check_fn must NOT log a WARNING when raft CLI is missing.

    This is the regression guard for issue #49234 — logging inside check_fn
    causes log spam because the function is called on every config load.
    """
    with patch("plugins.platforms.raft.adapter.shutil.which", return_value=None), \
         patch("plugins.platforms.raft.adapter.AIOHTTP_AVAILABLE", True):
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.raft.adapter"):
            raft_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_raft_requirements must be silent (no WARNING logs), "
        f"but emitted: {[r.getMessage() for r in warnings]}"
    )


def test_check_silent_when_aiohttp_missing(raft_check, caplog):
    """check_fn must NOT log a WARNING when aiohttp is missing."""
    with patch("plugins.platforms.raft.adapter.AIOHTTP_AVAILABLE", False):
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.raft.adapter"):
            raft_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_raft_requirements must be silent (no WARNING logs), "
        f"but emitted: {[r.getMessage() for r in warnings]}"
    )
