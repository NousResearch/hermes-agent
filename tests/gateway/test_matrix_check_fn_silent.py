"""Regression tests for Matrix check_matrix_requirements() silence contract.

The caller in ``gateway/run.py`` ``_create_adapter()`` already emits a
warning when ``check_matrix_requirements()`` returns ``False``.  Logging
inside the function itself duplicates that warning on every
adapter-creation attempt (initial connect, reconnect, profile load).

Credential checks must therefore be silent — return True/False without
emitting WARNING or higher.  This matches the convention enforced for
platform plugin ``check_fn`` callbacks (see raft fix PR #49240 and
mattermost fix PR #49465).
"""

import logging
from unittest.mock import patch

import pytest


@pytest.fixture
def matrix_check():
    from gateway.platforms.matrix import check_matrix_requirements

    return check_matrix_requirements


def test_check_returns_false_when_no_credentials(matrix_check, monkeypatch):
    """check_fn returns False when neither token nor password is set."""
    monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("MATRIX_PASSWORD", raising=False)
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.com")
    assert matrix_check() is False


def test_check_returns_false_when_homeserver_missing(matrix_check, monkeypatch):
    """check_fn returns False when MATRIX_HOMESERVER is not set."""
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token")
    monkeypatch.delenv("MATRIX_HOMESERVER", raising=False)
    assert matrix_check() is False


def test_check_silent_when_no_credentials(matrix_check, monkeypatch, caplog):
    """check_fn must NOT log WARNING+ when credentials are missing.

    The caller in gateway/run.py _create_adapter() already logs a warning
    when check_fn returns False.  Logging here creates duplicate warnings.
    """
    monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("MATRIX_PASSWORD", raising=False)
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.com")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.matrix"):
        matrix_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_matrix_requirements credential checks must be silent "
        f"(no WARNING logs), but emitted: {[r.getMessage() for r in warnings]}"
    )


def test_check_silent_when_homeserver_missing(matrix_check, monkeypatch, caplog):
    """check_fn must NOT log WARNING+ when MATRIX_HOMESERVER is missing."""
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token")
    monkeypatch.delenv("MATRIX_HOMESERVER", raising=False)
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.matrix"):
        matrix_check()

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"check_matrix_requirements credential checks must be silent "
        f"(no WARNING logs), but emitted: {[r.getMessage() for r in warnings]}"
    )
