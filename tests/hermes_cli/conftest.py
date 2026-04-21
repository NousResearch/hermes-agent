"""Fixtures for hermes_cli tests."""

import logging
import pytest


@pytest.fixture(autouse=True)
def setup_logger():
    """Set up logger to capture WARNING and above messages."""
    logger = logging.getLogger("hermes_cli.config")
    logger.setLevel(logging.DEBUG)
    yield
    logger.setLevel(logging.WARNING)
