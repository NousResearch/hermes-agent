"""Shared fixtures for the AG-UI adapter test package."""

import logging

import pytest

from tests.agui_adapter.test_modernization_probe import peer


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers = list(root.handlers)
    loggers = [logging.getLogger(name) for name in ("", "httpx", "httpcore", "openai")]
    levels = [logger.level for logger in loggers]
    try:
        yield
    finally:
        for handler in root.handlers:
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)
