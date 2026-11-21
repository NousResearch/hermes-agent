"""Bounded, redacted catalog bootstrap diagnostics, never inherited console output."""
import contextvars
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from agent.monitoring.redaction import redact_bounded

_home = contextvars.ContextVar('catalog_diagnostic_home', default=None)
_lock = threading.Lock()


class _Sink(RotatingFileHandler):
    def handleError(self, record):
        # Diagnostics are best-effort. logging's default prints private data to stderr.
        return


def write(home, text):
    """Bound bytes on disk; retain no profile handles or terminal fallback."""
    try:
        with _lock:
            path = Path(home) / 'logs/skill-catalog.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = _Sink(path, maxBytes=262144, backupCount=1)
            try:
                record = logging.LogRecord('skill-catalog', logging.INFO, '', 0,
                                           redact_bounded(text, limit=2048), (), None)
                handler.emit(record)
            finally:
                handler.close()
    except OSError:
        return


class _Contain(logging.Filter):
    def filter(self, record):
        home = _home.get()
        if home is None:
            return True
        write(home, record.getMessage())
        return False


def enter(home):
    logger = logging.getLogger('hermes_cli.plugins')
    with _lock:
        if not any(isinstance(item, _Contain) for item in logger.filters):
            logger.addFilter(_Contain())
    return _home.set(home)


def leave(token):
    _home.reset(token)
