"""Process-wide gateway tool-call log handler."""

import logging
import threading

_tool_call_logger_lock = threading.Lock()


def _tool_call_logger() -> logging.Logger:
    """Process-wide ``hermes.tool_calls`` Logger + one RotatingFileHandler on logs/tool_calls.log.
    Named Loggers live in ``logging.Logger.manager.loggerDict`` forever, so the former per-turn name
    (``hermes.tool_calls.<id(log_queue)>``) leaked one Logger per logged turn (#62950); a single
    shared handler also keeps concurrent turns from double-writing lines."""
    tool_logger = logging.getLogger("hermes.tool_calls")
    with _tool_call_logger_lock:
        if not tool_logger.handlers:
            from logging.handlers import RotatingFileHandler
            from agent.redact import RedactingFormatter
            from gateway.run import _hermes_home

            log_dir = _hermes_home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                log_dir / "tool_calls.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
            )
            handler.setFormatter(RedactingFormatter("%(message)s"))
            tool_logger.setLevel(logging.INFO)
            tool_logger.propagate = False
            tool_logger.addHandler(handler)
    return tool_logger
