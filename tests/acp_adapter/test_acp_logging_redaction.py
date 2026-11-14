"""ACP adapter stderr logging must go through RedactingFormatter.

``_setup_logging`` clears root handlers and installs its own stderr handler;
before the fix it used a plain ``logging.Formatter`` — zero redaction on a
surface that logs request/response internals. See issue #77484.
"""

import logging

from acp_adapter.entry import _setup_logging

SECRET = "sk-proj-AbCdEf1234567890SecretValue999"


def test_acp_stderr_handler_redacts_secrets():
    saved_handlers = logging.getLogger().handlers[:]
    saved_level = logging.getLogger().level
    try:
        _setup_logging()
        root = logging.getLogger()
        assert root.handlers, "ACP logging setup installed no handler"
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        record = logging.LogRecord(
            name="acp.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="request failed: OPENROUTER_API_KEY=%s",
            args=(SECRET,),
            exc_info=None,
        )
        out = handler.format(record)
        assert SECRET not in out
        assert "OPENROUTER_API_KEY=" in out
    finally:
        root = logging.getLogger()
        root.handlers.clear()
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
