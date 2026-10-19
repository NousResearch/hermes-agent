"""Regression tests for credential-safe Uvicorn diagnostics."""
import logging
import sys
from io import StringIO

import pytest
from uvicorn.logging import AccessFormatter, DefaultFormatter

from hermes_cli.dashboard_auth.log_redaction import (
    GatewayCredentialFilter, redact_query_credentials, redacted_uvicorn_log_config,
)


@pytest.mark.parametrize("query", [
    "token=synthetic-secret", "ticket=synthetic-secret", "internal=synthetic-secret",
    "%74oken=synthetic-secret", "ToKeN=synthetic-secret", "refresh_token=synthetic-secret",
])
def test_encoded_repeated_query_fields_are_redacted(query):
    text = f"ws://example.test/api/ws?profile=demo&{query}&token=second-secret"
    result = redact_query_credentials(text)
    assert "synthetic-secret" not in result
    assert "second-secret" not in result
    assert "profile=demo" in result
    assert result.count("[REDACTED]") == 2


def test_access_formatter_argument_contract_is_preserved():
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                               '%s - "%s %s HTTP/%s" %d',
                               ("127.0.0.1", "GET", "/api/ws?token=synthetic-secret", "1.1", 403), None)
    assert GatewayCredentialFilter().filter(record)
    rendered = AccessFormatter("%(message)s", use_colors=False).format(record)
    assert "synthetic-secret" not in rendered
    assert "403" in rendered and "[REDACTED]" in rendered


def test_protocol_template_and_exception_diagnostics_are_redacted():
    try:
        raise RuntimeError("failure at /api/ws?token=exception-secret")
    except RuntimeError:
        record = logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 1,
                                   "failure at /api/ws?token=%s", ("template-secret",), sys.exc_info())
    record.stack_info = "/api/ws?ticket=stack-secret"
    assert GatewayCredentialFilter().filter(record)
    rendered = DefaultFormatter("%(message)s", use_colors=False).format(record)
    assert all(secret not in rendered for secret in ("template-secret", "exception-secret", "stack-secret"))
    assert "RuntimeError" in rendered
    # The exception appears both as traceback source and as its rendered value.
    assert rendered.count("[REDACTED]") == 4


def test_config_is_independent_and_protects_protocol_and_access_channels():
    from uvicorn.config import LOGGING_CONFIG
    before = repr(LOGGING_CONFIG)
    first, second = redacted_uvicorn_log_config(), redacted_uvicorn_log_config()
    first["filters"].clear()
    assert second["filters"]
    assert repr(LOGGING_CONFIG) == before
    for logger in second["loggers"].values():
        assert "gateway_query_credentials" in logger["filters"]
    for handler in second["handlers"].values():
        assert "gateway_query_credentials" in handler["filters"]


def test_live_uvicorn_handler_redacts_protocol_url_without_replacing_log_config(monkeypatch):
    from hermes_cli.web_server_lifecycle import _uvicorn_console_logging

    output = StringIO()
    parent = logging.getLogger("uvicorn")
    access = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(parent, "handlers", [logging.StreamHandler(output)])
    monkeypatch.setattr(access, "handlers", [])
    monkeypatch.setattr(sys, "stdout", StringIO())
    monkeypatch.setattr(parent, "level", logging.INFO)
    _uvicorn_console_logging()
    logging.getLogger("uvicorn.error").error("rejected /api/ws?ticket=synthetic-secret&profile=demo")
    rendered = output.getvalue()
    assert "synthetic-secret" not in rendered
    assert "ticket=[REDACTED]" in rendered and "profile=demo" in rendered
