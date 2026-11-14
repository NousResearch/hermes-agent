"""Log-path URL credential masking (#133713): a log file is a persistence sink, so the formatter
masks credential-named query values and URL userinfo that live tool flows deliberately keep."""

import logging

from agent.redact import RedactingFormatter


def _format(message: str) -> str:
    record = logging.LogRecord("httpx2", logging.INFO, "", 0, message, (), None)
    return RedactingFormatter("%(message)s").format(record)


def test_penpot_style_user_token_param_is_masked_in_logs():
    secret = "OPAQUE" + "x" * 30
    line = _format(f"HTTP Request: POST http://127.0.0.1:9001/mcp/stream?userToken={secret}&v=1 200")
    assert secret not in line
    assert "?userToken=***&v=1" in line
