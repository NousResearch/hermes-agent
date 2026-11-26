"""Log-path URL credential masking (#133713): a log file is a persistence sink, so the formatter
masks credential-named query values and URL userinfo that live tool flows deliberately keep."""

import logging

import pytest

from agent.redact import RedactingFormatter, redact_sensitive_text


def _format(message: str) -> str:
    record = logging.LogRecord("httpx2", logging.INFO, "", 0, message, (), None)
    return RedactingFormatter("%(message)s").format(record)


def test_penpot_style_user_token_param_is_masked_in_logs():
    secret = "OPAQUE" + "x" * 30
    line = _format(f"HTTP Request: POST http://127.0.0.1:9001/mcp/stream?userToken={secret}&v=1 200")
    assert secret not in line
    assert "?userToken=***&v=1" in line

class TestRedactingFormatterUrls:
    """Moved from test_redact.py (over its size cap)."""

    @staticmethod
    def _record(message: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg=message, args=(), exc_info=None,
        )

    def test_log_path_masks_sensitive_url_query_params(self):
        # issue #133713: a remote MCP URL with token auth must not reach disk verbatim
        formatter = RedactingFormatter("%(message)s")
        result = formatter.format(
            self._record(
                'HTTP Request: POST https://mcp.example.com/mcp?token=SECRETVALUE "HTTP/1.1 200 OK"'
            )
        )
        assert "SECRETVALUE" not in result
        assert "token=***" in result
        assert "https://mcp.example.com/mcp" in result

    def test_log_path_masks_url_userinfo(self):
        formatter = RedactingFormatter("%(message)s")
        result = formatter.format(
            self._record("connect failed: https://user:hunter2@mcp.example.com/mcp")
        )
        assert "hunter2" not in result
        assert "//***:***@mcp.example.com" in result

    def test_log_path_keeps_public_query_params_and_url_free_lines(self):
        formatter = RedactingFormatter("%(message)s")
        url_line = "GET https://example.com/list?page=2&sort=name done"
        assert formatter.format(self._record(url_line)) == url_line
        plain = "state codes like A=1;B=2 survive when no URL is present"
        assert formatter.format(self._record(plain)) == plain


def test_apostrophe_inside_query_credential_is_masked_whole():
    # An apostrophe inside the value used to end the match and leave the tail on disk.
    tail = "opaque" + "Tail0123456789"
    line = _format(f"connect https://mcp.example/mcp?token=abc'{tail}&v=1 failed")
    assert tail not in line
    assert "?token=***&v=1 failed" in line


def test_quoted_url_keeps_its_closing_apostrophe():
    line = _format("url='https://mcp.example/mcp?token=" + "SECRET" + "value123' next")
    assert "SECRETvalue123" not in line
    assert line.endswith("?token=***' next")


@pytest.mark.parametrize("before, after", [
    ("get('https://a.example/v1?key={s}')", "get('https://a.example/v1?key=***')"),
    ("['https://a.example/x?token={s}', 'https://b.example/y']", "['https://a.example/x?token=***', 'https://b.example/y']"),
    ("u = 'https://a.example/x?token={s}'.strip()", "u = 'https://a.example/x?token=***'.strip()"),
])
def test_closing_quote_before_punctuation_survives_the_mask(before, after):
    # The strict pass is shared with every egress caller (context compression, cron errors), so
    # only the credential goes: the quote and the code after it stay.
    secret = "abcdef" + "123456"
    out = redact_sensitive_text(before.format(s=secret), force=True, redact_url_credentials=True)
    assert out == after
