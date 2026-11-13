"""Cookie / Set-Cookie credential masking (#80936)."""

import pytest

from hermes_agent.agent.redact import redact_sensitive_text, redact_terminal_output


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    # Also patch the module-level snapshot so it reflects the cleared env var
    monkeypatch.setattr("hermes_agent.agent.redact._REDACT_ENABLED", True)


class TestCookieCredentials:
    def test_cookie_values_masked_in_assignment_and_inline_header_forms(self):
        for text, secret in (
            ("cookie=opaque-session-value-1234567890; Path=/", "opaque-session-value"),
            ("export SESSION_COOKIE=opaqueSessionValue1234567890", "opaqueSessionValue"),
            ("INFO upstream Cookie: a=1; sid=opaqueSessionValue1234567890; theme=dark user=bob",
             "opaqueSessionValue"),
            ("Set-Cookie: opaqueSessionValue1234567890", "opaqueSessionValue"),
            ('export SESSION_COOKIE="a=1; sid=opaqueSessionValue1234567890"', "opaqueSessionValue"),
            ('INFO upstream Cookie: sid="opaqueSessionValue1234567890"', "opaqueSessionValue"),
            ("INFO upstream x-api-key: $opaqueSessionValue1234567890", "opaqueSessionValue"),
            ("INFO upstream x-api-key: &opaqueSessionValue1234567890", "opaqueSessionValue"),
            ("INFO upstream x-api-key: prefix|opaqueSessionValue1234567890", "opaqueSessionValue"),
            ("INFO upstream x-api-key : opaqueSessionValue1234567890", "opaqueSessionValue"),
            ("INFO upstream Cookie: sid=pre'opaqueSessionValue1234567890", "opaqueSessionValue"),
            ("export SESSION_COOKIE='sid='\"opaqueSessionValue1234567890\"", "opaqueSessionValue"),
        ):
            assert secret not in redact_sensitive_text(text), text
        # Terminal output scrubs with code_file=True (``curl -i``); the header is still a credential.
        rc_line = "export SESSION_COOKIE='sid='\\''opaqueSessionValue1234567890'"
        assert "opaqueSessionValue" not in redact_sensitive_text(rc_line, file_read=True, secret_file=True)
        dump = "HTTP/2 200\nset-cookie: sid=opaqueSessionValue1234567890; Path=/\ncontent-type: text/html"
        assert "opaqueSessionValue" not in redact_terminal_output(dump, "curl -i https://example.com")

    def test_cookie_prose_unchanged(self):
        # #80936 false positives: prose, cookie-ish identifiers and URL params are not credentials.
        for text in (
            "cookies enabled now",
            "cookie_policy: strict",
            "mycookiejar=1",
            'Cookie.set("name", value)',
            "The cookie: it was delicious",
            "the Set-Cookie: header sets state",
            "if cookie == expected:",
            "GET https://example.com/cb?cookie=abc123&x=1",
            # Code and placeholders, the same shapes main leaves alone for ``token``.
            "cookie = getCookie()",
            "self.cookie = cookie",
            "cookie=None",
            "cookie=$(cat jar)",
            '{"cookie": "chocolate"}',
            '{"apikey": "${API_KEY}"}',
            '{"x-api-key": "${API_KEY}"}',
        ):
            assert redact_sensitive_text(text) == text, text
            assert redact_sensitive_text(text, force=True) == text, text
        # A masked value never swallows the text around it (JSON neighbours, shell separators).
        out = redact_sensitive_text('{"output": "cookie=\'opaqueSessionValue1234567890\'", "model": "m"}')
        assert out.endswith('\'", "model": "m"}') and "opaqueSessionValue" not in out
        for text, tail in (
            ("SESSION_COOKIE='opaqueSessionValue1234567890';echo done", "';echo done"),
            ('{"output":"cookie=opaqueSessionValue1234567890","model":"m"}', '","model":"m"}'),
            ("SESSION_COOKIE=opaqueSessionValue1234567890;echo done", "7890;echo done"),
            ("INFO x-api-key: opaqueSessionValue1234567890&& echo ok", "***&& echo ok"),
        ):
            out = redact_sensitive_text(text)
            assert out.endswith(tail) and "opaqueSessionValue" not in out, out
