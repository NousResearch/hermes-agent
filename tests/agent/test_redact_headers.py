"""Header redaction must preserve the syntax consumed by diagnostic tools."""

import json
import shlex

import pytest

from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Redaction on regardless of HERMES_REDACT_SECRETS left over from other test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def test_json_headers_remain_parseable_without_exposing_credentials():
    for name in ("x-api-key", "X-Goog-Api-Key", "apikey", "Cookie", "Set-Cookie"):
        for value in ('abcd="opaque-session-secret-1234567890"; Path=/',
                      's=short"value', "opaque\\session\\credential1234567890"):
            payload = {"headers": {name: value}, "model": "example", "messages": []}
            serialized = json.dumps(payload)
            redacted = redact_sensitive_text(serialized, force=True)
            restored = json.loads(redacted)
            assert restored["headers"][name] != value
            assert "opaque-session-secret" not in restored["headers"][name]
            assert restored["model"] == payload["model"]
            assert restored["messages"] == payload["messages"]
            file_redacted = json.loads(redact_sensitive_text(serialized, force=True, file_read=True, secret_file=True))
            assert file_redacted["headers"][name].startswith("«redacted")
            assert redact_sensitive_text(serialized, force=True, code_file=True) == serialized
        if "cookie" not in name.lower():  # a slash-led opaque key is a credential, not a path
            slash_key = json.dumps({name: "/abcdef0123456789abcdef0123456789"})
            assert "0123456789abcdef0123" not in redact_sensitive_text(slash_key, force=True)
        vendor_key = json.dumps({name: "ghp_" + "A1b2C3d4" * 5})
        assert "«redacted:ghp_" in redact_sensitive_text(vendor_key, force=True, file_read=True, secret_file=True)


def test_raw_headers_mask_all_values_without_consuming_shell_or_next_line():
    value = "sid=opaque-session-secret-1234567890; second=another-credential"
    for name in ("Cookie", "sEt-CoOkIe", "x-api-key"):
        raw = f"{name}: {value}\nX-Trace: unchanged"
        assert redact_sensitive_text(raw, force=True) == f"{name}: ***\nX-Trace: unchanged"
        for flag in ("-H", "--header"):
            for quote in ("'", '"'):
                command = f"curl {flag} {quote}{name}: {value}{quote} https://example.com/path"
                output = redact_sensitive_text(command, force=True)
                assert shlex.split(output) == ["curl", flag, f"{name}: ***", "https://example.com/path"]
    for benign in ("the cookie: is a browser feature", 'Cookie.set("name", value)'):
        assert redact_sensitive_text(benign, force=True) == benign


class TestXKeyHeaderNames:
    """``x-<name>-key`` custom API-key headers (``x-acme-key`` authenticates an MCP server, and its
    line in config.yaml was masked on no surface: the header list only knew fixed names)."""

    V = "fakeAcmeKeyValue0123456789abcdef"

    @pytest.mark.parametrize("template", [
        "x-acme-key: {v}",
        "      x-acme-key: {v}",            # config.yaml mcp_servers.<x>.headers
        "X-Acme-Key: {v}",
        'x-acme-key: "{v}"',
        "x-functions-key: {v}",
        "curl -H 'x-acme-key: {v}' http://127.0.0.1:8000/mcp",
        'curl -H "x-acme-key: {v}" http://127.0.0.1:8000/mcp',
        '{{"x-acme-key": "{v}"}}',
    ])
    def test_x_key_header_masked(self, template):
        text = template.format(v=self.V)
        for result in (redact_sensitive_text(text, force=True),
                       redact_sensitive_text(text, force=True, file_read=True, secret_file=True)):
            assert self.V not in result
            assert "acme-key" in result.lower() or "functions-key" in result

    def test_x_acme_key_masked_in_code_file_terminal_output(self):
        from agent.redact import redact_terminal_output

        out = f"> x-acme-key: {self.V}\n< HTTP/1.1 200 OK"
        assert self.V not in redact_terminal_output(out, "curl -v http://127.0.0.1:8000/mcp", force=True)

    @pytest.mark.parametrize("name", ["x-acme-key", "x-api-key"])
    def test_quoted_header_behind_the_read_file_gutter_is_masked(self, name):
        out = redact_sensitive_text(f'5|      {name}: "{self.V}"', force=True, file_read=True)
        assert out == f'5|      {name}: ***'

    def test_short_header_value_keeps_closing_quote(self):
        text = 'curl -H "x-acme-key: short123" http://127.0.0.1:8000/mcp'
        assert redact_sensitive_text(text, force=True) == 'curl -H "x-acme-key: ***" http://127.0.0.1:8000/mcp'

    @pytest.mark.parametrize("text", [
        "x-monkey: banana",
        "x-keyboard: us",
        "x-request-id: 1234567890abcdef",
        "inbox-key: meeting notes",
        "primary-key: id",
        "sort-key: name",
        "the api-key-rotation-key: guide",
        "hot-key: ctrl-k",
    ])
    def test_other_dash_key_words_unchanged(self, text):
        assert redact_sensitive_text(text, force=True) == text
