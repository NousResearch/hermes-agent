"""Header redaction must preserve the syntax consumed by diagnostic tools."""

import json
import shlex

from agent.redact import redact_sensitive_text


def test_json_headers_remain_parseable_without_exposing_credentials():
    for name in ("x-api-key", "X-Goog-Api-Key", "apikey", "Cookie", "Set-Cookie"):
        for value in ('abcd="opaque-session-secret-1234567890"; Path=/',
                      'short"value', "opaque\\session\\credential1234567890"):
            payload = {"headers": {name: value}, "model": "example", "messages": []}
            serialized = json.dumps(payload)
            redacted = redact_sensitive_text(serialized, force=True)
            restored = json.loads(redacted)
            assert restored["headers"][name] != value
            assert "opaque-session-secret" not in restored["headers"][name]
            assert restored["model"] == payload["model"]
            assert restored["messages"] == payload["messages"]
            assert redact_sensitive_text(serialized, force=True, code_file=True) == serialized


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
