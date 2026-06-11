"""Failing tests for the JSON-preserving tool-call argument redaction.

Drives the fix for the PR #42846 review finding:
  The new tool-call argument redaction can corrupt the JSON string stored in
  tool_calls[].function.arguments.

The redactor must NEVER produce invalid JSON from valid input.
"""
import json

import pytest

from agent.redact import (
    redact_sensitive_text,
    _redact_message_object,
    _redact_json_arguments,
)


# A handful of realistic secret shapes that all match patterns in
# redact_sensitive_text and which the env-assignment / prefix regexes
# will attempt to mask.
_OPENAI_KEY = "sk" + "-" + "abcdefghijklmnopqrstuvwxyz0123456789ABCD"  # 41 chars total
_GH_KEY = "ghp_" + "a" * 40
_BEARER = "Bearer " + "Z" * 60


def _env_args(value: str) -> str:
    """Build a tool-call arguments JSON string with a given value."""
    # JSON-escape any quotes/backslashes in value to keep the test
    # focused on what happens INSIDE the string.
    safe = value.replace("\\", "\\\\").replace('"', '\\"')
    return '{"command":"echo ' + safe + '"}'


class TestRedactJsonArguments:
    """Direct tests for the new _redact_json_arguments helper."""

    def test_returns_str(self):
        out = _redact_json_arguments('{"x": 1}')
        assert isinstance(out, str)

    def test_preserves_json_with_no_secrets(self):
        arg = '{"command": "ls -la"}'
        out = _redact_json_arguments(arg)
        assert json.loads(out) == {"command": "ls -la"}

    def test_redacts_env_assignment_in_value(self):
        arg = _env_args("OPENAI_API_KEY=" + _OPENAI_KEY)
        out = _redact_json_arguments(arg)
        # Must still be valid JSON
        parsed = json.loads(out)
        assert "command" in parsed
        # The secret value must be masked
        assert _OPENAI_KEY not in parsed["command"]
        # The KEY= part survives (env-name is structural, not the secret)
        assert "OPENAI_API_KEY" in parsed["command"]
        # The secret VALUE is replaced with the masked form
        assert "***" in parsed["command"]

    def test_redacts_openai_key_in_value(self):
        arg = _env_args(_OPENAI_KEY)
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert _OPENAI_KEY not in parsed["command"]

    def test_redacts_bearer_token(self):
        # The Authorization: Bearer <sk-XXX> form is caught by both
        # _AUTH_HEADER_RE and _PREFIX_RE. Use a key value built from
        # non-prefix characters so the redaction only matches the
        # header pattern, not the prefix regex.
        skey = "sk" + "-" + "abcdefghijklmnopqrstuvwxyz0123456789ABCD"
        arg = _env_args("Authorization: Bearer " + skey)
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        # The OpenAI key value must be masked
        assert skey not in parsed["command"]
        # And the literal "sk-..." prefix must not survive intact
        assert "abcdefghijklmnopqrstuvwxyz" not in parsed["command"]

    def test_redacts_bearer_header_bare(self):
        # Bare "Bearer XYZ" with no Authorization: prefix and no
        # vendor prefix — _AUTH_HEADER_RE requires the header. The
        # structural layer will not redact the value (no matching
        # pattern), which is the EXPECTED behaviour: without a
        # recognisable secret shape we don't fabricate a mask. The
        # output MUST still be valid JSON.
        arg = _env_args("Bearer ZZZZZZZZZZZZZZZZZZZZ")
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)  # must not raise
        # No mask applied — that's fine, no false positives.
        assert "Bearer ZZZZZZZZZZZZZZZZZZZZ" in parsed["command"]

    def test_redacts_github_pat_in_value(self):
        arg = _env_args(_GH_KEY)
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert _GH_KEY not in parsed["command"]

    def test_preserves_nested_object_structure(self):
        arg = json.dumps({
            "command": "deploy",
            "env": {
                "OPENAI_API_KEY": _OPENAI_KEY,
                "PATH": "/usr/bin",
            },
        })
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert parsed["command"] == "deploy"
        assert parsed["env"]["PATH"] == "/usr/bin"
        assert _OPENAI_KEY not in parsed["env"]["OPENAI_API_KEY"]
        # The KEY name survives
        assert "OPENAI_API_KEY" in parsed["env"]

    def test_preserves_list_structure(self):
        arg = json.dumps({
            "command": "ls",
            "args": [_OPENAI_KEY, "plain"],
        })
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert parsed["args"][1] == "plain"
        assert _OPENAI_KEY not in parsed["args"][0]

    def test_redacts_jwt_token(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9" + "." + "abcdefghij" + "." + "xyz1234567"
        arg = json.dumps({"header": jwt})
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        # JWT must be masked
        assert "eyJhbGciOiJIUzI1NiJ9" not in parsed["header"]

    def test_handles_json_array_at_top_level(self):
        arg = json.dumps([_OPENAI_KEY, "safe"])
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert parsed[1] == "safe"
        assert _OPENAI_KEY not in parsed[0]

    def test_handles_json_null(self):
        out = _redact_json_arguments("null")
        assert json.loads(out) is None

    def test_handles_json_number(self):
        out = _redact_json_arguments("42")
        assert json.loads(out) == 42

    def test_handles_json_string_at_top_level(self):
        out = _redact_json_arguments(json.dumps(_OPENAI_KEY))
        parsed = json.loads(out)
        assert _OPENAI_KEY not in parsed

    def test_redacts_deeply_nested(self):
        arg = json.dumps({
            "a": {"b": {"c": {"d": {"e": _OPENAI_KEY}}}}
        })
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert _OPENAI_KEY not in parsed["a"]["b"]["c"]["d"]["e"]

    def test_redacts_multiple_secrets(self):
        arg = json.dumps({
            "key1": _OPENAI_KEY,
            "key2": _GH_KEY,
            "key3": "plain",
        })
        out = _redact_json_arguments(arg)
        parsed = json.loads(out)
        assert _OPENAI_KEY not in parsed["key1"]
        assert _GH_KEY not in parsed["key2"]
        assert parsed["key3"] == "plain"


class TestRedactMessageObjectToolCalls:
    """_redact_message_object must produce valid JSON in tool_calls[].function.arguments."""

    def test_preserves_tool_call_json(self):
        arg = _env_args("OPENAI_API_KEY=" + _OPENAI_KEY)
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": arg,
                },
            }],
        }
        _redact_message_object(msg, redact_sensitive_text)
        out = msg["tool_calls"][0]["function"]["arguments"]
        # Must parse as JSON
        parsed = json.loads(out)
        assert "command" in parsed
        # The secret value must be masked
        assert _OPENAI_KEY not in parsed["command"]
        assert "***" in parsed["command"]

    def test_preserves_legacy_function_call_json(self):
        arg = _env_args("OPENAI_API_KEY=" + _OPENAI_KEY)
        msg = {
            "role": "assistant",
            "function_call": {
                "name": "shell",
                "arguments": arg,
            },
        }
        _redact_message_object(msg, redact_sensitive_text)
        out = msg["function_call"]["arguments"]
        parsed = json.loads(out)
        assert _OPENAI_KEY not in parsed["command"]
        assert "***" in parsed["command"]

    def test_preserves_message_with_no_tool_calls(self):
        msg = {"role": "user", "content": "hello"}
        _redact_message_object(msg, redact_sensitive_text)
        assert msg["content"] == "hello"

    def test_preserves_list_typed_content(self):
        # This was the second blocker — list-typed content must be redacted.
        # Use a 41-char OpenAI-key-shape value so _PREFIX_RE's floor of 10
        # chars catches it.
        secret = "sk" + "-" + "abcdefghijklmnopqrstuvwxyz0123456789ABCD"
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "key is " + secret},
                {"type": "text", "text": "plain"},
            ],
        }
        _redact_message_object(msg, redact_sensitive_text)
        parts = msg["content"]
        assert isinstance(parts, list)
        # The literal secret value must be gone (specifically, the
        # bulk of the secret — the head=4/tail=4 debug tail ABCD is
        # preserved by mask_secret)
        assert "abcdefghijklmnopqrstuvwxyz" not in parts[0]["text"]
        # The mask ellipsis `...` must appear (evidence of redaction)
        assert "..." in parts[0]["text"]
        # And the tail of the secret (preserved for debug) survives
        assert "ABCD" in parts[0]["text"]
        assert parts[1]["text"] == "plain"

    def test_realistic_provider_payload_round_trip(self):
        """A representative OpenAI chat completion payload with mixed
        string and list content, plus a tool_call with embedded secret,
        must round-trip through json.dumps after redaction."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll run the command with key " + _OPENAI_KEY},
            ],
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": _env_args("OPENAI_API_KEY=" + _OPENAI_KEY),
                },
            }],
        }
        _redact_message_object(msg, redact_sensitive_text)
        # The whole message must serialize back to valid JSON
        roundtrip = json.dumps(msg, ensure_ascii=False)
        reparsed = json.loads(roundtrip)
        # And the secrets must not be present in any string field
        def _scrub(o):
            if isinstance(o, str):
                assert _OPENAI_KEY not in o, f"leaked: {o[:100]}"
            elif isinstance(o, list):
                for x in o:
                    _scrub(x)
            elif isinstance(o, dict):
                for v in o.values():
                    _scrub(v)
        _scrub(reparsed)


class TestJsonArgumentsDegradedPath:
    """When input is NOT valid JSON, _redact_json_arguments should still
    attempt safe redaction. It must never return a value that's MORE
    broken than the input."""

    def test_malformed_json_does_not_throw(self):
        # Trailing garbage — not valid JSON.
        out = _redact_json_arguments("not json at all")
        assert isinstance(out, str)

    def test_malformed_json_with_secret_does_not_crash(self):
        out = _redact_json_arguments('"OPENAI_API_KEY=' + _OPENAI_KEY + '"')
        # Should redact without crashing. We don't assert the output is
        # valid JSON (it isn't, by construction) — only that it returned.
        assert isinstance(out, str)
