"""Tests for ``_redact_tool_progress_args`` in gateway.run.

The gateway's tool-progress renderer echoes tool-call arguments into chat
(Discord/Telegram/etc.), a verbatim JSON dump in verbose mode, a human
preview otherwise, or a terminal command block. Because that bubble is an
egress boundary, credential-shaped string values must be redacted before
they can reach any render path, even though the outbound *response* text is
already redacted.

These tests pin the helper's contract: credential-shaped strings are redacted
with the existing ``redact_sensitive_text(force=True)`` primitive **at any
depth** (nested dicts/lists included), across several common credential
shapes, non-string values and dict keys pass through untouched, and a
redactor failure never breaks the bubble (fail-soft).

A fixture disables the global redaction flag (``security.redact_secrets:
false``) to prove ``force=True`` still redacts at this chat-egress boundary,
the same intent as ``_redact_gateway_user_facing_secrets``.
"""

from __future__ import annotations

import json

import pytest

from gateway.run import _redact_tool_progress_args

# Synthetic credentials in the shapes shipped by real providers (never real
# secrets). The full literal must never survive redaction; a head/tail mask
# may remain for debuggability.
_FAKE_OPENAI_KEY = "sk-proj-" + "a" * 40
_FAKE_ANTHROPIC_KEY = "sk-ant-" + "b" * 36
_FAKE_GITHUB_KEY = "ghp_" + "c" * 36
_FAKE_SK_KEY = "sk-" + "d" * 24

_FAKE_ALL = [
    _FAKE_OPENAI_KEY,
    _FAKE_ANTHROPIC_KEY,
    _FAKE_GITHUB_KEY,
    _FAKE_SK_KEY,
]


@pytest.fixture(autouse=True)
def _redaction_globally_disabled(monkeypatch):
    """Simulate security.redact_secrets: false so force=True must still win."""
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)


def _assert_clean(payload: object) -> None:
    dumped = json.dumps(payload)
    for secret in _FAKE_ALL:
        assert secret not in dumped


class TestRedactToolProgressArgs:
    def test_redacts_query_string_value(self):
        args = {"query": _FAKE_OPENAI_KEY}
        out = _redact_tool_progress_args(args)
        assert _FAKE_OPENAI_KEY not in out["query"]
        assert len(out["query"]) < len(_FAKE_OPENAI_KEY)

    def test_redacts_all_credential_shapes_top_level(self):
        for secret in _FAKE_ALL:
            out = _redact_tool_progress_args({"value": secret})
            assert secret not in json.dumps(out)

    def test_redacts_embedded_in_terminal_command(self):
        args = {"command": f"cat config.yaml && echo {_FAKE_GITHUB_KEY}"}
        out = _redact_tool_progress_args(args)
        _assert_clean(out)

    def test_redacts_nested_dict_value(self):
        args = {"options": {"api_key": _FAKE_OPENAI_KEY}}
        out = _redact_tool_progress_args(args)
        assert _FAKE_OPENAI_KEY not in out["options"]["api_key"]
        assert len(out["options"]["api_key"]) < len(_FAKE_OPENAI_KEY)

    def test_redacts_nested_list_value(self):
        args = {"headers": [_FAKE_ANTHROPIC_KEY, "normal"]}
        out = _redact_tool_progress_args(args)
        assert _FAKE_ANTHROPIC_KEY not in json.dumps(out)
        assert out["headers"][1] == "normal"

    def test_redacts_double_nested_value(self):
        args = {"a": {"b": {"c": _FAKE_SK_KEY}}}
        out = _redact_tool_progress_args(args)
        _assert_clean(out)

    def test_normal_string_values_unchanged(self):
        args = {"query": "what is the weather?", "units": "metric"}
        out = _redact_tool_progress_args(args)
        assert out["query"] == "what is the weather?"
        assert out["units"] == "metric"

    def test_non_string_values_pass_through(self):
        args = {"temperature": 0.7, "top_k": 5, "flag": None}
        out = _redact_tool_progress_args(args)
        assert out["temperature"] == 0.7
        assert out["top_k"] == 5
        assert out["flag"] is None

    def test_empty_or_non_dict_returns_unchanged(self):
        assert _redact_tool_progress_args({}) == {}
        assert _redact_tool_progress_args(None) is None
        assert _redact_tool_progress_args("not-a-dict") == "not-a-dict"
        assert _redact_tool_progress_args(["list"]) == ["list"]

    def test_does_not_mutate_input_dict(self):
        args = {"query": _FAKE_OPENAI_KEY}
        snapshot = json.loads(json.dumps(args))
        _redact_tool_progress_args(args)
        assert args == snapshot

    def test_verbose_json_dump_is_clean(self):
        args = {"query": _FAKE_OPENAI_KEY, "nested": {"k": _FAKE_GITHUB_KEY}}
        out = _redact_tool_progress_args(args)
        _assert_clean(out)

    def test_bubble_survives_redactor_error(self, monkeypatch):
        """If the redactor module itself fails (import error), the bubble
        must still render with the original args. redact_sensitive_text is
        called bare (repo convention, it is trusted), so this pins the
        outer crash net, not a per-string guard."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "agent.redact":
                raise ImportError("redactor module unavailable")

            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        args = {"query": _FAKE_OPENAI_KEY, "nested": {"k": _FAKE_GITHUB_KEY}}
        out = _redact_tool_progress_args(args)
        assert out == args

    def test_deeply_nested_payload_partially_redacted(self):
        """A payload nested past Python's recursion limit must not fail open:
        a secret at a reachable depth is still redacted, and only the
        too-deep branch passes through unchanged."""
        root = {"query": _FAKE_OPENAI_KEY, "nested": {}}
        cur = root["nested"]
        for _ in range(1500):
            nxt = {"payload": "plain"}
            cur["next"] = nxt
            cur = nxt

        out = _redact_tool_progress_args(root)
        assert _FAKE_OPENAI_KEY not in out["query"]
        assert len(out["query"]) < len(_FAKE_OPENAI_KEY)
        assert isinstance(out["nested"]["next"], dict)

    def test_secret_in_dict_key_is_not_redacted(self):
        """Dict keys are never touched: the renderer depends on exact key
        lookups (args.get('command'), list(args.keys()), preview selection
        by key name), so key-walking redaction would silently break the
        display pipeline."""
        key = _FAKE_OPENAI_KEY
        args = {key: "value"}
        out = _redact_tool_progress_args(args)
        assert list(out.keys()) == list(args.keys())
        assert out[key] == "value"
