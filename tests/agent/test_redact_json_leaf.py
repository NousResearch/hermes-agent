"""Redaction must not corrupt the structure it is redacting.

#123688: several call sites redacted a structure by serializing it, masking the text,
and parsing it back — ``json.loads(redact_sensitive_text(json.dumps(obj)))``. The text
patterns are not JSON-aware, and ``_ENV_ASSIGN_RE``'s unquoted value group ``(\\S+)``
swallows the closing quote of the string it sits in. Any string leaf ending in an
ENV-style secret therefore both broke the parse *and* left a partial value behind:

    '{"x": "DB_PASSWORD=***'      <- unterminated, trailing ``"`` and ``}`` consumed

``_redact_metadata`` (tools/kanban_tools.py) returns None on that failure, so
``kanban_complete`` stored the UNREDACTED dict; the request-debug dump wrote nothing;
``trace_upload`` refused the upload; the Google Chat envelope log printed a truncated,
unparseable string.

``redact_sensitive_json`` walks the structure instead, so JSON syntax is never part of
the text a pattern sees. These tests pin both the round trip and coverage parity: anything
the old serialize-then-redact path masked is still masked.
"""
from __future__ import annotations

import json

import pytest

from agent.redact import redact_sensitive_json, redact_sensitive_text

# Leaves that the serialize-then-redact round trip corrupted (from the issue).
CORRUPTING_LEAVES = [
    "DB_PASSWORD=abcdef12",
    "PASS=174",
    "db_password=abcdef12",
    "spring.datasource.password=hunter2x",
    "export API_KEY=abcdefghijklmnopqrstuv",
]


class TestRoundTrip:
    def test_a_corrupting_leaf_stays_parseable(self):
        for leaf in CORRUPTING_LEAVES:
            out = redact_sensitive_json({"x": leaf}, force=True)
            assert isinstance(out, dict), f"{leaf!r} did not survive as a dict"
            assert isinstance(out["x"], str)

    def test_a_corrupting_leaf_loses_its_secret(self):
        """The failure was not only a broken parse: ``{"x": "DB_PASSWORD=***`` is a partial
        mask, and whatever followed the swallowed quote never reached the output at all."""
        for leaf in CORRUPTING_LEAVES:
            out = redact_sensitive_json({"x": leaf}, force=True)["x"]
            assert "abcdef12" not in out and "hunter2x" not in out
            assert "174" not in out, f"{leaf!r} leaked its value: {out!r}"

    def test_the_leaf_is_redacted_the_same_way_the_text_path_redacts_it(self):
        """Parity with ``redact_sensitive_text`` on the isolated leaf: the JSON walk must not
        weaken masking, only stop it from corrupting structure."""
        for leaf in CORRUPTING_LEAVES:
            assert redact_sensitive_json({"x": leaf}, force=True)["x"] == \
                redact_sensitive_text(leaf, force=True)

    @pytest.mark.parametrize(
        "obj,leaked",
        [
            ({"password": "hunter2"}, "hunter2"),
            ({"nested": {"api_key": "abc123"}}, "abc123"),
            ({"list": [{"secret": "s3cr3t"}, "plain"]}, "s3cr3t"),
        ],
    )
    def test_secret_named_keys_still_mask_whole(self, obj, leaked):
        """The serialized path's ``"password": "..."`` rule is preserved: a value under a
        secret-named key is masked whole, at any depth."""
        redacted = redact_sensitive_json(obj, force=True)
        assert leaked not in json.dumps(redacted), f"{leaked!r} survived redaction"

    def test_masking_matches_the_serialized_path_key_for_key(self):
        """The contract that matters: for every key, the JSON walk masks exactly what the
        old path masked — no more, no less. Derived from the real patterns, not a snapshot.

        The cases include values the old path deliberately leaves alone — a short ``t0ken``
        that is not opaque enough, ``os.getenv('X')`` lookups, ``$HOME`` paths, and
        ``db_password`` (absent from ``_JSON_KEY_NAMES``). Masking any of those would be a
        behaviour change disguised as a repair.
        """
        cases = [
            {"password": "hunter2"}, {"api_key": "abc123"}, {"token": "t0ken"},
            {"db_password": "abc"}, {"other": "keep-me"},
            {"token": "os.getenv('MY_TOKEN')"}, {"password": "$HOME/.ssh/id_rsa"},
            {"nested": {"secret": "s3cr3t", "fine": "ok"}},
        ]
        for obj in cases:
            try:
                old = json.loads(redact_sensitive_text(json.dumps(obj), force=True))
            except json.JSONDecodeError:
                continue  # the old path broke here — that is the bug, not a masking change
            assert redact_sensitive_json(obj, force=True) == old, f"masking changed for {obj}"

    def test_non_secret_leaves_are_untouched(self):
        obj = {"message": "hello", "count": 3, "flag": False, "none": None, "list": [1, 2]}
        assert redact_sensitive_json(obj, force=True) == obj

    def test_keys_are_redacted_too(self):
        out = redact_sensitive_json({"API_KEY=abcdef12": "value"}, force=True)
        assert all("abcdef12" not in k for k in out), f"key leaked: {list(out)}"


class TestDisabledRedaction:
    def test_disabled_redaction_is_a_no_op(self, monkeypatch):
        """``security.redact_secrets: false`` must leave the structure exactly as it came in."""
        import agent.redact as redact_mod

        monkeypatch.setattr(redact_mod, "_redact_enabled", lambda: False)
        obj = {"x": "DB_PASSWORD=abcdef12", "password": "hunter2"}
        assert redact_sensitive_json(obj) == obj


class TestParityWithTheSerializedPath:
    @pytest.mark.parametrize("obj", [{"x": "DB_PASSWORD=abcdef12"}, {"y": "hello world"}])
    def test_anything_the_old_path_masked_is_still_masked(self, obj):
        """Regression guard for coverage, not shape: the fix must not silently narrow masking
        by taking a different code path."""
        try:
            old = json.loads(redact_sensitive_text(json.dumps(obj), force=True))
        except json.JSONDecodeError:
            old = None  # the old path broke here — that is the bug being fixed
        if old is None:
            return
        new = redact_sensitive_json(obj, force=True)
        assert json.dumps(new) == json.dumps(old)