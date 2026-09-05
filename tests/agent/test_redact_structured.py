"""Invariant tests for fail-closed structure-aware redaction."""
from __future__ import annotations

import json

import pytest

from agent.redact import redact_sensitive_text
from agent.redact_structured import redact_structured


def _settled(payload) -> bool:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return redact_sensitive_text(serialized, force=True) == serialized


CORRUPTING = [
    {"cmd": "psql; export DB_PASSWORD=xyz"},
    {"hdr": "x-api-key: abc123"},
    {"backslash": "path\\to\\OPENAI_API_KEY=sk-abcdefghijklmnopqrst"},
]

SURVIVING = [
    {"password": "hunter2horse"},
    {"api_key": "abc123def456ghi"},
    {"token": "sk-proj-abc123def456ghi789jkl012"},
    {"note": "ghp_" + "A" * 40},
    {"db": "postgres://user:supersecret@localhost:5432/db"},
]

SENTINEL = "[redaction-unverified]"


@pytest.mark.parametrize("payload", CORRUPTING + SURVIVING)
def test_result_is_a_fixed_point_of_the_authoritative_redactor(payload):
    assert _settled(redact_structured(payload))


@pytest.mark.parametrize("payload", CORRUPTING)
def test_secret_is_removed_for_delimiter_sensitive_payloads(payload):
    out = redact_structured(payload)

    for key, raw in payload.items():
        assert out[key] != raw, f"{key} survived unredacted"
    assert _settled(out)


def test_nested_dict_list_and_tuple_are_walked_without_losing_their_shapes():
    payload = {
        "deep": {"inner": {"password": "hunter2horse"}},
        "rows": [{"api_key": "abc123def456ghi"}, ["x-api-key: abc123"]],
        "pair": ("OPENAI_API_KEY=sk-abcdefghijklmnopqrst", "plain"),
    }

    out = redact_structured(payload)

    assert out["deep"]["inner"]["password"] != "hunter2horse"
    assert out["rows"][0]["api_key"] != "abc123def456ghi"
    assert out["rows"][1][0] != "x-api-key: abc123"
    assert out["pair"][0] != "OPENAI_API_KEY=sk-abcdefghijklmnopqrst"
    assert isinstance(out["rows"], list)
    assert isinstance(out["pair"], tuple)
    assert _settled(out)


def test_benign_values_pass_through_verbatim():
    payload = {
        "summary": "fixed the bug in run_agent.py",
        "changed_files": ["run_agent.py", "cli.py"],
        "nested": {"tuple": ("a", "b")},
        "number": 12,
        "ratio": 0.75,
        "ok": True,
        "nothing": None,
        "empty": "",
    }

    assert redact_structured(payload) == payload


def test_nested_secret_fields_keep_parent_credential_context():
    payload = {"password": ["hunter2horse", {"value": "another-password"}],
               "safe": {"value": "keep"}}
    out = redact_structured(payload)
    assert "hunter2horse" not in json.dumps(out)
    assert "another-password" not in json.dumps(out)
    assert out["safe"] == payload["safe"]


def test_scalar_types_are_preserved():
    payload = {"n": 5, "f": 1.5, "b": False, "nil": None}

    out = redact_structured(payload)

    assert out == payload
    assert isinstance(out["n"], int) and not isinstance(out["n"], bool)
    assert isinstance(out["f"], float)
    assert isinstance(out["b"], bool)
    assert out["nil"] is None


@pytest.mark.parametrize(
    "key",
    ("password", "api_key", "bearer", "key_material", "access_token"),
)
def test_dict_key_context_is_preserved(key):
    assert redact_structured({key: "abc123def456ghi"})[key] != "abc123def456ghi"


def test_nonserializable_object_with_secret_carrying_str_fails_closed():
    class Carrier:
        def __str__(self) -> str:
            return "OPENAI_API_KEY=sk-abcdefghijklmnopqrst"

    out = redact_structured({"obj": Carrier()})

    assert "sk-abcdefghijklmnopqrst" not in json.dumps(out, default=str)
    assert _settled(out)


def test_nonserializable_benign_leaf_does_not_raise():
    out = redact_structured({"s": {"a"}})

    assert _settled(out)


def test_redactor_exception_propagates_without_returning_raw_data(monkeypatch):
    def boom(_text, *, force=False, **kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", boom)

    with pytest.raises(RuntimeError, match="redactor unavailable"):
        redact_structured({"tok": "sk-abcdefghijklmnopqrst"})


def test_truncating_redactor_cannot_return_raw_data(monkeypatch):
    def truncating(text, *, force=False, **kwargs):
        return text[: max(0, len(text) - 6)]

    monkeypatch.setattr("agent.redact.redact_sensitive_text", truncating)

    out = redact_structured({"tok": "sk-abcdefghijklmnopqrst"})

    assert "sk-abcdefghijklmnopqrst" not in json.dumps(out)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"tok": "sk-abcdefghijklmnopqrst"}, {"_redaction": SENTINEL}),
        (["sk-abcdefghijklmnopqrst"], [SENTINEL]),
        (("sk-abcdefghijklmnopqrst",), [SENTINEL]),
        ("sk-abcdefghijklmnopqrst", SENTINEL),
    ],
)
def test_never_settling_redactor_returns_content_free_sentinel(
    monkeypatch, payload, expected
):
    def always_changes(text, *, force=False, **kwargs):
        return text + "!"

    monkeypatch.setattr("agent.redact.redact_sensitive_text", always_changes)

    assert redact_structured(payload) == expected
