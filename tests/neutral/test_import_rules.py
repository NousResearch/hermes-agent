"""The import payload rules (``session_rules.imports.validate_import_payload``).

The cases pin the item and error shapes, the error order and texts, the size accounting that leaves out ``timings`` and counts a rejected entry before its limit check, and the 64-bit integer bounds.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persistence.housekeeping.maintenance import IMPORT_MAX_MESSAGES_PER_SESSION, IMPORT_MAX_TOTAL_MESSAGES
from session_rules.imports import IMPORT_INT_MAX, IMPORT_INT_MIN, validate_import_payload

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
MiB = 1024 * 1024
INT_RANGE = "is outside SQLite's integer range"


def _msg(role, content, offset, **fields):
    return {"role": role, "content": content, "timestamp": BASE + offset, **fields}


def _s(sid, messages=(), **fields):
    return {"id": sid, "source": "cli", "started_at": BASE, "messages": list(messages), **fields}


def _errors(sessions):
    return validate_import_payload(sessions)[1]


def _failed(sessions):
    """``{session_id: error}`` of the rejected entries."""
    return {error["session_id"]: error["error"] for error in _errors(sessions)}


def test_a_valid_session_normalizes_and_timings_do_not_count_toward_the_size():
    messages = [_msg("user", "hi", 0, token_count="3", reasoning_details='[{"k": 1}]'), _msg("assistant", [{"type": "text", "text": "x"}], 1)]
    raw = _s(" a ", messages, timings={"pad": "x" * (6 * MiB)}, model_config={"_branched_from": "p"})
    normalized, errors = validate_import_payload([raw])
    assert errors == []
    [item] = normalized
    assert set(item) == {"index", "session", "messages"} and item["index"] == 0
    session = item["session"]
    assert session["id"] == "a" and session["model_config"] == '{"_branched_from": "p"}'
    assert session["title"] is None and session["parent_session_id"] is None
    first, second = item["messages"]
    assert first["token_count"] == 3 and first["tool_name"] is None and first["reasoning_details"] == '[{"k": 1}]'
    assert second["content"] == [{"type": "text", "text": "x"}] and second["token_count"] is None
    assert raw["id"] == " a " and "tool_name" not in raw["messages"][0]


def test_duplicate_and_missing_ids_and_non_objects():
    assert _errors([_s("a"), _s("a"), _s(" a ")]) == [
        {"index": 1, "error": "duplicate session id", "session_id": "a"},
        {"index": 2, "error": "duplicate session id", "session_id": "a"},
    ]
    assert _errors(["text", 5, None, _s(""), {"source": "cli"}]) == [
        {"index": 0, "error": "session must be an object"},
        {"index": 1, "error": "session must be an object"},
        {"index": 2, "error": "session must be an object"},
        {"index": 3, "error": "session id is required"},
        {"index": 4, "error": "session id is required"},
    ]


def test_message_list_shape():
    assert _failed([_s("a") | {"messages": "nope"}]) == {"a": "messages must be a list"}
    assert _failed([_s("a", [_msg("user", "x", 0), "string message"])]) == {"a": "messages must contain only objects"}
    assert _failed([_s("many", [_msg("user", "m", 0)] * (IMPORT_MAX_MESSAGES_PER_SESSION + 1))]) == {"many": "messages exceeds the per-session import limit"}


def test_size_limits_count_a_rejected_entry_before_its_check():
    normalized, errors = validate_import_payload([_s("big", [_msg("user", "x" * (5 * MiB + 10), 0)]), _s("after")])
    assert errors == [{"index": 0, "error": "session exceeds the import size limit", "session_id": "big"}]
    assert [item["index"] for item in normalized] == [1]
    # Seven sessions of 4.5 MiB: the sixth crosses 25 MiB, and its bytes stay counted, so the seventh is refused too.
    errors = _errors([_s(f"t{i}", [_msg("user", "y" * (4 * MiB + 500 * 1024), 0)]) for i in range(7)])
    assert errors == [
        {"index": 5, "error": "import exceeds the total size limit", "session_id": "t5"},
        {"index": 6, "error": "import exceeds the total size limit", "session_id": "t6"},
    ]


def test_the_total_message_limit():
    per_session = 9_000
    sessions = [_s(f"n{i}", [_msg("user", "m", 0)] * per_session) for i in range(6)]
    assert per_session * 5 <= IMPORT_MAX_TOTAL_MESSAGES < per_session * 6
    assert _errors(sessions) == [{"index": 5, "error": "messages exceeds the total import limit", "session_id": "n5"}]


def test_integer_bounds():
    assert (IMPORT_INT_MIN, IMPORT_INT_MAX) == (-(1 << 63), (1 << 63) - 1)
    assert _failed([
        _s("tok-hi", [_msg("user", "x", 0, token_count=1 << 63)]), _s("tok-max", [_msg("user", "x", 0, token_count=(1 << 63) - 1)]),
        _s("tok-lo", [_msg("user", "x", 0, token_count=-(1 << 63) - 1)]), _s("content-hi", [_msg("user", 1 << 63, 0)]),
        _s("content-bool", [_msg("user", True, 0)]), _s("in-hi", input_tokens=1 << 63), _s("in-max", input_tokens=(1 << 63) - 1),
        _s("api-lo", api_call_count=-(1 << 63) - 1), _s("tok-text", [_msg("user", "x", 0, token_count="seven")]),
    ]) == {
        "tok-hi": f"token_count {INT_RANGE}", "tok-lo": f"token_count {INT_RANGE}", "content-hi": f"session.messages[0].content {INT_RANGE}",
        "in-hi": f"input_tokens {INT_RANGE}", "api-lo": f"api_call_count {INT_RANGE}", "tok-text": "token_count must be an integer",
    }


def test_non_finite_numbers_name_their_path():
    assert _failed([
        _s("cfg", model_config='{"a": 1e309}'), _s("ended", ended_at=float("inf")), _s("cost", estimated_cost_usd=float("nan")),
        _s("content", [_msg("user", float("nan"), 0)]), _s("sidecar", [_msg("user", "x", 0, reasoning_details="[1e999]")]),
        _s("sidecar-obj", [_msg("user", "x", 0, codex_reasoning_items=[{"v": float("inf")}])]),
    ]) == {
        "cfg": "non-finite number at session.model_config.a", "ended": "non-finite number at session.ended_at",
        "cost": "non-finite number at session.estimated_cost_usd", "content": "non-finite number at session.messages[0].content",
        "sidecar": "non-finite number at session.messages[0].reasoning_details[0]",
        "sidecar-obj": "non-finite number at session.messages[0].codex_reasoning_items[0].v",
    }


def test_json_and_text_fields():
    assert _failed([
        _s("cfg-list", model_config="[1]"), _s("cfg-bad", model_config="{nope"), _s("cfg-dict", model_config={"_branched_from": "x"}),
        _s("cfg-type", model_config=5), _s("title", title=5), _s("parent", parent_session_id=7), _s("sidecar-bad", [_msg("user", "x", 0, reasoning_details="{nope")]),
        _s("tool-name", [_msg("tool", "x", 0, tool_name=3)]), _s("role", [{"content": "x"}]), _s("role-empty", [_msg("", "x", 0)]),
        _s("bytes", [_msg("user", b"raw", 0)]),
    ]) == {
        "cfg-list": "model_config must be a JSON object", "cfg-bad": "model_config must be valid JSON", "cfg-type": "model_config must be a JSON object",
        "title": "title must be a string", "parent": "parent_session_id must be a string", "tool-name": "tool_name must be a string",
        "role": "messages[0].role must be a non-empty string", "role-empty": "messages[0].role must be a non-empty string",
        # Bytes are not JSON: the size measure refuses them before the content rule runs.
        "bytes": "session must be JSON serializable",
    }


def test_timestamps_must_fall_in_the_trusted_window():
    assert _failed([
        _s("ts-neg", [_msg("user", "x", -BASE - 5)]), _s("ts-text", [{"role": "user", "content": "x", "timestamp": "soon"}]),
        _s("ts-numeric-text", [{"role": "user", "content": "x", "timestamp": "1700000000"}]), _s("start-empty", started_at=""),
        _s("start-text", started_at="abc"), _s("start-far", started_at=5e12), _s("start-none", started_at=None),
    ]) == {
        "ts-neg": "session.messages[0].timestamp must be a supported Unix timestamp",
        "ts-text": "session.messages[0].timestamp must be a supported Unix timestamp",
        "start-empty": "session.started_at must be a supported Unix timestamp",
        "start-text": "session.started_at must be a supported Unix timestamp",
        "start-far": "session.started_at must be a supported Unix timestamp",
    }


def test_a_timestamp_past_the_float_range_raises():
    # QUIRK: float() overflows and the OverflowError escapes the validator.
    with pytest.raises(OverflowError):
        validate_import_payload([_s("huge", [{"role": "user", "content": "x", "timestamp": 10 ** 400}])])


def test_a_session_that_is_not_json_is_refused():
    normalized, errors = validate_import_payload([_s("set", extra={1, 2}), _s("ok-after")])
    assert errors == [{"index": 0, "error": "session must be JSON serializable", "session_id": "set"}]
    assert [item["session"]["id"] for item in normalized] == ["ok-after"]
