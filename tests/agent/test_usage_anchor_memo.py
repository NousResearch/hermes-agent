"""Anchor-local memoization preserves the full priced-prefix identity contract."""

import json
import random
from copy import deepcopy
from datetime import datetime
from operator import delitem, setitem
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import usage_anchor as ua


def _messages(count=400):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row {i}"}
            for i in range(count)]


def _counter(monkeypatch):
    counter = Mock(wraps=ua.message_fingerprint)
    monkeypatch.setattr(ua, "message_fingerprint", counter)
    return counter


def _full_match(messages, anchor):
    """The base algorithm, deliberately taking the uncached prefix path."""
    count = int(anchor["base_count"])
    if count <= 0 or len(messages) < count:
        return False
    last = messages[count - 1]
    return (isinstance(last, dict)
            and last.get("role") == anchor["base_last_role"]
            and ua.message_fingerprint(last) == anchor["base_last_fp"]
            and ua._priced_prefix_fingerprint(messages, count) == anchor["base_prefix_fp"])


def test_unchanged_transcript_never_rehashes_rows(monkeypatch):
    """Red on base: ten checks hash 4,010 rows."""
    messages = _messages()
    anchor = ua.capture_usage_anchor(10_000, 100, messages)
    counter = _counter(monkeypatch)
    for _ in range(10):
        assert ua.anchored_context_tokens(messages, anchor) == 10_100
    assert counter.call_count == 0


def test_cli_redraw_never_rehashes_rows(monkeypatch):
    """Exercise the classic CLI snapshot; red on base."""
    from cli import HermesCLI

    messages = _messages()
    anchor = ua.capture_usage_anchor(10_000, 100, messages)
    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "test-model"
    cli.session_start = datetime.now()
    cli.agent = SimpleNamespace(
        _session_messages=messages, _turn_base_usage_anchor=anchor,
        context_compressor=SimpleNamespace(last_prompt_tokens=1, context_length=100_000),
    )
    counter = _counter(monkeypatch)
    for _ in range(10):
        assert cli._get_status_bar_snapshot()["context_tokens"] == 10_100
    assert counter.call_count == 0


def test_stale_last_row_fails_on_one_hash(monkeypatch):
    """A changed last priced row is rejected before the prefix walk, as on base."""
    plain = ua._serialize(ua.capture_usage_anchor(10_000, 100, _messages()))
    db = SimpleNamespace(get_session_model_config_value=lambda *args: plain)
    fresh = _messages()
    fresh[-1]["content"] = "edited"
    counter = _counter(monkeypatch)
    assert ua.persisted_anchor_tokens(db, "test-session", fresh) is None
    assert counter.call_count == 1


def test_same_length_earlier_rewrite_is_stale():
    messages = _messages(4)
    anchor = ua.capture_usage_anchor(100, 10, messages)
    messages[0]["content"] = "ROW 0"
    assert not _full_match(messages, anchor)
    assert ua.anchored_context_tokens(messages, anchor) is None


def test_equal_replacement_row_is_rehashed(monkeypatch):
    messages = _messages(4)
    anchor = ua.capture_usage_anchor(100, 10, messages)
    messages[0] = dict(messages[0])
    counter = _counter(monkeypatch)
    assert ua.anchored_context_tokens(messages, anchor) == 110
    assert any(call.args[0] is messages[0] for call in counter.call_args_list)
    counter.reset_mock()
    assert ua.anchored_context_tokens(messages, anchor) == 110
    assert counter.call_count == 0  # Red on base; a successful match refreshes the memo.


@pytest.mark.parametrize("row", [1, 3])
def test_in_place_tool_call_mutation_is_stale(row):
    messages = _messages(4)
    messages[row]["tool_calls"] = [{"id": "call", "function": {"arguments": "{}"}}]
    anchor = ua.capture_usage_anchor(100, 10, messages)
    messages[row]["tool_calls"][0]["function"]["arguments"] = '{"changed":true}'
    assert not _full_match(messages, anchor)
    assert ua.anchored_context_tokens(messages, anchor) is None


def test_multimodal_content_is_always_rehashed(monkeypatch):
    messages = _messages(4)
    messages[0]["content"] = [{"type": "text", "text": "image caption"}]
    anchor = ua.capture_usage_anchor(100, 10, messages)
    counter = _counter(monkeypatch)
    for _ in range(2):
        counter.reset_mock()
        assert ua.anchored_context_tokens(messages, anchor) == 110
        assert any(call.args[0] is messages[0] for call in counter.call_args_list)
    messages[0]["content"][0]["text"] = "rewritten caption"
    assert ua.anchored_context_tokens(messages, anchor) is None


def test_persisted_round_trip_warms_memo(monkeypatch):
    """Restoration validates once, then checks avoid hashing; red on base."""
    messages = _messages(4)
    anchor = ua.capture_usage_anchor(100, 10, messages)
    plain = ua._serialize(anchor)
    assert type(plain) is dict
    assert set(plain) == {"prompt_tokens", "completion_tokens", "base_count",
                          "base_last_role", "base_last_fp", "base_prefix_fp"}
    assert plain == anchor
    assert json.dumps(anchor) == json.dumps(plain)
    db = SimpleNamespace(get_session_model_config_value=lambda *args: plain)
    agent = SimpleNamespace(session_id="test-session", _session_db=db)
    counter = _counter(monkeypatch)
    ua.restore_usage_anchor(agent, deepcopy(messages))
    assert counter.call_count > 0
    restored_messages = _messages(4)
    assert ua.anchored_context_tokens(restored_messages, agent._usage_anchor) == 110
    counter.reset_mock()
    assert ua.anchored_context_tokens(restored_messages, agent._usage_anchor) == 110
    assert counter.call_count == 0


def test_persisted_tokens_use_memo_capable_anchor(monkeypatch):
    messages = _messages(4)
    plain = ua._serialize(ua.capture_usage_anchor(100, 10, messages))
    db = SimpleNamespace(get_session_model_config_value=lambda *args: plain)
    original = ua.anchored_context_tokens
    seen = []

    def observe(messages, anchor):
        seen.append(anchor)
        return original(messages, anchor)

    monkeypatch.setattr(ua, "anchored_context_tokens", observe)
    assert ua.persisted_anchor_tokens(db, "test-session", messages) == 110
    counter = _counter(monkeypatch)
    assert original(messages, seen[0]) == 110
    assert counter.call_count == 0  # Red on base.


@pytest.mark.parametrize("field", ["base_count", "base_last_role", "base_last_fp", "base_prefix_fp"])
def test_public_anchor_edits_still_match_base(field):
    messages = _messages(4)
    anchor = ua.capture_usage_anchor(100, 10, messages)
    anchor[field] = 3 if field == "base_count" else "changed"
    assert ua._anchor_matches(messages, anchor) == _full_match(messages, anchor)


@pytest.mark.parametrize("seed", range(5))
def test_randomized_edit_scripts_match_full_hash(seed):
    rng = random.Random(seed)
    messages = _messages(8)
    messages[1]["tool_calls"] = [{"id": "call", "function": {"arguments": "{}"}}]
    anchor = ua.capture_usage_anchor(100, 10, messages)
    for _ in range(40):
        if len(messages) < 3:
            messages = _messages(8)
            messages[1]["tool_calls"] = []
            anchor = ua.capture_usage_anchor(100, 10, messages)
        op = rng.choice(("append", "edit", "replace", "mutate", "truncate"))
        index = rng.randrange(len(messages))
        edits = {
            "append": lambda: messages.append({"role": "user", "content": "next"}),
            "edit": lambda: setitem(messages[index], "content", f"edit {rng.randrange(100)}"),
            "replace": lambda: setitem(messages, index, deepcopy(messages[index])),
            "mutate": lambda: messages[1].setdefault("tool_calls", []).append({"id": f"call-{index}"}),
            "truncate": lambda: delitem(messages, slice(index, None)),
        }
        edits[op]()
        expected = _full_match(messages, anchor)
        assert ua._anchor_matches(messages, anchor) == expected
        assert ua._anchor_matches(messages, dict(anchor)) == expected
