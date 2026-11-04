"""Unit tests for scripts/replay_refusal.py.

Tests use a provider mock (no real API calls).
Verifies:
1. Message loading and truncation logic
2. Tool result exclusion in bisect mode
3. Refusal detection from mocked provider response
4. Correct message index identified when mock returns refusal → stop after exclusion

Run with: pytest tests/scripts/test_replay_refusal.py -v
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — the script lives in scripts/ at the repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # tests/scripts/ → root
sys.path.insert(0, str(_REPO_ROOT))

from scripts.replay_refusal import (  # noqa: E402
    _convert_to_anthropic_messages,
    _exclude_tool_result,
    _safe_content_summary,
    _sha256,
    _tool_result_indices,
    _truncate_to_tool_result_n,
    _replay_once,
)


# ---------------------------------------------------------------------------
# Fixtures — a minimal but realistic message sequence
# ---------------------------------------------------------------------------

def _make_messages() -> list[dict]:
    """Return a synthetic message list with 2 tool round-trips.

    Structure:
      user(0) → assistant[tool_use A](1) → tool A result(2)
              → assistant[tool_use B](3) → tool B result(4)
              → user follow-up(5)
    """
    return [
        {"role": "user", "content": "Hello, run some tools."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_A",
                    "call_id": "tc_A",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command":"echo hello"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_A",
            "tool_name": "terminal",
            "content": "hello\n",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_B",
                    "call_id": "tc_B",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"/tmp/x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc_B",
            "tool_name": "read_file",
            "content": "file content goes here",
        },
        {"role": "user", "content": "Thanks."},
    ]


# ---------------------------------------------------------------------------
# Tests: utility helpers
# ---------------------------------------------------------------------------


def test_sha256_deterministic():
    h = _sha256("hello")
    assert h == hashlib.sha256(b"hello").hexdigest()
    assert len(h) == 64


def test_sha256_bytes():
    assert _sha256(b"abc") == _sha256("abc")


def test_safe_content_summary_never_exposes_content():
    msg = {"role": "tool", "tool_name": "terminal", "content": "SECRET DATA"}
    summary = _safe_content_summary(msg)
    assert summary["char_count"] == len("SECRET DATA")
    assert "SECRET DATA" not in str(summary)
    assert len(summary["sha256"]) == 64
    assert summary["tool_name"] == "terminal"


def test_safe_content_summary_list_content():
    msg = {
        "role": "tool",
        "tool_name": "read_file",
        "content": [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}],
    }
    summary = _safe_content_summary(msg)
    # "foo" + " " + "bar" = 7 chars (space separated)
    assert summary["char_count"] == 7


# ---------------------------------------------------------------------------
# Tests: message filtering
# ---------------------------------------------------------------------------


def test_tool_result_indices(tmp_path):
    msgs = _make_messages()
    indices = _tool_result_indices(msgs)
    assert indices == [2, 4], f"expected [2, 4], got {indices}"


def test_truncate_to_tool_result_n_first():
    msgs = _make_messages()
    truncated = _truncate_to_tool_result_n(msgs, 1)
    # Should include up to and including index 2 (first tool result)
    assert truncated == msgs[:3]
    # Still has the first tool result
    assert truncated[-1]["role"] == "tool"
    assert truncated[-1]["tool_call_id"] == "tc_A"


def test_truncate_to_tool_result_n_all():
    msgs = _make_messages()
    truncated = _truncate_to_tool_result_n(msgs, 2)
    # 2 tool results = include everything up to index 4
    assert truncated == msgs[:5]


def test_truncate_to_tool_result_n_exceeds():
    msgs = _make_messages()
    truncated = _truncate_to_tool_result_n(msgs, 99)
    assert truncated == msgs  # no truncation


def test_truncate_to_tool_result_n_zero():
    msgs = _make_messages()
    assert _truncate_to_tool_result_n(msgs, 0) == msgs  # 0 means no truncation


def test_exclude_tool_result_replaces_content():
    msgs = _make_messages()
    result = _exclude_tool_result(msgs, 0)  # exclude the FIRST tool result (0-based)
    # The structure length is unchanged
    assert len(result) == len(msgs)
    # The first tool result's content has been replaced
    assert result[2]["content"] == "[tool result excluded by replay_refusal.py bisect]"
    # The second tool result is untouched
    assert result[4]["content"] == "file content goes here"


def test_exclude_tool_result_does_not_mutate_original():
    msgs = _make_messages()
    _ = _exclude_tool_result(msgs, 0)
    assert msgs[2]["content"] == "hello\n"


def test_exclude_tool_result_out_of_range():
    msgs = _make_messages()
    result = _exclude_tool_result(msgs, 99)
    assert result == msgs  # no change


# ---------------------------------------------------------------------------
# Tests: message conversion
# ---------------------------------------------------------------------------


def test_convert_basic_user_assistant():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    system, converted = _convert_to_anthropic_messages(msgs)
    assert system is None
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    # Content should be a list of blocks
    assert isinstance(converted[0]["content"], list)
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][0]["text"] == "Hi"


def test_convert_system_extracted():
    msgs = [
        {"role": "system", "content": "You are a tester."},
        {"role": "user", "content": "Run a test"},
    ]
    system, converted = _convert_to_anthropic_messages(msgs)
    assert system == "You are a tester."
    assert all(m["role"] != "system" for m in converted)


def test_convert_tool_result_merged_into_user_turn():
    msgs = _make_messages()
    _, converted = _convert_to_anthropic_messages(msgs)
    # Find the user turn that follows the first assistant tool_use
    # It should contain a tool_result block
    tool_result_turns = [
        m for m in converted
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m.get("content", []) if isinstance(b, dict))
    ]
    assert len(tool_result_turns) >= 1, "No tool_result blocks found in user turns"


def test_convert_assistant_with_tool_calls():
    msgs = [
        {"role": "user", "content": "Do it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                }
            ],
        },
    ]
    _, converted = _convert_to_anthropic_messages(msgs)
    asst = next(m for m in converted if m["role"] == "assistant")
    tool_use_blocks = [b for b in asst["content"] if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "terminal"
    assert tool_use_blocks[0]["input"] == {"command": "ls"}


def test_convert_starts_with_user():
    # Messages that start with assistant should have a user turn prepended
    msgs = [
        {"role": "assistant", "content": "I go first"},
        {"role": "user", "content": "OK"},
    ]
    _, converted = _convert_to_anthropic_messages(msgs)
    assert converted[0]["role"] == "user"


def test_convert_no_duplicate_consecutive_roles():
    # Two consecutive user messages must be merged
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
    ]
    _, converted = _convert_to_anthropic_messages(msgs)
    roles = [m["role"] for m in converted]
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1], f"Consecutive {roles[i]} at index {i}"


# ---------------------------------------------------------------------------
# Tests: _replay_once with mocked provider
# ---------------------------------------------------------------------------


def _make_mock_client(stop_reason: str, text_content: str = "") -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns *stop_reason*."""
    block = SimpleNamespace(type="text", text=text_content)
    response = SimpleNamespace(
        stop_reason=stop_reason,
        content=[block] if text_content else [],
    )
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_replay_once_stop():
    client = _make_mock_client("stop")
    msgs = _make_messages()
    result = _replay_once(client, msgs, model="test-model", max_tokens=16)
    assert result["stop_reason"] == "stop"
    assert result["error"] is None
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0


def test_replay_once_refusal():
    client = _make_mock_client("refusal")
    msgs = _make_messages()
    result = _replay_once(client, msgs, model="test-model", max_tokens=16)
    assert result["stop_reason"] == "refusal"
    assert result["refusal_len"] == 0  # no text in the mock refusal


def test_replay_once_refusal_with_text():
    client = _make_mock_client("refusal", text_content="Policy violation detected.")
    msgs = _make_messages()
    result = _replay_once(client, msgs, model="test-model", max_tokens=16)
    assert result["stop_reason"] == "refusal"
    assert result["refusal_len"] == len("Policy violation detected.")
    assert "Policy" in result["refusal_text_snippet"]


def test_replay_once_api_exception():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("network error")
    msgs = _make_messages()
    result = _replay_once(client, msgs, model="test-model", max_tokens=16)
    assert result["stop_reason"] == "error"
    assert "network error" in (result.get("error") or "")


# ---------------------------------------------------------------------------
# Tests: bisect integration — mock provider simulates refusal only when
#        the last tool result (tc_B) is present; excluding it clears the refusal
# ---------------------------------------------------------------------------


def test_bisect_identifies_trigger_tool_result(capsys):
    """When the second tool result causes refusal, bisect should identify rank=2."""
    msgs = _make_messages()

    def _smart_create(**kwargs):
        """Return refusal if tc_B tool_result block is present in the messages."""
        messages_arg = kwargs.get("messages", [])
        for turn in messages_arg:
            if turn.get("role") != "user":
                continue
            for block in turn.get("content", []):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") == "tc_B"
                    # Only trigger if content is NOT the placeholder
                    and block.get("content") != "[tool result excluded by replay_refusal.py bisect]"
                ):
                    return SimpleNamespace(stop_reason="refusal", content=[])
        return SimpleNamespace(stop_reason="stop", content=[SimpleNamespace(type="text", text="OK")])

    client = MagicMock()
    client.messages.create.side_effect = _smart_create

    n_tool_results = len(_tool_result_indices(msgs))
    assert n_tool_results == 2

    # Baseline — should be refusal (tc_B is present)
    baseline = _replay_once(client, msgs, model="mock", max_tokens=16)
    assert baseline["stop_reason"] == "refusal", "baseline should refuse"

    # Exclude rank 1 (tc_A) — refusal should PERSIST (tc_B still there)
    excluded_1 = _exclude_tool_result(msgs, 0)
    result_1 = _replay_once(client, excluded_1, model="mock", max_tokens=16)
    assert result_1["stop_reason"] == "refusal", (
        "refusal should persist when tc_A excluded (tc_B still present)"
    )

    # Exclude rank 2 (tc_B) — refusal should CLEAR
    excluded_2 = _exclude_tool_result(msgs, 1)
    result_2 = _replay_once(client, excluded_2, model="mock", max_tokens=16)
    assert result_2["stop_reason"] == "stop", (
        "refusal should clear when tc_B (the trigger) is excluded"
    )
