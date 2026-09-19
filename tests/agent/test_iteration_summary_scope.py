"""Regression coverage for max-iteration summary input scoping."""

from __future__ import annotations

import json
import types
from unittest.mock import patch


def _summary(stale: str) -> str:
    from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER

    return f"{SUMMARY_PREFIX}\n{stale}\n{_SUMMARY_END_MARKER}"


def test_scope_excludes_standalone_compaction_handoff():
    from agent.chat_completion_helpers import _iteration_summary_messages
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

    stale = "STALE_COMPACTION_HANDOFF"
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "assistant", "content": _summary(stale), COMPRESSED_SUMMARY_METADATA_KEY: True},
        {"role": "user", "content": "Continue"},
    ]
    agent = types.SimpleNamespace(_persist_user_message_idx=3)

    scoped = _iteration_summary_messages(agent, messages)

    assert stale not in json.dumps(scoped)


def test_scope_uses_live_user_view_from_merged_compaction_row():
    from agent.chat_completion_helpers import _iteration_summary_messages
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
        COMPRESSED_SUMMARY_METADATA_KEY,
    )

    stale = "STALE_MERGED_COMPACTION"
    merged = {
        "role": "user",
        "content": _summary(stale) + "\n\nContinue",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
        COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: True,
    }
    agent = types.SimpleNamespace(_persist_user_message_idx=0)

    scoped = _iteration_summary_messages(agent, [merged])

    assert stale not in json.dumps(scoped)
    assert scoped[0]["content"] == "Continue"


def test_scope_drops_old_task_and_keeps_current_tool_result():
    from agent.chat_completion_helpers import _iteration_summary_messages

    stale = "STALE_MODEL_CATALOG_RESULT"
    current = "CURRENT_OCR_TESTS_PASS"
    messages = [
        {"role": "user", "content": "Inspect providers"},
        {"role": "assistant", "content": stale},
        {"role": "user", "content": "Continue OCR"},
        {"role": "assistant", "content": "OCR handoff"},
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": current},
    ]
    agent = types.SimpleNamespace(_persist_user_message_idx=4)

    scoped = _iteration_summary_messages(agent, messages)
    wire = json.dumps(scoped)

    assert stale not in wire
    assert "OCR handoff" in wire
    assert current in wire
