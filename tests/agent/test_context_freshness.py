"""Contracts for the first-milestone context-freshness layer.

The resolver must drop reasoning that predates the latest compaction
summary and cap replay to the last N post-summary assistant turns.
Persisted message dicts are left unchanged — only request assembly
filters replay.
"""

from agent.context_freshness import (
    DEFAULT_REASONING_REPLAY_KEEP_LAST,
    PREVIOUS_SUMMARY_UNVERIFIED_GUIDANCE,
    clamp_reasoning_replay_keep_last,
    last_compaction_index,
    reasoning_replay_keep_indices,
)
from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.context_compressor import SUMMARY_PREFIX


def _reasoning(blob: str) -> list[dict]:
    return [{"type": "reasoning", "encrypted_content": blob, "id": f"rs_{blob}"}]


def _assistant(text: str, blob: str | None = None) -> dict:
    msg = {"role": "assistant", "content": text}
    if blob is not None:
        msg["codex_reasoning_items"] = _reasoning(blob)
    return msg


def _summary() -> dict:
    return {
        "role": "user",
        "content": f"{SUMMARY_PREFIX}\n## Goal\nPrior work.",
    }


def test_clamp_keep_last_rejects_bool_and_fraction():
    assert clamp_reasoning_replay_keep_last(None) == DEFAULT_REASONING_REPLAY_KEEP_LAST
    assert clamp_reasoning_replay_keep_last(True) == DEFAULT_REASONING_REPLAY_KEEP_LAST
    assert clamp_reasoning_replay_keep_last(1.5) == DEFAULT_REASONING_REPLAY_KEEP_LAST
    assert clamp_reasoning_replay_keep_last(-3) == 0
    assert clamp_reasoning_replay_keep_last(0) == 0
    assert clamp_reasoning_replay_keep_last(4) == 4


def test_last_compaction_index_finds_latest_summary():
    messages = [
        {"role": "user", "content": "start"},
        _assistant("old", "old_blob"),
        _summary(),
        {"role": "user", "content": "continue"},
        _summary(),
        _assistant("new", "new_blob"),
    ]
    assert last_compaction_index(messages) == 4


def test_keep_indices_drop_pre_summary_and_cap_recent():
    messages = [
        {"role": "user", "content": "a"},
        _assistant("one", "b1"),
        {"role": "user", "content": "b"},
        _assistant("two", "b2"),
        _summary(),
        {"role": "user", "content": "c"},
        _assistant("three", "b3"),
        {"role": "user", "content": "d"},
        _assistant("four", "b4"),
        {"role": "user", "content": "e"},
        _assistant("five", "b5"),
    ]
    kept = reasoning_replay_keep_indices(messages, keep_last=2)
    assert kept == frozenset({8, 10})
    # History is not mutated.
    assert messages[1]["codex_reasoning_items"][0]["encrypted_content"] == "b1"


def test_keep_indices_zero_keep_last_is_summary_bound_only():
    messages = [
        _assistant("old", "old"),
        _summary(),
        _assistant("new1", "n1"),
        _assistant("new2", "n2"),
        _assistant("new3", "n3"),
    ]
    kept = reasoning_replay_keep_indices(messages, keep_last=0)
    assert kept == frozenset({2, 3, 4})


def test_adapter_replays_only_kept_reasoning():
    messages = [
        {"role": "user", "content": "start"},
        _assistant("stale interpretation", "stale"),
        _summary(),
        {"role": "user", "content": "correction: use postgres"},
        _assistant("ack", "recent1"),
        {"role": "user", "content": "and pin 16"},
        _assistant("ack2", "recent2"),
        {"role": "user", "content": "go"},
        _assistant("working", "recent3"),
    ]
    items = _chat_messages_to_responses_input(
        messages, reasoning_replay_keep_last=2
    )
    blobs = [
        item.get("encrypted_content")
        for item in items
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert "stale" not in blobs
    assert blobs == ["recent2", "recent3"]
    # Persisted history still holds the dropped blob.
    assert messages[1]["codex_reasoning_items"][0]["encrypted_content"] == "stale"


def test_adapter_keeps_native_compaction_checkpoint_outside_keep_window():
    messages = [
        {
            "role": "assistant",
            "content": "old",
            "codex_reasoning_items": [
                {
                    "type": "compaction",
                    "encrypted_content": "checkpoint_blob",
                    "id": "cmp_1",
                },
                {
                    "type": "reasoning",
                    "encrypted_content": "stale_reason",
                    "id": "rs_old",
                },
            ],
        },
        _summary(),
        _assistant("new", "fresh"),
    ]
    items = _chat_messages_to_responses_input(
        messages,
        reasoning_replay_keep_last=1,
        native_compaction_eligible=True,
    )
    blobs = [
        (item.get("type"), item.get("encrypted_content"))
        for item in items
        if isinstance(item, dict) and item.get("encrypted_content")
    ]
    assert ("compaction", "checkpoint_blob") in blobs
    assert ("reasoning", "stale_reason") not in blobs
    assert ("reasoning", "fresh") in blobs


def test_adapter_default_still_replays_a_single_recent_item():
    messages = [
        {"role": "user", "content": "hello"},
        _assistant("hi", "enc123"),
        {"role": "user", "content": "follow up"},
    ]
    items = _chat_messages_to_responses_input(messages)
    blobs = [
        item.get("encrypted_content")
        for item in items
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert blobs == ["enc123"]


def test_unverified_guidance_is_actionable():
    assert "unverified" in PREVIOUS_SUMMARY_UNVERIFIED_GUIDANCE.lower()
    assert "new turns" in PREVIOUS_SUMMARY_UNVERIFIED_GUIDANCE.lower()
    assert "ground truth" in PREVIOUS_SUMMARY_UNVERIFIED_GUIDANCE.lower()
