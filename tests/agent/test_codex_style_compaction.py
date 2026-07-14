"""Behavioral tests for the Codex-style replacement-history algorithm."""

from agent.codex_style_compaction import (
    CODEX_COMPACT_USER_MESSAGE_MAX_TOKENS,
    build_compacted_history,
    collect_user_messages,
)


def test_collect_user_messages_excludes_old_summaries_and_non_user_items():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nold summary"},
        {"role": "tool", "content": "large tool output"},
        {"role": "user", "content": "latest question"},
    ]

    assert collect_user_messages(messages) == [
        "old question",
        "latest question",
    ]


def test_build_history_keeps_recent_user_messages_and_appends_summary():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "second"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "third"},
    ]

    result = build_compacted_history(messages, "[SUMMARY]\ncompleted work")

    assert result == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "user", "content": "[SUMMARY]\ncompleted work"},
    ]


def test_build_history_uses_recent_user_token_budget():
    # Four characters are the Hermes rough-token convention used by the
    # existing compressor. This makes the 20K Codex boundary deterministic.
    old = "o" * 32_000       # ~8K tokens
    middle = "m" * 40_000    # ~10K tokens
    newest = "n" * 8_000     # ~2K tokens
    messages = [
        {"role": "user", "content": old},
        {"role": "user", "content": middle},
        {"role": "user", "content": newest},
    ]

    result = build_compacted_history(messages, "summary")
    selected = result[:-1]

    assert len(selected) == 3
    assert selected[0]["content"] == old
    assert selected[1]["content"] == middle
    assert selected[2]["content"] == newest
    assert sum(len(item["content"]) // 4 for item in selected) <= CODEX_COMPACT_USER_MESSAGE_MAX_TOKENS


def test_build_history_truncates_oldest_selected_user_message_to_remaining_budget():
    first = "a" * 80_000       # ~20K tokens
    newest = "b" * 8_000       # ~2K tokens
    messages = [
        {"role": "user", "content": first},
        {"role": "user", "content": newest},
    ]

    result = build_compacted_history(messages, "summary")
    selected = result[:-1]

    assert selected[-1]["content"] == newest
    assert len(selected[0]["content"]) // 4 == CODEX_COMPACT_USER_MESSAGE_MAX_TOKENS - len(newest) // 4
    assert selected[0]["content"].startswith("a")
