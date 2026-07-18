from __future__ import annotations

from agent.iteration_budget import (
    CURRENT_TURN_ID_KEY,
    _current_turn_user_index,
    _latest_unseen_tool_result,
    kanban_checkpoint_warning,
)


def test_warning_fires_once_at_configured_budget_ratio() -> None:
    assert (
        kanban_checkpoint_warning(
            used=44,
            max_total=60,
            ratio=0.75,
            is_kanban=True,
            already_emitted=False,
        )
        is None
    )

    warning = kanban_checkpoint_warning(
        used=45,
        max_total=60,
        ratio=0.75,
        is_kanban=True,
        already_emitted=False,
    )
    assert warning is not None
    assert "45/60" in warning
    assert "atomic claim" in warning
    assert "Do not claim PASS" in warning
    assert len(warning) < 700

    assert (
        kanban_checkpoint_warning(
            used=46,
            max_total=60,
            ratio=0.75,
            is_kanban=True,
            already_emitted=True,
        )
        is None
    )


def test_warning_never_changes_non_kanban_turns() -> None:
    assert (
        kanban_checkpoint_warning(
            used=59,
            max_total=60,
            ratio=0.75,
            is_kanban=False,
            already_emitted=False,
        )
        is None
    )


def test_checkpoint_targets_only_a_tool_result_not_sent_before() -> None:
    old = {"role": "tool", "tool_call_id": "old", "content": "old result"}
    new = {"role": "tool", "tool_call_id": "new", "content": "new result"}
    messages = [
        {"role": "user", "content": "task"},
        old,
        {"role": "assistant", "content": "continuing"},
        new,
    ]

    assert _latest_unseen_tool_result(
        messages,
        seen_tool_call_ids={"old"},
        seen_legacy_message_ids={id(old)},
    ) == (3, "new")
    assert (
        _latest_unseen_tool_result(
            messages,
            seen_tool_call_ids={"old", "new"},
            seen_legacy_message_ids={id(old), id(new)},
        )
        is None
    )


def test_checkpoint_never_reanchors_before_current_turn_or_to_legacy_result() -> None:
    historical = {
        "role": "tool",
        "tool_call_id": "historical",
        "content": "old result",
    }
    legacy = {"role": "tool", "content": "rebuilt old result"}
    current = {
        "role": "tool",
        "tool_call_id": "current",
        "content": [{"type": "text", "text": "new multimodal result"}],
    }
    messages = [
        historical,
        legacy,
        {"role": "user", "content": "current turn"},
        current,
    ]

    assert _latest_unseen_tool_result(
        messages,
        seen_tool_call_ids=set(),
        seen_legacy_message_ids=set(),
        min_index=3,
        require_tool_call_id=True,
    ) == (3, "current")
    assert (
        _latest_unseen_tool_result(
            messages[:3],
            seen_tool_call_ids=set(),
            seen_legacy_message_ids=set(),
            min_index=2,
            require_tool_call_id=True,
        )
        is None
    )


def test_current_turn_boundary_survives_compaction_copies() -> None:
    historical = {
        "role": "user",
        "content": "same text",
        CURRENT_TURN_ID_KEY: "old-turn",
    }
    current = {
        "role": "user",
        "content": "same text",
        CURRENT_TURN_ID_KEY: "current-turn",
    }
    compressed = [message.copy() for message in [historical, current]]

    assert _current_turn_user_index(compressed, "current-turn") == 1
    assert _current_turn_user_index(compressed, "missing-turn") is None
