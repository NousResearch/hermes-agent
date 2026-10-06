"""Regression for #80449 — an oversized in-progress turn must stay compressible.

When one turn (opening user message + many individually small tool groups) grows past
the protected-tail soft ceiling, anchoring the cut back to the turn-opening request kept
the whole turn verbatim: compaction re-fired with an empty summarizable window and the
session sat over threshold. The cut must instead land on a tool-group-aligned mid-turn
boundary, and the active request must survive the handoff.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _estimate_msg_budget_tokens,
)


_ACTIVE_REQUEST = "Inspect every shard and preserve the active request exactly."
_TOKEN_BUDGET = 250


def _make_compressor(**overrides) -> ContextCompressor:
    """Compressor with an explicit small tail budget so the ceiling is reachable in a short transcript."""
    kwargs = {
        "model": "test/model",
        "threshold_percent": 0.85,
        "protect_first_n": 0,
        "protect_last_n": 3,
        "quiet_mode": True,
    }
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(**kwargs)
        _ = instance.context_length
    instance.tail_token_budget = _TOKEN_BUDGET
    return instance


@pytest.fixture()
def compressor() -> ContextCompressor:
    return _make_compressor()


def _tool_group(index: int) -> list[dict]:
    call_id = f"call_{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "inspect_shard",
                        # Large enough for the complete turn to exceed the
                        # ceiling, but below the phase-1 argument-prune limit.
                        "arguments": "x" * 440,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            # Keep each result below the phase-1 result-prune floor. The bug is
            # aggregate turn size, not one individually oversized result.
            "content": f"result-{index}:" + "r" * 110,
        },
    ]


def _oversized_active_turn() -> list[dict]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older request completed"},
        {"role": "user", "content": _ACTIVE_REQUEST},
    ]
    for index in range(10):
        messages.extend(_tool_group(index))
    return messages


def _assert_tool_pairs_are_complete(messages: list[dict]) -> None:
    call_ids = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls") or []
    }
    result_ids = {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    }
    assert call_ids == result_ids


def test_oversized_active_turn_uses_a_mid_turn_tool_boundary(
    compressor: ContextCompressor,
) -> None:
    messages = _oversized_active_turn()
    head_end = compressor._protect_head_size(messages)

    cut = compressor._find_tail_cut_by_tokens(
        messages,
        head_end,
        token_budget=_TOKEN_BUDGET,
    )

    active_user_idx = next(
        index
        for index, message in enumerate(messages)
        if message.get("content") == _ACTIVE_REQUEST
    )
    tail_tokens = sum(_estimate_msg_budget_tokens(msg) for msg in messages[cut:])

    assert cut > active_user_idx
    # Tool-group alignment may retain one additional indivisible group beyond
    # the scalar ceiling. It must not retain the whole active turn.
    max_group_tokens = max(
        sum(_estimate_msg_budget_tokens(msg) for msg in _tool_group(index))
        for index in range(10)
    )
    assert tail_tokens <= int(_TOKEN_BUDGET * 1.5) + max_group_tokens
    assert tail_tokens < sum(
        _estimate_msg_budget_tokens(msg) for msg in messages[active_user_idx:]
    )
    assert messages[cut]["role"] == "assistant"
    _assert_tool_pairs_are_complete(messages[head_end:cut])
    _assert_tool_pairs_are_complete(messages[cut:])


def test_full_compaction_preserves_active_request_and_tool_pairs(
    compressor: ContextCompressor,
) -> None:
    messages = _oversized_active_turn()

    # Exercise the deterministic handoff too: even when the summary model is
    # unavailable, splitting the turn must not lose the opening request.
    with patch.object(compressor, "_generate_summary", return_value=None):
        compressed = compressor.compress(messages, current_tokens=90_000)

    summary_rows = [
        message
        for message in compressed
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(summary_rows) == 1
    assert _ACTIVE_REQUEST in str(summary_rows[0].get("content"))
    assert sum(
        _ACTIVE_REQUEST in str(message.get("content"))
        for message in compressed
    ) == 1
    assert len(compressed) < len(messages)
    _assert_tool_pairs_are_complete(compressed)


def test_n_user_tail_guarantee_outranks_the_split() -> None:
    """compression.min_tail_user_messages is a user-facing promise (#70250).

    The oversized-turn exception must not void it: with N > 1 the N-user tail
    anchor wins even when one turn alone exceeds the soft ceiling.
    """
    compressor = _make_compressor(protect_first_n=1, min_tail_user_messages=3)

    user_turns = ["first request", "second request", _ACTIVE_REQUEST]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "earlier request"},
        {"role": "assistant", "content": "earlier request completed"},
    ]
    for text in user_turns:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": f"{text} completed"})
    for index in range(10):
        messages.extend(_tool_group(index))

    cut = compressor._find_tail_cut_by_tokens(
        messages,
        compressor._protect_head_size(messages),
        token_budget=_TOKEN_BUDGET,
    )

    tail = messages[cut:]
    assert [m["content"] for m in tail if m.get("role") == "user"] == user_turns


def test_a_tail_that_fits_the_budget_still_anchors_the_active_request() -> None:
    """The exception is for a turn that overflows the budget, not for one that fits.

    With the whole transcript inside the tail budget, keeping the active request
    verbatim costs nothing, so the anchor must still hold and the exception must
    not fire just because the turn happens to be built from tool groups.
    """
    compressor = _make_compressor()
    compressor.tail_token_budget = 10_000
    messages = _oversized_active_turn()

    cut = compressor._find_tail_cut_by_tokens(messages, compressor._protect_head_size(messages))

    active_user_idx = next(
        index
        for index, message in enumerate(messages)
        if message.get("content") == _ACTIVE_REQUEST
    )
    assert cut <= active_user_idx, "active request must stay inside the protected tail"
    assert any(
        m.get("content") == _ACTIVE_REQUEST for m in messages[cut:]
    )


def test_active_request_survives_repeated_compaction_and_restart(tmp_path) -> None:
    # Fallback compaction (no LLM summary) + SQLite reload between cycles:
    # the active request must be recognized from persisted content alone.
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, _SUMMARY_END_MARKER
    from agent.conversation_compression import _ensure_compressed_has_user_turn
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    session_id = "active-turn-restart"
    db.create_session(session_id, "test")
    messages = _oversized_active_turn()
    try:
        for cycle in range(3):
            if cycle:
                for index in range(10 * cycle, 10 * (cycle + 1)):
                    messages.extend(_tool_group(index))
            original = messages
            compressor = _make_compressor()
            with patch.object(compressor, "_generate_summary", return_value=None):
                messages = compressor.compress(original, current_tokens=90_000, force=True)
            _ensure_compressed_has_user_turn(original, messages)
            assert len(messages) < len(original)
            _assert_tool_pairs_are_complete(messages)
            # Historical summaries may quote the request. Count only actionable
            # text after their boundary, not those explicitly historical quotes.
            user_content = "\n".join(
                str(m.get("content")).rsplit(_SUMMARY_END_MARKER, 1)[-1]
                for m in messages if m["role"] == "user"
            )
            assert user_content.count(_ACTIVE_REQUEST) == 1
            assert user_content.count(_INFLIGHT_TASK_REPLAY_HEADER) == 1
            assert user_content.rfind(_ACTIVE_REQUEST) > user_content.rfind(_SUMMARY_END_MARKER)
            db.archive_and_compact(session_id, messages)
            db.close()
            db = SessionDB(db_path=db_path)
            messages = db.get_messages_as_conversation(session_id)
    finally:
        db.close()

    # Only a replay after the LAST end marker is live: a carrier merged into a
    # newer summary's prior context is history, and a leftover flag is not content.
    from agent.context_compressor import (
        SUMMARY_PREFIX,
        _MERGED_PRIOR_CONTEXT_HEADER,
        _MERGED_SUMMARY_DELIMITER,
    )

    old = f"{SUMMARY_PREFIX}\nold\n\n{_SUMMARY_END_MARKER}\n\n{_INFLIGHT_TASK_REPLAY_HEADER}\ndo X"
    tail_merged = (
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{old}\n\n{_MERGED_SUMMARY_DELIMITER}\n\n"
        f"{SUMMARY_PREFIX}\nnew\n\n{_SUMMARY_END_MARKER}"
    )
    detect = ContextCompressor._has_merged_inflight_replay
    assert detect({"role": "user", "content": old})
    assert not detect({"role": "user", "content": tail_merged})
    assert not detect({"role": "user", "content": "hi", "_inflight_replay_merged": True})


@pytest.mark.parametrize(
    "payload, can_split",
    [
        ([{"type": "audio", "source": {"data": "AA=="}}], False),
        ([{"type": "text", "text": _ACTIVE_REQUEST}], True),
    ],
    ids=["audio", "text-parts"],
)
def test_split_requires_a_request_that_can_be_restated_as_text(payload, can_split):
    compressor = _make_compressor()
    messages = _oversized_active_turn()
    messages[3]["content"] = payload
    cut = compressor._find_tail_cut_by_tokens(messages, compressor._protect_head_size(messages))
    assert (cut > 3) is can_split
    with patch.object(compressor, "_generate_summary", return_value=None):
        compressed = compressor.compress(messages, current_tokens=90_000, force=True)
    if can_split:
        assert len(compressed) < len(messages)
        assert any(_ACTIVE_REQUEST in str(m.get("content")) for m in compressed)
    else:
        assert any(m.get("content") == payload for m in compressed)
    _assert_tool_pairs_are_complete(compressed)
