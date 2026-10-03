"""Regression for #80449 — an oversized in-progress turn must stay compressible.

When one turn (opening user message + many individually small tool groups) grows past
the protected-tail soft ceiling, anchoring the cut back to the turn-opening request kept
the whole turn verbatim: compaction re-fired with an empty summarizable window and the
session sat over threshold. The cut must instead land on a tool-group-aligned mid-turn
boundary, and the active request must survive the handoff.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    _ACTIVE_TASK_MAX_CHARS,
    _INFLIGHT_TASK_REPLAY_HEADER,
    _SUMMARY_END_MARKER,
    _authored_request_text,
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


def _oversized_active_turn(request: Any = _ACTIVE_REQUEST, groups: int = 10) -> list[dict]:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older request completed"},
        {"role": "user", "content": request},
    ]
    for index in range(groups):
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


_LONG_QUOTE = "Earlier assistant answer. " * 80


def _gateway_reply(text: str, *, own: bool = False, discord_id: str | None = None) -> str:
    """Build the row exactly as the gateway does, so a pointer format change fails here."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    platform = Platform.DISCORD if discord_id else Platform.TELEGRAM
    source = SessionSource(platform=platform, chat_id="1", chat_type="dm")
    event = MessageEvent(
        text=text, source=source, message_id=discord_id,
        reply_to_message_id="7", reply_to_text=_LONG_QUOTE, reply_to_is_own_message=own,
    )
    with patch("gateway.session._discord_tools_loaded", return_value=True):
        return GatewayRunner._prepend_inbound_reply_context(event, source, text)


@pytest.mark.parametrize(
    "gateway_kwargs",
    [{}, {"own": True}, {"discord_id": "42"}],
    ids=["reply", "own-reply", "discord-note"],
)
def test_reply_pointer_does_not_count_toward_the_request_size(gateway_kwargs: dict[str, Any]) -> None:
    """A short reply to a long answer must still split: the quote is not the request.

    The gateway prepends ``[Replying to: "<quoted message>"]``; measured whole, a reply to a
    long assistant answer exceeded the active-task cap, the split never fired, and every
    automatic pass hit an empty window (structural backoff) while the turn kept growing.
    """
    request = _gateway_reply(_ACTIVE_REQUEST, **gateway_kwargs)
    assert len(request) > _ACTIVE_TASK_MAX_CHARS
    assert _authored_request_text(request) == _ACTIVE_REQUEST
    # Past the cap the deterministic snapshot must drop the quote, not elide the request away.
    snapshot = ContextCompressor._latest_user_task_snapshot([{"role": "user", "content": request}])
    assert _ACTIVE_REQUEST in (snapshot or "")
    compressor = _make_compressor()
    compressor.tail_token_budget = 1_000
    messages = _oversized_active_turn(request, 30)
    active_user_idx = 3

    cut = compressor._find_tail_cut_by_tokens(messages, compressor._protect_head_size(messages))
    assert cut > active_user_idx

    with patch.object(compressor, "_generate_summary", return_value=None):
        compressed = compressor.compress(messages, current_tokens=90_000)
    assert len(compressed) < len(messages)
    _assert_tool_pairs_are_complete(compressed)
    actionable = "\n".join(
        str(m.get("content")).rsplit(_SUMMARY_END_MARKER, 1)[-1]
        for m in compressed if m["role"] == "user"
    )
    assert actionable.count(_ACTIVE_REQUEST) == 1


def test_restated_reply_keeps_splitting_on_later_compactions() -> None:
    """After the first split the request is restated behind the replay header; the next
    passes must still measure the authored text, not the header plus the quote.

    ``protect_first_n=3`` (the default) keeps the restated row standalone, so the second
    pass anchors on it instead of on the summary carrier.
    """
    compressor = _make_compressor(protect_first_n=3)
    compressor.tail_token_budget = 1_000
    messages = _oversized_active_turn(_gateway_reply(_ACTIVE_REQUEST), 30)
    for cycle in range(3):
        with patch.object(compressor, "_generate_summary", return_value=None):
            compressed = compressor.compress(messages, current_tokens=90_000)
        # One cycle adds 30 tool groups (60 rows); a refused split reclaims only a handful.
        assert len(messages) - len(compressed) >= 30, f"cycle {cycle}: split refused"
        _assert_tool_pairs_are_complete(compressed)
        messages = list(compressed)
        for index in range(30):
            messages.extend(_tool_group(100 * (cycle + 1) + index))


def test_a_long_active_request_still_splits_and_survives_verbatim() -> None:
    """A long but token-bounded request (e.g. a /goal continuation prompt) must not pin the turn.

    The row-size guard is the token soft ceiling; a character cap on the request made every
    long-goal session uncompressible (empty window, structural backoff, context overflow).
    """
    compressor = _make_compressor()
    compressor.tail_token_budget = 1_000  # soft ceiling must hold the long request row itself
    long_request = " ".join(f"step-{i}" for i in range(400))
    assert len(long_request) > _ACTIVE_TASK_MAX_CHARS
    messages = _oversized_active_turn()
    messages[3]["content"] = long_request
    for index in range(10, 40):
        messages.extend(_tool_group(index))

    cut = compressor._find_tail_cut_by_tokens(messages, compressor._protect_head_size(messages))
    assert cut > 3

    with patch.object(compressor, "_generate_summary", return_value=None):
        compressed = compressor.compress(messages, current_tokens=90_000, force=True)
    assert len(compressed) < len(messages)
    _assert_tool_pairs_are_complete(compressed)
    live = "\n".join(
        str(m.get("content")).rsplit(_SUMMARY_END_MARKER, 1)[-1] for m in compressed if m["role"] == "user"
    )
    assert live.count(long_request) == 1


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


def _tail_group(index: int) -> list[dict]:
    """A small group so the region after the latest user turn stays under the soft ceiling."""
    call_id = f"tail_{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "inspect_shard", "arguments": "x" * 100},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": f"t-{index}:" + "r" * 50},
    ]


def _textless_oversized_turn() -> list[dict]:
    """#131412 shape: a completed older turn, then an oversized active turn with no text reply."""
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older request completed"},
    ]
    for index in range(10):
        messages.extend(_tool_group(index))
    # Newest text-bearing assistant reply: the older turn's closer.
    messages.append({"role": "assistant", "content": "older turn finished"})
    # The oversized active turn: a normal opening request, then tool groups with no
    # interleaved text reply (assistant rows carry only tool_calls).
    messages.append({"role": "user", "content": _ACTIVE_REQUEST})
    for index in range(10, 20):
        messages.extend(_tool_group(index))
    # A final short user nudge plus a small tail region: the latest user turn stays
    # inside the token-budget tail, so the #80449 user-anchor split does not fire.
    messages.append({"role": "user", "content": "keep going"})
    messages.extend(_tail_group(0))
    messages.extend(_tail_group(1))
    return messages


def test_assistant_anchor_cannot_retain_a_textless_oversized_turn(
    compressor: ContextCompressor,
) -> None:
    """The assistant anchor needs the same soft-ceiling bound as the user anchor (#131412).

    When the newest text-bearing assistant is the previous turn's closer, anchoring the
    cut to it retains the whole oversized active turn and the middle collapses to
    nothing — the session wedges in no_progress. The cut must keep a tool-group-aligned
    mid-turn boundary instead.
    """
    messages = _textless_oversized_turn()
    head_end = compressor._protect_head_size(messages)
    active_user_idx = next(
        index for index, message in enumerate(messages)
        if message.get("content") == _ACTIVE_REQUEST
    )

    cut = compressor._find_tail_cut_by_tokens(messages, head_end, token_budget=_TOKEN_BUDGET)

    latest_user_idx = next(
        index for index, message in enumerate(messages)
        if message.get("role") == "user" and message.get("content") == "keep going"
    )
    # The walk's tool-group-aligned cut: only the active turn's last two groups ride the
    # tail with the #10896-anchored nudge; the rest of the turn stays summarizable.
    assert active_user_idx < cut == latest_user_idx - 4
    _assert_tool_pairs_are_complete(messages[head_end:cut])
    _assert_tool_pairs_are_complete(messages[cut:])


def _reasoning_heavy_small_turn() -> list[dict]:
    """Under the ceiling on the wire, over it only if stale thinking were charged (#84371)."""
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older turn finished"},
    ]
    for index in range(2):
        group = _tail_group(index)
        group[0]["reasoning_content"] = "t" * 1200
        messages.extend(group)
    messages.append({"role": "user", "content": "keep going"})
    for index in range(2, 6):
        messages.extend(_tail_group(index))
    return messages


def _plain_text_oversized_region() -> list[dict]:
    """Over the ceiling, but in plain text rows with no tool-call bodies to summarize."""
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "older request"},
        {"role": "assistant", "content": "older turn finished"},
    ]
    messages.extend({"role": "user", "content": f"note-{index}:" + "n" * 600} for index in range(6))
    messages.append({"role": "user", "content": "keep going"})
    messages.extend(_tail_group(0) + _tail_group(1))
    return messages


@pytest.mark.parametrize(
    ("build", "allow_split_turn"),
    [
        # Rolling micro-compaction consumes complete exchanges only (allow_split_turn=False).
        (_textless_oversized_turn, False),
        # Stale thinking never reaches the wire on this route, so the escape must price the
        # region like the walk does (#84371) and not drop a reply that fits (#29824).
        (_reasoning_heavy_small_turn, True),
        # The escape exists to free tool-call bodies; an over-ceiling region of plain text
        # rows carries none, so the reply anchor still binds.
        (_plain_text_oversized_region, True),
    ],
)
def test_assistant_anchor_still_binds(
    compressor: ContextCompressor, build, allow_split_turn: bool,
) -> None:
    """The #131412 escape fires only for a wire-oversized tool region with splitting allowed."""
    messages = build()
    head_end = compressor._protect_head_size(messages)

    cut = compressor._find_tail_cut_by_tokens(
        messages, head_end, token_budget=_TOKEN_BUDGET, allow_split_turn=allow_split_turn,
    )

    older_closer_idx = next(
        index for index, message in enumerate(messages)
        if message.get("content") == "older turn finished"
    )
    # The anchor pulls the cut back to the older turn's closer, aligned before any tool group.
    assert cut == compressor._align_boundary_backward(messages, older_closer_idx)
