"""Regression tests: async delegation results survive context compaction.

The failure mode: a background ``delegate_task`` result lands as a synthetic
user-role row, the agent keeps working after it (fixing issues, answering the
user), and by the next compaction the token-budget tail walk pushes the result
row into the summarised middle.  The summariser rolls it up; the agent's next
context no longer contains the verdict and it concludes "the subagent result
hasn't landed" and re-does the delegated work.

The tail anchor added in ``_find_tail_cut_by_tokens``
(``_ensure_last_delegation_completions_in_tail``) pulls the cut back to the
most recent delegation completion row(s) so they survive verbatim.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
    ContextCompressor,
)


@pytest.fixture()
def compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    instance.tail_token_budget = 10
    return instance


def _append_tool_run(messages: list[dict], prefix: str, count: int = 6) -> None:
    for index in range(count):
        call_id = f"{prefix}-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "x" * 400,
                },
            ]
        )


def _compress(compressor: ContextCompressor, messages: list[dict]) -> list[dict]:
    with patch.object(
        compressor,
        "_generate_summary",
        return_value=f"{SUMMARY_PREFIX}\nsummary of older work",
    ):
        return compressor.compress(messages, current_tokens=90_000)


def _completion_rows(result: list[dict], completion: str) -> list[dict]:
    return [message for message in result if message.get("content") == completion]


# ---------------------------------------------------------------------------
# Core regression: the result is NOT the last user message, yet must survive
# ---------------------------------------------------------------------------


def test_delegation_completion_survives_when_not_last_user_message(compressor):
    completion = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_ab12cd]\n"
        "The independent reviewer verdict: FAIL with 3 critical issues."
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
    ]
    # Older middle turns that the token budget (10) could never protect.
    messages += [
        {"role": "user", "content": f"older question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"older reply {index}"}
        for index in range(6)
    ]
    # The result lands...
    messages.append({"role": "user", "content": completion})
    # ...the agent keeps working after it...
    messages.append({"role": "assistant", "content": "the reviewer is right, fixing"})
    _append_tool_run(messages, "work")
    # ...and the user keeps talking, so the completion is NOT the last user row.
    messages.append({"role": "user", "content": "what did the reviewer find?"})
    messages.append({"role": "assistant", "content": "three critical issues, fixing now"})

    result = _compress(compressor, messages)

    rows = _completion_rows(result, completion)
    assert len(rows) == 1
    assert not rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)
    # The middle was genuinely summarised (otherwise the test proves nothing).
    summary_rows = [
        message for message in result if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(summary_rows) == 1
    # The work done AFTER the result arrived must also survive.
    assert result[-1].get("content") == "three critical issues, fixing now"


def test_single_delegation_completion_survives_verbatim(compressor):
    completion = (
        "[ASYNC DELEGATION COMPLETE — deleg_ef34gh]\n"
        "Subagent finished; full task source below."
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
        {"role": "user", "content": "older work 1"},
        {"role": "assistant", "content": "older reply 1"},
        {"role": "user", "content": "older work 2"},
        {"role": "assistant", "content": "older reply 2"},
        {"role": "user", "content": completion},
        {"role": "assistant", "content": "acting on the result"},
        {"role": "user", "content": "later question"},
        {"role": "assistant", "content": "later reply"},
    ]

    result = _compress(compressor, messages)

    rows = _completion_rows(result, completion)
    assert len(rows) == 1
    assert not rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)


def test_recompression_keeps_delegation_completion_verbatim(compressor):
    """A second compression pass must not summarise the anchored result."""
    completion = (
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_ij56kl]\n"
        "Fan-out of 2 subagents finished; consolidated results below."
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
    ]
    # Older middle turns that the tiny token budget could never protect.
    messages += [
        {"role": "user", "content": f"older question {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"older reply {index}"}
        for index in range(6)
    ]
    # The result lands deep in the transcript...
    messages.append({"role": "user", "content": completion})
    messages.append({"role": "assistant", "content": "working from the completion"})
    _append_tool_run(messages, "tail")
    messages += [
        {"role": "user", "content": "next question"},
        {"role": "assistant", "content": "next reply"},
    ]

    first = _compress(compressor, messages)
    rows = _completion_rows(first, completion)
    assert len(rows) == 1
    assert not rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)

    second = _compress(compressor, first)
    rows = _completion_rows(second, completion)
    assert len(rows) == 1
    assert not rows[0].get(COMPRESSED_SUMMARY_METADATA_KEY)


# ---------------------------------------------------------------------------
# Marker recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "[ASYNC DELEGATION COMPLETE — deleg_aa11]\nresult",
        "[ASYNC DELEGATION COMPLETE - deleg_aa11]\nresult",  # ASCII hyphen
        "[ASYNC DELEGATION BATCH COMPLETE — deleg_bb22]\nresult",
        "[ASYNC DELEGATION BATCH COMPLETE - deleg_bb22]\nresult",
        "[ASYNC DELEGATION COMPLETE — deleg_cc33]",
    ],
)
def test_delegation_completion_markers_recognized(compressor, content):
    assert compressor._is_delegation_completion_message(
        {"role": "user", "content": content}
    )


def test_display_kind_marker_recognized_without_header(compressor):
    assert compressor._is_delegation_completion_message(
        {
            "role": "user",
            "content": "opaque delegation context",
            "display_kind": "async_delegation_complete",
        }
    )


def test_assistant_quote_of_header_is_not_a_completion_row(compressor):
    # An assistant reply that merely quotes the header must not be anchored.
    assert not compressor._is_delegation_completion_message(
        {
            "role": "assistant",
            "content": '[ASYNC DELEGATION COMPLETE — deleg_dd44] said FAIL, fixing',
        }
    )


def test_plain_user_message_is_not_a_completion_row(compressor):
    assert not compressor._is_delegation_completion_message(
        {"role": "user", "content": "can you review the plugin?"}
    )


# ---------------------------------------------------------------------------
# Anchor behaviour
# ---------------------------------------------------------------------------


def test_anchor_is_noop_without_delegation_completions(compressor):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "reply"},
    ]
    cut = compressor._find_tail_cut_by_tokens(messages, head_end=1)
    assert compressor._ensure_last_delegation_completions_in_tail(
        messages, cut, head_end=1
    ) == cut


def test_anchor_is_noop_when_completion_already_in_tail(compressor):
    completion = "[ASYNC DELEGATION COMPLETE — deleg_ee55]\nresult"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": completion},
        {"role": "assistant", "content": "acting"},
    ]
    cut = len(messages) - 2  # tail starts AT the completion
    assert compressor._ensure_last_delegation_completions_in_tail(
        messages, cut, head_end=1
    ) == cut


def test_anchor_pulls_cut_back_to_earliest_of_recent_completions(compressor):
    first = "[ASYNC DELEGATION COMPLETE — deleg_ff66]\noldest result"
    second = "[ASYNC DELEGATION BATCH COMPLETE — deleg_gg77]\nnewer result"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": first},
        {"role": "assistant", "content": "handled oldest"},
        {"role": "user", "content": second},
        {"role": "assistant", "content": "handled newest"},
    ]
    cut = len(messages)  # tail empty — both completions in the middle region
    new_cut = compressor._ensure_last_delegation_completions_in_tail(
        messages, cut, head_end=1
    )
    # Both completions are anchored; cut lands at the earliest one.
    assert new_cut == 3
    assert messages[new_cut].get("content") == first


def test_anchor_respects_max_tail_delegation_completions(compressor):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    for index in range(4):
        messages.append(
            {
                "role": "user",
                "content": f"[ASYNC DELEGATION COMPLETE — deleg_0{index}]\nresult {index}",
            }
        )
        messages.append({"role": "assistant", "content": f"handled {index}"})

    compressor.max_tail_delegation_completions = 2
    cut = len(messages)
    new_cut = compressor._ensure_last_delegation_completions_in_tail(
        messages, cut, head_end=1
    )
    # Completions sit at indices 3, 5, 7, 9 (deleg_00..deleg_03).  Walking
    # backward from the end, the 2 most recent are 9 and 7, so the cut lands
    # at 7 — the OLDEST anchored one — NOT all the way back at 3.
    assert new_cut == 7
    assert "[ASYNC DELEGATION COMPLETE — deleg_02]" in messages[new_cut]["content"]


# ---------------------------------------------------------------------------
# No-bridge edge: summary must merge INTO the completion row WITHOUT burying
# the verdict under the [PRIOR CONTEXT] header
# ---------------------------------------------------------------------------


def test_no_bridge_merge_keeps_verdict_after_summary_boundary(compressor):
    completion = (
        "[ASYNC DELEGATION COMPLETE — deleg_hh88]\n"
        "The verdict: FAIL with 3 critical issues."
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": "initial reply"},
        # One compressible middle turn — NO assistant bridge before the
        # completion, so the summary must merge INTO the completion row.
        {"role": "user", "content": "middle work"},
        {"role": "user", "content": completion},
        {"role": "assistant", "content": "acting on the result"},
        {"role": "user", "content": "later question"},
        {"role": "assistant", "content": "later reply"},
    ]

    result = _compress(compressor, messages)

    merged_rows = [
        message
        for message in result
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(merged_rows) == 1
    merged_content = merged_rows[0].get("content", "")
    # The verdict text survives intact, placed AFTER the summary boundary so
    # the model treats it as the actionable message to respond to...
    assert completion in merged_content
    assert _SUMMARY_END_MARKER in merged_content
    assert merged_content.index(completion) > merged_content.index(_SUMMARY_END_MARKER)
    # ...and it is NOT buried under the reference-only PRIOR CONTEXT header.
    assert "[PRIOR CONTEXT" not in merged_content
    # Alternation is preserved (no adjacent user rows in the output).
    for previous, current in zip(result, result[1:]):
        assert (previous.get("role"), current.get("role")) != ("user", "user")
