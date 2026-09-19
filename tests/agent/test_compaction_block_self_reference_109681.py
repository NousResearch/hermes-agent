"""Regression coverage for #109681: the assembled compaction block must not
re-embed its own previous content.

Observed on a 544-message session: once the context reached the compaction
threshold, ``[CONTEXT COMPACTION]`` fired every turn and the assembled block
grew to 51,710 chars, then stalled at *exactly* that size across 4 consecutive
compactions — a stable fixed point, not a transient. Each pass re-embedded:

1. the same "Historical Task Snapshot", grounded on the previous pass's own
   ``[STILL IN PROGRESS …]`` restatement instead of the real user ask,
2. the previous pass's restatement verbatim, quoted again as a "real user
   message" in the lean ``## User Messages (verbatim, newest first)`` section,
3. the mechanically appended sections, none of which were re-derived.

The summary-token ceiling governed the *generated summary* only, never the
assembled block, so the block could never drop back below the size that
triggered compaction.

These tests drive real-length samples through consecutive compactions and pin
three properties:

(a) the assembled block never exceeds its end-to-end ceiling;
(b) no previous pass's restatement is re-embedded into the block — neither as
    the Historical Task Snapshot ground nor as a verbatim user message;
(c) a block that sits over its ceiling without shrinking registers as a fixed
    point, so the stall is identifiable instead of silent.
"""

import re
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    SUMMARY_PREFIX,
    ContextCompressor,
    _INFLIGHT_TASK_REPLAY_HEADER,
    _LEAN_ANCHOR_HEADING,
    _LEAN_USER_MESSAGES_HEADING,
    _SUMMARY_END_MARKER,
    _build_verbatim_user_section,
)

TASK = "fix the flaky compaction path in agent/context_compressor.py and ship it"
HISTORICAL_TASK_HEADING = "## Historical Task Snapshot"


def _mk_compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
        )
    compressor.tail_token_budget = 500
    # A 100K-context model's summary allowance (5% of context, capped at 10K tokens) — pinned here
    # because the lazy property would otherwise re-resolve the unpatched 256K fallback.
    compressor.max_summary_tokens = 5_000
    return compressor


def _tool_pairs(count: int, start: int = 0, output_chars: int = 5_000) -> List[Dict[str, Any]]:
    """Assistant(tool_calls) + tool result pairs with real-length tool output."""
    turns: List[Dict[str, Any]] = []
    for i in range(start, start + count):
        turns.append({
            "role": "assistant",
            "content": f"step {i}: checked the block builder, running the suite next.",
            "tool_calls": [{"id": f"c{i}", "function": {"name": "terminal", "arguments": "{}"}}],
        })
        turns.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": (f"line {i}: assembled block measured at {i * 137} chars; "
                        "context_compressor.py:3140 still appends the section. ") * (output_chars // 110),
        })
    return turns


def _long_session() -> List[Dict[str, Any]]:
    """system + one real user ask + a long tool transcript (the incident shape)."""
    return [
        {"role": "system", "content": "You are Hermes. Keep working until the task is done."},
        {"role": "user", "content": TASK},
        *_tool_pairs(80),
    ]


def _saturated_summary_body() -> str:
    """The block a saturating summarizer preserves: a ~21K narrative over the ceiling, plus the
    stale mechanically appended sections an earlier pass left in it (the #109681 re-embedding).

    The in-flight action list is kept to a couple of lines on purpose: ``_ground_historical_task_snapshot``
    rewrites that section from the compacted window on the next pass, so a bulkier one would come back
    under the ceiling on its own and there would be no stall left to detect in (c).
    """
    actions = "\n".join(
        f"{i}. READ agent/context_compressor.py:{100 + i} — inspected the handoff block builder "
        "[tool: read_file]"
        for i in range(1, 141)
    )
    log = "\n".join(
        f"- compaction pass {i}: assembled block measured, sections re-derived from the window"
        for i in range(1, 71)
    )
    stale_users = (
        f"{_LEAN_USER_MESSAGES_HEADING}\n"
        f"> {_INFLIGHT_TASK_REPLAY_HEADER}\n"
        f"> {TASK}\n"
        "(Every real user message from the compacted region, quoted verbatim. These are the user's "
        "actual words and override any paraphrase of them above.)\n"
    )
    stale_anchors = (
        f"{_LEAN_ANCHOR_HEADING}\n"
        "PRs/issues: #109681(x4), #100818, #86366(x2)\n"
        "commits: 9939e3375e2, d131988d53c\n"
        "files: agent/context_compressor.py(x6), tests/agent/test_context_compressor.py\n"
        "(Exact identifiers from the compacted region — use these verbatim.)\n"
    )
    return (
        f"{HISTORICAL_TASK_HEADING}\n"
        + "In-flight action list: inspected the block builder, measured the assembled block, "
        "re-ran the suite, next commit the fix. " * 2 + "\n\n"
        "## Goal\nShip the self-reference fix for the assembled compaction block.\n\n"
        "## Constraints & Preferences\nDo not change what compaction drops; keep the handoff contract.\n\n"
        f"## Completed Actions\n{actions}\n\n"
        "## Active State\nworktree D:/code/wt-109681 on branch fix/109681-descr; suite green.\n\n"
        "## Blocked\nNone. The block keeps re-embedding its own previous content.\n\n"
        "## Key Decisions\nTrim the mechanically re-derivable sections first.\n\n"
        "## Resolved Questions\nNone\n\n"
        "## Relevant Files\nagent/context_compressor.py\n\n"
        "## Critical Context\nThe summary ceiling governs the generated summary, not the assembled block.\n\n"
        f"## Detailed Session Log (oldest first)\n{log}\n\n"
        f"{stale_users}\n{stale_anchors}"
    )


def _summary_body_from_prompt(prompt: str) -> str:
    match = re.search(r"PREVIOUS SUMMARY:\n(.*?)\n\nNEW TURNS TO INCORPORATE:", prompt, re.S)
    return match.group(1).strip() if match else ""


def _saturating_llm(first_body: str):
    """A summarizer at saturation: PRESERVE the previous summary, add nothing new."""

    def _call(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        body = _summary_body_from_prompt(prompt) or first_body
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = body
        return response

    return _call


def _carrier(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith(SUMMARY_PREFIX):
            return content
    raise AssertionError("no compaction handoff carrier in the transcript")


def _handoff_body(carrier: str) -> str:
    """The assembled block proper: prefix stripped, merged live ask excluded."""
    text = carrier[len(SUMMARY_PREFIX):] if carrier.startswith(SUMMARY_PREFIX) else carrier
    marker = text.find(_SUMMARY_END_MARKER)
    return text[:marker].rstrip() if marker >= 0 else text.rstrip()


def _section(body: str, heading: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(heading)}\n(.*?)(?=^## |\Z)", body)
    return match.group(1) if match else ""


def test_task_snapshot_and_verbatim_section_skip_a_previous_restatement():
    """The block's own restatement is scaffolding, not the user's ask."""
    compressor = _mk_compressor()
    restatement = {
        "role": "user",
        "content": _INFLIGHT_TASK_REPLAY_HEADER + "\n" + TASK,
    }
    messages = [
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": "on it"},
        restatement,
    ]

    snapshot = compressor._latest_user_task_snapshot(messages)
    assert snapshot is not None and TASK in snapshot
    assert _INFLIGHT_TASK_REPLAY_HEADER not in snapshot

    section = _build_verbatim_user_section(messages)
    assert TASK in section
    assert _INFLIGHT_TASK_REPLAY_HEADER not in section

    # A transcript whose only user row is the restatement grounds nothing: the
    # block must not cite its own previous output as human intent.
    assert compressor._latest_user_task_snapshot([restatement]) is None


def test_stale_lean_sections_are_re_derived_not_re_embedded():
    """A section an earlier pass left in the block is rebuilt from this window, keeping one copy."""
    compressor = _mk_compressor()
    turns = [{"role": "user", "content": TASK}, {"role": "assistant", "content": "on it"}]
    stale_users = (
        f"{_LEAN_USER_MESSAGES_HEADING}\n"
        "> words from an earlier window\n"
        "(Every real user message from the compacted region, quoted verbatim. These are the user's "
        "actual words and override any paraphrase of them above.)\n"
    )
    summary = (
        f"{HISTORICAL_TASK_HEADING}\nUser asked: stale example\n\n"
        f"## Goal\nship the fix\n\n{stale_users}"
    )

    out = compressor._augment_summary_lean(summary, turns)

    assert out.count(_LEAN_USER_MESSAGES_HEADING) == 1
    section = _section(out, _LEAN_USER_MESSAGES_HEADING)
    assert TASK in section, "the re-derived section must quote this window's real ask"
    assert "words from an earlier window" not in section, "the stale copy must not ride along"


def test_consecutive_compactions_do_not_reembed_the_previous_block():
    """Real-length samples, 4 passes: ceiling holds, no self-reference, stall visible."""
    compressor = _mk_compressor()
    cap = compressor._handoff_block_cap_chars()
    assert cap == 20_000  # 100K-ctx model: summary allowance (5K tokens) at 4 chars/token

    transcript = _long_session()
    bodies: List[str] = []
    with patch("agent.context_compressor.call_llm", side_effect=_saturating_llm(_saturated_summary_body())):
        for pass_no in range(1, 5):
            if pass_no > 1:
                transcript = transcript + _tool_pairs(2, start=1_000 + pass_no * 10)
            transcript = compressor.compress(transcript, current_tokens=200_000, force=True)
            bodies.append(_handoff_body(_carrier(transcript)))

    # (a) End-to-end ceiling: the assembled block, not just the generated summary.
    for index, body in enumerate(bodies, start=1):
        assert len(body) <= cap, f"pass {index}: block {len(body)} chars over the {cap}-char ceiling"

    # (b) The block never re-embeds its own previous restatement.
    for index, body in enumerate(bodies, start=1):
        assert _INFLIGHT_TASK_REPLAY_HEADER not in _section(body, HISTORICAL_TASK_HEADING), (
            f"pass {index}: the Historical Task Snapshot cites the block's own restatement"
        )
        assert _INFLIGHT_TASK_REPLAY_HEADER not in _section(body, _LEAN_USER_MESSAGES_HEADING), (
            f"pass {index}: the verbatim user section re-quotes the block's own restatement"
        )

    # The in-flight ask itself stays actionable after the boundary (#100818).
    replay_rows = [
        str(message.get("content"))
        for message in transcript
        if _INFLIGHT_TASK_REPLAY_HEADER in str(message.get("content"))
    ]
    assert replay_rows, "the in-flight ask must stay actionable after the boundary (#100818)"
    assert TASK in replay_rows[-1].split(_INFLIGHT_TASK_REPLAY_HEADER, 1)[1]

    # (c) The stall is identifiable: over the ceiling and not shrinking => fixed point.
    assert getattr(compressor, "_last_handoff_block_fixed_point", False) is True
    assert compressor._last_handoff_block_len <= cap
