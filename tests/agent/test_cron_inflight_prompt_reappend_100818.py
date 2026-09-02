"""Regression coverage for #100818: compaction of a single-prompt session
(the cron shape) must not leave the model with nothing to obey.

A cron run is one user message — the job prompt — followed by nothing but
assistant/tool turns. When ContextCompressor fires mid-run, that prompt is
folded into the handoff summary and no user message survives *after* it.
SUMMARY_PREFIX then reads literally:

    If no user message appears AFTER this summary, do nothing.

so the model correctly does nothing, the scheduler sees the ``[SILENT]``
sentinel, and records ``last_status: ok`` — a silent failure.

The fix re-appends the in-flight user task after the handoff so the prefix's
"latest user message" pointer resolves to the job prompt again. The
#80622 contract is unchanged: an idle session with no in-flight task must
still be left with nothing to act on.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    _SUMMARY_END_MARKER,
    SUMMARY_PREFIX,
    ContextCompressor,
)


JOB_SENTINEL = "CRON_JOB_PROMPT_sentinel_brief_the_inbox_and_write_a_digest"


def _make_compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        compressor = ContextCompressor(
            model="test",
            quiet_mode=True,
            protect_first_n=2,
            protect_last_n=2,
        )
    compressor.tail_token_budget = 500
    return compressor


def _tool_pairs(count: int, start: int = 0) -> List[Dict[str, Any]]:
    """``count`` assistant(tool_calls) + tool result pairs."""
    turns: List[Dict[str, Any]] = []
    for i in range(start, start + count):
        turns.append(
            {
                "role": "assistant",
                "content": f"step {i}",
                "tool_calls": [
                    {"id": f"c{i}", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            }
        )
        turns.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": ("tool output " * 200) + f" {i}",
            }
        )
    return turns


def _cron_transcript() -> List[Dict[str, Any]]:
    """system + one user job prompt + many tool turns, NO trailing user."""
    return [
        {
            "role": "system",
            "content": "You are Hermes. Cron preamble: if nothing to report, "
            "return [SILENT].",
        },
        {"role": "user", "content": JOB_SENTINEL},
        *_tool_pairs(40),
    ]


def _compress(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = (
        "## Historical Task Snapshot\nUser asked: '" + JOB_SENTINEL + "'\n"
        "## Summary\nRan a bunch of terminal steps."
    )
    compressor = _make_compressor()
    with patch("agent.context_compressor.call_llm", return_value=response):
        return compressor.compress(messages, current_tokens=200_000, force=True)


def _handoff_idx(compressed: List[Dict[str, Any]]) -> int:
    """Index of the handoff row (standalone summary or merged carrier)."""
    for idx in range(len(compressed) - 1, -1, -1):
        content = compressed[idx].get("content")
        text = content if isinstance(content, str) else str(content)
        if SUMMARY_PREFIX[:60] in text or _SUMMARY_END_MARKER in text:
            return idx
    return -1


def _text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


def _actionable_user_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        m
        for m in rows
        if ContextCompressor._is_actionable_user_turn(m)
        and not ContextCompressor._is_synthetic_compression_user_turn(m)
    ]


def test_cron_job_prompt_survives_after_the_handoff():
    """The in-flight job prompt must be readable AFTER the summary boundary."""
    compressed = _compress(_cron_transcript())

    idx = _handoff_idx(compressed)
    assert idx >= 0, "expected a compaction handoff in the compressed transcript"

    after = compressed[idx + 1:]
    # The re-append may also land inside the handoff carrier itself, after the
    # end marker (the alternation-safe merge layout) — accept either shape.
    carrier_tail = _text(compressed[idx]).split(_SUMMARY_END_MARKER)[-1]

    job_after_summary = any(
        JOB_SENTINEL in _text(m) for m in _actionable_user_rows(after)
    ) or (JOB_SENTINEL in carrier_tail)
    assert job_after_summary, (
        "the in-flight cron job prompt must appear in a user message AFTER the "
        "handoff summary — SUMMARY_PREFIX orders the model to do nothing "
        "otherwise (#100818)"
    )


def test_model_is_not_left_without_a_user_message_after_the_handoff():
    """The 'no user message after this summary → do nothing' branch of
    SUMMARY_PREFIX must not be what a mid-run cron compaction produces."""
    compressed = _compress(_cron_transcript())

    idx = _handoff_idx(compressed)
    after = compressed[idx + 1:]
    has_user_after = bool(_actionable_user_rows(after)) or bool(
        _text(compressed[idx]).split(_SUMMARY_END_MARKER)[-1].strip()
    )
    assert has_user_after, (
        "compaction left no user message after the handoff; the model is "
        "instructed to do nothing and the cron run fails silently"
    )


def test_role_alternation_and_head_are_preserved():
    """The re-append must not create two same-role rows in a row, and must
    not disturb the cached head prefix."""
    messages = _cron_transcript()
    compressed = _compress([dict(m) for m in messages])

    assert compressed[0]["role"] == "system"
    visible = [
        m.get("role")
        for m in compressed
        if not (
            m.get("role") == "tool"
            or (m.get("role") == "assistant" and m.get("tool_calls"))
        )
    ]
    for previous, current in zip(visible, visible[1:]):
        assert not (previous == current == "user"), (
            f"consecutive user rows in compressed transcript: {visible}"
        )


def test_idle_session_without_inflight_task_is_not_reanimated():
    """#80622 must hold: a session whose only user-role row is an inherited
    handoff has no in-flight task, so compaction must not manufacture one."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are Hermes."},
        {
            "role": "user",
            "content": (
                SUMMARY_PREFIX
                + "\n## Historical Task Snapshot\nUser asked: 'a finished task'\n\n"
                + _SUMMARY_END_MARKER
            ),
        },
        *_tool_pairs(40),
    ]
    compressed = _compress(messages)

    assert not _actionable_user_rows(compressed), (
        "no real user turn existed before compaction — none may be invented"
    )


def test_completed_exchange_is_not_replayed():
    """Only an in-flight task is re-appended. A turn that already produced a
    final assistant reply must not be handed back to the model as a fresh
    instruction."""
    messages = [
        *_cron_transcript(),
        {"role": "assistant", "content": "Digest written. Nothing else to do."},
    ]
    compressed = _compress(messages)

    idx = _handoff_idx(compressed)
    after = compressed[idx + 1:]
    assert not any(
        JOB_SENTINEL in _text(m) for m in _actionable_user_rows(after)
    ), "a completed exchange must not be re-appended as a new user instruction"


# ---------------------------------------------------------------------------
# Follow-up coverage (salvage of #101170)
# ---------------------------------------------------------------------------


def _pending_tail_transcript() -> List[Dict[str, Any]]:
    """Cron shape whose LAST row is an assistant tool_calls turn still awaiting
    its result — compaction fired inside the tool-execution window."""
    msgs = _cron_transcript()
    msgs.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "pending", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        }
    )
    return msgs


def test_pending_trailing_tool_call_survives_the_replay():
    """The replay row must not make a genuinely in-flight tool_call look
    orphaned: _sanitize_tool_pairs runs before the re-append, so the trailing
    assistant(tool_calls) keeps its calls and the late tool result still pairs."""
    out = _compress(_pending_tail_transcript())
    pending = [
        m
        for m in out
        if m.get("role") == "assistant"
        and any(tc.get("id") == "pending" for tc in (m.get("tool_calls") or []))
    ]
    assert pending, "trailing in-flight tool_calls were stripped"
    replay_idx = max(
        i for i, m in enumerate(out)
        if m.get("role") == "user" and JOB_SENTINEL in str(m.get("content"))
    )
    assert replay_idx > out.index(pending[0])


def _compress_with(protect_first_n: int, compression_count: int, messages):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = SUMMARY_PREFIX + "\n## Summary\nran steps."
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        compressor = ContextCompressor(
            model="test",
            quiet_mode=True,
            protect_first_n=protect_first_n,
            protect_last_n=2,
        )
    compressor.tail_token_budget = 500
    compressor.compression_count = compression_count
    with patch.object(
        compressor, "_generate_summary", return_value=response.choices[0].message.content
    ):
        return compressor.compress(messages, current_tokens=90_000)


def _job_copies(messages) -> int:
    return sum(
        str(m.get("content")).count(JOB_SENTINEL)
        for m in messages
        if m.get("role") == "user"
    )


def test_merged_restatement_is_not_anchored_twice():
    """Later cycle: protect_first_n has decayed, the prompt is summarised away
    and the summary is pinned to role=user followed by an exempt tool tail, so
    the restatement is MERGED onto the carrier. The later
    _ensure_compressed_has_user_turn pass must see intent as present instead
    of inserting a second copy of the job prompt."""
    from agent.conversation_compression import _ensure_compressed_has_user_turn

    original = [{"role": "user", "content": JOB_SENTINEL}, *_tool_pairs(40)]
    out = _compress_with(2, 1, original)
    assert any(m.get("_inflight_replay_merged") for m in out), "expected merge layout"
    assert _job_copies(out) == 1
    assert _ensure_compressed_has_user_turn(original, out) == "already_present"
    assert _job_copies(out) == 1


def test_restatement_survives_repeated_compactions_without_stacking():
    """A task alive across three compactions is restated exactly once per
    output — one copy, one header, always after the summary."""
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER

    out = [{"role": "user", "content": JOB_SENTINEL}, *_tool_pairs(40)]
    for cycle in (1, 2, 3):
        extra = _tool_pairs(40, start=100 * cycle) if cycle > 1 else []
        out = _compress_with(2, cycle, out + extra)
        users = [m for m in out if m.get("role") == "user"]
        assert _job_copies(out) == 1, cycle
        assert max(
            str(m.get("content")).count(_INFLIGHT_TASK_REPLAY_HEADER) for m in users
        ) == 1, cycle
        last = str(users[-1].get("content"))
        assert last.rfind(JOB_SENTINEL) > last.rfind(_SUMMARY_END_MARKER), cycle


def test_flagged_scaffolding_row_is_never_the_inflight_task():
    """A trailing user-role scaffolding row flagged synthetic (todo snapshot)
    must not be mistaken for the live request and replayed as an instruction."""
    msgs = _cron_transcript()
    msgs.append(
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]\n- x",
            "_todo_snapshot_synthetic": True,
        }
    )
    found = ContextCompressor._find_inflight_user_task(msgs)
    assert found is not None
    assert JOB_SENTINEL in str(found.get("content"))
