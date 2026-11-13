"""Compaction mid-way through a machinery wake turn must restate THAT wake.

A kanban/background-process/delegation notification is delivered to the agent
as a user-role row typed ``display_kind="internal_notification"`` (or another
machinery kind). When the agent's work on that wake grows past the context
threshold, compaction re-appends "the active request" after the handoff
(#100818). ``_find_inflight_user_task`` skipped every ``display_kind`` row, so
it walked past the live wake and restated an OLDER user row instead -- an
earlier human ask, or a stale notification for a different, already-finished
task. The model then resumed the wrong task (often answering ``[SILENT]`` to a
stale "nothing to do" notification) and the work it was actually doing was
dropped.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from agent.context_compressor import _SUMMARY_END_MARKER, SUMMARY_PREFIX, ContextCompressor

HUMAN_ASK = "HUMAN_ASK_sentinel_please_study_all_lessons"
OLD_WAKE = "[kanban] Task t_old completed. OLD_WAKE_sentinel"
LIVE_WAKE = "[kanban] Task t_live completed. LIVE_WAKE_sentinel"


def _tool_pairs(count: int, start: int = 0) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for i in range(start, start + count):
        turns.append({"role": "assistant", "content": "",
                      "tool_calls": [{"id": f"c{i}", "function": {"name": "read_file", "arguments": "{}"}}]})
        turns.append({"role": "tool", "tool_call_id": f"c{i}", "content": ("file text " * 200) + f" {i}"})
    return turns


def _wake(text: str, kind: str = "internal_notification") -> dict[str, Any]:
    return {"role": "user", "content": text, "display_kind": kind}


def _transcript(live_kind: str = "internal_notification") -> list[dict[str, Any]]:
    """Human ask -> answered; stale wake -> [SILENT]; live wake -> still working (tool turns, no reply)."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": HUMAN_ASK},
        {"role": "assistant", "content": "On it, I'll report back."},
        _wake(OLD_WAKE),
        {"role": "assistant", "content": "[SILENT]"},
        _wake(LIVE_WAKE, live_kind),
        *_tool_pairs(40),
    ]


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


def test_live_wake_is_the_inflight_task():
    picked = ContextCompressor._find_inflight_user_task(_transcript())
    assert picked is not None and LIVE_WAKE in _text(picked)


def test_every_machinery_kind_counts_as_the_live_task():
    for kind in ("internal_notification", "process_complete", "async_delegation_complete"):
        picked = ContextCompressor._find_inflight_user_task(_transcript(kind))
        assert picked is not None and LIVE_WAKE in _text(picked), kind


def test_answered_wake_is_not_replayed():
    messages = [*_transcript(), {"role": "assistant", "content": "Done -- summary sent to the user."}]
    assert ContextCompressor._find_inflight_user_task(messages) is None


def test_stale_wake_never_beats_a_newer_human_ask():
    """Only the NEWEST user row may be a machinery wake; an older notification stays excluded."""
    messages = [
        {"role": "system", "content": "You are Hermes."},
        _wake(OLD_WAKE),
        {"role": "assistant", "content": "[SILENT]"},
        {"role": "user", "content": HUMAN_ASK},
        *_tool_pairs(5),
    ]
    picked = ContextCompressor._find_inflight_user_task(messages)
    assert picked is not None and HUMAN_ASK in _text(picked)


def test_compress_restates_the_live_wake_not_an_older_row():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)
    compressor.tail_token_budget = 500
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Summary\nRead many files for the live task."
    with patch("agent.context_compressor.call_llm", return_value=response):
        compressed = compressor.compress(_transcript(), current_tokens=200_000, force=True)

    idx = max(i for i, m in enumerate(compressed)
              if SUMMARY_PREFIX[:60] in _text(m) or _SUMMARY_END_MARKER in _text(m))
    restated = _text(compressed[idx]).split(_SUMMARY_END_MARKER)[-1] + "".join(
        _text(m) for m in compressed[idx + 1:] if m.get("role") == "user")
    assert LIVE_WAKE in restated
    assert OLD_WAKE not in restated and HUMAN_ASK not in restated

def test_live_wake_survives_later_synthetic_todo_snapshot():
    """A post-compaction TODO snapshot is not a newer real user task."""
    messages = _transcript()
    messages.append({
        "role": "user",
        "content": "[ACTIVE TODOS] Continue the wake task.",
        "_todo_snapshot_synthetic": True,
    })
    messages.extend(_tool_pairs(1, start=40))
    picked = ContextCompressor._find_inflight_user_task(messages)
    assert picked is not None and LIVE_WAKE in _text(picked)
