"""Tests for live subagent (delegate_task) progress encoding (M12a).

build_subagent_progress_update turns a list_active_subagents() snapshot into a
delegate-batch tool_call_update the VS Code FleetModel parses. The subagent_id
"sa-<task_index>-<uuid>" encodes the index; the FleetModel keys children by
K = index + 1, so we assert the emitted "Task K" lines line up.
"""
import time

import acp_adapter.events as ev
from acp_adapter.events import build_subagent_progress_update


def _text_of(update):
    # ACP tool_call_update.content[0] is a text content block.
    block = update.content[0]
    # acp.tool_content(acp.text_block(...)) — drill to the text.
    for attr in ("content", "text"):
        block = getattr(block, attr, block)
        if isinstance(block, str):
            return block
    return str(update.content)


def test_none_when_no_indexed_subagents():
    assert build_subagent_progress_update("tc1", []) is None
    assert build_subagent_progress_update("tc1", None) is None
    # ids without the sa-<index>- prefix are skipped → nothing to report
    assert build_subagent_progress_update("tc1", [{"subagent_id": "weird", "status": "running"}]) is None


def test_running_subagents_emit_task_lines_keyed_by_index_plus_one():
    snap = [
        {"subagent_id": "sa-0-abc", "status": "running", "tool_count": 3},
        {"subagent_id": "sa-1-def", "status": "running", "tool_count": 0},
    ]
    u = build_subagent_progress_update("tcX", snap)
    assert u is not None
    assert u.session_update == "tool_call_update"
    assert u.tool_call_id == "tcX"
    assert u.kind == "execute"
    assert u.status == "in_progress"
    text = _text_of(u)
    assert text.startswith("Delegation progress:")
    # index 0 -> Task 1, index 1 -> Task 2
    assert "🔄 Task 1: running (3 tools)" in text
    assert "🔄 Task 2: running (0 tools)" in text


def test_done_and_failed_glyphs():
    snap = [
        {"subagent_id": "sa-0-a", "status": "completed", "tool_count": 5},
        {"subagent_id": "sa-1-b", "status": "failed", "tool_count": 2},
    ]
    text = _text_of(build_subagent_progress_update("tc", snap))
    assert "✅ Task 1: completed (5 tools)" in text
    assert "❌ Task 2: failed (2 tools)" in text


def test_rows_sorted_by_task_number():
    snap = [
        {"subagent_id": "sa-2-c", "status": "running", "tool_count": 1},
        {"subagent_id": "sa-0-a", "status": "running", "tool_count": 1},
    ]
    text = _text_of(build_subagent_progress_update("tc", snap))
    body = text.split("\n")[1:]
    assert body[0].startswith("🔄 Task 1"), body
    assert body[1].startswith("🔄 Task 3"), body


def test_missing_tool_count_omits_suffix():
    snap = [{"subagent_id": "sa-0-a", "status": "running"}]
    text = _text_of(build_subagent_progress_update("tc", snap))
    assert "Task 1: running" in text
    assert "tools)" not in text


def test_poll_thread_emits_progress_then_stops(monkeypatch):
    """Deterministically verify the poll glue: the daemon thread reads the
    registry, emits a progress frame via _send_update, and stops on demand."""
    import tools.delegate_tool as dt

    sent = []
    monkeypatch.setattr(ev, "_send_update", lambda conn, sid, loop, update: sent.append(update))
    monkeypatch.setattr(ev, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(
        dt, "list_active_subagents",
        lambda: [{"subagent_id": "sa-0-x", "status": "running", "tool_count": 2}],
    )

    polls = {}
    ev._start_delegate_poll(object(), "sess", object(), "tcZ", polls)
    assert "tcZ" in polls  # poll registered
    time.sleep(0.15)  # let it emit (identical frames are deduped to one)
    ev._stop_delegate_poll("tcZ", polls)
    time.sleep(0.1)
    assert "tcZ" not in polls  # stopped + cleaned up
    assert len(sent) >= 1
    assert sent[0].tool_call_id == "tcZ"
    assert sent[0].session_update == "tool_call_update"
    n_after_stop = len(sent)
    time.sleep(0.1)
    assert len(sent) == n_after_stop  # no frames emitted after stop


def test_poll_is_noop_without_a_polls_dict():
    # Backward-compat: callers that don't pass delegate_polls get no poll.
    ev._start_delegate_poll(object(), "s", object(), "tc", None)  # must not raise
