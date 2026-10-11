"""/goal session-state fold for the compaction boundary (#133643).

The todo list is re-folded into the transcript on every compaction by ``_fold_todo_snapshot``; the
goal and its completion contract live in ``state_meta``, not the transcript, so without an
equivalent fold a mid-turn compaction keeps them only inside the summary's lossy free-text
``## Goal`` section and the Verification/Constraints/Stop-condition fields the judge enforces are
routinely paraphrased away for the rest of the turn.

Kept in its own topical module (not the ``conversation_compression`` facade) so the facade stays at
its line cap.
"""

from __future__ import annotations

from typing import Any

# Header for the goal block re-folded at a compaction boundary. Deliberately distinct from the
# between-turn continuation prompt and non-imperative: this block reminds the model what its standing
# goal and completion contract ARE, not to keep working (the judge decides continuation).
GOAL_PRESERVED_HEADER = "[Your standing /goal was preserved across context compression]"
GOAL_SNAPSHOT_FLAG = "_goal_snapshot_synthetic"
# The between-turn continuation prompt's opening line: a row that already carries it already carries
# the goal, so the fold must not add a duplicate block.
GOAL_CONTINUATION_PREFIX = "[Continuing toward your standing goal]"


def is_goal_snapshot_row(message: Any) -> bool:
    """A user row the goal fold injected, or a real user row a goal block was folded into."""
    from agent.conversation_compression import _message_text

    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if message.get(GOAL_SNAPSHOT_FLAG):
        return True
    return GOAL_PRESERVED_HEADER in _message_text(message)


def strip_stale_goal_snapshot(content: Any) -> Any:
    """Remove a previously folded goal block from message content.

    The block is always folded LAST into a row (after the todo snapshot), so it runs to the end of the
    string / its own trailing text part — unlike the todo strip, which truncates at its header."""
    if isinstance(content, str):
        idx = content.find(GOAL_PRESERVED_HEADER)
        return content[:idx].rstrip() if idx != -1 else content
    if isinstance(content, list):
        cleaned = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text") or "")
                idx = text.find(GOAL_PRESERVED_HEADER)
                if idx != -1:
                    if stripped := text[:idx].rstrip():
                        cleaned.append({**part, "text": stripped})
                    continue
            cleaned.append(part)
        return cleaned
    return content


def _goal_snapshot_is_only_content(content: Any, stripped: Any) -> bool:
    """Whether stripping the goal block leaves no structured content (the row was scaffolding alone)."""
    if isinstance(content, str) and isinstance(stripped, str):
        return not stripped.strip()
    if isinstance(content, list) and isinstance(stripped, list):
        return not stripped
    return False


def fold_goal_snapshot(agent: Any, compressed: list) -> None:
    """Strip earlier preserved-goal snapshots and fold the live active /goal block in (in place).

    Mirror of ``_fold_todo_snapshot`` for ``/goal``. Runs after the todo fold, so the goal block is
    appended into whatever trailing user row the todo fold produced (real, or the todo fold's own
    synthetic row) and two synthetic user rows never sit adjacent.
    """
    from agent.context_compressor import _append_text_to_content
    from agent.conversation_compression import _message_text, _replace_message_content
    from hermes_cli.goals import load_goal, render_preserved_goal_block

    state = load_goal(getattr(agent, "session_id", "") or "")
    if state is None:
        # No goal row, or the store is unavailable: we cannot prove an existing snapshot stale, so
        # preserve it rather than risk deleting the only copy (mirrors the un-rehydrated todo store).
        return
    goal_block = render_preserved_goal_block(state)
    # The stored goal is authoritative now: drop any earlier snapshot so repeated compactions refresh
    # rather than stack it, and a finished/paused goal stops reading as a standing one (#26981, #34197).
    for idx in range(len(compressed) - 1, -1, -1):
        row = compressed[idx]
        if not is_goal_snapshot_row(row):
            continue
        stripped = strip_stale_goal_snapshot(row.get("content"))
        if stripped == row.get("content"):
            continue
        if row.get(GOAL_SNAPSHOT_FLAG) and _goal_snapshot_is_only_content(row.get("content"), stripped):
            compressed.pop(idx)
            if idx < len(compressed):
                # Deleting a standalone row can expose two assistant rows; reuse the replay repair.
                agent._repair_message_sequence(compressed)
        else:
            _replace_message_content(row, stripped)
            # A real user row that merely carried a block keeps its real provenance after the strip.
            row.pop(GOAL_SNAPSHOT_FLAG, None)
        break
    if not goal_block:
        return
    tail = compressed[-1] if compressed and isinstance(compressed[-1], dict) else None
    if tail is not None and tail.get("role") == "user":
        if GOAL_CONTINUATION_PREFIX in _message_text(tail):
            return  # continuation-opened turn already carries the goal verbatim
        base = tail.get("content")
        separator = "\n\n" if isinstance(base, str) and base.strip() else ""
        _replace_message_content(tail, _append_text_to_content(base, f"{separator}{goal_block}"))
    else:
        compressed.append({"role": "user", "content": goal_block, GOAL_SNAPSHOT_FLAG: True})


def fold_state_snapshots(agent: Any, compressed: list) -> None:
    """Re-fold the live todo list and the active /goal into *compressed* (in place).

    The single entry point the compaction commit boundary calls, in place of the bare todo fold.
    """
    from agent.conversation_compression import _fold_todo_snapshot

    _fold_todo_snapshot(agent, compressed)
    fold_goal_snapshot(agent, compressed)