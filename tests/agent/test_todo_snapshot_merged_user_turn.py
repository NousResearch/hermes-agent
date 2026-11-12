"""A user turn merged after a folded todo snapshot survives the next compaction boundary (#137652).

The pre-call repair (``_merge_consecutive_users``) can join the next user message onto a row that already
carries a snapshot: after a boundary folds the snapshot into the trailing user row and the turn ends with
no assistant reply (an API error, ``/stop``), or when a standalone snapshot row is followed by the next
prompt. The stale-snapshot strip used to cut from the header to the end of the content, deleting that
merged message. A snapshot block now ends with ``TODO_INJECTION_FOOTER``, so the strip removes only the
block.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_runtime_helpers import _merge_consecutive_users
from agent.conversation_compression import _fold_todo_snapshot, _strip_stale_todo_snapshot
from tools.todo_tool import TODO_INJECTION_FOOTER, TODO_INJECTION_HEADER, TodoStore


def _agent(*items):
    store = TodoStore()
    store.write([{"id": str(i), "content": c, "status": "in_progress"} for i, c in enumerate(items, 1)])
    return SimpleNamespace(_todo_store=store, _repair_message_sequence=lambda rows: None)


def _boundary_then_merge(agent, rows, followup):
    _fold_todo_snapshot(agent, rows)
    rows.append({"role": "user", "content": followup})
    rows, repairs = _merge_consecutive_users(rows)
    assert repairs == 1
    return rows


def test_issue_repro_merged_user_turn_survives_the_next_boundary():
    agent = _agent("ship the release")
    rows = _boundary_then_merge(agent, [{"role": "user", "content": "first request"}],
                                "SECOND MESSAGE: please also update the docs")
    _fold_todo_snapshot(agent, rows)

    content = rows[-1]["content"]
    assert content.startswith("first request\n\nSECOND MESSAGE: please also update the docs\n\n")
    # The stale block is replaced by the live one, never stacked.
    assert content.count(TODO_INJECTION_HEADER) == 1
    assert content.count(TODO_INJECTION_FOOTER) == 1
    assert content.endswith(TODO_INJECTION_FOOTER)


def test_prompt_merged_onto_a_standalone_snapshot_row_survives():
    """The snapshot comes first: a synthetic snapshot row absorbed the next prompt."""
    agent = _agent("task")
    rows = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "done"}]
    rows = _boundary_then_merge(agent, rows, "NEXT PROMPT")
    assert rows[-1].get("_todo_snapshot_synthetic") is True  # the merge kept the flag

    _fold_todo_snapshot(agent, rows)

    assert [row["role"] for row in rows] == ["user", "assistant", "user"]
    assert rows[-1]["content"].startswith("NEXT PROMPT\n\n" + TODO_INJECTION_HEADER)
    # Real user text now: no longer todo-only scaffolding.
    assert "_todo_snapshot_synthetic" not in rows[-1]


def test_multiline_todo_content_is_stripped_whole():
    """A blank line inside a todo item does not end the block."""
    agent = _agent("step one\n\nstep two (pending)\nstep three")
    rows = _boundary_then_merge(agent, [{"role": "user", "content": "first"}], "second")
    _fold_todo_snapshot(agent, rows)

    content = rows[-1]["content"]
    assert content.startswith("first\n\nsecond\n\n" + TODO_INJECTION_HEADER)
    assert content.count("step three") == 1


def test_skill_reload_notice_strips_with_its_block():
    notice = "[Skills pruned during compression — reload before acting on these tasks]\n- skill_view(name='x')"
    text = f"first\n\n{TODO_INJECTION_HEADER}\n- [>] 1. task (in_progress)\n\n{notice}\n{TODO_INJECTION_FOOTER}\n\nsecond"
    assert _strip_stale_todo_snapshot(text) == "first\n\nsecond"


def test_block_without_footer_still_strips_to_the_end():
    """Rows written before the footer existed keep the old strip."""
    text = f"first\n\n{TODO_INJECTION_HEADER}\n- [>] 1. task (in_progress)"
    assert _strip_stale_todo_snapshot(text) == "first"
    assert _strip_stale_todo_snapshot(f"{TODO_INJECTION_HEADER}\n- [ ] 1. task (pending)") == ""


def test_structured_text_part_keeps_text_after_the_block():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    block = f"{TODO_INJECTION_HEADER}\n- [>] 1. task (in_progress)\n{TODO_INJECTION_FOOTER}"
    content = [image, {"type": "text", "text": f"look\n\n{block}\n\nand this"}, {"type": "text", "text": block}]
    assert _strip_stale_todo_snapshot(content) == [image, {"type": "text", "text": "look\n\nand this"}]
