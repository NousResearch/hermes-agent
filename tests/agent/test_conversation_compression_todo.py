"""A user turn merged after a todo snapshot survives the next compaction boundary (#137652).

``_fold_todo_snapshot`` appends the live todo list to the trailing user row. If that turn ends without an
assistant reply (API error, ``/stop``), the next user message makes a ``user;user`` run, and the pre-call
repair ``_merge_consecutive_users`` joins it onto the row that already carries the snapshot. The stale-snapshot
strip at the next boundary used to cut from ``TODO_INJECTION_HEADER`` to the end of the content, deleting
that later user message. Only the snapshot block (header, task lines, optional reload notice) is stale.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_runtime_helpers import _merge_consecutive_users
from agent.conversation_compression import (
    _PRUNED_SKILL_RELOAD_NOTICE_HEADER,
    _fold_todo_snapshot,
    _strip_stale_todo_snapshot,
)
from tools.todo_tool import TODO_INJECTION_HEADER, TodoStore

SECOND = "SECOND MESSAGE: please also update the docs"
NOTICE = _PRUNED_SKILL_RELOAD_NOTICE_HEADER + "\nreload skill_view(name='old-skill') first."


def _item_line(marker: str, item_id: str, content: str, status: str, depth: int = 0) -> str:
    """Match TodoStore.format_for_injection length-prefixed id+content rendering."""
    return (
        f"{'  ' * depth}- {marker} #{len(item_id)}:{item_id}. "
        f"#{len(content)}:{content} ({status})"
    )


def _agent(*todos: dict) -> SimpleNamespace:
    store = TodoStore()
    store.write(list(todos))
    return SimpleNamespace(_todo_store=store, _repair_message_sequence=lambda rows: None)


def test_issue_repro_merged_user_turn_survives_second_boundary():
    agent = _agent({"id": "1", "content": "ship the release", "status": "in_progress"})
    rows = [{"role": "user", "content": "first request"}]
    _fold_todo_snapshot(agent, rows)
    rows.append({"role": "user", "content": SECOND})
    rows, repairs = _merge_consecutive_users(rows)
    assert repairs == 1 and len(rows) == 1

    _fold_todo_snapshot(agent, rows)

    content = rows[-1]["content"]
    assert content.startswith("first request\n\n" + SECOND + "\n\n" + TODO_INJECTION_HEADER)
    assert content.count(TODO_INJECTION_HEADER) == 1
    assert content.endswith(_item_line("[>]", "1", "ship the release", "in_progress"))


def test_refreshed_snapshot_replaces_stale_tasks_without_dropping_text():
    agent = _agent({"id": "1", "content": "old task", "status": "in_progress"})
    rows = [{"role": "user", "content": "first request"}]
    _fold_todo_snapshot(agent, rows)
    rows.append({"role": "user", "content": SECOND})
    rows, _ = _merge_consecutive_users(rows)
    agent._todo_store.write([{"id": "2", "content": "new task", "status": "pending"}])

    _fold_todo_snapshot(agent, rows)

    content = rows[-1]["content"]
    assert SECOND in content and "first request" in content
    assert "old task" not in content and "new task" in content
    assert content.count(TODO_INJECTION_HEADER) == 1


def test_strip_keeps_text_after_snapshot_and_reload_notice():
    tasks = "\n".join([
        _item_line("[ ]", "t1", "old", "pending"),
        _item_line("[>]", "t2", "b", "in_progress"),
    ])
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{tasks}\n\n{NOTICE}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == f"first request\n\n{SECOND}"


def test_strip_keeps_text_after_snapshot_without_notice_and_nested_tasks():
    tasks = "\n".join([
        _item_line("[>]", "p", "parent", "in_progress"),
        _item_line("[ ]", "c", "child", "pending", depth=1),
    ])
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{tasks}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == f"first request\n\n{SECOND}"


def test_strip_keeps_text_when_snapshot_leads_the_row():
    content = f"{TODO_INJECTION_HEADER}\n{_item_line('[ ]', 't1', 'old', 'pending')}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == SECOND


def test_strip_removes_every_stacked_snapshot_but_keeps_each_turn():
    block = f"{TODO_INJECTION_HEADER}\n{_item_line('[ ]', 't1', 'old', 'pending')}"
    content = f"a\n\n{block}\n\nb\n\n{block}\n\n{NOTICE}\n\nc"
    assert _strip_stale_todo_snapshot(content) == "a\n\nb\n\nc"


def test_trailing_snapshot_and_notice_still_strip_to_prefix():
    content = (
        f"real user words\n\n{TODO_INJECTION_HEADER}\n"
        f"{_item_line('[ ]', 't1', 'old', 'pending')}\n\n{NOTICE}"
    )
    assert _strip_stale_todo_snapshot(content) == "real user words"
    assert _strip_stale_todo_snapshot(
        f"{TODO_INJECTION_HEADER}\n{_item_line('[ ]', 't1', 'old', 'pending')}"
    ) == ""


def test_list_part_keeps_text_after_snapshot():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    text = (
        f"words\n\n{TODO_INJECTION_HEADER}\n"
        f"{_item_line('[ ]', 't1', 'old', 'pending')}\n\n{SECOND}"
    )
    content = [{"type": "text", "text": text}, image]
    assert _strip_stale_todo_snapshot(content) == [{"type": "text", "text": f"words\n\n{SECOND}"}, image]


MULTILINE = "prepare release notes\n\ninclude migration steps"
# Status suffix embedded in content must not end the item early (review #4257241880).
SUFFIX_IN_CONTENT = "prepare pending section (pending)\ninclude migration steps"


def test_multiline_todo_keeps_one_copy_across_two_folds_and_merged_turn():
    agent = _agent({"id": "1", "content": MULTILINE, "status": "pending"})
    rows = [{"role": "user", "content": "first request"}]
    _fold_todo_snapshot(agent, rows)
    rows.append({"role": "user", "content": SECOND})
    rows, _ = _merge_consecutive_users(rows)

    _fold_todo_snapshot(agent, rows)
    _fold_todo_snapshot(agent, rows)

    content = rows[-1]["content"]
    assert content == (
        f"first request\n\n{SECOND}\n\n{TODO_INJECTION_HEADER}\n"
        f"{_item_line('[ ]', '1', MULTILINE, 'pending')}"
    )
    assert content.count("include migration steps") == 1


def test_status_suffix_inside_content_keeps_one_copy_across_two_folds():
    """Content ending a mid-item line with `` (pending)`` must not early-cut the strip."""
    agent = _agent({"id": "1", "content": SUFFIX_IN_CONTENT, "status": "pending"})
    rows = [{"role": "user", "content": "first request"}]
    _fold_todo_snapshot(agent, rows)
    rows.append({"role": "user", "content": SECOND})
    rows, _ = _merge_consecutive_users(rows)

    _fold_todo_snapshot(agent, rows)
    _fold_todo_snapshot(agent, rows)

    content = rows[-1]["content"]
    assert content == (
        f"first request\n\n{SECOND}\n\n{TODO_INJECTION_HEADER}\n"
        f"{_item_line('[ ]', '1', SUFFIX_IN_CONTENT, 'pending')}"
    )
    assert content.count(TODO_INJECTION_HEADER) == 1
    assert content.count("include migration steps") == 1
    # Must not leave the second content line as ordinary user text before the live snapshot.
    before_header, _, after_header = content.partition(TODO_INJECTION_HEADER)
    assert "include migration steps" not in before_header
    assert "include migration steps" in after_header


def test_strip_status_suffix_inside_content_keeps_following_user_turn():
    rendered = _item_line("[ ]", "1", SUFFIX_IN_CONTENT, "pending")
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{rendered}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == f"first request\n\n{SECOND}"
    assert "include migration steps" not in _strip_stale_todo_snapshot(content)


def test_strip_multiline_nested_tasks_and_notice_keep_later_turn():
    tasks = "\n".join([
        _item_line("[>]", "p", "parent\n\nmore", "in_progress"),
        _item_line("[ ]", "c", "child\n\nx", "pending", depth=1),
    ])
    content = f"a\n\n{TODO_INJECTION_HEADER}\n{tasks}\n\n{NOTICE}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == f"a\n\n{SECOND}"

def test_legacy_unprefixed_snapshot_stripped_when_store_completed():
    """F1: pre-#N: snapshots must still clean up (review #4258260151)."""
    agent = _agent({"id": "t1", "content": "old task", "status": "completed"})
    rows = [{
        "role": "user",
        "content": (
            "human request\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t1. old task (pending)"
        ),
    }]
    _fold_todo_snapshot(agent, rows)
    assert rows[0]["content"] == "human request"


def test_legacy_unprefixed_snapshot_keeps_later_user_turn():
    legacy = "- [ ] t1. old task (pending)"
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{legacy}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == f"first request\n\n{SECOND}"


# Old writer emitted raw multiline content; a content line may end in a status-looking suffix
# (review comment 6179098126). The item must run to the last matching suffix, not the first.
LEGACY_STATUS_IN_CONTENT = "- [ ] t1. step one\nlog says (pending)\nstep three (pending)"


def test_legacy_status_looking_content_line_does_not_truncate_item():
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{LEGACY_STATUS_IN_CONTENT}"
    stripped = _strip_stale_todo_snapshot(content)
    assert stripped == "first request"
    assert "step three" not in stripped


def test_legacy_status_looking_content_line_keeps_later_user_turn():
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{LEGACY_STATUS_IN_CONTENT}\n\n{SECOND}"
    stripped = _strip_stale_todo_snapshot(content)
    assert stripped == f"first request\n\n{SECOND}"
    assert "step three" not in stripped and "log says" not in stripped


def test_legacy_status_looking_content_line_with_next_item_and_notice():
    tasks = f"{LEGACY_STATUS_IN_CONTENT}\n- [>] t2. other (in_progress)"
    content = f"first request\n\n{TODO_INJECTION_HEADER}\n{tasks}\n\n{NOTICE}\n\n{SECOND}"
    assert _strip_stale_todo_snapshot(content) == f"first request\n\n{SECOND}"


def test_legacy_stacked_snapshots_keep_turn_between_them():
    block = f"{TODO_INJECTION_HEADER}\n{LEGACY_STATUS_IN_CONTENT}"
    content = f"a\n\n{block}\n\nb\n\n{block}\n\nc"
    assert _strip_stale_todo_snapshot(content) == "a\n\nb\n\nc"


def test_legacy_synthetic_only_row_clears_when_store_completed():
    agent = _agent({"id": "t1", "content": "old task", "status": "completed"})
    rows = [{
        "role": "user",
        "content": f"{TODO_INJECTION_HEADER}\n- [ ] t1. old task (pending)",
        "_todo_snapshot_synthetic": True,
    }]
    _fold_todo_snapshot(agent, rows)
    assert rows == []


def test_whitespace_id_keeps_one_copy_across_repeated_folds():
    """F2: IDs with spaces must round-trip (review #4258260170)."""
    agent = _agent({"id": "release notes", "content": "ship the release", "status": "pending"})
    rows = [{"role": "user", "content": "human request"}]
    for _ in range(3):
        _fold_todo_snapshot(agent, rows)
    content = rows[0]["content"]
    assert content.count("ship the release") == 1
    assert content.count(TODO_INJECTION_HEADER) == 1
    assert content == (
        f"human request\n\n{TODO_INJECTION_HEADER}\n"
        f"{_item_line('[ ]', 'release notes', 'ship the release', 'pending')}"
    )


def test_tab_and_newline_ids_round_trip_nested_and_complete():
    tab_id = "release\tnotes"
    nl_id = "release\nnotes"
    agent = _agent(
        {"id": tab_id, "content": "parent work", "status": "in_progress"},
        {"id": nl_id, "content": "child work", "status": "pending", "parent": tab_id},
    )
    rows = [{"role": "user", "content": "human request"}]
    _fold_todo_snapshot(agent, rows)
    _fold_todo_snapshot(agent, rows)
    content = rows[0]["content"]
    assert content.count("parent work") == 1
    assert content.count("child work") == 1
    assert _item_line("[>]", tab_id, "parent work", "in_progress") in content
    assert _item_line("[ ]", nl_id, "child work", "pending", depth=1) in content

    agent._todo_store.write([
        {"id": tab_id, "content": "parent work", "status": "completed"},
        {"id": nl_id, "content": "child work", "status": "completed", "parent": tab_id},
    ])
    _fold_todo_snapshot(agent, rows)
    assert rows[0]["content"] == "human request"
