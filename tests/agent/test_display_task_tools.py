"""Tests for the Claude Agent SDK's TaskCreate/TaskUpdate/TaskGet/TaskList
display handling — friendly verbs, single-task previews (not counts), and
the compact progress-tree rendering.

Ported from Solar's live-validated WIP in agent/display.py (2026-08-20).
Deliberately NOT aliased onto native `todo_list`: these tools take one task per
call and carry no `todos` array, so routing them through todo_list's preview
logic would misrender e.g. a TaskCreate call as "reading task list".
"""

from agent.display import (
    TASK_TOOLS,
    build_tool_preview,
    get_cute_tool_message,
    get_tool_verb,
    verb_drops_preview,
)


class TestTaskToolsMembership:
    def test_all_four_task_tools_registered(self):
        assert TASK_TOOLS == {"TaskCreate", "TaskUpdate", "TaskGet", "TaskList"}


class TestTaskToolPreview:
    """build_tool_preview() shows the task itself, never a count."""

    def test_task_create_shows_subject(self):
        preview = build_tool_preview("TaskCreate", {"subject": "Port task labels to Main"})
        assert preview == "Port task labels to Main"

    def test_task_create_no_subject_returns_none(self):
        # Empty dict short-circuits earlier in build_tool_preview (no args
        # to preview at all) — confirm that path, not a TaskCreate-specific one.
        assert build_tool_preview("TaskCreate", {}) is None

    def test_task_update_shows_subject_and_status(self):
        preview = build_tool_preview(
            "TaskUpdate", {"taskId": "7", "subject": "Port task labels", "status": "in_progress"}
        )
        assert preview == "Port task labels → in_progress"

    def test_task_update_falls_back_to_id_without_subject(self):
        preview = build_tool_preview("TaskUpdate", {"taskId": "7", "status": "completed"})
        assert preview == "#7 → completed"

    def test_task_update_status_only_no_id_or_subject(self):
        preview = build_tool_preview("TaskUpdate", {"status": "completed"})
        assert preview == "→ completed"

    def test_task_get_shows_id(self):
        assert build_tool_preview("TaskGet", {"taskId": "3"}) == "#3"

    def test_task_get_no_id_returns_none(self):
        # No taskId and no other truthy arg -> build_tool_preview's leading
        # `if not args: return None` guard fires before reaching TaskGet's
        # own branch; still None either way.
        assert build_tool_preview("TaskGet", {"taskId": ""}) is None

    def test_task_list_returns_none_not_a_count(self):
        """The returned list is the content; there's nothing to preview from args."""
        assert build_tool_preview("TaskList", {"anything": "x"}) is None

    def test_task_create_preview_is_not_a_count(self):
        """Regression guard: must never render '1 task(s)' the way todo would."""
        preview = build_tool_preview("TaskCreate", {"subject": "Do the thing"})
        assert "task(s)" not in (preview or "")


class TestTaskToolVerbs:
    def test_task_create_verb(self):
        assert get_tool_verb("TaskCreate") == "Adding task"

    def test_task_update_verb(self):
        assert get_tool_verb("TaskUpdate") == "Updating task"

    def test_task_get_verb(self):
        assert get_tool_verb("TaskGet") == "Reading task"

    def test_task_list_verb(self):
        assert get_tool_verb("TaskList") == "Reading the task list"

    def test_task_list_drops_preview(self):
        """TaskList takes no meaningful args -- verb should stand alone."""
        assert verb_drops_preview("TaskList") is True

    def test_task_create_keeps_preview(self):
        assert verb_drops_preview("TaskCreate") is False


class TestTaskToolCuteMessage:
    """get_cute_tool_message()'s compact progress-tree rendering."""

    def test_task_create_message(self):
        msg = get_cute_tool_message("TaskCreate", {"subject": "Write tests"}, 0.4)
        assert "task" in msg
        assert "add" in msg
        assert "Write tests" in msg
        assert "0.4s" in msg

    def test_task_update_message(self):
        msg = get_cute_tool_message(
            "TaskUpdate", {"taskId": "7", "subject": "Write tests", "status": "completed"}, 0.2
        )
        assert "update" in msg
        assert "completed" in msg

    def test_task_get_message(self):
        msg = get_cute_tool_message("TaskGet", {"taskId": "9"}, 0.1)
        assert "read" in msg
        assert "#9" in msg

    def test_task_list_message_has_no_count(self):
        """Regression guard: must render 'list', never a task count."""
        msg = get_cute_tool_message("TaskList", {}, 0.3)
        assert "list" in msg
        assert "task(s)" not in msg

    def test_task_tools_use_clipboard_glyph_like_todo(self):
        msg = get_cute_tool_message("TaskCreate", {"subject": "x"}, 0.1)
        assert "📋" in msg
