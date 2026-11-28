"""Snapshot rendering and fail-open dispatch for #31001."""
import json
import logging
import queue

import pytest

from gateway.todo_progress import TodoChecklist, _dispatch_todo_progress, _render_todo_checklist


@pytest.mark.parametrize("status,glyph", [("pending", "[ ]"), ("in_progress", "[>]"),
                                         ("completed", "[x]"), ("cancelled", "[-]"), ("other", "[ ]")])
def test_status_and_id(status, glyph):
    result = _render_todo_checklist(json.dumps({"todos": [{"content": "Work", "id": "t1", "status": status}]}))
    assert f"{glyph} Work (t1)" in result


@pytest.mark.parametrize("result", [None, "", "not json", [], {}, {"todos": None},
                                    {"todos": [None]}, {"todos": [{"content": 1}]}])
def test_invalid_snapshot_falls_through(result):
    q = queue.Queue()
    assert not _dispatch_todo_progress(q, "tool.completed", "todo", result, logging.getLogger())
    assert q.empty()


def test_empty_list_clears_previous_tasks():
    assert _render_todo_checklist({"todos": []}) == "📋 Task List\nNo tasks."


def test_compact_single_line_content():
    result = _render_todo_checklist({"todos": [{"content": "A" * 120 + "\nExtra", "id": "t" * 100}]})
    assert "A" * 80 in result and "A" * 81 not in result
    assert len(result.splitlines()) == 2
    assert "t" * 33 not in result


@pytest.mark.parametrize("name", ["todo", "todo_list"])
def test_dispatch_preserves_snapshot_identity(name):
    q = queue.Queue()
    assert _dispatch_todo_progress(q, "tool.completed", name, {"todos": []}, logging.getLogger())
    assert isinstance(q.get_nowait(), TodoChecklist)


@pytest.mark.parametrize("event,name", [("tool.started", "todo"), ("tool.completed", "terminal")])
def test_unrelated_event_falls_through(event, name):
    q = queue.Queue()
    assert not _dispatch_todo_progress(q, event, name, {"todos": []}, logging.getLogger())
    assert q.empty()
