"""Display glue for tools that exist only on the Claude Agent SDK runtime.

The SDK harness exposes per-task CRUD tools (``TaskCreate``/``TaskUpdate``/``TaskGet``/``TaskList``)
and its own ``ToolSearch``; none has a native Hermes counterpart, so :mod:`agent.tool_identity`
deliberately does not alias them. This sibling owns their previews, verbs and completion lines;
:mod:`agent.display` merges the tables below into its own, so the facade carries no SDK-specific
bodies. Helpers are imported lazily from ``agent.display`` to avoid an import cycle.
"""
from __future__ import annotations

# Per-task CRUD tools exposed by the Claude Agent SDK harness.
#
# Deliberately NOT aliased to native ``todo_list`` in :mod:`agent.tool_identity`,
# which maps only tools whose semantics match.  ``todo_list`` rewrites the whole
# list from a ``todos`` array; these operate on one task per call and carry no
# such array, so the alias would route ``TaskCreate`` down the
# ``todos_arg is None`` branch and render "reading task list" for a call that
# creates one.  Wrong is worse than absent.
TASK_TOOLS: frozenset[str] = frozenset({
    "TaskCreate", "TaskUpdate", "TaskGet", "TaskList",
})


def _task_fields(args: dict) -> tuple[str, str, str]:
    """``(subject, task_id, status)`` of a per-task tool call, each ``""`` when absent."""
    from agent.display import _oneline

    return (
        _oneline(str(args.get("subject") or "")).strip(),
        str(args.get("taskId") or "").strip(),
        str(args.get("status") or "").strip(),
    )


# Previews for the per-task tools show the task itself, not a count: ``todo_list``
# renders "planning 3 task(s)" because one call really does carry three, while
# these take one task per call, so a count would read "1 task(s)" every time --
# true, and useless.  The subject is what tells the user which task moved.
def _preview_task_create(args: dict, max_len: int) -> str | None:
    from agent.display import _truncate_preview

    subject = _task_fields(args)[0]
    return _truncate_preview(subject, max_len) if subject else None


def _preview_task_update(args: dict, max_len: int) -> str | None:
    from agent.display import _truncate_preview

    subject, task_id, status = _task_fields(args)
    head = subject or (f"#{task_id}" if task_id else "")
    preview = " ".join(p for p in (head, f"→ {status}" if status else "") if p)
    return _truncate_preview(preview, max_len) if preview else None


def _preview_task_get(args: dict, max_len: int) -> str | None:
    from agent.display import _truncate_preview

    task_id = _task_fields(args)[1]
    return _truncate_preview(f"#{task_id}", max_len) if task_id else None


# Merged into agent.display._PREVIEW_BUILDERS.
SDK_PREVIEW_BUILDERS = {
    "TaskCreate": _preview_task_create, "TaskUpdate": _preview_task_update,
    "TaskGet": _preview_task_get,
    "TaskList": lambda _args, _limit: None,  # no meaningful arguments; the returned list is the content
}

# Merged into agent.display._TOOL_VERBS. The SDK harness splits native ``todo_list`` into per-task
# calls, so each gets its own verb rather than borrowing "Updating tasks" -- reading the list and
# adding one differ. ``ToolSearch`` is not a Hermes tool and is deliberately absent from the
# identity map (no native counterpart), but users see it constantly on runtimes that defer tool
# schemas, so it still earns a verb.
SDK_TOOL_VERBS: dict[str, str] = {
    "TaskCreate": "Adding task", "TaskUpdate": "Updating task",
    "TaskGet": "Reading task", "TaskList": "Reading the task list",
    "ToolSearch": "Loading tools",
}
# ``TaskList`` takes no arguments: no preview appended.
SDK_TOOL_VERBS_NO_PREVIEW: frozenset[str] = frozenset({"TaskList"})
# ``ToolSearch`` reads as search-style phrasing ("Loading tools for ...").
SDK_TOOL_VERBS_FOR_CONNECTOR: frozenset[str] = frozenset({"ToolSearch"})

_CUTE_TASK_VERBS = {"TaskCreate": "add", "TaskUpdate": "update", "TaskGet": "read", "TaskList": "list"}


def _cute_task(tool_name: str, args: dict) -> str:
    from agent.display import _cute_trunc

    detail = _cute_trunc(SDK_PREVIEW_BUILDERS[tool_name](args, 0) or "")
    return f"┊ 📋 task      {f'{_CUTE_TASK_VERBS[tool_name]} {detail}'.rstrip()}"


# Merged into agent.display._CUTE_LINES: tool -> f(args, result) -> completion line.
SDK_CUTE_LINES = {
    "TaskCreate": lambda a, _r: _cute_task("TaskCreate", a),
    "TaskUpdate": lambda a, _r: _cute_task("TaskUpdate", a),
    "TaskGet": lambda a, _r: _cute_task("TaskGet", a),
    "TaskList": lambda a, _r: _cute_task("TaskList", a),
}
