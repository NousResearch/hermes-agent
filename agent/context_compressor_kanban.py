"""Keep the newest bounded Kanban assignment through tool-result compression."""
import json
import os


def _assignment(message, calls):
    name, arguments = calls.get(message.get("tool_call_id", ""), (None, ""))
    if message.get("role") != "tool" or name != "kanban_show":
        return None
    try:
        payload = json.loads(message.get("content", ""))
        arguments = json.loads(arguments)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(arguments, dict):
        return None
    task = payload.get("task")
    worker_task_id = os.environ.get("HERMES_KANBAN_TASK")
    task_id = arguments.get("task_id") or worker_task_id
    if worker_task_id != task_id:
        return None
    if not isinstance(task, dict) or not isinstance(task_id, str) or not task_id or len(task_id) > 128 or task.get("id") != task_id:
        return None
    if not isinstance(task.get("title"), str) or not isinstance(task.get("body"), str):
        return None
    return task_id, payload


def _bounded(text, budget):
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= budget:
        return encoded.decode("utf-8")
    suffix = "\n<truncated>"
    return encoded[:budget - len(suffix)].decode("utf-8", errors="ignore") + suffix


def newest_assignment_summary(messages, index, calls):
    current = _assignment(messages[index], calls)
    if current is None:
        return None
    task_id, payload = current
    if any((later := _assignment(message, calls)) is not None and later[0] == task_id
           for message in messages[index + 1:]):
        return None
    task = payload["task"]
    projected = {"id": task_id, "title": _bounded(task["title"], 1024),
                 "body": _bounded(task["body"], 8 * 1024)}
    for key in ("status", "workspace_access"):
        if isinstance(task.get(key), str):
            projected[key] = _bounded(task[key], 128)
    summary = {"task": projected}
    spec = payload.get("protected_task_spec")
    # Preserve an existing producer contract; never promote ordinary task text
    # into a protected egress grant or bring back historical board material.
    if (isinstance(spec, dict) and spec.get("version") == "v1"
            and isinstance(spec.get("title"), str) and isinstance(spec.get("body"), str)):
        summary["protected_task_spec"] = {
            "version": "v1", "title": _bounded(spec["title"], 1024),
            "body": _bounded(spec["body"], 8 * 1024),
        }
    return json.dumps(summary, ensure_ascii=False)
