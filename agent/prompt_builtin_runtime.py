"""Turn-local metadata for prompt-backed built-ins such as ``/learn``.

Prompt built-ins remain ordinary user turns.  This module carries their origin beside the
turn, collects authoritative tool outcomes without parsing assistant prose, and emits one
additive completion hook after the whole turn settles.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional


_PENDING_KEY = "_pending_prompt_builtin_runs"
_COMMAND_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


@dataclass
class PromptBuiltinRun:
    origin: dict[str, str]
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_current_run: ContextVar[Optional[PromptBuiltinRun]] = ContextVar(
    "prompt_builtin_run", default=None
)


def make_prompt_builtin_origin(command: str, raw_args: str = "") -> dict[str, str]:
    """Return the stable, JSON-safe origin carried by a prompt-backed command turn."""
    return {
        "name": str(command or "").strip().lstrip("/"),
        "raw_args": str(raw_args or ""),
        "run_id": uuid.uuid4().hex,
    }


def normalize_prompt_builtin_origin(origin: Any) -> Optional[dict[str, str]]:
    """Validate and normalize turn provenance before it enters the runtime context."""
    if not isinstance(origin, Mapping):
        return None
    name = origin.get("name")
    raw_args = origin.get("raw_args", "")
    run_id = origin.get("run_id")
    if not isinstance(name, str) or not isinstance(raw_args, str):
        return None
    name = name.strip().lstrip("/")
    if not _COMMAND_RE.fullmatch(name):
        return None
    if run_id in (None, ""):
        run_id = uuid.uuid4().hex
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        return None
    return {"name": name, "raw_args": raw_args, "run_id": run_id}


def stage_prompt_builtin(
    owner: MutableMapping[str, Any] | Any,
    message: str,
    origin: Mapping[str, Any],
) -> None:
    """Associate an origin with an exact queued prompt without changing model-visible text."""
    record = {"message": message, "origin": dict(origin)}
    if isinstance(owner, MutableMapping):
        owner.setdefault(_PENDING_KEY, []).append(record)
        return
    pending = getattr(owner, _PENDING_KEY, None)
    if not isinstance(pending, list):
        pending = []
        setattr(owner, _PENDING_KEY, pending)
    pending.append(record)


def take_prompt_builtin(owner: MutableMapping[str, Any] | Any, message: Any) -> Optional[dict[str, Any]]:
    """Consume the oldest origin whose exact queued prompt reached the turn boundary."""
    pending = owner.get(_PENDING_KEY) if isinstance(owner, MutableMapping) else getattr(owner, _PENDING_KEY, None)
    if not isinstance(pending, list) or not isinstance(message, str):
        return None
    for index, record in enumerate(pending):
        if isinstance(record, dict) and record.get("message") == message:
            pending.pop(index)
            if not pending:
                if isinstance(owner, MutableMapping):
                    owner.pop(_PENDING_KEY, None)
                else:
                    delattr(owner, _PENDING_KEY)
            origin = record.get("origin")
            return dict(origin) if isinstance(origin, Mapping) else None
    return None


def begin_prompt_builtin(origin: Optional[Mapping[str, Any]]) -> Optional[Token]:
    """Bind one prompt-built-in run for tool observers in this context tree."""
    normalized = normalize_prompt_builtin_origin(origin)
    if normalized is None:
        return None
    return _current_run.set(PromptBuiltinRun(normalized))


def _json_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return {}
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _skill_operations(args: Mapping[str, Any]) -> list[dict[str, Any]]:
    operations = args.get("operations")
    if isinstance(operations, list):
        return [dict(op) for op in operations if isinstance(op, Mapping)]
    return [{
        key: args.get(key)
        for key in ("name", "action", "file_path", "category")
        if args.get(key) is not None
    }]


def observe_tool_result(tool_name: str, args: Any, result: Any) -> None:
    """Collect structured ``skill_manage`` outcomes for the active prompt-built-in turn."""
    run = _current_run.get()
    if run is None or tool_name != "skill_manage" or not isinstance(args, Mapping):
        return
    parsed = _json_result(result)
    operations = _skill_operations(args)
    if not operations:
        return
    status = (
        "staged" if parsed.get("success") and parsed.get("staged")
        else "saved" if parsed.get("success")
        else "failed"
    )
    result_rows = parsed.get("results") if isinstance(parsed.get("results"), list) else []
    artifacts: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        row = result_rows[index] if index < len(result_rows) and isinstance(result_rows[index], Mapping) else {}
        name = str(row.get("name") or operation.get("name") or args.get("name") or "")
        action = str(row.get("action") or operation.get("action") or args.get("action") or "")
        path = row.get("path") or (parsed.get("path") if len(operations) == 1 else None)
        artifact = {
            "kind": "skill",
            "name": name,
            "action": action,
            "status": status,
            "path": str(path) if path else None,
        }
        file_path = row.get("file_path") or operation.get("file_path")
        if file_path:
            artifact["file_path"] = str(file_path)
        artifacts.append(artifact)
    with run.lock:
        run.artifacts.extend(artifacts)


def finish_prompt_builtin(
    token: Optional[Token], result: Any, *, session_id: str, task_id: str, platform: str,
) -> Any:
    """Attach completion metadata and fire ``post_prompt_builtin_run`` exactly once."""
    if token is None:
        return result
    run = _current_run.get()
    _current_run.reset(token)
    if run is None:
        return result
    result_dict = result if isinstance(result, dict) else {}
    status = (
        "interrupted" if result_dict.get("interrupted")
        else "failed" if result_dict.get("failed") or result_dict.get("error")
        else "completed"
    )
    with run.lock:
        artifacts = [dict(artifact) for artifact in run.artifacts]
    completion = {
        "command": run.origin["name"],
        "request": run.origin["raw_args"],
        "run_id": run.origin["run_id"],
        "session_id": str(session_id or ""),
        "task_id": str(task_id or ""),
        "platform": str(platform or ""),
        "status": status,
        "artifacts": artifacts,
    }
    if isinstance(result, dict):
        result["prompt_builtin_completion"] = completion
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if has_hook("post_prompt_builtin_run"):
            invoke_hook("post_prompt_builtin_run", **completion)
    except Exception:
        # Hooks are observers; their discovery or execution can never fail the user turn.
        pass
    return result
