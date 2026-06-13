"""Normalize common coding-agent tool aliases to Hermes tool names.

Some OpenAI-compatible coding-agent gateways expose or bias models toward local
tool names such as ``shell``, ``write``, and ``edit`` even when Hermes sends its
own tool schema. Normalize the small, lossless subset before invalid-tool
validation so the agent executes the intended Hermes tool instead of feeding a
recoverable-but-often-ignored error back to the model.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


def normalize_tool_call_aliases(tool_calls: Iterable[Any] | None, valid_tool_names: set[str]) -> None:
    """Rewrite recognized tool-call aliases in-place when Hermes has a target.

    ``tool_calls`` are NormalizedResponse ToolCall objects (or OpenAI-like
    objects) with ``tc.function.name`` and ``tc.function.arguments``.  The
    function is intentionally conservative: it only rewrites when the mapped
    Hermes tool is currently available, and only changes argument keys that have
    direct equivalents.
    """
    if not tool_calls or not valid_tool_names:
        return

    for tool_call in tool_calls:
        fn = getattr(tool_call, "function", tool_call)
        raw_name = getattr(fn, "name", None)
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        normalized = name.lower().replace("-", "_")
        args = _parse_args(getattr(fn, "arguments", None))

        mapped_name: str | None = None
        mapped_args: dict[str, Any] | None = None

        if normalized in {"bash", "shell", "runterminalcmd", "run_terminal_cmd"}:
            mapped_name = "terminal"
            mapped_args = _compact({
                "command": _first_string(args, "command", "cmd", "script"),
                "workdir": _first_string(args, "workdir", "workingDirectory", "cwd", "directory"),
                "timeout": _number(args.get("timeout")),
            })
        elif normalized in {"write", "writefile", "write_file", "createfile", "create_file"}:
            mapped_name = "write_file"
            mapped_args = _compact({
                "path": _first_string(args, "path", "filePath", "file", "filename"),
                "content": _first_string(args, "content", "fileText", "text", "new_contents", "newContents"),
            })
        elif normalized in {"read", "readfile", "read_file", "openfile", "open_file"}:
            mapped_name = "read_file"
            mapped_args = _compact({
                "path": _first_string(args, "path", "filePath", "file", "filename"),
                "offset": _number(args.get("offset")),
                "limit": _number(args.get("limit")),
            })
        elif normalized in {"grep", "search", "searchfiles", "search_files"}:
            mapped_name = "search_files"
            context = _number(args.get("context"))
            if context is None:
                before = _number(args.get("contextBefore"))
                after = _number(args.get("contextAfter"))
                context = max(v for v in (before, after, 0) if v is not None)
            mapped_args = _compact({
                "pattern": _first_string(args, "pattern", "query", "regex", "search"),
                "path": _first_string(args, "path", "directory", "cwd"),
                "file_glob": _first_string(args, "glob", "include", "file_glob"),
                "limit": _number(args.get("headLimit")) or _number(args.get("limit")),
                "offset": _number(args.get("offset")),
                "context": context,
            })
        elif normalized in {"glob", "fileglob", "file_glob", "findfiles", "find_files", "ls", "list"}:
            mapped_name = "search_files"
            pattern = _first_string(args, "globPattern", "pattern", "glob")
            if not pattern:
                pattern = "*"
            mapped_args = _compact({
                "pattern": pattern,
                "target": "files",
                "path": _first_string(args, "path", "targetDirectory", "directory", "cwd"),
                "limit": _number(args.get("limit")),
            })
        elif normalized in {"edit", "editfile", "edit_file", "replacefile", "replace_file", "searchreplace", "search_replace"}:
            mapped_name = "patch"
            old_string = _first_string(args, "old_string", "oldString", "oldText", "old", "search")
            new_string = _first_string(args, "new_string", "newString", "newText", "replacement", "content")
            patch_text = _first_string(args, "patch", "patchContent")
            if patch_text:
                mapped_args = {"mode": "patch", "patch": patch_text}
            else:
                mapped_args = _compact({
                    "mode": "replace",
                    "path": _first_string(args, "path", "filePath", "file", "filename"),
                    "old_string": old_string,
                    "new_string": new_string,
                })

        if not mapped_name or mapped_name not in valid_tool_names or mapped_args is None:
            continue
        # Do not convert to an unusable call with missing required arguments.
        if mapped_name == "terminal" and not mapped_args.get("command"):
            continue
        if mapped_name in {"write_file", "read_file"} and not mapped_args.get("path"):
            continue
        if mapped_name == "write_file" and "content" not in mapped_args:
            continue
        if mapped_name == "search_files" and not mapped_args.get("pattern"):
            continue
        if mapped_name == "patch" and mapped_args.get("mode") == "replace" and not (
            mapped_args.get("path") and "old_string" in mapped_args and "new_string" in mapped_args
        ):
            continue

        fn.name = mapped_name
        fn.arguments = json.dumps(mapped_args, ensure_ascii=False)


def _parse_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _first_string(args: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _compact(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
