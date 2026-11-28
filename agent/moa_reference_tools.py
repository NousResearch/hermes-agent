"""Read-only tools for Mixture-of-Agents reference advisors.

A reference advisor normally sees only a flattened, truncated transcript: it
cannot open the file under discussion, look at the screenshot, or list the
directory, so its advice is guesswork about material it never saw. This module
gives advisors an on-demand look at the machine without giving them any way to
change it.

The tool set is fixed and read-only — ``read_file``, ``list_dir``,
``view_image`` — and no other name is dispatchable. ``read_file`` reuses the
acting agent's own reader, so the credential-store denylist, device-path
blocks, special-file guards and document extraction all apply unchanged.
``view_image`` routes through the auxiliary vision model, the same path
``vision_analyze`` uses when the main model is not multimodal.

``run_reference_tool_loop`` is the bounded driver: at most ``max_iterations``
tool rounds, a shared character budget for tool output, and a wall-clock
deadline after which the advisor is asked for its advice with the tools
withdrawn. The final response is always a plain text completion, so the
caller keeps its existing text extraction and accounting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 8000
MAX_LIST_ENTRIES = 200
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_TOOL_BUDGET_CHARS = 40000

READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a text file on this machine, with line numbers and pagination. "
            "PDF, Office, notebook, OpenDocument, RTF, EPUB and SQLite files are "
            "extracted to text. Images are not readable here — use view_image. "
            "Credential stores are refused."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~/ path to the file"},
                "offset": {"type": "integer", "description": "1-indexed first line (default 1)", "minimum": 1},
                "limit": {"type": "integer", "description": "Lines to read (default 400, max 2000)", "minimum": 1, "maximum": 2000},
            },
            "required": ["path"],
        },
    },
}

LIST_DIR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List the entries of a directory on this machine (names, kind, size).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~/ path to the directory"},
            },
            "required": ["path"],
        },
    },
}

VIEW_IMAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "view_image",
        "description": (
            "Look at an image file on this machine (screenshot, photo, diagram, chart) "
            "and get a description of what it shows. Pass a question to focus the look."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~/ path to the image"},
                "question": {"type": "string", "description": "What to look for in the image (optional)"},
            },
            "required": ["path"],
        },
    },
}

REFERENCE_TOOL_SCHEMAS: list[dict[str, Any]] = [READ_FILE_TOOL, LIST_DIR_TOOL, VIEW_IMAGE_TOOL]

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: max(0, limit - 200)]
    return f"{head}\n[... truncated {len(text) - len(head)} chars; narrow the read with offset/limit ...]"


def _log_arg(arguments: Any, limit: int = 120) -> str:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[:limit]}…"


def _read_file(path: str, offset: int = 1, limit: int = 400) -> str:
    from tools.file_tools import read_file_tool

    try:
        offset = max(1, int(offset or 1))
        limit = max(1, min(int(limit or 400), 2000))
    except (TypeError, ValueError):
        offset, limit = 1, 400
    result = read_file_tool(path, offset, limit, task_id="moa-reference")
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return result
    if payload.get("is_image"):
        return f"{path} is an image file — call view_image on it instead of read_file."
    return result


def _list_dir(path: str) -> str:
    target = os.path.expanduser(str(path or ".").strip() or ".")
    if not os.path.isdir(target):
        return f"error: not a directory: {target}"
    rows: list[str] = []
    try:
        with os.scandir(target) as entries:
            ordered = sorted(entries, key=lambda e: (not e.is_dir(), e.name.lower()))
            for entry in ordered[:MAX_LIST_ENTRIES]:
                try:
                    is_dir = entry.is_dir()
                    size = 0 if is_dir else entry.stat().st_size
                except OSError:
                    is_dir, size = False, 0
                rows.append(f"{'dir' if is_dir else 'file'} {entry.name}" + ("" if is_dir else f"  {size}B"))
            extra = len(ordered) - len(rows)
    except OSError as exc:
        return f"error: {exc}"
    header = f"dir: {target}"
    if extra > 0:
        rows.append(f"... {extra} more entries not shown")
    return "\n".join([header, *rows])


def _view_image(path: str, question: str = "") -> str:
    from tools.vision_tools import vision_analyze_tool

    target = os.path.expanduser(str(path or "").strip())
    if not target:
        return "error: view_image needs a path"
    if not os.path.isfile(target):
        return f"error: not a file: {target}"
    prompt = str(question or "").strip() or (
        "详细描述这张图：画面主体、关键细节、配色、以及图中出现的任何文字。"
    )
    raw = asyncio.run(vision_analyze_tool(target, prompt, None, task_id="moa-reference"))
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return f"error: {payload.get('error') or payload.get('analysis') or 'vision failed'}"
        return str(payload.get("analysis") or raw)
    return raw


_HANDLERS: dict[str, Callable[..., str]] = {
    "read_file": _read_file,
    "list_dir": _list_dir,
    "view_image": _view_image,
}


def execute_reference_tool(name: str, arguments: Any, *, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Run one read-only tool call. Unknown names and failures come back as text."""
    handler = _HANDLERS.get(str(name or "").strip())
    if handler is None:
        return f"error: unknown tool '{name}'. Available read-only tools: {', '.join(_HANDLERS)}"
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return "error: tool arguments must be a JSON object"
    elif isinstance(arguments, dict):
        args = arguments
    else:
        args = {}
    if not isinstance(args, dict):
        return "error: tool arguments must be a JSON object"
    try:
        try:
            result = handler(**{k: v for k, v in args.items() if k in handler.__code__.co_varnames})
        except TypeError:
            # A model inventing an argument name must not lose the read: retry with the path.
            result = handler(str(args.get("path", "")))
    except Exception as exc:
        logger.warning("MoA reference tool %s failed", name, exc_info=True)
        return f"error: {exc}"
    return _truncate(str(result), max(0, max_chars))


def _first_message(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "message", None)


def _tool_call_payload(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", "") or "",
        "type": "function",
        "function": {
            "name": getattr(function, "name", "") or "",
            "arguments": getattr(function, "arguments", "") or "{}",
        },
    }


def run_reference_tool_loop(
    send: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Any],
    messages: list[dict[str, Any]],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tool_budget_chars: int = DEFAULT_TOOL_BUDGET_CHARS,
    deadline: float | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Drive an advisor through bounded read-only tool rounds.

    ``send(messages, tools)`` performs one model call (``tools=None`` withdraws the
    tool set). Returns every response received and the final message list; the last
    response is the advisor's answer. Never raises for a tool failure — a failed
    tool becomes text the advisor can react to.
    """
    convo = list(messages)
    responses: list[Any] = []
    budget = max(0, int(tool_budget_chars))
    rounds = 0
    while True:
        expired = deadline is not None and time.monotonic() > deadline
        use_tools = rounds < max_iterations and budget > 0 and not expired
        response = send(convo, REFERENCE_TOOL_SCHEMAS if use_tools else None)
        responses.append(response)
        message = _first_message(response)
        calls = getattr(message, "tool_calls", None)
        if not use_tools or not isinstance(calls, list) or not calls:
            return responses, convo
        convo.append({
            "role": "assistant",
            "content": getattr(message, "content", "") or "",
            "tool_calls": [_tool_call_payload(call) for call in calls],
        })
        for call in calls:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "") or ""
            arguments = getattr(function, "arguments", "") or "{}"
            result = execute_reference_tool(
                name, arguments, max_chars=min(MAX_TOOL_RESULT_CHARS, budget),
            )
            budget -= len(result)
            logger.info(
                "MoA reference tool %s(%s) -> %d chars", name, _log_arg(arguments), len(result),
            )
            convo.append({
                "role": "tool",
                "tool_call_id": getattr(call, "id", "") or "",
                "content": result,
            })
        rounds += 1
