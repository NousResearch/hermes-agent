"""OpenViking tool surface: ``viking_*`` tool schemas, name/status taxonomy, and the
parsing of tool calls and tool results out of message history.

Extracted byte-verbatim from ``plugins/memory/openviking/__init__.py`` (2K-law fracture);
import these names from THIS module — the package ``__init__`` only imports the few it
calls itself and is not a re-export surface.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any, Dict, List, Optional

from agent.message_content import flatten_message_text


# -- Tool schemas -----------------------------------------------------------

def _tool_schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}


def _str(description: str, **extra) -> dict:
    return {"type": "string", **extra, "description": description}


SEARCH_SCHEMA = _tool_schema(
    "viking_search",
    "Semantic search over the OpenViking knowledge base. Returns ranked results with viking:// URIs for deeper reading. "
    "Use mode='deep' for complex queries that need reasoning across multiple sources, 'fast' for simple lookups.",
    {
        "query": _str("Search query."),
        "mode": _str("Search depth (default: auto).", enum=["auto", "fast", "deep"]),
        "scope": _str("Viking URI prefix to scope search (e.g. 'viking://resources/docs/')."),
        "limit": {"type": "integer", "description": "Max results (default: 10)."},
    },
    ["query"],
)

READ_SCHEMA = _tool_schema(
    "viking_read",
    "Read one or a few specific viking:// URIs returned by viking_search or viking_browse. Three detail levels:\n"
    "  abstract — ~100 token summary (L0)\n  overview — ~2k token key points (L1)\n  full — complete content (L2)\n"
    "Start with abstract/overview, only use full when you need details. For multiple strong candidates, pass uris with up to three URIs.",
    {
        "uri": _str("Single viking:// URI to read."),
        "uris": {"type": "array", "items": {"type": "string"}, "description": "Optional batch of up to three viking:// URIs to read."},
        "level": _str("Detail level (default: overview).", enum=["abstract", "overview", "full"]),
    },
    [],
)

BROWSE_SCHEMA = _tool_schema(
    "viking_browse",
    "Browse the OpenViking knowledge store like a filesystem.\n  list — show directory contents\n  tree — show hierarchy\n  stat — show metadata for a URI",
    {
        "action": _str("Browse action.", enum=["tree", "list", "stat"]),
        "path": _str("Viking URI path (default: viking://). Examples: 'viking://resources/', 'viking://~/memories/'."),
    },
    ["action"],
)

REMEMBER_SCHEMA = _tool_schema(
    "viking_remember",
    "Submit important long-term information to OpenViking through session memory extraction. Success means the source was "
    "submitted, not that a distinct memory file was created. OpenViking can add, merge, or skip the final memory. Use this tool "
    "when OpenViking should decide how to retain the information. Do not use it when an exact memory file or URI is required. "
    "If the message is accepted but commit fails, it normally remains live and unextracted because server auto-commit is "
    "disabled by default; follow the returned recovery instructions.",
    {"content": _str("The information to remember.")},
    ["content"],
)

FORGET_SCHEMA = _tool_schema(
    "viking_forget",
    "Delete one OpenViking memory file by exact viking:// URI. Use only when the user explicitly asks to forget or delete a "
    "specific memory and you have the exact memory file URI. Resources, skills, sessions, directories, generated summaries, "
    "and broad deletes are rejected.",
    {"uri": _str("Exact viking:// memory file URI ending in .md.")},
    ["uri"],
)

ADD_RESOURCE_SCHEMA = _tool_schema(
    "viking_add_resource",
    "Add a remote URL or local file/directory to the OpenViking knowledge base. Remote resources must be public http(s), git, "
    "or ssh URLs. Local files are uploaded first using OpenViking temp_upload. The system automatically parses, indexes, and "
    "generates summaries.",
    {
        "url": _str("Remote URL or local file/directory path to add."),
        "reason": _str("Why this resource is relevant (improves search)."),
        "to": _str("Optional target viking:// URI for the resource."),
        "parent": _str("Optional parent viking:// URI. Cannot be used with to."),
        "instruction": _str("Optional processing instruction for semantic extraction."),
        "wait": {"type": "boolean", "description": "Whether to wait for processing to complete."},
        "timeout": {"type": "number", "description": "Timeout in seconds when wait is true."},
    },
    ["url"],
)

_TOOL_SCHEMAS = [SEARCH_SCHEMA, READ_SCHEMA, BROWSE_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA, ADD_RESOURCE_SCHEMA]
# Recall tools (read-only) whose results are never re-ingested — echoing recalled
# memory back into the transcript would re-store it. Write tools are deliberately absent.
_OPENVIKING_RECALL_TOOL_NAMES = {SEARCH_SCHEMA["name"], READ_SCHEMA["name"], BROWSE_SCHEMA["name"]}
# viking_* tool name -> provider method (resolved via getattr so instance patches apply).
_TOOL_HANDLERS = {schema["name"]: "_tool_" + schema["name"].removeprefix("viking_") for schema in _TOOL_SCHEMAS}
# Inbound tool-result status aliases -> canonical "error" / "completed" (else "pending").
_TOOL_STATUS_ERROR_ALIASES = {"error", "failed", "failure"}
_TOOL_STATUS_COMPLETED_ALIASES = {"completed", "complete", "success", "succeeded"}


# -- Message / tool-call parsing --------------------------------------------


_message_text = flatten_message_text  # OpenAI-style string/list content -> text


def _tool_part(tool_id: str, tool_name: str, tool_input: Dict[str, Any], tool_status: str, **extra) -> Dict[str, Any]:
    return {"type": "tool", "tool_id": tool_id, "tool_name": tool_name, "tool_input": tool_input, **extra, "tool_status": tool_status}


def _tool_call_id(tool_call: Dict[str, Any]) -> str:
    return str(tool_call.get("id") or tool_call.get("tool_call_id") or "")


def _tool_call_name(tool_call: Dict[str, Any]) -> str:
    function = tool_call.get("function")
    return str((function.get("name") if isinstance(function, dict) else tool_call.get("name")) or "")


def _is_openviking_recall_tool_name(tool_name: Any) -> bool:
    return str(tool_name or "").strip().lower() in _OPENVIKING_RECALL_TOOL_NAMES


def _tool_call_input(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function")
    raw_args = function.get("arguments") if isinstance(function, dict) else None
    if raw_args is None:
        raw_args = tool_call.get("args")
    if raw_args is None or (isinstance(raw_args, str) and not raw_args.strip()):
        return {}
    if not isinstance(raw_args, str):
        return raw_args if isinstance(raw_args, dict) else {"value": raw_args}
    with suppress(Exception):
        parsed = json.loads(raw_args)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": raw_args}


def _tool_result_status(message: Dict[str, Any]) -> str:
    raw_status = str(message.get("status") or message.get("tool_status") or "").lower()
    if raw_status in _TOOL_STATUS_ERROR_ALIASES:
        return "error"
    if raw_status in _TOOL_STATUS_COMPLETED_ALIASES:
        return "completed"
    text = _message_text(message.get("content")).strip()
    parsed = None
    if text:
        with suppress(Exception):
            parsed = json.loads(text)
    if isinstance(parsed, dict):
        exit_code = parsed.get("exit_code")
        if (str(parsed.get("status") or "").lower() in _TOOL_STATUS_ERROR_ALIASES or parsed.get("success") is False
                or bool(parsed.get("error")) or (isinstance(exit_code, int) and exit_code != 0)):
            return "error"
    return "completed"


def _rfind_message(messages: List[Any], role: str, start: int, expected: Any = None) -> Optional[int]:
    """Index of the last ``role`` message at or before ``start`` (matching ``expected`` text if given)."""
    expected_text = None if expected is None else _message_text(expected).strip()
    for idx in range(start, -1, -1):
        message = messages[idx]
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        if expected_text is None or (expected_text and _message_text(message.get("content")).strip() == expected_text):
            return idx
    return None


def _index_tool_calls(messages: List[Dict[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], set[str], set[str]]:
    """-> (assistant tool_calls by id, ids with a result in the slice, recall-tool ids to drop)."""
    tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
    completed_tool_ids: set[str] = set()
    skipped_tool_ids: set[str] = set()
    for message in messages:
        if message.get("role") == "tool":
            if tool_id := str(message.get("tool_call_id") or message.get("id") or ""):
                completed_tool_ids.add(tool_id)
                if _is_openviking_recall_tool_name(message.get("name")):
                    skipped_tool_ids.add(tool_id)
        elif message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict) and (tool_id := _tool_call_id(tool_call)):
                    tool_name = _tool_call_name(tool_call)
                    tool_calls_by_id[tool_id] = {"tool_name": tool_name, "tool_input": _tool_call_input(tool_call)}
                    if _is_openviking_recall_tool_name(tool_name):
                        skipped_tool_ids.add(tool_id)
    return tool_calls_by_id, completed_tool_ids, skipped_tool_ids
