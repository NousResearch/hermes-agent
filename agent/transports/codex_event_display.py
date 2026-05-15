"""Surface codex app-server tool calls through Hermes' tool-progress display.

When Hermes runs the codex_app_server runtime, the agent loop is owned by the
codex CLI subprocess instead of run_agent.py — which means the "tool.started" /
"tool.completed" events Hermes' display path expects (gateway tool-progress
bubbles, CLI activity feed) never fire from the standard call sites in
run_agent.py. Without a bridge, codex-runtime turns appear opaque: the bot
takes a long time and then a message appears, with no indication that shell
commands or file edits ran in between.

This module is the bridge. It consumes raw `note: dict` notifications from
codex's JSON-RPC stream (delivered via `CodexAppServerSession`'s `on_event`
hook) and translates them into the same `progress_callback(event_type,
tool_name, preview, args)` shape the agent already uses for native tools.
The gateway's existing tool-progress rendering, dedup, queue, and edit logic
then work without changes.

Mapping (item type → display name):

| Codex item type    | Display name            | Notes                              |
|--------------------|-------------------------|------------------------------------|
| commandExecution   | exec_command            | matches codex_event_projector      |
| fileChange         | apply_patch             | matches codex_event_projector      |
| mcpToolCall        | mcp.<server>.<tool>     | user MCP servers                   |
| mcpToolCall        | <tool>                  | server="hermes-tools" — see below  |
| dynamicToolCall    | <tool>                  | matches codex_event_projector      |
| webSearch          | web_search              | codex built-in tool                |
| reasoning          | (skipped)               | not a tool call                    |
| agentMessage       | (skipped)               | assistant text, not a tool         |
| userMessage        | (skipped)               | user echo, not a tool              |

Special case: the "hermes-tools" MCP server is Hermes' own tool callback
exposed to codex as an MCP server. When codex invokes
web_search/browser_*/vision_analyze/etc. through it, the inner Hermes
dispatch runs in a separate hermes-tools-mcp-server subprocess that
does NOT have access to the parent agent's tool_progress_callback —
so the inner call can never surface its own native progress event.
The codex-level mcpToolCall event IS the display event for that call,
and we drop the mcp.hermes-tools.* namespacing so users see
"web_search" rather than "mcp.hermes-tools.web_search" — matching how
they think about these tools.

Streaming output deltas (item/<type>/outputDelta, item/<type>/delta) are
ignored — only `item/started` and `item/completed` produce progress events,
matching how Hermes renders native tools (start + completion, no intra-call
stdout). Verbose stdout surfaces in the final response if the model decides
to include it.

Threading: the events are delivered on the agent's main thread from inside
CodexAppServerSession.run_turn (via _client.take_notification). No locking
needed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Methods we surface. Anything else (turn/started, item/<type>/outputDelta,
# turn/completed, etc.) is ignored. We re-render on every started/completed
# pass — the gateway's dedup-by-tool-name handles the "same tool many times
# in a row" case.
_RENDER_METHODS = frozenset({"item/started", "item/completed"})

# Item types we surface. Everything else (reasoning, agentMessage,
# userMessage, plan, hookPrompt, collabAgentToolCall, ...) is dropped.
_TOOL_ITEM_TYPES = frozenset({
    "commandExecution",
    "fileChange",
    "mcpToolCall",
    "dynamicToolCall",
    "webSearch",
})

# Internal MCP server that wraps Hermes' native tools. When codex calls
# back through it, the inner dispatch runs in a SEPARATE
# hermes-tools-mcp-server subprocess that has no access to the parent
# agent's tool_progress_callback — so the inner call can never surface
# its own native progress event. The codex-level mcpToolCall event IS
# the display event for those calls; we strip the mcp.hermes-tools.*
# namespacing and emit the bare tool name. See module docstring for
# the full design note.
_INTERNAL_MCP_SERVER = "hermes-tools"

# Length limits — keep previews short enough for chat platforms but
# informative enough to identify the call. The gateway's
# tool_preview_length config will further trim if configured.
_PREVIEW_MAX_LEN = 200


def _truncate(s: str, max_len: int = _PREVIEW_MAX_LEN) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _preview_command(item: dict) -> tuple[str, dict]:
    """Build the tool-progress preview for a commandExecution item.

    Returns (preview_string, args_dict). The args dict matches the shape
    the codex_event_projector emits so any downstream consumer that
    inspects the args sees consistent fields.
    """
    command = item.get("command") or ""
    cwd = item.get("cwd") or ""
    args = {"command": command, "cwd": cwd}
    return _truncate(command), args


def _preview_file_change(item: dict) -> tuple[str, dict]:
    """Build the preview for a fileChange item.

    Codex puts the changeset on the item under `changes`: a list of
    `{kind: {type: add|update|delete}, path: str}`. The same `changes`
    field is present on item/started (so we can show what's about to
    change) and item/completed.
    """
    changes = item.get("changes") or []
    kinds: dict[str, int] = {}
    paths: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        kind = (change.get("kind") or {}).get("type") or "update"
        kinds[kind] = kinds.get(kind, 0) + 1
        p = change.get("path") or ""
        if p:
            paths.append(p)
    counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    if paths:
        head = paths[0]
        if len(paths) > 1:
            head = f"{head} +{len(paths) - 1}"
        preview = f"{counts}: {head}" if counts else head
    else:
        preview = counts or "1 change"
    args = {"changes": [
        {
            "kind": (c.get("kind") or {}).get("type") or "update",
            "path": c.get("path") or "",
        }
        for c in changes if isinstance(c, dict)
    ]}
    return _truncate(preview), args


def _preview_mcp_tool_call(item: dict) -> tuple[str, dict]:
    """Build the preview for an mcpToolCall item.

    The display name is constructed by the caller (it needs `server` and
    `tool` to decide whether to skip). This helper only builds the preview
    and args dict.
    """
    raw_args = item.get("arguments") or {}
    if not isinstance(raw_args, dict):
        raw_args = {"arguments": raw_args}
    # Prefer a primary-arg preview matching how Hermes-native tool
    # progress builds previews (path/query/command/etc.).
    preview_keys = ("path", "query", "command", "url", "file", "name")
    for k in preview_keys:
        v = raw_args.get(k)
        if isinstance(v, str) and v.strip():
            return _truncate(v), raw_args
    # Fall back to first string-valued arg.
    for v in raw_args.values():
        if isinstance(v, str) and v.strip():
            return _truncate(v), raw_args
    # No string args — just compact-serialize the whole arg dict.
    try:
        preview = json.dumps(raw_args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        preview = ""
    return _truncate(preview), raw_args


def _preview_dynamic_tool_call(item: dict) -> tuple[str, dict]:
    """Same shape as mcpToolCall: arguments dict with optional primary arg."""
    return _preview_mcp_tool_call(item)


def _preview_web_search(item: dict) -> tuple[str, dict]:
    """Codex's built-in web_search tool. Query lives under `query`."""
    query = item.get("query") or ""
    args = {"query": query}
    return _truncate(query), args


def _classify(item: dict) -> Optional[tuple[str, str, dict]]:
    """Map a codex item to (display_name, preview, args) or None to skip.

    None means: not a tool item, or an internal item we want to suppress
    (e.g. an mcpToolCall through the hermes-tools server, which will fire
    its own native progress event downstream).
    """
    item_type = item.get("type") or ""
    if item_type not in _TOOL_ITEM_TYPES:
        return None

    if item_type == "commandExecution":
        preview, args = _preview_command(item)
        return ("exec_command", preview, args)

    if item_type == "fileChange":
        preview, args = _preview_file_change(item)
        return ("apply_patch", preview, args)

    if item_type == "mcpToolCall":
        server = item.get("server") or "mcp"
        tool = item.get("tool") or "unknown"
        preview, args = _preview_mcp_tool_call(item)
        if server == _INTERNAL_MCP_SERVER:
            # The hermes-tools MCP server is a separate subprocess that
            # doesn't have access to this agent's tool_progress_callback,
            # so the inner Hermes dispatch can't surface a native progress
            # event. Emit the bare tool name here (web_search,
            # browser_navigate, vision_analyze, ...) so the codex-level
            # event IS the display event. Drop the mcp.hermes-tools.*
            # namespacing since the user thinks of these as Hermes tools,
            # not as MCP calls.
            return (tool, preview, args)
        return (f"mcp.{server}.{tool}", preview, args)

    if item_type == "dynamicToolCall":
        tool = item.get("tool") or "unknown"
        preview, args = _preview_dynamic_tool_call(item)
        return (tool, preview, args)

    if item_type == "webSearch":
        preview, args = _preview_web_search(item)
        return ("web_search", preview, args)

    return None


def make_progress_bridge(
    get_progress_callback: Callable[[], Optional[Callable[..., Any]]],
) -> Callable[[dict], None]:
    """Build an `on_event(note: dict)` adapter that translates codex
    notifications into Hermes' progress_callback shape.

    Args:
        get_progress_callback: a zero-arg callable returning the agent's
            current `tool_progress_callback` (or None). MUST be late-binding
            — the gateway swaps the agent's progress_callback closure per
            turn (each turn has its own progress queue, dedup state, and
            cleanup tracking), and the CodexAppServerSession lives across
            turns. If we captured the callback once at session creation,
            tool events on turn N+1 would fire into turn N's dead queue
            and the user would see nothing past the first turn.

            Typical call site: `make_progress_bridge(lambda: self.tool_progress_callback)`.

    Returns:
        A callable suitable for passing as `on_event` to
        CodexAppServerSession.__init__.

    The returned adapter wraps every callback invocation in try/except so a
    misbehaving progress callback can never crash the codex transport read
    loop. Errors are logged at debug.
    """

    def _bridge(note: dict) -> None:
        try:
            # Late-bind the callback on every event. The agent's
            # tool_progress_callback is per-turn state; capturing it once
            # would route every codex turn's tool events into the first
            # turn's queue.
            progress_callback = get_progress_callback()
            if progress_callback is None:
                return
            method = note.get("method", "")
            if method not in _RENDER_METHODS:
                return
            params = note.get("params") or {}
            item = params.get("item") or {}
            if not isinstance(item, dict):
                return
            classified = _classify(item)
            if classified is None:
                return
            display_name, preview, args = classified
            event_type = (
                "tool.started" if method == "item/started"
                else "tool.completed"
            )
            progress_callback(event_type, display_name, preview, args)
        except Exception:  # pragma: no cover - display path is best-effort
            logger.debug(
                "codex tool-progress bridge failed", exc_info=True
            )

    return _bridge
