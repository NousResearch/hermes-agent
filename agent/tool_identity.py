"""Canonical tool identity across runtimes.

Hermes renders tool progress from curated tables keyed on *native* tool
names (``terminal``, ``read_file``, ``patch``, ...).  Every runtime that
is not the native tool executor speaks a different vocabulary for the same
operations::

    native          Claude Agent SDK / CLI       Codex app-server
    terminal        Bash                         exec_command
    read_file       Read                         --
    write_file      Write                        --
    patch           Edit / MultiEdit             apply_patch
    search_files    Grep / Glob                  --
    web_search      WebSearch                    web_search
    delegate_task   Task / Agent                 --
    todo_list       TodoWrite                    --

The display layer historically matched on native names only, so the verb
table (``_TOOL_VERBS``), the argument-preview table (``build_tool_preview``)
and the gateway's shell code-block gate all silently missed on every
non-native runtime: the user saw ``Bash: "..."`` instead of ``Running ...``,
and never got the fenced command block.

Codex appeared to be exempt, but only by accident -- Hermes' own tools reach
it through the ``hermes-tools`` MCP server and the bridge strips that
namespace, so they arrive already native.  Codex's *own* shell and patch
paths (``exec_command``/``apply_patch``) were just as broken.

This module is the single place that maps a foreign ``(name, args)`` pair
onto its native equivalent.  It is **display-only**: callers keep passing
the raw name everywhere else (activity summaries, ``tool_calls`` records,
logs, approval prompts), so nothing downstream loses the runtime's real
identity.
"""

from __future__ import annotations

from typing import Any

# MCP tool names arrive in two conventions.  The Claude Agent SDK and the
# Anthropic wire format use ``mcp__<server>__<tool>``; the Codex app-server
# reports ``server``/``tool`` separately, which its bridge renders as
# ``mcp.<server>.<tool>``.  Both are handled.
_MCP_UNDERSCORE_PREFIX = "mcp__"
_MCP_DOT_PREFIX = "mcp."

# Hermes' own tools, when routed to a child runtime, are exposed through
# this MCP server.  The user thinks of them as Hermes tools rather than as
# MCP calls, so the namespace is stripped and the bare native name kept.
# Third-party servers keep their namespace: they have no curated verb, and
# hiding which server a tool came from would be actively misleading.
INTERNAL_MCP_SERVER = "hermes-tools"


# Separator between server and tool in a third-party MCP display label.
# A middle dot rather than ``__``: the underscores are wire scaffolding, the
# server is information.
MCP_SERVER_SEPARATOR = " · "


def _split_mcp(tool_name: str) -> tuple[str, str] | None:
    """Split an MCP tool name into ``(server, tool)``, or None if not MCP.

    Handles both conventions: ``mcp__<server>__<tool>`` (Claude Agent SDK /
    Anthropic wire format) and ``mcp.<server>.<tool>`` (Codex app-server
    bridge).  A malformed name with no tool part yields None so callers fall
    through to treating it as an ordinary name.
    """
    if tool_name.startswith(_MCP_UNDERSCORE_PREFIX):
        parts = tool_name.split("__", 2)
    elif tool_name.startswith(_MCP_DOT_PREFIX):
        parts = tool_name.split(".", 2)
    else:
        return None
    if len(parts) == 3 and parts[1] and parts[2]:
        return parts[1], parts[2]
    return None


def strip_mcp_namespace(tool_name: str) -> str:
    """Return the bare tool name for internal-MCP-routed Hermes tools.

    ``mcp__hermes-tools__read_file`` -> ``read_file``.  Third-party servers
    (``mcp__linear__get_issue``) are returned unchanged.
    """
    split = _split_mcp(tool_name)
    if split is not None and split[0] == INTERNAL_MCP_SERVER:
        return split[1]
    return tool_name


def display_tool_label(tool_name: str) -> str:
    """Return the human-facing name for a tool that has no curated verb.

    The no-verb fallback is the one rendering path that prints a tool's own
    name to the user, so it is where raw wire names leak.  ``mcp__`` and the
    doubled underscores are transport scaffolding -- the user did not choose
    them and they carry no meaning.

    Hermes' own tools drop the namespace entirely: they *are* Hermes tools,
    merely arriving by another road, and they render identically no matter
    which runtime called them.  Third-party servers keep their identity but
    lose the scaffolding (``mcp__linear__get_issue`` -> ``linear · get_issue``)
    -- hiding which server ran a tool would be misleading, but spelling it in
    wire format is just noise.
    """
    if not tool_name:
        return tool_name
    split = _split_mcp(tool_name)
    if split is None:
        return canonical_tool_name(tool_name)
    server, tool = split
    if server == INTERNAL_MCP_SERVER:
        return canonical_tool_name(tool)
    return f"{server}{MCP_SERVER_SEPARATOR}{tool}"


# Foreign tool name -> Hermes-native tool name.
#
# Only mappings where the *semantics* match are listed.  A tool with no
# honest native counterpart is deliberately absent (``ToolSearch``,
# ``BashOutput``, ``KillShell``, ``ExitPlanMode``) and keeps rendering
# under its own name rather than being forced into a verb that would
# misdescribe it.
_TOOL_NAME_ALIASES: dict[str, str] = {
    # --- Legacy Hermes names (accepted by dispatch for resumed sessions) ---
    "todo": "todo_list",
    "cronjob": "cronjob_manage",
    "process": "process_manage",
    "tour": "gui_tour",
    "tip": "show_tip",
    # --- Claude Agent SDK / Claude Code CLI ---
    "Bash": "terminal",
    "Read": "read_file",
    "NotebookRead": "read_file",
    "Write": "write_file",
    "Edit": "patch",
    "MultiEdit": "patch",
    "NotebookEdit": "patch",
    "Glob": "search_files",
    "Grep": "search_files",
    "WebSearch": "web_search",
    "WebFetch": "web_extract",
    "Task": "delegate_task",
    "Agent": "delegate_task",
    "TodoWrite": "todo_list",
    "Skill": "skill_view",
    # --- Codex app-server ---
    "exec_command": "terminal",
    "apply_patch": "patch",
}

# Per-canonical-tool argument aliases: ``canonical_key -> foreign keys``.
#
# Scoped per tool rather than applied globally because the same foreign key
# means different things to different tools -- a global ``query -> pattern``
# rewrite would corrupt ``web_search`` arguments.
_ARG_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "terminal": {"command": ("cmd",)},
    "read_file": {"path": ("file_path", "filepath", "notebook_path")},
    "write_file": {"path": ("file_path", "filepath")},
    "patch": {"path": ("file_path", "filepath", "notebook_path")},
    "search_files": {"pattern": ("glob",), "path": ("file_path",)},
    "web_extract": {"urls": ("url",)},
    "skill_view": {"name": ("skill", "skill_name")},
    "delegate_task": {
        "goal": ("description", "prompt", "task", "instructions"),
    },
}


def _codex_patch_path(args: dict) -> str | None:
    """Derive a display path from the Codex ``apply_patch`` changes list.

    Codex reports ``{"changes": [{"kind": ..., "path": ...}, ...]}`` where
    native ``patch`` expects a single ``path``.  Multi-file patches render
    as ``first.py +2 more`` so the count is not silently dropped.
    """
    changes = args.get("changes")
    if not isinstance(changes, list):
        return None
    paths = [
        str(change["path"])
        for change in changes
        if isinstance(change, dict) and change.get("path")
    ]
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    return f"{paths[0]} +{len(paths) - 1} more"


def canonical_tool_name(tool_name: str) -> str:
    """Map any runtime's tool name onto its Hermes-native equivalent.

    Returns *tool_name* unchanged when it is already native, or when no
    honest native counterpart exists.
    """
    if not tool_name:
        return tool_name
    name = strip_mcp_namespace(tool_name)
    return _TOOL_NAME_ALIASES.get(name, name)


def canonical_tool_args(tool_name: str, args: Any) -> dict:
    """Return *args* with foreign argument keys mapped to native ones.

    The input dict is never mutated: a copy is made only when an alias
    actually applies, so the common native case allocates nothing.  Native
    keys already present always win over an alias.
    """
    if not isinstance(args, dict):
        return {}
    canonical = canonical_tool_name(tool_name)
    result = args

    # Codex's patch shape needs restructuring, not just a key rename.
    if canonical == "patch" and "path" not in result:
        derived = _codex_patch_path(result)
        if derived is not None:
            result = dict(result)
            result["path"] = derived

    aliases = _ARG_ALIASES.get(canonical)
    if not aliases:
        return result

    for native_key, foreign_keys in aliases.items():
        if native_key in result:
            continue
        for foreign_key in foreign_keys:
            value = result.get(foreign_key)
            if value in (None, "", [], {}):
                continue
            if result is args:
                result = dict(args)
            result[native_key] = value
            break
    return result


def canonicalize_tool_call(tool_name: str, args: Any) -> tuple[str, dict]:
    """Return the native ``(tool_name, args)`` pair for display purposes."""
    return canonical_tool_name(tool_name), canonical_tool_args(tool_name, args)
