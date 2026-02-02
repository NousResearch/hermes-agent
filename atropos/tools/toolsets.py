"""
Toolsets (Hermes-Agent inspired).

Toolsets are named groups of tools with optional composition (includes).
They are used to decide which tools are advertised to the model and/or enabled
for a particular environment run.

This module is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TypedDict


class ToolsetDef(TypedDict):
    description: str
    tools: List[str]
    includes: List[str]


TOOLSETS: Dict[str, ToolsetDef] = {
    # Primitive building blocks
    "filesystem": {
        "description": "Read/write files in the current workspace.",
        "tools": ["read_file", "write_file"],
        "includes": [],
    },
    "terminal": {
        "description": "Terminal/command execution tools.",
        # Prefer `terminal` for Hermes compatibility; keep `bash` as a legacy alias.
        "tools": ["terminal", "bash"],
        "includes": [],
    },
    "terminal_stateful": {
        "description": "Stateful terminal execution (enables persistent background processes like tmux).",
        "tools": ["terminal_stateful", "tmux"],
        "includes": [],
    },
    "sandbox": {
        "description": "Standard sandbox tools (terminal + filesystem).",
        "tools": [],
        "includes": ["terminal", "filesystem"],
    },
    # External tools (executed via ToolServer)
    "web": {
        "description": "Web research and content extraction tools (external).",
        "tools": ["web_search", "web_extract", "web_crawl"],
        "includes": [],
    },
    "vision": {
        "description": "Vision/image analysis tools (external).",
        "tools": ["vision_analyze"],
        "includes": [],
    },
    "image_gen": {
        "description": "Image generation tools (external).",
        "tools": ["image_generate"],
        "includes": [],
    },
    "moa": {
        "description": "Advanced reasoning tools (Mixture-of-Agents, external).",
        "tools": ["mixture_of_agents"],
        "includes": [],
    },
    # Convenience presets
    "default": {
        "description": "Default toolset for code-agent tasks.",
        "tools": [],
        "includes": ["sandbox"],
    },
    "debugging": {
        "description": "Debugging toolkit (terminal + web).",
        "tools": [],
        "includes": ["sandbox", "web"],
    },
    "research": {
        "description": "Research toolkit (web + vision + reasoning).",
        "tools": [],
        "includes": ["web", "vision", "moa"],
    },
    "safe": {
        "description": "Safe toolkit without terminal access.",
        "tools": [],
        "includes": ["web", "vision", "image_gen", "moa"],
    },
    "full": {
        "description": "All common tools (sandbox + external).",
        "tools": [],
        "includes": ["sandbox", "web", "vision", "image_gen", "moa"],
    },
}


def get_toolset(name: str) -> Optional[ToolsetDef]:
    return TOOLSETS.get(name)


def get_toolset_names() -> List[str]:
    return list(TOOLSETS.keys())


def validate_toolset(name: str) -> bool:
    return name in {"all", "*"} or name in TOOLSETS


def resolve_toolset(name: str, visited: Optional[Set[str]] = None) -> List[str]:
    """
    Recursively resolve a toolset to a list of tool names.

    Includes are expanded depth-first with cycle protection.
    """
    if visited is None:
        visited = set()

    if name in {"all", "*"}:
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            all_tools.update(resolve_toolset(toolset_name, visited=set()))
        return sorted(all_tools)

    if name in visited:
        # Cycle: return empty to avoid infinite recursion.
        return []

    visited.add(name)
    toolset = TOOLSETS.get(name)
    if toolset is None:
        return []

    tools: Set[str] = set(toolset.get("tools", []))
    for included in toolset.get("includes", []):
        tools.update(resolve_toolset(included, visited=set(visited)))
    return sorted(tools)


def resolve_multiple_toolsets(toolset_names: List[str]) -> List[str]:
    tools: Set[str] = set()
    for name in toolset_names:
        tools.update(resolve_toolset(name))
    return sorted(tools)


def get_toolset_info(name: str) -> Optional[Dict[str, Any]]:
    toolset = get_toolset(name)
    if toolset is None:
        return None
    resolved = resolve_toolset(name)
    return {
        "name": name,
        "description": toolset["description"],
        "direct_tools": toolset["tools"],
        "includes": toolset["includes"],
        "resolved_tools": resolved,
        "tool_count": len(resolved),
        "is_composite": bool(toolset["includes"]),
    }
