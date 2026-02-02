"""
Toolset resolution for Hermes-Agent Atropos integration.

We primarily reuse Hermes-Agent toolsets (`toolsets.py`), but Atropos training/envs
need a few extra sandbox-oriented toolsets that Hermes doesn't expose by default
(e.g. filesystem + stateful terminal).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import toolsets as hermes_toolsets


ATROPOS_TOOLSETS: Dict[str, Dict[str, Any]] = {
    "filesystem": {
        "description": "Read/write files in the sandbox workspace.",
        "tools": ["read_file", "write_file"],
        "includes": [],
    },
    "terminal_stateful": {
        "description": "Stateful terminal execution (tmux/TUI support) inside the sandbox.",
        "tools": ["terminal_stateful", "tmux"],
        "includes": [],
    },
    "sandbox": {
        "description": "Sandbox tools (terminal + filesystem).",
        "tools": [],
        "includes": ["terminal", "filesystem"],
    },
    "default": {
        "description": "Default toolset for Atropos AgentEnv tasks.",
        "tools": [],
        "includes": ["sandbox"],
    },
    "full": {
        "description": "All Hermes tools plus Atropos sandbox additions.",
        "tools": [],
        "includes": ["all", "filesystem", "sandbox", "terminal_stateful"],
    },
}


def validate_toolset(name: str) -> bool:
    if name in {"all", "*"}:
        return True
    return hermes_toolsets.validate_toolset(name) or name in ATROPOS_TOOLSETS


def resolve_toolset(name: str, visited: Optional[Set[str]] = None) -> List[str]:
    if visited is None:
        visited = set()

    if name in {"all", "*"}:
        # Union Hermes + Atropos toolsets.
        all_tools: Set[str] = set()
        for tname in hermes_toolsets.get_toolset_names():
            all_tools.update(resolve_toolset(tname, visited=set()))
        for tname, spec in ATROPOS_TOOLSETS.items():
            # Avoid recursion: some Atropos toolsets (e.g. "full") include "all".
            if tname == "full" or "all" in (spec.get("includes") or []):
                continue
            all_tools.update(resolve_toolset(tname, visited=set()))
        return sorted(all_tools)

    if name in ATROPOS_TOOLSETS:
        if name in visited:
            return []
        visited.add(name)
        spec = ATROPOS_TOOLSETS[name]
        tools: Set[str] = set(spec.get("tools", []))
        for inc in spec.get("includes", []):
            tools.update(resolve_toolset(inc, visited=set(visited)))
        return sorted(tools)

    # Fall back to Hermes toolsets.
    # IMPORTANT: do not pre-add `name` to `visited` here; Hermes' resolver uses
    # `visited` for its own cycle detection and will treat the presence of `name`
    # as a circular dependency.
    return sorted(hermes_toolsets.resolve_toolset(name, visited=set(visited)))


def resolve_multiple_toolsets(names: List[str]) -> List[str]:
    tools: Set[str] = set()
    for name in names:
        tools.update(resolve_toolset(name, visited=set()))
    return sorted(tools)
