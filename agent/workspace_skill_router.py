"""Deterministic, workspace-scoped skill preloads for current user turns.

The system prompt's skill catalogue remains advisory.  A workspace may own a
small routing contract where a missed skill would violate its invariants; this
module applies that contract by adding the already-resolved skill content to
the current user message.  It never mutates a session's cached system prompt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_VKS_WORKSPACE_NAME = "vks-pattern-library"
_VKS_CONTEXT_FILE = ".hermes.md"
_UI_SIGNAL = re.compile(
    r"\b(?:redesenh\w*|reskin\w*|blueprint\s+visual|design\s+system|"
    r"hierarquia\s+visual|interface(?:s)?|ui)\b",
    re.IGNORECASE,
)
_NON_UI_SIGNAL = re.compile(r"\b(?:sem|without)\s+(?:interface|ui)\b", re.IGNORECASE)


def _is_vks_pattern_library(cwd: str | Path | None) -> bool:
    if cwd is None:
        return False
    try:
        workspace = Path(cwd).expanduser().resolve()
    except OSError:
        return False
    return workspace.name == _VKS_WORKSPACE_NAME and (workspace / _VKS_CONTEXT_FILE).is_file()


def _requires_vks_design_director(user_message: Any, cwd: str | Path | None) -> bool:
    if not isinstance(user_message, str) or not _is_vks_pattern_library(cwd):
        return False
    return not _NON_UI_SIGNAL.search(user_message) and bool(_UI_SIGNAL.search(user_message))


def route_workspace_skill_context(
    user_message: Any, *, cwd: str | Path | None, task_id: str | None = None,
) -> tuple[Any, list[str]]:
    """Return the user-message payload with required VKS UI guidance prepended.

    Only the VKS Pattern Library workspace receives this binding.  Missing or
    disabled skills fail loudly rather than silently degrading to advisory
    catalogue matching.
    """
    if not _requires_vks_design_director(user_message, cwd):
        return user_message, []

    from agent.skill_commands import build_preloaded_skills_prompt

    prompt, loaded, missing = build_preloaded_skills_prompt(["vks-design-director"], task_id=task_id)
    if missing or loaded != ["vks-design-director"] or not prompt:
        raise RuntimeError(
            "VKS UI routing requires the enabled vks-design-director skill; "
            f"loaded={loaded!r}, missing={missing!r}."
        )

    return (
        "[VKS WORKSPACE ROUTING: vks-design-director is active for this UI turn.]\n\n"
        f"{prompt}\n\n{user_message}",
        loaded,
    )
