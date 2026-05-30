"""Volatile operating-context blocks for the system prompt.

Plugins register hooks here instead of patching :mod:`agent.system_prompt`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

OperatingPromptHook = Callable[[Any], List[str]]

_HOOKS: list[OperatingPromptHook] = []


def register_operating_prompt_hook(hook: OperatingPromptHook) -> None:
    """Register a callable that returns extra volatile prompt blocks for an agent."""

    if hook not in _HOOKS:
        _HOOKS.append(hook)


def collect_operating_prompt_blocks(agent: Any) -> List[str]:
    """Return all registered operating prompt blocks for this agent session."""

    blocks: List[str] = []
    for hook in _HOOKS:
        try:
            part = hook(agent)
            if not part:
                continue
            if isinstance(part, str):
                blocks.append(part)
            else:
                blocks.extend(str(p) for p in part if p)
        except Exception as exc:
            logger.debug("Operating prompt hook failed: %s", exc)
    return blocks
