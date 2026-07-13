"""Small optional integrations with the cmux terminal multiplexer.

This module is deliberately dependency-free and fail-soft. Hermes should behave
exactly the same outside cmux, when the cmux CLI is missing, or when the socket
is unavailable.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Mapping
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PREFIX = "Goal: "
_DEFAULT_MAX_CHARS = 60
_TIMEOUT_SECONDS = 2.0
_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_BULLET_RE = re.compile(r"^[-*+]\s+")


def _compact_goal_text(goal: str) -> str:
    """Return a single-line, title-friendly version of a goal string."""
    text = str(goal or "").strip()
    if text.lower().startswith("/goal"):
        text = text[5:].strip()
    # Prefer the first meaningful line over concatenating a detailed checklist.
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate = _MARKDOWN_BULLET_RE.sub("", candidate).strip()
        if candidate:
            text = candidate
            break
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.strip("`*_#[](){}<>\"'“”‘’") or "Goal"


def build_goal_workspace_title(
    goal: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    prefix: str = _DEFAULT_PREFIX,
) -> str:
    """Build a compact cmux workspace title for a standing goal."""
    safe_prefix = str(prefix or "")
    try:
        limit = int(max_chars or _DEFAULT_MAX_CHARS)
    except Exception:
        limit = _DEFAULT_MAX_CHARS
    limit = max(8, limit)

    title = f"{safe_prefix}{_compact_goal_text(goal)}"
    if len(title) <= limit:
        return title

    ellipsis = "…"
    cutoff = max(1, limit - len(ellipsis))
    return title[:cutoff].rstrip() + ellipsis


def _cmux_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    section = (config or {}).get("cmux") if isinstance(config, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def rename_cmux_workspace_for_goal(
    goal: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> Optional[str]:
    """Rename the current cmux workspace for a newly-set goal.

    Returns the title when a rename command was attempted, otherwise ``None``.
    All failures are swallowed and logged at debug level because cmux is an
    optional host UI nicety, not part of Hermes goal semantics.
    """
    cfg = _cmux_config(config)
    if cfg.get("auto_rename_workspace_on_goal", True) is False:
        return None

    active_env = os.environ if env is None else env
    workspace_id = str(active_env.get("CMUX_WORKSPACE_ID") or "").strip()
    if not workspace_id:
        return None

    title = build_goal_workspace_title(
        goal,
        max_chars=cfg.get("goal_title_max_chars", _DEFAULT_MAX_CHARS),
        prefix=cfg.get("goal_title_prefix", _DEFAULT_PREFIX),
    )
    cmd = ["cmux", "rename-workspace", "--workspace", workspace_id, title]
    call = runner or subprocess.run
    try:
        call(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.debug("cmux workspace rename for goal failed: %s", exc)
        return None
    return title
