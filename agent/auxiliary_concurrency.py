"""Per-task concurrency limit for auxiliary LLM calls (``auxiliary.<task>.max_concurrency``, #23324)."""

from __future__ import annotations

from typing import Optional


def _get_task_max_concurrency(task: Optional[str]) -> Optional[int]:
    """``auxiliary.<task>.max_concurrency`` as a positive int, or None. Vision uses this key for
    its encode/resize CPU pool; its LLM calls stay concurrent."""
    if not task or task == "vision":
        return None
    from agent.auxiliary_client import _get_auxiliary_task_config
    try:
        value = int(_get_auxiliary_task_config(task).get("max_concurrency"))
    except (TypeError, ValueError):  # missing (None) or malformed
        return None
    return value if value > 0 else None
