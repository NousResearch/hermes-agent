"""Argument merging for pre-tool directives; callbacks still see the original payload."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def merge_pre_tool_modification(tool_name: str, original: Any, accumulated: dict | None,
                               partial: Any, modified_keys: set) -> dict | None:
    """Keep last-writer-wins merging, but report conflicting writes without argument values."""
    if not isinstance(partial, dict) or not partial:
        return accumulated
    previous = accumulated if accumulated is not None else (original if isinstance(original, dict) else {})
    conflicts = [key for key in partial if key in modified_keys and previous[key] != partial[key]]
    if conflicts:
        logger.warning(
            "pre_tool_call modify conflict for tool '%s' on argument key(s): %s; later hook wins. "
            "Each callback receives the original arguments, not previous modifications.",
            tool_name, ", ".join(str(key) for key in conflicts))
    modified_keys.update(partial)
    return {**previous, **partial}
