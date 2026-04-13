import logging
import json
import re
import os
from datetime import datetime
from pathlib import Path
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Tools that must never run concurrently (interactive / user-facing).
# When any of these appear in a batch, we fall back to sequential execution.
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Read-only tools with no shared mutable session state.
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "honcho_context",
    "honcho_profile",
    "honcho_search",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

# File tools can run concurrently when they target independent paths.
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# Maximum number of concurrent worker threads for parallel tool execution.
_MAX_TOOL_WORKERS = 8

# Patterns that indicate a terminal command may modify/delete files.
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# Output redirects that overwrite files (> but not >>)
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')

def _is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False

def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve(); the file may not exist yet.
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # Empty paths shouldn't reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]

def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True

# Threshold at which tool results are saved to a file instead of kept inline.
_LARGE_RESULT_CHARS = 100_000

# How many characters of the original result to include as an inline preview
_LARGE_RESULT_PREVIEW_CHARS = 1_500


def _save_oversized_tool_result(function_name: str, function_result: str) -> str:
    """Replace oversized tool results with a file reference + preview."""
    original_len = len(function_result)
    if original_len <= _LARGE_RESULT_CHARS:
        return function_result

    try:
        response_dir = os.path.join(get_hermes_home(), "cache", "tool_responses")
        os.makedirs(response_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = re.sub(r"[^\w\-]", "_", function_name)[:40]
        filename = f"{safe_name}_{timestamp}.txt"
        filepath = os.path.join(response_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(function_result)

        preview = function_result[:_LARGE_RESULT_PREVIEW_CHARS]
        return (
            f"{preview}\n\n"
            f"[Large tool response: {original_len:,} characters total — "
            f"only the first {_LARGE_RESULT_PREVIEW_CHARS:,} shown above. "
            f"Full output saved to: {filepath}\n"
            f"Use read_file or search_files on that path to access the rest.]"
        )
    except Exception as exc:
        logger.warning("Failed to save large tool result to file: %s", exc)
        return (
            function_result[:_LARGE_RESULT_CHARS]
            + f"\n\n[Truncated: tool response was {original_len:,} chars, "
            f"exceeding the {_LARGE_RESULT_CHARS:,} char limit. "
            f"File save failed: {exc}]"
        )
