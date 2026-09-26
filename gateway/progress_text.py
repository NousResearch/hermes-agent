"""Plain-language progress text shared by the Slack task card and the long-running heartbeat.

Only curated, low-risk previews are shown (search queries, web pages, skill names). Commands,
file paths and raw arguments never appear, so progress stays readable and safe in shared channels.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Tools whose preview is safe and useful to show after the verb.
_SAFE_PREVIEW_TOOLS = frozenset({"web_search", "web_extract", "browser_navigate", "skill_view"})
_STEP_MAX = 80


def _compact(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _mcp_label(tool_name: str) -> Optional[str]:
    # mcp__<server>__<tool> -> "Using <Server>"
    parts = tool_name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp" and parts[1]:
        server = parts[1].replace("_", " ").replace("-", " ").strip()
        return f"Using {server.title()}" if server else None
    return None


def friendly_step(tool_name: Any, preview: Any = None) -> str:
    """One plain-language line for a tool step, e.g. "Searching the web for pump specs"."""
    name = str(tool_name or "").strip()
    if not name:
        return "Working"
    from agent.display import get_tool_verb, tool_verb_connector, verb_drops_preview
    verb = get_tool_verb(name)
    if verb:
        shown = _compact(preview, 50) if (preview and name in _SAFE_PREVIEW_TOOLS
                                          and not verb_drops_preview(name)) else ""
        return _compact(f"{verb}{tool_verb_connector(name)}{shown}" if shown else verb, _STEP_MAX)
    return _mcp_label(name) or _compact(f"Using {name.replace('_', ' ')}", _STEP_MAX)


# Extra heartbeat lines from plugins (e.g. the KC subagent indicator). A provider receives
# chat_id and the turn's message ids and returns a list of lines (or nothing).
_providers: List[Callable[..., Optional[Iterable[str]]]] = []
_providers_lock = threading.Lock()


def register_progress_line_provider(fn: Callable[..., Optional[Iterable[str]]]) -> None:
    with _providers_lock:
        if fn not in _providers:
            _providers.append(fn)


def extra_progress_lines(**turn: Any) -> List[str]:
    with _providers_lock:
        providers = list(_providers)
    lines: List[str] = []
    for fn in providers:
        try:
            lines.extend(str(line) for line in (fn(**turn) or ()) if line)
        except Exception:
            logger.debug("progress line provider failed", exc_info=True)
    return lines


def heartbeat_text(elapsed_min: int, current_tool: Any = None, extra: Iterable[str] = ()) -> str:
    """The 3-minute check-in: elapsed time, what the agent is doing now, plus plugin lines."""
    lines = [f"⏳ Still working · {elapsed_min} min"]
    if current_tool:
        # current_tool may hold several parallel tool names joined by commas.
        first = str(current_tool).split(",")[0].strip()
        if first and first != "_thinking":
            lines.append(f"Now: {friendly_step(first)}")
    lines.extend(extra)
    return "\n".join(lines)
