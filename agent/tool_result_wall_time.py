"""Model-only wall-time header on tool results.

The executor measures every real tool call's duration but only forwarded it to hooks,
telemetry and the UI; the model could not tell a 30 s command from an instant one. The
measured duration is persisted on the tool message as ``duration_ms`` (its own session-DB
column, never a wire field) and ``build_api_messages`` prepends ``Wall time: N.NNN seconds``
to the MODEL-BOUND copy only.

Persisted content stays the tool's raw output on purpose: the loop guardrails key on
byte-identical result streaks and stub duplicate results, JSON-shaped results are parsed by
error classifiers, the compression summariser reads raw content, and every UI surface
renders the unchanged text. The header is deterministic from the stored number, so replayed
requests (next iteration, resume) send identical bytes and the prompt-cache prefix stays
stable. Synthetic results (blocked calls, parse errors, interrupt backfills) carry no
duration and therefore no header.

Port of MoonshotAI/kimi-code#3966 (itself after codex's ``Wall time`` response payload).
"""

from __future__ import annotations

from typing import Any

WALL_TIME_KEY = "duration_ms"


def wall_time_header(duration_ms: Any) -> str | None:
    """``Wall time: N.NNN seconds`` for a measured duration, else ``None``."""
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)) or duration_ms < 0:
        return None
    return f"Wall time: {duration_ms / 1000:.3f} seconds"


def prepend_wall_time(api_msg: Any, duration_ms: Any) -> None:
    """Prepend the header to a wire copy's content (string, or the first text part of a list)."""
    header = wall_time_header(duration_ms)
    if header is None:
        return
    content = api_msg.get("content")
    if isinstance(content, str):
        api_msg["content"] = f"{header}\n{content}" if content else header
        return
    if isinstance(content, list):
        first = content[0] if content else None
        if isinstance(first, dict) and first.get("type") == "text" and isinstance(first.get("text"), str):
            api_msg["content"] = [{**first, "text": f"{header}\n{first['text']}"}, *content[1:]]
        else:
            api_msg["content"] = [{"type": "text", "text": header}, *content]
