"""Formatting helpers for `hermes sessions search`."""

from __future__ import annotations

from typing import Any


_HIGHLIGHT_START = ">>>"
_HIGHLIGHT_END = "<<<"


def _single_line(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _clean_snippet(snippet: Any) -> str:
    text = str(snippet or "")
    text = text.replace(_HIGHLIGHT_START, "").replace(_HIGHLIGHT_END, "")
    return _single_line(text)


def format_session_search_results(payload: dict[str, Any], *, query: str) -> str:
    """Render the session_search discovery payload for terminal users."""
    if not payload.get("success"):
        error = payload.get("error") or payload.get("message") or "session search failed"
        return f"Error: {error}"

    results = payload.get("results") or []
    if not results:
        return (
            f'No sessions matched "{query}".\n'
            "Tip: session search uses FTS5 syntax, so try fewer terms, an exact "
            'phrase like "auth refactor", OR, NOT, or a prefix such as deploy*. '
            "Use `hermes sessions list` to browse recent sessions."
        )

    count = int(payload.get("count") or len(results))
    lines = [f'Found {count} matching session(s) for "{query}":']
    for index, hit in enumerate(results, start=1):
        session_id = str(hit.get("session_id") or "").strip()
        title = _single_line(hit.get("title") or "(untitled)", limit=80)
        source = str(hit.get("source") or "unknown")
        when = str(hit.get("when") or "unknown")
        role = str(hit.get("matched_role") or "message")
        message_id = hit.get("match_message_id")
        snippet = _clean_snippet(hit.get("snippet"))

        lines.append(f"{index}. {title}")
        lines.append(f"   id: {session_id}  source: {source}  when: {when}")
        if message_id is not None:
            lines.append(f"   match: {role} message {message_id}")
        else:
            lines.append(f"   match: {role}")
        if snippet:
            lines.append(f"   snippet: {snippet}")
        if session_id:
            lines.append(f"   resume: hermes --resume {session_id}")
    return "\n".join(lines)
