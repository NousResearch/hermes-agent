"""The one field set behind ``/status`` on every surface.

The CLI (``cli_session_mixin._show_session_status``), the messaging gateway
(``gateway/slash_commands_status._handle_status_command``) and the TUI backend
(``tui_gateway/methods_session`` ``session.status``) all report the same session facts —
id, home path, title, model route, created / last-activity stamps, lifetime tokens, running
flag. Each used to derive them independently (three ``getattr(agent, "model")`` fallback
chains, three ``updated_at`` candidate scans, three timestamp formats to keep aligned).
They now all call :func:`build_status_fields`; a surface only adds its own header, labels
(the gateway translates through ``t("gateway.status.*")``) and surface-specific extras.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

STATUS_STAMP = "%Y-%m-%d %H:%M"

UNKNOWN_MODEL = "(unknown)"
UNKNOWN_PROVIDER = "unknown"

# Newest-first: a row carries at most one of these, depending on which writer last touched it.
_LAST_ACTIVITY_FIELDS = ("updated_at", "last_updated_at", "last_activity_at")

# English labels shared by the CLI and TUI renderers (the gateway has its own i18n catalog).
STATUS_LABELS = {
    "session_id": "Session ID",
    "path": "Path",
    "title": "Title",
    "model": "Model",
    "created": "Created",
    "last_activity": "Last Activity",
    "tokens": "Tokens",
    "agent_running": "Agent Running",
}


def timestamp_or(value: Any, default: datetime | None) -> datetime | None:
    """``datetime.fromtimestamp(value)`` or *default* when the value is missing/unparseable."""
    if not value:
        return default
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OverflowError, OSError):
        return default


def session_timestamps(meta: dict, created_fallback: datetime | None = None) -> tuple[datetime, datetime]:
    """``(created, last_activity)`` from a SessionDB row; last activity falls back to created."""
    created = timestamp_or(meta.get("started_at"), created_fallback) or datetime.now()
    updated = next(
        (stamp for stamp in (timestamp_or(meta.get(f), None) for f in _LAST_ACTIVITY_FIELDS) if stamp),
        created,
    )
    return created, updated


def build_status_fields(
    session_id: str,
    agent: Any,
    meta: dict | None,
    *,
    title: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    created: datetime | None = None,
    last_activity: datetime | None = None,
    created_fallback: datetime | None = None,
    tokens: int | None = None,
    agent_running: bool = False,
) -> dict[str, Any]:
    """Common ``/status`` facts, pre-formatted for display.

    ``agent`` wins for model / provider / tokens; the keyword values are the surface's fallback
    (CLI ``self.model``, TUI metadata mirror, gateway's resolved route) and are also what a
    surface passes when it has no live agent. ``created`` / ``last_activity`` override the
    ``meta`` row scan for surfaces whose session store is authoritative (gateway SessionEntry).
    """
    from hermes_constants import display_hermes_home

    meta = meta or {}
    if created is None or last_activity is None:
        row_created, row_updated = session_timestamps(meta, created_fallback)
        created = created or row_created
        last_activity = last_activity or row_updated
    if tokens is None:
        tokens = getattr(agent, "session_total_tokens", 0) or 0
    row_title = meta.get("title") if title is None else title
    return {
        "session_id": str(session_id or ""),
        "path": display_hermes_home(),
        "title": (row_title or "").strip(),
        "model": getattr(agent, "model", None) or model or "",
        "provider": getattr(agent, "provider", None) or provider or "",
        "created": created.strftime(STATUS_STAMP),
        "last_activity": last_activity.strftime(STATUS_STAMP),
        "tokens": f"{int(tokens or 0):,}",
        "agent_running": bool(agent_running),
    }


def status_lines(fields: dict[str, Any], *keys: str) -> list[str]:
    """``Label: value`` lines for *keys* in order (English surfaces). An empty title is skipped;
    an unresolved model/provider shows a placeholder rather than an empty parenthesis."""
    lines: list[str] = []
    for key in keys:
        value = fields[key]
        if key == "title" and not value:
            continue
        if key == "model":
            value = f"{fields['model'] or UNKNOWN_MODEL} ({fields['provider'] or UNKNOWN_PROVIDER})"
        elif key == "agent_running":
            value = "Yes" if value else "No"
        lines.append(f"{STATUS_LABELS[key]}: {value}")
    return lines
