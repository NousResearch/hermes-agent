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
from pathlib import Path
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


def _clean_str(value: Any) -> str:
    """``str(value).strip()`` for a possibly-missing wire value (``None`` → ``""``)."""
    return "" if value is None else str(value).strip()


def resolve_display_identity(
    *,
    agent: Any = None,
    mirror: dict | None = None,
    pending: dict | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    custom_resolver: Any = None,
) -> tuple[str, str]:
    """The ``(model, provider)`` every status surface reports for one session.

    One shared precedence — pending switch, metadata mirror, live agent, configured
    default — so the status bar and ``/status`` cannot disagree. ``custom_resolver``
    resolves the ``"custom"`` sentinel against the owning profile when a surface can
    reach it.
    """
    mirror = mirror or {}
    pending = pending or {}
    tiers = (
        (pending.get("display_model"), pending.get("display_provider")),
        (mirror.get("model"), mirror.get("provider")),
        (getattr(agent, "model", None), getattr(agent, "provider", None)),
        (configured_model, configured_provider),
    )
    model = provider = ""
    for tier_model, tier_provider in tiers:
        if _clean_str(tier_model):
            model = _clean_str(tier_model)
            provider = _clean_str(tier_provider)
            break
    if not model:
        return "", ""
    if not provider:
        for _, tier_provider in tiers:
            candidate = _clean_str(tier_provider)
            if candidate:
                provider = candidate
                break
    if provider == "custom" and "provider" not in mirror and custom_resolver is not None:
        resolved = custom_resolver(agent) or ""
        if resolved:
            provider = resolved
    return model, provider


def build_status_fields(
    session_id: str,
    agent: Any,
    meta: dict | None,
    *,
    title: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    mirror: dict | None = None,
    pending: dict | None = None,
    custom_resolver: Any = None,
    created: datetime | None = None,
    last_activity: datetime | None = None,
    created_fallback: datetime | None = None,
    tokens: int | None = None,
    agent_running: bool = False,
    home: str | Path | None = None,
) -> dict[str, Any]:
    """Common ``/status`` facts, pre-formatted for display.

    Model / provider come from :func:`resolve_display_identity`; ``model`` /
    ``provider`` are the configured seed (last tier). ``created`` / ``last_activity``
    override the ``meta`` row scan for surfaces whose session store is authoritative.
    ``home`` is an explicit owning-profile path for multiplexed surfaces.
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
    display_model, display_provider = resolve_display_identity(
        agent=agent, mirror=mirror, pending=pending, custom_resolver=custom_resolver,
        configured_model=model, configured_provider=provider)
    return {
        "session_id": str(session_id or ""),
        "path": display_hermes_home(Path(home) if home else None),
        "title": (row_title or "").strip(),
        "model": display_model,
        "provider": display_provider,
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
