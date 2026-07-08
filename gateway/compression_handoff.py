"""Default-disabled handoff artifact generation for compression boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from hermes_cli.config import cfg_get, get_hermes_home
from utils import is_truthy_value


_DEFAULT_MIN_COMPRESSION_COUNT = 2
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class CompressionHandoffResult:
    """Result of a compression handoff write."""

    path: Path
    notice: str
    notify: bool


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _sanitize_component(value: Any, default: str) -> str:
    text = str(value or "").strip() or default
    text = _SAFE_COMPONENT_RE.sub("-", text).strip(".-_")
    return text[:80] or default


def compression_handoff_enabled(config: dict[str, Any] | None) -> bool:
    """Return whether compression handoff artifacts are enabled."""

    return is_truthy_value(
        cfg_get(config, "compression", "handoff", "enabled", default=False),
        default=False,
    )


def _handoff_output_dir(
    config: dict[str, Any] | None,
    *,
    hermes_home: Path | None = None,
) -> Path:
    configured = cfg_get(config, "compression", "handoff", "output_dir", default="")
    home = hermes_home or get_hermes_home()
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = home / path
        return path
    return home / "handoffs" / "compression"


def should_write_compression_handoff(
    event_type: str,
    context: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> bool:
    """Gate compression handoff generation.

    The feature is default-disabled and only reacts to successful compression
    lifecycle events that have crossed the configured repeated-compression
    threshold.
    """

    if event_type != "session:compress":
        return False
    if not compression_handoff_enabled(config):
        return False
    context = context or {}
    threshold = _coerce_positive_int(
        cfg_get(
            config,
            "compression",
            "handoff",
            "min_compression_count",
            default=_DEFAULT_MIN_COMPRESSION_COUNT,
        ),
        _DEFAULT_MIN_COMPRESSION_COUNT,
    )
    count = _coerce_positive_int(context.get("compression_count"), 0)
    return count >= threshold


def render_compression_handoff_artifact(
    context: dict[str, Any],
    *,
    generated_at: datetime | None = None,
) -> str:
    """Render a deterministic metadata-first handoff artifact."""

    ts = generated_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    generated = ts.astimezone(timezone.utc).isoformat(timespec="seconds")
    platform = str(context.get("platform") or "").strip() or "unknown"
    session_id = str(context.get("session_id") or "").strip() or "unknown"
    old_session_id = str(context.get("old_session_id") or "").strip()
    compression_count = str(context.get("compression_count") or "unknown")
    in_place = bool(context.get("in_place"))

    old_line = old_session_id or "same session / unavailable"
    boundary = "in-place compaction" if in_place else "session rotation"

    return f"""# Compression handoff — {session_id}

Generated: {generated}
Source: Hermes `session:compress` lifecycle event
Mutation status: GENERATED_LOCAL_ARTIFACT_ONLY

## Boundary

- Platform: {platform}
- Current session id: `{session_id}`
- Previous session id: `{old_line}`
- Boundary mode: {boundary}
- Compression count: {compression_count}

## What happened

Hermes compacted this conversation after repeated context pressure. This file is a deterministic handoff seed generated from compression-boundary metadata. It is **not** a full transcript audit and does not dump prior message bodies.

## Fresh-session seed prompt

Resume the work from session `{session_id}`. Treat this handoff as a compression-boundary marker, then inspect durable artifacts, repo state, and session history before continuing. Prefer a fresh `/new` session if the active thread feels stale.

## Operator note

If this was delivered through a messaging surface, the next practical action is to open a fresh session and point the agent at this artifact plus any project run summaries.
"""


def write_compression_handoff_artifact(
    context: dict[str, Any],
    config: dict[str, Any] | None,
    *,
    hermes_home: Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Write a compression handoff artifact and return its path."""

    ts = generated_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    session_id = _sanitize_component(context.get("session_id"), "session")
    count = _sanitize_component(context.get("compression_count"), "n")
    stamp = ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _handoff_output_dir(config, hermes_home=hermes_home)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stamp}-compression-{count}-{session_id}.md"
    path.write_text(
        render_compression_handoff_artifact(context, generated_at=ts),
        encoding="utf-8",
    )
    return path


def maybe_write_compression_handoff(
    event_type: str,
    context: dict[str, Any] | None,
    config: dict[str, Any] | None,
    *,
    hermes_home: Path | None = None,
    generated_at: datetime | None = None,
) -> CompressionHandoffResult | None:
    """Write a handoff artifact when the config/event gates pass."""

    if not should_write_compression_handoff(event_type, context, config):
        return None
    context = context or {}
    path = write_compression_handoff_artifact(
        context,
        config,
        hermes_home=hermes_home,
        generated_at=generated_at,
    )
    notify = is_truthy_value(
        cfg_get(config, "compression", "handoff", "notify", default=True),
        default=True,
    )
    count = context.get("compression_count") or "?"
    notice = (
        f"DONE — session compressed {count} times; handoff artifact: {path}. "
        "Suggested next step: start a fresh session seeded from that artifact."
    )
    return CompressionHandoffResult(path=path, notice=notice, notify=notify)
