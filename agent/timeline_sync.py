from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class ElapsedDescription:
    seconds: float
    text: str
    bucket: str
    guidance: str


def coerce_epoch_seconds(value: Any) -> Optional[float]:
    """Best-effort conversion to epoch seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return float(value.timestamp())
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        if rem:
            return f"{minutes} minutes {rem} seconds"
        return f"{minutes} minutes"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        if minutes:
            return f"{hours} hours {minutes} minutes"
        return f"{hours} hours"
    days, hours = divmod(hours, 24)
    if hours:
        return f"{days} days {hours} hours"
    return f"{days} days"


def describe_elapsed(seconds: float | int | None) -> ElapsedDescription:
    """Classify an elapsed time gap for language guidance."""
    if seconds is None:
        seconds = 0
    seconds_f = max(0.0, float(seconds))
    if seconds_f < 120:
        bucket = "immediate"
        guidance = "Treat this as a live continuation."
    elif seconds_f < 30 * 60:
        bucket = "recent"
        guidance = "Treat this as recent, but do not assume it happened just now."
    elif seconds_f < 6 * 3600:
        bucket = "earlier_today"
        guidance = "Treat this as earlier today; state the gap if timing matters."
    elif seconds_f < 24 * 3600:
        bucket = "long_gap_today"
        guidance = "Treat this as a long same-day gap; avoid saying it just happened."
    else:
        bucket = "older"
        guidance = "Treat this as a prior-day or older context unless the user says otherwise."
    return ElapsedDescription(
        seconds=seconds_f,
        text=_format_duration(seconds_f),
        bucket=bucket,
        guidance=guidance,
    )


def _default_db_path() -> Path:
    override = os.getenv("HERMES_TIMELINE_SYNC_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return get_hermes_home() / "state.db"


def _decode_preview(content: Any, max_chars: int = 120) -> str:
    if content is None:
        return ""
    text = str(content)
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                parts: list[str] = []
                for item in parsed:
                    if isinstance(item, dict):
                        value = item.get("text") or item.get("content") or ""
                        if value:
                            parts.append(str(value))
                    elif item:
                        parts.append(str(item))
                text = " ".join(parts) or text
            elif isinstance(parsed, dict):
                text = str(parsed.get("text") or parsed.get("content") or text)
        except Exception:
            pass
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    text = re.sub(r"\[Image attached at: [^\]]+\]", "[Image attached]", text)
    return text[: max(0, max_chars - 1)] + "…" if len(text) > max_chars else text


def _is_synthetic_gateway_preview(preview: str) -> bool:
    """Return True for gateway-maintenance text that should not shape time sense."""
    text = " ".join(str(preview or "").strip().split()).lower()
    if not text:
        return True
    synthetic_prefixes = (
        "[system note:",
        "[your active task list was preserved across context compression]",
        "[context compaction",
        "[timeline sync]",
        "[rhythm context]",
        "[expression context]",
    )
    if any(text.startswith(prefix) for prefix in synthetic_prefixes):
        return True
    synthetic_markers = (
        "[thread context — prior messages in this thread",
        "[thread context - prior messages in this thread",
    )
    return any(marker in text for marker in synthetic_markers)


def _collapse_replay_clusters(
    events: list[dict[str, Any]],
    *,
    max_span_seconds: float = 1.0,
    min_cluster_size: int = 3,
) -> list[dict[str, Any]]:
    """Collapse same-source replay bursts caused by compression/session restore."""
    if len(events) < min_cluster_size:
        return events

    ordered = sorted(events, key=lambda event: float(event.get("timestamp") or 0.0), reverse=True)
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        first = ordered[index]
        cluster = [first]
        first_ts = float(first.get("timestamp") or 0.0)
        source = str(first.get("source") or "")
        role = str(first.get("role") or "")
        next_index = index + 1
        while next_index < len(ordered):
            candidate = ordered[next_index]
            candidate_ts = float(candidate.get("timestamp") or 0.0)
            if source != str(candidate.get("source") or "") or role != str(candidate.get("role") or ""):
                break
            if abs(first_ts - candidate_ts) > max_span_seconds:
                break
            cluster.append(candidate)
            next_index += 1

        if len(cluster) >= min_cluster_size:
            result.append(cluster[0])
        else:
            result.extend(cluster)
        index = next_index
    return result


def get_last_user_message_time(*, db_path: str | Path | None = None, session_id: str) -> Optional[float]:
    if not session_id:
        return None
    path = Path(db_path) if db_path else _default_db_path()
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute(
            """
            SELECT timestamp FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return None
    return float(row[0]) if row else None


def get_recent_events(
    *,
    db_path: str | Path | None = None,
    since_ts: float,
    limit: int = 8,
    exclude_session_id: str | None = None,
    roles: Iterable[str] = ("user",),
) -> list[dict[str, Any]]:
    """Return recent cross-session events ordered by real timestamp."""
    path = Path(db_path) if db_path else _default_db_path()
    if not path.exists() or limit <= 0:
        return []
    role_values = tuple(roles) or ("user",)
    placeholders = ",".join("?" for _ in role_values)
    params: list[Any] = [float(since_ts), *role_values]
    exclude_sql = ""
    if exclude_session_id:
        exclude_sql = "AND m.session_id != ?"
        params.append(exclude_session_id)
    query_limit = max(int(limit), int(limit) * 4)
    params.append(query_limit)
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT m.timestamp, s.source, m.session_id, m.role, m.content, s.title
            FROM messages m
            LEFT JOIN sessions s ON s.id = m.session_id
            WHERE m.timestamp >= ?
              AND m.role IN ({placeholders})
              {exclude_sql}
            ORDER BY m.timestamp DESC, m.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    events: list[dict[str, Any]] = []
    for row in rows:
        preview = _decode_preview(row["content"])
        if _is_synthetic_gateway_preview(preview):
            continue
        events.append(
            {
                "timestamp": float(row["timestamp"]),
                "source": row["source"] or "unknown",
                "session_id": row["session_id"],
                "role": row["role"],
                "preview": preview,
                "title": row["title"] or "",
            }
        )
    return _collapse_replay_clusters(events)[: int(limit)]


def _format_time(ts: float, now: datetime) -> str:
    tz = now.tzinfo or datetime.now().astimezone().tzinfo
    return datetime.fromtimestamp(ts, tz=tz).strftime("%H:%M")


def _coerce_now(now: Any = None) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=datetime.now().astimezone().tzinfo)
    epoch = coerce_epoch_seconds(now)
    if epoch is None:
        return datetime.now().astimezone()
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone()


def build_timeline_context(
    *,
    db_path: str | Path | None = None,
    now: Any = None,
    session_id: str = "",
    platform: str = "",
    recent_window_minutes: int = 30,
    max_events: int = 8,
    include_other_platforms: bool = True,
) -> str:
    """Build an ephemeral per-turn timeline context block."""
    current = _coerce_now(now)
    path = Path(db_path) if db_path else _default_db_path()
    current_ts = current.timestamp()
    last_user_ts = get_last_user_message_time(db_path=path, session_id=session_id)
    elapsed = describe_elapsed(current_ts - last_user_ts) if last_user_ts else None

    lines = [
        "[Timeline sync]",
        f"Current real time: {current.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}.",
    ]
    if platform:
        lines.append(f"Current platform: {platform}.")
    if session_id:
        lines.append(f"Current session: {session_id}.")
    if elapsed:
        lines.append(f"Elapsed since last user message in this session: {elapsed.text}.")
        lines.append(f"Interpretation: {elapsed.guidance}")
    else:
        lines.append("Elapsed since last user message in this session: unknown.")

    if include_other_platforms:
        since_ts = current_ts - max(0, int(recent_window_minutes)) * 60
        events = get_recent_events(
            db_path=path,
            since_ts=since_ts,
            limit=max_events,
            exclude_session_id=session_id or None,
        )
        if events:
            lines.append(f"Recent cross-platform events in the last {recent_window_minutes} minutes:")
            for event in events:
                source = str(event.get("source") or "unknown").upper()
                when = _format_time(float(event["timestamp"]), current)
                preview = event.get("preview") or "(empty)"
                lines.append(f"- {when} {source}: {preview}")
        else:
            lines.append(f"Recent cross-platform events in the last {recent_window_minutes} minutes: none found.")

    lines.append('Instruction: use real elapsed time when saying "just now", "earlier", or "last time". Avoid saying "just now" when the gap is not immediate.')
    lines.append("[/Timeline sync]")
    return "\n".join(lines)
