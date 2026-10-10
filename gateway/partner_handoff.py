"""Partner handoff delivery notices.

A handoff is created in one conversation and delivered into another. Delivery
order is not creation order (concurrent sessions are independent writers), so
the delivered text must carry the creation timestamp beside the id — otherwise
the receiver cannot tell which of two handoffs is newer when one supersedes
the other.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union


def format_handoff_created(ts: float | str | datetime | None = None) -> str | None:
    """Creation timestamp for a handoff header, as ``YYYY-MM-DDTHH:MM:SS.mmmZ``.

    Accepts epoch seconds (``int``/``float`` or a numeric string), a ``datetime``,
    or an ISO-8601 string, all normalized to constant-width milliseconds so
    lexicographic order matches creation order. A non-empty string in no
    recognized shape is passed through stripped (documented legacy behavior);
    ``None``/empty/uninterpretable input returns ``None`` so callers can omit
    the ``created`` segment.
    """
    if ts is None:
        return None
    if isinstance(ts, bool):
        return None
    if isinstance(ts, (int, float)):
        try:
            epoch = float(ts)
        except (TypeError, ValueError):
            return None
        return _format_epoch(epoch)
    if isinstance(ts, datetime):
        return _format_datetime(ts)
    if isinstance(ts, str):
        text = ts.strip()
        if not text:
            return None
        try:
            return _format_epoch(float(text))
        except (TypeError, ValueError, OverflowError, OSError):
            pass
        try:
            iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
            return _format_datetime(datetime.fromisoformat(iso))
        except (TypeError, ValueError):
            return text
    return None


def _format_epoch(epoch: float) -> str | None:
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return _format_datetime(dt)


def _format_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _short_id(handoff_id: str) -> str:
    return str(handoff_id or "")[:8] or "unknown"


def _who(requester: str, partner: str) -> str:
    if isinstance(requester, str) and requester.strip():
        return requester.strip()
    if isinstance(partner, str) and partner.strip():
        return partner.strip()
    return "someone"


def _header(handoff_id: str, requester: str, partner: str, ts: float | str | datetime | None = None) -> str:
    short = _short_id(handoff_id)
    who = _who(requester, partner)
    created = format_handoff_created(ts)
    if created:
        return f"[Handoff {short} · created {created} · from your conversation with {who}]"
    return f"[Handoff {short} · from your conversation with {who}]"


def human_notice(
    handoff_id: str,
    requester: str,
    partner: str,
    intent: str,
    ts: float | str | datetime | None = None,
) -> str:
    """User-visible handoff text: header (id + creation time) plus intent."""
    header = _header(handoff_id, requester, partner, ts)
    return f"{header}\n{intent or ''}\n"


def agent_notice(
    handoff_id: str,
    requester: str,
    partner: str,
    intent: str,
    ts: float | str | datetime | None = None,
) -> str:
    """Agent-facing handoff text: same ordering signal (id + creation time)."""
    header = _header(handoff_id, requester, partner, ts)
    return f"{header}\n{intent or ''}\n"
