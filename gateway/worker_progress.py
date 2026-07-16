"""Structured gateway worker-progress contract and safe renderer."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit


WORKER_PROGRESS_EVENT = "worker.progress"
WORKER_PROGRESS_QUEUE_KIND = "__worker_progress__"

WORKER_PHASES = frozenset(
    {
        "preparing",
        "inspecting",
        "drafting",
        "editing",
        "validating",
        "visual_verification",
        "complete",
        "blocked",
    }
)
WORKER_STATUSES = frozenset({"active", "complete", "blocked"})
_DELEGATED_COMPLETE_STATUSES = frozenset({"complete", "completed", "ok", "success"})
_DELEGATED_BLOCKED_STATUSES = frozenset({"error", "failed", "timeout"})
_SECRET_QUERY_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "key",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer|password|secret|signature|token)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer|password|secret|token)\b\s*[:=]"
    r"|sk-[A-Za-z0-9_-]{12,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r")",
    re.IGNORECASE,
)

_PHASE_TEXT = {
    "preparing": "Preparing the workspace",
    "inspecting": "Inspecting the request",
    "drafting": "Drafting the change",
    "editing": "Editing files",
    "validating": "Validating the result",
    "visual_verification": "Checking the browser view",
    "complete": "Work complete",
    "blocked": "Blocked",
}


@dataclass(frozen=True)
class WorkerProgressNotice:
    """Trusted, bounded worker-progress state for human-facing gateways."""

    phase: str = "preparing"
    status: str = "active"
    preview_url: str | None = None


def _payload_mapping(args: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(args, Mapping):
        payload.update(args)
    progress = kwargs.get("progress")
    if isinstance(progress, Mapping):
        payload.update(progress)
    payload.update(kwargs)
    return payload


def _safe_token(value: Any, allowed: frozenset[str], default: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    return token if token in allowed else default


def sanitize_preview_url(value: Any) -> str | None:
    """Return a Slack-safe HTTPS preview URL, or None for untrusted input."""
    url = str(value or "").strip()
    if not url or len(url) > 512:
        return None
    if any(ch.isspace() or ch in "<>`" for ch in url):
        return None
    if _SECRET_VALUE_RE.search(url):
        return None
    try:
        parsed = urlsplit(url)
    except Exception:
        return None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    if parsed.fragment:
        return None
    if _SECRET_KEY_RE.search(parsed.path):
        return None
    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.strip().lower()
        if normalized_key in _SECRET_QUERY_KEYS or _SECRET_KEY_RE.search(key):
            return None
        if _SECRET_VALUE_RE.search(str(_value or "")):
            return None
    return url


def notice_from_worker_progress_event(
    event_type: str,
    *,
    args: Any = None,
    kwargs: Mapping[str, Any] | None = None,
) -> WorkerProgressNotice | None:
    """Parse trusted worker-progress events into a renderable notice.

    This intentionally ignores free-form text such as previews, task prompts,
    tool names, stdout, paths, and logs. The only accepted external data is a
    finite phase/status plus an optional sanitized HTTPS preview URL.
    """
    kwargs = kwargs or {}
    payload = _payload_mapping(args, kwargs)

    if event_type == WORKER_PROGRESS_EVENT:
        phase = _safe_token(payload.get("phase"), WORKER_PHASES, "preparing")
        status = _safe_token(payload.get("status"), WORKER_STATUSES, "active")
    elif event_type == "subagent.start":
        phase = "preparing"
        status = "active"
    elif event_type == "subagent.complete":
        delegated_status = (
            str(payload.get("status") or "").strip().lower().replace("-", "_")
        )
        if delegated_status in _DELEGATED_BLOCKED_STATUSES:
            phase = "blocked"
            status = "blocked"
        elif delegated_status in _DELEGATED_COMPLETE_STATUSES or not delegated_status:
            phase = "complete"
            status = "complete"
        else:
            phase = "complete"
            status = "complete"
    else:
        return None

    if phase in {"complete", "blocked"}:
        status = phase
    elif status in {"complete", "blocked"}:
        phase = status

    return WorkerProgressNotice(
        phase=phase,
        status=status,
        preview_url=sanitize_preview_url(payload.get("preview_url")),
    )


def render_worker_progress_notice(notice: WorkerProgressNotice) -> str:
    """Render a canned, human-readable Slack/gateway progress card."""
    phase_text = _PHASE_TEXT.get(notice.phase, _PHASE_TEXT["preparing"])
    if notice.status == "blocked":
        header = "⚠️ Worker progress"
    elif notice.status == "complete":
        header = "✅ Worker progress"
    else:
        header = "⏳ Worker progress"

    lines = [header, phase_text]
    if notice.preview_url and notice.status in {"complete", "blocked"}:
        lines.append(f"<{notice.preview_url}|Open preview>")
    return "\n".join(lines)
