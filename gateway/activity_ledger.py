"""Realtime gateway activity ledger.

This module records small, redacted turn facts from gateway lifecycle hooks.
It is intentionally a sidecar JSONL writer: records are not memory, are not
loaded into prompts, and do not contain transcripts or raw tool output.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home
from hermes_time import now as hermes_now
from hermes_cli.config import cfg_get, read_raw_config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MAX_PREVIEW_CHARS = 500
MAX_PREVIEW_CHARS_CAP = 500

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b"
    r"(?P<key>api[_-]?key|token|secret|password|private[_ -]?key|client[_-]?secret|cookie)"
    r"\b\s*(?P<sep>[:=])\s*(?P<quote>['\"]?)"
    r"(?P<value>[^\s,'\"&;]+)(?P=quote)"
)
_BARE_BEARER_RE = re.compile(
    r"(?i)\b(bearer)\s+([A-Za-z0-9._~+/=-]{12,})"
)


@dataclass(frozen=True)
class ActivityLedgerSettings:
    enabled: bool = False
    capture_turns: bool = True
    max_preview_chars: int = DEFAULT_MAX_PREVIEW_CHARS


@dataclass
class _TurnAccumulator:
    session_id: str
    platform: str = ""
    message_preview: str = ""
    tool_names: list[str] = field(default_factory=list)


_turns: Dict[str, _TurnAccumulator] = {}
_lock = threading.Lock()


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    return bool(value)


def _coerce_preview_chars(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_PREVIEW_CHARS
    if parsed < 0:
        return DEFAULT_MAX_PREVIEW_CHARS
    return min(parsed, MAX_PREVIEW_CHARS_CAP)


def load_settings(config: Optional[Dict[str, Any]] = None) -> ActivityLedgerSettings:
    """Return activity ledger settings from raw config.yaml values."""
    cfg = read_raw_config() if config is None else config
    section = cfg_get(cfg, "activity_ledger", default={})
    if not isinstance(section, dict):
        section = {}
    return ActivityLedgerSettings(
        enabled=_coerce_bool(section.get("enabled"), False),
        capture_turns=_coerce_bool(section.get("capture_turns"), True),
        max_preview_chars=_coerce_preview_chars(
            section.get("max_preview_chars", DEFAULT_MAX_PREVIEW_CHARS)
        ),
    )


def redact_activity_text(text: Any) -> str:
    """Apply forced secret redaction plus ledger-local conservative patterns."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""

    redacted = redact_sensitive_text(text, force=True)

    def _assignment_sub(match: re.Match[str]) -> str:
        quote = match.group("quote") or ""
        return (
            f"{match.group('key')}{match.group('sep')}"
            f"{quote}***{quote}"
        )

    redacted = _SENSITIVE_ASSIGNMENT_RE.sub(_assignment_sub, redacted)
    redacted = _BARE_BEARER_RE.sub(lambda m: f"{m.group(1)} ***", redacted)
    return redacted


def make_preview(text: Any, max_chars: int = DEFAULT_MAX_PREVIEW_CHARS) -> str:
    """Return a single-line redacted preview capped at 500 chars."""
    limit = _coerce_preview_chars(max_chars)
    preview = redact_activity_text(text)
    preview = re.sub(r"\s+", " ", preview).strip()
    if limit == 0:
        return ""
    if len(preview) <= limit:
        return preview
    if limit <= 3:
        return preview[:limit]
    return preview[: limit - 3].rstrip() + "..."


def _turn_key(context: Dict[str, Any]) -> str:
    session_id = str(context.get("session_id") or "").strip()
    if session_id:
        return session_id
    platform = str(context.get("platform") or "").strip()
    user_id = str(context.get("user_id") or "").strip()
    return f"{platform}:{user_id}" if platform or user_id else ""


def _tool_names_from_context(context: Dict[str, Any]) -> list[str]:
    names: list[str] = []

    raw_names = context.get("tool_names")
    if isinstance(raw_names, (list, tuple, set)):
        for item in raw_names:
            name = str(item or "").strip()
            if name:
                names.append(name)

    raw_tools = context.get("tools")
    if isinstance(raw_tools, (list, tuple)):
        for item in raw_tools:
            name = ""
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("tool_name") or "").strip()
            elif item is not None:
                name = str(item).strip()
            if name:
                names.append(name)

    return names


def _extend_tool_names(acc: _TurnAccumulator, names: Iterable[str]) -> None:
    seen = set(acc.tool_names)
    for name in names:
        clean = str(name or "").strip()
        if clean and clean not in seen:
            acc.tool_names.append(clean)
            seen.add(clean)


def _ledger_file(kind: str, when: datetime) -> Path:
    return get_hermes_home() / "activity-ledger" / when.date().isoformat() / f"{kind}.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def _record_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{secrets.token_hex(4)}"


def _build_turn_record(
    acc: _TurnAccumulator,
    *,
    response_preview: str,
    when: datetime,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": _record_id("turn"),
        "time": when.isoformat(),
        "date": when.date().isoformat(),
        "type": "turn",
        "session_id": acc.session_id,
        "platform": acc.platform,
        "message_preview": acc.message_preview,
        "response_preview": response_preview,
        "tool_names": list(acc.tool_names),
        "reportable": "unknown",
        "visibility": "private",
    }


def _handle_agent_start(context: Dict[str, Any], settings: ActivityLedgerSettings) -> None:
    key = _turn_key(context)
    session_id = str(context.get("session_id") or "").strip()
    if not key or not session_id:
        return

    acc = _TurnAccumulator(
        session_id=session_id,
        platform=str(context.get("platform") or "").strip(),
        message_preview=make_preview(context.get("message"), settings.max_preview_chars),
    )
    with _lock:
        _turns[key] = acc


def _handle_agent_step(context: Dict[str, Any]) -> None:
    key = _turn_key(context)
    if not key:
        return
    session_id = str(context.get("session_id") or "").strip()
    with _lock:
        acc = _turns.get(key)
        if acc is None and session_id:
            acc = _TurnAccumulator(
                session_id=session_id,
                platform=str(context.get("platform") or "").strip(),
            )
            _turns[key] = acc
        if acc is not None:
            _extend_tool_names(acc, _tool_names_from_context(context))


def _handle_agent_end(context: Dict[str, Any], settings: ActivityLedgerSettings) -> None:
    key = _turn_key(context)
    session_id = str(context.get("session_id") or "").strip()
    if not key or not session_id:
        return

    with _lock:
        acc = _turns.pop(key, None)
    if acc is None:
        acc = _TurnAccumulator(
            session_id=session_id,
            platform=str(context.get("platform") or "").strip(),
            message_preview=make_preview(context.get("message"), settings.max_preview_chars),
        )

    if not acc.platform:
        acc.platform = str(context.get("platform") or "").strip()
    if not acc.message_preview:
        acc.message_preview = make_preview(context.get("message"), settings.max_preview_chars)
    _extend_tool_names(acc, _tool_names_from_context(context))

    when = hermes_now()
    record = _build_turn_record(
        acc,
        response_preview=make_preview(context.get("response"), settings.max_preview_chars),
        when=when,
    )
    _append_jsonl(_ledger_file("turns", when), record)


def _discard_turn(context: Dict[str, Any]) -> None:
    key = _turn_key(context)
    if key:
        with _lock:
            _turns.pop(key, None)


def handle_gateway_hook(event_type: str, context: Optional[Dict[str, Any]]) -> None:
    """Built-in gateway hook entry point.

    All exceptions are swallowed so activity-ledger failures never affect the
    user-visible turn.
    """
    try:
        ctx = context if isinstance(context, dict) else {}
        settings = load_settings()
        if not settings.enabled or not settings.capture_turns:
            if event_type in {"agent:start", "agent:end"}:
                _discard_turn(ctx)
            return

        if event_type == "agent:start":
            _handle_agent_start(ctx, settings)
        elif event_type == "agent:step":
            _handle_agent_step(ctx)
        elif event_type == "agent:end":
            _handle_agent_end(ctx, settings)
    except Exception:
        logger.warning("activity ledger hook failed for %s", event_type, exc_info=True)


def _clear_state_for_tests() -> None:
    with _lock:
        _turns.clear()
