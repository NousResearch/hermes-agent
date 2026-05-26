"""Privacy-safe orchestration routing telemetry.

This module owns the append-only JSONL surface used by coordinator/orchestrator
paths (delegate_task, kanban_create, and future routers) to record route choices
without preserving prompt/task content or credentials.

Records are metadata-only. Content-bearing fields (goal, context, prompts,
messages, bodies, summaries, outputs, tool args) are excluded, and credential-
shaped fields are redacted before serialization. Writes are opt-in via
``logging.orchestration_telemetry.enabled`` or
``HERMES_ORCHESTRATION_TELEMETRY=1`` and best-effort so they never affect the
user-facing orchestration path.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - platform-dependent import
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

try:  # pragma: no cover - platform-dependent import
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - POSIX fallback
    msvcrt = None  # type: ignore

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TELEMETRY_FILENAME = "orchestration-routing.jsonl"
_MAX_STRING_CHARS = 512
_MAX_LIST_ITEMS = 80
_MAX_DICT_ITEMS = 80
_MAX_LOG_BYTES = 50 * 1024 * 1024
_MAX_ROTATIONS = 3

_WRITE_LOCK = threading.Lock()
_SKIP = object()

# Exact/suffix content-bearing keys. Keep this intentionally conservative: if a
# value could carry user task text, prompts, model outputs, or raw tool args, do
# not persist it in the routing log.
_CONTENT_KEYS = {
    "args",
    "arguments",
    "body",
    "child_summary",
    "context",
    "description",
    "final_response",
    "goal",
    "message",
    "messages",
    "metadata",
    "output",
    "preview",
    "prompt",
    "response",
    "result",
    "summary",
    "system_prompt",
    "task",
    "title",
    "tool_args",
    "user_message",
}
_CONTENT_SUFFIXES = (
    "_args",
    "_arguments",
    "_body",
    "_context",
    "_description",
    "_message",
    "_messages",
    "_output",
    "_preview",
    "_prompt",
    "_response",
    "_result",
    "_summary",
    "_task",
    "_title",
)

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|auth[_-]?token|base[_-]?url|cookie|credential|"
    r"dsn|oauth|password|private[_-]?key|secret|token|connection[_-]?string)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)(api[_-]?key=)[^&\s]+"),
    re.compile(r"(?i)(token=)[^&\s]+"),
    re.compile(r"(?i)(password=)[^&\s]+"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"),
)


_TRUE_VALUES = {"1", "true", "on", "yes", "enabled"}
_FALSE_VALUES = {"0", "false", "off", "no", "disabled"}


def _telemetry_enabled() -> bool:
    """Return whether route telemetry should be written.

    Environment wins for process-local smoke tests and launchd/systemd wrappers.
    Otherwise profile config can enable normal orchestrator sessions with:

    ``logging.orchestration_telemetry.enabled: true``
    """
    raw = os.environ.get("HERMES_ORCHESTRATION_TELEMETRY", "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        logging_cfg = cfg.get("logging") if isinstance(cfg, dict) else {}
        telemetry_cfg = (
            logging_cfg.get("orchestration_telemetry")
            if isinstance(logging_cfg, dict)
            else {}
        )
        if isinstance(telemetry_cfg, dict):
            return bool(telemetry_cfg.get("enabled", False))
    except Exception:
        logger.debug("orchestration telemetry config check failed", exc_info=True)
    return False


def telemetry_path(path: Optional[os.PathLike[str] | str] = None) -> Path:
    """Return the JSONL path for orchestration telemetry.

    Defaults to ``<HERMES_HOME>/logs/orchestration-routing.jsonl`` and resolves
    HERMES_HOME at call time so profile/context-local overrides are honored.
    """
    if path is not None:
        return Path(path)
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / TELEMETRY_FILENAME


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _is_content_key(key: str) -> bool:
    lower = str(key).lower()
    return lower in _CONTENT_KEYS or lower.endswith(_CONTENT_SUFFIXES)


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_RE.search(str(key)))


def _redact_secret_values(text: str) -> str:
    redacted = text
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", redacted)
    return redacted


def _safe_string(value: str) -> str:
    text = _redact_secret_values(value)
    if len(text) > _MAX_STRING_CHARS:
        return text[:_MAX_STRING_CHARS] + "…[truncated]"
    return text


def _sanitize_value(
    key: str,
    value: Any,
    *,
    excluded: set[str],
    redacted: set[str],
) -> Any:
    if value is None:
        return None
    if _is_content_key(key):
        excluded.add(str(key))
        return _SKIP
    if _is_secret_key(key):
        redacted.add(str(key))
        return "[REDACTED]"
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        safe = _safe_string(value)
        if safe != value:
            redacted.add(str(key))
        return safe
    if isinstance(value, Path):
        # Paths are routing metadata when explicitly supplied (e.g. log path or
        # workspace kind), but still cap them like strings.
        return _safe_string(str(value))
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for idx, (child_key, child_value) in enumerate(value.items()):
            if idx >= _MAX_DICT_ITEMS:
                result["_truncated_items"] = len(value) - _MAX_DICT_ITEMS
                break
            child_key_str = str(child_key)
            safe_child = _sanitize_value(
                child_key_str,
                child_value,
                excluded=excluded,
                redacted=redacted,
            )
            if safe_child is not _SKIP and safe_child is not None:
                result[child_key_str] = safe_child
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        result_list = []
        seq = list(value)
        for idx, item in enumerate(seq):
            if idx >= _MAX_LIST_ITEMS:
                result_list.append({"_truncated_items": len(seq) - _MAX_LIST_ITEMS})
                break
            safe_item = _sanitize_value(
                f"{key}_item",
                item,
                excluded=excluded,
                redacted=redacted,
            )
            if safe_item is not _SKIP and safe_item is not None:
                result_list.append(safe_item)
        return result_list
    return _safe_string(str(value))


def _runtime_context() -> dict[str, Any]:
    """Return non-content runtime identifiers for the current route decision."""
    context: dict[str, Any] = {}

    def _session_env(name: str) -> str:
        try:
            from gateway.session_context import get_session_env

            return get_session_env(name, "") or ""
        except Exception:
            return os.environ.get(name, "") or ""

    session_id = _session_env("HERMES_SESSION_ID")
    if session_id:
        context["session_id"] = session_id
    platform = _session_env("HERMES_SESSION_PLATFORM")
    if platform:
        context["platform"] = platform
    profile = os.environ.get("HERMES_PROFILE")
    if profile:
        context["profile"] = profile
    kanban_task_id = os.environ.get("HERMES_KANBAN_TASK")
    if kanban_task_id:
        context["kanban_task_id"] = kanban_task_id
    kanban_run_id = os.environ.get("HERMES_KANBAN_RUN_ID")
    if kanban_run_id:
        context["kanban_run_id"] = kanban_run_id
    return context


def build_record(
    event_type: str,
    *,
    surface: str,
    status: Optional[str] = None,
    timestamp: Optional[str] = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a sanitized telemetry record without writing it."""
    excluded: set[str] = set()
    redacted: set[str] = set()

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp or _utc_now(),
        "event_type": str(event_type),
        "surface": str(surface),
    }
    if status:
        record["status"] = str(status)

    runtime = _runtime_context()
    if runtime:
        record["runtime"] = runtime

    for key, value in fields.items():
        if value is None:
            continue
        safe_value = _sanitize_value(str(key), value, excluded=excluded, redacted=redacted)
        if safe_value is not _SKIP and safe_value is not None:
            record[str(key)] = safe_value

    record.setdefault(
        "classification",
        {"domain": "unknown", "sensitivity": "unknown", "risk": "unknown"},
    )
    record.setdefault("context_sources", [])
    record.setdefault("retries_escalations", {"retry_count": 0, "escalated": False})
    record.setdefault("quality", {"user_correction_signal": "not_captured"})

    record["privacy"] = {
        "mode": "metadata_only",
        "content_fields_excluded": sorted(excluded),
        "secret_fields_redacted": sorted(redacted),
        "policy": (
            "Task content, prompts, messages, summaries, raw tool args/results, "
            "and credential-shaped fields are not persisted."
        ),
    }
    return record


def _lock(lock_fd) -> None:
    if fcntl is not None:  # pragma: no branch - depends on platform
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
    elif msvcrt is not None:  # pragma: no cover - Windows only
        locking = getattr(msvcrt, "locking")
        lk_lock = getattr(msvcrt, "LK_LOCK")
        locking(lock_fd.fileno(), lk_lock, 1)


def _unlock(lock_fd) -> None:
    if fcntl is not None:  # pragma: no branch - depends on platform
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows only
        try:
            locking = getattr(msvcrt, "locking")
            lk_unlock = getattr(msvcrt, "LK_UNLCK")
            locking(lock_fd.fileno(), lk_unlock, 1)
        except OSError:
            pass


def _rotate_if_needed(target: Path) -> None:
    """Best-effort bounded retention for the append-only JSONL log."""
    try:
        if not target.exists() or target.stat().st_size < _MAX_LOG_BYTES:
            return
        for idx in range(_MAX_ROTATIONS - 1, 0, -1):
            src = target.with_name(f"{target.name}.{idx}")
            dst = target.with_name(f"{target.name}.{idx + 1}")
            if src.exists():
                if idx + 1 > _MAX_ROTATIONS:
                    src.unlink(missing_ok=True)
                else:
                    src.replace(dst)
        target.replace(target.with_name(f"{target.name}.1"))
    except Exception:
        logger.debug("orchestration telemetry rotation failed", exc_info=True)


def _read_lines_locked(target: Path) -> list[str]:
    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_fd:
        try:
            _lock(lock_fd)
            return target.read_text(encoding="utf-8").splitlines()
        finally:
            _unlock(lock_fd)


def append_event(
    event_type: str,
    *,
    surface: str,
    status: Optional[str] = None,
    path: Optional[os.PathLike[str] | str] = None,
    **fields: Any,
) -> Optional[Path]:
    """Append a metadata-only routing event to JSONL.

    Returns the path written, or ``None`` if telemetry is disabled or a write
    failed. Failures are logged at debug level and never raised so routing cannot
    be broken by observability.
    """
    if not _telemetry_enabled():
        return None
    try:
        target = telemetry_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        record = build_record(event_type, surface=surface, status=status, **fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        lock_path = target.with_suffix(target.suffix + ".lock")
        with _WRITE_LOCK:
            with lock_path.open("a+", encoding="utf-8") as lock_fd:
                try:
                    _lock(lock_fd)
                    _rotate_if_needed(target)
                    with target.open("a", encoding="utf-8") as out:
                        out.write(line)
                finally:
                    _unlock(lock_fd)
        return target
    except Exception:
        logger.debug("orchestration telemetry append failed", exc_info=True)
        return None


def read_events(
    *,
    limit: int = 20,
    path: Optional[os.PathLike[str] | str] = None,
) -> list[dict[str, Any]]:
    """Read the most recent telemetry records from JSONL.

    Invalid/corrupt lines are skipped to keep the observability surface tolerant
    of partial writes or manual edits.
    """
    target = telemetry_path(path)
    if not target.exists():
        return []
    try:
        raw_lines = target.read_text(encoding="utf-8").splitlines()
    except Exception:
        logger.debug("orchestration telemetry read failed", exc_info=True)
        return []
    events: list[dict[str, Any]] = []
    for line in raw_lines[-max(1, int(limit)) :]:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


__all__ = [
    "SCHEMA_VERSION",
    "TELEMETRY_FILENAME",
    "append_event",
    "build_record",
    "read_events",
    "telemetry_path",
]
