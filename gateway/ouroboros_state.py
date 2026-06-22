"""File-backed Ouroboros recent-ID state for gateway platform contexts.

The gateway may share a single Hermes conversation/session key across multiple
Discord users or threads.  This module keeps the latest Ouroboros identifiers
scoped by the platform routing context instead of by conversation alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

try:  # Prefer the project-wide helper because it preserves symlinks/modes.
    from utils import atomic_json_write as _atomic_json_write
except Exception:  # pragma: no cover - fallback for import-isolated use.
    _atomic_json_write = None

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - fallback for import-isolated use.
    get_hermes_home = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OooStateContext:
    """Gateway routing context used to scope Ouroboros recent IDs."""

    platform: str
    guild_id: str | None
    channel_id: str | None
    thread_id: str | None
    user_id: str | None
    profile: str = "default"


@dataclass
class OooRecentState:
    """Recent Ouroboros identifiers remembered for one scoped context."""

    interview_session_id: str | None = None
    pm_session_id: str | None = None
    auto_session_id: str | None = None
    last_session_id: str | None = None
    last_job_id: str | None = None
    last_execution_id: str | None = None
    last_lineage_id: str | None = None
    last_seed_id: str | None = None
    last_seed_path: str | None = None
    last_seed_content: str | None = None
    last_id_kind: str | None = None
    last_start_idempotency_key: str | None = None
    last_start_tool: str | None = None
    last_start_args_fingerprint: str | None = None
    updated_at: str | None = None


_STATE_FIELD_NAMES = {field.name for field in fields(OooRecentState)}
_STORAGE_VERSION = 1
_STATE_LOCKS_GUARD = threading.Lock()
_STATE_LOCKS: dict[Path, threading.RLock] = {}


_ID_KEY_MAP = {
    "interview_session_id": "interview_session_id",
    "pm_session_id": "pm_session_id",
    "auto_session_id": "auto_session_id",
    "session_id": "last_session_id",
    "last_session_id": "last_session_id",
    "job_id": "last_job_id",
    "last_job_id": "last_job_id",
    "execution_id": "last_execution_id",
    "last_execution_id": "last_execution_id",
    "lineage_id": "last_lineage_id",
    "last_lineage_id": "last_lineage_id",
    "seed_id": "last_seed_id",
    "last_seed_id": "last_seed_id",
}


def _default_state_path() -> Path:
    if get_hermes_home is not None:
        return get_hermes_home() / "gateway" / "ouroboros_state.json"
    home = os.environ.get("HERMES_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".hermes"
    return base / "gateway" / "ouroboros_state.json"


def _resolved_state_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def _lock_for_state_path(path: Path) -> threading.RLock:
    key = _resolved_state_path(path)
    with _STATE_LOCKS_GUARD:
        lock = _STATE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _STATE_LOCKS[key] = lock
        return lock


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_context_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_state_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _state_from_dict(data: Any) -> OooRecentState:
    if not isinstance(data, dict):
        return OooRecentState()
    kwargs = {name: _coerce_state_value(data.get(name)) for name in _STATE_FIELD_NAMES}
    return OooRecentState(**kwargs)


def _fallback_atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_json_atomic(path: Path, data: Any) -> None:
    if _atomic_json_write is not None:
        _atomic_json_write(path, data, sort_keys=True)
        return
    _fallback_atomic_json_write(path, data)


class OooStateStore:
    """Small atomic JSON store for gateway-scoped Ouroboros recent IDs."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _default_state_path()
        self._lock = _lock_for_state_path(self.path)

    def context_key(self, ctx: OooStateContext) -> str:
        """Return a stable, separator-safe key for a platform/user/thread scope."""

        context_payload = {
            "platform": _coerce_context_value(ctx.platform),
            "guild_id": _coerce_context_value(ctx.guild_id),
            "channel_id": _coerce_context_value(ctx.channel_id),
            "thread_id": _coerce_context_value(ctx.thread_id),
            "user_id": _coerce_context_value(ctx.user_id),
            "profile": _coerce_context_value(ctx.profile or "default"),
        }
        serialized = json.dumps(
            context_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"ooo:{digest}"

    def load(self, ctx: OooStateContext) -> OooRecentState:
        """Load recent state for *ctx*, returning an empty state if absent."""

        data = self._read_store()
        contexts = data.get("contexts")
        if not isinstance(contexts, dict):
            return OooRecentState()
        return _state_from_dict(contexts.get(self.context_key(ctx)))

    def save(self, ctx: OooStateContext, state: OooRecentState) -> None:
        """Persist *state* for *ctx* without modifying its fields."""

        with self._lock:
            data = self._read_store()
            contexts = data.setdefault("contexts", {})
            if not isinstance(contexts, dict):
                contexts = {}
                data["contexts"] = contexts
            data["version"] = _STORAGE_VERSION
            contexts[self.context_key(ctx)] = asdict(state)
            _write_json_atomic(self.path, data)

    def update(self, ctx: OooStateContext, **fields_to_update: Any) -> OooRecentState:
        """Merge provided fields into the scoped state and refresh updated_at."""

        unknown = set(fields_to_update) - _STATE_FIELD_NAMES
        if unknown:
            raise ValueError(f"Unknown Ouroboros state field(s): {', '.join(sorted(unknown))}")

        with self._lock:
            data = self._read_store()
            contexts = data.setdefault("contexts", {})
            if not isinstance(contexts, dict):
                contexts = {}
                data["contexts"] = contexts

            key = self.context_key(ctx)
            current = asdict(_state_from_dict(contexts.get(key)))
            for name, value in fields_to_update.items():
                if name == "updated_at":
                    continue
                current[name] = _coerce_state_value(value)
            current["updated_at"] = _utc_now_iso()

            data["version"] = _STORAGE_VERSION
            contexts[key] = current
            _write_json_atomic(self.path, data)
            return OooRecentState(**current)

    def _read_store(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": _STORAGE_VERSION, "contexts": {}}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            self._quarantine_corrupt_file()
            return {"version": _STORAGE_VERSION, "contexts": {}}

        if not isinstance(data, dict):
            return {"version": _STORAGE_VERSION, "contexts": {}}
        contexts = data.get("contexts")
        if not isinstance(contexts, dict):
            data = {"version": _STORAGE_VERSION, "contexts": {}}
        else:
            data.setdefault("version", _STORAGE_VERSION)
        return data

    def _quarantine_corrupt_file(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        corrupt_path = self.path.with_name(f"{self.path.name}.corrupt.{timestamp}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.path, corrupt_path)
        except OSError:
            # Best-effort quarantine only.  The caller still gets a safe empty
            # in-memory state rather than an exception from malformed JSON.
            pass


def _extract_ids_from_mapping(payload: dict[str, Any]) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for source_key, target_key in _ID_KEY_MAP.items():
        value = payload.get(source_key)
        if value is None:
            continue
        extracted[target_key] = str(value)
    return extracted


def _wrapped_dict(payload: dict[str, Any], wrapper_key: str) -> dict[str, Any] | None:
    value = payload.get(wrapper_key)
    if isinstance(value, dict):
        return value
    if wrapper_key == "result" and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_ids(payload: dict[str, Any]) -> dict[str, str]:
    """Extract common flat or wrapped Ouroboros/MCP identifiers for state updates."""

    if not isinstance(payload, dict):
        return {}

    extracted = _extract_ids_from_mapping(payload)
    for wrapper_key in ("result", "structuredContent", "content"):
        wrapped = _wrapped_dict(payload, wrapper_key)
        if wrapped is None:
            continue
        for target_key, value in _extract_ids_from_mapping(wrapped).items():
            extracted.setdefault(target_key, value)
    return extracted
