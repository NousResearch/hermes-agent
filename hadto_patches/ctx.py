"""ctx-native runtime helpers for Hermes coding sessions.

This module binds Hermes coding sessions to ctx-managed tasks and worktrees.
It is intentionally conservative:

- Only sessions with coding-capable toolsets are eligible by default.
- ACP/editor sessions are excluded because they already carry an explicit cwd.
- ctx session creation is optional and only happens when a configured provider
  is explicitly supported by the daemon.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib import error, request

from hermes_cli.config import get_hermes_home, load_config
from utils import atomic_json_write

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

_BINDINGS_LOCK = threading.RLock()
_BINDINGS_LOCK_STATE = threading.local()
_CTX_DAEMON_RECOVERY_LOCK = threading.Lock()
_DEFAULT_DAEMON_URL = "http://127.0.0.1:19876"
_DEFAULT_DATA_DIR = "~/.ctx-data"
_DEFAULT_CTX_BINDING_STALE_HOURS = 12
_ACTIVE_CODEX_HANDOFF_REASON = "ctx task handed off to delegated Codex run"
_CTX_DAEMON_SERVICE_CANDIDATES = ("ctx-daemon.service", "ctx.service")
_CTX_CLOSED_POOL_SIGNATURE = "attempted to acquire a connection on a closed pool"
_CTX_RECOVERY_COOLDOWN_SECONDS = 10.0
_CTX_RECOVERY_WAIT_TIMEOUT_SECONDS = 20.0
_last_ctx_recovery_attempt = 0.0
_TASK_ENV_OVERRIDES: Dict[str, Dict[str, Any]] = {}


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]) -> None:
    """Register task-local environment overrides shared with terminal/file tools."""
    if not task_id:
        return
    _TASK_ENV_OVERRIDES[task_id] = dict(overrides or {})


def clear_task_env_overrides(task_id: str) -> None:
    """Drop task-local environment overrides after a session retires."""
    if not task_id:
        return
    _TASK_ENV_OVERRIDES.pop(task_id, None)


def get_task_env_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return a shallow copy of the task-local override map."""
    if not task_id:
        return {}
    return dict(_TASK_ENV_OVERRIDES.get(task_id, {}))


def get_task_cwd(task_id: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """Resolve the effective cwd for a task-bound session."""
    overrides = get_task_env_overrides(task_id)
    if overrides.get("cwd"):
        return str(overrides["cwd"])
    env_cwd = os.getenv("TERMINAL_CWD")
    if env_cwd:
        return env_cwd
    return default


@dataclass
class CtxBinding:
    """Resolved ctx binding for a Hermes session."""

    active: bool
    reason: str
    session_id: str
    platform: str = ""
    repo_root: Optional[str] = None
    workspace_id: Optional[str] = None
    workspace_root: Optional[str] = None
    task_id: Optional[str] = None
    worktree_id: Optional[str] = None
    worktree_path: Optional[str] = None
    ctx_session_id: Optional[str] = None
    ctx_session_provider_id: Optional[str] = None
    ctx_session_model_id: Optional[str] = None
    daemon_url: Optional[str] = None
    source: str = "none"

    @property
    def signature(self) -> str:
        if not self.active:
            return f"inactive:{self.reason}"
        parts = [
            self.workspace_id or "",
            self.task_id or "",
            self.worktree_id or "",
            self.ctx_session_id or "",
            self.ctx_session_provider_id or "",
            self.ctx_session_model_id or "",
        ]
        return "|".join(parts)

    def system_prompt_note(self) -> str:
        if not self.active:
            return ""
        details = [
            f"ctx task `{self.task_id}`",
            f"workspace `{self.workspace_id}`",
            f"worktree `{self.worktree_path}`",
        ]
        if self.ctx_session_id:
            details.append(
                f"ctx session `{self.ctx_session_id}`"
            )
        return (
            "[System note: This coding session is attached to "
            + ", ".join(details)
            + ". Use the ctx-managed worktree as the source of truth. "
            "Do not create a nested Hermes worktree. "
            "If the codex_delegate tool is available, act as the engineering manager: "
            "launch bounded local Codex implementation runs, inspect their status and "
            "results, and course-correct with follow-up runs instead of delegating "
            "the Linear issue to cloud Codex. "
            "When dispatching Codex workers, require them not to call Slack, Moshi, "
            "webhook URLs, or external status-report/notification endpoints unless "
            "the task explicitly requires notification work.]"
        )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except Exception:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ctx_binding_age_hours(record: Dict[str, Any], *, now: datetime) -> Optional[float]:
    ts = _parse_timestamp(record.get("updated_at") or record.get("created_at"))
    if not ts:
        return None
    return (now - ts).total_seconds() / 3600


def _ctx_bindings_path(bindings_path: Optional[Path] = None) -> Path:
    if bindings_path is not None:
        return Path(bindings_path).expanduser()
    return get_hermes_home() / "ctx" / "session_bindings.json"


def _default_state_db_path(bindings_path: Optional[Path] = None) -> Path:
    resolved_bindings_path = _ctx_bindings_path(bindings_path)
    if resolved_bindings_path.name == "session_bindings.json" and resolved_bindings_path.parent.name == "ctx":
        return resolved_bindings_path.parent.parent / "state.db"
    return resolved_bindings_path.parent / "state.db"


def _load_bindings(bindings_path: Optional[Path] = None) -> Dict[str, Any]:
    path = _ctx_bindings_path(bindings_path)
    if not path.exists():
        return {"version": 1, "sessions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read ctx bindings from %s", path, exc_info=True)
        return {"version": 1, "sessions": {}}
    if not isinstance(data, dict):
        return {"version": 1, "sessions": {}}
    data.setdefault("version", 1)
    data.setdefault("sessions", {})
    if not isinstance(data["sessions"], dict):
        data["sessions"] = {}
    return data


def _save_bindings(data: Dict[str, Any], bindings_path: Optional[Path] = None) -> None:
    atomic_json_write(_ctx_bindings_path(bindings_path), data)


def _ctx_bindings_lock_path(bindings_path: Optional[Path] = None) -> Path:
    path = _ctx_bindings_path(bindings_path)
    return path.with_suffix(path.suffix + ".lock")


@contextmanager
def _bindings_store_lock(bindings_path: Optional[Path] = None):
    """Serialize ctx binding read/modify/write cycles across processes."""

    with _BINDINGS_LOCK:
        if getattr(_BINDINGS_LOCK_STATE, "depth", 0) > 0:
            _BINDINGS_LOCK_STATE.depth += 1
            try:
                yield
            finally:
                _BINDINGS_LOCK_STATE.depth -= 1
            return

        lock_path = _ctx_bindings_lock_path(bindings_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            _BINDINGS_LOCK_STATE.depth = 1
            try:
                yield
            finally:
                _BINDINGS_LOCK_STATE.depth = 0
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        with lock_path.open("r+" if msvcrt else "a+") as lock_file:
            if fcntl:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            else:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)

            _BINDINGS_LOCK_STATE.depth = 1
            try:
                yield
            finally:
                _BINDINGS_LOCK_STATE.depth = 0
                if fcntl:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                elif msvcrt:
                    try:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except (OSError, IOError):
                        pass


def _load_binding_record(session_id: str, bindings_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _bindings_store_lock(bindings_path):
        data = _load_bindings(bindings_path)
        record = data.get("sessions", {}).get(session_id)
        return dict(record) if isinstance(record, dict) else None


def _set_binding_record_inactive(
    record: Dict[str, Any],
    *,
    reason: str,
    updated_at: Optional[str] = None,
) -> bool:
    updated_at = updated_at or _utcnow_iso()
    changed = bool(record.get("active")) or str(record.get("reason") or "") != reason
    record["active"] = False
    record["reason"] = reason
    record["updated_at"] = updated_at
    if "created_at" not in record:
        record["created_at"] = updated_at
    return changed


def _set_binding_record_active(
    record: Dict[str, Any],
    *,
    reason: str,
    updated_at: Optional[str] = None,
) -> bool:
    updated_at = updated_at or _utcnow_iso()
    changed = (not bool(record.get("active"))) or str(record.get("reason") or "") != reason
    record["active"] = True
    record["reason"] = reason
    record["updated_at"] = updated_at
    if "created_at" not in record:
        record["created_at"] = updated_at
    return changed


def _open_session_db(*, state_db_path: Optional[Path] = None, bindings_path: Optional[Path] = None):
    db_path = Path(state_db_path).expanduser() if state_db_path is not None else _default_state_db_path(bindings_path)
    if not db_path.exists():
        return None
    try:
        from hermes_state import SessionDB

        return SessionDB(db_path=db_path)
    except Exception:
        logger.debug("Failed to open SessionDB while normalizing ctx bindings", exc_info=True)
        return None


def _load_active_codex_handoff_indexes(
    *,
    runs_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Dict[str, set[str]]:
    indexes: Dict[str, set[str]] = {
        "binding_session_id": set(),
        "task_id": set(),
        "ctx_session_id": set(),
        "worktree_id": set(),
        "worktree_path": set(),
    }
    try:
        normalize_codex_runs = _load_normalize_codex_runs()
    except Exception:
        logger.debug("Failed to import Codex delegate helpers for ctx handoff detection", exc_info=True)
        return indexes

    try:
        normalize_codex_runs(runs_path=runs_path, now=now)
    except Exception:
        logger.debug("Failed to normalize Codex runs before ctx handoff detection", exc_info=True)

    resolved_runs_path = Path(runs_path).expanduser() if runs_path is not None else get_hermes_home() / "codex" / "runs.json"
    if not resolved_runs_path.exists():
        return indexes

    try:
        payload = json.loads(resolved_runs_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to load Codex runs from %s for ctx handoff detection", resolved_runs_path, exc_info=True)
        return indexes

    runs = payload.get("runs", {}) if isinstance(payload, dict) else {}
    if not isinstance(runs, dict):
        return indexes

    for raw_record in runs.values():
        if not isinstance(raw_record, dict):
            continue
        if str(raw_record.get("status") or "").strip().lower() not in {"running", "unknown"}:
            continue
        values = {
            "binding_session_id": str(raw_record.get("task_id") or "").strip(),
            "task_id": str(raw_record.get("ctx_task_id") or "").strip(),
            "ctx_session_id": str(raw_record.get("ctx_session_id") or "").strip(),
            "worktree_id": str(raw_record.get("ctx_worktree_id") or "").strip(),
            "worktree_path": str(_normalize_path(raw_record.get("ctx_worktree_path")) or "").strip(),
        }
        for key, value in values.items():
            if value:
                indexes[key].add(value)
    return indexes


def _load_normalize_codex_runs():
    """Load Codex run normalization from the Hadto plugin or legacy core path."""
    try:
        from hadto_hermes_plugin.tools.codex_delegate import normalize_codex_runs

        return normalize_codex_runs
    except ImportError:
        plugin_root = get_hermes_home() / "plugins" / "hadto"
        plugin_root_str = str(plugin_root)
        if plugin_root.exists() and plugin_root_str not in sys.path:
            sys.path.insert(0, plugin_root_str)
            try:
                from hadto_hermes_plugin.tools.codex_delegate import normalize_codex_runs

                return normalize_codex_runs
            except ImportError:
                pass

    from tools.codex_delegate_tool import normalize_codex_runs

    return normalize_codex_runs


def _binding_has_active_codex_handoff(
    record: Dict[str, Any],
    *,
    active_codex_indexes: Optional[Dict[str, set[str]]] = None,
) -> bool:
    indexes = active_codex_indexes or {
        "binding_session_id": set(),
        "task_id": set(),
        "ctx_session_id": set(),
        "worktree_id": set(),
        "worktree_path": set(),
    }
    candidates = {
        "binding_session_id": str(record.get("session_id") or "").strip(),
        "task_id": str(record.get("task_id") or "").strip(),
        "ctx_session_id": str(record.get("ctx_session_id") or "").strip(),
        "worktree_id": str(record.get("worktree_id") or "").strip(),
        "worktree_path": str(_normalize_path(record.get("worktree_path")) or "").strip(),
    }
    return any(value and value in indexes.get(key, set()) for key, value in candidates.items())


def _ctx_binding_retirement_reason(
    record: Dict[str, Any],
    *,
    session_db,
    active_codex_indexes: Optional[Dict[str, set[str]]] = None,
    now: Optional[datetime] = None,
    stale_hours: int = _DEFAULT_CTX_BINDING_STALE_HOURS,
) -> Optional[str]:
    if not bool(record.get("active")):
        return None

    existing_reason = str(record.get("reason") or "").strip()
    if existing_reason.lower().startswith("ctx binding retired:"):
        return existing_reason

    worktree_path = record.get("worktree_path")
    if worktree_path and not Path(worktree_path).exists():
        return "ctx binding retired: worktree missing"
    if _binding_has_active_codex_handoff(record, active_codex_indexes=active_codex_indexes):
        return None

    session_id = str(record.get("session_id") or "").strip()
    if session_db and session_id:
        try:
            session = session_db.get_session(session_id)
        except Exception:
            logger.debug("Failed to read Hermes session state for ctx session %s", session_id, exc_info=True)
            session = None
        if isinstance(session, dict) and session.get("ended_at") is not None:
            end_reason = str(session.get("end_reason") or "session ended").strip()
            return f"ctx binding retired: session ended ({end_reason})"

    now = now or datetime.now(timezone.utc)
    age_hours = _ctx_binding_age_hours(record, now=now)
    if age_hours is not None and age_hours >= stale_hours:
        return f"ctx binding retired: stale active binding (>{stale_hours}h)"
    return None


def _clear_binding_overrides(session_ids: Iterable[str]) -> None:
    ids = [str(session_id or "").strip() for session_id in session_ids if str(session_id or "").strip()]
    if not ids:
        return

    for session_id in ids:
        try:
            clear_task_env_overrides(session_id)
        except Exception:
            logger.debug("Failed to clear task env overrides for ctx session %s", session_id, exc_info=True)


def normalize_ctx_bindings(
    *,
    session_id: Optional[str] = None,
    bindings_path: Optional[Path] = None,
    state_db_path: Optional[Path] = None,
    codex_runs_path: Optional[Path] = None,
    stale_hours: int = _DEFAULT_CTX_BINDING_STALE_HOURS,
    now: Optional[datetime] = None,
) -> Dict[str, str]:
    """Deactivate persisted ctx bindings that no longer map to live sessions."""

    retired: Dict[str, str] = {}
    cleared_overrides: list[str] = []
    bindings_path = _ctx_bindings_path(bindings_path)
    session_db = _open_session_db(state_db_path=state_db_path, bindings_path=bindings_path)
    now = now or datetime.now(timezone.utc)
    active_codex_indexes = _load_active_codex_handoff_indexes(runs_path=codex_runs_path, now=now)
    try:
        with _bindings_store_lock(bindings_path):
            data = _load_bindings(bindings_path)
            sessions = data.setdefault("sessions", {})
            if not isinstance(sessions, dict):
                sessions = {}
                data["sessions"] = sessions

            session_ids = [session_id] if session_id else list(sessions.keys())
            changed = False
            for current_session_id in session_ids:
                record = sessions.get(current_session_id)
                if not isinstance(record, dict):
                    continue
                if _binding_has_active_codex_handoff(record, active_codex_indexes=active_codex_indexes):
                    if _set_binding_record_active(
                        record,
                        reason=_ACTIVE_CODEX_HANDOFF_REASON,
                        updated_at=now.isoformat(),
                    ):
                        changed = True
                    continue
                reason = _ctx_binding_retirement_reason(
                    record,
                    session_db=session_db,
                    active_codex_indexes=active_codex_indexes,
                    now=now,
                    stale_hours=stale_hours,
                )
                if not reason:
                    continue
                if _set_binding_record_inactive(record, reason=reason, updated_at=now.isoformat()):
                    changed = True
                retired[current_session_id] = reason
                cleared_overrides.append(current_session_id)
            if changed:
                _save_bindings(data, bindings_path)
    finally:
        if session_db is not None:
            try:
                session_db.close()
            except Exception:
                logger.debug("Failed to close SessionDB after ctx binding normalization", exc_info=True)

    _clear_binding_overrides(cleared_overrides)
    return retired


def retire_ctx_binding(
    session_id: str,
    *,
    reason: str,
    preserve_codex_handoff: bool = False,
    codex_runs_path: Optional[Path] = None,
) -> bool:
    """Mark a persisted ctx binding inactive and clear its worktree override."""

    session_id = str(session_id or "").strip()
    reason = str(reason or "").strip()
    if not session_id or not reason:
        return False

    changed = False
    cleared_overrides: list[str] = []
    active_codex_indexes = (
        _load_active_codex_handoff_indexes(runs_path=codex_runs_path) if preserve_codex_handoff else None
    )
    with _bindings_store_lock():
        data = _load_bindings()
        record = data.get("sessions", {}).get(session_id)
        if isinstance(record, dict):
            if preserve_codex_handoff and _binding_has_active_codex_handoff(
                record,
                active_codex_indexes=active_codex_indexes,
            ):
                if _set_binding_record_active(record, reason=_ACTIVE_CODEX_HANDOFF_REASON):
                    changed = True
            elif _set_binding_record_inactive(record, reason=reason):
                changed = True
                cleared_overrides.append(session_id)
            if changed:
                _save_bindings(data)

    _clear_binding_overrides(cleared_overrides)
    return changed


def _persist_binding(binding: CtxBinding) -> None:
    with _bindings_store_lock():
        data = _load_bindings()
        sessions = data.setdefault("sessions", {})
        record = asdict(binding)
        record["updated_at"] = _utcnow_iso()
        existing = sessions.get(binding.session_id)
        if isinstance(existing, dict) and "created_at" in existing:
            record["created_at"] = existing["created_at"]
        else:
            record["created_at"] = record["updated_at"]
        sessions[binding.session_id] = record
        _save_bindings(data)


def describe_existing_ctx_binding(session_id: str) -> Optional[CtxBinding]:
    normalize_ctx_bindings(session_id=session_id)
    record = _load_binding_record(session_id)
    if not record:
        return None
    return _binding_from_record(record, reason=str(record.get("reason") or "existing"))


def _binding_from_record(record: Dict[str, Any], *, reason: str) -> CtxBinding:
    return CtxBinding(
        active=bool(record.get("active")),
        reason=reason,
        session_id=str(record.get("session_id") or ""),
        platform=str(record.get("platform") or ""),
        repo_root=record.get("repo_root"),
        workspace_id=record.get("workspace_id"),
        workspace_root=record.get("workspace_root"),
        task_id=record.get("task_id"),
        worktree_id=record.get("worktree_id"),
        worktree_path=record.get("worktree_path"),
        ctx_session_id=record.get("ctx_session_id"),
        ctx_session_provider_id=record.get("ctx_session_provider_id"),
        ctx_session_model_id=record.get("ctx_session_model_id"),
        daemon_url=record.get("daemon_url"),
        source=str(record.get("source") or "record"),
    )


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(Path(path).expanduser())


def _path_within_root(path: Optional[str], root: Optional[str]) -> bool:
    if not path or not root:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _guess_repo_root(candidate: Optional[str]) -> Optional[str]:
    starts = []
    for raw in (candidate, os.getcwd(), os.getenv("TERMINAL_CWD")):
        normalized = _normalize_path(raw)
        if normalized and normalized not in starts:
            starts.append(normalized)

    fallback = None
    for start in starts:
        if fallback is None:
            fallback = start
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=start,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return _normalize_path(result.stdout.strip())
        except Exception:
            continue
    return fallback


def _load_ctx_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    raw = cfg.get("ctx", {}) if isinstance(cfg, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "coding_mode": str(raw.get("coding_mode", "auto") or "auto").strip().lower(),
        "coding_toolsets": [
            str(item).strip()
            for item in (raw.get("coding_toolsets") or ["terminal", "file", "code_execution"])
            if str(item).strip()
        ],
        "daemon_url": str(raw.get("daemon_url") or "").strip(),
        "auth_token": str(raw.get("auth_token") or "").strip(),
        "workspace_id": str(raw.get("workspace_id") or "").strip(),
        "data_dir": str(raw.get("data_dir") or _DEFAULT_DATA_DIR).strip(),
        "session_provider_id": str(raw.get("session_provider_id") or "").strip(),
        "session_model_id": str(raw.get("session_model_id") or "").strip(),
        "session_execution_environment": str(raw.get("session_execution_environment") or "host").strip(),
    }


def is_ctx_candidate(
    *,
    enabled_toolsets: Optional[Iterable[str]],
    platform: Optional[str],
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    ctx_cfg = _load_ctx_config(config)
    if not ctx_cfg["enabled"]:
        return False
    if str(platform or "").strip().lower() == "acp":
        return False

    mode = ctx_cfg["coding_mode"]
    if mode in {"always", "on", "true", "1"}:
        return True
    if mode in {"off", "false", "0", "disabled"}:
        return False

    toolsets = {str(item).strip() for item in (enabled_toolsets or []) if str(item).strip()}
    return bool(toolsets & set(ctx_cfg["coding_toolsets"]))


def _find_auth_material(ctx_cfg: Dict[str, Any]) -> tuple[str, str]:
    daemon_url = (
        os.getenv("CTX_DAEMON_URL")
        or ctx_cfg.get("daemon_url")
        or ""
    ).strip()
    token = (
        os.getenv("CTX_DAEMON_AUTH_TOKEN")
        or ctx_cfg.get("auth_token")
        or ""
    ).strip()

    probe_paths = [
        Path("~/.ctx/auth.json").expanduser(),
        Path("~/.config/ctx/auth.json").expanduser(),
        Path(os.getenv("CTX_DATA_DIR") or ctx_cfg.get("data_dir") or _DEFAULT_DATA_DIR).expanduser() / "daemon_auth.json",
        Path("~/.ctx-data/daemon_auth.json").expanduser(),
    ]
    for path in probe_paths:
        if token and daemon_url:
            break
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not daemon_url:
            daemon_url = str(payload.get("daemon_url") or "").strip()
        if not token:
            token = str(payload.get("token") or "").strip()

    daemon_url = daemon_url or _DEFAULT_DAEMON_URL
    return daemon_url.rstrip("/"), token


def _ctx_error_is_closed_pool(detail: str) -> bool:
    return _CTX_CLOSED_POOL_SIGNATURE in str(detail or "").lower()


def _wait_for_ctx_daemon_ready(daemon_url: str, token: str) -> None:
    deadline = time.monotonic() + _CTX_RECOVERY_WAIT_TIMEOUT_SECONDS
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{daemon_url.rstrip('/')}/api/workspaces"

    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        req = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=2) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)

    raise RuntimeError(
        f"ctx daemon did not become ready after restart at {daemon_url}: {last_error}"
    )


def _restart_ctx_daemon_service(daemon_url: str, token: str) -> None:
    global _last_ctx_recovery_attempt

    with _CTX_DAEMON_RECOVERY_LOCK:
        now = time.monotonic()
        if now - _last_ctx_recovery_attempt < _CTX_RECOVERY_COOLDOWN_SECONDS:
            logger.info("ctx daemon recovery already attempted recently; waiting for readiness")
            _wait_for_ctx_daemon_ready(daemon_url, token)
            return

        last_failure = None
        for service_name in _CTX_DAEMON_SERVICE_CANDIDATES:
            result = subprocess.run(
                ["systemctl", "--user", "restart", service_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                _last_ctx_recovery_attempt = time.monotonic()
                logger.warning("Restarted %s after ctx closed-pool error", service_name)
                _wait_for_ctx_daemon_ready(daemon_url, token)
                return
            last_failure = RuntimeError(
                f"{service_name} restart failed: {result.stderr.strip() or result.stdout.strip()}"
            )

    raise last_failure or RuntimeError("No ctx daemon service restart command succeeded")


class _CtxDaemonClient:
    def __init__(self, daemon_url: str, token: str):
        self.daemon_url = daemon_url.rstrip("/")
        self.token = token

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        _allow_closed_pool_recovery: bool = True,
    ) -> Any:
        url = f"{self.daemon_url}{path}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with request.urlopen(req, timeout=20) as resp:
                body = resp.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if (
                _allow_closed_pool_recovery
                and exc.code >= 500
                and _ctx_error_is_closed_pool(detail)
            ):
                logger.warning(
                    "ctx daemon reported a closed pool on %s %s; restarting daemon and retrying once",
                    method.upper(),
                    path,
                )
                _restart_ctx_daemon_service(self.daemon_url, self.token)
                return self._request_json(
                    method,
                    path,
                    payload=payload,
                    _allow_closed_pool_recovery=False,
                )
            raise RuntimeError(f"ctx daemon {method.upper()} {path} -> HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"ctx daemon unreachable at {url}: {exc}") from exc

        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def list_workspaces(self) -> list[dict]:
        data = self._request_json("GET", "/api/workspaces")
        return data if isinstance(data, list) else []

    def get_workspace(self, workspace_id: str) -> Optional[dict]:
        data = self._request_json("GET", f"/api/workspaces/{workspace_id}")
        return data if isinstance(data, dict) else None

    def create_task(self, workspace_id: str, title: str, prompt: str) -> dict:
        data = self._request_json(
            "POST",
            f"/api/workspaces/{workspace_id}/tasks",
            payload={"title": title, "prompt": prompt},
        )
        if not isinstance(data, dict):
            raise RuntimeError("ctx daemon returned an unexpected task payload")
        return data

    def list_providers(self) -> list[dict]:
        data = self._request_json("GET", "/api/providers")
        return data if isinstance(data, list) else []

    def create_session(
        self,
        task_id: str,
        *,
        provider_id: str,
        model_id: str,
        execution_environment: str,
    ) -> dict:
        data = self._request_json(
            "POST",
            f"/api/tasks/{task_id}/sessions",
            payload={
                "provider_id": provider_id,
                "model_id": model_id,
                "execution_environment": execution_environment,
            },
        )
        if not isinstance(data, dict):
            raise RuntimeError("ctx daemon returned an unexpected session payload")
        return data


def _resolve_workspace(
    client: _CtxDaemonClient,
    *,
    workspace_id: str,
    repo_root: Optional[str],
) -> Optional[dict]:
    if workspace_id:
        workspace = client.get_workspace(workspace_id)
        if workspace:
            return workspace
        raise RuntimeError(f"Configured ctx workspace {workspace_id} was not found")

    candidate_root = _normalize_path(repo_root)
    if not candidate_root:
        return None

    for workspace in client.list_workspaces():
        root_path = _normalize_path(workspace.get("root_path"))
        if not root_path:
            continue
        if candidate_root == root_path or _path_within_root(candidate_root, root_path):
            return workspace
    return None


def _resolve_provider_session(
    client: _CtxDaemonClient,
    ctx_cfg: Dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    provider_id = (
        os.getenv("HERMES_CTX_SESSION_PROVIDER_ID")
        or ctx_cfg.get("session_provider_id")
        or ""
    ).strip()
    model_id = (
        os.getenv("HERMES_CTX_SESSION_MODEL_ID")
        or ctx_cfg.get("session_model_id")
        or ""
    ).strip()
    if not provider_id or not model_id:
        return None, None

    providers = client.list_providers()
    for provider in providers:
        if provider.get("provider_id") == provider_id:
            usable = bool((provider.get("usability") or {}).get("usable"))
            if usable and provider.get("installed"):
                return provider_id, model_id
            logger.info(
                "ctx provider %s configured for Hermes but not usable: %s",
                provider_id,
                provider.get("usability"),
            )
            return None, None
    logger.info("ctx provider %s not advertised by daemon; skipping ctx session creation", provider_id)
    return None, None


def _task_title_for_session(session_id: str, prompt: str, repo_root: Optional[str]) -> str:
    line = (prompt or "").strip().splitlines()[0] if prompt else ""
    line = line.strip()
    if line:
        return line[:96]
    repo_name = Path(repo_root).name if repo_root else "session"
    return f"Hermes coding session {repo_name} {session_id}"[:96]


def maybe_bind_ctx_session(
    *,
    session_id: str,
    enabled_toolsets: Optional[Iterable[str]],
    platform: Optional[str],
    prompt: str = "",
    repo_root: Optional[str] = None,
    dry_run: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> CtxBinding:
    """Resolve and optionally create a ctx binding for a Hermes session."""

    session_id = str(session_id or "").strip()
    platform = str(platform or "").strip()
    if not session_id:
        return CtxBinding(active=False, reason="missing session id", session_id="", platform=platform)

    if not is_ctx_candidate(enabled_toolsets=enabled_toolsets, platform=platform, config=config):
        return CtxBinding(active=False, reason="ctx disabled for this session", session_id=session_id, platform=platform)

    normalize_ctx_bindings()
    record = _load_binding_record(session_id)
    if record:
        binding = _binding_from_record(record, reason="reused persisted binding")
        if binding.active and binding.worktree_path and Path(binding.worktree_path).exists():
            if not dry_run:
                register_task_env_overrides(session_id, {"cwd": binding.worktree_path})
            return binding

    ctx_cfg = _load_ctx_config(config)
    daemon_url, token = _find_auth_material(ctx_cfg)
    if not token:
        return CtxBinding(
            active=False,
            reason="ctx auth token not found",
            session_id=session_id,
            platform=platform,
            daemon_url=daemon_url,
        )

    resolved_repo_root = _guess_repo_root(repo_root)
    client = _CtxDaemonClient(daemon_url, token)

    try:
        workspace = _resolve_workspace(
            client,
            workspace_id=ctx_cfg["workspace_id"],
            repo_root=resolved_repo_root,
        )
    except Exception as exc:
        logger.warning("ctx workspace resolution failed for session %s: %s", session_id, exc)
        return CtxBinding(
            active=False,
            reason=str(exc),
            session_id=session_id,
            platform=platform,
            repo_root=resolved_repo_root,
            daemon_url=daemon_url,
        )

    if not workspace:
        return CtxBinding(
            active=False,
            reason="no matching ctx workspace",
            session_id=session_id,
            platform=platform,
            repo_root=resolved_repo_root,
            daemon_url=daemon_url,
        )

    workspace_id = str(workspace.get("id") or "")
    workspace_root = _normalize_path(workspace.get("root_path"))
    if dry_run:
        return CtxBinding(
            active=True,
            reason="ctx workspace available",
            session_id=session_id,
            platform=platform,
            repo_root=resolved_repo_root,
            workspace_id=workspace_id,
            workspace_root=workspace_root,
            daemon_url=daemon_url,
            source="dry-run",
        )

    try:
        task = client.create_task(
            workspace_id,
            title=_task_title_for_session(session_id, prompt, resolved_repo_root),
            prompt=prompt or f"Hermes coding session {session_id}",
        )
    except Exception as exc:
        logger.warning("ctx task creation failed for session %s: %s", session_id, exc)
        return CtxBinding(
            active=False,
            reason=f"ctx task creation failed: {exc}",
            session_id=session_id,
            platform=platform,
            repo_root=resolved_repo_root,
            workspace_id=workspace_id,
            workspace_root=workspace_root,
            daemon_url=daemon_url,
        )

    task_id = str(task.get("id") or "")
    worktree_id = str(task.get("primary_worktree_id") or "")
    data_dir = Path(os.getenv("CTX_DATA_DIR") or ctx_cfg.get("data_dir") or _DEFAULT_DATA_DIR).expanduser()
    worktree_path = str(data_dir / "worktrees" / workspace_id / worktree_id) if worktree_id else None

    provider_id, model_id = _resolve_provider_session(client, ctx_cfg)
    ctx_session_id = None
    if provider_id and model_id:
        try:
            session_payload = client.create_session(
                task_id,
                provider_id=provider_id,
                model_id=model_id,
                execution_environment=ctx_cfg["session_execution_environment"],
            )
            ctx_session_id = str(session_payload.get("id") or "")
            worktree_id = str(session_payload.get("worktree_id") or worktree_id)
            if worktree_id:
                worktree_path = str(data_dir / "worktrees" / workspace_id / worktree_id)
            logger.info(
                "Bound Hermes session %s to ctx task %s and ctx session %s",
                session_id,
                task_id,
                ctx_session_id,
            )
        except Exception as exc:
            logger.warning(
                "ctx session creation skipped for Hermes session %s: %s",
                session_id,
                exc,
            )
            provider_id = None
            model_id = None

    binding = CtxBinding(
        active=True,
        reason="ctx task bound",
        session_id=session_id,
        platform=platform,
        repo_root=resolved_repo_root,
        workspace_id=workspace_id,
        workspace_root=workspace_root,
        task_id=task_id,
        worktree_id=worktree_id,
        worktree_path=worktree_path,
        ctx_session_id=ctx_session_id,
        ctx_session_provider_id=provider_id,
        ctx_session_model_id=model_id,
        daemon_url=daemon_url,
        source="ctx-daemon",
    )
    _persist_binding(binding)

    if binding.worktree_path:
        register_task_env_overrides(session_id, {"cwd": binding.worktree_path})

    logger.info(
        "ctx binding active for Hermes session %s: workspace=%s task=%s worktree=%s",
        session_id,
        workspace_id,
        task_id,
        worktree_id,
    )
    return binding
