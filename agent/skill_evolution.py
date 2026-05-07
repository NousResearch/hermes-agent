"""Pending queue helpers for background skill evolution changes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


VALID_EVOLUTION_MODES = {"auto", "confirm", "readonly"}
_LOCAL_QUEUE_LOCK = threading.RLock()


def normalize_evolution_mode(value: Any) -> str:
    """Return a supported evolution mode, defaulting to auto."""
    if not isinstance(value, str):
        return "auto"
    mode = value.strip().lower()
    if mode in VALID_EVOLUTION_MODES:
        return mode
    return "auto"


def get_evolution_mode(config: dict[str, Any] | None = None) -> str:
    """Read skills.evolution_mode from config, defaulting to auto."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()

    skills_config = config.get("skills") if isinstance(config, dict) else None
    if not isinstance(skills_config, dict):
        return "auto"
    return normalize_evolution_mode(skills_config.get("evolution_mode"))


def pending_queue_path() -> Path:
    """Return the profile-aware pending queue path."""
    return get_hermes_home() / "skills" / ".evolution_pending.json"


def pending_changes_dir() -> Path:
    """Return the profile-aware directory for rich pending change artifacts."""
    return get_hermes_home() / "skills" / ".pending"


@contextmanager
def _queue_lock():
    """Serialize pending queue read-modify-write cycles across processes."""
    lock_path = pending_queue_path().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with _LOCAL_QUEUE_LOCK:
        if fcntl is None and msvcrt is None:
            yield
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        fd = open(lock_path, "r+" if msvcrt else "a+")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()


def _empty_queue() -> dict[str, list[dict[str, Any]]]:
    return {"changes": []}


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _normalize_change(change: dict[str, Any]) -> dict[str, Any]:
    payload = change.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    normalized = {
        "id": str(change.get("id", "")),
        "created_at": str(change.get("created_at", "")),
        "action": str(change.get("action", "")),
        "name": str(change.get("name", "")),
        "payload": payload,
        "origin": str(change.get("origin", "background_review") or "background_review"),
        "status": str(change.get("status", "pending") or "pending"),
    }
    for key in ("manifest_path", "snapshot_path", "diff_path"):
        value = change.get(key)
        if value:
            normalized[key] = str(value)
    return normalized


def _artifact_path(change_id: str, value: Any) -> Path | None:
    if not value:
        return None
    base = _pending_artifact_dir(change_id)
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve()
        resolved.relative_to(base.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _normalize_stored_change(change: dict[str, Any]) -> dict[str, Any] | None:
    normalized = _normalize_change(change)
    change_id = normalized.get("id", "")
    for key in ("manifest_path", "snapshot_path", "diff_path"):
        if key not in normalized:
            continue
        resolved = _artifact_path(change_id, normalized.get(key))
        if resolved is None:
            return None
        normalized[key] = str(resolved)
    return normalized


def _read_queue() -> dict[str, list[dict[str, Any]]]:
    path = pending_queue_path()
    data = _read_json_file(path)

    changes = data.get("changes") if isinstance(data, dict) else None
    if not isinstance(changes, list):
        return _empty_queue()
    return {
        "changes": [
            normalized
            for change in changes
            if isinstance(change, dict)
            and change.get("id")
            for normalized in [_normalize_stored_change(change)]
            if normalized is not None
        ]
    }


def _write_queue(queue: dict[str, list[dict[str, Any]]]) -> None:
    path = pending_queue_path()
    _write_json_file(path, queue)


def _strip_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"snapshot", "snapshot_content", "diff", "diff_content"}
    }


def _pending_artifact_dir(change_id: str) -> Path:
    return pending_changes_dir() / change_id


def _safe_remove_pending_artifacts(change_id: str) -> None:
    if not change_id:
        return
    target = _pending_artifact_dir(change_id)
    try:
        root = pending_changes_dir().resolve()
        resolved = target.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return
    shutil.rmtree(target, ignore_errors=True)


def _write_pending_artifacts(change: dict[str, Any]) -> dict[str, Any]:
    payload = change.get("payload") if isinstance(change.get("payload"), dict) else {}
    change_id = str(change.get("id", ""))
    pending_dir = _pending_artifact_dir(change_id)
    pending_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = _strip_artifact_payload(payload)
    enriched = {**change, "payload": manifest_payload}

    snapshot = payload.get("snapshot", payload.get("snapshot_content"))
    if isinstance(snapshot, str):
        snapshot_path = pending_dir / "snapshot" / "SKILL.md"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(snapshot, encoding="utf-8")
        enriched["snapshot_path"] = str(snapshot_path)

    diff = payload.get("diff", payload.get("diff_content"))
    if isinstance(diff, str):
        diff_path = pending_dir / "diff.md"
        diff_path.write_text(diff, encoding="utf-8")
        enriched["diff_path"] = str(diff_path)

    manifest = {
        **enriched,
        "manifest_path": "manifest.json",
    }
    if enriched.get("snapshot_path"):
        manifest["snapshot_path"] = "snapshot/SKILL.md"
    if enriched.get("diff_path"):
        manifest["diff_path"] = "diff.md"
    _write_json_file(pending_dir / "manifest.json", manifest)
    enriched["manifest_path"] = str(pending_dir / "manifest.json")
    return _normalize_change(enriched)


def _read_manifest_change(manifest_path: Path) -> dict[str, Any] | None:
    data = _read_json_file(manifest_path)
    if not isinstance(data, dict) or not data.get("id"):
        return None

    base = manifest_path.parent
    change_id = str(data.get("id", ""))
    if change_id != base.name:
        return None
    change = dict(data)
    for key in ("manifest_path", "snapshot_path", "diff_path"):
        value = change.get(key)
        if not value:
            continue
        resolved = _artifact_path(change_id, value)
        if resolved is None:
            return None
        change[key] = str(resolved)
    return _normalize_change(change)


def _read_manifest_changes() -> list[dict[str, Any]]:
    root = pending_changes_dir()
    if not root.exists():
        return []
    changes: list[dict[str, Any]] = []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        change = _read_manifest_change(entry / "manifest.json")
        if change:
            changes.append(change)
    return changes


def _merge_changes(manifest_changes: list[dict[str, Any]], queue_changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for change in queue_changes:
        merged[change["id"]] = change
    for change in manifest_changes:
        merged[change["id"]] = {**merged.get(change["id"], {}), **change}
    return sorted(merged.values(), key=lambda change: change.get("created_at", ""))


def _parse_created_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def queue_pending_change(action: str, name: str, payload: dict) -> dict:
    """Append a background skill evolution change to the pending queue."""
    change = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "name": name,
        "payload": payload if isinstance(payload, dict) else {},
        "origin": "background_review",
        "status": "pending",
    }

    with _queue_lock():
        change = _write_pending_artifacts(change)
        queue = _read_queue()
        queue["changes"].append(change)
        _write_queue(queue)
    return {"success": True, **change}


def _list_pending_changes_unlocked() -> list[dict]:
    queue_changes = _read_queue()["changes"]
    manifest_changes = _read_manifest_changes()
    return _merge_changes(manifest_changes, queue_changes)


def list_pending_changes() -> list[dict]:
    """Return pending changes, tolerating absent or malformed queue files."""
    with _queue_lock():
        return _list_pending_changes_unlocked()


def approve_pending_change(change_id: str, apply_func: Callable[[dict], dict]) -> dict:
    """Apply and remove a pending change only when the apply callback succeeds."""
    with _queue_lock():
        queue = {"changes": _list_pending_changes_unlocked()}
        for index, change in enumerate(queue["changes"]):
            if change.get("id") != change_id:
                continue

            if change.get("status") == "applying":
                return {
                    "success": False,
                    "change_id": change_id,
                    "error": (
                        "Pending change is already marked as applying. "
                        "It may have been applied before queue cleanup failed; "
                        "reject it after inspection or recreate the proposal."
                    ),
                }

            applying_change = _write_pending_artifacts({**change, "status": "applying"})
            queue["changes"][index] = applying_change
            try:
                _write_queue(queue)
            except BaseException:
                _write_pending_artifacts({**change, "status": "pending"})
                raise

            apply_result = apply_func(change)
            if not isinstance(apply_result, dict):
                apply_result = {"success": False, "error": "Apply function returned a non-dict result"}
            if not apply_result.get("success"):
                queue["changes"][index] = _write_pending_artifacts({**applying_change, "status": "pending"})
                _write_queue(queue)
                return {
                    "success": False,
                    "change_id": change_id,
                    "apply_result": apply_result,
                }

            del queue["changes"][index]
            _write_queue(queue)
            _safe_remove_pending_artifacts(change_id)
            return {
                "success": True,
                "applied_change_id": change_id,
                "apply_result": apply_result,
            }

    return {"success": False, "error": f"Pending change not found: {change_id}"}


def reject_pending_change(change_id: str) -> dict:
    """Remove a pending change without applying it."""
    with _queue_lock():
        queue = {"changes": _list_pending_changes_unlocked()}
        for index, change in enumerate(queue["changes"]):
            if change.get("id") != change_id:
                continue

            del queue["changes"][index]
            _write_queue(queue)
            _safe_remove_pending_artifacts(change_id)
            return {"success": True, "rejected_change_id": change_id}

    return {"success": False, "error": f"Pending change not found: {change_id}"}


def cleanup_expired_pending_changes(ttl_days: int | None = None) -> list[str]:
    """Remove pending changes older than ``ttl_days`` and return their ids."""
    if ttl_days is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            ttl_days = int((cfg.get("skills") or {}).get("pending_ttl_days", 30))
        except Exception:
            ttl_days = 30
    try:
        ttl_days = int(ttl_days)
    except (TypeError, ValueError):
        ttl_days = 30
    if ttl_days <= 0:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    removed: list[str] = []
    with _queue_lock():
        changes = _list_pending_changes_unlocked()
        kept: list[dict[str, Any]] = []
        for change in changes:
            created = _parse_created_at(change.get("created_at"))
            if created is not None and created < cutoff:
                change_id = str(change.get("id", ""))
                if change_id:
                    removed.append(change_id)
                    _safe_remove_pending_artifacts(change_id)
                continue
            kept.append(change)
        _write_queue({"changes": kept})
    return removed
