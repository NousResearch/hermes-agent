"""Hermes home data vault guard.

This module protects user-owned identity and memory files from update,
distribution, setup, and uninstall flows.  It intentionally depends only on
the standard library so it can run before optional Hermes dependencies are
available.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


class VaultGuardError(RuntimeError):
    """Raised when a required vault backup or restore step cannot complete."""


VAULT_BACKUP_ENV = "HERMES_VAULT_BACKUP_ROOT"
DEFAULT_KEEP_BACKUPS = 30

ESSENTIAL_RELATIVE_PATHS: tuple[str, ...] = (
    "SOUL.md",
    "memories/MEMORY.md",
    "memories/USER.md",
)

PROTECTED_RELATIVE_PATHS: tuple[str, ...] = (
    *ESSENTIAL_RELATIVE_PATHS,
    "config.yaml",
    ".env",
    "auth.json",
    "cron",
    "skills",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().strftime("%Y%m%d-%H%M%S-%f")


def _safe_reason(reason: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in reason.strip())
    cleaned = cleaned.strip(".-_")
    return cleaned[:64] or "manual"


def _coerce_home(hermes_home: str | os.PathLike[str] | None = None) -> Path:
    if hermes_home is not None:
        return Path(hermes_home).expanduser().resolve(strict=False)
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()).expanduser().resolve(strict=False)
    except Exception:
        raw = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        return Path(raw).expanduser().resolve(strict=False)


def _find_workspace_root() -> Path | None:
    here = Path(__file__).resolve(strict=False)
    for parent in here.parents:
        if (parent / "runtime").is_dir() and (parent / "local-state").is_dir():
            return parent
    return None


def get_vault_backup_root(hermes_home: str | os.PathLike[str] | None = None) -> Path:
    """Return the backup root for protected Hermes data snapshots."""
    override = os.environ.get(VAULT_BACKUP_ENV)
    if override:
        return Path(override).expanduser().resolve(strict=False)
    workspace = _find_workspace_root()
    if workspace is not None:
        return workspace / "local-state" / "hermes-vault-backups"
    return _coerce_home(hermes_home) / "backups" / "vault"


def _relative_target(home: Path, relative_path: str) -> Path:
    target = home / Path(relative_path)
    try:
        target.resolve(strict=False).relative_to(home.resolve(strict=False))
    except ValueError as exc:
        raise VaultGuardError(f"protected path escapes HERMES_HOME: {relative_path}") from exc
    return target


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarize_dir(path: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*"), key=lambda p: str(p.relative_to(path)).lower()):
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix()
        file_count += 1
        size = child.stat().st_size
        total_bytes += size
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return {
        "file_count": file_count,
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _copy_into_snapshot(src: Path, dst: Path) -> None:
    if src.is_dir() and not src.is_symlink():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _entry_for_source(home: Path, snapshot_root: Path, relative_path: str) -> dict[str, Any]:
    source = _relative_target(home, relative_path)
    snapshot = snapshot_root / Path(relative_path)
    entry: dict[str, Any] = {
        "relative_path": relative_path,
        "source": str(source),
        "snapshot": str(snapshot),
    }

    if not source.exists():
        entry.update({"kind": "missing", "status": "missing"})
        return entry

    try:
        _copy_into_snapshot(source, snapshot)
        if source.is_dir() and not source.is_symlink():
            entry.update({"kind": "directory", "status": "backed_up"})
            entry.update(_summarize_dir(source))
        else:
            entry.update(
                {
                    "kind": "file",
                    "status": "backed_up",
                    "bytes": source.stat().st_size,
                    "sha256": _file_sha256(source),
                }
            )
    except Exception as exc:
        entry.update({"kind": "error", "status": "error", "error": repr(exc)})
    return entry


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _entry_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("relative_path")): entry
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict) and entry.get("relative_path")
    }


def create_vault_backup(
    hermes_home: str | os.PathLike[str] | None = None,
    *,
    reason: str = "manual",
    protected_paths: Iterable[str] = PROTECTED_RELATIVE_PATHS,
    required_paths: Iterable[str] = ESSENTIAL_RELATIVE_PATHS,
    keep: int = DEFAULT_KEEP_BACKUPS,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Create a protected snapshot and return its manifest.

    ``raise_on_error`` raises only when a required existing path fails to copy;
    missing required paths are recorded but do not prevent capturing whatever
    still exists.
    """
    home = _coerce_home(hermes_home)
    backup_root = get_vault_backup_root(home)
    backup_dir = backup_root / f"{_timestamp()}-{os.getpid()}-{_safe_reason(reason)}"
    snapshot_root = backup_dir / "snapshot"
    backup_dir.mkdir(parents=True, exist_ok=False)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    entries = [_entry_for_source(home, snapshot_root, rel) for rel in dict.fromkeys(protected_paths)]
    now = _utc_now().isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now,
        "reason": reason,
        "hermes_home": str(home),
        "backup_dir": str(backup_dir),
        "snapshot_root": str(snapshot_root),
        "entries": entries,
    }

    errors = [entry for entry in entries if entry.get("status") == "error"]
    required = set(required_paths)
    required_errors = [entry for entry in errors if entry.get("relative_path") in required]
    manifest["errors"] = errors
    manifest["required_errors"] = required_errors

    manifest_path = backup_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    (backup_root / "LATEST.txt").write_text(str(manifest_path), encoding="utf-8")
    _prune_backups(backup_root, keep=keep)

    if raise_on_error and required_errors:
        raise VaultGuardError(
            "failed to back up required Hermes data: "
            + ", ".join(str(entry.get("relative_path")) for entry in required_errors)
        )
    return manifest


def _manifest_sort_key(item: tuple[Path, dict[str, Any]]) -> str:
    return str(item[1].get("created_at") or item[0].stat().st_mtime_ns)


def list_vault_backups(
    hermes_home: str | os.PathLike[str] | None = None,
    *,
    backup_root: str | os.PathLike[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    root = Path(backup_root).expanduser().resolve(strict=False) if backup_root else get_vault_backup_root(hermes_home)
    if not root.is_dir():
        return []
    found: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in root.glob("*/manifest.json"):
        manifest = _load_manifest(manifest_path)
        if manifest is not None:
            found.append((manifest_path.parent, manifest))
    return sorted(found, key=_manifest_sort_key, reverse=True)


def _prune_backups(backup_root: Path, *, keep: int) -> None:
    if keep <= 0:
        return
    backups = list_vault_backups(backup_root=backup_root)
    for backup_dir, _manifest in backups[keep:]:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _snapshot_path(backup_dir: Path, relative_path: str) -> Path:
    return backup_dir / "snapshot" / Path(relative_path)


def _delete_target_for_restore(target: Path) -> None:
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)


def restore_from_backup(
    backup_dir: str | os.PathLike[str],
    hermes_home: str | os.PathLike[str] | None = None,
    *,
    relative_paths: Iterable[str] = ESSENTIAL_RELATIVE_PATHS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Restore selected paths from a specific vault snapshot."""
    home = _coerce_home(hermes_home)
    backup = Path(backup_dir).expanduser().resolve(strict=False)
    results: list[dict[str, Any]] = []
    for rel in dict.fromkeys(relative_paths):
        src = _snapshot_path(backup, rel)
        dst = _relative_target(home, rel)
        item: dict[str, Any] = {"relative_path": rel, "source": str(src), "target": str(dst)}
        if not src.exists():
            item["status"] = "missing_in_backup"
            results.append(item)
            continue
        if dst.exists() and not overwrite:
            item["status"] = "skipped_existing"
            results.append(item)
            continue
        try:
            if dst.exists():
                _delete_target_for_restore(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_into_snapshot(src, dst)
            item["status"] = "restored"
        except Exception as exc:
            item.update({"status": "error", "error": repr(exc)})
        results.append(item)
    return {"backup_dir": str(backup), "hermes_home": str(home), "results": results}


def restore_from_latest_backup(
    hermes_home: str | os.PathLike[str] | None = None,
    *,
    relative_paths: Iterable[str] = ESSENTIAL_RELATIVE_PATHS,
    overwrite: bool = False,
) -> dict[str, Any]:
    backups = list_vault_backups(hermes_home)
    if not backups:
        return {"hermes_home": str(_coerce_home(hermes_home)), "results": [], "error": "no_backups"}
    backup_dir, _manifest = backups[0]
    return restore_from_backup(backup_dir, hermes_home, relative_paths=relative_paths, overwrite=overwrite)


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _same_text(a: str, b: str) -> bool:
    return a.strip().replace("\r\n", "\n") == b.strip().replace("\r\n", "\n")


def _find_backup_with_snapshot(
    hermes_home: Path,
    relative_path: str,
    predicate: Callable[[Path], bool] | None = None,
) -> Path | None:
    for backup_dir, manifest in list_vault_backups(hermes_home):
        entries = _entry_map(manifest)
        entry = entries.get(relative_path)
        if entry and entry.get("status") != "backed_up":
            continue
        snapshot = _snapshot_path(backup_dir, relative_path)
        if not snapshot.exists():
            continue
        if predicate is not None and not predicate(snapshot):
            continue
        return backup_dir
    return None


def run_startup_integrity_check(
    hermes_home: str | os.PathLike[str] | None = None,
    *,
    default_soul_text: str | None = None,
    suspicious_memory_bytes: int = 256,
) -> dict[str, Any]:
    """Restore obvious identity/memory loss from the most recent good snapshot.

    This is deliberately conservative: it restores missing files, a SOUL.md
    that is exactly the default template, and a very small MEMORY.md only when
    a larger backup exists.
    """
    home = _coerce_home(hermes_home)
    actions: list[dict[str, Any]] = []

    def file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    soul = _relative_target(home, "SOUL.md")
    if not soul.exists():
        backup = _find_backup_with_snapshot(home, "SOUL.md", lambda p: file_size(p) > 20)
        if backup:
            actions.append({"relative_path": "SOUL.md", "backup_dir": backup, "reason": "missing"})
    elif default_soul_text and _same_text(_read_text_or_empty(soul), default_soul_text):
        backup = _find_backup_with_snapshot(
            home,
            "SOUL.md",
            lambda p: file_size(p) > 20 and not _same_text(_read_text_or_empty(p), default_soul_text),
        )
        if backup:
            actions.append({"relative_path": "SOUL.md", "backup_dir": backup, "reason": "default-template"})

    user_md = _relative_target(home, "memories/USER.md")
    if not user_md.exists():
        backup = _find_backup_with_snapshot(home, "memories/USER.md", lambda p: file_size(p) > 0)
        if backup:
            actions.append({"relative_path": "memories/USER.md", "backup_dir": backup, "reason": "missing"})

    memory_md = _relative_target(home, "memories/MEMORY.md")
    if not memory_md.exists():
        backup = _find_backup_with_snapshot(home, "memories/MEMORY.md", lambda p: file_size(p) > 0)
        if backup:
            actions.append({"relative_path": "memories/MEMORY.md", "backup_dir": backup, "reason": "missing"})
    elif file_size(memory_md) < suspicious_memory_bytes:
        current_size = file_size(memory_md)
        backup = _find_backup_with_snapshot(
            home,
            "memories/MEMORY.md",
            lambda p: file_size(p) > max(suspicious_memory_bytes, current_size + 64),
        )
        if backup:
            actions.append({"relative_path": "memories/MEMORY.md", "backup_dir": backup, "reason": "suspicious-small"})

    if not actions:
        return {"hermes_home": str(home), "actions": [], "results": []}

    create_vault_backup(home, reason="pre-autorestore", keep=max(DEFAULT_KEEP_BACKUPS, 60))
    results: list[dict[str, Any]] = []
    for action in actions:
        restore_result = restore_from_backup(
            action["backup_dir"],
            home,
            relative_paths=(action["relative_path"],),
            overwrite=True,
        )
        for result in restore_result.get("results", []):
            result["reason"] = action["reason"]
            results.append(result)
    return {"hermes_home": str(home), "actions": actions, "results": results}


def verify_recent_backup_manifest(
    manifest_path: str | os.PathLike[str],
    hermes_home: str | os.PathLike[str] | None = None,
    *,
    max_age_minutes: int = 15,
) -> dict[str, Any]:
    """Validate that a destructive action has a fresh vault backup."""
    home = _coerce_home(hermes_home)
    path = Path(manifest_path).expanduser().resolve(strict=False)
    manifest = _load_manifest(path)
    if manifest is None:
        raise VaultGuardError(f"vault backup manifest is unreadable: {path}")
    if Path(str(manifest.get("hermes_home", ""))).resolve(strict=False) != home:
        raise VaultGuardError("vault backup was created for a different HERMES_HOME")
    created_raw = str(manifest.get("created_at") or "")
    try:
        created = datetime.fromisoformat(created_raw)
    except ValueError as exc:
        raise VaultGuardError("vault backup manifest has an invalid timestamp") from exc
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_seconds = (_utc_now() - created.astimezone(timezone.utc)).total_seconds()
    if age_seconds > max_age_minutes * 60:
        raise VaultGuardError("vault backup is too old for this destructive action")
    required_errors = manifest.get("required_errors") or []
    if required_errors:
        raise VaultGuardError("vault backup has required copy errors")
    return manifest
