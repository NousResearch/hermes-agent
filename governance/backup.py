"""Verified backup helpers for governance-controlled edits."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shlex
import shutil
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from .evidence import append_hash_chained_event, sha256_file, utc_now_iso


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _path_hash(path: Path) -> str:
    return _sha256_tree(path) if path.is_dir() else sha256_file(path)


def _parse_utc_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _path_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "regular_file"
    if path.exists():
        return "special"
    return "missing"


def default_backup_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return get_hermes_home() / "governance" / "backups" / stamp


def create_verified_backup(source: str | Path, *, backup_root: str | Path | None = None, operation: str = "governance edit") -> Dict[str, Any]:
    """Copy *source* to *backup_root*, verify bytes/tree hash, and log manifest.

    This helper is deliberately side-effect limited: it only reads the source,
    writes the backup copy/manifest, and appends a hash-chained governance row.
    Restoring a backup remains an explicit live action requiring normal approval.
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    if not source_path.is_file() and not source_path.is_dir():
        raise ValueError(f"Unsupported backup path type: {source_path}")

    root = Path(backup_root).expanduser().resolve() if backup_root is not None else default_backup_root().resolve()
    root.mkdir(parents=True, exist_ok=True)
    backup_path = root / source_path.name
    if backup_path.exists():
        suffix = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:10]
        backup_path = root / f"{source_path.name}.{suffix}"

    if source_path.is_dir():
        shutil.copytree(source_path, backup_path, symlinks=True)
    else:
        shutil.copy2(source_path, backup_path)

    original_hash = _path_hash(source_path)
    backup_hash = _path_hash(backup_path)
    verified = original_hash == backup_hash
    manifest = {
        "schema_version": "governance.backup.v1",
        "event_type": "verified_backup",
        "created_at_utc": utc_now_iso(),
        "operation": operation,
        "original_path": str(source_path),
        "backup_path": str(backup_path),
        "path_type": _path_type(source_path),
        "original_sha256": original_hash,
        "backup_sha256": backup_hash,
        "verified_exact_match": verified,
        "size_bytes": source_path.stat().st_size if source_path.is_file() else None,
        "rollback_argv_live_requires_approval": ["cp", "-a", str(backup_path), str(source_path)],
        "rollback_command_live_requires_approval": f"cp -a {shlex.quote(str(backup_path))} {shlex.quote(str(source_path))}",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    append_hash_chained_event("backup_manifests", manifest)
    return manifest


def has_recent_verified_backup(path: str | Path, *, max_age_seconds: int = 24 * 60 * 60) -> bool:
    """Return whether the governance log has a current, intact backup for *path*.

    The check is intentionally conservative.  A manifest row alone is not proof:
    the source must still exist, the manifest must be fresh, the backup path must
    still exist, and recorded source/backup hashes must match the actual bytes.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or (not source_path.is_file() and not source_path.is_dir()):
        return False
    log_path = get_hermes_home() / "governance" / "backup_manifests.jsonl"
    if not log_path.exists():
        return False
    try:
        current_hash = _path_hash(source_path)
    except OSError:
        return False

    now = datetime.now(timezone.utc)
    newest_ok = False
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("original_path") != str(source_path):
                continue
            if not row.get("verified_exact_match"):
                continue
            created_at = _parse_utc_timestamp(row.get("created_at_utc") or row.get("timestamp_utc"))
            if created_at is None:
                continue
            age_seconds = (now - created_at).total_seconds()
            if age_seconds < -300 or age_seconds > max_age_seconds:
                continue
            if row.get("original_sha256") != current_hash:
                continue
            backup_path_value = row.get("backup_path")
            if not isinstance(backup_path_value, str) or not backup_path_value:
                continue
            backup_path = Path(backup_path_value).expanduser()
            if not backup_path.exists():
                continue
            if source_path.is_file() and not backup_path.is_file():
                continue
            if source_path.is_dir() and not backup_path.is_dir():
                continue
            try:
                actual_backup_hash = _path_hash(backup_path)
            except OSError:
                continue
            if row.get("backup_sha256") != actual_backup_hash:
                continue
            if row.get("backup_sha256") != row.get("original_sha256"):
                continue
            newest_ok = True
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return newest_ok
