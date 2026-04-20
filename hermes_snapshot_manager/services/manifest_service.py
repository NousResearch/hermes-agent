from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from hermes_snapshot_manager.core.hashing import sha256_file


ISO = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def build_manifest(snapshot_id: str, source_root: Path, files_root: Path, label: str | None, trigger_type: str, file_paths: Iterable[Path]) -> dict:
    files: list[dict] = []
    for path in sorted(file_paths):
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(files_root)),
                "sha256": sha256_file(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "mode": oct(stat.st_mode & 0o777),
                "file_type": "file" if path.is_file() else "other",
            }
        )
    return {
        "snapshot_id": snapshot_id,
        "created_at": utc_now(),
        "source_root": str(source_root),
        "label": label,
        "trigger_type": trigger_type,
        "files": files,
    }


def write_manifest(manifest: dict, output_path: Path) -> str:
    raw = json.dumps(manifest, indent=2, sort_keys=True)
    output_path.write_text(raw, encoding="utf-8")
    return sha256_file(output_path)
