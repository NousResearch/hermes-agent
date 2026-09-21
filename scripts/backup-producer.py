#!/usr/bin/env python3
"""Daily backup producer — creates timestamped archive of critical KenseiAgent state.

Run as no_agent cron script. Silent when successful.
Output goes to /home/kensei/backups/daily/kensei-YYYYMMDD-HHMM.tar.gz
"""
import os, sys, tarfile, json, io
from datetime import datetime, timezone
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
BACKUP_ROOT = Path(os.path.expanduser("~/backups/daily"))
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
ARCHIVE_NAME = f"kensei-{TIMESTAMP}.tar.gz"
ARCHIVE_PATH = BACKUP_ROOT / ARCHIVE_NAME
MANIFEST_PATH = BACKUP_ROOT / f"kensei-{TIMESTAMP}.manifest.json"

# Paths to include — relative to HERMES_HOME
INCLUDE_PATHS = [
    "config.yaml",
    "kanban.db",
    "cron/jobs.json",
    "governance/",
    "SOUL.md",
]
# auth.json excluded — contains API keys and credentials.
# Backup of secrets must use a separate encrypted path.

def build_manifest(archive_path: Path, files: list[tuple[str, int]]) -> dict:
    import hashlib
    sha = hashlib.sha256()
    with open(archive_path, "rb") as af:
        for chunk in iter(lambda: af.read(8192), b""):
            sha.update(chunk)
    return {
        "archive": str(archive_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(s for _, s in files),
        "sha256": sha.hexdigest(),
        "files": [{"path": p, "size": s} for p, s in files],
    }

def main():
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    
    included = []
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        for pattern in INCLUDE_PATHS:
            # Support directory patterns ending with /
            if pattern.endswith("/"):
                dir_path = HERMES_HOME / pattern.rstrip("/")
                if dir_path.is_dir():
                    for f in dir_path.rglob("*"):
                        if f.is_file() and "__pycache__" not in str(f):
                            rel = str(f.relative_to(HERMES_HOME))
                            size = f.stat().st_size
                            tar.add(f, arcname=rel)
                            included.append((rel, size))
            else:
                f = HERMES_HOME / pattern
                if f.is_file():
                    tar.add(f, arcname=pattern)
                    included.append((pattern, f.stat().st_size))
    
    # Write manifest
    manifest = build_manifest(ARCHIVE_PATH, included)
    with open(MANIFEST_PATH, "w") as mf:
        json.dump(manifest, mf, indent=2)
    
    # Signal success — silent when healthy
    print(f"Backup created: {ARCHIVE_NAME} ({manifest['file_count']} files, {manifest['total_bytes']} bytes)")

if __name__ == "__main__":
    main()
