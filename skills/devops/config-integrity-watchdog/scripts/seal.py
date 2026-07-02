#!/usr/bin/env python3
"""Seal the current config.yaml as the canonical baseline.

Appends a signed entry to the integrity log and commits it to the
dotfiles git repository, creating a tamper-evident anchor.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


CONFIG_PATH = Path(os.environ.get("HERMES_CONFIG", "~/.hermes/config.yaml")).expanduser()
DOTFILES_DIR = Path(os.environ.get("HERMES_DOTFILES_DIR", "~/Dev/dotfiles")).expanduser()
LOG_PATH = DOTFILES_DIR / "hermes" / "config_integrity.jsonl"


def _hash_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        print(f"ERROR: Cannot read {path}: {e}", file=sys.stderr)
        return None


def _git_commit(message: str) -> bool:
    try:
        subprocess.run(
            ["git", "add", str(LOG_PATH.relative_to(DOTFILES_DIR))],
            cwd=DOTFILES_DIR, check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=DOTFILES_DIR, capture_output=True
        )
        if result.returncode == 0:
            return True  # Nothing to commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=DOTFILES_DIR, check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: git commit failed: {e.stderr.decode()}", file=sys.stderr)
        return False


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found: {CONFIG_PATH}", file=sys.stderr)
        return 1

    config_hash = _hash_file(CONFIG_PATH)
    if config_hash is None:
        return 1

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "seal",
        "hash": config_hash,
        "config_path": str(CONFIG_PATH),
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    committed = _git_commit(f"integrity: seal config.yaml [{config_hash[:12]}]")
    if not committed:
        print("WARNING: Integrity log written but not committed to git.")
        print("  Fingerprint stored as mutable fallback only.")
    else:
        print(f"Sealed: {config_hash[:16]}...")
        print(f"   Log: {LOG_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
