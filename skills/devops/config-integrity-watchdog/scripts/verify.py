#!/usr/bin/env python3
"""Verify config.yaml integrity against the sealed baseline.

Exit codes:
  0 - Config matches canonical baseline
  1 - Config has been tampered
  2 - Integrity log itself has uncommitted changes (log tampering)
  3 - No baseline found (run seal.py first)
"""
import hashlib
import json
import os
import subprocess
import sys
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


def _log_has_uncommitted_changes() -> bool:
    if not DOTFILES_DIR.is_dir():
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", str(LOG_PATH.relative_to(DOTFILES_DIR))],
            cwd=DOTFILES_DIR, capture_output=True, text=True, check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def _load_baseline() -> Optional[str]:
    if not LOG_PATH.exists():
        return None
    baseline = None
    try:
        for line in LOG_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("event") == "seal":
                    baseline = entry.get("hash")
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return baseline


def main() -> int:
    # Check log file integrity first
    if _log_has_uncommitted_changes():
        print("INTEGRITY LOG TAMPERED")
        print(f"   {LOG_PATH} has uncommitted changes -- the log itself may have been modified.")
        return 2

    baseline = _load_baseline()
    if baseline is None:
        print("No baseline found. Run seal.py first.")
        return 3

    current = _hash_file(CONFIG_PATH)
    if current is None:
        return 1

    if current == baseline:
        print(f"Config integrity OK")
        print(f"   Hash: {current[:16]}... matches sealed baseline")
        return 0
    else:
        print(f"CONFIG TAMPERED")
        print(f"   Baseline: {baseline[:16]}...")
        print(f"   Current:  {current[:16]}...")
        print(f"   Run restore.py to revert to canonical config.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
