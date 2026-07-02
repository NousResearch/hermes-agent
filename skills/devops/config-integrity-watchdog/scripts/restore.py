#!/usr/bin/env python3
"""Restore config.yaml to the canonical sealed baseline.

Reads the sealed hash and config path from the integrity log,
restores the config from the dotfiles repo, then re-seals.

Note: restoration uses git HEAD of the dotfiles repo. If the tampered config
was committed to dotfiles, the committed tampered version will be restored.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


CONFIG_PATH = Path(os.environ.get("HERMES_CONFIG", "~/.hermes/config.yaml")).expanduser()
DOTFILES_DIR = Path(os.environ.get("HERMES_DOTFILES_DIR", "~/Dev/dotfiles")).expanduser()
LOG_PATH = DOTFILES_DIR / "hermes" / "config_integrity.jsonl"
CANONICAL_CONFIG = DOTFILES_DIR / "hermes" / "config.yaml"


def _hash_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_baseline_entry() -> Optional[dict]:
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
                    baseline = entry
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return baseline


def _git_restore_canonical() -> bool:
    """Restore config.yaml from the last committed version in dotfiles git."""
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", "hermes/config.yaml"],
            cwd=DOTFILES_DIR, check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: git restore failed: {e.stderr.decode()}", file=sys.stderr)
        return False


def _append_log_entry(event: str, **kwargs: object) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> int:
    baseline = _load_baseline_entry()
    if baseline is None:
        print("ERROR: No baseline found. Run seal.py first.", file=sys.stderr)
        return 1

    baseline_hash = baseline["hash"]
    current_hash = _hash_file(CONFIG_PATH)

    if current_hash == baseline_hash:
        print("Config already matches baseline -- no restore needed.")
        return 0

    print(f"Restoring config from git baseline ({baseline_hash[:16]}...)...")

    # Back up tampered config
    backup = CONFIG_PATH.with_suffix(
        f".pre-restore-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(CONFIG_PATH, backup)
    print(f"   Backed up tampered config to: {backup.name}")

    # Log tamper detection
    _append_log_entry("tamper_detected", actual_hash=current_hash, baseline_hash=baseline_hash)

    # Restore from git
    restored = _git_restore_canonical()
    if not restored:
        # Fallback: if config.yaml is a symlink to CANONICAL_CONFIG, and CANONICAL_CONFIG exists,
        # try to restore CANONICAL_CONFIG from git directly
        print("   Attempting direct symlink target restore...", file=sys.stderr)
        target = CONFIG_PATH.resolve() if CONFIG_PATH.is_symlink() else None
        if target and target.exists():
            try:
                subprocess.run(
                    ["git", "checkout", "HEAD", "--", str(target.relative_to(DOTFILES_DIR))],
                    cwd=DOTFILES_DIR, check=True, capture_output=True
                )
                restored = True
            except (subprocess.CalledProcessError, ValueError):
                pass

    if not restored:
        print("ERROR: Could not restore from git. Manual intervention required.", file=sys.stderr)
        _append_log_entry("restore_failed")
        return 1

    # If CONFIG_PATH is not the same file as CANONICAL_CONFIG (e.g. not a symlink into dotfiles),
    # copy the freshly-restored canonical file back to CONFIG_PATH.
    try:
        canonical_resolved = CANONICAL_CONFIG.resolve()
        config_resolved = CONFIG_PATH.resolve() if CONFIG_PATH.exists() else CONFIG_PATH
        if canonical_resolved != config_resolved and CANONICAL_CONFIG.exists():
            shutil.copy2(CANONICAL_CONFIG, CONFIG_PATH)
    except (OSError, ValueError):
        pass

    # Verify restoration
    new_hash = _hash_file(CONFIG_PATH)
    if new_hash != baseline_hash:
        print(
            f"WARNING: Restored hash {new_hash[:16] if new_hash else 'None'} "
            f"doesn't match baseline {baseline_hash[:16]}",
            file=sys.stderr,
        )

    # Re-seal
    _append_log_entry("restore", restored_hash=new_hash)

    # Commit the updated log
    try:
        subprocess.run(
            ["git", "add", str(LOG_PATH.relative_to(DOTFILES_DIR))],
            cwd=DOTFILES_DIR, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m",
             f"integrity: restore config.yaml [{new_hash[:12] if new_hash else 'unknown'}]"],
            cwd=DOTFILES_DIR, check=True, capture_output=True
        )
    except subprocess.CalledProcessError:
        pass  # Log written even if git commit fails

    print(f"Config restored and re-sealed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
