"""Shared core logic for config-integrity-watchdog.

Extracted from seal.py, verify.py, and restore.py so that both the
standalone scripts and the Hermes CLI integration can call the same
functions without subprocess indirection.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _default_config_path() -> Path:
    return Path(os.environ.get("HERMES_CONFIG", "~/.hermes/config.yaml")).expanduser()


def _default_dotfiles_dir() -> Path:
    return Path(os.environ.get("HERMES_DOTFILES_DIR", "~/Dev/dotfiles")).expanduser()


def _log_path(dotfiles_dir: Path) -> Path:
    return dotfiles_dir / "hermes" / "config_integrity.jsonl"


def _canonical_config_path(dotfiles_dir: Path) -> Path:
    return dotfiles_dir / "hermes" / "config.yaml"


def hash_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        print(f"ERROR: Cannot read {path}: {e}", file=sys.stderr)
        return None


def git_commit(dotfiles_dir: Path, log_path: Path, message: str) -> bool:
    try:
        subprocess.run(
            ["git", "add", str(log_path.relative_to(dotfiles_dir))],
            cwd=dotfiles_dir, check=True, capture_output=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=dotfiles_dir, capture_output=True,
        )
        if result.returncode == 0:
            return True  # Nothing to commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=dotfiles_dir, check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: git commit failed: {e.stderr.decode()}", file=sys.stderr)
        return False


def append_log_entry(log_path: Path, event: str, **kwargs: object) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_baseline(log_path: Path) -> Optional[str]:
    """Return the most recent sealed hash from the integrity log, or None."""
    if not log_path.exists():
        return None
    baseline = None
    try:
        for line in log_path.read_text().splitlines():
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


def load_baseline_entry(log_path: Path) -> Optional[dict]:
    """Return the most recent seal entry dict from the integrity log, or None."""
    if not log_path.exists():
        return None
    baseline = None
    try:
        for line in log_path.read_text().splitlines():
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


def log_has_uncommitted_changes(dotfiles_dir: Path, log_path: Path) -> bool:
    if not dotfiles_dir.is_dir():
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", str(log_path.relative_to(dotfiles_dir))],
            cwd=dotfiles_dir, capture_output=True, text=True, check=True,
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# High-level operations (return int exit code)
# ---------------------------------------------------------------------------


def seal(
    config_path: Optional[Path] = None,
    dotfiles_dir: Optional[Path] = None,
) -> int:
    """Hash config.yaml and append a seal entry to the integrity log.

    Returns 0 on success, 1 on error.
    """
    config_path = config_path or _default_config_path()
    dotfiles_dir = dotfiles_dir or _default_dotfiles_dir()
    log_path = _log_path(dotfiles_dir)

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    config_hash = hash_file(config_path)
    if config_hash is None:
        return 1

    log_path.parent.mkdir(parents=True, exist_ok=True)

    append_log_entry(log_path, "seal", hash=config_hash, config_path=str(config_path))

    committed = git_commit(
        dotfiles_dir, log_path,
        f"integrity: seal config.yaml [{config_hash[:12]}]",
    )
    if not committed:
        print("WARNING: Integrity log written but not committed to git.")
        print("  Fingerprint stored as mutable fallback only.")
    else:
        print(f"Sealed: {config_hash[:16]}...")
        print(f"   Log: {log_path}")

    return 0


def verify(
    config_path: Optional[Path] = None,
    dotfiles_dir: Optional[Path] = None,
) -> int:
    """Verify config.yaml integrity against the sealed baseline.

    Exit codes:
      0 — Config matches canonical baseline
      1 — Config has been tampered
      2 — Integrity log itself has uncommitted changes (log tampering)
      3 — No baseline found (run seal first)
    """
    config_path = config_path or _default_config_path()
    dotfiles_dir = dotfiles_dir or _default_dotfiles_dir()
    log_path = _log_path(dotfiles_dir)

    if log_has_uncommitted_changes(dotfiles_dir, log_path):
        print("INTEGRITY LOG TAMPERED")
        print(
            f"   {log_path} has uncommitted changes "
            "-- the log itself may have been modified."
        )
        return 2

    baseline = load_baseline(log_path)
    if baseline is None:
        print("No baseline found. Run `hermes config seal` first.")
        return 3

    current = hash_file(config_path)
    if current is None:
        return 1

    if current == baseline:
        print("Config integrity OK")
        print(f"   Hash: {current[:16]}... matches sealed baseline")
        return 0
    else:
        print("CONFIG TAMPERED")
        print(f"   Baseline: {baseline[:16]}...")
        print(f"   Current:  {current[:16]}...")
        print("   Run `hermes config restore` to revert to canonical config.")
        return 1


def restore(
    config_path: Optional[Path] = None,
    dotfiles_dir: Optional[Path] = None,
) -> int:
    """Restore config.yaml to the sealed baseline from git.

    Returns 0 on success, 1 on error.
    """
    config_path = config_path or _default_config_path()
    dotfiles_dir = dotfiles_dir or _default_dotfiles_dir()
    log_path = _log_path(dotfiles_dir)
    canonical_config = _canonical_config_path(dotfiles_dir)

    baseline = load_baseline_entry(log_path)
    if baseline is None:
        print("ERROR: No baseline found. Run `hermes config seal` first.", file=sys.stderr)
        return 1

    baseline_hash = baseline["hash"]
    current_hash = hash_file(config_path)

    if current_hash == baseline_hash:
        print("Config already matches baseline -- no restore needed.")
        return 0

    print(f"Restoring config from git baseline ({baseline_hash[:16]}...)...")

    # Back up tampered config
    backup = config_path.with_suffix(
        f".pre-restore-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(config_path, backup)
    print(f"   Backed up tampered config to: {backup.name}")

    # Log tamper detection
    append_log_entry(
        log_path, "tamper_detected",
        actual_hash=current_hash,
        baseline_hash=baseline_hash,
    )

    # Restore from git
    restored = False
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", "hermes/config.yaml"],
            cwd=dotfiles_dir, check=True, capture_output=True,
        )
        restored = True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: git restore failed: {e.stderr.decode()}", file=sys.stderr)

    if not restored:
        # Fallback: try via symlink target
        print("   Attempting direct symlink target restore...", file=sys.stderr)
        target = config_path.resolve() if config_path.is_symlink() else None
        if target and target.exists():
            try:
                subprocess.run(
                    ["git", "checkout", "HEAD", "--",
                     str(target.relative_to(dotfiles_dir))],
                    cwd=dotfiles_dir, check=True, capture_output=True,
                )
                restored = True
            except (subprocess.CalledProcessError, ValueError):
                pass

    if not restored:
        print(
            "ERROR: Could not restore from git. Manual intervention required.",
            file=sys.stderr,
        )
        append_log_entry(log_path, "restore_failed")
        return 1

    # If CONFIG_PATH is not the same file as CANONICAL_CONFIG, copy it back.
    try:
        canonical_resolved = canonical_config.resolve()
        config_resolved = config_path.resolve() if config_path.exists() else config_path
        if canonical_resolved != config_resolved and canonical_config.exists():
            shutil.copy2(canonical_config, config_path)
    except (OSError, ValueError):
        pass

    new_hash = hash_file(config_path)
    if new_hash != baseline_hash:
        print(
            f"WARNING: Restored hash {new_hash[:16] if new_hash else 'None'} "
            f"doesn't match baseline {baseline_hash[:16]}",
            file=sys.stderr,
        )

    append_log_entry(log_path, "restore", restored_hash=new_hash)

    # Commit the updated log
    try:
        subprocess.run(
            ["git", "add", str(log_path.relative_to(dotfiles_dir))],
            cwd=dotfiles_dir, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m",
             f"integrity: restore config.yaml [{new_hash[:12] if new_hash else 'unknown'}]"],
            cwd=dotfiles_dir, check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        pass  # Log written even if git commit fails

    print("Config restored and re-sealed.")
    return 0
