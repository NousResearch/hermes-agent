"""Fleet control — batch start / stop / clean all Hermes gateway profiles.

Operates on macOS launchd LaunchAgents (``ai.hermes.gateway*.plist``).
Designed to be called from the ``/all`` gateway slash command.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLIST_GLOB = "ai.hermes.gateway*.plist"
_LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def _discover_plist_files() -> List[Path]:
    """Return all Hermes gateway plist files sorted by name."""
    return sorted(_LAUNCH_AGENTS_DIR.glob(_PLIST_GLOB))


def _label_from_plist(plist_path: Path) -> str:
    """Derive the launchd label from a plist filename (strip .plist extension)."""
    return plist_path.stem


def _profile_from_label(label: str) -> str:
    """Extract human-readable profile name from a launchd label.

    ``ai.hermes.gateway``       → ``default``
    ``ai.hermes.gateway-dd``    → ``dd``
    """
    prefix = "ai.hermes.gateway"
    if label == prefix:
        return "default"
    if label.startswith(prefix + "-"):
        return label[len(prefix) + 1:]
    return label


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def fleet_stop(exclude_profiles: set[str] | None = None) -> Tuple[List[str], List[str], List[str]]:
    """Unload Hermes gateway LaunchAgents.

    Args:
        exclude_profiles: Profile names to skip (e.g. the bot receiving the command).

    Returns (stopped, skipped, errors).
    """
    domain = _launchd_domain()
    plists = _discover_plist_files()
    exclude = {str(p or "").strip().lower() for p in (exclude_profiles or set())}
    stopped: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    for plist in plists:
        label = _label_from_plist(plist)
        profile = _profile_from_label(label)
        if profile in exclude:
            skipped.append(profile)
            continue
        target = f"{domain}/{label}"
        try:
            subprocess.run(
                ["launchctl", "disable", target],
                check=False, timeout=30,
                capture_output=True,
            )
            subprocess.run(
                ["launchctl", "bootout", target],
                check=False, timeout=30,
                capture_output=True,
            )
            logger.info("[fleet] Unloaded %s (%s)", label, profile)
            stopped.append(profile)
        except Exception as exc:
            logger.warning("[fleet] Failed to unload %s: %s", label, exc)
            errors.append(f"{profile}: {exc}")

    return stopped, skipped, errors


def fleet_start(exclude_profiles: set[str] | None = None) -> Tuple[List[str], List[str]]:
    """Bootstrap + kickstart all Hermes gateway LaunchAgents.

    Args:
        exclude_profiles: Profile names to skip.

    Returns (started, errors).
    """
    domain = _launchd_domain()
    plists = _discover_plist_files()
    exclude = {str(p or "").strip().lower() for p in (exclude_profiles or set())}
    started: List[str] = []
    errors: List[str] = []

    for plist in plists:
        label = _label_from_plist(plist)
        profile = _profile_from_label(label)
        if profile in exclude:
            continue
        target = f"{domain}/{label}"
        try:
            subprocess.run(
                ["launchctl", "enable", target],
                check=False, timeout=30,
                capture_output=True,
            )
            # bootout first (ignore errors if not loaded)
            subprocess.run(
                ["launchctl", "bootout", target],
                check=False, timeout=30,
                capture_output=True,
            )
            # bootstrap to load
            subprocess.run(
                ["launchctl", "bootstrap", domain, str(plist)],
                check=True, timeout=30,
                capture_output=True,
            )
            subprocess.run(
                ["launchctl", "kickstart", "-k", target],
                check=False, timeout=30,
                capture_output=True,
            )
            logger.info("[fleet] Started %s (%s)", label, profile)
            started.append(profile)
        except Exception as exc:
            logger.warning("[fleet] Failed to start %s: %s", label, exc)
            errors.append(f"{profile}: {exc}")

    return started, errors


def fleet_clean() -> List[str]:
    """Clean relay queues, session state, and active tasks for a fresh start.

    Returns list of cleaned items (for reporting).
    """
    cleaned: List[str] = []
    hermes_home = Path.home() / ".hermes"
    backup_root = _new_backup_root(hermes_home)
    cleaned.append(f"backup: {backup_root}")

    # 1. Clear profile/global sessions: this is the fleet-wide equivalent of /new.
    _reset_dir(hermes_home / "sessions", backup_root / "global", "global/sessions", cleaned)
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for profile_home in sorted(profiles_dir.iterdir()):
            if not profile_home.is_dir() or "backup" in profile_home.name.lower():
                continue
            _reset_dir(
                profile_home / "sessions",
                backup_root / "profiles" / profile_home.name,
                f"{profile_home.name}/sessions",
                cleaned,
            )
            dedup = profile_home / "feishu_seen_message_ids.json"
            if dedup.exists():
                _backup_file(dedup, backup_root / "profiles" / profile_home.name / dedup.name)
                dedup.unlink()
                cleaned.append(f"{profile_home.name}/dedup")

    # 2. Clear cross-bot relay runtime state.
    relay_dir = hermes_home / "cross_bot_relay"
    if relay_dir.is_dir():
        relay_backup = backup_root / "cross_bot_relay"
        relay_backup.mkdir(parents=True, exist_ok=True)
        for name in ("_task_events.jsonl", "_registry.json", "_role_registry.json", "group_profiles.json"):
            src = relay_dir / name
            if src.exists():
                _backup_file(src, relay_backup / name)

        task_events = relay_dir / "_task_events.jsonl"
        task_events.parent.mkdir(parents=True, exist_ok=True)
        task_events.write_text("", encoding="utf-8")
        cleaned.append("relay/task_events: cleared")

        for claim_dir_name in ("_event_claims", "_system_command_claims"):
            claim_dir = relay_dir / claim_dir_name
            count = _delete_files_under(claim_dir)
            if count:
                cleaned.append(f"relay/{claim_dir_name}: {count} files")

        for profile_dir in sorted(relay_dir.iterdir()):
            if not profile_dir.is_dir() or profile_dir.name.startswith("_"):
                continue
            count = _delete_files_under(profile_dir)
            if count:
                cleaned.append(f"relay/{profile_dir.name}: {count} files")

    # 3. Mark queued/running JM tasks cancelled.
    tasks_db = hermes_home / "jm_tasks.db"
    if tasks_db.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(tasks_db))
            cursor = conn.execute(
                "UPDATE tasks SET status='cancelled' WHERE status IN ('pending', 'queued', 'running')"
            )
            if cursor.rowcount > 0:
                cleaned.append(f"jm_tasks.db: cancelled {cursor.rowcount} active tasks")
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("[fleet] Failed to clean jm_tasks.db: %s", exc)

    return cleaned


def fleet_start_clean_detached(delay_seconds: float = 1.5) -> Path:
    """Start a detached fleet clean+restart process and return its log path."""
    hermes_home = Path.home() / ".hermes"
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fleet_all_start_clean.log"
    project_root = Path(__file__).resolve().parents[1]
    with log_path.open("a", encoding="utf-8") as log_fh:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hermes_cli.fleet_ctl",
                "start-clean",
                "--delay",
                str(delay_seconds),
            ],
            cwd=str(project_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return log_path


def _new_backup_root(hermes_home: Path) -> Path:
    backups = hermes_home / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = backups / f"new_all_roles_{stamp}"
    suffix = 1
    while root.exists():
        suffix += 1
        root = backups / f"new_all_roles_{stamp}_{suffix}"
    root.mkdir(parents=True)
    return root


def _backup_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _reset_dir(path: Path, backup_parent: Path, label: str, cleaned: List[str]) -> None:
    count = sum(1 for f in path.rglob("*") if f.is_file()) if path.exists() else 0
    if path.exists():
        backup_parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(backup_parent / path.name))
    path.mkdir(parents=True, exist_ok=True)
    cleaned.append(f"{label}: {count} files")


def _delete_files_under(path: Path) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            child.unlink()
            count += 1
    return count


def _run_start_clean(delay_seconds: float) -> int:
    time.sleep(max(0.0, delay_seconds))
    print(f"[{datetime.now().isoformat()}] /all start c: stopping fleet", flush=True)
    stopped, skipped, stop_errors = fleet_stop()
    print(f"stopped={len(stopped)} skipped={len(skipped)} errors={len(stop_errors)}", flush=True)
    print(f"[{datetime.now().isoformat()}] /all start c: cleaning state", flush=True)
    cleaned = fleet_clean()
    for item in cleaned:
        print(f"cleaned: {item}", flush=True)
    print(f"[{datetime.now().isoformat()}] /all start c: starting fleet", flush=True)
    started, start_errors = fleet_start()
    print(f"started={len(started)} errors={len(start_errors)}", flush=True)
    for err in stop_errors + start_errors:
        print(f"error: {err}", flush=True)
    return 1 if (stop_errors or start_errors) else 0


def main(argv: List[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("Usage: python -m hermes_cli.fleet_ctl start-clean [--delay seconds]")
        return 2
    command = args.pop(0)
    if command != "start-clean":
        print(f"Unknown command: {command}")
        return 2
    delay = 1.5
    if "--delay" in args:
        idx = args.index("--delay")
        try:
            delay = float(args[idx + 1])
        except (IndexError, ValueError):
            print("Invalid --delay value")
            return 2
    return _run_start_clean(delay)


if __name__ == "__main__":
    raise SystemExit(main())
