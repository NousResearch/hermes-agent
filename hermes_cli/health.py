"""
Health command for hermes CLI.

Shows database health, WAL file size, session counts, checkpoint disk usage,
cron job status, and maintenance recommendations.
"""

import os
import time
from pathlib import Path
from typing import List


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (KB / MB / GB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def _dir_size(path: Path) -> int:
    """Return total size in bytes of all files under *path* (best-effort)."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _relative_time(ts) -> str:
    """Format a Unix timestamp or ISO string as a relative age."""
    if ts is None:
        return "never"
    try:
        if isinstance(ts, str):
            from datetime import datetime, timezone
            text = ts.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            ts = parsed.timestamp()
        delta = time.time() - float(ts)
    except Exception:
        return str(ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    if delta < 172800:
        return "yesterday"
    if delta < 604800:
        return f"{int(delta / 86400)}d ago"
    days = int(delta / 86400)
    return f"{days} days ago"


# ---------------------------------------------------------------------------
# Database section
# ---------------------------------------------------------------------------

def _db_info(db_path: Path) -> dict:
    """Collect SQLite database metrics.

    Returns a dict with keys:
      db_size, wal_size, page_count, page_size,
      total_sessions, active_chains, compressed_sessions, total_messages
    """
    result = {
        "db_size": 0,
        "wal_size": 0,
        "page_count": 0,
        "page_size": 4096,
        "total_sessions": 0,
        "active_chains": 0,
        "compressed_sessions": 0,
        "total_messages": 0,
        "error": None,
    }

    try:
        result["db_size"] = db_path.stat().st_size if db_path.exists() else 0
    except OSError:
        pass

    wal_path = db_path.with_suffix(".db-wal")
    try:
        result["wal_size"] = wal_path.stat().st_size if wal_path.exists() else 0
    except OSError:
        pass

    if not db_path.exists():
        return result

    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=3.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("PRAGMA page_count").fetchone()
            if row:
                result["page_count"] = row[0]
            row = conn.execute("PRAGMA page_size").fetchone()
            if row:
                result["page_size"] = row[0]

            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            if row:
                result["total_sessions"] = row[0]

            # Active chains: sessions without a parent (root of a chain)
            row = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NULL"
            ).fetchone()
            if row:
                result["active_chains"] = row[0]

            # Compressed sessions: sessions that have a parent (continuation after compression)
            row = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NOT NULL"
            ).fetchone()
            if row:
                result["compressed_sessions"] = row[0]

            row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            if row:
                result["total_messages"] = row[0]
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Checkpoint section
# ---------------------------------------------------------------------------

def _checkpoint_info(checkpoint_base: Path) -> dict:
    """Collect checkpoint directory metrics.

    Returns a dict with keys:
      repo_count, total_size, oldest_mtime, repos_over_50_commits
    """
    result = {
        "repo_count": 0,
        "total_size": 0,
        "oldest_mtime": None,
        "repos_over_50_commits": 0,
    }

    if not checkpoint_base.exists():
        return result

    import subprocess

    oldest = None
    repos_over_50 = 0
    total_size = 0

    try:
        entries = [p for p in checkpoint_base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        return result

    for repo_dir in entries:
        result["repo_count"] += 1
        repo_size = _dir_size(repo_dir)
        total_size += repo_size

        # Newest mtime in this repo
        try:
            newest = 0.0
            for p in repo_dir.rglob("*"):
                try:
                    m = p.stat().st_mtime
                    if m > newest:
                        newest = m
                except OSError:
                    pass
            if newest > 0:
                if oldest is None or newest < oldest:
                    oldest = newest
        except OSError:
            pass

        # Count commits in the shadow repo
        try:
            env = os.environ.copy()
            env["GIT_DIR"] = str(repo_dir)
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_CONFIG_GLOBAL"] = os.devnull
            proc = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
            if proc.returncode == 0:
                count = int(proc.stdout.strip())
                if count > 50:
                    repos_over_50 += 1
        except Exception:
            pass

    result["total_size"] = total_size
    result["oldest_mtime"] = oldest
    result["repos_over_50_commits"] = repos_over_50
    return result


# ---------------------------------------------------------------------------
# Cron section
# ---------------------------------------------------------------------------

def _cron_info() -> dict:
    """Collect cron job metrics.

    Returns a dict with keys:
      active_jobs, failed_jobs, last_run_ts
    """
    result = {
        "active_jobs": 0,
        "failed_jobs": 0,
        "last_run_ts": None,
        "error": None,
    }

    try:
        from cron.jobs import list_jobs

        jobs = list_jobs(include_disabled=True)
        active = 0
        failed = 0
        last_run_ts = None

        for job in jobs:
            if job.get("enabled", True) and job.get("state") != "paused":
                active += 1
            if job.get("last_status") == "error":
                failed += 1
            lr = job.get("last_run_at")
            if lr:
                try:
                    from datetime import datetime, timezone
                    text = lr.strip()
                    if text.endswith("Z"):
                        text = text[:-1] + "+00:00"
                    parsed = datetime.fromisoformat(text)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    ts = parsed.timestamp()
                    if last_run_ts is None or ts > last_run_ts:
                        last_run_ts = ts
                except Exception:
                    pass

        result["active_jobs"] = active
        result["failed_jobs"] = failed
        result["last_run_ts"] = last_run_ts
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

WAL_WARN_BYTES = 5 * 1024 * 1024   # 5 MB
CHECKPOINT_COMMIT_WARN = 50


def _recommendations(db: dict, ckpt: dict) -> List[str]:
    """Build a list of recommendation strings (may be empty)."""
    recs = []

    if db.get("wal_size", 0) > WAL_WARN_BYTES:
        wal_mb = db["wal_size"] / 1024 / 1024
        recs.append(
            f"WAL file > {wal_mb:.0f} MB — run 'hermes db vacuum' to checkpoint"
        )

    over = ckpt.get("repos_over_50_commits", 0)
    if over > 0:
        recs.append(
            f"{over} checkpoint repo{'s' if over != 1 else ''} > {CHECKPOINT_COMMIT_WARN} commits"
            " — run 'hermes checkpoints prune'"
        )

    return recs


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

def _get_hermes_home_path() -> Path:
    """Return the hermes home directory as a Path (lazy import for testability)."""
    from hermes_cli.config import get_hermes_home
    return Path(get_hermes_home())


def _get_checkpoint_base() -> Path:
    """Return the checkpoint base directory (lazy import for testability)."""
    from tools.checkpoint_manager import CHECKPOINT_BASE
    return CHECKPOINT_BASE


def run_health(args) -> None:
    """Print the health dashboard to stdout."""
    hermes_home = _get_hermes_home_path()
    db_path = hermes_home / "state.db"
    checkpoint_base = _get_checkpoint_base()

    db = _db_info(db_path)
    ckpt = _checkpoint_info(checkpoint_base)
    cron = _cron_info()
    recs = _recommendations(db, ckpt)

    # ---- Database section ----
    print()
    print("Database")

    db_size_str = _fmt_bytes(db["db_size"])
    wal_size = db.get("wal_size", 0)
    if wal_size > 0:
        wal_str = f"  (WAL: {_fmt_bytes(wal_size)}"
        if wal_size > WAL_WARN_BYTES:
            wal_str += " — consider running 'hermes db vacuum'"
        wal_str += ")"
    else:
        wal_str = ""

    if db.get("error"):
        print(f"  state.db:      (error: {db['error']})")
    else:
        print(f"  state.db:      {db_size_str}{wal_str}")
        print(
            f"  Sessions:      {db['total_sessions']} total"
            f"  ({db['active_chains']} active chains, {db['compressed_sessions']} compressed)"
        )
        print(f"  Messages:      {db['total_messages']} total")

    # ---- Checkpoints section ----
    print()
    print("Checkpoints")
    if ckpt["repo_count"] == 0:
        print("  No checkpoint repos found")
    else:
        total_str = _fmt_bytes(ckpt["total_size"])
        print(f"  {ckpt['repo_count']} repos  ({total_str} total)")
        if ckpt["oldest_mtime"] is not None:
            print(f"  Oldest: {_relative_time(ckpt['oldest_mtime'])}")

    # ---- Cron section ----
    print()
    print("Cron")
    if cron.get("error"):
        print(f"  (unavailable: {cron['error']})")
    else:
        last_run_str = _relative_time(cron["last_run_ts"]) if cron["last_run_ts"] else "never"
        print(f"  {cron['active_jobs']} jobs active  (last run: {last_run_str})")
        if cron["failed_jobs"] > 0:
            print(f"  {cron['failed_jobs']} failed (retry pending)")

    # ---- Recommendations section ----
    if recs:
        print()
        print("Recommendations")
        for rec in recs:
            print(f"  ! {rec}")

    print()
