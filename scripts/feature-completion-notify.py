#!/usr/bin/env python3
"""
Feature-completion auto-close + notify.

Scans all kanban boards for parent tasks where:
- Parent is in 'todo', 'blocked', 'scheduled', or 'running'
- Every linked child is 'done' or 'archived'
- Parent has at least 1 child (safety: no orphans)

When found: auto-completes the parent, which triggers any
notify-subscribe subscriptions Sahil registered on that task.

Runs as no_agent cron — zero token cost.
State tracked in /home/kensei/.hermes/data/feature-completion-state.json
to avoid double-closing.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Cross-process write lock to prevent WAL checkpoint races under concurrent write load
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from kanban_write_lock import write_lock
except ImportError:
    write_lock = None

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
BOARDS_ROOT = HERMES_HOME / "kanban" / "boards"
STATE_FILE = HERMES_HOME / "data" / "feature-completion-state.json"

logger = logging.getLogger("feature-completion-notify")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Set by main() from the --dry-run CLI flag. When True, every write path
# (task UPDATE, save_state, corrupt-DB quarantine/rename/copy/unlink, and
# quarantine-sentinel touch/unlink) is suppressed. Read paths run exactly as
# in live mode so the proposed completions are still computed and printed.
_DRY_RUN = False


def _is_db_corrupt(db_path: Path) -> bool:
    """Return True if the SQLite file cannot serve the queries we need.

    A bare `SELECT 1` will pass on a schema-corrupt DB (the header page is fine
    but the schema is gone), so we probe for the specific tables this script
    reads. Avoids the cost of `PRAGMA integrity_check` on a healthy DB while
    still catching `database disk image is malformed` and missing-schema cases
    like `no such table: task_links`.
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        try:
            conn.execute("SELECT 1 FROM task_links LIMIT 0")
            conn.execute("SELECT 1 FROM tasks LIMIT 0")
        finally:
            conn.close()
        return False
    except sqlite3.DatabaseError:
        return True
    except Exception:
        # Treat other errors (locked, IO) as transient — don't quarantine on those
        return False


# Stop the auto-recreate loop. When we quarantine a corrupt ops/kanban.db,
# some other process (kanban CLI, hermes itself) re-creates an empty
# 4 KB shell DB on the next event. That triggers our corruption handler
# again, every cron tick. Sentinel file records "we just cleaned this up,
# don't re-quarantine for N hours" so the loop settles.
QUARANTINE_SENTINEL_TTL_HOURS = 6


def _quarantine_sentinel_path(db_path: Path) -> Path:
    return db_path.parent / ".quarantine-cooldown"


def _quarantine_recent(db_path: Path) -> bool:
    """Return True if we already quarantined this board within the cooldown window."""
    sentinel = _quarantine_sentinel_path(db_path)
    if not sentinel.exists():
        return False
    try:
        age_h = (datetime.now().timestamp() - sentinel.stat().st_mtime) / 3600
        return age_h < QUARANTINE_SENTINEL_TTL_HOURS
    except OSError:
        return False


def _mark_quarantined(db_path: Path) -> None:
    if _DRY_RUN:
        return
    try:
        sentinel = _quarantine_sentinel_path(db_path)
        sentinel.touch(exist_ok=True)
    except OSError:
        pass


def _clear_quarantine_sentinel(db_path: Path) -> None:
    """Clear the sentinel once we confirm a fresh DB is healthy (e.g. ops was rebuilt)."""
    if _DRY_RUN:
        return
    try:
        _quarantine_sentinel_path(db_path).unlink(missing_ok=True)
    except OSError:
        pass


def _quarantine_corrupt_db(db_path: Path) -> Optional[str]:
    """Move a corrupt DB out of the boards dir so subsequent runs skip it.

    Returns the quarantine path on success, None on failure. Never raises —
    a failed quarantine should not crash the cron. Suppressed entirely in
    --dry-run (the corrupt DB is left in place for inspection).
    """
    if _DRY_RUN:
        logger.info("dry-run: would quarantine corrupt DB %s (left in place)", db_path)
        return None
    try:
        date_dir = db_path.parent / f"quarantine-{datetime.now().strftime('%Y-%m-%d')}"
        date_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        dest = date_dir / f"{db_path.name}.corrupt.{stamp}"
        try:
            db_path.rename(dest)
        except OSError:
            shutil.copy2(str(db_path), str(dest))
            db_path.unlink()
        return str(dest)
    except Exception as exc:
        logger.warning("Quarantine failed for %s: %s", db_path, exc)
        return None


def get_parents_with_children(db_path: Path) -> dict[str, list[str]]:
    """Return {parent_id: [child_ids]} for all parent-child links."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT parent_id, child_id FROM task_links"
    ).fetchall()
    conn.close()
    result: dict[str, list[str]] = {}
    for row in rows:
        pid = row["parent_id"]
        cid = row["child_id"]
        result.setdefault(pid, []).append(cid)
    return result

def get_task_statuses(db_path: Path, task_ids: list[str]) -> dict[str, str]:
    """Return {task_id: status} for given task IDs."""
    if not task_ids:
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(task_ids))
    rows = conn.execute(
        f"SELECT id, status FROM tasks WHERE id IN ({placeholders})",
        task_ids
    ).fetchall()
    conn.close()
    return {r["id"]: r["status"] for r in rows}

def get_task_title(db_path: Path, task_id: str) -> Optional[str]:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT title FROM tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return row[0] if row else None

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def process_board(board_dir: Path, state: dict) -> list[str]:
    """Process a single board. Returns list of new completion strings.

    Isolated so that one corrupt/errored board never aborts the whole cron.
    """
    db_path = board_dir / "kanban.db"

    # If the file is missing entirely but the board directory is freshly
    # nuked (we quarantined it, or it never existed), nothing to do.
    if not db_path.exists():
        return []

    # Cooldown check: don't re-quarantine within the TTL. If the file was
    # re-created as an empty shell, the schema check below will catch that
    # case, but only ONCE per cooldown — otherwise the cron loops.
    if _quarantine_recent(db_path):
        # Re-probe: if the file has actually been rebuilt with a real
        # schema, clear the sentinel and process normally.
        if not _is_db_corrupt(db_path):
            _clear_quarantine_sentinel(db_path)
            logger.info("Board %s recovered from corruption, resuming processing", board_dir.name)
        else:
            return []

    # Pre-flight: is the DB readable? If not, quarantine and skip — never raise.
    if _is_db_corrupt(db_path):
        qpath = _quarantine_corrupt_db(db_path)
        _mark_quarantined(db_path)
        logger.warning(
            "Quarantined corrupt kanban DB: %s -> %s (cooldown %sh)",
            db_path, qpath or "<quarantine failed; leaving in place>",
            QUARANTINE_SENTINEL_TTL_HOURS,
        )
        return []

    try:
        parents = get_parents_with_children(db_path)
    except sqlite3.DatabaseError as exc:
        # Caught between pre-flight and read (race / partial write) — quarantine and skip
        qpath = _quarantine_corrupt_db(db_path)
        _mark_quarantined(db_path)
        logger.warning(
            "DB became corrupt mid-read (%s): %s -> %s (cooldown %sh)",
            exc, db_path, qpath or "<quarantine failed>",
            QUARANTINE_SENTINEL_TTL_HOURS,
        )
        return []

    if not parents:
        return []

    all_child_ids = [cid for children in parents.values() for cid in children]
    all_ids = list(parents.keys()) + all_child_ids
    statuses = get_task_statuses(db_path, all_ids)

    new_completions: list[str] = []
    terminal = {"done", "archived", "completed"}
    completable = {"todo", "blocked", "scheduled", "running", "ready"}

    for parent_id, child_ids in parents.items():
        parent_status = statuses.get(parent_id, "unknown")
        child_statuses = [statuses.get(cid, "unknown") for cid in child_ids]

        if parent_status in terminal:
            continue
        if parent_status not in completable:
            continue
        if not all(cs in terminal for cs in child_statuses):
            continue
        if not child_ids:
            continue
        if parent_id in state:
            continue

        # Auto-complete (under cross-process write lock). Suppressed in
        # --dry-run: the parent is reported as a proposal without mutating
        # the DB.
        title = get_task_title(db_path, parent_id) or parent_id
        if not _DRY_RUN:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                if write_lock is not None:
                    with write_lock(conn):
                        now_ts = int(datetime.now(timezone.utc).timestamp())
                        conn.execute(
                            "UPDATE tasks SET status='done', completed_at=? WHERE id=?",
                            (now_ts, parent_id)
                        )
                        conn.commit()
                else:
                    now_ts = int(datetime.now(timezone.utc).timestamp())
                    conn.execute(
                        "UPDATE tasks SET status='done', completed_at=? WHERE id=?",
                        (now_ts, parent_id)
                    )
                    conn.commit()
            finally:
                conn.close()

        state[parent_id] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "children": child_ids,
            "title": title,
            "board": board_dir.name,
        }
        new_completions.append(f"{board_dir.name}/{parent_id}: {title}")

    return new_completions


def _process_root_board(db_path: Path, state: dict) -> list[str]:
    """Process the default board (root kanban.db, not under boards/).

    Same logic as process_board but takes a direct db path instead of a
    board directory (which has no subdir structure for the default board).
    """
    _parent_dir = db_path.parent  # ~/.hermes/kanban/
    # Monkey-patch process_board to use a temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Symlink the DB so process_board's db_path resolution works
        (tmp / "kanban.db").symlink_to(db_path.resolve())
        return process_board(tmp, state)


def _parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="Feature-completion auto-close + notify")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report proposed parent completions to stdout without mutating "
             "boards, state, quarantine, or sentinels.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    global _DRY_RUN
    args = _parse_args(argv)
    _DRY_RUN = bool(args.dry_run)
    state = load_state()
    new_completions: list[str] = []

    if BOARDS_ROOT.exists():
        for board_dir in sorted(BOARDS_ROOT.iterdir()):
            if not board_dir.is_dir():
                continue
            if board_dir.name.startswith("quarantine-"):
                continue
            try:
                new_completions.extend(process_board(board_dir, state))
            except Exception as exc:
                logger.warning("Unexpected error processing %s: %s", board_dir, exc)
    else:
        logger.info("Boards root %s does not exist", BOARDS_ROOT)

    # Scan the default board (root kanban.db, not under boards/)
    _default_db = HERMES_HOME / "kanban.db"
    if _default_db.exists() and not _is_db_corrupt(_default_db):
        try:
            _default_board = _default_db.parent  # ~/.hermes/kanban/
            new_completions.extend(_process_root_board(_default_db, state))
        except Exception as exc:
            logger.warning("Error processing default board: %s", exc)

    if not _DRY_RUN:
        save_state(state)

    if new_completions:
        verb = "would auto-close" if _DRY_RUN else "auto-closed"
        summary = "\n".join(f"  - {c}" for c in new_completions)
        print(f"Feature-completion: {len(new_completions)} parent(s) {verb}:\n{summary}")
    # Silent when nothing to report.

if __name__ == "__main__":
    main()
