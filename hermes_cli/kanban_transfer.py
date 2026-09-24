"""Kanban board export / import — move a whole board between machines.

Backs ``hermes kanban export|import``, the ``/boards/{slug}/export`` and
``/boards/import`` REST endpoints, and the desktop board switcher. Archive
layout (``<slug>.tar.gz``, one top-level dir named for the source slug):
``manifest.json`` (format/version/provenance/counts), ``board.json`` (display
metadata, machine-local fields stripped), ``kanban.db`` (consistent snapshot),
``attachments/<task>/…`` (unless --no-attachments), ``logs/<task>.log`` (only
with --include-logs).

Two things make this more than ``tar czf`` of the board directory: the DB is
live (WAL mode, dispatcher may be mid-write) so export uses SQLite's online
backup instead of a file copy that would miss the ``-wal`` sidecar; and rows
carry machine-local state (claims, PIDs, heartbeats, absolute paths, gateway
chat subscriptions, session ids) that would import a stranger's claims or push
events into a stranger's Telegram thread — scrubbed on export and re-scrubbed
on import (an archive is untrusted); see :func:`_scrub_local_state` and
:func:`_relocate_imported_rows`. Imports always land as a **new** board (slug
auto-suffixes on collision), never ``default``, so the importer can ignore the
default board's split on-disk layout.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.archive_safe import (
    archive_root_dirs,
    copy_regular_files,
    make_targz,
    safe_extract_targz,
)

ARCHIVE_FORMAT = "hermes-kanban-board"
ARCHIVE_FORMAT_VERSION = 1

# Statuses from which the dispatcher can still act on a task. A task whose
# workspace cannot be rebuilt on this machine is parked in ``triage`` only
# if it is in one of these — terminal and already-parked tasks are left
# alone rather than having their history rewritten.
_DISPATCHABLE_STATUSES = ("ready", "running", "todo", "scheduled")
_COUNTED_TABLES = ("tasks", "task_links", "task_comments", "task_events", "task_runs", "task_attachments")


def _placeholders(items) -> str:
    return ", ".join("?" * len(items))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _snapshot_db(source: Path, target: Path) -> None:
    """Consistent copy of ``source`` via the online-backup API (a file copy
    would miss pages still in the ``-wal`` sidecar and could tear)."""
    with contextlib.closing(sqlite3.connect(str(source))) as src, \
            contextlib.closing(sqlite3.connect(str(target))) as dst:
        src.backup(dst)


def _scrub_local_state(conn: sqlite3.Connection) -> None:
    """Strip machine-local runtime state (claims, PIDs, and above all the
    gateway chat ids subscribed to task events). Caller owns the transaction.
    Run on export and again on import (an archive is untrusted input)."""
    conn.execute("DELETE FROM kanban_notify_subs")
    conn.execute(
        """
        UPDATE tasks
           SET claim_lock           = NULL,
               claim_expires        = NULL,
               worker_pid           = NULL,
               current_run_id       = NULL,
               last_heartbeat_at    = NULL,
               session_id           = NULL,
               project_id           = NULL,
               consecutive_failures = 0,
               last_failure_error   = NULL
        """
    )
    # A task caught mid-run is not running anywhere the importer can see.
    # Send it back to the queue rather than shipping a phantom claim.
    conn.execute("UPDATE tasks SET status = 'ready' WHERE status = 'running'")
    conn.execute(
        """
        UPDATE task_runs
           SET status            = 'released',
               outcome           = COALESCE(outcome, 'reclaimed'),
               ended_at          = COALESCE(ended_at, ?),
               last_heartbeat_at = NULL
         WHERE status = 'running'
        """,
        (int(time.time()),),
    )
    conn.execute("UPDATE task_runs SET claim_lock = NULL, worker_pid = NULL")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _count_rows(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in _COUNTED_TABLES}


def export_board(
    board: Optional[str],
    output_path: str,
    *,
    include_attachments: bool = True,
    include_logs: bool = False,
) -> dict[str, Any]:
    """Export ``board`` to a ``tar.gz`` (suffix optional on ``output_path``);
    returns a summary dict. Workspaces are never included — large,
    machine-local, rebuilt on demand."""
    slug = kb._normalize_board_slug(board) or kb.get_current_board()
    if not kb.board_exists(slug):
        raise ValueError(f"board {slug!r} does not exist")

    db_path = kb.kanban_db_path(slug)
    if not db_path.exists():
        raise FileNotFoundError(f"board {slug!r} has no database at {db_path}")

    base = str(Path(output_path).expanduser()).removesuffix(".tar.gz").removesuffix(".tgz")
    Path(base).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / slug
        staged.mkdir(parents=True)

        _snapshot_db(db_path, staged / "kanban.db")
        # The snapshot is a private file with no other writers, so plain
        # commit/close is enough — no need for the board DB's WAL dance.
        with contextlib.closing(sqlite3.connect(str(staged / "kanban.db"))) as snapshot:
            _scrub_local_state(snapshot)
            snapshot.commit()
            counts = _count_rows(snapshot)

        meta = kb.read_board_metadata(slug)
        # Both name a location on the exporting machine; the importer
        # resolves its own.
        meta.pop("db_path", None)
        meta["default_workdir"] = None
        meta["project_id"] = None
        _write_json(staged / "board.json", meta)

        attachments = copy_regular_files(kb.attachments_root(slug), staged / "attachments") if include_attachments else 0
        logs = copy_regular_files(kb.worker_logs_dir(slug), staged / "logs") if include_logs else 0

        try:
            from hermes_cli import __version__ as hermes_version
        except Exception:
            hermes_version = ""

        manifest = {
            "format": ARCHIVE_FORMAT,
            "format_version": ARCHIVE_FORMAT_VERSION,
            "board": slug,
            "board_name": meta.get("name") or slug,
            "exported_at": int(time.time()),
            "hermes_version": str(hermes_version),
            "includes": {"attachments": bool(include_attachments), "logs": bool(include_logs)},
            "counts": {**counts, "attachment_files": attachments, "log_files": logs},
        }
        _write_json(staged / "manifest.json", manifest)

        archive = make_targz(base, tmpdir, slug)

    return {
        "board": slug,
        "archive": archive,
        "size": Path(archive).stat().st_size,
        "counts": manifest["counts"],
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _available_slug(preferred: str) -> str:
    """``preferred`` or the first free ``<preferred>-N``. ``default`` always
    exists, so a default-board export lands as ``default-2``."""
    if not kb.board_exists(preferred):
        return preferred
    # Leave headroom for the suffix inside the 64-char slug limit.
    stem = preferred[:58].rstrip("-_") or "board"
    n = 2
    while kb.board_exists(f"{stem}-{n}"):
        n += 1
    return f"{stem}-{n}"


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError("archive is not a Hermes kanban board export (no manifest.json)")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"archive manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != ARCHIVE_FORMAT:
        raise ValueError(
            "archive is not a Hermes kanban board export "
            f"(format={manifest.get('format') if isinstance(manifest, dict) else None!r})"
        )
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > ARCHIVE_FORMAT_VERSION:
        raise ValueError(
            f"archive format version {version!r} is newer than this Hermes "
            f"understands (max {ARCHIVE_FORMAT_VERSION}) — update Hermes and retry"
        )
    return manifest


def _read_board_metadata(path: Path) -> dict[str, Any]:
    """Read an archive's ``board.json``, tolerating a missing/broken file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _relocate_imported_rows(conn: sqlite3.Connection, slug: str) -> tuple[dict[str, int], list[str]]:
    """Re-anchor an imported board's rows to this machine; returns ``(stats, warnings)``.

    * Attachment rows are repointed at this board's tree; rows whose blob
      did not travel (``--no-attachments``) are dropped, since a dangling row
      breaks download in every UI.
    * Workspace paths are cleared. ``scratch`` regenerates on next claim;
      dispatchable ``dir``/``worktree`` tasks are parked in ``triage``,
      otherwise the dispatcher claims them, fails to build a workspace, and
      burns them into the failure breaker.
    * Runtime state is scrubbed again (untrusted input, one UPDATE).
    """
    warnings: list[str] = []
    now = int(time.time())
    attachments_dir = kb.attachments_root(slug)

    with kb.write_txn(conn):
        _scrub_local_state(conn)

        dropped = rehomed = 0
        for row in conn.execute("SELECT id, task_id, stored_path FROM task_attachments").fetchall():
            landed = attachments_dir / row["task_id"] / Path(row["stored_path"]).name
            if landed.is_file():
                conn.execute("UPDATE task_attachments SET stored_path = ? WHERE id = ?", (str(landed), row["id"]))
                rehomed += 1
            else:
                conn.execute("DELETE FROM task_attachments WHERE id = ?", (row["id"],))
                dropped += 1
        if dropped:
            warnings.append(f"{dropped} attachment record(s) dropped — the files were not in the archive")

        parked = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM tasks WHERE workspace_kind IN ('dir', 'worktree') "
                f"AND status IN ({_placeholders(_DISPATCHABLE_STATUSES)})",
                _DISPATCHABLE_STATUSES,
            ).fetchall()
        ]
        conn.execute("UPDATE tasks SET workspace_path = NULL, branch_name = NULL")
        if parked:
            conn.execute(f"UPDATE tasks SET status = 'triage' WHERE id IN ({_placeholders(parked)})", parked)
            warnings.append(
                f"{len(parked)} task(s) moved to triage — their workspace was a directory or git "
                f"worktree on the exporting machine and needs to be pointed somewhere on this one"
            )

        for row in conn.execute("SELECT id FROM tasks").fetchall():
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, 'imported', ?, ?)",
                (row["id"], json.dumps({"board": slug, "parked": row["id"] in parked}, ensure_ascii=False), now),
            )

    return {"attachments": rehomed, "parked": len(parked)}, warnings


def import_board(
    archive_path: str,
    slug: Optional[str] = None,
    *,
    activate: bool = False,
) -> dict[str, Any]:
    """Import an archive as a NEW board (``slug`` overrides the archive's;
    either way it auto-suffixes if taken). Returns a summary dict."""
    archive = Path(archive_path).expanduser()
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    roots = archive_root_dirs(archive)
    if len(roots) != 1:
        raise ValueError("a kanban board archive must contain exactly one top-level directory")
    archive_root = roots.pop()

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        safe_extract_targz(archive, staging)
        extracted = staging / archive_root

        manifest = _read_manifest(extracted)
        staged_db = extracted / "kanban.db"
        if not staged_db.is_file():
            raise ValueError("archive is missing kanban.db")

        requested = kb._normalize_board_slug(slug or manifest.get("board") or archive_root)
        if not requested:
            raise ValueError(
                "cannot determine a board name from the archive — pass one "
                "explicitly with --as <slug>"
            )
        target = _available_slug(requested)

        staged_meta = _read_board_metadata(extracted / "board.json")

        board_root = kb.board_dir(target)
        board_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_db), str(board_root / "kanban.db"))
        for tree in ("attachments", "logs"):
            src = extracted / tree
            if src.is_dir():
                shutil.move(str(src), str(board_root / tree))

    # Rewritten rather than moved across: the archive's copy names a slug
    # and a workdir that belong to the exporting machine.
    name = str(staged_meta.get("name") or manifest.get("board_name") or target)
    kb.write_board_metadata(
        target,
        name=name,
        description=str(staged_meta.get("description") or ""),
        icon=str(staged_meta.get("icon") or ""),
        color=str(staged_meta.get("color") or ""),
        archived=False,
    )
    # Bring the imported schema up to this install's version before the
    # relocation pass writes to it.
    kb.init_db(board=target)

    with kbc.connect_closing(board=target) as conn:
        stats, warnings = _relocate_imported_rows(conn, target)
        counts = _count_rows(conn)

    if activate:
        kb.set_current_board(target)

    return {
        "board": target,
        "requested_board": requested,
        "renamed": target != requested,
        "name": name,
        "path": str(kb.board_dir(target)),
        "db_path": str(kb.kanban_db_path(target)),
        "source": {
            "board": manifest.get("board"),
            "exported_at": manifest.get("exported_at"),
            "hermes_version": manifest.get("hermes_version"),
        },
        "counts": counts,
        "attachments_restored": stats["attachments"],
        "tasks_parked": stats["parked"],
        "warnings": warnings,
        "activated": bool(activate),
    }
