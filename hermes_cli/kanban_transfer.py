"""Kanban board export / import — move a whole board between machines.

Backs ``hermes kanban export|import``, the matching ``/boards/{slug}/export``
and ``/boards/import`` REST endpoints, and the desktop board switcher's
Export/Import items.

Archive layout (``<slug>.tar.gz``, one top-level directory named for the
source board's slug)::

    <slug>/
      manifest.json          format + version + provenance + row counts
      board.json             display metadata, machine-local fields stripped
      kanban.db              consistent snapshot of the board database
      attachments/<task>/…   attachment blobs (unless --no-attachments)
      logs/<task>.log        worker logs (only with --include-logs)

Two things make this more than a ``tar czf`` of the board directory.

**The database is live.** Kanban runs in WAL mode and a dispatcher may be
mid-write, so copying ``kanban.db`` off the filesystem yields a torn
snapshot that is missing whatever still sits in the ``-wal`` file. Export
goes through SQLite's online-backup API instead, which produces a
consistent single-file image of a database that is being written to.

**Rows carry machine-local state.** Claims, PIDs, heartbeats, absolute
workspace and attachment paths, gateway chat subscriptions, and session
ids are all meaningful only on the machine that wrote them. Shipping them
verbatim is how an imported board arrives holding claims owned by a
process on somebody else's laptop, or starts pushing task events into a
stranger's Telegram thread. Everything machine-local is scrubbed on the
export side (so the archive itself never carries it) and defensively
re-scrubbed on import; see :func:`_scrub_local_state` and
:func:`_relocate_imported_rows`.

Imports always land as a **new** board — the slug auto-suffixes on
collision — so an import can never mutate a board that is already there.
That also means an imported board is never ``default``, which is what
lets the import side ignore the default board's split on-disk layout
(``<root>/kanban.db`` beside ``<root>/kanban/attachments/``) and put
everything inside one ``boards/<slug>/`` directory.
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _snapshot_db(source: Path, target: Path) -> None:
    """Write a consistent copy of ``source`` to ``target``.

    Uses SQLite's online-backup API rather than a file copy: in WAL mode
    a just-committed page can still live in the ``-wal`` sidecar, so
    copying only ``kanban.db`` loses recent writes and can produce a
    torn image if the dispatcher commits mid-copy.
    """
    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _scrub_local_state(conn: sqlite3.Connection) -> None:
    """Strip machine-local runtime state. Caller owns the transaction.

    Runs on the export side so the archive itself never carries another
    machine's claims, PIDs, or — the one that actually matters for a
    board shared with someone else — the gateway chat ids subscribed to
    its task events. Repeated on import because an archive is untrusted
    input.
    """
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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _count_rows(conn: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "tasks", "task_links", "task_comments",
        "task_events", "task_runs", "task_attachments",
    )
    return {
        t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        for t in tables
    }


def export_board(
    board: Optional[str],
    output_path: str,
    *,
    include_attachments: bool = True,
    include_logs: bool = False,
) -> dict[str, Any]:
    """Export ``board`` to a ``tar.gz`` archive. Returns a summary dict.

    ``output_path`` may be given with or without the ``.tar.gz`` suffix.
    Workspaces are never included: they are git worktrees and scratch
    trees that are large, machine-local, and rebuilt on demand.
    """
    slug = kb._normalize_board_slug(board) or kb.get_current_board()
    if not kb.board_exists(slug):
        raise ValueError(f"board {slug!r} does not exist")

    db_path = kb.kanban_db_path(slug)
    if not db_path.exists():
        raise FileNotFoundError(f"board {slug!r} has no database at {db_path}")

    output = Path(output_path).expanduser()
    base = str(output).removesuffix(".tar.gz").removesuffix(".tgz")
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

        attachments = 0
        if include_attachments:
            attachments = copy_regular_files(
                kb.attachments_root(slug), staged / "attachments"
            )
        logs = 0
        if include_logs:
            logs = copy_regular_files(
                kb.worker_logs_dir(slug), staged / "logs"
            )

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
            "includes": {
                "attachments": bool(include_attachments),
                "logs": bool(include_logs),
            },
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
    """Return ``preferred``, or the first free ``<preferred>-N`` variant.

    ``default`` always reports as existing, so an archive exported from a
    default board naturally lands as ``default-2`` instead of colliding
    with the importer's own default board.
    """
    if not kb.board_exists(preferred):
        return preferred
    # Leave headroom for the suffix inside the 64-char slug limit.
    stem = preferred[:58].rstrip("-_") or "board"
    n = 2
    while True:
        candidate = f"{stem}-{n}"
        if not kb.board_exists(candidate):
            return candidate
        n += 1


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError(
            "archive is not a Hermes kanban board export (no manifest.json)"
        )
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


def _relocate_imported_rows(
    conn: sqlite3.Connection, slug: str
) -> tuple[dict[str, int], list[str]]:
    """Re-anchor an imported board's rows to this machine.

    Returns ``(stats, warnings)``. Three things move:

    * Attachment rows are repointed at this board's attachments tree.
      Rows whose blob did not travel (an export made with
      ``--no-attachments``) are dropped, because a row pointing at a file
      that does not exist breaks download in every UI that lists it.
    * Workspace paths are cleared. ``scratch`` tasks regenerate one under
      this board on the next claim, so they are simply reset. ``dir`` and
      ``worktree`` tasks cannot be resolved without a path that means
      something here, so any that are still dispatchable are parked in
      ``triage`` — otherwise the dispatcher claims them, fails to build a
      workspace, and burns them straight into the failure breaker.
    * Runtime state is scrubbed again. Export already did this, but an
      archive is an untrusted input and the cost is one UPDATE.
    """
    warnings: list[str] = []
    now = int(time.time())
    attachments_dir = kb.attachments_root(slug)

    with kb.write_txn(conn):
        _scrub_local_state(conn)

        dropped = 0
        rehomed = 0
        for row in conn.execute(
            "SELECT id, task_id, stored_path FROM task_attachments"
        ).fetchall():
            landed = attachments_dir / row["task_id"] / Path(row["stored_path"]).name
            if landed.is_file():
                conn.execute(
                    "UPDATE task_attachments SET stored_path = ? WHERE id = ?",
                    (str(landed), row["id"]),
                )
                rehomed += 1
            else:
                conn.execute(
                    "DELETE FROM task_attachments WHERE id = ?", (row["id"],)
                )
                dropped += 1
        if dropped:
            warnings.append(
                f"{dropped} attachment record(s) dropped — the files were not "
                f"in the archive"
            )

        parked = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM tasks WHERE workspace_kind IN ('dir', 'worktree') "
                f"AND status IN ({', '.join('?' * len(_DISPATCHABLE_STATUSES))})",
                _DISPATCHABLE_STATUSES,
            ).fetchall()
        ]
        conn.execute("UPDATE tasks SET workspace_path = NULL, branch_name = NULL")
        if parked:
            conn.execute(
                f"UPDATE tasks SET status = 'triage' "
                f"WHERE id IN ({', '.join('?' * len(parked))})",
                parked,
            )
            warnings.append(
                f"{len(parked)} task(s) moved to triage — their workspace was a "
                f"directory or git worktree on the exporting machine and needs "
                f"to be pointed somewhere on this one"
            )

        for row in conn.execute("SELECT id FROM tasks").fetchall():
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, 'imported', ?, ?)",
                (
                    row["id"],
                    json.dumps(
                        {
                            "board": slug,
                            "parked": row["id"] in parked,
                        },
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )

    return {"attachments": rehomed, "parked": len(parked)}, warnings


def import_board(
    archive_path: str,
    slug: Optional[str] = None,
    *,
    activate: bool = False,
) -> dict[str, Any]:
    """Import a board archive as a new board. Returns a summary dict.

    ``slug`` overrides the name from the archive. Either way the final
    slug auto-suffixes if it is taken, so an import never merges into or
    overwrites an existing board.
    """
    archive = Path(archive_path).expanduser()
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    roots = archive_root_dirs(archive)
    if len(roots) != 1:
        raise ValueError(
            "a kanban board archive must contain exactly one top-level directory"
        )
    archive_root = roots.pop()

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        safe_extract_targz(archive, staging)
        extracted = staging / archive_root

        manifest = _read_manifest(extracted)
        staged_db = extracted / "kanban.db"
        if not staged_db.is_file():
            raise ValueError("archive is missing kanban.db")

        requested = kb._normalize_board_slug(
            slug or manifest.get("board") or archive_root
        )
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

    with kb.connect_closing(board=target) as conn:
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
