"""Tests for kanban board export / import (``hermes_cli.kanban_transfer``).

The contract these pin down is "a board survives the trip to another
machine, and nothing that only made sense on the exporting machine comes
with it":

* Content round-trips — tasks, comments, links, events, attachment blobs.
* Runtime state does not — claims, worker PIDs, heartbeats, session ids,
  and gateway chat subscriptions are gone on the far side.
* Paths are re-anchored — attachment rows point into the importing
  board's tree, and tasks whose workspace cannot be rebuilt here are
  parked instead of being fed to the dispatcher.
* An import never mutates a board that already exists.
* A hostile archive cannot write outside the import destination.
"""

from __future__ import annotations

import json
import sys
import tarfile
import time
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_transfer as kt
from hermes_cli.archive_safe import normalize_archive_parts, safe_extract_targz


@pytest.fixture
def kanban_root(tmp_path, monkeypatch):
    """Point kanban at an empty root, and hand back a switcher.

    Export and import have to run against two different machines' state.
    Calling the returned function re-points every kanban path helper at a
    fresh root, which is as close to "the other machine" as a unit test
    gets.
    """
    def _use(name: str) -> Path:
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
        for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                    "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_BOARD"):
            monkeypatch.delenv(var, raising=False)
        kb._INITIALIZED_PATHS.clear()
        return root

    _use("source")
    return _use


def _seed_board(slug: str = "alpha") -> dict[str, str]:
    """Create a board with one task of each interesting shape."""
    kb.create_board(slug, name="Alpha Board")
    ids = {}
    with kbc.connect_closing(board=slug) as conn:
        ids["scratch"] = kb.create_task(
            conn, title="scratch task", body="body", assignee="coder"
        )
        ids["worktree"] = kb.create_task(
            conn, title="worktree task", assignee="coder",
            workspace_kind="worktree", workspace_path="/exporter/repo",
        )
        kb.add_comment(conn, ids["scratch"], "brooklyn", "a comment")
        kb.link_tasks(conn, ids["scratch"], ids["worktree"])
        kb.store_attachment_bytes(
            conn, ids["scratch"], "notes.txt", b"hello attachment", board=slug
        )
    return ids


def _claim(task_id: str, slug: str = "alpha") -> None:
    """Put a task into the state a live worker would leave behind."""
    with kbc.connect_closing(board=slug) as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='running', claim_lock='lock-1', "
                "claim_expires=?, worker_pid=4242, last_heartbeat_at=?, "
                "session_id='sess-xyz', consecutive_failures=2 WHERE id=?",
                (int(time.time()) + 600, int(time.time()), task_id),
            )


def _subscribe(task_id: str, slug: str = "alpha") -> None:
    with kbc.connect_closing(board=slug) as conn:
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO kanban_notify_subs "
                "(task_id, platform, chat_id, thread_id, created_at) "
                "VALUES (?, 'telegram', '12345', '', ?)",
                (task_id, int(time.time())),
            )


def _tasks_by_title(slug: str) -> dict[str, dict]:
    with kbc.connect_closing(board=slug) as conn:
        return {
            row["title"]: dict(row)
            for row in conn.execute("SELECT * FROM tasks").fetchall()
        }


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_round_trip_preserves_content(kanban_root, tmp_path):
    _seed_board()
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    kanban_root("target")
    result = kt.import_board(archive)

    assert result["counts"]["tasks"] == 2
    assert result["counts"]["task_comments"] == 1
    assert result["counts"]["task_links"] == 1

    tasks = _tasks_by_title(result["board"])
    assert set(tasks) == {"scratch task", "worktree task"}
    assert tasks["scratch task"]["body"] == "body"
    assert tasks["scratch task"]["assignee"] == "coder"


def test_attachment_blob_travels_and_is_readable(kanban_root, tmp_path):
    _seed_board()
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    target_root = kanban_root("target")
    result = kt.import_board(archive)

    with kbc.connect_closing(board=result["board"]) as conn:
        row = conn.execute(
            "SELECT filename, stored_path FROM task_attachments"
        ).fetchone()

    assert row["filename"] == "notes.txt"
    stored = Path(row["stored_path"])
    # Re-anchored under the importing machine's board, not the exporter's.
    assert target_root in stored.parents
    assert stored.read_bytes() == b"hello attachment"


def test_export_without_attachments_drops_the_rows(kanban_root, tmp_path):
    _seed_board()
    archive = kt.export_board(
        "alpha", str(tmp_path / "alpha"), include_attachments=False
    )["archive"]

    kanban_root("target")
    result = kt.import_board(archive)

    # A row whose blob never travelled would be a broken download link in
    # every UI that lists it, so it is dropped and reported.
    assert result["counts"]["task_attachments"] == 0
    assert any("attachment" in w for w in result["warnings"])


def test_workspaces_are_never_exported(kanban_root, tmp_path):
    _seed_board()
    workspace = kb.workspaces_root("alpha") / "junk"
    workspace.mkdir(parents=True)
    (workspace / "huge.bin").write_bytes(b"x" * 1024)

    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
    assert not any("workspaces" in name for name in names)


# ---------------------------------------------------------------------------
# Machine-local state does not travel
# ---------------------------------------------------------------------------

def test_claimed_task_arrives_unclaimed_and_queued(kanban_root, tmp_path):
    ids = _seed_board()
    _claim(ids["scratch"])
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    kanban_root("target")
    result = kt.import_board(archive)

    task = _tasks_by_title(result["board"])["scratch task"]
    # A claim held by a PID on the exporting machine must not survive, or
    # the importing dispatcher inherits a lock nothing will ever release.
    assert task["status"] == "ready"
    assert task["claim_lock"] is None
    assert task["claim_expires"] is None
    assert task["worker_pid"] is None
    assert task["last_heartbeat_at"] is None
    assert task["current_run_id"] is None
    assert task["session_id"] is None
    assert task["consecutive_failures"] == 0


def test_gateway_subscriptions_never_travel(kanban_root, tmp_path):
    ids = _seed_board()
    _subscribe(ids["scratch"])
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    # Not merely dropped on import — the chat id must not be in the file
    # at all, because the archive is the thing that gets shared.
    kanban_root("target")
    result = kt.import_board(archive)
    with kbc.connect_closing(board=result["board"]) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM kanban_notify_subs"
        ).fetchone()[0] == 0

    assert b"12345" not in Path(archive).read_bytes()


def test_unresolvable_workspaces_are_parked_not_dispatched(kanban_root, tmp_path):
    _seed_board()
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    kanban_root("target")
    result = kt.import_board(archive)
    tasks = _tasks_by_title(result["board"])

    # The worktree lived on the exporting machine's disk. Letting the
    # dispatcher claim this would fail workspace resolution twice and trip
    # the failure breaker, so it waits for a human instead.
    assert tasks["worktree task"]["status"] == "triage"
    assert tasks["worktree task"]["workspace_path"] is None
    assert result["tasks_parked"] == 1

    # A scratch task needs no path — it regenerates one under this board.
    assert tasks["scratch task"]["status"] == "ready"
    assert tasks["scratch task"]["workspace_path"] is None


def test_board_metadata_loses_exporter_local_paths(kanban_root, tmp_path):
    kb.create_board("alpha", name="Alpha Board",
                    default_workdir="/exporter/repo", project_id="proj-1")
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    kanban_root("target")
    result = kt.import_board(archive)
    meta = kb.read_board_metadata(result["board"])

    assert meta["name"] == "Alpha Board"
    assert meta["default_workdir"] is None
    assert meta["project_id"] is None


# ---------------------------------------------------------------------------
# Import never overwrites
# ---------------------------------------------------------------------------

def test_slug_collision_creates_a_new_board(kanban_root, tmp_path):
    _seed_board()
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    kanban_root("target")
    first = kt.import_board(archive)
    second = kt.import_board(archive)

    assert first["board"] == "alpha"
    assert second["board"] != first["board"]
    assert second["renamed"] is True
    assert second["requested_board"] == "alpha"
    # The board that was already there is untouched.
    assert len(_tasks_by_title(first["board"])) == 2


def test_import_never_targets_the_default_board(kanban_root, tmp_path):
    with kbc.connect_closing(board="default") as conn:
        kb.create_task(conn, title="exported default task")
    archive = kt.export_board("default", str(tmp_path / "default"))["archive"]

    kanban_root("target")
    with kbc.connect_closing(board="default") as conn:
        kb.create_task(conn, title="local default task")

    result = kt.import_board(archive)

    assert result["board"] != "default"
    assert set(_tasks_by_title("default")) == {"local default task"}
    assert set(_tasks_by_title(result["board"])) == {"exported default task"}


def test_explicit_slug_is_honoured(kanban_root, tmp_path):
    _seed_board()
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    kanban_root("target")
    result = kt.import_board(archive, "renamed-board", activate=True)

    assert result["board"] == "renamed-board"
    assert result["renamed"] is False
    assert kb.get_current_board() == "renamed-board"


# ---------------------------------------------------------------------------
# Hostile / malformed input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("member", [
    "../escape.txt",
    "board/../../escape.txt",
    "/etc/passwd",
    "C:\\Windows\\system32",
    "board\\..\\..\\escape.txt",
])
def test_traversal_members_are_rejected(member):
    with pytest.raises(ValueError):
        normalize_archive_parts(member)


def test_extract_refuses_a_symlink_member(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        link = tarfile.TarInfo("payload/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)

    with pytest.raises(ValueError, match="Unsupported archive member"):
        safe_extract_targz(archive, tmp_path / "out")


def test_import_rejects_a_non_kanban_archive(kanban_root, tmp_path):
    payload = tmp_path / "notaboard"
    payload.mkdir()
    (payload / "readme.txt").write_text("hi")
    archive = tmp_path / "other.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="notaboard")

    with pytest.raises(ValueError, match="manifest"):
        kt.import_board(str(archive))


def test_import_rejects_a_future_format_version(kanban_root, tmp_path):
    _seed_board()
    archive = Path(kt.export_board("alpha", str(tmp_path / "alpha"))["archive"])

    staged = tmp_path / "restage"
    safe_extract_targz(archive, staged)
    manifest_path = staged / "alpha" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = kt.ARCHIVE_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    bumped = tmp_path / "bumped.tar.gz"
    with tarfile.open(bumped, "w:gz") as tf:
        tf.add(staged / "alpha", arcname="alpha")

    kanban_root("target")
    with pytest.raises(ValueError, match="newer than this Hermes"):
        kt.import_board(str(bumped))


# ---------------------------------------------------------------------------
# Concurrency. ``_available_slug`` only OBSERVED that a slug was free; between
# that observation and the ``shutil.move`` that installs the DB, a second
# importer could claim the same slug and the later move replaced the first
# board's ``kanban.db`` -- a silent, total loss of an imported board
# (FleetReview on PR #785). Two REAL processes, because the window is between
# two syscalls and a mock cannot occupy it.
# ---------------------------------------------------------------------------

def _import_in_child(home: str, archive: str, marker: str, barrier, q) -> None:
    """Run a real ``import_board`` in a fresh interpreter, barrier-synced."""
    import os
    import sys as _sys

    for key in [k for k in os.environ if k.startswith("HERMES_KANBAN")]:
        os.environ.pop(key, None)
    os.environ["HERMES_KANBAN_SANDBOX"] = "1"
    os.environ["HERMES_HOME"] = home
    _sys.path.insert(0, str(_WORKTREE))
    from hermes_cli import kanban_transfer as _kt

    barrier.wait(timeout=60)
    try:
        result = _kt.import_board(Path(archive), slug="shared")
        q.put((marker, result["board"], None))
    except Exception as exc:  # noqa: BLE001
        q.put((marker, None, f"{type(exc).__name__}: {exc}"))


@pytest.mark.timeout(300)
def test_concurrent_imports_of_the_same_slug_never_overwrite(kanban_root, tmp_path):
    """Two importers racing on one slug must produce two boards, no loss."""
    import multiprocessing
    import sqlite3
    from hermes_cli.kanban_db_connect import connect_closing

    archives: dict[str, str] = {}
    expected: dict[str, str] = {}
    for marker in ("a", "b"):
        kanban_root("src-" + marker)
        kb.create_board("shared", name="Shared")
        with connect_closing(board="shared") as conn:
            expected[marker] = kb.create_task(
                conn, title="card-" + marker, assignee="daedalus",
                workspace_kind="dir", workspace_path=str(tmp_path),
            )
        archives[marker] = kt.export_board(
            "shared", str(tmp_path / ("arc-" + marker)))["archive"]

    home = kanban_root("target")

    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_import_in_child,
                    args=(str(home), archives[m], m, barrier, queue))
        for m in ("a", "b")
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(240)
    results = [queue.get(timeout=30) for _ in procs]

    errors = [err for _m, _b, err in results if err]
    assert not errors, results
    slugs = [board for _m, board, _e in results]
    assert len(set(slugs)) == 2, f"both importers claimed the same slug: {slugs}"

    # Every imported card must still be readable somewhere. A replaced
    # kanban.db shows up here as a missing id, which is what this test exists
    # to catch -- the two-scan shape is deliberate: WHICH board holds which
    # card depends on who won the race, and that is not the invariant.
    survived: set[str] = set()
    for slug in slugs:
        db = kb.board_dir(slug) / "kanban.db"
        assert db.is_file(), f"board {slug} has no kanban.db"
        conn = sqlite3.connect(db)
        try:
            survived |= {row[0] for row in conn.execute("SELECT id FROM tasks")}
        finally:
            conn.close()
    assert set(expected.values()) <= survived, (
        f"an imported board was overwritten: expected {sorted(expected.values())}, "
        f"found {sorted(survived)}"
    )
