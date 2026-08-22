"""Contract tests for the safe Kanban admin archive/unarchive kernel.

Public kernel under test (added by a later lane; absent at write time):

    admin_archive_graph(
        conn, root_ids, *, reason, actor,
        dry_run=False, allow_promotions=False, force_running=False,
    ) -> dict
    admin_unarchive(conn, task_ids=None, *, group_id=None, actor) -> dict

Lane-1 pre-kernel gate: TestFixtureSelfChecks must pass today against
direct SQL only; TestAdminArchiveGraph and TestAdminUnarchive must fail
exclusively through the single kernel shim (KERNEL-MISSING marker),
never through import, fixture, or runtime errors. Every FAILED short
summary line must contain the marker, so kernel test method names are
deliberately short enough that the reason survives `-rf` width
truncation next to the node id.

Isolation: every connection is opened on an explicit db_path under a
per-test temporary HERMES_HOME. Nothing in this module resolves the
real ~/.hermes kanban root, the develop-builds board, or another
worktree DB.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

GROUP_ID_RE = re.compile(r"^ag_[0-9a-f]{16}$")
KERNEL_MISS = "KERNEL-MISSING: admin_archive_graph"
REASON = "admin archive contract lane 1"
ACTOR = "dozer"
DEAD_PID = 424242


# ---------------------------------------------------------------------------
# Kernel access: exactly one shim. No direct kb.admin_* calls anywhere else.
# ---------------------------------------------------------------------------


def _kernel(name, *args, **kwargs):
    try:
        fn = getattr(kb, name)
    except AttributeError:
        # Pre-kernel gate: every kernel op fails with the exact card marker.
        # The try/except AttributeError shape (not hasattr) leaves unrelated
        # implementation failures unmasked once the kernel exists.
        pytest.fail("KERNEL-MISSING: admin_archive_graph")
    return fn(*args, **kwargs)


def _archive(conn, root_ids, **kwargs):
    kwargs.setdefault("reason", REASON)
    kwargs.setdefault("actor", ACTOR)
    return _kernel("admin_archive_graph", conn, list(root_ids), **kwargs)


def _unarchive(conn, **kwargs):
    kwargs.setdefault("actor", ACTOR)
    return _kernel("admin_unarchive", conn, **kwargs)


def _call(fn, *args, **kwargs):
    """Invoke a kernel op; return (result, exception) without raising.

    The one exception: a pre-kernel shim miss (KERNEL-MISSING marker)
    is re-raised verbatim so every kernel test fails with exactly that
    marker before the kernel exists.
    """
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - contract: any refusal form
        if KERNEL_MISS in str(exc):
            raise
        return None, exc


def assert_refused(result, exc, *needles):
    """A refusal must surface (error dict or raised exception) and list
    every id / flag named in ``needles``."""
    if exc is not None:
        blob = str(exc)
    else:
        assert result is not None
        blob = json.dumps(result)
    for needle in needles:
        assert needle in blob, f"{needle!r} missing from refusal: {blob}"


def assert_success(result, exc) -> dict:
    """Assert kernel success; return the result narrowed to dict."""
    assert exc is None, f"expected success, kernel raised: {exc!r}"
    assert result is not None
    assert not result.get("error"), f"unexpected error: {result.get('error')}"
    return result


def assert_failed(result, exc):
    """Assert a refusal of any form (error dict or raised exception)."""
    assert exc is not None or (result is not None and result.get("error")), (
        "expected a refusal, got success"
    )


# ---------------------------------------------------------------------------
# Connection, graph-building, and inspection helpers (direct SQL only).
# ---------------------------------------------------------------------------


def make_conn(db_path):
    """Open a board DB at an explicit path. No default-path resolution."""
    conn = kb.connect(db_path=Path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


class FailConn:
    """Attribute-forwarding connection that raises on its Nth execute.

    ``fail_at`` is 1-based: the Nth ``execute`` call raises
    sqlite3.OperationalError; earlier and later calls pass through.
    Counts every execute, including read-only and transaction-control
    statements.
    """

    def __init__(self, conn, fail_at):
        self._conn = conn
        self._fail_at = int(fail_at)
        self.execute_count = 0

    def execute(self, sql, *args, **kwargs):
        self.execute_count += 1
        if self.execute_count == self._fail_at:
            raise sqlite3.OperationalError("injected failure")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._conn.close()
        return False


def make_fail_once_conn(db_path, fail_at):
    return FailConn(kb.connect(db_path=Path(db_path)), fail_at)


class FailWriteConn:
    """Attribute-forwarding connection that fails on the Nth data write.

    ``fail_on_write`` is 1-based over INSERT/UPDATE/DELETE statements
    only (read-only and transaction-control statements pass through).
    Failing on the Nth write guarantees at least one earlier data
    mutation succeeded inside the transaction window, so a correct
    transactional kernel must roll it all back.
    """

    def __init__(self, conn, fail_on_write):
        self._conn = conn
        self._fail_on_write = int(fail_on_write)
        self.write_count = 0

    def execute(self, sql, *args, **kwargs):
        head = (sql or "").lstrip().split(None, 1)
        if head and head[0].upper() in ("INSERT", "UPDATE", "DELETE"):
            self.write_count += 1
            if self.write_count == self._fail_on_write:
                raise sqlite3.OperationalError("injected write failure")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._conn.close()
        return False


def make_fail_write_conn(db_path, fail_on_write):
    return FailWriteConn(kb.connect(db_path=Path(db_path)), fail_on_write)


def create(db, title, **kwargs):
    return kb.create_task(db, title=title, **kwargs)


def link(db, parent, child):
    kb.link_tasks(db, parent, child)


def _statuses(db):
    return {
        row["id"]: row["status"]
        for row in db.execute("SELECT id, status FROM tasks").fetchall()
    }


def _links_of(db):
    return {
        (row["parent_id"], row["child_id"])
        for row in db.execute("SELECT parent_id, child_id FROM task_links").fetchall()
    }


def _events_by_kind(db, task_id, kind):
    return db.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id",
        (task_id, kind),
    ).fetchall()


def _payloads(db, task_id, kind):
    out = []
    for row in _events_by_kind(db, task_id, kind):
        out.append(json.loads(row["payload"]) if row["payload"] else None)
    return out


def _admin_event_rows(db):
    return db.execute(
        "SELECT task_id, payload FROM task_events WHERE kind = 'admin_archived' "
        "ORDER BY task_id, id"
    ).fetchall()


def _comments_of(db, task_id):
    return db.execute(
        "SELECT id, author, body FROM task_comments WHERE task_id = ? "
        "ORDER BY id",
        (task_id,),
    ).fetchall()


def _runs_of(db, task_id):
    return db.execute(
        "SELECT id, status, outcome, ended_at FROM task_runs WHERE task_id = ? "
        "ORDER BY id",
        (task_id,),
    ).fetchall()


def _table_snapshot(db, table, task_id=None):
    """Set of full-row value tuples from ``table``.

    Optionally restricted to rows whose ``task_id`` column equals
    ``task_id``. Keyed by the row's values (not a named ``id`` column,
    which some join tables like ``task_links`` lack) so it is
    schema-agnostic and usable for byte/row-identical preservation
    checks.
    """
    sql = f"SELECT * FROM {table}"
    params = ()
    if task_id is not None:
        sql += " WHERE task_id = ?"
        params = (task_id,)
    return {
        tuple(r[k] for k in r.keys())
        for r in db.execute(sql, params).fetchall()
    }


def set_status(db, task_id, status):
    db.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def make_task_running(db, task_id):
    """Promote to ready, claim it, then stamp a fake (dead) worker pid."""
    set_status(db, task_id, "ready")
    claimed = kb.claim_task(db, task_id)
    assert claimed is not None and claimed.status == "running"
    db.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (DEAD_PID, task_id))


def make_task_blocked_nonsticky(db, task_id, failures):
    """status=blocked with NO 'blocked' event -> not a sticky block."""
    db.execute(
        "UPDATE tasks SET status = 'blocked', consecutive_failures = ? "
        "WHERE id = ?",
        (failures, task_id),
    )


def make_task_blocked_sticky(db, task_id, failures):
    """status=blocked preceded by a worker 'blocked' event -> sticky."""
    db.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'blocked', ?, ?)",
        (task_id, json.dumps({"reason": "sticky handoff"}), int(time.time())),
    )
    make_task_blocked_nonsticky(db, task_id, failures)


def make_task_blocked_with_resume(db, task_id, failures, resume_status):
    """status=blocked whose newest covered event carries resume_status."""
    db.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'blocked', ?, ?)",
        (task_id, json.dumps({"resume_status": resume_status}), int(time.time())),
    )
    make_task_blocked_nonsticky(db, task_id, failures)


def assert_topology(db, expected_statuses, expected_links, expected_task_count):
    assert _statuses(db) == expected_statuses
    assert _links_of(db) == expected_links
    n = db.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    assert n == expected_task_count


def spy_recompute(monkeypatch):
    """Patch kb.recompute_ready with a counting spy; return the call log."""
    calls = []
    original = kb.recompute_ready

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(kb, "recompute_ready", spy)
    return calls


def new_board(tmp_path, monkeypatch, index):
    """A second (or Nth) isolated board inside the same test."""
    home = tmp_path / f"hermes-home-{index}"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return kb.kanban_db_path(f"contractboard{index}")


# ---------------------------------------------------------------------------
# Canonical graph builders (normative topologies).
# External (non-closure) tasks are created via the blocked initial status
# and moved to their normative status by direct UPDATE: create_task only
# accepts initial_status in {"running", "blocked"}.
# ---------------------------------------------------------------------------


def build_h1(db):
    """A(todo)->B(todo); A->C1(todo); D(done external)->C1."""
    a = create(db, "h1 A", initial_status="blocked")
    b = create(db, "h1 B", parents=[a])
    c1 = create(db, "h1 C1", parents=[a])
    d = create(db, "h1 D", initial_status="blocked")
    link(db, d, c1)
    set_status(db, a, "todo")
    set_status(db, d, "done")
    return {"a": a, "b": b, "c1": c1, "d": d}


def build_h2(db):
    """H1 plus A->C3, B->C3, D2(done external)->C3. No archived E."""
    ids = build_h1(db)
    c3 = create(db, "h2 C3", parents=[ids["a"], ids["b"]])
    d2 = create(db, "h2 D2", initial_status="blocked")
    link(db, d2, c3)
    set_status(db, d2, "done")
    return {**ids, "c3": c3, "d2": d2}


def build_n1(db):
    """A->C2(todo); Q(todo external)->C2. Q MUST be todo, never ready."""
    a = create(db, "n1 A", initial_status="blocked")
    c2 = create(db, "n1 C2", parents=[a])
    q = create(db, "n1 Q", initial_status="blocked")
    link(db, q, c2)
    set_status(db, a, "todo")
    set_status(db, q, "todo")
    return {"a": a, "c2": c2, "q": q}


def build_i1(db):
    """P(done external)->A(root) ONLY. No extra A->B edge or task."""
    p = create(db, "i1 P", initial_status="blocked")
    a = create(db, "i1 A")
    link(db, p, a)
    set_status(db, a, "todo")
    set_status(db, p, "done")
    return {"p": p, "a": a}


def build_dm(db):
    """A->B, A->C, B->D, C->D, all todo."""
    a = create(db, "dm A", initial_status="blocked")
    b = create(db, "dm B", parents=[a])
    c = create(db, "dm C", parents=[a])
    d = create(db, "dm D", parents=[b, c])
    set_status(db, a, "todo")
    return {"a": a, "b": b, "c": c, "d": d}


def build_ap(db):
    """A->X; E(archived external)->X."""
    a = create(db, "ap A", initial_status="blocked")
    x = create(db, "ap X", parents=[a])
    e = create(db, "ap E", initial_status="blocked")
    link(db, e, x)
    set_status(db, a, "todo")
    set_status(db, e, "archived")
    return {"a": a, "x": x, "e": e}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Primary isolated board: explicit DB path under a temp HERMES_HOME."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    db_path = kb.kanban_db_path("contractboard")
    assert db_path.parent == home / "kanban" / "boards" / "contractboard"
    init = make_conn(db_path)
    init.close()
    return db_path


@pytest.fixture(autouse=True)
def _guard_db_connect(tmp_path, monkeypatch):
    """Fail-closed per-test guard over ``kb.connect``.

    Refuses to open ANY connection whose resolved DB path lies outside
    this test's ``tmp_path``, *before* :func:`kb.connect` can create the
    file/sidecars. This is deliberately stricter than
    ``tests/conftest.py``'s production-root deny-list: it refuses by the
    test's own temporary root without needing to know the real home root
    ahead of time, so a stray path pointing anywhere outside the current
    test's temporary tree is refused on every platform.
    """
    original = kb.connect
    root = Path(tmp_path).resolve()

    def guarded(db_path=None, *, board=None):
        if db_path is None:
            # No explicit path -> kb.connect would resolve via env/board
            # defaults. Under the guard that resolution is never allowed to
            # run, because the resolved target is unknowable ahead of time.
            pytest.fail(
                "FAIL-CLOSED: kb.connect called without an explicit db_path "
                "(default/board-path resolution is not permitted under the "
                "archive guard)"
            )
        resolved = Path(db_path).resolve()
        if not resolved.is_relative_to(root):
            pytest.fail(
                f"FAIL-CLOSED: refused kb.connect to DB path outside this "
                f"test's tmp_path: {resolved}"
            )
        return original(db_path=db_path, board=board)

    monkeypatch.setattr(kb, "connect", guarded)


def assert_result_keys(result, keys):
    missing = set(keys) - set(result.keys())
    assert not missing, f"result missing keys: {sorted(missing)}"


def assert_execution_shape(result):
    assert_result_keys(
        result,
        {
            "dry_run",
            "archive_group_id",
            "root_ids",
            "tasks",
            "skipped_archived",
            "internal_edges",
            "external_edges",
            "would_promote",
            "active_runs",
            "archived_ids",
            "warnings",
        },
    )
    for key in ("root_ids", "skipped_archived", "would_promote", "active_runs",
                "archived_ids"):
        assert result[key] == sorted(result[key]), key
    ids = [t["id"] for t in result["tasks"]]
    assert ids == sorted(ids)
    for entry in result["tasks"]:
        assert set(entry.keys()) == {
            "id", "title", "status", "workspace_path", "workspace_exists",
        }
        assert isinstance(entry["workspace_exists"], bool)
    for edge in result["internal_edges"]:
        assert set(edge.keys()) == {"parent_id", "child_id"}
    assert result["internal_edges"] == sorted(
        result["internal_edges"], key=lambda e: (e["parent_id"], e["child_id"])
    )
    for edge in result["external_edges"]:
        assert set(edge.keys()) == {"direction", "parent_id", "child_id"}
        assert edge["direction"] in ("inbound", "outbound")
    assert result["external_edges"] == sorted(
        result["external_edges"],
        key=lambda e: (e["parent_id"], e["child_id"], e["direction"]),
    )
    for warning in result["warnings"]:
        assert set(warning.keys()) == {"code", "parent_id", "root_id"}
        assert warning["code"] == "live_external_parent"
    assert result["warnings"] == sorted(
        result["warnings"], key=lambda w: (w["root_id"], w["parent_id"])
    )
    json.dumps(result)


# ---------------------------------------------------------------------------
# TestFixtureSelfChecks — no kernel calls; must pass pre-kernel.
# ---------------------------------------------------------------------------


class TestFixtureSelfChecks:
    def test_db_connection_is_isolated_under_temp_home(self, board):
        home = Path(os.environ["HERMES_HOME"])
        real_home = Path.home() / ".hermes"
        real_kanban = real_home / "kanban"
        db_path = Path(board)
        assert db_path.is_file()
        # The DB sits inside this test's temporary HERMES_HOME tree...
        assert db_path.is_relative_to(home)
        assert home == db_path.parents[3]
        # ...and outside the real home / real kanban root.
        assert home != real_home
        assert not db_path.is_relative_to(real_home)
        assert not real_kanban.is_relative_to(home)
        # The connection the test opens must actually point at this file.
        db = make_conn(board)
        with db:
            file_path = db.execute("PRAGMA database_list").fetchone()[2]
        assert Path(file_path).resolve() == db_path.resolve()

    def test_board_path_cannot_resolve_live_roots(self, board):
        home = Path(os.environ["HERMES_HOME"])
        real_home = Path.home() / ".hermes"
        slug = Path(board).parent.name
        assert kb.kanban_db_path(slug) == Path(board)
        assert kb.boards_root() == home / "kanban" / "boards"
        # A request for the live develop-builds board must resolve inside
        # this test's temporary home, never into the real kanban root.
        dev_builds = kb.kanban_db_path("develop-builds")
        assert dev_builds == home / "kanban" / "boards" / "develop-builds" / "kanban.db"
        assert not dev_builds.is_relative_to(real_home)
        assert real_home not in home.parents

    def test_make_fail_once_conn_raises_exactly_on_nth_execute(self, board):
        conn = make_fail_once_conn(board, fail_at=3)
        with conn:
            conn.execute("SELECT 1")
            conn.execute("SELECT 2")
            assert conn.execute_count == 2
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("SELECT 3")
            assert conn.execute_count == 3
            # Later executes pass through.
            conn.execute("SELECT 4")
            conn.execute("SELECT 5")
        assert conn.execute_count == 5

    def test_make_fail_write_conn_raises_on_nth_write(self, board):
        # Reads and transaction-control statements never count toward the
        # write counter; a real data write does, and fails on the Nth one.
        conn2 = make_fail_write_conn(board, fail_on_write=1)
        with conn2:
            conn2.execute("SELECT 1")
            conn2.execute("BEGIN")
            conn2.execute("COMMIT")
            assert conn2.write_count == 0
            with pytest.raises(sqlite3.OperationalError):
                # A real INSERT counts as write #1 and trips the failure.
                conn2.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES ('t_x', 'note', NULL, 0)"
                )
            assert conn2.write_count == 1

    def test_guard_refuses_db_path_outside_tmp_path(self, board, tmp_path):
        # A resolved DB path outside THIS test's tmp_path must be refused
        # by the fail-closed guard before sqlite3 can create anything.
        outside = tmp_path.parent / "guard-escape"
        db_target = outside / "escaped.db"
        outside.mkdir(parents=True, exist_ok=True)
        assert not db_target.is_relative_to(Path(tmp_path).resolve())
        for suffix in ("", "-wal", "-shm"):
            assert not Path(str(db_target) + suffix).exists()
        with pytest.raises(pytest.fail.Exception):
            kb.connect(db_path=db_target)
        # The refused call created no DB, WAL, or SHM file.
        for suffix in ("", "-wal", "-shm"):
            assert not Path(str(db_target) + suffix).exists()

    @pytest.mark.parametrize("builder_name", ["h1", "h2", "n1", "i1", "dm", "ap"])
    def test_graph_builder_topology_by_direct_sql(self, board, builder_name):
        db = make_conn(board)
        with db:
            ids = getattr(self, f"_build_{builder_name}")(db)
            self._expect_topology(builder_name, db, ids)

    # -- builder wrappers + exact-topology expectations -------------------

    def _build_h1(self, db):
        return build_h1(db)

    def _build_h2(self, db):
        return build_h2(db)

    def _build_n1(self, db):
        return build_n1(db)

    def _build_i1(self, db):
        return build_i1(db)

    def _build_dm(self, db):
        return build_dm(db)

    def _build_ap(self, db):
        return build_ap(db)

    def _expect_topology(self, name, db, ids):
        if name == "h1":
            assert_topology(
                db,
                {ids["a"]: "todo", ids["b"]: "todo",
                 ids["c1"]: "todo", ids["d"]: "done"},
                {(ids["a"], ids["b"]), (ids["a"], ids["c1"]),
                 (ids["d"], ids["c1"])},
                4,
            )
        elif name == "h2":
            assert_topology(
                db,
                {ids["a"]: "todo", ids["b"]: "todo",
                 ids["c1"]: "todo", ids["d"]: "done",
                 ids["c3"]: "todo", ids["d2"]: "done"},
                {(ids["a"], ids["b"]), (ids["a"], ids["c1"]),
                 (ids["d"], ids["c1"]),
                 (ids["a"], ids["c3"]), (ids["b"], ids["c3"]),
                 (ids["d2"], ids["c3"])},
                6,
            )
        elif name == "n1":
            assert_topology(
                db,
                {ids["a"]: "todo", ids["c2"]: "todo", ids["q"]: "todo"},
                {(ids["a"], ids["c2"]), (ids["q"], ids["c2"])},
                3,
            )
        elif name == "i1":
            assert_topology(
                db,
                {ids["p"]: "done", ids["a"]: "todo"},
                {(ids["p"], ids["a"])},
                2,
            )
        elif name == "dm":
            assert_topology(
                db,
                {ids[k]: "todo" for k in ("a", "b", "c", "d")},
                {(ids["a"], ids["b"]), (ids["a"], ids["c"]),
                 (ids["b"], ids["d"]), (ids["c"], ids["d"])},
                4,
            )
        elif name == "ap":
            assert_topology(
                db,
                {ids["a"]: "todo", ids["x"]: "todo", ids["e"]: "archived"},
                {(ids["a"], ids["x"]), (ids["e"], ids["x"])},
                3,
            )


# ---------------------------------------------------------------------------
# TestAdminArchiveGraph — dominated-closure archive kernel contract.
# ---------------------------------------------------------------------------


class TestAdminArchiveGraph:
    @pytest.fixture
    def ids(self, board):
        db = make_conn(board)
        built = build_h1(db)
        db.close()
        return built

    def test_dm(self, board, tmp_path, monkeypatch):
        # Deterministic task ids so two independent runs produce
        # byte-comparable results (the random group id is normalized out).
        counter = {"i": 0}

        def det_id():
            counter["i"] += 1
            return f"t_det{counter['i']:04d}"

        monkeypatch.setattr(kb, "_new_task_id", det_id)

        def run_once(db_path):
            db = make_conn(db_path)
            built = build_dm(db)
            result = assert_success(*_call(_archive, db, [built["a"], built["a"]]))
            db.close()
            return result, built

        result, built = run_once(board)
        assert result["dry_run"] is False
        assert GROUP_ID_RE.fullmatch(result["archive_group_id"])
        assert_execution_shape(result)
        assert result["root_ids"] == [built["a"]]
        assert result["skipped_archived"] == []
        assert result["warnings"] == []
        assert result["would_promote"] == []
        assert result["active_runs"] == []
        assert result["archived_ids"] == sorted(
            [built[k] for k in ("a", "b", "c", "d")]
        )
        assert [t["id"] for t in result["tasks"]] == sorted(
            [built[k] for k in ("a", "b", "c", "d")]
        )
        assert result["internal_edges"] == sorted(
            [
                {"parent_id": built["a"], "child_id": built["b"]},
                {"parent_id": built["a"], "child_id": built["c"]},
                {"parent_id": built["b"], "child_id": built["d"]},
                {"parent_id": built["c"], "child_id": built["d"]},
            ],
            key=lambda e: (e["parent_id"], e["child_id"]),
        )
        assert result["external_edges"] == []

        counter["i"] = 0
        second_path = new_board(tmp_path, monkeypatch, "dm2")
        result2, built2 = run_once(second_path)
        assert built == built2  # deterministic ids across runs
        normalized_a = dict(result)
        normalized_a["archive_group_id"] = None
        normalized_b = dict(result2)
        normalized_b["archive_group_id"] = None
        assert json.loads(json.dumps(normalized_a)) == json.loads(
            json.dumps(normalized_b)
        )

    def test_h1_ref(self, board, ids, monkeypatch):
        db = make_conn(board)
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_archive, db, [ids["a"]])
        assert_refused(result, exc, ids["c1"], "--allow-promotions")
        if result is not None:
            assert result["archived_ids"] == []
        statuses = _statuses(db)
        assert statuses[ids["a"]] == "todo"
        assert statuses[ids["b"]] == "todo"
        assert statuses[ids["c1"]] == "todo"
        assert statuses[ids["d"]] == "done"
        assert len(_admin_event_rows(db)) == 0
        assert calls == []  # a refusal never recomputes
        db.close()

    def test_h1_up(self, board, ids, monkeypatch):
        db = make_conn(board)
        calls = spy_recompute(monkeypatch)
        result = assert_success(
            *_call(_archive, db, [ids["a"]], allow_promotions=True)
        )
        assert result["dry_run"] is False
        assert GROUP_ID_RE.fullmatch(result["archive_group_id"])
        assert result["archived_ids"] == sorted([ids["a"], ids["b"]])
        assert result["would_promote"] == [ids["c1"]]
        statuses = _statuses(db)
        assert statuses[ids["a"]] == "archived"
        assert statuses[ids["b"]] == "archived"
        assert statuses[ids["c1"]] == "ready"
        assert statuses[ids["d"]] == "done"
        assert len(calls) == 1  # recompute runs exactly once, after commit
        group = result["archive_group_id"]
        for task_id in (ids["a"], ids["b"]):
            payloads = _payloads(db, task_id, "admin_archived")
            assert len(payloads) == 1
            assert payloads[0]["archive_group_id"] == group
        db.close()

    def test_h2(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        built = build_h2(db)
        result, exc = _call(_archive, db, [built["a"]])
        assert_refused(result, exc, built["c1"], built["c3"],
                       "--allow-promotions")
        statuses = _statuses(db)
        assert statuses[built["a"]] == "todo"
        assert statuses[built["b"]] == "todo"

        result2 = assert_success(
            *_call(_archive, db, [built["a"]], allow_promotions=True)
        )
        assert result2["would_promote"] == sorted([built["c1"], built["c3"]])
        statuses = _statuses(db)
        assert statuses[built["a"]] == "archived"
        assert statuses[built["b"]] == "archived"
        assert statuses[built["c1"]] == "ready"
        assert statuses[built["c3"]] == "ready"
        assert statuses[built["d"]] == "done"
        assert statuses[built["d2"]] == "done"
        db.close()

    def test_n1(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        built = build_n1(db)
        result = assert_success(*_call(_archive, db, [built["a"]]))
        assert result["would_promote"] == []
        statuses = _statuses(db)
        assert statuses[built["a"]] == "archived"
        assert statuses[built["c2"]] == "todo"
        assert statuses[built["q"]] == "todo"
        assert result["external_edges"] == [
            {"direction": "outbound", "parent_id": built["a"],
             "child_id": built["c2"]}
        ]
        assert result["warnings"] == []
        db.close()

    def test_i1(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        built = build_i1(db)
        result = assert_success(*_call(_archive, db, [built["a"]]))
        assert result["warnings"] == [
            {"code": "live_external_parent", "parent_id": built["p"],
             "root_id": built["a"]}
        ]
        assert result["external_edges"] == [
            {"direction": "inbound", "parent_id": built["p"],
             "child_id": built["a"]}
        ]
        assert result["archived_ids"] == [built["a"]]
        statuses = _statuses(db)
        assert statuses[built["a"]] == "archived"
        assert statuses[built["p"]] == "done"
        db.close()

    def test_ap(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        built = build_ap(db)
        result = assert_success(*_call(_archive, db, [built["a"]]))
        assert result["archived_ids"] == sorted([built["a"], built["x"]])
        assert result["warnings"] == []
        assert result["external_edges"] == [
            {"direction": "inbound", "parent_id": built["e"],
             "child_id": built["x"]}
        ]
        statuses = _statuses(db)
        assert statuses[built["x"]] == "archived"
        assert statuses[built["e"]] == "archived"
        db.close()

    def _build_eligibility_board(self, db):
        """R(ready) + D(done external); six boundary children (parents R + D)
        covering every recompute_ready eligibility branch."""
        r = create(db, "elig R")
        d = create(db, "elig D", initial_status="blocked")
        set_status(db, d, "done")
        c1 = create(db, "elig C1 todo", parents=[r, d])
        c4 = create(db, "elig C4 nonsticky", parents=[r, d])
        c5 = create(db, "elig C5 sticky", parents=[r, d])
        c6 = create(db, "elig C6 at limit", parents=[r, d])
        c7 = create(db, "elig C7 resume", parents=[r, d])
        c8 = create(db, "elig C8 per-task limit", parents=[r, d],
                    max_retries=5)
        make_task_blocked_nonsticky(db, c4, failures=1)
        make_task_blocked_sticky(db, c5, failures=0)
        make_task_blocked_nonsticky(db, c6, failures=kb.DEFAULT_FAILURE_LIMIT)
        make_task_blocked_with_resume(db, c7, failures=0, resume_status="review")
        make_task_blocked_nonsticky(db, c8, failures=3)
        children = {
            "c1": c1, "c4": c4, "c5": c5, "c6": c6, "c7": c7, "c8": c8,
        }
        return {"r": r, "d": d, **children}, r, children

    def test_elig(self, board, tmp_path, monkeypatch):
        # Independent oracle: same topology in a second board; archive the
        # root with direct SQL, then run the stock recompute_ready.
        db2_path = new_board(tmp_path, monkeypatch, "elig-oracle")
        db2 = make_conn(db2_path)
        built2, r2, _ = self._build_eligibility_board(db2)
        set_status(db2, r2, "archived")
        for key in sorted(k for k in built2 if k not in ("r", "d")):
            kb.recompute_ready(db2)
        oracle = {key: kb.get_task(db2, built2[key]).status
                  for key in built2 if key not in ("r", "d")}
        promoted_keys = {key for key, status in oracle.items()
                         if status in ("ready", "review")}
        db2.close()

        db = make_conn(board)
        built, r, children = self._build_eligibility_board(db)

        # Refusal phase: hazards present -> default refuses, no recompute.
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_archive, db, [r])
        assert_refused(result, exc, "--allow-promotions")
        assert calls == []
        assert _statuses(db)[r] == "ready"

        # Override phase: would_promote must equal the oracle's promoted set.
        calls = spy_recompute(monkeypatch)
        result = assert_success(
            *_call(_archive, db, [r], allow_promotions=True)
        )
        assert result["would_promote"] == sorted(
            children[key] for key in ("c1", "c4", "c7", "c8")
        )
        # The kernel board's random task ids can never equal the oracle
        # board's random ids (each board generates an independent id space),
        # so compare promoted fixture KEYS instead: the canonical set is
        # exactly {c1, c4, c7, c8}.
        id_to_key = {tid: key for key, tid in children.items()}
        assert {id_to_key[tid] for tid in result["would_promote"]} == promoted_keys
        statuses = _statuses(db)
        assert statuses[r] == "archived"
        for key in ("c1", "c4", "c5", "c6", "c7", "c8"):
            assert statuses[children[key]] == oracle[key], key
        assert statuses[children["c1"]] == "ready"
        assert statuses[children["c4"]] == "ready"
        assert statuses[children["c5"]] == "blocked"
        assert statuses[children["c6"]] == "blocked"
        assert statuses[children["c7"]] == "review"
        assert statuses[children["c8"]] == "ready"
        assert len(calls) == 1
        db.close()

    def test_run(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "run A")
        b = create(db, "run B", parents=[a])
        make_task_running(db, a)
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_archive, db, [a])
        assert_refused(result, exc, a, "--force-running")
        if result is not None:
            assert result["active_runs"] == [a]
            assert result["archived_ids"] == []
        statuses = _statuses(db)
        assert statuses[a] == "running"
        assert statuses[b] == "todo"
        assert calls == []
        db.close()

    def test_claim(self, board, tmp_path, monkeypatch):
        # Active claim/run detection must fire even when task status alone
        # is no longer "running" (a corrupted/stale status field).
        db = make_conn(board)
        a = create(db, "claim A")
        b = create(db, "claim B", parents=[a])
        make_task_running(db, a)
        set_status(db, a, "ready")  # status is NOT running now...
        row = db.execute(
            "SELECT status, current_run_id, claim_lock, worker_pid "
            "FROM tasks WHERE id = ?",
            (a,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["current_run_id"] is not None  # ...but the claim/run remain
        assert row["claim_lock"] is not None
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_archive, db, [a])
        assert_refused(result, exc, a, "--force-running")
        if result is not None:
            assert result["active_runs"] == [a]
            assert result["archived_ids"] == []
        assert _statuses(db)[a] == "ready"
        assert calls == []
        db.close()

    def test_force(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "force A")
        b = create(db, "force B", parents=[a])
        make_task_running(db, a)
        monkeypatch.setattr(
            kb, "_terminate_reclaimed_worker", lambda *a, **k: {},
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        result = assert_success(*_call(_archive, db, [a], force_running=True))
        assert result["active_runs"] == [a]
        assert a in result["archived_ids"]
        assert b in result["archived_ids"]
        statuses = _statuses(db)
        assert statuses[a] == "archived"
        assert statuses[b] == "archived"
        row = db.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (a,),
        ).fetchone()
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None
        runs = _runs_of(db, a)
        assert runs[-1]["outcome"] == "reclaimed"
        assert runs[-1]["ended_at"] is not None
        payloads = _payloads(db, a, "admin_archived")
        assert len(payloads) == 1
        assert payloads[0]["prior_status"] == "running"
        db.close()

    def test_dry(self, board, tmp_path):
        db = make_conn(board)
        built = build_h1(db)
        # Establish a stable read transaction and read mark BEFORE the
        # before-snapshot, so ordinary SQLite read-lock bookkeeping in the
        # WAL/SHM cannot perturb the byte comparison of a write-free (dry)
        # planner. Without the mark, the kernel's first read would itself
        # create a WAL/SHM read mark between the two snapshots.
        db.execute("BEGIN")
        db.execute("SELECT COUNT(*) FROM tasks")
        files = []
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(board) + suffix)
            if candidate.exists():
                files.append(candidate)
        before = {str(p): p.read_bytes() for p in files}
        result = assert_success(*_call(_archive, db, [built["a"]], dry_run=True))
        after = {str(p): p.read_bytes() for p in files}
        assert before == after
        assert result["dry_run"] is True
        assert result["archive_group_id"] is None
        assert result["archived_ids"] == []
        assert result["root_ids"] == [built["a"]]
        assert result["would_promote"] == [built["c1"]]
        assert result["active_runs"] == []
        group_ids = [
            row["id"]
            for row in db.execute("SELECT id FROM tasks").fetchall()
            if row["id"].startswith("ag_")
        ]
        assert group_ids == []
        assert len(_admin_event_rows(db)) == 0
        assert _statuses(db)[built["a"]] == "todo"
        db.close()

    def test_pres(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        # A real child link A->B plus per-task workspace directories.
        ws_a = Path(board).parent / "workspaces" / "pres-a"
        ws_a.mkdir(parents=True)
        (ws_a / "a.txt").write_bytes(b"A-workspace\x00bytes\n")
        a = create(db, "pres A", workspace_kind="dir", workspace_path=str(ws_a))
        b = create(db, "pres B", parents=[a])
        ws_b = Path(board).parent / "workspaces" / "pres-b"
        ws_b.mkdir(parents=True)
        (ws_b / "b.txt").write_bytes(b"B-workspace\n")
        db.execute(
            "UPDATE tasks SET workspace_kind = 'dir', workspace_path = ? "
            "WHERE id = ?",
            (str(ws_b), b),
        )
        # Comment + prior event + attachment on A; prior event on B.
        kb.add_comment(db, a, ACTOR, "note before archive")
        db.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'edited', ?, ?)",
            (a, json.dumps({"field": "title"}), int(time.time())),
        )
        db.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'scoped', ?, ?)",
            (b, json.dumps({"scope": 7}), int(time.time())),
        )
        att_dir = kb.attachments_root(Path(board).parent.name) / a
        att_dir.mkdir(parents=True)
        blob = att_dir / "spec.pdf"
        blob.write_bytes(b"%PDF-fake-bytes")
        db.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (a, "spec.pdf", str(blob), "application/pdf",
             blob.stat().st_size, ACTOR, int(time.time())),
        )
        # One real run row on A (a task_runs row, not just a status).
        db.execute(
            "INSERT INTO task_runs (task_id, profile, step_key, status, "
            "outcome, started_at) VALUES (?, ?, NULL, 'running', 'running', ?)",
            (a, ACTOR, int(time.time())),
        )

        before = {
            "comments": _table_snapshot(db, "task_comments"),
            "events": _table_snapshot(db, "task_events"),
            "atts": _table_snapshot(db, "task_attachments"),
            "links": _table_snapshot(db, "task_links"),
            "runs": _table_snapshot(db, "task_runs"),
        }
        a_bytes_before = (ws_a / "a.txt").read_bytes()
        b_bytes_before = (ws_b / "b.txt").read_bytes()
        blob_before = blob.read_bytes()
        row_before_a = tuple(
            db.execute("SELECT workspace_path, workspace_kind FROM tasks WHERE id=?",
                       (a,)).fetchone()
        )
        row_before_b = tuple(
            db.execute("SELECT workspace_path, workspace_kind FROM tasks WHERE id=?",
                       (b,)).fetchone()
        )

        cleanup_calls = []
        monkeypatch.setattr(
            kb, "_cleanup_workspace",
            lambda conn, task_id: cleanup_calls.append(task_id),
        )
        result = assert_success(*_call(_archive, db, [a]))

        assert cleanup_calls == []  # archive never reapplies the completer
        assert _statuses(db)[a] == "archived"
        assert _statuses(db)[b] == "archived"
        # Files byte-identical.
        assert (ws_a / "a.txt").read_bytes() == a_bytes_before
        assert (ws_b / "b.txt").read_bytes() == b_bytes_before
        assert blob.is_file() and blob.read_bytes() == blob_before
        # Workspace fields row-identical.
        row_after_a = tuple(
            db.execute("SELECT workspace_path, workspace_kind FROM tasks WHERE id=?",
                       (a,)).fetchone()
        )
        row_after_b = tuple(
            db.execute("SELECT workspace_path, workspace_kind FROM tasks WHERE id=?",
                       (b,)).fetchone()
        )
        assert row_after_a == row_before_a
        assert row_after_b == row_before_b
        # Every pre-archive row survives identically (subset of after).
        after = {
            "comments": _table_snapshot(db, "task_comments"),
            "events": _table_snapshot(db, "task_events"),
            "atts": _table_snapshot(db, "task_attachments"),
            "links": _table_snapshot(db, "task_links"),
            "runs": _table_snapshot(db, "task_runs"),
        }
        for key in before:
            assert before[key] <= after[key], key
        # The ONLY allowed additions: one audit comment on root a, and the
        # admin_archived events (one per archived task).
        new_comments = after["comments"] - before["comments"]
        new_events = after["events"] - before["events"]
        assert len(new_comments) == 1
        new_admin_count = (
            len([r for r in db.execute(
                "SELECT 1 FROM task_events WHERE kind='admin_archived'").fetchall()])
        )
        assert new_admin_count == 2  # a and b
        assert len(_events_by_kind(db, a, "admin_archived")) == 1
        assert len(_events_by_kind(db, b, "admin_archived")) == 1
        assert len(new_events) >= 2
        # Archive audit comment carries the exact root body. _comments_of
        # orders by id ascending, so "[0]" is the pre-archive note and the
        # audit comment is the NEWEST row ("[-1]").
        comments_a = _comments_of(db, a)
        assert len(comments_a) == 2  # the pre-archive note + exactly one audit
        assert comments_a[0]["body"] == "note before archive"  # prior row intact
        assert comments_a[-1]["author"] == ACTOR
        group = result["archive_group_id"]
        assert comments_a[-1]["body"] == (
            f"Admin-archived graph {group} (2 tasks): {REASON}"
        )
        assert _statuses(db)[a] == "archived"
        db.close()

    def test_ghost(self, board, ids):
        db = make_conn(board)
        snapshot_statuses = _statuses(db)
        snapshot_admin = len(_admin_event_rows(db))
        snapshot_links = _links_of(db)
        result, exc = _call(_archive, db, [ids["a"], "zz_ghost_1"])
        assert_refused(result, exc, "zz_ghost_1")
        if result is not None:
            assert result["archived_ids"] == []
        assert _statuses(db) == snapshot_statuses
        assert len(_admin_event_rows(db)) == snapshot_admin
        assert _links_of(db) == snapshot_links
        result2, exc2 = _call(_archive, db, ["zz_ghost_2"])
        assert_refused(result2, exc2, "zz_ghost_2")
        assert _statuses(db) == snapshot_statuses
        db.close()

    def test_skip(self, board, ids):
        db = make_conn(board)
        result1 = assert_success(
            *_call(_archive, db, [ids["a"]], allow_promotions=True)
        )
        admin_count = len(_admin_event_rows(db))
        assert admin_count == 2  # a and b each carry one admin_archived
        b_admin_events = len(_events_by_kind(db, ids["b"], "admin_archived"))
        assert b_admin_events == 1
        result2 = assert_success(*_call(_archive, db, [ids["a"]]))
        assert result2["skipped_archived"] == sorted([ids["a"], ids["b"]])
        assert result2["archived_ids"] == []
        assert len(_admin_event_rows(db)) == admin_count
        assert len(_events_by_kind(db, ids["b"], "admin_archived")) == (
            b_admin_events
        )
        assert _statuses(db)[ids["b"]] == "archived"
        db.close()

    def test_edges(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        built = build_h2(db)
        result = assert_success(
            *_call(_archive, db, [built["a"]], allow_promotions=True)
        )
        expected_outbound = sorted(
            [
                {"direction": "outbound", "parent_id": built["a"],
                 "child_id": built["c1"]},
                {"direction": "outbound", "parent_id": built["a"],
                 "child_id": built["c3"]},
                {"direction": "outbound", "parent_id": built["b"],
                 "child_id": built["c3"]},
            ],
            key=lambda e: (e["parent_id"], e["child_id"], e["direction"]),
        )
        assert result["external_edges"] == expected_outbound
        assert result["internal_edges"] == [
            {"parent_id": built["a"], "child_id": built["b"]},
        ]
        # D->C1 touches no closure member: it must not appear at all.
        assert (built["d"], built["c1"]) not in {
            (e["parent_id"], e["child_id"]) for e in result["external_edges"]
        }
        # Inbound side: I1 topology.
        db_i1 = make_conn(new_board(tmp_path, monkeypatch, "i1-edges"))
        i1 = build_i1(db_i1)
        result_i1 = assert_success(*_call(_archive, db_i1, [i1["a"]]))
        assert result_i1["external_edges"] == [
            {"direction": "inbound", "parent_id": i1["p"],
             "child_id": i1["a"]}
        ]
        # Invariant: a non-archived inbound parent targets only a root.
        for warning in result_i1["warnings"]:
            assert warning["root_id"] in result_i1["root_ids"]
            assert warning["parent_id"] not in result_i1["archived_ids"]
        db.close()
        db_i1.close()

    def test_reason(self, board, ids):
        db = make_conn(board)
        result, exc = _call(_archive, db, [ids["a"]], reason="   ",
                            actor=ACTOR)
        assert_failed(result, exc)
        result2, exc2 = _call(_archive, db, [ids["a"]], reason=REASON,
                              actor="  ")
        assert_failed(result2, exc2)
        assert _statuses(db)[ids["a"]] == "todo"
        assert len(_admin_event_rows(db)) == 0
        db.close()

    def test_rb_sql(self, board, tmp_path, monkeypatch):
        db_path = new_board(tmp_path, monkeypatch, "rollback")
        db = make_conn(db_path)
        built = build_dm(db)
        statuses_before = _statuses(db)
        events_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_events"
        ).fetchone()["n"]
        comments_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_comments"
        ).fetchone()["n"]
        db.close()
        failing = make_fail_write_conn(db_path, fail_on_write=2)
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_archive, failing, [built["a"]])
        assert_failed(result, exc)
        probe = make_conn(db_path)
        assert _statuses(probe) == statuses_before
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_events")
            .fetchone()["n"] == events_before
        )
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_comments")
            .fetchone()["n"] == comments_before
        )
        assert len(_admin_event_rows(probe)) == 0
        assert calls == []  # recompute never runs on a rolled-back archive
        probe.close()

    def test_rbev(self, board, tmp_path, monkeypatch):
        db_path = new_board(tmp_path, monkeypatch, "rollback-ev")
        db = make_conn(db_path)
        built = build_dm(db)
        statuses_before = _statuses(db)
        events_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_events"
        ).fetchone()["n"]
        comments_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_comments"
        ).fetchone()["n"]
        db.close()
        db = make_conn(db_path)
        calls = spy_recompute(monkeypatch)
        orig = kb._append_event

        def boom(conn, task_id, kind, payload=None, **kwargs):
            if kind == "admin_archived":
                raise RuntimeError("injected admin_archived failure")
            return orig(conn, task_id, kind, payload=payload, **kwargs)

        monkeypatch.setattr(kb, "_append_event", boom)
        result, exc = _call(_archive, db, [built["a"]])
        assert_failed(result, exc)
        probe = make_conn(db_path)
        assert _statuses(probe) == statuses_before
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_events")
            .fetchone()["n"] == events_before
        )
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_comments")
            .fetchone()["n"] == comments_before
        )
        assert len(_admin_event_rows(probe)) == 0
        assert calls == []
        db.close()
        probe.close()

    def test_rbc(self, board, tmp_path, monkeypatch):
        db_path = new_board(tmp_path, monkeypatch, "rollback-cmt")
        db = make_conn(db_path)
        built = build_dm(db)
        statuses_before = _statuses(db)
        events_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_events"
        ).fetchone()["n"]
        comments_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_comments"
        ).fetchone()["n"]
        db.close()
        db = make_conn(db_path)
        calls = spy_recompute(monkeypatch)

        def boom(conn, task_id, author, body):
            raise RuntimeError("injected audit comment failure")

        monkeypatch.setattr(kb, "add_comment", boom)
        result, exc = _call(_archive, db, [built["a"]])
        assert_failed(result, exc)
        probe = make_conn(db_path)
        assert _statuses(probe) == statuses_before
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_events")
            .fetchone()["n"] == events_before
        )
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_comments")
            .fetchone()["n"] == comments_before
        )
        assert len(_admin_event_rows(probe)) == 0
        assert calls == []
        db.close()
        probe.close()

    def test_audit(self, board, ids):
        db = make_conn(board)
        result = assert_success(
            *_call(_archive, db, [ids["a"], ids["a"]], allow_promotions=True)
        )
        group = result["archive_group_id"]
        root_comments = _comments_of(db, ids["a"])
        # Exactly one audit comment on the deduped root, exact body.
        assert len(root_comments) == 1
        assert root_comments[0]["author"] == ACTOR
        assert root_comments[0]["body"] == (
            f"Admin-archived graph {group} (2 tasks): {REASON}"
        )
        for other in (ids["b"], ids["c1"], ids["d"]):
            assert _comments_of(db, other) == []
        payloads = _payloads(db, ids["a"], "admin_archived")
        assert len(payloads) == 1
        assert payloads[0]["archive_group_id"] == group
        assert payloads[0]["actor"] == ACTOR
        assert payloads[0]["reason"] == REASON
        db.close()


# ---------------------------------------------------------------------------
# TestAdminUnarchive — transactional group/direct unarchive kernel contract.
# ---------------------------------------------------------------------------


class TestAdminUnarchive:
    def test_group_rt(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "unr A")
        c = create(db, "unr C", parents=[a])
        # c lands todo (parent a not done) -> the gated-todo restore case.
        result_a = assert_success(*_call(_archive, db, [a]))
        group = result_a["archive_group_id"]
        assert GROUP_ID_RE.fullmatch(group)
        assert set(result_a["archived_ids"]) == {a, c}

        calls = spy_recompute(monkeypatch)
        result = assert_success(*_call(_unarchive, db, group_id=group))
        assert set(result.keys()) >= {
            "group_id", "restored_ids", "skipped_ids", "missing_workspaces",
        }
        assert result["group_id"] == group
        assert result["restored_ids"] == sorted([a, c])
        assert result["restored_ids"] == sorted(result["restored_ids"])
        assert result["skipped_ids"] == []
        assert result["missing_workspaces"] == []
        assert len(calls) == 1  # recompute runs exactly once, after commit
        statuses = _statuses(db)
        assert statuses[a] == "ready"
        assert statuses[c] == "todo"  # parent restored ready, not done
        for task_id in (a, c):
            unarchived = _payloads(db, task_id, "admin_unarchived")
            assert len(unarchived) == 1
            assert unarchived[0]["archive_group_id"] == group
            assert unarchived[0]["actor"] == ACTOR
        db.close()

    def test_prior(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "prior A")
        make_task_running(db, a)
        result_a = assert_success(*_call(_archive, db, [a], force_running=True))
        group = result_a["archive_group_id"]
        assert GROUP_ID_RE.fullmatch(group)
        payloads = _payloads(db, a, "admin_archived")
        assert payloads[0]["prior_status"] == "running"
        result = assert_success(*_call(_unarchive, db, group_id=group))
        assert result["restored_ids"] == [a]
        assert _statuses(db)[a] == "ready"  # restored ready, not running
        db.close()

    def test_missing(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        ws = Path(board).parent / "workspaces" / "gone"
        ws.mkdir(parents=True)
        (ws / "x.txt").write_text("x\n", encoding="utf-8")
        a = create(db, "ws A", workspace_kind="dir", workspace_path=str(ws))
        result_a = assert_success(*_call(_archive, db, [a]))
        group = result_a["archive_group_id"]
        shutil.rmtree(ws)
        result = assert_success(*_call(_unarchive, db, group_id=group))
        assert result["restored_ids"] == [a]
        assert result["missing_workspaces"] == [str(ws)]
        assert all(isinstance(p, str) for p in result["missing_workspaces"])
        assert _statuses(db)[a] == "ready"
        db.close()

    def test_direct(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "d A")
        b = create(db, "d B")
        m = create(db, "d M")
        x = create(db, "d X")
        # All archiving happens BEFORE the spy is installed so the spy
        # only observes the unarchive calls under test.
        r1 = assert_success(*_call(_archive, db, [a]))
        r2 = assert_success(*_call(_archive, db, [b]))
        # Direct setup transition: b keeps its latest admin_archived event
        # but is brought back to the NON-archived "ready" status, preserving
        # the latest-admin-event-but-currently-nonarchived refusal case used
        # below as the all-or-nothing mixed target.
        set_status(db, b, "ready")
        rx = assert_success(*_call(_archive, db, [x]))
        kb.archive_task(db, m)  # ordinary archive, not admin
        assert _statuses(db)[m] == "archived"
        assert _statuses(db)[b] == "ready"
        # Latest-archive-mismatch: a plain 'archived' event now sits on
        # top of x's admin archive event.
        db.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'archived', NULL, ?)",
            (x, int(time.time())),
        )

        calls = spy_recompute(monkeypatch)

        # Unknown id mixed with a valid one: all-or-nothing.
        result, exc = _call(_unarchive, db, task_ids=[a, "zz_nope"])
        assert_refused(result, exc, "zz_nope")
        if result is not None:
            assert result["restored_ids"] == []
        assert _statuses(db)[a] == "archived"

        # Non-admin-archived target.
        result, exc = _call(_unarchive, db, task_ids=[m])
        assert_refused(result, exc, m)
        if result is not None:
            assert result["restored_ids"] == []
        assert _statuses(db)[m] == "archived"

        # Non-archived target mixed with a valid archived one.
        result, exc = _call(_unarchive, db, task_ids=[a, b])
        assert_refused(result, exc, b)
        if result is not None:
            assert result["restored_ids"] == []
        assert _statuses(db)[a] == "archived"
        assert _statuses(db)[b] == "ready"

        # Latest-archive-mismatch target.
        result, exc = _call(_unarchive, db, task_ids=[x])
        assert_refused(result, exc, x)
        if result is not None:
            assert result["restored_ids"] == []
        assert _statuses(db)[x] == "archived"
        assert calls == []  # refusals never recompute

        # Valid direct-id success: recomputes exactly once after commit.
        result = assert_success(*_call(_unarchive, db, task_ids=[a]))
        assert result["restored_ids"] == [a]
        assert _statuses(db)[a] == "ready"
        assert len(calls) == 1
        db.close()

    def test_supersede(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "sup A")
        b = create(db, "sup B")
        r1 = assert_success(*_call(_archive, db, [a, b]))
        g1 = r1["archive_group_id"]
        restored = assert_success(*_call(_unarchive, db, group_id=g1))
        assert restored["restored_ids"] == sorted([a, b])
        r2 = assert_success(*_call(_archive, db, [b]))
        g2 = r2["archive_group_id"]
        assert g1 != g2

        # b's latest admin archive is g2; a is no longer archived. The group
        # g1 unarchive must report BOTH the already-unarchived member (a)
        # and the member superseded by a newer archive (b) in sorted order.
        result = assert_success(*_call(_unarchive, db, group_id=g1))
        assert result["restored_ids"] == []
        assert result["skipped_ids"] == sorted([a, b])
        statuses = _statuses(db)
        assert statuses[a] == "ready"
        assert statuses[b] == "archived"

        # The newer group still restores b.
        result2 = assert_success(*_call(_unarchive, db, group_id=g2))
        assert result2["restored_ids"] == [b]
        assert _statuses(db)[b] == "ready"
        db.close()

    def test_unk(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "ug A")
        r = assert_success(*_call(_archive, db, [a]))
        group = r["archive_group_id"]
        calls = spy_recompute(monkeypatch)
        ghost_group = "ag_" + "f" * 16
        result, exc = _call(_unarchive, db, group_id=ghost_group)
        assert_refused(result, exc, ghost_group)
        if result is not None:
            assert result["restored_ids"] == []
        assert _statuses(db)[a] == "archived"
        assert calls == []
        db.close()

    def test_modes(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "mm A")
        r = assert_success(*_call(_archive, db, [a]))
        group = r["archive_group_id"]
        # Both modes at once.
        with pytest.raises((ValueError, TypeError)):
            _kernel("admin_unarchive", db, task_ids=[a], group_id=group,
                    actor=ACTOR)
        # Empty direct-id mode.
        with pytest.raises((ValueError, TypeError)):
            _kernel("admin_unarchive", db, task_ids=[], actor=ACTOR)
        # Whitespace-only actor: group and direct modes.
        with pytest.raises((ValueError, TypeError)):
            _unarchive(db, group_id=group, actor="   ")
        with pytest.raises((ValueError, TypeError)):
            _kernel("admin_unarchive", db, task_ids=[a], actor="  ")
        # No mode at all / no actor at all.
        with pytest.raises((ValueError, TypeError)):
            _kernel("admin_unarchive", db, task_ids=[a])
        assert _statuses(db)[a] == "archived"
        db.close()

    def test_malformed(self, board, tmp_path, monkeypatch):
        db = make_conn(board)
        a = create(db, "mal A")
        c = create(db, "mal C", parents=[a])
        r = assert_success(*_call(_archive, db, [a]))
        group = r["archive_group_id"]
        assert set(r["archived_ids"]) == {a, c}
        # Corrupt a's latest admin_archived payload to invalid JSON.
        db.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'admin_archived' "
            "AND id = (SELECT MAX(id) FROM task_events "
            "          WHERE task_id = ? AND kind = 'admin_archived')",
            ("not-valid-json{{", a, a),
        )
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_unarchive, db, task_ids=[a])
        assert_failed(result, exc)
        if result is not None:
            assert result["restored_ids"] == []
        assert _statuses(db)[a] == "archived"
        assert _statuses(db)[c] == "archived"
        assert len(_events_by_kind(db, a, "admin_unarchived")) == 0
        assert len(_events_by_kind(db, c, "admin_unarchived")) == 0
        assert calls == []
        # Now a decodeable-but-incomplete payload (missing archive_group_id).
        db.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'admin_archived' "
            "AND id = (SELECT MAX(id) FROM task_events "
            "          WHERE task_id = ? AND kind = 'admin_archived')",
            (json.dumps({"prior_status": "todo"}), a, a),
        )
        result2, exc2 = _call(_unarchive, db, task_ids=[a])
        assert_failed(result2, exc2)
        if result2 is not None:
            assert result2["restored_ids"] == []
        assert _statuses(db)[a] == "archived"
        assert _statuses(db)[c] == "archived"
        assert calls == []
        db.close()

    def test_rb_sql(self, board, tmp_path, monkeypatch):
        db_path = new_board(tmp_path, monkeypatch, "unrollback")
        db = make_conn(db_path)
        a = create(db, "rb A")
        c = create(db, "rb C", parents=[a])
        r = assert_success(*_call(_archive, db, [a]))
        group = r["archive_group_id"]
        statuses_before = _statuses(db)
        events_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_events"
        ).fetchone()["n"]
        comments_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_comments"
        ).fetchone()["n"]
        db.close()
        failing = make_fail_write_conn(db_path, fail_on_write=2)
        calls = spy_recompute(monkeypatch)
        result, exc = _call(_unarchive, failing, group_id=group)
        assert_failed(result, exc)
        probe = make_conn(db_path)
        assert _statuses(probe) == statuses_before
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_events")
            .fetchone()["n"] == events_before
        )
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_comments")
            .fetchone()["n"] == comments_before
        )
        assert len(_events_by_kind(probe, a, "admin_unarchived")) == 0
        assert len(_events_by_kind(probe, c, "admin_unarchived")) == 0
        assert calls == []  # recompute never runs on a rolled-back unarchive
        probe.close()

    def test_rbev(self, board, tmp_path, monkeypatch):
        db_path = new_board(tmp_path, monkeypatch, "unrollback-ev")
        db = make_conn(db_path)
        a = create(db, "rb2 A")
        c = create(db, "rb2 C", parents=[a])
        r = assert_success(*_call(_archive, db, [a]))
        group = r["archive_group_id"]
        statuses_before = _statuses(db)
        events_before = db.execute(
            "SELECT COUNT(*) AS n FROM task_events"
        ).fetchone()["n"]
        db.close()
        db = make_conn(db_path)
        calls = spy_recompute(monkeypatch)
        orig = kb._append_event

        def boom(conn, task_id, kind, payload=None, **kwargs):
            if kind == "admin_unarchived":
                raise RuntimeError("injected admin_unarchived failure")
            return orig(conn, task_id, kind, payload=payload, **kwargs)

        monkeypatch.setattr(kb, "_append_event", boom)
        result, exc = _call(_unarchive, db, group_id=group)
        assert_failed(result, exc)
        probe = make_conn(db_path)
        assert _statuses(probe) == statuses_before
        assert (
            probe.execute("SELECT COUNT(*) AS n FROM task_events")
            .fetchone()["n"] == events_before
        )
        assert len(_events_by_kind(probe, a, "admin_unarchived")) == 0
        assert len(_events_by_kind(probe, c, "admin_unarchived")) == 0
        assert calls == []
        db.close()
        probe.close()
