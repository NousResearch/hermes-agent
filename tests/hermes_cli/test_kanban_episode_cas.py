"""Governed WAIT_AND_RESUME episodes: exact-episode park/unblock CAS.

``schedule_task_governed`` mints an opaque per-episode token inside the park's
write txn; ``unblock_task_governed`` resumes only while that park is still the
task's current wait episode. Legacy ``schedule_task``/``unblock_task`` keep
their exact semantics and mint nothing.
"""

from __future__ import annotations

import ast
import json
import re
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_episode as kbe

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_RE = re.compile(r"^hkep1\.(?P<task>\S+)\.(?P<nonce>[0-9a-f]{32})$")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


def _ready(conn, title="t", **kw) -> str:
    tid = kb.create_task(conn, title=title, assignee="worker", **kw)
    assert kb.get_task(conn, tid).status in ("ready", "todo")
    return tid


def _running(conn) -> tuple[str, int]:
    tid = _ready(conn)
    claimed = kb.claim_task(conn, tid, claimer="testhost:1")
    assert claimed is not None and claimed.current_run_id
    return tid, int(claimed.current_run_id)


def _snapshot(conn, tid):
    """Everything a lifecycle mutation could touch for ``tid``."""
    return (
        tuple(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()),
        [tuple(r) for r in conn.execute("SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,))],
        [tuple(r) for r in conn.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,))],
    )


def _last_event(conn, tid):
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    return row["kind"], json.loads(row["payload"]) if row["payload"] else None


def _conflict(code, fn, *a, **kw):
    with pytest.raises(kbe.EpisodeConflict) as ei:
        fn(*a, **kw)
    assert ei.value.code == code, (ei.value.code, str(ei.value))
    return ei.value


# ---------------------------------------------------------------------------
# R1 — token minted atomically with the park, returned only after commit
# ---------------------------------------------------------------------------

def test_pre_launch_park_mints_token_in_park_event(conn):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH", reason="capacity")
    m = TOKEN_RE.match(token)
    assert m and m["task"] == tid
    assert kb.get_task(conn, tid).status == "scheduled"
    kind, payload = _last_event(conn, tid)
    assert kind == "scheduled"
    assert payload == {"reason": "capacity", "episode_token": token, "mode": "PRE_LAUNCH"}


def test_token_format_and_nonce_uniqueness(conn):
    tokens = {kbe._mint_episode_token("t_abcd1234") for _ in range(2000)}
    assert len(tokens) == 2000
    for tok in tokens:
        m = TOKEN_RE.match(tok)
        assert m and m["task"] == "t_abcd1234" and len(m["nonce"]) == 32  # 128 bits
    a, b = _ready(conn, "a"), _ready(conn, "b")
    assert kbe.schedule_task_governed(conn, a, mode="PRE_LAUNCH") != kbe.schedule_task_governed(
        conn, b, mode="PRE_LAUNCH")


def test_forced_commit_failure_returns_no_token_and_persists_nothing(conn, monkeypatch):
    tid = _ready(conn)
    before = _snapshot(conn, tid)
    real = kbc._execute_boundary_with_retry

    def failing(c, stmt):
        if stmt == "COMMIT":
            raise sqlite3.OperationalError("disk I/O error (forced)")
        return real(c, stmt)

    monkeypatch.setattr(kbc, "_execute_boundary_with_retry", failing)
    with pytest.raises(sqlite3.OperationalError):
        kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    monkeypatch.setattr(kbc, "_execute_boundary_with_retry", real)

    other = kbc.connect()
    try:
        assert _snapshot(other, tid) == before
        assert not other.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND payload LIKE '%episode_token%'", (tid,),
        ).fetchone()
    finally:
        other.close()


def test_legacy_schedule_mints_no_token_and_keeps_payload(conn):
    tid = _ready(conn)
    assert kb.schedule_task(conn, tid, reason="later") is True
    assert _last_event(conn, tid) == ("scheduled", {"reason": "later"})


# ---------------------------------------------------------------------------
# R2 — exact source / run binding
# ---------------------------------------------------------------------------

def test_mid_run_park_requires_exact_run(conn):
    tid, run_id = _running(conn)
    before = _snapshot(conn, tid)
    err = _conflict("RUN_MISMATCH", kbe.schedule_task_governed, conn, tid, mode="MID_RUN",
                    expected_run_id=run_id + 1000)
    assert err.current_status == "running"
    assert _snapshot(conn, tid) == before

    token = kbe.schedule_task_governed(conn, tid, mode="MID_RUN", expected_run_id=run_id)
    assert TOKEN_RE.match(token)
    task = kb.get_task(conn, tid)
    assert task.status == "scheduled" and task.current_run_id is None
    run = conn.execute("SELECT outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["outcome"] == "scheduled" and run["ended_at"] is not None


def test_mid_run_park_ignores_leaked_run_id_on_a_waiting_task(conn):
    tid, run_id = _running(conn)
    assert kb.block_task(conn, tid, reason="human")
    with kb.write_txn(conn):  # leaked pointer, the case _reclaim_dangling_run exists for
        conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, tid))
    before = _snapshot(conn, tid)
    _conflict("SOURCE_IS_WAIT", kbe.schedule_task_governed, conn, tid, mode="MID_RUN",
              expected_run_id=run_id)
    assert _snapshot(conn, tid) == before


def test_mid_run_on_non_running_task_is_status_not_allowed(conn):
    tid = _ready(conn)
    _conflict("STATUS_NOT_ALLOWED", kbe.schedule_task_governed, conn, tid, mode="MID_RUN", expected_run_id=1)


def test_pre_launch_refuses_claimed_or_running_task(conn):
    tid, _run = _running(conn)  # the dispatcher won the race
    before = _snapshot(conn, tid)
    _conflict("NOT_IDLE", kbe.schedule_task_governed, conn, tid, mode="PRE_LAUNCH")
    assert _snapshot(conn, tid) == before

    other = _ready(conn, "claimed-but-ready")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock = 'h:1' WHERE id = ?", (other,))
    _conflict("NOT_IDLE", kbe.schedule_task_governed, conn, other, mode="PRE_LAUNCH")


@pytest.mark.parametrize("park", ["blocked", "scheduled", "triage"])
@pytest.mark.parametrize("mode", ["PRE_LAUNCH", "MID_RUN"])
def test_source_already_waiting_is_rejected(conn, park, mode):
    tid = _ready(conn)
    if park == "blocked":
        assert kb.block_task(conn, tid, reason="human")
    elif park == "scheduled":
        assert kb.schedule_task(conn, tid)
    else:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
    before = _snapshot(conn, tid)
    kw = {"expected_run_id": 1} if mode == "MID_RUN" else {}
    err = _conflict("SOURCE_IS_WAIT", kbe.schedule_task_governed, conn, tid, mode=mode, **kw)
    assert err.current_status == park
    assert _snapshot(conn, tid) == before


def test_pre_launch_on_terminal_task_and_unknown_task(conn):
    tid = _ready(conn)
    assert kb.complete_task(conn, tid, result="done", force=True)
    _conflict("STATUS_NOT_ALLOWED", kbe.schedule_task_governed, conn, tid, mode="PRE_LAUNCH")
    _conflict("TASK_NOT_FOUND", kbe.schedule_task_governed, conn, "t_nope0000", mode="PRE_LAUNCH")


@pytest.mark.parametrize("kw", [
    {"mode": "MID_RUN"}, {"mode": "PRE_LAUNCH", "expected_run_id": 3}, {"mode": "WHENEVER"},
])
def test_invalid_mode_arguments_raise_value_error_without_writing(conn, kw):
    tid = _ready(conn)
    before = _snapshot(conn, tid)
    with pytest.raises(ValueError) as ei:
        kbe.schedule_task_governed(conn, tid, **kw)
    assert not isinstance(ei.value, kbe.EpisodeConflict)
    assert _snapshot(conn, tid) == before


# ---------------------------------------------------------------------------
# R3 — exact-episode unblock CAS
# ---------------------------------------------------------------------------

def test_governed_unblock_success_lands_ready(conn):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    assert kbe.unblock_task_governed(conn, tid, expected_episode_token=token) == "ready"
    assert kb.get_task(conn, tid).status == "ready"
    assert _last_event(conn, tid) == ("unblocked", None)


def test_same_token_cannot_be_used_twice(conn):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    kbe.unblock_task_governed(conn, tid, expected_episode_token=token)
    before = _snapshot(conn, tid)
    err = _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=token)
    assert err.current_status == "ready"
    assert _snapshot(conn, tid) == before


def test_stale_e1_cannot_unblock_e2_after_governed_repark(conn):
    tid = _ready(conn)
    e1 = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    assert kb.unblock_task(conn, tid)  # human/legacy unblock ends E1
    e2 = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    before = _snapshot(conn, tid)
    err = _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=e1)
    assert err.current_status == "scheduled"  # same task, same status: only the episode differs
    assert _snapshot(conn, tid) == before
    assert kbe.unblock_task_governed(conn, tid, expected_episode_token=e2) == "ready"


def test_stale_e1_cannot_unblock_after_human_repark_without_token(conn):
    tid = _ready(conn)
    e1 = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    assert kb.unblock_task(conn, tid)
    assert kb.schedule_task(conn, tid, reason="human re-park")  # legacy E2, no token
    before = _snapshot(conn, tid)
    _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=e1)
    assert _snapshot(conn, tid) == before


def test_stale_after_blocked_to_scheduled_human_transition(conn):
    tid = _ready(conn)
    e1 = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    assert kb.unblock_task(conn, tid)
    assert kb.block_task(conn, tid, reason="human")
    assert kb.schedule_task(conn, tid)  # legacy blocked -> scheduled re-park
    _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=e1)


def test_unblock_of_task_not_waiting_is_stale(conn):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    with kb.write_txn(conn):  # direct status write without any event
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
    _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=token)


@pytest.mark.parametrize("bad", [
    "", "garbage", "hkep2.t_x.0123456789abcdef0123456789abcdef",
    "hkep1.t_x.0123456789ABCDEF0123456789ABCDEF", "hkep1.t_x.0123", "hkep1..0123456789abcdef0123456789abcdef",
    "hkep1.t x.0123456789abcdef0123456789abcdef",
])
def test_malformed_token(conn, bad):
    tid = _ready(conn)
    kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    before = _snapshot(conn, tid)
    _conflict("TOKEN_MALFORMED", kbe.unblock_task_governed, conn, tid, expected_episode_token=bad)
    assert _snapshot(conn, tid) == before


def test_foreign_token(conn):
    a, b = _ready(conn, "a"), _ready(conn, "b")
    ta = kbe.schedule_task_governed(conn, a, mode="PRE_LAUNCH")
    kbe.schedule_task_governed(conn, b, mode="PRE_LAUNCH")
    before = _snapshot(conn, b)
    _conflict("TOKEN_FOREIGN", kbe.unblock_task_governed, conn, b, expected_episode_token=ta)
    assert _snapshot(conn, b) == before


def test_well_formed_token_never_issued_is_stale(conn):
    tid = _ready(conn)
    kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    forged = kbe._mint_episode_token(tid)  # right shape, unknown nonce
    _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=forged)


def test_unknown_task(conn):
    _conflict("TASK_NOT_FOUND", kbe.unblock_task_governed, conn, "t_nope0000",
              expected_episode_token=kbe._mint_episode_token("t_nope0000"))


def test_neutral_events_do_not_stale_the_episode(conn, tmp_path):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    kb.add_comment(conn, tid, "human", "still waiting on capacity")
    assert kb.edit_task(conn, tid, title="renamed", body="new body")
    assert kb.edit_task(conn, tid, priority=5)
    blob = tmp_path / "note.txt"
    blob.write_text("x", encoding="utf-8")
    att = kb.add_attachment(conn, tid, filename="note.txt", stored_path=str(blob), size=1)
    kb.delete_attachment(conn, att)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "terminal_worker_reaped", {"pid": 1})
    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))}
    assert {"commented", "edited", "reprioritized", "attached", "attachment_removed",
            "terminal_worker_reaped"} <= kinds
    assert kbe.unblock_task_governed(conn, tid, expected_episode_token=token) == "ready"


@pytest.mark.parametrize("boundary", ["assigned", "linked", "promoted", "status", "a_future_kind_nobody_classified"])
def test_any_non_neutral_event_stales_fail_closed(conn, boundary):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    with kb.write_txn(conn):
        kb._append_event(conn, tid, boundary, None)
    assert kb.get_task(conn, tid).status == "scheduled"  # status alone would still match
    before = _snapshot(conn, tid)
    _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=token)
    assert _snapshot(conn, tid) == before


def test_real_reassign_during_wait_stales(conn):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    assert kb.assign_task(conn, tid, "other-profile")
    _conflict("EPISODE_STALE", kbe.unblock_task_governed, conn, tid, expected_episode_token=token)


def test_concurrent_same_token_unblock_exactly_one_succeeds(kanban_home):
    with kbc.connect() as c:
        tid = _ready(c)
        token = kbe.schedule_task_governed(c, tid, mode="PRE_LAUNCH")
    barrier = threading.Barrier(8)
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        c = kbc.connect()
        try:
            barrier.wait()
            try:
                out = "ok:" + kbe.unblock_task_governed(c, tid, expected_episode_token=token)
            except kbe.EpisodeConflict as e:
                out = e.code
            with lock:
                results.append(out)
        finally:
            c.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(results) == ["EPISODE_STALE"] * 7 + ["ok:ready"]
    with kbc.connect() as c:
        n = c.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'unblocked'",
                      (tid,)).fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# Successful unblock keeps upstream semantics (shared _unblock_locked body)
# ---------------------------------------------------------------------------

def _unblock_outcome(conn, tid):
    row = conn.execute(
        "SELECT status, current_run_id, consecutive_failures, last_failure_error, block_kind, "
        "block_recurrences FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    return tuple(row), _last_event(conn, tid)


@pytest.mark.parametrize("parent_done", [True, False])
def test_governed_and_legacy_unblock_are_equivalent(conn, parent_done):
    outcomes = []
    for governed in (False, True):
        parent = _ready(conn, "parent")
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        if parent_done:
            assert kb.complete_task(conn, parent, result="ok", force=True)
        with kb.write_txn(conn):  # make the child parkable PRE_LAUNCH and carry counters
            conn.execute(
                "UPDATE tasks SET status = 'ready', consecutive_failures = 2, last_failure_error = 'boom', "
                "block_kind = 'capability', block_recurrences = 1 WHERE id = ?", (child,))
        if governed:
            token = kbe.schedule_task_governed(conn, child, mode="PRE_LAUNCH")
            kbe.unblock_task_governed(conn, child, expected_episode_token=token)
        else:
            assert kb.schedule_task(conn, child)
            assert kb.unblock_task(conn, child)
        outcomes.append(_unblock_outcome(conn, child))
    assert outcomes[0] == outcomes[1]
    (status, run, failures, error, kind, rec), _event = outcomes[1]
    assert status == ("ready" if parent_done else "todo")  # parent re-gate
    assert (run, failures, error) == (None, 0, None)
    assert (kind, rec) == ("capability", 1)  # recurrence counters survive unblock


def test_mid_run_review_phase_resumes_ready_like_legacy_scheduled(conn):
    """``scheduled`` never resumes into ``review`` upstream; governed keeps that."""
    tid, run_id = _running(conn)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "changes_requested", {"resume_status": "review"}, run_id=run_id)
    token = kbe.schedule_task_governed(conn, tid, mode="MID_RUN", expected_run_id=run_id)
    assert kbe.unblock_task_governed(conn, tid, expected_episode_token=token) == "ready"


def test_governed_unblock_recovers_dangling_run(conn):
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    with kb.write_txn(conn):  # leaked open run, no event
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', 1)", (tid,))
        conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (cur.lastrowid, tid))
        leaked = cur.lastrowid
    assert kbe.unblock_task_governed(conn, tid, expected_episode_token=token) == "ready"
    run = conn.execute("SELECT outcome, ended_at FROM task_runs WHERE id = ?", (leaked,)).fetchone()
    assert run["outcome"] == "reclaimed" and run["ended_at"] is not None
    assert kb.get_task(conn, tid).current_run_id is None


def test_governed_calls_change_no_schema(conn):
    before = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
    tid = _ready(conn)
    token = kbe.schedule_task_governed(conn, tid, mode="PRE_LAUNCH")
    kbe.unblock_task_governed(conn, tid, expected_episode_token=token)
    after = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
    assert [tuple(r) for r in before] == [tuple(r) for r in after]


# ---------------------------------------------------------------------------
# Classification guard: every emitted event kind is neutral or a boundary
# ---------------------------------------------------------------------------

# Kinds that END a wait episode. A new kind must be added to exactly one of
# this set or ``kbe.EPISODE_NEUTRAL_EVENT_KINDS`` — anything unlisted fails here,
# and at runtime it is treated as a boundary (fail-closed).
EPISODE_BOUNDARY_EVENT_KINDS = frozenset({
    "archive_worker_termination", "archived", "assigned", "blocked", "block_loop_detected",
    "changes_requested", "claim_extended", "claim_rejected", "claimed", "completed",
    "completion_blocked_empty_result", "completion_blocked_hallucination", "crashed", "created",
    "decomposed", "dependency_wait", "descendant_invalidated", "gave_up", "heartbeat", "imported",
    "linked", "model_override_set", "pr_acceptance", "promoted", "promoted_manual", "rate_limited",
    "reasoning_effort_set", "reclaim_deferred", "reclaimed", "reconciled", "respawn_guarded",
    "review_reopened", "review_requested", "scheduled", "skipped_nonspawnable", "spawn_failed",
    "spawned", "specified", "stale", "status", "suspected_hallucinated_references", "timed_out",
    "tip_scratch_workspace", "unblocked", "unlinked", "worker_registered",
    "workspace_cleanup_deferred_shared",
})

# Call sites whose event kind is computed; their possible values are listed in
# the boundary set. A new dynamic site must be reviewed and added here.
KNOWN_DYNAMIC_EVENT_SITES = frozenset({
    ("hermes_cli/kanban_db.py", "_set_task_override"),
    ("hermes_cli/kanban_db.py", "block_task"),
    ("hermes_cli/kanban_db_dispatch.py", "_record_task_failure"),
    ("hermes_cli/kanban_db_dispatch.py", "_reclaim_dead_workers"),
})


def _emitted_event_kinds():
    literal: set[str] = set()
    dynamic: set[tuple[str, str]] = set()
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(("tests/", "tests-js/", "website/", "node_modules/", "venv/", ".venv/")):
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "_append_event" not in src and "INSERT INTO task_events" not in src:
            continue
        tree = ast.parse(src)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name != "_append_event" or len(node.args) < 3:
                    continue
                arg = node.args[2]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literal.add(arg.value)
                else:
                    dynamic.add((rel, fn.name))
        literal.update(re.findall(r"INSERT INTO task_events[^;]*?VALUES \([^)]*?'([a-z_]+)'", src))
    return literal, dynamic


def test_every_emitted_event_kind_is_classified():
    literal, dynamic = _emitted_event_kinds()
    neutral = kbe.EPISODE_NEUTRAL_EVENT_KINDS
    assert not neutral & EPISODE_BOUNDARY_EVENT_KINDS
    unclassified = literal - neutral - EPISODE_BOUNDARY_EVENT_KINDS
    assert not unclassified, f"classify new event kinds as neutral or boundary: {sorted(unclassified)}"
    assert dynamic == KNOWN_DYNAMIC_EVENT_SITES, f"review dynamic event sites: {sorted(dynamic ^ KNOWN_DYNAMIC_EVENT_SITES)}"
    assert neutral <= literal  # no stale names in the allow-list


def test_neutral_allowlist_is_the_minimal_adjudicated_set():
    """Widening the allow-list weakens the fence; it must be a deliberate edit here."""
    assert kbe.EPISODE_NEUTRAL_EVENT_KINDS == frozenset({
        "commented", "edited", "reprioritized", "attached", "attachment_removed",
        "terminal_worker_reaped",
    })
