"""The consumer-side CONTENT skew gate on the dispatcher tick.

Ruling ``t_8fed34c8`` (platform-stl, Design Authority, 2026-10-10), R2 + R3.

**R2 -- the signal must be CONTENT, not the ref.** ``gateway.code_skew.detect_code_skew()``
fingerprints only ``.git/HEAD`` -> ref -> sha and never reads the working tree. The measured
incident (gateway pid 56908, still serving at 17:23 on 2026-10-10) moved the WORKING TREE with
HEAD unchanged -- an interrupted merge leaves HEAD at the pre-merge commit and the recovery
restores paths "to HEAD" -- so the ref-only signal returned ``None`` while 120 paths differed
from HEAD and every dispatcher spawn died ``cannot import name 'ADVISORY_SKILLS_ENV'``. The test
that proves R2 is therefore: **a working-tree edit with HEAD unchanged must be DETECTED**, and it
must be detected by the gate that ships, not by a probe the gate does not make.

**R3 -- the consumer is the tick.** On proven skew the tick spawns nothing, records a NON-GREEN
status on a surface a human reads, and files EXACTLY ONE acting card to ``default`` (the ops
seat), idempotency-keyed ``<tree-fingerprint>:<boot-pid>:<window-start>``. The check is O(tick),
never O(cards): no per-card import walk on the happy path.
"""

from __future__ import annotations

import json
import os

import pytest

from gateway import code_skew
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import tree_fingerprint as tf


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_boot_state(monkeypatch):
    """Every test starts with NO boot record, an empty memo, and an inert ref half.

    The boot record is process-global by design (one record, recorded once), so a test must never
    inherit another test's -- and the ref half is disarmed unless a test arms it, so a content-only
    finding cannot be mistaken for a ref finding.
    """
    tf.reset_boot()
    tf._content_memo.clear()
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    yield
    tf.reset_boot()
    tf._content_memo.clear()


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A sandbox board: the status record must land in the sandbox home, never the live one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A small fake source tree that ``hermes_cli.tree_fingerprint`` is pointed at."""
    root = tmp_path / "checkout"
    for package, module in (
        ("hermes_cli", "kanban_db.py"),
        ("agent", "skill_commands.py"),
        ("gateway", "run.py"),
        ("tools", "kanban_tools.py"),
    ):
        directory = root / package
        directory.mkdir(parents=True)
        (directory / module).write_text(f"# {package}/{module}\nVALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(tf, "default_root", lambda: root)
    return root


def _edit(root, package="hermes_cli", module="kanban_db.py", body="VALUE = 2\n"):
    """Change the WORKING TREE only -- no ref moves, exactly the incident's shape."""
    (root / package / module).write_text(f"# {package}/{module}\n{body}", encoding="utf-8")


def _arm_ref(monkeypatch, *, boot: str, disk: str) -> None:
    """Point the ref half of the signal at fixed revisions (the fast path, not the decision)."""
    monkeypatch.setattr(code_skew, "_boot_fingerprint", f"git:refs/heads/main:{boot}")
    monkeypatch.setattr(code_skew, "_fingerprint", lambda: f"git:refs/heads/main:{disk}")


def _actor_rows(conn):
    return conn.execute(
        "SELECT id, assignee, idempotency_key FROM tasks WHERE idempotency_key LIKE 'tree-skew:%'"
    ).fetchall()


def _status_record():
    path = kbd.tick_yield_status_path()
    return path, (json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)


# ---------------------------------------------------------------------------------------------
# (a) the load-bearing one: a working-tree edit with HEAD unchanged
# ---------------------------------------------------------------------------------------------


def test_a_working_tree_edit_with_head_unchanged_is_detected(board, tree, monkeypatch):
    """R2's acceptance test. A ref-only guard fails this by construction; the shipped gate passes.

    Proves both halves in one place: the ref-only probe the incident defeated reports NO skew,
    and the tick still refuses.
    """
    conn = board
    tf.record_boot()
    _edit(tree)
    # HEAD did not move (the interrupted merge / restore-to-HEAD shape), so the fast path is blind.
    _arm_ref(monkeypatch, boot="a" * 40, disk="a" * 40)
    assert code_skew.detect_code_skew() is None, "the ref-only signal must be blind here"
    assert tf.detect_skew() is not None, "the content signal must see the working-tree edit"

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    assert result.tick_yielded.startswith(kbd.TICK_YIELD_PREFIX)
    assert result.spawned == [], "a refused tick spawns nothing"
    assert result.promoted == 0 and result.reclaimed == 0, "a refused tick mutates no board state"


# ---------------------------------------------------------------------------------------------
# (b) HEAD-only drift is still caught (the ref half rides along as a fast path)
# ---------------------------------------------------------------------------------------------


def test_head_only_drift_is_detected(board, tree, monkeypatch):
    conn = board
    tf.record_boot()
    # The files are byte-identical; only the revision moved (a checkout of an identical tree).
    _arm_ref(monkeypatch, boot="a" * 40, disk="b" * 40)
    assert tf.detect_skew() is None, "content alone cannot see a pure ref move"

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    assert result.tick_yielded.startswith(kbd.TICK_YIELD_PREFIX)
    assert "b" * 10 in result.tick_yielded, result.tick_yielded


# ---------------------------------------------------------------------------------------------
# (c) no false positive on a clean tree
# ---------------------------------------------------------------------------------------------


def test_a_clean_tree_is_not_refused(board, tree):
    conn = board
    tf.record_boot()

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    assert result.tick_yielded == ""
    assert _actor_rows(conn) == []
    path, record = _status_record()
    assert record is None, f"a healthy tick must leave no non-green marker ({path})"


def test_a_process_that_never_recorded_a_boot_fingerprint_fails_open(board, tree):
    """A one-shot CLI dispatch imports fresh from disk: it cannot be stale, so it never refuses."""
    conn = board
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert result.tick_yielded == ""


def test_an_unreadable_tree_fails_open(board, tree, monkeypatch):
    """An IO error is 'cannot certify identity', never proof of skew."""
    conn = board
    tf.record_boot()
    monkeypatch.setattr(tf, "content_fingerprint", lambda root=None: None)
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert result.tick_yielded == ""


def test_a_byte_identical_rewrite_reads_clean(board, tree):
    """The contract is CONTENT, not mtime: re-writing the same bytes (a `git checkout`, a re-applied
    override, the restore after a break) is not drift and must not refuse a healthy tick."""
    conn = board
    tf.record_boot()
    boot_fingerprint = tf.boot_record()["fingerprint"]

    _edit(tree, body="VALUE = 1\n")  # byte-identical to what the fixture wrote

    assert tf.content_fingerprint() == boot_fingerprint
    assert tf.detect_skew() is None
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert result.tick_yielded == ""


# ---------------------------------------------------------------------------------------------
# (d) exactly ONE acting card, idempotent within the window
# ---------------------------------------------------------------------------------------------


def test_n_ticks_on_a_skewed_tree_file_exactly_one_actor_card(board, tree):
    conn = board
    tf.record_boot()
    _edit(tree)

    for _ in range(5):
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
        assert result.tick_yielded.startswith(kbd.TICK_YIELD_PREFIX)

    rows = _actor_rows(conn)
    assert len(rows) == 1, f"one skewed process must resolve to ONE card, got {len(rows)}"
    assert rows[0]["assignee"] == kbd.TICK_YIELD_ACTOR_ASSIGNEE == "default"
    assert rows[0]["idempotency_key"].startswith(f"{kbd.TICK_YIELD_ACTOR_IDEMPOTENCY_PREFIX}")


def test_the_actor_key_is_the_fingerprint_the_pid_and_the_window(board, tree):
    conn = board
    tf.record_boot()
    _edit(tree)

    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    key = _actor_rows(conn)[0]["idempotency_key"]

    boot_record = tf.boot_record()
    assert boot_record is not None
    window = int(key.rsplit(":", 1)[-1])
    assert key == (
        f"{kbd.TICK_YIELD_ACTOR_IDEMPOTENCY_PREFIX}{boot_record['fingerprint']}"
        f":{boot_record['pid']}:{window}"
    )
    assert window % kbd.TICK_YIELD_WINDOW_SECONDS == 0


def test_a_different_frozen_process_may_file_its_own_card(board, tree, monkeypatch):
    """The window buckets TICKS, not processes: the pid half is what keeps one card per process."""
    conn = board
    tf.record_boot()
    _edit(tree)
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    boot_record = tf.boot_record()
    assert boot_record is not None
    monkeypatch.setattr(tf, "_boot_record", {**boot_record, "pid": int(boot_record["pid"]) + 1})
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    assert len(_actor_rows(conn)) == 2


# ---------------------------------------------------------------------------------------------
# the non-green status surface
# ---------------------------------------------------------------------------------------------


def test_a_refused_tick_records_a_non_green_status_a_human_reads(board, tree, caplog):
    conn = board
    tf.record_boot()
    _edit(tree)

    with caplog.at_level("ERROR", logger="hermes_cli.kanban_db_dispatch"):
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    path, status_record = _status_record()
    assert status_record is not None, "a refusing tick must leave a record a human reads"
    assert status_record["yielded"] is True
    assert status_record["boot_fingerprint"] != status_record["disk_fingerprint"]
    boot_record = tf.boot_record()
    assert boot_record is not None
    assert status_record["pid"] == boot_record["pid"]
    assert status_record["reason"] == result.tick_yielded
    # Loud, not silent: the status record's path is in the same line.
    messages = [r.getMessage() for r in caplog.records]
    assert any(kbd.TICK_YIELD_PREFIX in message for message in messages)
    assert any(str(path) in message for message in messages)


def test_the_healthy_tick_retires_its_own_non_green_marker(board, tree):
    """Once the process serves normally again the marker goes -- nobody has to remember to clear it."""
    conn = board
    tf.record_boot()
    _edit(tree)
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    _path, record = _status_record()
    assert record is not None

    _edit(tree, body="VALUE = 1\n" + "#" * 40)  # still skewed: the marker must SURVIVE this
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert _status_record()[1] is not None

    # Rewind to the recorded content (the reload door's effect, without a bounce): the marker clears.
    tf.reset_boot()
    tf.record_boot()
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert _status_record()[1] is None


def test_a_foreign_marker_is_never_cleared_by_a_healthy_process(board, tree):
    """Clearing a LIVE gateway's marker from another process would read green over a frozen one.

    The upstream form of this change retires a foreign marker only when its recorder is PROVEN
    dead; that clause needs the liveness-witness substrate, so it rides that card (see the live
    tree). What this branch guarantees is the safety half: another process's refusal is never
    silently cleared.
    """
    conn = board
    tf.record_boot()
    _edit(tree)
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    path, status_record = _status_record()
    assert status_record is not None
    # The marker now names a FOREIGN process (as a sibling gateway's would).
    path.write_text(json.dumps({**status_record, "pid": os.getpid() + 1}), encoding="utf-8")

    kbd._clear_tick_yield_status()

    assert path.exists(), "a foreign process's refusal must not be cleared by a sibling"

    # This process's OWN marker does retire on a healthy tick.
    path.write_text(json.dumps({**status_record, "pid": os.getpid()}), encoding="utf-8")
    kbd._clear_tick_yield_status()
    assert not path.exists()


# ---------------------------------------------------------------------------------------------
# (e) cost: O(tick), never O(cards)
# ---------------------------------------------------------------------------------------------


def test_the_skew_check_runs_once_per_tick_not_per_card(
    board, tree, monkeypatch, all_assignees_spawnable,
):
    conn = board
    tf.record_boot()
    calls: list = []
    real = tf.content_fingerprint

    def _counting(root=None):
        calls.append(root)
        return real(root)

    monkeypatch.setattr(tf, "content_fingerprint", _counting)

    # Empty board: one tick, one check.
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)
    assert len(calls) == 1, f"one tick must make one check, made {len(calls)}"

    # A board with several spawnable cards: STILL one check -- the check is per tick, not per card.
    for index in range(4):
        kb.create_task(conn, title=f"card {index}", assignee="alice")
    calls.clear()
    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 4242)

    assert len(result.spawned) == 4, "the clean-tick control must still spawn the cards"
    assert len(calls) == 1, (
        f"the check must not run per card: {len(calls)} checks for {len(result.spawned)} cards"
    )


def test_the_steady_state_check_does_not_reread_file_bytes(board, tree, monkeypatch):
    """The memo keeps a per-tick check cheap: a stat walk, not 32MB of hashing, every tick."""
    tf.record_boot()
    tf._content_memo.clear()  # the boot snapshot warmed the memo; this test measures the ticks
    reads: list = []
    real_digest = tf._digest_files

    def _counting(files, root):
        reads.append(len(files))
        return real_digest(files, root)

    monkeypatch.setattr(tf, "_digest_files", _counting)
    tf.content_fingerprint()
    tf.content_fingerprint()
    tf.content_fingerprint()

    assert len(reads) == 1, f"an unchanged tree must be digested once, not {len(reads)} times"
    _edit(tree)
    tf.content_fingerprint()
    assert len(reads) == 2, "a moved file must force exactly one re-digest"
