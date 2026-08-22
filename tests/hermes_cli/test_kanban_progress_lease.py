"""Progress lease vs liveness lease on the Kanban board.

Incident being fixed: dispatcher-spawned workers streamed/reasoned for
thousands of tokens with zero tool calls, zero board transitions and zero
artifacts.  ``AIAgent._touch_activity`` bridged *every* stream chunk / API
wait into ``tasks.last_heartbeat_at`` (and extended ``claim_expires``), so
the claim was renewed indefinitely and no recovery path ever fired.

The fix splits the single lease in two:

* **liveness** (``last_heartbeat_at`` + ``claim_expires``) — "the process and
  the provider are responsive".  Renewed by any agent activity, including raw
  stream tokens.  This is what keeps a slow-but-healthy single API call alive.
* **progress** (``last_progress_at``) — "something observable happened outside
  the model's token stream".  Renewed ONLY by deterministic, semantically
  meaningful events that the model cannot author directly: an attempted /
  in-flight / completed tool invocation (via the centralized tool
  middleware), a durable board state transition, or the claim that started
  the run.  Free text a model writes -- including ``kanban_heartbeat``
  notes -- is commentary and never renews the lease.

These tests pin that invariant at the DB layer, plus the reclaim path that
distinguishes dead / wedged / no-progress workers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# Captured before any fixture stubs it, so the real-process tests below can
# put the genuine /proc reader back without unwinding the rest of the fixture.
_REAL_CAPTURE_WORKER_IDENTITY = kb.capture_worker_identity


def _fake_identity(pid, **overrides):
    """A synthetic Linux birth record for a pid that does not exist.

    Shaped like a dispatcher-spawned worker: its own session and process-group
    leader (``sid == pgid == pid``), which is what ``start_new_session=True``
    produces and what ``verify_worker_ownership`` requires before it will
    authorise a process-GROUP signal.
    """
    ident = {
        "v": kb.WORKER_IDENTITY_VERSION,
        "scheme": "linux_proc",
        "pid": int(pid),
        "starttime": 987654,
        "pgid": int(pid),
        "sid": int(pid),
        "ppid": 1,
        "boot_id": "0000-boot-under-test",
    }
    ident.update(overrides)
    return ident


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB.

    Also stubs ``capture_worker_identity``. Reclaim now refuses to signal any
    pid whose recorded birth identity does not verify against the live
    process, and these tests use synthetic pids that have no ``/proc`` entry —
    without a stub every reclaim would correctly decline, testing the refusal
    path and nothing else. The stub is the *same* function at spawn and at
    verification, so a test that claims a task and then reclaims it exercises
    the real store-then-re-verify round trip; tests that want the refusal
    path override it explicitly.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        kb, "capture_worker_identity", lambda pid: _fake_identity(pid),
    )
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _running_task(conn, *, title="t", assignee="a", pid=4242):
    """Create + claim a task on this host, with a worker pid recorded."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid, claimer=f"{host}:worker")
    kb._set_worker_pid(conn, tid, pid)
    return tid


def _mock_worker(monkeypatch, *, alive=True, dies_on_signal=True):
    """Fake host-local worker process. Returns the list of (pid, signal) the
    reclaim path sent, so tests can assert termination actually happened."""
    state = {"alive": alive}
    sent: list = []

    def _pid_alive(_pid):
        return state["alive"]

    def _signal_fn(pid, sig):
        sent.append((pid, sig))
        if dies_on_signal:
            state["alive"] = False

    monkeypatch.setattr(kb, "_pid_alive", _pid_alive)
    return sent, _signal_fn


def _cols(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _row(conn, tid):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()


def _run_row(conn, tid):
    return conn.execute(
        "SELECT r.* FROM task_runs r JOIN tasks t ON t.current_run_id = r.id "
        "WHERE t.id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind):
    return [
        json.loads(r["payload"]) if r["payload"] else {}
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        )
    ]


# ---------------------------------------------------------------------------
# Schema + migration
# ---------------------------------------------------------------------------

def test_fresh_schema_carries_progress_columns(kanban_home):
    """Both the task and its run carry an independent progress lease."""
    with kb.connect() as conn:
        assert "last_progress_at" in _cols(conn, "tasks")
        assert "last_progress_at" in _cols(conn, "task_runs")


def test_migration_restores_progress_columns_and_preserves_rows(kanban_home):
    """An existing board that predates the progress lease must migrate in
    place — additively, without losing rows or the in-flight claim."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute("ALTER TABLE tasks DROP COLUMN last_progress_at")
        conn.execute("ALTER TABLE task_runs DROP COLUMN last_progress_at")
        conn.commit()
        assert "last_progress_at" not in _cols(conn, "tasks")

        kb._migrate_add_optional_columns(conn)

        assert "last_progress_at" in _cols(conn, "tasks")
        assert "last_progress_at" in _cols(conn, "task_runs")
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "running"
        # Legacy rows migrate to NULL — never to "now", which would hand a
        # wedged pre-upgrade worker a fresh lease it never earned.
        assert _row(conn, tid)["last_progress_at"] is None


def test_migration_is_idempotent(kanban_home):
    with kb.connect() as conn:
        kb._migrate_add_optional_columns(conn)
        kb._migrate_add_optional_columns(conn)
        assert "last_progress_at" in _cols(conn, "tasks")


def test_task_dataclass_exposes_progress_lease(kanban_home):
    """Dashboard/API serialisation is ``asdict(Task)`` — the field has to be
    on the dataclass or the UI can never show it."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        task = kb.get_task(conn, tid)
        assert hasattr(task, "last_progress_at")
        run = kb.get_run(conn, task.current_run_id)
        assert hasattr(run, "last_progress_at")


# ---------------------------------------------------------------------------
# The causal invariant: liveness never renews progress
# ---------------------------------------------------------------------------

def test_claim_stamps_the_progress_lease(kanban_home):
    """A claim is itself a state transition: the run starts with progress."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        assert _row(conn, tid)["last_progress_at"] is not None
        assert _run_row(conn, tid)["last_progress_at"] is not None


def test_stream_activity_renews_liveness_but_not_progress(kanban_home):
    """THE regression. ``heartbeat_worker`` / ``heartbeat_claim`` are the
    liveness bridge that ``_touch_activity`` drives on every stream chunk.
    They must move ``last_heartbeat_at`` and ``claim_expires`` — and must
    leave ``last_progress_at`` exactly where it was."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale_progress = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ?, last_heartbeat_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (stale_progress, stale_progress, stale_progress, tid),
        )
        conn.commit()

        host = kb._claimer_id().split(":", 1)[0]
        for _ in range(5):  # thousands of tokens, no tools
            kb.heartbeat_claim(conn, tid, claimer=f"{host}:worker")
            kb.heartbeat_worker(conn, tid)

        row = _row(conn, tid)
        assert row["last_heartbeat_at"] > stale_progress, "liveness must renew"
        assert row["claim_expires"] > stale_progress, "claim TTL must renew"
        assert row["last_progress_at"] == stale_progress, (
            "raw stream/API activity must NOT renew the progress lease"
        )


def test_tool_invocation_renews_progress(kanban_home):
    with kb.connect() as conn:
        tid = _running_task(conn)
        old = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (old, tid)
        )
        conn.execute(
            "UPDATE task_runs SET last_progress_at = ? WHERE id = ?",
            (old, _run_row(conn, tid)["id"]),
        )
        conn.commit()

        receipt = kb.record_progress(conn, tid, source=kb.PROGRESS_SOURCE_TOOL)

        assert receipt["recorded"] is True
        assert receipt["source"] == kb.PROGRESS_SOURCE_TOOL
        assert _row(conn, tid)["last_progress_at"] > old
        assert _run_row(conn, tid)["last_progress_at"] > old


def test_board_state_transition_renews_progress(kanban_home):
    """A durable board transition is progress evidence in its own right —
    whichever surface (worker tool, CLI, dashboard) produced it."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        old = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (old, tid)
        )
        conn.commit()

        kb.add_comment(conn, tid, author="worker", body="hotspot: a.py — churn")

        assert _row(conn, tid)["last_progress_at"] > old


def test_dispatcher_bookkeeping_events_do_not_renew_progress(kanban_home):
    """``heartbeat`` / ``claim_extended`` are lease bookkeeping, not work.
    If they renewed progress the whole guard would be self-defeating."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        old = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (old, tid)
        )
        conn.commit()

        with kb.write_txn(conn):
            kb._append_event(conn, tid, "heartbeat", {"note": None})
            kb._append_event(conn, tid, "claim_extended", {"reason": "pid_alive"})

        assert _row(conn, tid)["last_progress_at"] == old


# ---------------------------------------------------------------------------
# Class boundary: model-authored text is NEVER progress evidence
# ---------------------------------------------------------------------------
# The first cut of this fix let ``kanban_heartbeat(note=...)`` renew the
# progress lease when the note was "evidence-bearing and new".  That is not a
# guard, it is a spelling test: the note is free text the model writes, so a
# worker that never leaves its reasoning loop can renew forever just by
# varying a sentence.  Nothing about the string is validated against the
# world.  The boundary now sits at the class of the signal, not its content:
# only the centralized tool middleware and durable board state transitions
# renew progress, and ``record_progress`` refuses everything else.

def test_record_progress_rejects_sources_outside_the_allowlist(kanban_home):
    """Enforced inside ``record_progress``, not at its call sites, so a
    future caller that invents a model-driven source gets a refusal rather
    than a lease renewal."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        old = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (old, tid)
        )
        conn.commit()

        for bogus in ("checkpoint", "heartbeat", "note", "model", ""):
            receipt = kb.record_progress(conn, tid, source=bogus)
            assert receipt["recorded"] is False, bogus
            assert receipt["reason"] == "unsupported_source", bogus
        assert _row(conn, tid)["last_progress_at"] == old


def test_record_progress_admits_only_the_tool_middleware(kanban_home):
    """``record_progress`` is the callable entry point and admits only the
    tool middleware. Board transitions renew inline (they already hold the
    write txn) and are pinned by PROGRESS_EVENT_KINDS."""
    assert not hasattr(kb, "PROGRESS_SOURCE_CHECKPOINT")
    assert kb.PROGRESS_RENEWAL_SOURCES == frozenset({kb.PROGRESS_SOURCE_TOOL})


def test_record_progress_has_no_note_channel(kanban_home):
    """The text channel into the lease is removed, not merely guarded — no
    argument carries model-authored bytes into ``last_progress_at``."""
    import inspect
    params = inspect.signature(kb.record_progress).parameters
    assert "note" not in params
    assert not hasattr(kb, "_normalize_progress_evidence")
    assert not hasattr(kb, "_last_progress_evidence")


def test_progress_never_writes_an_event_row(kanban_home):
    """Progress is a column stamp only. No ``progress`` event exists to be
    mined for 'the worker said it did something'."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        for _ in range(5):
            kb.record_progress(conn, tid, source=kb.PROGRESS_SOURCE_TOOL)
        assert _events(conn, tid, "progress") == []


def test_progress_is_rejected_for_a_superseded_run(kanban_home):
    """A worker whose run was already reclaimed must not resurrect the
    lease of whatever run replaced it."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        run_id = _run_row(conn, tid)["id"]
        receipt = kb.record_progress(
            conn, tid, source=kb.PROGRESS_SOURCE_TOOL,
            expected_run_id=run_id + 1000,
        )
        assert receipt["recorded"] is False
        assert receipt["reason"] == "superseded"


# ---------------------------------------------------------------------------
# No-progress reclaim
# ---------------------------------------------------------------------------

def test_no_progress_reclaim_fires_when_liveness_is_fresh(kanban_home, monkeypatch):
    """The reproduced incident, end to end at the DB layer: PID alive,
    heartbeat seconds old, progress lease hours old."""
    killed, signal_fn = _mock_worker(monkeypatch)

    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (now - 5, now - 7200, now + 900, tid),
        )
        conn.execute(
            "UPDATE task_runs SET last_progress_at = ?, started_at = ? "
            "WHERE task_id = ?",
            (now - 7200, now - 7300, tid),
        )
        conn.commit()

        reclaimed = kb.detect_no_progress_running(
            conn,
            no_progress_timeout_seconds=2700,
            signal_fn=signal_fn,
        )

        assert reclaimed == [tid]
        assert kb.get_task(conn, tid).status in {"ready", "changes_requested"}
        assert killed, "the wedged worker must be terminated before reclaim"

        payload = _events(conn, tid, "no_progress")[0]
        assert payload["progress_age_seconds"] >= 7200
        assert payload["timeout_seconds"] == 2700
        assert payload["heartbeat_age_seconds"] <= 60
        # The whole point: classify *why*, so an operator can tell a
        # reasoning loop from a dead process.
        assert payload["worker_state"] == "alive"
        assert payload["liveness"] == "fresh"
        assert payload["classification"] == "no_progress"

        run = conn.execute(
            "SELECT outcome, status, error FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run["outcome"] == "no_progress"
        assert run["status"] == "no_progress"


def test_no_progress_reclaim_leaves_an_operator_receipt(kanban_home, monkeypatch):
    _killed, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (now, now - 7200, tid),
        )
        conn.commit()
        kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=signal_fn,
        )
        bodies = [c.body for c in kb.list_comments(conn, tid)]
        assert any("no observable progress" in b for b in bodies)


def test_slow_healthy_api_call_is_not_reclaimed(kanban_home, monkeypatch):
    """A single long tool-free API call keeps its claim: liveness carries it
    and the progress lease has not yet expired.  Backward compatibility for
    legitimate long model calls is the whole reason liveness stays."""
    _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (now, now - 1800, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700,
        ) == []
        assert kb.get_task(conn, tid).status == "running"


def test_no_progress_disabled_when_timeout_is_zero(kanban_home, monkeypatch):
    _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (int(time.time()), 1, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=0,
        ) == []
        assert kb.get_task(conn, tid).status == "running"


def test_no_progress_falls_back_to_run_start_for_legacy_rows(kanban_home, monkeypatch):
    """A run claimed before the upgrade has ``last_progress_at IS NULL``.
    It must be measured from the run's own start, not treated as
    infinitely stale (which would reclaim every in-flight worker the
    moment the operator upgrades)."""
    _killed, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_progress_at = NULL, last_heartbeat_at = ? "
            "WHERE id = ?",
            (now, tid),
        )
        conn.execute(
            "UPDATE task_runs SET last_progress_at = NULL, started_at = ? "
            "WHERE task_id = ?",
            (now - 60, tid),
        )
        conn.commit()

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700,
        ) == []

        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (now - 9000, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn,
        ) == [tid]


def test_no_progress_defers_while_our_worker_survives_termination(
    kanban_home, monkeypatch,
):
    """Never release a claim next to a worker that refused to die — that
    spawns a duplicate. Same rule the TTL path already follows."""
    _killed, signal_fn = _mock_worker(monkeypatch, dies_on_signal=False)
    monkeypatch.setattr(kb.time, "sleep", lambda _s: None)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (now, now - 7200, tid),
        )
        conn.commit()

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=signal_fn,
        ) == []
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid, "reclaim_deferred")


def test_no_progress_counts_a_failure_so_it_cannot_spin_forever(
    kanban_home, monkeypatch,
):
    """Unlike the absent-heartbeat ``stale`` path, a no-progress reclaim is
    an evidence-based worker fault: the model was demonstrably live and
    produced nothing. Respawning it unboundedly is the spin loop the
    circuit breaker exists to stop."""
    _killed, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        before = kb.get_task(conn, tid).consecutive_failures
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (now, now - 7200, tid),
        )
        conn.commit()
        kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=signal_fn,
        )
        assert kb.get_task(conn, tid).consecutive_failures == before + 1


def test_dead_worker_is_left_to_the_crash_path(kanban_home, monkeypatch):
    """Classification, not overlap: a dead PID is a crash, and the crash
    path owns it (it carries exit-code forensics this pass has no access
    to)."""
    _killed, signal_fn = _mock_worker(monkeypatch, alive=False)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (now, now - 7200, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=signal_fn,
        ) == []


# ---------------------------------------------------------------------------
# Dispatcher wiring + config
# ---------------------------------------------------------------------------

def test_dispatch_tick_runs_the_no_progress_pass(kanban_home, monkeypatch):
    _killed, _signal_fn = _mock_worker(monkeypatch)
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda *a, **kw: {"prev_pid": None, "host_local": True,
                          "termination_attempted": True, "terminated": True,
                          "sigkill": False},
    )
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (now, now - 7200, now + 900, tid),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None,
            no_progress_timeout_seconds=2700,
        )
        assert result.no_progress == [tid]


def test_dispatch_tick_no_progress_defaults_to_enabled(kanban_home, monkeypatch):
    """Backward-compatible *enough*: the guard is on by default, with a
    window wide enough for legitimate long model calls."""
    assert kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS >= 1800
    _killed, _signal_fn = _mock_worker(monkeypatch)
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda *a, **kw: {"prev_pid": None, "host_local": True,
                          "termination_attempted": True, "terminated": True,
                          "sigkill": False},
    )
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (
                now,
                now - kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS - 600,
                now + 900,
                tid,
            ),
        )
        conn.commit()
        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)
        assert result.no_progress == [tid]


def test_resolve_no_progress_timeout_seconds_contract():
    d = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    assert kb.resolve_no_progress_timeout_seconds(None) == d
    assert kb.resolve_no_progress_timeout_seconds(0) == 0        # explicit off
    assert kb.resolve_no_progress_timeout_seconds(900) == 900
    assert kb.resolve_no_progress_timeout_seconds("900") == 900
    # Invalid / nonsensical values keep the guard on rather than silently
    # disabling it.
    assert kb.resolve_no_progress_timeout_seconds("banana") == d
    assert kb.resolve_no_progress_timeout_seconds(-5) == d


def test_config_default_exposes_the_no_progress_timeout(tmp_path, monkeypatch):
    """Real config load against a temp HERMES_HOME — the gateway and the CLI
    both read the effective value from here."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.config import DEFAULT_CONFIG, load_config

    assert "no_progress_timeout_seconds" in DEFAULT_CONFIG["kanban"]
    cfg = load_config()
    assert (
        cfg["kanban"]["no_progress_timeout_seconds"]
        == kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    )


def test_user_config_override_propagates(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(
        "kanban:\n  no_progress_timeout_seconds: 600\n", encoding="utf-8"
    )

    from hermes_cli.config import load_config

    cfg = load_config()
    assert kb.resolve_no_progress_timeout_seconds(
        cfg["kanban"]["no_progress_timeout_seconds"]
    ) == 600


def test_resolve_no_progress_timeout_rejects_bools(caplog):
    """``True`` is an ``int`` in Python, so a YAML ``no_progress_timeout_seconds:
    true`` would otherwise resolve to a one-second progress bound and reclaim
    every healthy worker on the next tick."""
    d = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    with caplog.at_level("WARNING"):
        assert kb.resolve_no_progress_timeout_seconds(True) == d
        assert kb.resolve_no_progress_timeout_seconds(False) == d
    assert "no_progress_timeout_seconds" in caplog.text


def test_resolve_no_progress_timeout_rejects_sub_minute_windows(caplog):
    """1..59s cannot express a real progress bound — a single model call
    routinely exceeds it — so it is a units typo (minutes written as
    seconds), not an intent. Fall back to the default and say so."""
    d = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    with caplog.at_level("WARNING"):
        for bad in (1, 30, 45, 59):
            assert kb.resolve_no_progress_timeout_seconds(bad) == d, bad
    assert "no_progress_timeout_seconds" in caplog.text
    # The boundary itself, and 0 (explicit disable), stay honoured.
    assert kb.resolve_no_progress_timeout_seconds(60) == 60
    assert kb.resolve_no_progress_timeout_seconds(0) == 0
    assert kb.MIN_NO_PROGRESS_TIMEOUT_SECONDS == 60


def test_resolve_no_progress_timeout_warns_on_garbage(caplog):
    d = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    with caplog.at_level("WARNING"):
        assert kb.resolve_no_progress_timeout_seconds("banana") == d
        assert kb.resolve_no_progress_timeout_seconds(-5) == d
    assert "no_progress_timeout_seconds" in caplog.text


# ---------------------------------------------------------------------------
# Failure accounting: configured limit + auto-block surfacing
# ---------------------------------------------------------------------------

def _stub_termination(monkeypatch):
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda *a, **kw: {"prev_pid": None, "host_local": True,
                          "termination_attempted": True, "terminated": True,
                          "sigkill": False, "process_group": True},
    )


def _age_progress(conn, tid, seconds):
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
        "claim_expires = ? WHERE id = ?",
        (now, now - seconds, now + 900, tid),
    )
    conn.commit()


def test_no_progress_honours_the_configured_failure_limit(kanban_home, monkeypatch):
    """The dispatcher's ``kanban.failure_limit`` must reach the failure
    recorder. Without it the pass silently used DEFAULT_FAILURE_LIMIT and a
    board configured to give up after one no-progress round kept respawning."""
    _mock_worker(monkeypatch)
    _stub_termination(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)

        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None,
            no_progress_timeout_seconds=2700,
            failure_limit=1,
        )
        assert result.no_progress == [tid]
        assert kb.get_task(conn, tid).status == "blocked"


def test_no_progress_auto_block_is_surfaced_on_the_dispatch_result(
    kanban_home, monkeypatch,
):
    """A circuit-breaker trip has to be visible to telemetry and the tick
    hook, exactly like the crash path's."""
    _mock_worker(monkeypatch)
    _stub_termination(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)

        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None,
            no_progress_timeout_seconds=2700,
            failure_limit=1,
        )
        assert result.auto_blocked == [tid]


def test_no_progress_below_the_limit_does_not_auto_block(kanban_home, monkeypatch):
    _mock_worker(monkeypatch)
    _stub_termination(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)

        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None,
            no_progress_timeout_seconds=2700,
            failure_limit=5,
        )
        assert result.no_progress == [tid]
        assert result.auto_blocked == []
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Resume parity: a reclaimed review run comes back as a review run
# ---------------------------------------------------------------------------

def test_no_progress_reclaim_of_a_review_run_resumes_in_review(
    kanban_home, monkeypatch,
):
    """``_resume_status_from_events`` reads the most recent lifecycle event.
    If ``no_progress`` is not one of the kinds it looks at, a reviewer run
    reclaimed for no progress is silently demoted to an implementation run."""
    _mock_worker(monkeypatch)
    _stub_termination(monkeypatch)
    host = kb._claimer_id().split(":", 1)[0]
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review-me", assignee="a")
        impl = kb.claim_task(conn, tid, claimer=f"{host}:impl")
        assert kb.request_review(
            conn, tid, summary="done", reviewer="rev",
            expected_run_id=impl.current_run_id,
        ) is True
        assert kb.claim_review_task(
            conn, tid, claimer=f"{host}:reviewer",
        ) is not None
        kb._set_worker_pid(conn, tid, 4242)
        _age_progress(conn, tid, 7200)

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700,
        ) == [tid]
        assert kb.get_task(conn, tid).status == "review"
        assert kb._resume_status_from_events(conn, tid) == "review"


# ---------------------------------------------------------------------------
# Dispatch-tick observer hook: a no-progress-only tick is not "idle"
# ---------------------------------------------------------------------------

def test_dispatch_tick_hook_classifies_a_no_progress_only_tick_as_ok(monkeypatch):
    """Unit: the classifier itself. A DispatchResult whose only populated
    field is ``no_progress`` must not be reported as an idle tick."""
    from hermes_cli import lifecycle

    seen: list[dict] = []
    monkeypatch.setattr(kb, "_kanban_observer_consumed", lambda _name: True)
    monkeypatch.setattr(
        lifecycle, "invoke_hook",
        lambda name, **kw: seen.append({"hook": name, **kw}),
    )

    result = kb.DispatchResult()
    result.no_progress = ["t-1"]
    kb._fire_dispatch_tick_hook(result, board="b")

    assert seen and seen[-1]["outcome"] == "ok"
    assert seen[-1]["result"].no_progress == ["t-1"]

    # ...and a genuinely empty tick is still idle.
    seen.clear()
    kb._fire_dispatch_tick_hook(kb.DispatchResult(), board="b")
    assert seen[-1]["outcome"] == "idle"


def test_dispatch_tick_hook_e2e_no_progress_only_tick_is_not_idle(
    kanban_home, monkeypatch,
):
    """E2E through ``dispatch_once``: the only thing that happened this tick
    was a progress-lease reclaim, and every dispatcher-health subscriber must
    be told the tick acted — this is precisely the incident being watched
    for, so reporting ``idle`` hides it."""
    from hermes_cli import lifecycle

    seen: list[dict] = []
    monkeypatch.setattr(kb, "_kanban_observer_consumed", lambda _name: True)
    monkeypatch.setattr(
        lifecycle, "invoke_hook",
        lambda name, **kw: seen.append({"hook": name, **kw}),
    )

    _mock_worker(monkeypatch)
    _stub_termination(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None,
            max_spawn=0,
            no_progress_timeout_seconds=2700,
        )

    assert result.no_progress == [tid]
    assert result.spawned == [] and result.crashed == [] and result.stale == []
    ticks = [s for s in seen if s.get("hook") == "on_kanban_dispatch_tick"]
    assert ticks, "dispatch tick hook did not fire"
    assert ticks[-1]["outcome"] == "ok"
    assert ticks[-1]["result"].no_progress == [tid]


def test_no_progress_is_in_the_dispatch_tick_outcome_classifier():
    """Unit backstop for the E2E above — the classifier's own field list."""
    import inspect
    src = inspect.getsource(kb._fire_dispatch_tick_hook)
    assert "result.no_progress" in src


# ---------------------------------------------------------------------------
# Worker termination: kill the process GROUP, not just the leader
# ---------------------------------------------------------------------------
# The dispatcher spawns workers with ``start_new_session=True``, so each worker
# is its own session and process-group leader (pgid == pid). Signalling only
# the leader leaves everything it spawned — the build, the crawl, the training
# run — alive and holding the workspace, CPU and file locks, while the board
# happily re-queues the card and spawns a second worker beside the orphans.
# The reclaim helper must take the whole group when, and only when, that group
# is one the dispatcher owns.

def _fake_killers():
    """(per-pid sends, group sends, signal_fn, killpg_fn)."""
    pid_sent: list = []
    grp_sent: list = []
    return (
        pid_sent, grp_sent,
        lambda pid, sig: pid_sent.append((pid, sig)),
        lambda pgid, sig: grp_sent.append((pgid, sig)),
    )


def _owned(pid=4242, **overrides):
    """(stored_identity_json, probe_fn) for a provably-owned synthetic pid."""
    ident = _fake_identity(pid, **overrides)
    return json.dumps(ident), (lambda p: dict(ident) if int(p) == int(pid) else None)


def test_terminate_kills_the_process_group_when_the_worker_leads_one(monkeypatch):
    import os as _os
    import signal as _signal

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(_os, "getpgid", lambda p: 4242 if p else 777)
    pid_sent, grp_sent, sig_fn, killpg_fn = _fake_killers()
    stored, probe = _owned()

    host = kb._claimer_id().split(":", 1)[0]
    info = kb._terminate_reclaimed_worker(
        4242, f"{host}:worker", stored_identity=stored, probe_fn=probe,
        signal_fn=sig_fn, killpg_fn=killpg_fn,
    )

    assert grp_sent == [(4242, _signal.SIGTERM)]
    assert pid_sent == []
    assert info["process_group"] is True
    assert info["terminated"] is True
    assert info["ownership"] == "proven"


def test_terminate_signals_nothing_when_the_worker_leads_no_group(monkeypatch):
    """Worker inherited the dispatcher's group (no ``start_new_session``).

    That group contains the dispatcher itself, so it can never be signalled —
    but neither can the bare pid. A process that is not the session leader we
    spawned is a process whose identity we have not established, and the whole
    point of the proof is that "probably ours" is not a category. Signal
    nothing and let the caller hold the claim.
    """
    import os as _os

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(_os, "getpgid", lambda p: 999)
    pid_sent, grp_sent, sig_fn, killpg_fn = _fake_killers()
    # sid/pgid say 999: alive, but not the session leader we spawned.
    stored, probe = _owned(sid=999, pgid=999)

    host = kb._claimer_id().split(":", 1)[0]
    info = kb._terminate_reclaimed_worker(
        4242, f"{host}:worker", stored_identity=stored, probe_fn=probe,
        signal_fn=sig_fn, killpg_fn=killpg_fn,
    )

    assert grp_sent == []
    assert pid_sent == []
    assert info["process_group"] is False
    assert info["signalled"] is False
    assert info["ownership"] == "unproven"
    assert info["ownership_reason"] == "not_dispatcher_session"
    assert kb._reclaim_hold_reason(info) == "ownership_unproven"


def test_terminate_never_signals_our_own_process_group(monkeypatch):
    """Defence in depth: if the pid we were handed somehow resolves to the
    dispatcher's own group, fall back to the single (proven) pid."""
    import os as _os

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(_os, "getpgid", lambda _p: 4242)  # incl. getpgid(0)
    pid_sent, grp_sent, sig_fn, killpg_fn = _fake_killers()
    stored, probe = _owned()

    host = kb._claimer_id().split(":", 1)[0]
    info = kb._terminate_reclaimed_worker(
        4242, f"{host}:worker", stored_identity=stored, probe_fn=probe,
        signal_fn=sig_fn, killpg_fn=killpg_fn,
    )

    assert grp_sent == []
    assert pid_sent and info["process_group"] is False


def test_terminate_falls_back_when_the_group_cannot_be_resolved(monkeypatch):
    import os as _os

    def _boom(_p):
        raise ProcessLookupError()

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(_os, "getpgid", _boom)
    pid_sent, grp_sent, sig_fn, killpg_fn = _fake_killers()
    stored, probe = _owned()

    host = kb._claimer_id().split(":", 1)[0]
    info = kb._terminate_reclaimed_worker(
        4242, f"{host}:worker", stored_identity=stored, probe_fn=probe,
        signal_fn=sig_fn, killpg_fn=killpg_fn,
    )
    assert grp_sent == []
    assert pid_sent and info["process_group"] is False


def test_windows_termination_stays_per_pid(monkeypatch):
    """Windows has no POSIX process groups here; ``os.killpg``/``os.getpgid``
    are absent. Behaviour must be exactly what it was before this change —
    a per-pid TerminateProcess — except that the pid must now carry a proven
    creation time, which is Windows' equivalent birth stamp."""
    import os as _os
    import signal as _signal

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.delattr(_os, "getpgid", raising=False)
    monkeypatch.delattr(_os, "killpg", raising=False)
    pid_sent, _grp, sig_fn, _killpg = _fake_killers()
    win_ident = {
        "v": kb.WORKER_IDENTITY_VERSION,
        "scheme": "create_time",
        "pid": 4242,
        "create_time": 1_700_000_000.5,
    }

    host = kb._claimer_id().split(":", 1)[0]
    info = kb._terminate_reclaimed_worker(
        4242, f"{host}:worker",
        stored_identity=json.dumps(win_ident),
        probe_fn=lambda _p: dict(win_ident),
        signal_fn=sig_fn,
    )

    assert pid_sent == [(4242, _signal.SIGTERM)]
    assert info["process_group"] is False
    assert info["ownership"] == "proven"
    assert kb._dispatcher_owned_pgid(4242) is None


def test_windows_without_a_provable_creation_time_defers(monkeypatch):
    """No psutil / no creation time on Windows means no proof, and no proof
    means no TerminateProcess — the pid may name something else entirely."""
    import os as _os

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.delattr(_os, "getpgid", raising=False)
    monkeypatch.delattr(_os, "killpg", raising=False)
    pid_sent, _grp, sig_fn, _killpg = _fake_killers()

    host = kb._claimer_id().split(":", 1)[0]
    info = kb._terminate_reclaimed_worker(
        4242, f"{host}:worker",
        stored_identity=None,
        probe_fn=lambda _p: None,
        signal_fn=sig_fn,
    )

    assert pid_sent == []
    assert info["ownership_unproven"] is True
    assert kb._reclaim_hold_reason(info) == "ownership_unproven"


def test_injected_signal_fn_without_a_group_killer_keeps_per_pid_semantics(
    monkeypatch,
):
    """A test that injects only a per-pid killer must never reach the real
    ``os.killpg`` — otherwise a fake pid could signal a live group."""
    import os as _os
    import signal as _signal

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(_os, "getpgid", lambda p: 4242 if p else 777)
    monkeypatch.setattr(
        _os, "killpg",
        lambda *_a: pytest.fail("real killpg reached with an injected signal_fn"),
    )
    pid_sent, _grp, sig_fn, _killpg = _fake_killers()
    stored, probe = _owned()

    host = kb._claimer_id().split(":", 1)[0]
    kb._terminate_reclaimed_worker(
        4242, f"{host}:worker", stored_identity=stored, probe_fn=probe,
        signal_fn=sig_fn,
    )
    assert pid_sent == [(4242, _signal.SIGTERM)]


@pytest.mark.skipif(
    not hasattr(os, "killpg") or not hasattr(os, "setsid"),
    reason="POSIX process groups only",
)
def test_reclaim_reaps_the_workers_descendants(tmp_path):
    """E2E with real processes: a dispatcher-shaped worker (own session) that
    spawned a long-running child. Reclaiming must leave neither behind."""
    import subprocess
    import sys

    childfile = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time, pathlib\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"pathlib.Path({str(childfile)!r}).write_text(str(p.pid))\n"
        "time.sleep(300)\n"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True,
    )
    try:
        for _ in range(100):
            if childfile.exists() and childfile.read_text().strip():
                break
            time.sleep(0.1)
        child_pid = int(childfile.read_text().strip())
        assert os.getpgid(worker.pid) == worker.pid, "worker must lead its group"
        assert os.getpgid(child_pid) == worker.pid, "child must join that group"

        host = kb._claimer_id().split(":", 1)[0]
        # Real capture, real verification: the identity is read from this
        # process's actual /proc entry at "spawn" time and re-read by the
        # reclaim, which is the round trip that has to work in production.
        stored = json.dumps(kb.capture_worker_identity(worker.pid))
        info = kb._terminate_reclaimed_worker(
            worker.pid, f"{host}:worker", stored_identity=stored,
        )

        def _gone(pid):
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                except (ProcessLookupError, PermissionError):
                    return True
                time.sleep(0.1)
            return False

        # The property that matters, asserted first: nothing the worker
        # spawned is left holding the workspace after the reclaim.
        worker.wait(timeout=10)
        assert _gone(child_pid), "descendant survived the reclaim"
        assert info["terminated"] is True
        assert info["process_group"] is True
    finally:
        for pid in (worker.pid,):
            try:
                os.killpg(pid, 9)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            worker.wait(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker process birth identity — ownership must be PROVEN before any signal
# ---------------------------------------------------------------------------
# A PID is a recycled integer. Every reclaim path in kanban_db used to signal
# that bare number, and since the process-group change above a single wrong
# PID no longer costs one wrong process — it costs a whole wrong process TREE.
# These tests pin the rule that closes it: re-read the birth identity recorded
# at spawn, and if it does not verify, signal NOTHING.

def test_proc_stat_parser_survives_a_comm_with_spaces_and_parentheses():
    """``comm`` is unescaped in /proc/<pid>/stat and may contain spaces AND
    parentheses. ``line.split()[21]`` reads a completely different field for
    such a process — which here would mean comparing a wrong start time and
    either refusing to reclaim a real worker or accepting a recycled PID."""
    tail = " ".join(str(n) for n in range(3, 53))  # fields 3..52
    for comm in ("python3", "my app (beta)", "a b c", "((()))", ") ("):
        raw = f"1234 ({comm}) {tail}\n"
        parsed = kb._parse_proc_pid_stat(raw)
        assert parsed is not None, comm
        assert parsed["ppid"] == 4
        assert parsed["pgid"] == 5
        assert parsed["sid"] == 6
        assert parsed["starttime"] == 22, comm

    # The naive parser everyone reaches for, shown failing on the same input,
    # so this test documents WHY the helper exists.
    naive = f"1234 (my app (beta)) {tail}".split()[21]
    assert int(naive) != 22


def test_proc_stat_parser_rejects_malformed_lines():
    assert kb._parse_proc_pid_stat("") is None
    assert kb._parse_proc_pid_stat("1234 no-parens 1 2 3") is None
    assert kb._parse_proc_pid_stat("1234 (sh) S 1 1") is None          # truncated
    assert kb._parse_proc_pid_stat("1234 (sh) S " + "x " * 60) is None  # non-numeric


def test_capture_worker_identity_reads_this_process(kanban_home, monkeypatch):
    """The real capture, against a real pid: our own."""
    monkeypatch.setattr(
        kb, "capture_worker_identity", _REAL_CAPTURE_WORKER_IDENTITY)
    ident = kb.capture_worker_identity(os.getpid())
    if ident is None:  # pragma: no cover - unusual platform
        pytest.skip("no birth identity available on this platform")
    assert ident["pid"] == os.getpid()
    assert ident["scheme"] in {"linux_proc", "create_time"}
    # Stable across reads — that is the entire property being relied on.
    assert kb.capture_worker_identity(os.getpid()) == ident


def test_capture_worker_identity_is_none_for_a_dead_pid(kanban_home, monkeypatch):
    monkeypatch.setattr(
        kb, "capture_worker_identity", _REAL_CAPTURE_WORKER_IDENTITY)
    assert kb.capture_worker_identity(0) is None
    assert kb.capture_worker_identity(-1) is None
    assert kb.capture_worker_identity(None) is None


def test_verify_rejects_a_recycled_pid():
    """Same pid number, different process: the starttime cannot match."""
    stored = json.dumps(_fake_identity(4242, starttime=111))
    live = _fake_identity(4242, starttime=999)  # reborn later
    assert kb.verify_worker_ownership(4242, stored, probe_fn=lambda _p: live) == (
        "none", "identity_mismatch",
    )


def test_verify_pins_the_identity_pid_field_independently():
    """The live birth record's pid is a proof leg, not decoration.

    Keep every other field—including the live dispatcher-session shape—equal,
    so deleting only the live-pid comparison makes this test approve.
    """
    stored = _fake_identity(4242)
    live = dict(stored, pid=4343)
    assert kb.verify_worker_ownership(
        4242, json.dumps(stored), probe_fn=lambda _p: live,
    ) == ("none", "identity_mismatch")


def test_verify_pins_the_stored_pid_field_for_sidless_identity():
    """Persisted PID remains load-bearing where no SID/PGID proof exists.

    This is the Windows/create-time shape. Every field except the stored PID
    agrees with the live process and requested PID, so deleting only the
    stored-pid comparison would incorrectly authorise a per-PID signal.
    """
    stored = {
        "v": kb.WORKER_IDENTITY_VERSION,
        "scheme": "create_time",
        "pid": 1111,
        "create_time": 1000.0,
    }
    live = dict(stored, pid=2222)
    assert kb.verify_worker_ownership(
        2222, json.dumps(stored), probe_fn=lambda _p: live,
    ) == ("none", "identity_mismatch")


def test_verify_pins_stored_sid_and_pgid_independently():
    """A stale session/group record must not be treated as today's leader.

    The live record is otherwise a valid dispatcher-owned leader. Removing
    only the stored-vs-live sid/pgid comparison would therefore authorise it.
    """
    stored = _fake_identity(4242, sid=999, pgid=999)
    live = _fake_identity(4242)
    assert kb.verify_worker_ownership(
        4242, json.dumps(stored), probe_fn=lambda _p: live,
    ) == ("none", "identity_mismatch")


def test_verify_rejects_a_pid_reborn_after_a_reboot():
    """``starttime`` is ticks since boot, so it aliases across reboots. The
    boot_id pin is what makes a pre-reboot record un-matchable."""
    stored = json.dumps(_fake_identity(4242, boot_id="boot-one"))
    live = _fake_identity(4242, boot_id="boot-two")  # identical starttime
    assert stored != json.dumps(live)
    assert kb.verify_worker_ownership(4242, stored, probe_fn=lambda _p: live) == (
        "none", "identity_mismatch",
    )


def test_capture_returns_no_linux_identity_without_a_boot_id(
    kanban_home, monkeypatch,
):
    """No boot id, no Linux identity — NULL, not a weaker one.

    ``starttime`` is clock ticks since boot. Recorded without the boot id it
    aliases across a restart, so a record minted that way is not a lesser
    proof, it is a proof that can be satisfied by the wrong process. And
    because a missing key compares equal to a missing key, two such records
    verify each other. Returning None stores NULL, which every reclaim path
    already reads as "unproven" and refuses to signal.
    """
    if sys.platform != "linux":  # pragma: no cover - platform-specific
        pytest.skip("linux birth records only")
    monkeypatch.setattr(
        kb, "capture_worker_identity", _REAL_CAPTURE_WORKER_IDENTITY)
    # Sanity: with a boot id this host does produce an identity, so a None
    # below is caused by the boot id and nothing else.
    assert kb.capture_worker_identity(os.getpid()) is not None

    for missing in (None, "", "   ", "\n"):
        monkeypatch.setattr(kb, "_read_linux_boot_id", lambda: missing)
        assert kb.capture_worker_identity(os.getpid()) is None, (
            f"boot_id {missing!r} must not mint an identity"
        )


def test_boot_id_reader_is_none_when_the_file_is_absent_or_empty(monkeypatch):
    """Unreadable, undecodable and empty all mean "no id to pin to"."""
    import builtins

    class _Reader:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            if isinstance(self._payload, BaseException):
                raise self._payload
            return self._payload

    def _run(payload):
        # Patch, call and RESTORE before asserting: ``open`` is patched
        # process-wide here, and a failing assert inside the window would
        # take pytest's own reporting down with it.
        if isinstance(payload, OSError):
            def _fake_open(*_a, **_kw):
                raise payload
        else:
            def _fake_open(*_a, **_kw):
                return _Reader(payload)
        monkeypatch.setattr(builtins, "open", _fake_open)
        try:
            return kb._read_linux_boot_id()
        finally:
            monkeypatch.undo()

    assert _run(OSError("no such file")) is None
    assert _run("") is None
    assert _run("   \n") is None
    assert _run(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")) is None
    assert _run("  7f3d-boot  \n") == "7f3d-boot"


def test_verify_never_approves_two_identities_that_both_lack_a_boot_id():
    """The None == None hole. Both sides missing the key is the shape a
    pre-reboot record and a post-reboot alias have in common, and a plain
    ``stored.get(...) != live.get(...)`` reads it as agreement."""
    ident = _fake_identity(4242)
    del ident["boot_id"]
    mode, reason = kb.verify_worker_ownership(
        4242, json.dumps(ident), probe_fn=lambda _p: dict(ident),
    )
    assert mode == "none", "two boot-id-less records must never verify"
    assert reason == "boot_id_absent"


def test_verify_rejects_a_boot_id_missing_on_either_side():
    """Either half missing is unprovable. Stored-missing is the legacy row
    written before the pin existed; live-missing is a host that has stopped
    exposing ``/proc/sys/kernel/random/boot_id``."""
    full = _fake_identity(4242)
    stored_missing = {k: v for k, v in full.items() if k != "boot_id"}
    assert kb.verify_worker_ownership(
        4242, json.dumps(stored_missing), probe_fn=lambda _p: dict(full),
    ) == ("none", "boot_id_absent")
    assert kb.verify_worker_ownership(
        4242, json.dumps(full), probe_fn=lambda _p: dict(stored_missing),
    ) == ("none", "boot_id_absent")
    # Present-but-blank is missing too, not a value that can match a blank.
    blank = _fake_identity(4242, boot_id="   ")
    assert kb.verify_worker_ownership(
        4242, json.dumps(blank), probe_fn=lambda _p: dict(blank),
    ) == ("none", "boot_id_absent")


def test_a_boot_id_less_record_cannot_alias_a_post_reboot_pid():
    """The reboot alias, spelled out: identical pid, identical starttime,
    different boot — the exact collision the pin exists to break. With no
    boot id on either side the two records are byte-identical, so the ONLY
    thing that can reject them is the absence check."""
    pre_reboot = _fake_identity(4242, starttime=555)
    del pre_reboot["boot_id"]
    post_reboot = _fake_identity(4242, starttime=555)  # same ticks, new boot
    del post_reboot["boot_id"]
    assert pre_reboot == post_reboot, "the alias is indistinguishable by value"

    mode, reason = kb.verify_worker_ownership(
        4242, json.dumps(pre_reboot), probe_fn=lambda _p: post_reboot,
    )
    assert (mode, reason) == ("none", "boot_id_absent")


def test_a_missing_boot_id_is_unknown_not_a_dead_worker(kanban_home):
    """``boot_id_absent`` must not read as ``identity_mismatch``.

    ``_identity_contradicts_liveness`` turns a mismatch into "the pid was
    recycled, close the run as crashed". An unevaluatable record is unknown,
    not contradicted — treating it as gone would close a run out from under a
    worker that is very much alive."""
    ident = _fake_identity(4242)
    del ident["boot_id"]
    with kb.connect() as conn:
        tid = _running_task(conn, pid=4242)
        conn.execute(
            "UPDATE tasks SET worker_identity = ? WHERE id = ?",
            (json.dumps(ident), tid),
        )
        conn.commit()
        assert kb._identity_contradicts_liveness(conn, tid, 4242) is False


def test_verify_rejects_absent_unreadable_and_malformed_identities():
    live = _fake_identity(4242)
    probe = lambda _p: live  # noqa: E731
    assert kb.verify_worker_ownership(4242, None, probe_fn=probe) == (
        "none", "identity_absent")
    assert kb.verify_worker_ownership(4242, "", probe_fn=probe) == (
        "none", "identity_absent")
    assert kb.verify_worker_ownership(4242, "not json", probe_fn=probe) == (
        "none", "identity_absent")
    assert kb.verify_worker_ownership(4242, "[1,2,3]", probe_fn=probe) == (
        "none", "identity_absent")
    assert kb.verify_worker_ownership(0, json.dumps(live), probe_fn=probe) == (
        "none", "no_pid")
    assert kb.verify_worker_ownership(
        4242, json.dumps(live), probe_fn=lambda _p: None,
    ) == ("none", "identity_unreadable")
    # A probe that raises is an unreadable identity, not a crash.
    def _boom(_p):
        raise OSError("boom")
    assert kb.verify_worker_ownership(
        4242, json.dumps(live), probe_fn=_boom,
    ) == ("none", "identity_unreadable")


def test_verify_rejects_a_scheme_change():
    """A board carried between platforms (or a restored DB) must not have a
    Linux starttime compared against a psutil epoch."""
    stored = json.dumps(_fake_identity(4242))
    live = {"v": 1, "scheme": "create_time", "pid": 4242, "create_time": 1.0}
    assert kb.verify_worker_ownership(4242, stored, probe_fn=lambda _p: live) == (
        "none", "identity_scheme_changed",
    )


def test_verify_authorises_the_group_only_for_a_session_leader():
    ident = _fake_identity(4242)
    assert kb.verify_worker_ownership(
        4242, json.dumps(ident), probe_fn=lambda _p: ident,
    ) == ("group", "ok")

    inherited = _fake_identity(4242, sid=999, pgid=999)
    assert kb.verify_worker_ownership(
        4242, json.dumps(inherited), probe_fn=lambda _p: inherited,
    ) == ("none", "not_dispatcher_session")


def test_verify_accepts_create_time_within_epsilon_and_rejects_beyond_it():
    base = {"v": 1, "scheme": "create_time", "pid": 4242, "create_time": 1000.0}
    near = dict(base, create_time=1000.0 + kb._CREATE_TIME_EPSILON / 2)
    far = dict(base, create_time=1000.0 + kb._CREATE_TIME_EPSILON + 1)
    assert kb.verify_worker_ownership(
        4242, json.dumps(base), probe_fn=lambda _p: near) == ("pid", "ok")
    assert kb.verify_worker_ownership(
        4242, json.dumps(base), probe_fn=lambda _p: far) == (
        "none", "identity_mismatch")


def test_spawn_persists_the_birth_identity_on_task_and_run(kanban_home):
    with kb.connect() as conn:
        tid = _running_task(conn, pid=4242)
        row = _row(conn, tid)
        run = _run_row(conn, tid)
        assert json.loads(row["worker_identity"])["pid"] == 4242
        assert json.loads(run["worker_identity"])["pid"] == 4242
        assert _events(conn, tid, "spawned")[0]["identity"] is True


def test_spawn_records_a_null_identity_when_capture_fails(kanban_home, monkeypatch):
    """An uncapturable worker is recorded honestly, not guessed at — and the
    ``spawned`` event says so, so an operator knows up front that this worker
    can never be force-reclaimed."""
    monkeypatch.setattr(kb, "capture_worker_identity", lambda _pid: None)
    with kb.connect() as conn:
        tid = _running_task(conn, pid=4242)
        assert _row(conn, tid)["worker_identity"] is None
        assert _events(conn, tid, "spawned")[0]["identity"] is False


def _wedge(conn, tid, *, now=None):
    """Age the progress lease past any plausible bound, keeping liveness fresh."""
    now = now or int(time.time())
    conn.execute(
        "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
        "claim_expires = ? WHERE id = ?",
        (now - 5, now - 7200, now + 900, tid),
    )
    conn.execute(
        "UPDATE task_runs SET last_progress_at = ?, started_at = ? "
        "WHERE task_id = ?",
        (now - 7200, now - 7300, tid),
    )
    conn.commit()
    return now


def test_no_progress_refuses_to_signal_a_legacy_row_and_holds_the_claim(
    kanban_home, monkeypatch,
):
    """A board upgraded while a worker was in flight has no identity for it.
    That worker is never force-killed — and the card is not requeued either,
    because requeueing beside a live process spawns a duplicate."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET worker_identity = NULL WHERE id = ?", (tid,))
        _wedge(conn, tid)

        reclaimed = kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn,
        )

        assert reclaimed == []
        assert sent == [], "a legacy row must not be signalled at all"
        assert kb.get_task(conn, tid).status == "running"
        deferred = _events(conn, tid, "reclaim_deferred")
        assert len(deferred) == 1
        assert deferred[0]["reason"] == "ownership_unproven"
        assert deferred[0]["ownership_reason"] == "identity_absent"
        assert deferred[0]["trigger"] == "no_progress_worker_alive"
        assert deferred[0]["signalled"] is False
        assert _events(conn, tid, "no_progress") == []


def test_deferred_reclaim_keeps_the_stale_progress_lease_so_it_re_fires(
    kanban_home, monkeypatch,
):
    """Holding must not launder the evidence. If the deferral reset
    ``last_progress_at``, the next tick would see a healthy card and the whole
    situation would go silent — the operator would never learn of it."""
    _sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET worker_identity = NULL WHERE id = ?", (tid,))
        _wedge(conn, tid)
        before = _row(conn, tid)["last_progress_at"]

        for _ in range(3):
            kb.detect_no_progress_running(
                conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn,
            )

        assert _row(conn, tid)["last_progress_at"] == before
        assert len(_events(conn, tid, "reclaim_deferred")) == 3


def test_no_progress_refuses_to_signal_a_recycled_pid(kanban_home, monkeypatch):
    """The amplified hazard, directly: the recorded worker exited and its pid
    now names something else. Nothing may be signalled — least of all its
    process group."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _wedge(conn, tid)
        # Whatever holds pid 4242 now was born at a different instant.
        monkeypatch.setattr(
            kb, "capture_worker_identity",
            lambda pid: _fake_identity(pid, starttime=13371337),
        )

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn,
        ) == []
        assert sent == []
        assert kb.get_task(conn, tid).status == "running"
        deferred = _events(conn, tid, "reclaim_deferred")[0]
        assert deferred["reason"] == "ownership_unproven"
        assert deferred["ownership_reason"] == "identity_mismatch"


def test_unproven_ownership_still_reclaims_once_the_pid_is_gone(
    kanban_home, monkeypatch,
):
    """The hold is bounded: it lasts only as long as the unknown process does.
    Nothing to kill means nothing to be wrong about, so the claim releases."""
    _sent, signal_fn = _mock_worker(monkeypatch, alive=False)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET worker_identity = NULL WHERE id = ?", (tid,))
        _wedge(conn, tid)
        # A dead host-local pid is the crash path's business, not ours; drop
        # the pid so this exercises the release rather than the crash skip.
        conn.execute("UPDATE tasks SET worker_pid = NULL WHERE id = ?", (tid,))
        conn.commit()

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn,
        ) == [tid]
        assert kb.get_task(conn, tid).status in {"ready", "changes_requested"}


@pytest.mark.parametrize(
    "path",
    ["ttl", "stale", "max_runtime"],
)
def test_every_reclaim_path_refuses_an_unprovable_worker(
    kanban_home, monkeypatch, path,
):
    """The point of the fix is that it closes the CLASS, not one entry point.
    Each watchdog that can signal must decline the same way."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET worker_identity = NULL WHERE id = ?", (tid,))
        if path == "ttl":
            conn.execute(
                "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
                "WHERE id = ?", (now - 10, now - 10_000, tid),
            )
            conn.commit()
            kb.release_stale_claims(conn, signal_fn=signal_fn)
        elif path == "stale":
            conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ?, started_at = ? "
                "WHERE id = ?", (now - 10_000, now - 20_000, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
                (now - 20_000, tid),
            )
            conn.commit()
            kb.detect_stale_running(
                conn, stale_timeout_seconds=60, signal_fn=signal_fn)
        else:
            conn.execute(
                "UPDATE tasks SET max_runtime_seconds = 1, started_at = ? "
                "WHERE id = ?", (now - 10_000, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
                (now - 10_000, tid),
            )
            conn.commit()
            assert kb.enforce_max_runtime(conn, signal_fn=signal_fn) == []

        assert sent == [], f"{path} signalled an unprovable pid"
        assert kb.get_task(conn, tid).status == "running"
        deferred = _events(conn, tid, "reclaim_deferred")
        assert deferred and deferred[0]["reason"] == "ownership_unproven"


@pytest.mark.parametrize("path", ["ttl", "stale", "max_runtime"])
def test_every_reclaim_path_still_terminates_a_proven_worker(
    kanban_home, monkeypatch, path,
):
    """The mirror image: proof present, so the reclaim proceeds exactly as
    before. A guard that only ever refuses is not a guard, it is an outage."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        if path == "ttl":
            conn.execute(
                "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
                "WHERE id = ?", (now - 10, now - 10_000, tid),
            )
            conn.commit()
            assert kb.release_stale_claims(conn, signal_fn=signal_fn) == 1
        elif path == "stale":
            conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ?, started_at = ? "
                "WHERE id = ?", (now - 10_000, now - 20_000, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
                (now - 20_000, tid),
            )
            conn.commit()
            assert kb.detect_stale_running(
                conn, stale_timeout_seconds=60, signal_fn=signal_fn) == [tid]
        else:
            conn.execute(
                "UPDATE tasks SET max_runtime_seconds = 1, started_at = ? "
                "WHERE id = ?", (now - 10_000, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
                (now - 10_000, tid),
            )
            conn.commit()
            assert kb.enforce_max_runtime(conn, signal_fn=signal_fn) == [tid]

        assert sent, f"{path} failed to terminate a proven worker"
        assert kb.get_task(conn, tid).status != "running"
        assert _events(conn, tid, "reclaim_deferred") == []


def test_timed_out_receipt_records_the_ownership_verdict(kanban_home, monkeypatch):
    _sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = 1, started_at = ? "
            "WHERE id = ?", (now - 10_000, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (now - 10_000, tid),
        )
        conn.commit()
        kb.enforce_max_runtime(conn, signal_fn=signal_fn)

        payload = _events(conn, tid, "timed_out")[0]
        assert payload["ownership"] == "proven"
        assert payload["ownership_reason"] == "ok"
        assert payload["signalled"] is True


def test_remote_claims_are_never_signalled(kanban_home, monkeypatch):
    """A pid recorded by another host names a process on THAT host. Signalling
    it here means signalling whatever happens to hold the number locally."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET claim_lock = ? WHERE id = ?",
            ("some-other-host:worker", tid),
        )
        _wedge(conn, tid)

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn,
        ) == [tid]
        assert sent == []
        payload = _events(conn, tid, "no_progress")[0]
        assert payload["worker_state"] == "remote"
        assert payload["ownership_reason"] == "remote_claim"
        # Remote is not an ownership failure — we simply cannot manage it —
        # so the claim is released rather than held forever.
        assert _events(conn, tid, "reclaim_deferred") == []


def test_reclaiming_clears_the_identity_with_the_pid(kanban_home, monkeypatch):
    """A released claim must not leave a previous worker's proof behind for the
    next one to inherit."""
    _sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _wedge(conn, tid)
        kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700, signal_fn=signal_fn)
        row = _row(conn, tid)
        assert row["worker_pid"] is None
        assert row["worker_identity"] is None


def test_operator_reclaim_releases_but_reports_that_it_could_not_kill(
    kanban_home, monkeypatch,
):
    """Operator-driven reclaim keeps its escape hatch — the card unsticks —
    but it still never signals an unprovable pid, and the receipt says so."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET worker_identity = NULL WHERE id = ?", (tid,))
        conn.commit()

        assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=signal_fn)
        assert sent == []
        assert kb.get_task(conn, tid).status != "running"
        payload = _events(conn, tid, "reclaimed")[-1]
        assert payload["ownership"] == "unproven"
        assert payload["ownership_reason"] == "identity_absent"


# ---------------------------------------------------------------------------
# The escalation window: the proof expires between SIGTERM and SIGKILL
# ---------------------------------------------------------------------------
# The ownership proof taken before the SIGTERM is five seconds old by the time
# the SIGKILL is due, and "still alive after SIGTERM" has two explanations of
# very different likelihood. A worker that ignores SIGTERM is rare. A worker
# that obeys it promptly and has its pid handed to something else inside the
# grace window is ordinary — that is exactly what a busy host does with a
# freed pid. ``_pid_alive`` cannot tell them apart: it asks about a NUMBER.
#
# So the escalation is the single most dangerous signal this file sends, and
# it is the one that used to be sent on the oldest evidence.

def _terminate_with_recycled_escalation(monkeypatch, *, recycle: bool):
    """Drive ``_terminate_reclaimed_worker`` through the grace window.

    The pid stays "alive" throughout (that is what makes the escalation fire).
    When ``recycle`` is True the birth record flips the moment the SIGTERM is
    delivered, modelling the worker exiting and its number being reused;
    when it is False the same process is still there, ignoring the signal.

    Returns ``(receipt, per_pid_signals, group_signals)``.
    """
    import signal as _signal

    pid = 4242
    ours = _fake_identity(pid, starttime=111)
    replacement = _fake_identity(pid, starttime=222)  # born later, own leader
    live = {"identity": ours}

    sent: list = []
    groups: list = []

    def _die_or_recycle():
        if recycle:
            live["identity"] = replacement

    def _signal_fn(p, sig):
        sent.append((int(p), sig))
        _die_or_recycle()

    def _killpg_fn(pgid, sig):
        groups.append((int(pgid), sig))
        _die_or_recycle()

    monkeypatch.setattr(kb, "_pid_alive", lambda _p: True)
    monkeypatch.setattr(kb.time, "sleep", lambda _s: None)
    # The synthetic pid has no /proc entry, so the real resolver would answer
    # None and quietly downgrade the test to per-pid semantics — hiding the
    # very thing being asserted (that no *group* is killed either).
    monkeypatch.setattr(kb, "_dispatcher_owned_pgid", lambda p: int(p))

    receipt = kb._terminate_reclaimed_worker(
        pid,
        f"{kb._claimer_id().split(':', 1)[0]}:worker",
        stored_identity=json.dumps(ours),
        signal_fn=_signal_fn,
        killpg_fn=_killpg_fn,
        probe_fn=lambda _p: live["identity"],
    )
    assert _signal.SIGTERM in [s for _p, s in groups + sent], (
        "the SIGTERM half must still have been delivered"
    )
    return receipt, sent, groups


def test_escalation_reverifies_and_never_sigkills_a_replacement_process(
    kanban_home, monkeypatch,
):
    """The worker dies on the SIGTERM and its pid is immediately reused.

    Nothing observable changes: the number is still alive, the claim is still
    ours, the grace window still elapses. Only the birth record moves — and it
    is the only thing that can tell us the process we are about to SIGKILL is
    a stranger's. Zero further signals may leave this function.
    """
    import signal as _signal

    receipt, sent, groups = _terminate_with_recycled_escalation(
        monkeypatch, recycle=True)

    escalations = [
        (p, sig) for p, sig in sent + groups if sig != _signal.SIGTERM
    ]
    assert escalations == [], (
        f"a SIGKILL reached the replacement process: {escalations}"
    )
    assert groups == [(4242, _signal.SIGTERM)], (
        "exactly one group signal — the authorised SIGTERM — and no more"
    )
    assert sent == [], "no per-pid signal either"
    assert receipt["sigkill"] is False
    assert receipt["escalation_ownership_unproven"] is True
    assert receipt["escalation_ownership_reason"] == "identity_mismatch"
    assert receipt["ownership_unproven"] is True
    assert receipt["ownership_reason"] == "escalation_ownership_unproven"
    assert receipt["terminated"] is False
    # And the claim is held, under a reason that names the stage that refused.
    assert kb._reclaim_hold_reason(receipt) == "escalation_ownership_unproven"


def test_escalation_still_sigkills_a_worker_that_is_provably_still_ours(
    kanban_home, monkeypatch,
):
    """The counter-case, without which the test above is satisfied by simply
    never escalating. Same pid, same aliveness, same grace window — but the
    birth record still verifies, so the SIGKILL goes to the whole group."""
    import signal as _signal

    receipt, sent, groups = _terminate_with_recycled_escalation(
        monkeypatch, recycle=False)

    assert groups == [
        (4242, _signal.SIGTERM), (4242, _signal.SIGKILL),
    ], "a proven worker that ignores SIGTERM must still be killed"
    assert sent == []
    assert receipt["sigkill"] is True
    assert receipt["escalation_ownership_unproven"] is False
    assert receipt["escalation_ownership_reason"] is None
    # It survived the SIGKILL in this fake, which is the other hold reason.
    assert kb._reclaim_hold_reason(receipt) == "worker_alive"


def test_escalation_reresolves_the_group_rather_than_reusing_it(
    kanban_home, monkeypatch,
):
    """The pgid resolved before the SIGTERM is as stale as the proof was.

    Ownership can hold at escalation while the GROUP has stopped being ours —
    the pid is still our worker but no longer leads a group we may signal.
    Reusing the earlier resolution would aim a SIGKILL at a group on the
    strength of a five-second-old reading, so the group is resolved again
    from the fresh verdict and the signal falls back to the proven pid alone.
    """
    import signal as _signal

    pid = 4242
    ours = _fake_identity(pid)
    resolutions: list = []
    groups: list = []
    sent: list = []

    def _resolve(p):
        resolutions.append(int(p))
        # Leads its own group for the SIGTERM; no longer resolvable after it.
        return int(p) if len(resolutions) == 1 else None

    monkeypatch.setattr(kb, "_pid_alive", lambda _p: True)
    monkeypatch.setattr(kb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(kb, "_dispatcher_owned_pgid", _resolve)

    receipt = kb._terminate_reclaimed_worker(
        pid,
        f"{kb._claimer_id().split(':', 1)[0]}:worker",
        stored_identity=json.dumps(ours),
        signal_fn=lambda p, sig: sent.append((int(p), sig)),
        killpg_fn=lambda g, sig: groups.append((int(g), sig)),
        probe_fn=lambda _p: ours,
    )
    assert len(resolutions) == 2, "the group must be resolved once per signal"
    assert groups == [(pid, _signal.SIGTERM)]
    assert sent == [(pid, _signal.SIGKILL)], (
        "the escalation must follow the fresh resolution, not the stale one"
    )
    assert receipt["sigkill"] is True
    assert receipt["process_group"] is True, (
        "the SIGTERM did reach a group, and the receipt keeps saying so"
    )


def test_escalation_refuses_when_the_pid_stopped_being_a_session_leader(
    kanban_home, monkeypatch,
):
    """A birth record that no longer matches at escalation is refused however
    it stopped matching — here the live sid/pgid have moved, which is what a
    reused pid inside another session looks like."""
    pid = 4242
    ours = _fake_identity(pid)
    stranger = _fake_identity(pid, sid=999, pgid=999)
    live = {"identity": ours}
    sent: list = []
    groups: list = []

    monkeypatch.setattr(kb, "_pid_alive", lambda _p: True)
    monkeypatch.setattr(kb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(kb, "_dispatcher_owned_pgid", lambda p: int(p))

    def _killpg_fn(pgid, sig):
        groups.append((int(pgid), sig))
        live["identity"] = stranger

    receipt = kb._terminate_reclaimed_worker(
        pid,
        f"{kb._claimer_id().split(':', 1)[0]}:worker",
        stored_identity=json.dumps(ours),
        signal_fn=lambda p, sig: sent.append((int(p), sig)),
        killpg_fn=_killpg_fn,
        probe_fn=lambda _p: live["identity"],
    )
    assert len(groups) == 1, "no second group signal"
    assert sent == [], "and no per-pid fallback either — none means none"
    assert receipt["escalation_ownership_unproven"] is True
    assert receipt["sigkill"] is False


def test_a_refused_escalation_holds_the_card_and_re_fires_next_tick(
    kanban_home, monkeypatch,
):
    """End to end through the reclaim path: refusing the SIGKILL must not
    quietly requeue. Releasing the claim beside a live unknown process is the
    duplicate-worker bug the refusal exists to avoid, and clearing the
    progress lease would reset the clock that detected it."""
    ours = _fake_identity(4242, starttime=111)
    replacement = _fake_identity(4242, starttime=222)
    live = {"identity": ours}

    monkeypatch.setattr(kb, "_pid_alive", lambda _p: True)
    monkeypatch.setattr(kb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        kb, "capture_worker_identity", lambda _p: live["identity"])

    def _signal_fn(_p, _sig):
        live["identity"] = replacement

    with kb.connect() as conn:
        tid = _running_task(conn, pid=4242)
        conn.execute(
            "UPDATE tasks SET worker_identity = ? WHERE id = ?",
            (json.dumps(ours), tid),
        )
        conn.commit()
        wedged_at = _wedge(conn, tid)

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=_signal_fn,
        ) == []
        assert kb.get_task(conn, tid).status == "running"

        deferred = _events(conn, tid, "reclaim_deferred")
        assert len(deferred) == 1
        assert deferred[0]["reason"] == "escalation_ownership_unproven"
        assert deferred[0]["trigger"] == "no_progress_worker_alive"
        assert deferred[0]["escalation_ownership_reason"] == "identity_mismatch"
        assert deferred[0]["sigkill"] is False

        # The stale progress lease survives, so the next tick re-detects and
        # re-reports rather than going silent.
        assert _row(conn, tid)["last_progress_at"] == wedged_at - 7200
        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=_signal_fn,
        ) == []
        assert len(_events(conn, tid, "reclaim_deferred")) == 2


# ---------------------------------------------------------------------------
# Real processes: the stale-PID process-group hazard, and its absence
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(os, "killpg") or not hasattr(os, "setsid")
    or not os.path.exists("/proc"),
    reason="Linux process groups + /proc only",
)
def test_a_recycled_pid_s_process_tree_is_never_signalled(kanban_home, monkeypatch):
    """The hazard, with real processes.

    Stand up an unrelated three-process tree in its own session — exactly what
    an operator's ``setsid``-launched job, another agent's dispatcher, or a
    freshly-recycled PID looks like — and tell the board that its worker is
    that tree's leader. Every reclaim path must send ZERO signals and all
    three processes must survive.

    Without the ownership proof this is a whole-tree kill: the pid leads its
    own group, ``getpgid(pid) == pid`` holds, and the old code took that as
    permission.
    """
    import subprocess
    import sys

    monkeypatch.setattr(
        kb, "capture_worker_identity", _REAL_CAPTURE_WORKER_IDENTITY)
    pidfile = kanban_home / "tree.pids"
    code = (
        "import subprocess, sys, time, pathlib\n"
        "kids = [subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(300)']) for _ in range(2)]\n"
        f"pathlib.Path({str(pidfile)!r}).write_text("
        "','.join(str(k.pid) for k in kids))\n"
        "time.sleep(300)\n"
    )
    stranger = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True,
    )
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            time.sleep(0.1)
        kids = [int(x) for x in pidfile.read_text().strip().split(",")]
        tree = [stranger.pid, *kids]
        assert len(tree) == 3
        assert os.getpgid(stranger.pid) == stranger.pid, "must lead its group"
        for k in kids:
            assert os.getpgid(k) == stranger.pid

        sent: list = []

        def _signal_fn(pid, sig):  # pragma: no cover - must never run
            sent.append((pid, sig))
            os.kill(pid, sig)

        host = kb._claimer_id().split(":", 1)[0]
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.claim_task(conn, tid, claimer=f"{host}:worker")
            # The board's record: our worker's pid, but the birth identity of
            # the process that pid USED to name. This is a PID reuse, exactly.
            kb._set_worker_pid(conn, tid, stranger.pid)
            stale = kb.capture_worker_identity(stranger.pid)
            stale["starttime"] = int(stale["starttime"]) - 5000
            conn.execute(
                "UPDATE tasks SET worker_identity = ? WHERE id = ?",
                (json.dumps(stale), tid),
            )
            now = int(time.time())
            conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
                "claim_expires = ?, started_at = ?, max_runtime_seconds = 1 "
                "WHERE id = ?",
                (now - 5, now - 7200, now - 10, now - 20_000, tid),
            )
            conn.execute(
                "UPDATE task_runs SET last_progress_at = ?, started_at = ? "
                "WHERE task_id = ?",
                (now - 7200, now - 20_000, tid),
            )
            conn.commit()

            kb.detect_no_progress_running(
                conn, no_progress_timeout_seconds=2700, signal_fn=_signal_fn)
            kb.detect_stale_running(
                conn, stale_timeout_seconds=60, signal_fn=_signal_fn)
            kb.enforce_max_runtime(conn, signal_fn=_signal_fn)
            kb.release_stale_claims(conn, signal_fn=_signal_fn)

            assert sent == [], "a reclaim signalled an unrelated process tree"
            # The property that matters: all three are still running.
            for pid in tree:
                os.kill(pid, 0)
            assert stranger.poll() is None
            assert kb.get_task(conn, tid).status == "running"
            deferred = _events(conn, tid, "reclaim_deferred")
            assert deferred, "the refusal must be visible, not silent"
            assert all(d["reason"] == "ownership_unproven" for d in deferred)
    finally:
        try:
            os.killpg(stranger.pid, 9)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            stranger.wait(timeout=5)
        except Exception:
            pass


@pytest.mark.skipif(
    not hasattr(os, "killpg") or not hasattr(os, "setsid")
    or not os.path.exists("/proc"),
    reason="Linux process groups + /proc only",
)
def test_a_genuinely_owned_process_tree_is_still_fully_reaped(kanban_home, monkeypatch):
    """Same shape, honest identity: the whole tree must go. This is the half
    of the contract that an over-eager guard would silently break."""
    import subprocess
    import sys

    monkeypatch.setattr(
        kb, "capture_worker_identity", _REAL_CAPTURE_WORKER_IDENTITY)
    pidfile = kanban_home / "owned.pids"
    code = (
        "import subprocess, sys, time, pathlib\n"
        "kids = [subprocess.Popen([sys.executable, '-c',"
        " 'import time; time.sleep(300)']) for _ in range(2)]\n"
        f"pathlib.Path({str(pidfile)!r}).write_text("
        "','.join(str(k.pid) for k in kids))\n"
        "time.sleep(300)\n"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True,
    )
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            time.sleep(0.1)
        kids = [int(x) for x in pidfile.read_text().strip().split(",")]
        tree = [worker.pid, *kids]

        host = kb._claimer_id().split(":", 1)[0]
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.claim_task(conn, tid, claimer=f"{host}:worker")
            kb._set_worker_pid(conn, tid, worker.pid)  # real capture
            now = int(time.time())
            conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
                "claim_expires = ? WHERE id = ?",
                (now - 5, now - 7200, now + 900, tid),
            )
            conn.execute(
                "UPDATE task_runs SET last_progress_at = ?, started_at = ? "
                "WHERE task_id = ?", (now - 7200, now - 7300, tid),
            )
            conn.commit()

            assert kb.detect_no_progress_running(
                conn, no_progress_timeout_seconds=2700) == [tid]

        worker.wait(timeout=15)
        for pid in tree:
            for _ in range(150):
                try:
                    os.kill(pid, 0)
                except (ProcessLookupError, PermissionError):
                    break
                time.sleep(0.1)
            else:  # pragma: no cover
                pytest.fail(f"pid {pid} survived a reclaim of its own group")
    finally:
        try:
            os.killpg(worker.pid, 9)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            worker.wait(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Schema: additive migration + rebuilt-table parity
# ---------------------------------------------------------------------------

def test_fresh_schema_carries_the_identity_columns(kanban_home):
    with kb.connect() as conn:
        assert "worker_identity" in _cols(conn, "tasks")
        assert "worker_identity" in _cols(conn, "task_runs")


def test_migration_adds_the_identity_columns_without_backfilling(kanban_home):
    """Additive and NULL — never invented. There is no way to reconstruct the
    birth record of a process that was already running at upgrade time, and a
    guessed one is exactly the false proof the column exists to prevent."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute("ALTER TABLE tasks DROP COLUMN worker_identity")
        conn.execute("ALTER TABLE task_runs DROP COLUMN worker_identity")
        conn.commit()
        assert "worker_identity" not in _cols(conn, "tasks")

    kb._INITIALIZED_PATHS.clear()
    with kb.connect() as conn:
        assert "worker_identity" in _cols(conn, "tasks")
        assert "worker_identity" in _cols(conn, "task_runs")
        row = _row(conn, tid)
        assert row is not None, "migration must not lose rows"
        assert row["status"] == "running", "the in-flight claim must survive"
        assert row["worker_identity"] is None
        # And a NULL identity is unprovable, so that worker is never signalled.
        assert kb.verify_worker_ownership(
            row["worker_pid"], row["worker_identity"],
        ) == ("none", "identity_absent")


def test_missing_identity_column_degrades_to_unproven_not_a_crash(kanban_home):
    """A board opened by a sibling process on an older build has no column.
    Reading it must answer "unproven", never raise into the dispatch tick."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute("ALTER TABLE tasks DROP COLUMN worker_identity")
        conn.commit()
        assert kb._stored_worker_identity(conn, tid) is None
        assert kb._stored_worker_identity(conn, "no-such-task") is None
        assert kb._stored_worker_identity(conn, None) is None


def test_no_progress_pass_degrades_when_the_column_is_missing(
    kanban_home, monkeypatch,
):
    """A board opened by a sibling process on an older build has no
    ``last_progress_at``. This pass is one of several in a dispatch tick, so
    raising here would take the crash, stale and TTL passes down with it and
    turn a version skew into a dispatcher outage. Answer "nothing to say"."""
    sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _wedge(conn, tid)
        conn.execute("ALTER TABLE tasks DROP COLUMN last_progress_at")
        conn.commit()

        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, signal_fn=signal_fn,
        ) == []
        assert sent == [], "and certainly no worker is signalled"
        assert kb.get_task(conn, tid).status == "running"


def test_no_progress_pass_does_not_swallow_a_real_database_error(
    kanban_home, monkeypatch,
):
    """The degradation above is scoped to a missing column. A locked or
    corrupt database is a different problem and must still surface."""
    import sqlite3 as _sqlite3

    with kb.connect() as conn:
        _running_task(conn)

        def _boom(*_a, **_kw):
            raise _sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(conn, "execute", _boom)
        with pytest.raises(_sqlite3.OperationalError):
            kb.detect_no_progress_running(
                conn, no_progress_timeout_seconds=60,
                signal_fn=lambda *_a: None,
            )


def test_no_progress_auto_blocks_reach_the_caller_without_a_side_channel(
    kanban_home, monkeypatch,
):
    """Trips are collected through a caller-owned list, not an attribute on
    the function object. One shared slot per process means two concurrent
    ticks — gateway watcher plus a CLI dispatch, or two boards — overwrite
    each other and one board's auto-block can be reported on another's
    result."""
    _sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 4 WHERE id = ?", (tid,))
        _wedge(conn, tid)

        out: list = []
        assert kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=60, failure_limit=5,
            auto_blocked_out=out, signal_fn=signal_fn,
        ) == [tid]
        assert out == [tid]
        assert kb.get_task(conn, tid).status == "blocked"
        assert not hasattr(kb.detect_no_progress_running, "_last_auto_blocked")


def test_max_runtime_uses_the_configured_failure_limit(kanban_home, monkeypatch):
    """``enforce_max_runtime`` shares the ``consecutive_failures`` counter
    with every other recovery pass, so it has to share the threshold too.
    Reading ``DEFAULT_FAILURE_LIMIT`` here gave a board that lowered
    ``kanban.failure_limit`` a different breaker for timeouts than for
    crashes — from the same counter."""
    _sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = 1, started_at = ?, "
            "consecutive_failures = 1 WHERE id = ?", (now - 10_000, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (now - 10_000, tid),
        )
        conn.commit()

        assert kb.enforce_max_runtime(
            conn, failure_limit=2, signal_fn=signal_fn) == [tid]
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 2
        assert task.status == "blocked", (
            "the configured limit of 2 must trip here, not DEFAULT_FAILURE_LIMIT"
        )


def test_max_runtime_below_the_configured_limit_still_retries(
    kanban_home, monkeypatch,
):
    """Counter-case: the same timeout under a higher limit must requeue."""
    _sent, signal_fn = _mock_worker(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = 1, started_at = ?, "
            "consecutive_failures = 1 WHERE id = ?", (now - 10_000, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (now - 10_000, tid),
        )
        conn.commit()

        assert kb.enforce_max_runtime(
            conn, failure_limit=9, signal_fn=signal_fn) == [tid]
        assert kb.get_task(conn, tid).status != "blocked"


def test_dispatch_tick_threads_the_failure_limit_into_the_timeout_pass(
    kanban_home, monkeypatch,
):
    """And the tick actually passes it — the wiring, not just the parameter."""
    seen: dict = {}
    real = kb.enforce_max_runtime

    def _spy(conn, **kw):
        seen.update(kw)
        return real(conn, **kw)

    monkeypatch.setattr(kb, "enforce_max_runtime", _spy)
    with kb.connect() as conn:
        kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None, failure_limit=7,
        )
    assert seen.get("failure_limit") == 7


def test_rebuilt_task_runs_matches_the_fresh_schema(kanban_home):
    """``_REBUILD_SPECS`` is a second, hand-maintained copy of the table
    definitions, so it can silently drift from ``SCHEMA_SQL`` — a rebuilt board
    would then be missing exactly the column a new feature just added. Compare
    the two directly."""
    with kb.connect() as conn:
        fresh = [
            (c["name"], (c["type"] or "").upper(), c["notnull"], c["pk"])
            for c in conn.execute("PRAGMA table_info(task_runs)")
        ]
    assert ("worker_identity", "TEXT", 0, 0) in fresh

    import sqlite3 as _sqlite3
    probe = _sqlite3.connect(":memory:")
    probe.row_factory = _sqlite3.Row
    for table, (create_sql, index_sqls) in kb._REBUILD_SPECS.items():
        probe.execute(create_sql)
        for idx in index_sqls:
            probe.execute(idx)
    rebuilt = [
        (c["name"], (c["type"] or "").upper(), c["notnull"], c["pk"])
        for c in probe.execute("PRAGMA table_info(task_runs)")
    ]
    probe.close()
    assert rebuilt == fresh, (
        "_REBUILD_SPECS['task_runs'] has drifted from SCHEMA_SQL"
    )


def test_a_drifted_board_is_rebuilt_with_the_identity_column(kanban_home):
    """End-to-end of the above: a legacy (TEXT-id) ``task_runs`` is rebuilt on
    connect, and the rebuild must carry the new column rather than dropping it
    back to the pre-feature shape."""
    with kb.connect() as conn:
        conn.execute("DROP TABLE task_runs")
        conn.execute(
            "CREATE TABLE task_runs ("
            " id TEXT PRIMARY KEY, task_id TEXT NOT NULL, profile TEXT,"
            " step_key TEXT, status TEXT NOT NULL, claim_lock TEXT,"
            " claim_expires INTEGER, worker_pid INTEGER,"
            " max_runtime_seconds INTEGER, last_heartbeat_at INTEGER,"
            " started_at INTEGER NOT NULL, ended_at INTEGER, outcome TEXT,"
            " summary TEXT, metadata TEXT, error TEXT)"
        )
        conn.execute(
            "INSERT INTO task_runs (id, task_id, status, started_at) "
            "VALUES ('legacy-1', 'T1', 'done', 1)"
        )
        conn.commit()
        assert kb._table_has_drifted(conn, "task_runs")

    kb._INITIALIZED_PATHS.clear()
    with kb.connect() as conn:
        assert not kb._table_has_drifted(conn, "task_runs")
        assert "worker_identity" in _cols(conn, "task_runs")
        assert "last_progress_at" in _cols(conn, "task_runs")
        kept = conn.execute(
            "SELECT task_id FROM task_runs WHERE task_id = 'T1'").fetchone()
        assert kept is not None, "the rebuild must preserve rows"


# ---------------------------------------------------------------------------
# Timeout resolver: non-finite values
# ---------------------------------------------------------------------------

def test_resolver_rejects_infinity_without_raising():
    """``.inf`` is a real YAML scalar and ``int(float("inf"))`` raises
    OverflowError — which is NOT a ValueError. Uncaught it would escape a
    config read and take down the dispatcher tick instead of falling back."""
    default = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    for value in (float("inf"), float("-inf"), float("nan")):
        assert kb.resolve_no_progress_timeout_seconds(value) == default


def test_resolver_rejects_a_non_finite_decimal():
    from decimal import Decimal
    default = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    assert kb.resolve_no_progress_timeout_seconds(Decimal("Infinity")) == default
    assert kb.resolve_no_progress_timeout_seconds(Decimal("NaN")) == default


# ---------------------------------------------------------------------------
# record_progress: the allowlist is a refusal, never an exception
# ---------------------------------------------------------------------------

def test_record_progress_refuses_an_unhashable_source(kanban_home):
    """``source in frozenset`` RAISES on an unhashable value rather than
    returning False, so the "refuse anything not on the allowlist" boundary
    would have thrown into the tool middleware instead of refusing."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        for bad in ([], {}, set(), ["tool_invocation"], {"source": "x"}):
            receipt = kb.record_progress(conn, tid, source=bad)
            assert receipt["recorded"] is False
            assert receipt["reason"] == "unsupported_source"
        # And nothing was written.
        assert _row(conn, tid)["last_progress_at"] is not None


def test_record_progress_refuses_unhashable_source_without_touching_the_lease(
    kanban_home,
):
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET last_progress_at = 1000 WHERE id = ?", (tid,))
        conn.commit()
        assert kb.record_progress(conn, tid, source=["x"])["recorded"] is False
        assert _row(conn, tid)["last_progress_at"] == 1000


# ---------------------------------------------------------------------------
# Human board transitions renew the progress lease (documented behaviour)
# ---------------------------------------------------------------------------

def test_a_human_comment_extends_the_workers_progress_lease(kanban_home):
    """Progress is renewed by board state transitions, and the transition
    table does not record who caused them — so an operator commenting from the
    dashboard renews a running worker's lease exactly as the worker's own
    comment would.

    This is a real hole in the guard and it is the deliberate trade documented
    in kanban.md: the dangerous direction (a worker renewing its own lease from
    text it authored) stays closed regardless. Pinned here so the behaviour is
    a decision rather than a surprise.
    """
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET last_progress_at = 1000 WHERE id = ?", (tid,))
        conn.commit()

        kb.add_comment(conn, tid, author="a-human", body="still looking at this")

        assert _row(conn, tid)["last_progress_at"] > 1000


def test_a_recycled_pid_does_not_hold_a_card_running_forever(
    kanban_home, monkeypatch,
):
    """The other half of refusing to signal an unprovable pid.

    Once reclaim declines to kill a recycled pid, something has to notice the
    real worker is gone — otherwise the card sits ``running`` forever emitting
    ``reclaim_deferred`` every tick. Crash detection asks the identity, not
    just the pid number: a contradicted birth stamp means our worker exited,
    however alive that integer looks.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 10_000, tid),
        )
        conn.commit()
        # Same pid, different process.
        monkeypatch.setattr(
            kb, "capture_worker_identity",
            lambda pid: _fake_identity(pid, starttime=999_999),
        )

        assert kb.detect_crashed_workers(conn) == [tid]
        payload = _events(conn, tid, "crashed")[0]
        assert payload["pid_recycled"] is True
        assert kb.get_task(conn, tid).status != "running"


def test_a_live_worker_with_a_matching_identity_is_not_called_crashed(
    kanban_home, monkeypatch,
):
    """The mirror: a genuinely running worker must never be declared crashed
    because the crash pass learned to doubt pids."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 10_000, tid),
        )
        conn.commit()
        assert kb.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, tid).status == "running"


def test_a_legacy_row_with_a_live_pid_is_not_called_crashed(
    kanban_home, monkeypatch,
):
    """Absent is *unknown*, not gone. A pre-upgrade row must not be requeued
    beside a worker that is genuinely still running just because we cannot
    prove anything about it."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET worker_identity = NULL, started_at = ? "
            "WHERE id = ?", (int(time.time()) - 10_000, tid),
        )
        conn.commit()
        assert kb.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, tid).status == "running"


def test_reconcile_does_not_defer_forever_on_a_recycled_pid(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET claim_expires = NULL WHERE id = ?", (tid,))
        conn.commit()
        # Matching identity: a real live worker, so the orphan is held.
        assert kb.reconcile_orphaned_running(conn) == []

        monkeypatch.setattr(
            kb, "capture_worker_identity",
            lambda pid: _fake_identity(pid, starttime=999_999),
        )
        assert kb.reconcile_orphaned_running(conn) == [tid]
