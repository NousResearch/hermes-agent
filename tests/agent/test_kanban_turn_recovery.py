"""Tests for kanban worker in-place turn recovery and the shared exit policy.

Unit contracts first (retry authority, the live run/claim proof, the recovery
loop), then call-site behaviour, then the goal-mode veto as one table.

Under the unified gateway runtime a dispatcher-spawned worker no longer runs the
agent: ``hermes_cli.kanban_worker_client`` submits the claim to the profile owner,
and the owner runs the turns in ``gateway.session_kanban.run_worker_turns`` against
the bound frame's db/run/claim carrier. The call-site tests therefore drive that
real owner seam (real board, real claim, real lease proof) and read the outcome the
dispatcher books — ``worker_exit_code`` — plus the client's exit/summary half. The
exit mapping itself is ``hermes_cli.turn_exit.turn_exit_code`` (pinned row by row by
``tests/gateway/test_kanban_result_exit.py``); the in-process one-shot mapper keeps
its stripped worker predicate pinned here.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from agent.kanban_turn_recovery import (
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    RECOVERY_DELAYS_SECONDS,
    build_recovery_nudge,
    kanban_task_id,
    kanban_turn_recovery_enabled,
    max_recovery_attempts,
    recover_failed_kanban_turns,
    recovery_delay_seconds,
    should_recover_turn,
    worker_claim_is_live,
)
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

KANBAN_ENV = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_GOAL_MODE",
    "HERMES_KANBAN_TURN_RECOVERY",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
)

#: Any "live" lease for these tests (2100-01-01 UTC).
_FAR_FUTURE_EXPIRY = 4_102_444_800


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in KANBAN_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _failed(*, retryable: bool = True, reason: str = "timeout",
            error: str = "peer closed connection") -> dict:
    return {"failed": True, "failure_retryable": retryable, "failure_reason": reason,
            "error": error, "final_response": f"API call failed after 3 retries: {error}",
            "completed": False, "messages": []}


def _success() -> dict:
    return {"failed": False, "completed": True, "final_response": "done", "messages": []}


def _interrupted() -> dict:
    """``agent/turn_recovery.py::abort_turn_on_interrupt`` already persisted and cleared
    the interrupt — re-entering the model would resurrect cancelled work."""
    return {**_failed(), "interrupted": True, "api_calls": 4}


def _settled() -> dict:
    """#87096: the bounded finalizer recorded a durable ``timed_out`` and released the
    claim while the result still reads ``completed=False`` — and here it ALSO carries a
    retryable failure stamp (the loop exhausted its provider retries, then hit the
    iteration cap). The settlement veto must win over the eligible-looking stamp."""
    return {**_failed(retryable=True), "turn_exit_reason": "max_iterations_reached(60/60)"}


def _partial() -> dict:
    """An outcome, not retry authority: truncation repair belongs in the conversation
    loop (#89289)."""
    return {"partial": True, "completed": False, "final_response": "cut off",
            "error": "truncated", "messages": []}


# ── enablement / budget ──────────────────────────────────────────────


def test_disabled_without_kanban_task(clear_kanban_env):
    assert kanban_task_id() is None
    assert kanban_turn_recovery_enabled() is False
    calls: list[str] = []
    attempts = recover_failed_kanban_turns(
        lambda nudge: calls.append(nudge), lambda: _failed(), sleep_fn=lambda s: None
    )
    assert attempts == 0
    assert calls == []


def test_env_zero_disables_and_blank_ids_are_not_workers(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "0")
    assert kanban_turn_recovery_enabled() is False
    assert should_recover_turn(_failed(), attempt=0) is False

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "   ")
    assert kanban_task_id() is None  # whitespace-only is NOT a worker, anywhere
    assert kanban_turn_recovery_enabled() is False
    clear_kanban_env.delenv("HERMES_KANBAN_TURN_RECOVERY", raising=False)
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "  t_probe  ")
    assert kanban_task_id() == "t_probe"  # padded ids are the same worker
    assert kanban_turn_recovery_enabled() is True


def test_max_attempts_parsing_and_clamp(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert max_recovery_attempts() == DEFAULT_MAX_RECOVERY_ATTEMPTS  # unset
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    assert max_recovery_attempts() == 2
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "99")
    assert max_recovery_attempts() == 10  # clamped
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "abc")
    assert max_recovery_attempts() == DEFAULT_MAX_RECOVERY_ATTEMPTS
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "false")
    assert max_recovery_attempts() == 0


def test_delay_schedule_repeats_last_entry():
    assert RECOVERY_DELAYS_SECONDS == (15.0, 45.0, 90.0)
    assert recovery_delay_seconds(1) == 15.0
    assert recovery_delay_seconds(3) == 90.0
    assert recovery_delay_seconds(9) == 90.0
    assert recovery_delay_seconds(0) == 15.0


# ── retry authority (the unit contract behind both drivers) ──────────


def test_retryable_failed_turn_is_eligible(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(reason="timeout"), attempt=0) is True
    assert should_recover_turn(_failed(reason="timeout"), attempt=2) is True  # budget 3
    assert should_recover_turn(_failed(reason="timeout"), attempt=3) is False


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_failed(retryable=False, reason="auth"), id="not-retryable"),
        pytest.param(_success(), id="success"),
        pytest.param(None, id="nothing-settled"),
        pytest.param(_failed(reason="rate_limit"), id="quota-wall"),
        pytest.param(_failed(reason="billing"), id="billing-wall"),
        pytest.param(_failed(reason="upstream_rate_limit"), id="upstream-quota-wall"),
        pytest.param(_interrupted(), id="interrupt"),
        pytest.param(_settled(), id="terminal-settlement"),
        pytest.param(_partial(), id="unfinished-nonfailed"),
        pytest.param({"completed": False, "final_response": "x"}, id="completed-false"),
    ],
)
def test_should_recover_turn_refuses(result, clear_kanban_env):
    """Nothing but a retryable failed turn is retry authority. The aggregator's upstream
    429 is the same quota-class wall as a direct one — the dispatcher owns the cooldown."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(result, attempt=0) is False


# ── live run/claim proof (the adversarial boundaries) ────────────────


def _make_board(tmp_path, *, status="running", task_pid=None, run_id=1, lock="lk",
                run_ended=None, run_pid=None, task_expires=_FAR_FUTURE_EXPIRY,
                run_expires=_FAR_FUTURE_EXPIRY):
    """A synthetic board whose task carries a LIVE, unexpired task+run lease by default
    (current-main dispatcher shape: ``claim_task`` sets ``claim_expires`` on both rows
    and ``heartbeat_claim`` mirrors the extension)."""
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, worker_pid INTEGER,"
        " current_run_id INTEGER, claim_lock TEXT, claim_expires INTEGER);"
        "CREATE TABLE task_runs (id INTEGER PRIMARY KEY, ended_at INTEGER,"
        " worker_pid INTEGER, claim_expires INTEGER);"
    )
    conn.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                 ("t_live", status, task_pid, run_id, lock, task_expires))
    if run_id is not None:
        conn.execute("INSERT INTO task_runs VALUES (?, ?, ?, ?)",
                     (run_id, run_ended, run_pid, run_expires))
    conn.commit()
    conn.close()
    return db


def _pin_carrier(monkeypatch, db, *, task="t_live", run_id="1", lock="lk"):
    """Pin the full dispatcher-spawn carrier: DB + run id + claim lock."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="owned-live"),
        pytest.param({"status": "ready"}, id="released-to-the-board"),
        pytest.param({"task_pid": 999_999_999}, id="other-process-owns-the-claim"),
        pytest.param({"run_ended": 12345}, id="run-already-closed"),
        pytest.param({"run_pid": 999_999_999}, id="run-handed-to-another-pid"),
        pytest.param({"run_id": None}, id="no-live-run-pointer"),
        pytest.param({"run_id": 2}, id="board-moved-to-another-run"),
        pytest.param({"lock": "other-lock"}, id="claim-lock-no-longer-ours"),
        pytest.param({"task_expires": 1}, id="task-lease-expired"),
        pytest.param({"run_expires": 1}, id="run-lease-expired"),
        pytest.param({"task_expires": None}, id="null-task-lease-is-not-unbounded"),
        pytest.param({"run_expires": None}, id="null-run-lease-is-not-unbounded"),
    ],
)
def test_claim_ownership_matrix(clear_kanban_env, tmp_path, overrides):
    """``worker_claim_is_live`` authorises model re-entry, so every broken coordinate must
    deny it — a NULL lease is not "unbounded", and the pinned carrier itself stays intact
    (the denial has to come from the board rows, not from a missing pin)."""
    board = {"task_pid": os.getpid(), "run_pid": os.getpid(), **overrides}
    db = _make_board(tmp_path, **board)
    _pin_carrier(clear_kanban_env, db)
    assert worker_claim_is_live() is (overrides == {})


def test_claim_check_fails_closed_without_a_board(clear_kanban_env, tmp_path):
    """No proof, no retry: a missing board must not authorise model re-entry."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_live")
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "missing.db"))
    assert worker_claim_is_live() is False
    clear_kanban_env.delenv("HERMES_KANBAN_DB", raising=False)
    clear_kanban_env.setenv("HERMES_KANBAN_BOARD", "no-such-board-xyz")
    assert worker_claim_is_live() is False  # and no ambient fallback rescues it


def test_missing_pinned_carrier_fails_closed(clear_kanban_env, tmp_path):
    """Round-2 P1: the dispatcher pins DB + run id + claim lock at spawn. Any MISSING
    coordinate means there is no exact authority carrier to re-prove — the proof fails
    closed instead of widening to ambient board state or skipping comparisons."""
    db = _make_board(tmp_path)  # a live board with a live lease
    _pin_carrier(clear_kanban_env, db)
    assert worker_claim_is_live() is True  # sanity: the full carrier proves live

    for missing, value in (("HERMES_KANBAN_DB", str(db)), ("HERMES_KANBAN_RUN_ID", "1"),
                           ("HERMES_KANBAN_CLAIM_LOCK", "lk")):
        clear_kanban_env.delenv(missing, raising=False)
        assert worker_claim_is_live() is False, missing  # a missing pin denies
        clear_kanban_env.setenv(missing, value)


def test_missing_db_pin_never_resolves_an_ambient_board(tmp_path, monkeypatch):
    """Round-2 P1: the old code reconstructed the board from ``HERMES_KANBAN_BOARD`` /
    the default when the pin was absent — the proof could silently move to whatever board
    is ambient NOW. With the pin gone the canonical resolver must not be consulted at all,
    even when it would return a fully live board whose row matches this worker (built with
    the real board API, so this is not a strawman)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="ambient live task")
        assert kb.claim_task(conn, task_id, claimer="lk", ttl_seconds=3600) is not None
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    import agent.kanban_turn_recovery as rec

    consulted: list = []
    real_resolve = kb.kanban_db_path

    def _spy(board=None):
        consulted.append(board)
        return real_resolve(board=board)

    monkeypatch.setattr(kb, "kanban_db_path", _spy)
    assert rec.worker_claim_is_live() is False
    assert consulted == []  # no ambient resolution was attempted


def test_lease_expired_during_the_backoff_blocks_model_reentry(
    clear_kanban_env, tmp_path, monkeypatch
):
    """Round-2 P1 (the reviewer's exact race): failure near TTL expiry → the pre-backoff
    proof passes → the 15/45/90s sleep crosses expiry → the post-backoff re-proof must FAIL
    on the unexpired-lease check even though status/run-id/lock/pid/ended_at are all
    unchanged. No model turn is made."""
    import agent.kanban_turn_recovery as rec

    db = _make_board(tmp_path, task_expires=1_600, run_expires=1_600)
    _pin_carrier(clear_kanban_env, db, run_id="1", lock="lk")
    clock = iter([1_000, 1_700])  # pre-backoff proof sees a live lease; the re-proof does not
    monkeypatch.setattr(rec, "_now", lambda: next(clock))
    turns: list[str] = []
    emitted: list[str] = []

    attempts = rec.recover_failed_kanban_turns(
        lambda nudge: turns.append(nudge), lambda: _failed(),
        sleep_fn=lambda s: None, emit=emitted.append,  # default claim_check: the real proof
    )
    assert attempts == 1
    assert turns == []  # the expired lease vetoed re-entry before the model call
    assert "no longer holds a live run/claim" in emitted[-1]


# ── the loop ─────────────────────────────────────────────────────────


def _loop_ok(result=None):
    return (lambda: True) if result is None else result


def test_recovery_loop_retries_until_success_and_reports(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    latest = {"r": _failed()}
    turns: list[str] = []
    delays: list[float] = []
    emitted: list[str] = []

    def turn_fn(nudge: str) -> None:
        turns.append(nudge)
        latest["r"] = _success()  # provider recovered

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=delays.append, emit=emitted.append,
        claim_check=_loop_ok(),
    )
    assert attempts == 1
    assert delays == [15.0]
    assert "Do NOT start over" in turns[0]
    assert "t_probe" in turns[0]
    assert len(emitted) == 1
    assert "[kanban]" in emitted[0] and "attempt 1/3" in emitted[0]


def test_recovery_loop_bounded_when_result_never_changes(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    delays: list[float] = []
    attempts = recover_failed_kanban_turns(
        lambda nudge: None,
        lambda: _failed(),  # stuck result: must NOT loop forever
        sleep_fn=delays.append, emit=lambda m: None, claim_check=_loop_ok(),
    )
    assert attempts == DEFAULT_MAX_RECOVERY_ATTEMPTS
    assert delays == [15.0, 45.0, 90.0]


def test_recovery_loop_order_is_check_sleep_check_turn(clear_kanban_env):
    """Round-5 finding F1, both halves in one pass: the proof is taken BEFORE the backoff
    and re-taken immediately before model re-entry (check → sleep → check → turn), so a run
    lost during the 15/45/90s sleep stops the loop WITHOUT re-entering the model."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "3")
    events: list[str] = []
    latest = {"r": _failed()}
    proofs = {"n": 0}

    def claim_check() -> bool:
        proofs["n"] += 1
        events.append("check")
        return proofs["n"] == 1  # first proof holds; the run is lost during the backoff

    def turn_fn(nudge: str) -> None:
        events.append("turn")  # a model entry: must NOT happen on the denied path
        latest["r"] = _success()

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=lambda s: events.append("sleep"),
        emit=lambda m: None, claim_check=claim_check,
    )
    assert events == ["check", "sleep", "check"]  # the lost run vetoed re-entry
    assert attempts == 1  # the attempt was authorised …

    # …and with the proof intact, the happy path is the full check → sleep → check → turn.
    events.clear()
    latest["r"] = _failed()
    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=lambda s: events.append("sleep"),
        emit=lambda m: None, claim_check=lambda: (events.append("check"), True)[1],
    )
    assert attempts == 1
    assert events == ["check", "sleep", "check", "turn"]


def test_nudge_terminal_contract(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    nudge = build_recovery_nudge(_failed(error="peer closed connection"), attempt=1, max_attempts=3)
    assert "t_probe" in nudge
    assert "1/3" in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "Do NOT start over" in nudge
    assert "peer closed connection" in nudge


# ── what this PR adds to the shared exit mapping ─────────────────────


def test_exit_mapping_composes_with_the_stripped_worker_predicate(monkeypatch):
    """``cli._single_query_exit_code`` is main's (its own contract test pins the mapping);
    this PR's contribution is that it asks ``kanban_task_id()`` whether this is a worker.
    Pin exactly that: surrounding whitespace must not decide worker-ness, or a
    bare-whitespace id would hand a quota wall the neutral 75."""
    from cli import _single_query_exit_code

    monkeypatch.setenv("HERMES_KANBAN_TASK", "   ")
    assert _single_query_exit_code(_failed(reason="rate_limit")) == 1
    monkeypatch.setenv("HERMES_KANBAN_TASK", "  t_probe  ")
    assert _single_query_exit_code(_failed(reason="rate_limit")) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_explicit_blank_carrier_never_falls_back_to_the_env_pin(clear_kanban_env):
    """Owner-side workers pass the bound frame's task id explicitly; a blank explicit id is
    not a worker and must not be rescued by whatever ``HERMES_KANBAN_TASK`` the owner
    process happens to carry."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_foreign")
    assert should_recover_turn(_failed(), attempt=0, task_id="   ") is False
    assert should_recover_turn(_failed(), attempt=0, task_id="  t_bound  ") is True


# ── call-site behaviour: the owner-side worker seam ──────────────────


@pytest.fixture
def owner_worker(tmp_path, clear_kanban_env, monkeypatch):
    """Run the real owner-side worker turns for one claimed task on a real board.

    ``install(script, ...)`` claims a task, records this interpreter's ``worker_bound``
    event (what ``bind_worker_context`` writes), and returns ``(run, entered, context)``:
    ``run()`` executes ``session_kanban.run_worker_turns`` and returns the exit code the
    dispatcher books (``worker_exit_code``); ``entered`` records every model entry's user
    message. ``script`` is a list of settled turn results or a callable returning one.
    The owner process deliberately carries a FOREIGN env carrier: the proof must come
    from the frame, never from the owner's environment.
    """
    import agent.kanban_turn_recovery as rec
    from gateway import session_kanban
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key, value in (("HERMES_KANBAN_TASK", "t_foreign"), ("HERMES_KANBAN_DB", str(tmp_path / "nope.db")),
                       ("HERMES_KANBAN_RUN_ID", "999"), ("HERMES_KANBAN_CLAIM_LOCK", "foreign")):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    monkeypatch.setattr("agent.skill_commands.build_preloaded_skills_prompt", lambda *a, **k: ("", [], []))

    def install(script, *, goal_mode=False, expires=None, recovery=None):
        if recovery is not None:
            monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", str(recovery))
        kb.init_db()
        path = kb.kanban_db_path()
        with connect_closing(path) as conn:
            task_id = kb.create_task(conn, title="recover", assignee="default")
            task = kb.claim_task(conn, task_id, claimer="private-host:owner")
            assert task is not None
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "worker_bound",
                                 {"pid": os.getpid(), "claim_lock": task.claim_lock}, run_id=task.current_run_id)
                if expires is not None:  # status/run/lock all still match: only the lease moves
                    conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (expires, task_id))
                    conn.execute("UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                                 (expires, task.current_run_id))
        context = {"db": str(path), "task_id": task_id, "run_id": task.current_run_id,
                   "claim_lock": task.claim_lock, "skills": [], "context": "", "goal_mode": goal_mode,
                   "goal_text": "g", "goal_max_turns": 2}
        entered: list[str] = []
        seen = {"n": 0}

        class _Agent:
            session_id = "s1"

            def __init__(self):
                self._session_db = self

            def get_messages_as_conversation(self, _sid):
                return [{"role": "user", "content": "from-store"}]

            def run_conversation(self, message, conversation_history=None):
                entered.append(message)
                result = script() if callable(script) else script[seen["n"]]
                seen["n"] += 1
                return result

        frame = {"text": "work", "policy": {"kanban_json": json.dumps(context)}}

        def run() -> int:
            session_kanban.run_worker_turns(_Agent(), frame, [])
            return session_kanban.worker_exit_code(path, context)

        return run, entered, context

    return install


def test_recovers_in_place(owner_worker):
    run, entered, context = owner_worker([_failed(), _success()], recovery=2)

    assert run() == 0  # settled successfully
    assert len(entered) == 2  # the failed turn, then the recovery turn
    assert "Do NOT start over" in entered[1]
    assert context["task_id"] in entered[1] and "t_foreign" not in entered[1]


def test_budget_exhausted_stops_and_releases(owner_worker):
    """Bounded, then released: the loop stops at the cap, and the exhausted transient wall
    still reaches the dispatcher as the neutral code."""
    run, entered, _ = owner_worker(_failed, recovery=1)  # never recovers

    assert run() == KANBAN_RATE_LIMIT_EXIT_CODE
    assert len(entered) == 2  # original + one recovery attempt


@pytest.mark.parametrize(
    ("script", "code"),
    [
        pytest.param([None], 1, id="nothing-settled"),
        pytest.param([_partial()], 1, id="denial-unfinished-turn"),
        pytest.param([_failed(reason="rate_limit")], KANBAN_RATE_LIMIT_EXIT_CODE, id="quota-wall"),
        pytest.param([_failed(reason="billing")], KANBAN_RATE_LIMIT_EXIT_CODE, id="billing-wall"),
        pytest.param([_failed(reason="upstream_rate_limit")], KANBAN_RATE_LIMIT_EXIT_CODE,
                     id="upstream-quota-wall"),
    ],
)
def test_single_turn_route_outcomes(script, code, owner_worker):
    """Every row takes exactly ONE model turn, so what is under test is what the worker does
    AFTER that turn: the neutral release for a quota wall (#48000-class boundary, including
    the aggregator's upstream 429) versus an honest 1 for a turn that is not retry authority
    (partial / nothing-settled) rather than a silent rc=0."""
    run, entered, _ = owner_worker(list(script))

    assert run() == code
    assert len(entered) == 1


def test_non_worker_frame_is_never_recovered(clear_kanban_env, monkeypatch):
    """A turn whose frame carries no kanban claim is not a worker, even inside an owner
    process whose environment names a task: one model entry, result returned untouched."""
    from gateway import session_kanban

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_foreign")
    entered: list[str] = []
    failed = _failed()

    class _Agent:
        def run_conversation(self, message, conversation_history=None):
            entered.append(message)
            return failed

    frame = {"text": "hello", "policy": {}}
    assert session_kanban.run_worker_turns(_Agent(), frame, []) is failed
    assert entered == ["hello"]


def test_claim_denial_stops_before_model_reentry(owner_worker, caplog):
    """A worker whose run/claim proof fails (lease lapsed, not yet reaped) gets no recovery
    nudge, says so, and still leaves the dispatcher its neutral release."""
    import logging

    run, entered, _ = owner_worker([_failed()], expires=1)
    with caplog.at_level(logging.WARNING, logger="agent.kanban_turn_recovery"):
        assert run() == KANBAN_RATE_LIMIT_EXIT_CODE
    assert len(entered) == 1
    assert "no longer holds a live run/claim" in caplog.text


def test_worker_client_reports_the_owner_exit_and_reply(owner_worker, monkeypatch, capsys):
    """The dispatcher-spawned process is the client half: it exits with the code the owner
    booked for this exact attempt, ends its log with the attempt's reply, and writes the
    exit trailer the dead-worker sweep reads — so an exhausted wall still releases as 75."""
    from contextlib import asynccontextmanager

    from hermes_cli import gateway_client, kanban_worker_client
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

    run, entered, context = owner_worker(_failed, recovery=1)
    assert run() == KANBAN_RATE_LIMIT_EXIT_CODE
    submitted = []

    class _Owner:
        async def rpc(self, method, **params):
            submitted.append(method)
            assert method == "kanban.run" and params["task_id"] == context["task_id"]
            return {"session_id": "s1", "receipt": {"status": "terminal", "admission_id": "a1"}}

    @asynccontextmanager
    async def _connect():
        yield _Owner()

    monkeypatch.setattr(gateway_client, "connect_gateway", _connect)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    for key, value in (("HERMES_KANBAN_BOARD", "default"), ("HERMES_KANBAN_TASK", context["task_id"]),
                       ("HERMES_KANBAN_RUN_ID", str(context["run_id"])),
                       ("HERMES_KANBAN_CLAIM_LOCK", context["claim_lock"]),
                       ("HERMES_KANBAN_DB", context["db"])):
        monkeypatch.setenv(key, value)
    capsys.readouterr()

    assert kanban_worker_client.main() == KANBAN_RATE_LIMIT_EXIT_CODE
    out, err = capsys.readouterr()
    assert submitted == ["kanban.run"]  # the client never re-enters the model itself
    assert "API call failed" in out
    assert f"{KANBAN_WORKER_EXIT_TRAILER}{KANBAN_RATE_LIMIT_EXIT_CODE}" in err


@pytest.mark.asyncio
async def test_non_worker_one_shot_prints_its_exit_summary(capsys):
    """Smoke pin kept from the pre-gateway file: a plain failed ``chat -q`` run exits 1 after
    printing the turn's text and its stderr ``session_id`` summary, and takes exactly one
    admission — recovery is never a client-side resubmit."""
    import asyncio

    from hermes_cli.gateway_chat_view import GatewayChatView

    submitted = []

    class _Owner:
        events: asyncio.Queue = asyncio.Queue()

        async def rpc(self, method, **params):
            submitted.append(method)
            self.events.put_nowait({"method": "event", "params": {
                "type": "message.complete", "session_id": "stored", "admission_id": "mine",
                "payload": {"text": "API call failed after 3 retries", "outcome": "failed"}}})
            return {"admission_id": "mine"}

    view = GatewayChatView(_Owner(), {"stored_session_id": "stored"}, quiet=True)
    assert await asyncio.wait_for(view.run("hello", oneshot=True), 2) == 1
    out, err = capsys.readouterr()
    assert submitted == ["prompt.submit"]
    assert "API call failed" in out and "session_id: stored" in err


# ── goal mode: a refused recovery stops ALL model entry (one table) ──


def _goal_cases():
    """One row per way model entry can be refused while goal mode is ON:
    ``(id, setup, script, expected exit code, expected model entries)``. Each row carries
    its own setup, because the refusal reason IS the setup."""
    neutral = KANBAN_RATE_LIMIT_EXIT_CODE
    return [
        pytest.param("recovery-denied", {"expires": 1}, _failed, neutral, 1),
        pytest.param("lease-expired-during-backoff", {"expires": 1_600, "clock": (1_000, 1_700)},
                     _failed, neutral, 1),
        pytest.param("quota-wall-billing", {}, lambda: _failed(reason="billing"), neutral, 1),
        pytest.param("quota-wall-upstream", {}, lambda: _failed(reason="upstream_rate_limit"),
                     neutral, 1),
        pytest.param("interrupt", {}, _interrupted, 130, 1),
        pytest.param("terminal-settlement", {}, _settled, neutral, 1),
        pytest.param("budget-exhausted", {"recovery": 1}, _failed, neutral, 2),
        pytest.param("unfinished-nonfailed", {}, _partial, 1, 1),
    ]


@pytest.fixture
def goal_spy(monkeypatch):
    """Spy on the goal continuation loop the owner-side worker enters. The loop is
    pre-existing (its status check sees only run identity, not the claim lease); the
    invariant under test is the call-site veto."""
    calls: list = []
    monkeypatch.setattr("hermes_cli.goals.run_kanban_goal_loop", lambda **kw: calls.append(kw))
    return calls


@pytest.mark.parametrize(("case_id", "setup", "script", "code", "entries"), _goal_cases())
def test_goal_mode_does_not_continue(case_id, setup, script, code, entries, owner_worker, goal_spy,
                                     monkeypatch):
    """With goal mode ON, every refusal path reaches the honest exit without a judge-driven
    continuation: the goal loop's own status check sees only run identity, not the claim
    lease, so continuing would re-enter the model under an authority this worker can no
    longer prove."""
    import agent.kanban_turn_recovery as rec

    run, entered, _ = owner_worker(script, goal_mode=True, expires=setup.get("expires"),
                                   recovery=setup.get("recovery"))
    if setup.get("clock"):
        clock = iter(setup["clock"])  # pre-backoff proof live; the re-proof is not
        monkeypatch.setattr(rec, "_now", lambda: next(clock))

    assert run() == code
    assert len(entered) == entries
    assert goal_spy == []  # no continuation after a refused recovery


def test_goal_mode_continues_after_a_successful_authorized_recovery(owner_worker, goal_spy):
    """Positive control: a settled, authorized recovery still continues in goal mode — the
    veto must not swallow the happy path — and the judge receives the RECOVERED response,
    not the initial provider error."""
    run, entered, _ = owner_worker([_failed(), _success()], goal_mode=True, recovery=2)

    assert run() == 0
    assert len(entered) == 2
    assert [call["first_response"] for call in goal_spy] == ["done"]
