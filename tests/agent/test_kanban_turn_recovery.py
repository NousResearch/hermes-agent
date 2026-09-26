"""Tests for kanban worker in-place turn recovery and the shared exit policy.

Covers the retry-authority rules (positive and typed: retryable failed turns
only; interrupts and durable terminal settlements are never retried), the live
run/claim proof taken before every attempt, the backoff schedule, the
continuation nudge contract, the single exit-code decision point shared by the
``chat -q`` and quiet ``-Q`` one-shot paths, and the call-site wiring in
``cli.py`` driven through ``cli.main`` (the settled-outcome hook the ``-q``
route reads, ``cli_chat_turn_mixin.py::_last_turn_result``, is main's own).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

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
from hermes_cli.kanban_db import (
    KANBAN_RATE_LIMIT_EXIT_CODE,
    KANBAN_TERMINAL_PROVIDER_EXIT_CODE,
)

KANBAN_ENV = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_TURN_RECOVERY",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in KANBAN_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _failed(*, retryable: bool = True, reason: str = "timeout",
            error: str = "peer closed connection") -> dict:
    return {
        "failed": True,
        "failure_retryable": retryable,
        "failure_reason": reason,
        "error": error,
        "final_response": f"API call failed after 3 retries: {error}",
        "completed": False,
        "messages": [],
    }


def _success() -> dict:
    return {"failed": False, "completed": True, "final_response": "done", "messages": []}


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


def test_env_zero_disables(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "0")
    assert kanban_turn_recovery_enabled() is False
    assert should_recover_turn(_failed(), attempt=0) is False


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


def test_kanban_task_id_strips_and_rejects_blank(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "   ")
    assert kanban_task_id() is None  # whitespace-only is NOT a worker, anywhere
    assert kanban_turn_recovery_enabled() is False
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "  t_probe  ")
    assert kanban_task_id() == "t_probe"
    assert kanban_turn_recovery_enabled() is True


# ── retry authority: positive and typed ──────────────────────────────


def test_retryable_failed_turn_is_eligible(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(reason="timeout"), attempt=0) is True
    assert should_recover_turn(_failed(reason="timeout"), attempt=2) is True  # budget 3
    assert should_recover_turn(_failed(reason="timeout"), attempt=3) is False


def test_not_retryable_is_not_recovered(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(retryable=False, reason="auth"), attempt=0) is False
    assert should_recover_turn(_success(), attempt=0) is False
    assert should_recover_turn(None, attempt=0) is False


def test_rate_limit_and_billing_are_left_to_the_dispatcher(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(reason="rate_limit"), attempt=0) is False
    assert should_recover_turn(_failed(reason="billing"), attempt=0) is False


def test_upstream_rate_limit_is_a_dispatcher_quota_wall(clear_kanban_env):
    """The aggregator's upstream 429 is the same quota-class wall as a direct 429:
    not retried in place, and released as the neutral 75 for a worker (main's
    ``_TRANSIENT_PROVIDER_REASONS``); a person's run books it as a plain 1."""
    from cli import _single_query_exit_code

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(reason="upstream_rate_limit"), attempt=0) is False
    assert _single_query_exit_code(_failed(reason="upstream_rate_limit")) == 75
    clear_kanban_env.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _single_query_exit_code(_failed(reason="upstream_rate_limit")) == 1


def test_interrupt_is_never_retry_authority(clear_kanban_env):
    """``agent/turn_recovery.py::abort_turn_on_interrupt`` already persisted and
    cleared the interrupt — re-entering the model would resurrect cancelled work."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    interrupted = {
        "final_response": "interrupted", "messages": [], "api_calls": 4,
        "completed": False, "interrupted": True,
        "failed": True, "failure_retryable": True, "failure_reason": "timeout",
    }
    assert should_recover_turn(interrupted, attempt=0) is False


def test_terminal_settlement_is_never_retry_authority(clear_kanban_env):
    """#87096: the bounded kanban finalizer records ``outcome="timed_out"`` and
    releases the claim while the result still reads ``completed=False`` — retrying
    would continue work after this exact run was terminalized."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    settled = {
        "final_response": "budget exhausted", "messages": [], "api_calls": 60,
        "completed": False, "failed": False,
        "turn_exit_reason": "max_iterations_reached(60/60)",
    }
    assert should_recover_turn(settled, attempt=0) is False
    # The sharp combination: the turn ALSO carries a retryable failure verdict (the
    # loop exhausted its provider retries and then hit the iteration cap, so the
    # finalizer terminalized the run). The settlement veto must win over the
    # otherwise-eligible failure stamp.
    sharp = {**_failed(retryable=True), "turn_exit_reason": "max_iterations_reached(60/60)"}
    assert should_recover_turn(sharp, attempt=0) is False
    # …and the exit policy still books it honestly rather than as a silent rc=0
    # (no failure_reason on a settled turn → not transient → honest 1).
    from cli import _single_query_exit_code

    assert _single_query_exit_code(settled) == 1


def test_incomplete_turn_is_not_retry_authority(clear_kanban_env):
    """``partial`` / ``completed=False`` describe an outcome; they are not retry
    authority (truncation repair belongs inside the conversation loop, #89289)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    partial = {"partial": True, "completed": False, "final_response": "cut off",
               "error": "truncated", "messages": []}
    assert should_recover_turn(partial, attempt=0) is False
    assert should_recover_turn({"completed": False, "final_response": "x"}, attempt=0) is False


# ── live run/claim proof ─────────────────────────────────────────────


#: Any "live" lease for these tests (2100-01-01 UTC).
_FAR_FUTURE_EXPIRY = 4_102_444_800


def _make_board(tmp_path, *, status="running", task_pid=None, run_id=1, lock="lk",
                run_ended=None, run_pid=None, task_expires=_FAR_FUTURE_EXPIRY,
                run_expires=_FAR_FUTURE_EXPIRY):
    """A synthetic board whose task carries a LIVE, unexpired task+run lease by
    default (current-main dispatcher shape: ``claim_task`` sets ``claim_expires``
    on both rows and ``heartbeat_claim`` mirrors the extension)."""
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, worker_pid INTEGER, "
        "current_run_id INTEGER, claim_lock TEXT, claim_expires INTEGER)"
    )
    conn.execute(
        "CREATE TABLE task_runs (id INTEGER PRIMARY KEY, ended_at INTEGER, "
        "worker_pid INTEGER, claim_expires INTEGER)"
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
        ("t_live", status, task_pid, run_id, lock, task_expires),
    )
    if run_id is not None:
        conn.execute(
            "INSERT INTO task_runs VALUES (?, ?, ?, ?)", (run_id, run_ended, run_pid, run_expires)
        )
    conn.commit()
    conn.close()
    return db


def _pin_carrier(monkeypatch, db, *, task="t_live", run_id="1", lock="lk"):
    """Pin the full dispatcher-spawn carrier: DB + run id + claim lock."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)


def test_claim_is_live_when_this_worker_still_owns_the_run(clear_kanban_env, tmp_path):
    db = _make_board(tmp_path, task_pid=os.getpid(), run_id=7, lock="lk", run_pid=os.getpid())
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_live")
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db))
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "7")
    clear_kanban_env.setenv("HERMES_KANBAN_CLAIM_LOCK", "lk")
    assert worker_claim_is_live() is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "ready"},                       # task was released back to the board
        {"task_pid": 999_999_999},                 # another process owns the claim
        {"run_ended": 12345},                      # run already closed (settled)
        {"run_pid": 999_999_999},                  # run handed to another pid
        {"run_id": None},                          # no live run pointer
        {"run_id": 2},                             # board moved to a different run id
        {"lock": "other-lock"},                    # claim lock no longer ours
        {"task_expires": 1},                       # task lease has EXPIRED
        {"run_expires": 1},                        # run lease has EXPIRED (heartbeat mirror)
        {"task_expires": None},                    # no provable task lease — NULL is not "unbounded"
        {"run_expires": None},                     # no provable run lease
    ],
)
def test_claim_is_not_live_when_ownership_is_broken(clear_kanban_env, tmp_path, overrides):
    board = {"task_pid": os.getpid(), "run_pid": os.getpid(), **overrides}
    db = _make_board(tmp_path, **board)
    _pin_carrier(clear_kanban_env, db)  # the pinned carrier itself is intact
    assert worker_claim_is_live() is False


def test_claim_check_fails_closed_without_a_board(clear_kanban_env, tmp_path):
    """No proof, no retry: a missing board must not authorise model re-entry."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_live")
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "missing.db"))
    assert worker_claim_is_live() is False
    clear_kanban_env.delenv("HERMES_KANBAN_DB", raising=False)
    clear_kanban_env.setenv("HERMES_KANBAN_BOARD", "no-such-board-xyz")
    assert worker_claim_is_live() is False  # and no ambient fallback rescues it


def test_missing_pinned_carrier_fails_closed(clear_kanban_env, tmp_path):
    """Round-2 P1: the dispatcher pins DB + run id + claim lock at spawn. Any
    MISSING coordinate means there is no exact authority carrier to re-prove —
    the proof fails closed instead of widening to ambient board state or
    skipping comparisons."""
    db = _make_board(tmp_path)  # a live board with a live lease
    _pin_carrier(clear_kanban_env, db)
    assert worker_claim_is_live() is True  # sanity: the full carrier proves live

    clear_kanban_env.delenv("HERMES_KANBAN_DB", raising=False)
    assert worker_claim_is_live() is False  # missing DB pin
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db))
    clear_kanban_env.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    assert worker_claim_is_live() is False  # missing run-id pin
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "1")
    clear_kanban_env.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    assert worker_claim_is_live() is False  # missing claim-lock pin


def test_missing_db_pin_never_resolves_an_ambient_board(tmp_path, monkeypatch):
    """Round-2 P1: the old code reconstructed the board from
    ``HERMES_KANBAN_BOARD`` / the default when the pin was absent — the proof
    could silently move to whatever board is ambient NOW. With the pin gone the
    canonical resolver must not be consulted at all, even when it would return a
    fully live board whose row matches this worker (built with the real board
    API so this is not a strawman)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="ambient live task")
        claimed = kb.claim_task(conn, task_id, claimer="lk", ttl_seconds=3600)
        assert claimed is not None  # running, lock+run pointer+unexpired TTL set
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


def test_lease_expired_during_the_backoff_blocks_model_reentry(clear_kanban_env, tmp_path, monkeypatch):
    """Round-2 P1 (the reviewer's exact race): failure near TTL expiry → the
    pre-backoff proof passes → the 15/45/90s sleep crosses expiry → the
    post-backoff re-proof must FAIL on the unexpired-lease check even though
    status/run-id/lock/pid/ended_at are all unchanged. No model turn is made."""
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


def test_recovery_loop_retries_until_success(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    latest = {"r": _failed()}
    turns: list[str] = []
    delays: list[float] = []

    def turn_fn(nudge: str) -> None:
        turns.append(nudge)
        latest["r"] = _success()  # provider recovered

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=delays.append, emit=lambda m: None,
        claim_check=_loop_ok(),
    )
    assert attempts == 1
    assert delays == [15.0]
    assert "Do NOT start over" in turns[0]
    assert "t_probe" in turns[0]


def test_recovery_loop_bounded_when_result_never_changes(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    turns: list[str] = []
    delays: list[float] = []
    attempts = recover_failed_kanban_turns(
        lambda nudge: turns.append(nudge),
        lambda: _failed(),  # stuck result: must NOT loop forever
        sleep_fn=delays.append, emit=lambda m: None, claim_check=_loop_ok(),
    )
    assert attempts == DEFAULT_MAX_RECOVERY_ATTEMPTS
    assert delays == [15.0, 45.0, 90.0]


def test_recovery_loop_checks_the_claim_before_every_attempt(clear_kanban_env):
    """The live run/claim proof is re-taken for EACH attempt. A run lost while we
    were backing off (the dispatcher can reclaim during the 15/45/90s sleep) must
    stop the loop WITHOUT re-entering the model — round-5 finding F1."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "3")
    checks: list[int] = []
    turns: list[str] = []
    emitted: list[str] = []

    def claim_check() -> bool:
        checks.append(len(checks) + 1)
        return len(checks) == 1  # first proof holds; the run is lost during the backoff

    attempts = recover_failed_kanban_turns(
        lambda nudge: turns.append(nudge), lambda: _failed(),
        sleep_fn=lambda s: None, emit=emitted.append, claim_check=claim_check,
    )
    assert checks == [1, 2]  # pre-backoff proof + the pre-entry re-proof
    assert attempts == 1     # the attempt was authorised …
    assert len(turns) == 0   # … but the run was gone before re-entry: no model turn
    assert "no longer holds a live run/claim" in emitted[-1]


def test_recovery_loop_rechecks_the_claim_after_the_backoff(clear_kanban_env):
    """Order pin: check -> sleep -> check -> turn. The re-proof sits immediately
    before model re-entry, not only before the backoff (round-5 finding F1)."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    events: list[str] = []
    latest = {"r": _failed()}

    def claim_check() -> bool:
        events.append("check")
        return True

    def sleeper(_seconds: float) -> None:
        events.append("sleep")

    def turn_fn(nudge: str) -> None:
        events.append("turn")
        latest["r"] = _success()

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=sleeper, emit=lambda m: None,
        claim_check=claim_check,
    )
    assert attempts == 1
    assert events == ["check", "sleep", "check", "turn"]


def test_recovery_emit_receives_status_line(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    emitted: list[str] = []
    latest = {"r": _failed()}
    recover_failed_kanban_turns(
        lambda nudge: latest.update(r=_success()), lambda: latest["r"],
        sleep_fn=lambda s: None, emit=emitted.append, claim_check=_loop_ok(),
    )
    assert len(emitted) == 1
    assert "[kanban]" in emitted[0]
    assert "attempt 1/3" in emitted[0]


def test_nudge_terminal_contract(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    nudge = build_recovery_nudge(_failed(error="peer closed connection"), attempt=1, max_attempts=3)
    assert "t_probe" in nudge
    assert "1/3" in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "Do NOT start over" in nudge
    assert "peer closed connection" in nudge


# ── shared exit-code policy ──────────────────────────────────────────


def test_single_query_exit_code_policy(monkeypatch):
    """The shared exit mapping is main's ``cli._single_query_exit_code`` — this PR
    composes with it (stripped worker predicate) and must not fork it. Pins the
    composed contract as main defines it today: transient provider walls release a
    worker as the neutral 75, terminal provider verdicts park it as EX_CONFIG 78,
    other failures and unfinished turns are honest 1s, interrupts are 130,
    success 0."""
    from cli import _single_query_exit_code

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert _single_query_exit_code(_success()) == 0
    # transient provider walls: neutral 75 for a worker — quota walls and the
    # broader transient set (a storm that outlasts in-place recovery is released,
    # not counted against the breaker)
    for reason in ("rate_limit", "billing", "upstream_rate_limit", "timeout", "overloaded"):
        assert _single_query_exit_code(_failed(reason=reason)) == KANBAN_RATE_LIMIT_EXIT_CODE
    # terminal provider verdicts (revoked credential, model gone, WAF block) are
    # main's EX_CONFIG: the dispatcher parks the card after ONE spawn instead of
    # re-spawning into the same wall
    for reason in ("auth", "auth_permanent", "model_not_found", "ssl_cert_verification",
                   "upstream_blocked"):
        assert _single_query_exit_code(_failed(reason=reason)) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    # a non-transient, non-terminal worker failure is an honest 1
    assert _single_query_exit_code(_failed(reason="some_unknown_reason")) == 1
    # unfinished / nothing settled: honest 1 — never a silent rc=0
    assert _single_query_exit_code(None) == 1
    assert _single_query_exit_code({"partial": True, "completed": False}) == 1
    # explicit cancellation: 130 — never retried, never continued
    assert _single_query_exit_code({**_failed(), "interrupted": True}) == 130

    # a person's run: no neutral release — a wall is just a failure
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _single_query_exit_code(_success()) == 0
    assert _single_query_exit_code(_failed(reason="rate_limit")) == 1
    assert _single_query_exit_code(_failed(reason="timeout")) == 1
    assert _single_query_exit_code(None) == 1


# ── call-site behaviour (drive cli.main with a FakeCLI) ──────────────


def _install_fake_cli(monkeypatch, chat_script, calls):
    """Mirror tests/hermes_cli/test_single_query_session_finalize.py's FakeCLI harness."""
    import cli as cli_mod

    class _Console:
        def print(self, *a, **k):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "single-query-session"
            self.agent = SimpleNamespace(session_id="single-query-session", platform="cli")

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, query, images=None):
            calls.append(("chat", query))
            result = chat_script() if callable(chat_script) else chat_script.pop(0)
            self._last_turn_result = result
            # Mirror production: chat() returns the settled turn's rendered response —
            # for a failed turn that IS the provider error text.
            return result.get("final_response", "") if isinstance(result, dict) else ""

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda fake_cli: None)
    monkeypatch.setattr(cli_mod, "_collect_query_images", lambda q, img: (q, []))
    monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda imgs: [])
    return cli_mod


@pytest.fixture
def live_claim(monkeypatch):
    """The board proof is a separate contract (unit-tested above); call-site tests
    run against a worker that owns its run, unless a test overrides it."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setattr(rec, "worker_claim_is_live", lambda: True)
    return rec


def _chats(calls):
    return [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]


def test_non_quiet_kanban_recovers_in_place(monkeypatch, live_claim):
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed(), _success()], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 0  # settled successfully
    chats = _chats(calls)
    assert len(chats) == 2
    assert "Do NOT start over" in chats[1][1]
    assert "t_probe" in chats[1][1]


def test_non_quiet_kanban_exits_nonzero_after_budget(monkeypatch, live_claim):
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, _failed, calls)  # never recovers

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 75  # exhausted transient wall: released, not counted
    assert len(_chats(calls)) == 2  # original + one recovery attempt


@pytest.mark.parametrize("reason", ["rate_limit", "billing", "upstream_rate_limit"])
def test_non_quiet_rate_limit_keeps_the_neutral_exit_code(monkeypatch, live_claim, reason):
    """#48000-class boundary: the production ``chat -q`` route must hand a quota
    wall (including the aggregator's upstream 429) to the dispatcher as the
    neutral EX_TEMPFAIL code, not a counted crash."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed(reason=reason)], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 75
    assert len(_chats(calls)) == 1  # declined in place: the dispatcher owns the cooldown


def test_non_quiet_interrupted_turn_is_not_retried(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    interrupted = {**_failed(), "interrupted": True}
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [interrupted], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 130  # main's interrupt exit: never retried, never continued
    assert len(_chats(calls)) == 1


def test_non_quiet_terminal_settlement_is_not_retried(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    settled = {**_failed(retryable=True),
               "turn_exit_reason": "max_iterations_reached(60/60)"}
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [settled], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    # never retried, never continued; the transient stamp books the neutral 75 —
    # the #87096 settlement already recorded the durable timed_out outcome.
    assert exc_info.value.code == 75
    assert len(_chats(calls)) == 1


def test_non_quiet_partial_is_not_retried(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    partial = {"partial": True, "completed": False, "final_response": "cut off",
               "error": "truncated", "messages": []}
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [partial], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(_chats(calls)) == 1


def test_non_quiet_skips_recovery_without_a_live_claim(monkeypatch, capsys):
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setattr(rec, "worker_claim_is_live", lambda: False)
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed()], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 75  # denial still exits non-zero; transient → release
    assert len(_chats(calls)) == 1
    assert "no longer holds a live run/claim" in capsys.readouterr().err


def test_non_quiet_kanban_no_settled_outcome_exits_nonzero(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [None], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(_chats(calls)) == 1


def test_non_kanban_failed_run_exits_one_without_recovery(monkeypatch, live_claim):
    """No HERMES_KANBAN_TASK: no in-place recovery; main's shared exit contract
    books a failed run as 1 (the non-quiet tail now exits unconditionally)."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed()], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(_chats(calls)) == 1  # no recovery outside kanban workers
    assert "summary" in calls


def test_whitespace_task_id_is_not_a_worker_anywhere(monkeypatch, live_claim):
    """A whitespace-only HERMES_KANBAN_TASK is not a worker: the exit mapping's
    stripped predicate must not hand a quota wall the neutral 75 (raw env
    truthiness would)."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "   ")
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed(reason="rate_limit")], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1  # not 75: not a worker, not neutralized
    assert len(_chats(calls)) == 1


def _install_quiet_fake_cli(monkeypatch, run_conversation):
    """Quiet (-Q) harness: minimal HermesCLI whose agent runs the given function."""
    import cli as cli_mod

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session", platform="cli", quiet_mode=False,
                suppress_status_output=False, stream_delta_callback=object(),
                tool_gen_callback=object(), run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            return {"signature": "same-route", "model": None, "runtime": None, "request_overrides": None}

        def _init_agent(self, **kwargs):
            return True

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda fake_cli: None)
    monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda imgs: [])
    return cli_mod


def test_quiet_kanban_recovers_in_place(monkeypatch, live_claim):
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed() if len(runs) == 1 else _success()

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert len(runs) == 2
    assert "Do NOT start over" in runs[1]


def test_quiet_rate_limit_keeps_the_neutral_exit_code(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed(reason="billing")

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 75
    assert len(runs) == 1


def test_quiet_partial_exits_nonzero_without_retrying(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return {"partial": True, "completed": False, "final_response": "cut off",
                "error": "truncated"}

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(runs) == 1


def test_quiet_kanban_no_settled_outcome_exits_nonzero(monkeypatch, live_claim):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return None

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(runs) == 1


def test_quiet_non_kanban_no_settled_outcome_is_booked_honestly(monkeypatch, live_claim):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return None

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1  # main's shared contract: unsettled is 1 for every caller
    assert len(runs) == 1


# ── goal-mode composition (round-3): a recovery refusal stops ALL model entry ──


@pytest.fixture
def goal_spy(monkeypatch):
    """Spy on the goal continuation loop at the cli-module seam. The goal loop
    itself is pre-existing (its status check sees only run identity, not the
    claim lease); the invariant under test here is the CALL-SITE gate."""
    import cli as cli_mod

    calls: list = []
    monkeypatch.setattr(cli_mod, "_run_kanban_goal_loop_q", lambda c, resp: calls.append(resp))
    return calls


def _quiet_goal_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")


def test_goal_mode_does_not_continue_after_recovery_denial(monkeypatch, capsys, goal_spy):
    """Round-3 P1: with goal mode ON, a DENIED recovery (claim lost) must reach
    the shared exit path without goal continuation — the goal loop's status
    check sees only run identity, so continuing would re-enter the model under
    an authority this process can no longer prove."""
    import agent.kanban_turn_recovery as rec

    _quiet_goal_env(monkeypatch)
    monkeypatch.setattr(rec, "worker_claim_is_live", lambda: False)  # recovery denied
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed()

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 75  # timeout stamp → transient release; gate still vetoes
    assert len(runs) == 1          # the failed turn only: no recovery nudge, and
    assert goal_spy == []          # NO goal continuation after the denial
    assert "no longer holds a live run/claim" in capsys.readouterr().err


def test_goal_mode_does_not_continue_when_the_lease_expires_during_the_backoff(
    clear_kanban_env, tmp_path, monkeypatch, goal_spy
):
    """Round-3 P1, the reviewer's exact reproduced fixture, end to end through
    ``cli.main``: both leases at 1600, pre-backoff proof at 1000 passes, the
    backoff crosses to 1700, the re-proof refuses — and goal mode (which sees a
    still-running identity-only status) must NOT start another model turn."""
    import agent.kanban_turn_recovery as rec

    db = _make_board(tmp_path, task_expires=1_600, run_expires=1_600)
    _pin_carrier(clear_kanban_env, db, run_id="1", lock="lk")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    clock = iter([1_000, 1_700])
    monkeypatch.setattr(rec, "_now", lambda: next(clock))
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed()

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 75  # timeout stamp → transient release; gate still vetoes
    assert len(runs) == 1   # goal-on: still exactly ONE model call (was 2 before the gate)
    assert goal_spy == []


@pytest.mark.parametrize("reason", ["billing", "upstream_rate_limit"])
def test_goal_mode_does_not_continue_on_a_quota_wall(monkeypatch, goal_spy, reason):
    """The neutral 75 must arrive without a sibling continuation making another
    quota-bound call first — goal mode is skipped for a quota-wall result."""
    _quiet_goal_env(monkeypatch)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed(reason=reason)

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 75
    assert len(runs) == 1
    assert goal_spy == []


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({**_failed(), "interrupted": True}, 130),  # explicit cancellation: 130, never continued
        ({**_failed(retryable=True), "turn_exit_reason": "max_iterations_reached(60/60)"},
         75),  # #87096 settlement: transient stamp → neutral release, never continued
    ],
    ids=["interrupt", "terminal-settlement"],
)
def test_goal_mode_does_not_continue_on_interrupt_or_terminal_settlement(
    monkeypatch, goal_spy, result, expected
):
    _quiet_goal_env(monkeypatch)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return result

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == expected
    assert len(runs) == 1
    assert goal_spy == []


def test_goal_mode_does_not_continue_after_the_recovery_budget_is_exhausted(
    monkeypatch, live_claim, goal_spy
):
    """Budget exhausted (recovery attempted, still failed): the goal loop must
    not take over as an unbounded continuation channel past the recovery cap."""
    import agent.kanban_turn_recovery as rec

    _quiet_goal_env(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed()

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 75  # exhausted transient wall: released, not counted
    assert len(runs) == 2   # original + the one authorised recovery attempt
    assert goal_spy == []   # …and nothing continues past the exhausted budget


@pytest.mark.parametrize(
    "result",
    [
        {"partial": True, "completed": False, "final_response": "cut off", "error": "truncated",
         "messages": []},
    ],
    ids=["partial"],
)
def test_goal_mode_does_not_continue_on_an_unfinished_nonfailed_turn(monkeypatch, goal_spy, result):
    """Pinned semantics for the gate's breadth: an unfinished-but-not-failed turn
    (partial / completed=0) also does NOT continue in goal mode — only a settled,
    authorized turn continues. Truncation/compression repair is owned inside the
    conversation loop (#89289) and by the dispatcher, not by the goal judge."""
    _quiet_goal_env(monkeypatch)
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return result

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(runs) == 1
    assert goal_spy == []


def test_goal_mode_continues_after_a_successful_authorized_recovery(
    monkeypatch, live_claim, goal_spy
):
    """Positive control: goal continuation is PRESERVED for a settled,
    authorized turn — recovery succeeds and the judge-driven loop runs with the
    recovered response."""
    import agent.kanban_turn_recovery as rec

    _quiet_goal_env(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return _failed() if len(runs) == 1 else _success()

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert len(runs) == 2              # original failed turn + the recovery turn
    assert goal_spy == ["done"]        # continuation happened, with the recovered response


# ── the same gate on the production `-q` route (goal_mode rides `-q`) ──


@pytest.fixture
def chat_goal_spy(monkeypatch):
    """Spy on the `-q` goal continuation loop. Same invariant as ``goal_spy``:
    the gate under test is the CALL-SITE, not the loop's own status check."""
    import cli as cli_mod

    calls: list = []
    monkeypatch.setattr(cli_mod, "_run_kanban_goal_loop_chat", lambda c, resp: calls.append(resp))
    return calls


def test_non_quiet_goal_mode_does_not_continue_after_recovery_denial(
    monkeypatch, chat_goal_spy
):
    """The dispatcher spawns goal_mode cards on the ``-q`` route
    (``kanban_db_dispatch.py`` builds ``chat -q`` and sets
    ``HERMES_KANBAN_GOAL_MODE=1``), so the round-3 finding has to be closed
    there too: a DENIED recovery (claim lost) reaches the shared exit path with
    zero further model entry."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setattr(rec, "worker_claim_is_live", lambda: False)  # recovery denied
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed()], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 75    # transient stamp → released, not counted
    assert len(_chats(calls)) == 1      # the failed turn only: no recovery nudge
    assert chat_goal_spy == []          # NO goal continuation after the denial


def test_non_quiet_goal_mode_continues_after_a_successful_authorized_recovery(
    monkeypatch, live_claim, chat_goal_spy
):
    """Positive control on the ``-q`` route: a settled, authorized turn still
    continues in goal mode — the gate must not swallow the happy path — and the
    judge receives the RECOVERED response, not the initial provider error."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [_failed(), _success()], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert len(_chats(calls)) == 2
    # …and the judge must receive the RECOVERED response, not the initial provider
    # error (re-review P2): the recovery callback's return value carries through.
    assert chat_goal_spy == ["done"]
