"""Tests for kanban worker in-place turn recovery (provider-failure class).

Covers the decision policy (bounded in-place retry of a failed worker turn),
the backoff schedule, the continuation nudge contract, the worker predicate,
and the exit-code behaviour of both one-shot paths (driven through
``cli.main`` with a FakeCLI — no source reading, no real board).
"""

from __future__ import annotations

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
    turn_is_unfinished,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_TURN_RECOVERY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _failed(*, retryable: bool = True, reason: str = "timeout", error: str = "peer closed connection") -> dict:
    return {
        "failed": True,
        "failure_retryable": retryable,
        "failure_reason": reason,
        "error": error,
        "final_response": f"API call failed after 3 retries: {error}",
        "messages": [],
    }


def _success() -> dict:
    return {"failed": False, "final_response": "done", "messages": []}


# ── enablement / budget ──────────────────────────────────────────────


def test_disabled_without_kanban_task(clear_kanban_env):
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
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "-4")
    assert max_recovery_attempts() == 0
    clear_kanban_env.setenv("HERMES_KANBAN_TURN_RECOVERY", "false")
    assert max_recovery_attempts() == 0


def test_delay_schedule_repeats_last_entry():
    assert RECOVERY_DELAYS_SECONDS == (15.0, 45.0, 90.0)
    assert recovery_delay_seconds(1) == 15.0
    assert recovery_delay_seconds(2) == 45.0
    assert recovery_delay_seconds(3) == 90.0
    assert recovery_delay_seconds(9) == 90.0
    assert recovery_delay_seconds(0) == 15.0


# ── eligibility policy ───────────────────────────────────────────────


def test_not_retryable_is_not_recovered(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(retryable=False, reason="auth"), attempt=0) is False
    assert should_recover_turn(_success(), attempt=0) is False
    assert should_recover_turn(None, attempt=0) is False


def test_rate_limit_and_billing_are_not_recovered(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(reason="rate_limit"), attempt=0) is False
    assert should_recover_turn(_failed(reason="billing"), attempt=0) is False
    assert should_recover_turn(_failed(reason="timeout"), attempt=0) is True


def test_budget_exhaustion_stops_recovery(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    assert should_recover_turn(_failed(), attempt=2) is True  # default budget 3
    assert should_recover_turn(_failed(), attempt=3) is False


# ── the loop ─────────────────────────────────────────────────────────


def test_recovery_loop_retries_until_success(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    latest = {"r": _failed()}
    turns: list[str] = []
    delays: list[float] = []

    def turn_fn(nudge: str) -> None:
        turns.append(nudge)
        latest["r"] = _success()  # provider recovered

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=delays.append, emit=lambda m: None
    )
    assert attempts == 1
    assert delays == [15.0]
    assert len(turns) == 1
    assert "Do NOT start over" in turns[0]
    assert "t_probe" in turns[0]


def test_recovery_loop_second_attempt(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    outcomes = [_failed(), _success()]
    latest = {"r": _failed()}
    delays: list[float] = []

    def turn_fn(nudge: str) -> None:
        latest["r"] = outcomes.pop(0)

    attempts = recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=delays.append, emit=lambda m: None
    )
    assert attempts == 2
    assert delays == [15.0, 45.0]


def test_recovery_loop_bounded_when_result_never_changes(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    turns: list[str] = []
    delays: list[float] = []
    attempts = recover_failed_kanban_turns(
        lambda nudge: turns.append(nudge),
        lambda: _failed(),  # stuck result: must NOT loop forever
        sleep_fn=delays.append,
        emit=lambda m: None,
    )
    assert attempts == DEFAULT_MAX_RECOVERY_ATTEMPTS
    assert len(turns) == DEFAULT_MAX_RECOVERY_ATTEMPTS
    assert delays == [15.0, 45.0, 90.0]


def test_recovery_emit_receives_status_line(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    emitted: list[str] = []
    latest = {"r": _failed()}

    def turn_fn(nudge: str) -> None:
        latest["r"] = _success()

    recover_failed_kanban_turns(
        turn_fn, lambda: latest["r"], sleep_fn=lambda s: None, emit=emitted.append
    )
    assert len(emitted) == 1
    assert "[kanban]" in emitted[0]
    assert "attempt 1/3" in emitted[0]


# ── nudge contract ───────────────────────────────────────────────────


def test_nudge_terminal_contract(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    nudge = build_recovery_nudge(_failed(error="peer closed connection"), attempt=1, max_attempts=3)
    assert "t_probe" in nudge
    assert "1/3" in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "Do NOT start over" in nudge
    assert "peer closed connection" in nudge


# ── worker predicate (single source of truth, D2) ────────────────────


def test_kanban_task_id_strips_and_rejects_blank(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "   ")
    assert kanban_task_id() is None  # whitespace-only is NOT a worker, anywhere
    assert kanban_turn_recovery_enabled() is False
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "  t_probe  ")
    assert kanban_task_id() == "t_probe"
    assert kanban_turn_recovery_enabled() is True


# ── call-site behaviour (drive cli.main with a FakeCLI) ──────────────


def _failed_live(*, retryable=True, reason="timeout"):
    return {"failed": True, "failure_retryable": retryable, "failure_reason": reason,
            "error": "Connection error.", "final_response": "API call failed", "messages": []}


def _install_fake_cli(monkeypatch, chat_script, calls):
    """Mirror tests/hermes_cli/test_single_query_session_finalize.py's FakeCLI harness."""
    import cli as cli_mod
    from types import SimpleNamespace

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
            self._last_turn_result = chat_script() if callable(chat_script) else chat_script.pop(0)

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda fake_cli: None)
    # Keep the harness away from real profile/board state.
    monkeypatch.setattr(cli_mod, "_collect_query_images", lambda q, img: (q, []))
    monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda imgs: [])
    return cli_mod


def test_single_query_kanban_recovers_in_place(monkeypatch):
    """A retryable failure in kanban context retries the turn in place; success -> no exit."""
    import agent.kanban_turn_recovery as rec
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    script = [_failed_live(), {"failed": False, "final_response": "done", "messages": []}]
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, script, calls)

    cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert chats[0] == ("chat", "hello")
    assert len(chats) == 2
    assert "Do NOT start over" in chats[1][1]
    assert "t_probe" in chats[1][1]


def test_single_query_kanban_exits_nonzero_after_budget(monkeypatch):
    """Still-failed after the recovery budget -> deterministic non-zero exit (honest crash)."""
    import agent.kanban_turn_recovery as rec
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, _failed_live, calls)  # never recovers

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1
    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert len(chats) == 2  # original + one recovery attempt


def test_single_query_non_kanban_unchanged(monkeypatch):
    """Without a kanban task the one-shot path keeps its historical behaviour (no exit 1)."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, _failed_live, calls)

    cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert len(chats) == 1  # no recovery outside kanban workers
    assert "summary" in calls


def _install_quiet_fake_cli(monkeypatch, run_conversation):
    """Quiet (-Q) harness: minimal HermesCLI whose agent runs the given function."""
    import cli as cli_mod
    from types import SimpleNamespace

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


def test_quiet_single_query_kanban_recovers(monkeypatch):
    """The -Q path gets the same in-place recovery before its exit-code block."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        if len(runs) == 1:
            return {"final_response": "", "error": "boom", "failed": True,
                    "failure_retryable": True, "failure_reason": "timeout", "messages": []}
        return {"final_response": "done", "failed": False, "messages": []}

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert len(runs) == 2
    assert "Do NOT start over" in runs[1]


# ── D1: a turn that settles NOTHING must not masquerade as a clean exit ──


def test_single_query_kanban_exits_nonzero_when_no_settled_outcome(monkeypatch):
    """D1: chat() can return without settling (credentials/init failure, a raising
    settle, a blocked reference). rc=0 there is the silent protocol-violation class —
    a kanban worker must exit non-zero instead."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [None], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1
    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert len(chats) == 1  # nothing settled -> nothing to retry in place


def test_single_query_non_kanban_no_settled_outcome_stays_clean(monkeypatch):
    """D1 control: outside kanban the historical exit behaviour is untouched."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [None], calls)

    cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert "summary" in calls  # normal exit path, no SystemExit


def test_whitespace_task_id_is_not_a_worker_anywhere(monkeypatch):
    """D2: a blank task id must read as "not a worker" to BOTH the recovery gate and
    the exit-code guard — never recovery-off-but-exit-1."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "   ")
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [None], calls)

    cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert len(chats) == 1  # no recovery attempt
    assert "summary" in calls  # and no forced non-zero exit


def test_quiet_kanban_exits_nonzero_when_no_settled_outcome(monkeypatch):
    """D1 on the quiet (-Q) path: no settled outcome in kanban context -> exit 1."""
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
    assert len(runs) == 1  # None result -> no in-place retry, honest exit instead


# ── D5: an INCOMPLETE turn (partial / completed=False) is unfinished work ──


def test_partial_turn_policy(clear_kanban_env):
    """D5: incomplete turns are recoverable in place; non-retryable failures are not."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_probe")
    partial = {"partial": True, "completed": False, "final_response": "cut off", "error": "truncated"}
    assert turn_is_unfinished(partial) is True
    assert turn_is_unfinished(_success()) is False
    assert should_recover_turn(partial, attempt=0) is True
    assert should_recover_turn({"completed": False, "final_response": "x"}, attempt=0) is True
    # a FAILED turn keeps its own policy even when also flagged partial
    assert should_recover_turn({**_failed(retryable=False), "partial": True}, attempt=0) is False
    assert should_recover_turn(_success(), attempt=0) is False
    clear_kanban_env.delenv("HERMES_KANBAN_TASK", raising=False)
    assert should_recover_turn(partial, attempt=0) is False


def test_partial_turn_recovers_in_place_then_exits(monkeypatch):
    """D5: a partial turn retries in place; still partial after the budget -> exit 1."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    partial = {"partial": True, "completed": False, "final_response": "cut off", "error": "truncated"}
    calls: list = []
    cli_mod = _install_fake_cli(monkeypatch, [partial, partial], calls)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    assert exc_info.value.code == 1
    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert len(chats) == 2
    assert "Do NOT start over" in chats[1][1]


def test_partial_turn_recovers_cleanly(monkeypatch):
    """D5: a partial turn followed by a settled success exits clean (no SystemExit)."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "2")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    partial = {"partial": True, "completed": False, "final_response": "cut off", "error": "truncated"}
    calls: list = []
    cli_mod = _install_fake_cli(
        monkeypatch, [partial, {"failed": False, "final_response": "done", "messages": []}], calls
    )

    cli_mod.main(query="hello", quiet=False, oneshot=True, toolsets="terminal")

    chats = [c for c in calls if isinstance(c, tuple) and c[0] == "chat"]
    assert len(chats) == 2
    assert "summary" in calls


def test_quiet_partial_turn_exits_nonzero_after_recovery(monkeypatch):
    """D5 on the quiet path: partial -> one in-place retry -> still partial -> exit 1."""
    import agent.kanban_turn_recovery as rec

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.setenv("HERMES_KANBAN_TURN_RECOVERY", "1")
    monkeypatch.setattr(rec, "RECOVERY_DELAYS_SECONDS", (0.0,))
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    runs: list = []

    def run_conversation(*, user_message, conversation_history):
        runs.append(user_message)
        return {"partial": True, "completed": False, "final_response": "cut off", "error": "truncated"}

    cli_mod = _install_quiet_fake_cli(monkeypatch, run_conversation)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert len(runs) == 2  # original + one in-place recovery attempt
    assert "Do NOT start over" in runs[1]
