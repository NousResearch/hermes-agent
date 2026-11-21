"""The one-shot exit funnel is the single reconciliation point.

`exit_single_query` is where every `-q` / `-Q` turn converges before `sys.exit`.
It must reconcile only when the caller supplies a settled turn result, and it
must never let a guard failure mask the real exit code. Both one-shot drivers
must hand their settled result to it.
"""
import sys

import pytest

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER, exit_single_query


def _guard(monkeypatch, action="recorded"):
    import agent.kanban_terminal_guard as guard

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return action

    monkeypatch.setattr(guard, "reconcile_worker_exit", fake)
    return calls


def test_reconciles_when_turn_result_is_supplied(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    calls = _guard(monkeypatch)
    result = {"completed": False, "messages": []}
    with pytest.raises(SystemExit) as exc:
        exit_single_query(0, result=result)
    assert exc.value.code == 0
    assert calls == [{"completed": False, "result": result}]


def test_skips_reconcile_without_turn_result(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    calls = _guard(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        exit_single_query(1)
    assert exc.value.code == 1
    assert calls == []


def test_never_raises_when_guard_crashes(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    import agent.kanban_terminal_guard as guard

    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(guard, "reconcile_worker_exit", boom)
    with pytest.raises(SystemExit) as exc:
        exit_single_query(0, result={"completed": False})
    assert exc.value.code == 0
    assert KANBAN_WORKER_EXIT_TRAILER in capsys.readouterr().err


def test_trailer_is_still_written(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    _guard(monkeypatch, action="skipped")
    with pytest.raises(SystemExit):
        exit_single_query(75, result={})
    assert f"{KANBAN_WORKER_EXIT_TRAILER}75" in capsys.readouterr().err


def _capture_exit(monkeypatch):
    import hermes_cli.quiet_single_query as qsq

    captured = {}

    def fake_exit(code, *, result=None):
        captured["code"] = code
        captured["result"] = result
        raise SystemExit(code)

    monkeypatch.setattr(qsq, "exit_single_query", fake_exit)
    return captured


def test_quiet_driver_passes_result_to_funnel(monkeypatch):
    """`-Q`: the settled turn result must reach the exit funnel, not just a code."""
    import contextlib

    import hermes_cli._early_recovery as early

    monkeypatch.setattr(early, "restore_interrupted_pull", lambda *a, **k: False)
    monkeypatch.setattr(early, "recover_if_needed", lambda *a, **k: False)
    import agent.kanban_turn_recovery as recovery
    import cli as cli_mod
    import hermes_cli.cli_single_query as csq

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    result = {"completed": False, "failed": False, "final_response": "ok", "messages": []}

    class _Agent:
        def run_conversation(self, **kwargs):
            return result

    class _Cli:
        session_id = "s-quiet"
        conversation_history = []
        agent = _Agent()
        _quiet_notify_linger_done = False

    captured = _capture_exit(monkeypatch)
    import hermes_cli.quiet_single_query as qsq
    monkeypatch.setattr(qsq, "bind_quiet_session_key", lambda key: contextlib.nullcontext())
    monkeypatch.setattr(qsq, "continue_quiet_notify_completions", lambda *a, **k: None)
    monkeypatch.setattr(qsq, "quiet_notify_linger_seconds", lambda: 0)
    monkeypatch.setattr(qsq, "write_turn_report", lambda *a, **k: None)
    monkeypatch.setattr(recovery, "recover_failed_kanban_turns", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_sync_cli_session_id_from_agent", lambda cli: None)
    monkeypatch.setattr(cli_mod, "_single_query_exit_code", lambda res, **kw: 0)

    with pytest.raises(SystemExit):
        csq._run_quiet_single_query(_Cli(), "hi")
    assert captured["result"] is result


def test_q_driver_passes_last_turn_result(monkeypatch):
    """`-q`: the non-quiet driver must hand `cli._last_turn_result` to the funnel."""
    import hermes_cli._early_recovery as early

    monkeypatch.setattr(early, "restore_interrupted_pull", lambda *a, **k: False)
    monkeypatch.setattr(early, "recover_if_needed", lambda *a, **k: False)
    import agent.kanban_turn_recovery as recovery
    import cli as cli_mod
    import hermes_cli.cli_single_query as csq
    import hermes_cli.observability.shared_metrics_startup as sms

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    result = {"completed": False, "failed": False, "messages": []}

    class _Console:
        def print(self, *a, **k):
            pass

    class _Cli:
        session_id = "s-q"
        _last_turn_result = result
        console = _Console()

        def _claim_active_session(self, *a, **k):
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, query, images=None):
            return "ok"

        def _print_exit_summary(self, **k):
            pass

    captured = _capture_exit(monkeypatch)
    monkeypatch.setattr(cli_mod, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli_mod, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli_mod, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda cli: None)
    monkeypatch.setattr(cli_mod, "_single_query_exit_code", lambda res, **kw: 0)
    monkeypatch.setattr(sms, "record_cli_one_shot_ready", lambda: None)
    monkeypatch.setattr(recovery, "recover_failed_kanban_turns", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        csq._run_single_query_mode(_Cli(), "hi", None, False, False)
    assert captured["result"] is result
