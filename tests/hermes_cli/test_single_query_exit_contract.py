"""One-shot runs map their outcome onto an exit code, on BOTH one-shot paths.

The Kanban dispatcher spawns workers as ``hermes ... chat -q <prompt>`` — the
non-quiet single-query path. That path had no exit contract at all: it ran the
turn and fell through to an implicit 0 whatever happened. The dispatcher's reap
classifier reads rc=0 with the task still ``running`` as a protocol violation
and blocks the card on the first occurrence, so one provider quota wall killed
the card permanently along with every card queued behind it.

The EX_TEMPFAIL sentinel that exists to prevent exactly that was wired into the
``-Q`` path only. These tests pin the contract on both.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


# --------------------------------------------------------------------------
# the shared mapping
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_inherited_kanban_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


@pytest.mark.parametrize("result", [
    {"final_response": "ok"},
    {"failed": False},
    None,
    "not a dict",
])
def test_a_run_that_did_not_fail_exits_zero(result):
    assert cli._single_query_exit_code(result) == 0


def test_a_plain_failure_exits_one():
    assert cli._single_query_exit_code({"failed": True, "failure_reason": "tool_error"}) == 1


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_a_kanban_worker_on_a_quota_wall_exits_with_the_sentinel(monkeypatch, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    result = {"failed": True, "failure_reason": reason}
    assert cli._single_query_exit_code(result) == KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_the_sentinel_is_for_kanban_workers_only(reason):
    """A human's one-shot run that hit a quota wall is an ordinary failure."""
    assert "HERMES_KANBAN_TASK" not in os.environ
    assert cli._single_query_exit_code({"failed": True, "failure_reason": reason}) == 1


def test_a_kanban_worker_failing_for_another_reason_still_exits_one(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert cli._single_query_exit_code({"failed": True, "failure_reason": "tool_error"}) == 1


# --------------------------------------------------------------------------
# the path the dispatcher actually spawns: `chat -q`, non-quiet
# --------------------------------------------------------------------------

def _fake_cli(turn_result):
    """A CLI stub exercising only what the non-quiet one-shot tail touches."""
    return SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=SimpleNamespace(print=lambda *a, **k: None),
        _show_security_advisories=lambda: None,
        chat=lambda *a, **k: "response",
        _print_exit_summary=lambda **k: None,
        _last_turn_result=turn_result,
    )


def _run_non_quiet(monkeypatch, turn_result):
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    stub = _fake_cli(turn_result)
    try:
        cli._run_single_query_mode(stub, "work kanban task t_abc123", None, False, True)
    except SystemExit as exc:
        return exc.code
    return None  # fell through without exiting


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_dispatcher_spawned_worker_signals_a_quota_wall_not_a_protocol_violation(monkeypatch, reason):
    """The regression. rc=0 here is read as a protocol violation and blocks the card."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason})
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE


def test_dispatcher_spawned_worker_reports_an_ordinary_failure_as_nonzero(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": "tool_error"})
    assert code == 1


def test_dispatcher_spawned_worker_that_succeeded_exits_zero(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code = _run_non_quiet(monkeypatch, {"final_response": "done"})
    assert code == 0


def test_a_human_one_shot_run_is_unaffected(monkeypatch):
    """No HERMES_KANBAN_TASK: the path must not start exiting non-zero on people."""
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": "rate_limit"})
    assert code is None


# --------------------------------------------------------------------------
# the plumbing that makes the outcome reachable from that path
# --------------------------------------------------------------------------

def test_settling_a_turn_records_the_raw_result():
    """chat() returns a rendered string; the exit mapping needs the result dict."""
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    stub = SimpleNamespace(
        _prompt_start_time=None,
        _prompt_duration=0.0,
        _flush_stream=lambda: None,
        conversation_history=[],
        agent=None,
    )
    result = {"failed": True, "failure_reason": "rate_limit"}
    turn = SimpleNamespace(
        result=result, use_streaming_tts=False, text_queue=None, tts_thread=None,
    )

    CLIChatTurnMixin._chat_settle_turn(stub, turn)

    assert stub._last_turn_result is result


def test_the_recorded_result_defaults_to_none():
    """A CLI that never completed a turn must map to 0, not raise."""
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    assert CLIChatTurnMixin._last_turn_result is None
    assert cli._single_query_exit_code(CLIChatTurnMixin._last_turn_result) == 0
