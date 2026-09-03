"""Exit-code contract for the human-facing ``-q`` single-query path.

A kanban worker rate-limited by the provider must exit with the EX_TEMPFAIL
sentinel so the dispatcher requeues it instead of classifying the run as a
protocol violation.
"""

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)


def _install_fake_cli(monkeypatch, calls, chat_result):
    class _Console:
        def print(self, *_args, **_kwargs):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "single-query-session"
            self._last_turn_result = None
            self.agent = SimpleNamespace(
                session_id="single-query-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            self._last_turn_result = chat_result
            return chat_result.get("final_response") if chat_result else None

        def _print_exit_summary(self, clear_screen=True):
            calls.append("summary")

    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )


def _run(monkeypatch, calls, chat_result):
    _install_fake_cli(monkeypatch, calls, chat_result)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="hello", quiet=False, toolsets="terminal")
    return exc_info.value.code


def test_single_query_kanban_rate_limit_exits_tempfail(monkeypatch):
    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "42")

    code = _run(
        monkeypatch,
        calls,
        {"final_response": "", "failed": True, "failure_reason": "rate_limit"},
    )

    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert calls[-1] == ("finalize", "single-query-session")


def test_single_query_non_kanban_failure_exits_one(monkeypatch):
    calls = []
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    code = _run(
        monkeypatch,
        calls,
        {"final_response": "", "failed": True, "failure_reason": "rate_limit"},
    )

    assert code == 1
    assert calls[-1] == ("finalize", "single-query-session")


def test_single_query_success_exits_zero(monkeypatch):
    calls = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "42")

    code = _run(monkeypatch, calls, {"final_response": "done", "failed": False})

    assert code == 0
    assert calls[-1] == ("finalize", "single-query-session")
