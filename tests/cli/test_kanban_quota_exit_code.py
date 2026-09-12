"""Regression tests for the kanban quota-wall exit sentinel on the non-quiet path.

A dispatcher-spawned worker runs ``hermes -p <profile> chat -q "work kanban task <id>"``
— the NON-quiet one-shot path. Before this fix that path always exited 0, so a worker
that bailed because the provider rate-limited / exhausted quota looked like a clean exit
with no terminal ``kanban_complete`` / ``kanban_block`` call. The dispatcher's reap
classifier scores that a ``protocol_violation``, which *counts a failure* and auto-blocks
the card after ``kanban.failure_limit`` — for a quota wall that says nothing whatsoever
about the task.

``hermes_cli/kanban_db.py`` already defines ``KANBAN_RATE_LIMIT_EXIT_CODE`` (75,
EX_TEMPFAIL) and the whole ``rate_limited`` requeue path that does NOT count a failure.
The sentinel was only emitted from the ``-Q`` branch (``_run_quiet_single_query``), so it
was unreachable for every worker the dispatcher actually spawns.
"""

import pytest

import cli as cli_mod
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE

RL_CODE = KANBAN_RATE_LIMIT_EXIT_CODE


# ── _kanban_exit_code: the shared decision ───────────────────────────────────

@pytest.mark.parametrize(
    "result, kanban, expected",
    [
        # Success — plain 0, whatever the env.
        ({"failed": False}, False, 0),
        ({"failed": False}, True, 0),
        ({"final_response": "ok"}, False, 0),
        ({"final_response": "ok"}, True, 0),
        # A result that is not a dict at all is success (chat() returns a str).
        ("just a string", False, 0),
        (None, False, 0),
        # Generic failure — 1, not the sentinel.
        ({"failed": True}, False, 1),
        ({"failed": True}, True, 1),
        ({"failed": True, "failure_reason": "context_overflow"}, True, 1),
        ({"failed": True, "failure_reason": "server_error"}, True, 1),
        # Quota wall in a kanban worker — the sentinel.
        ({"failed": True, "failure_reason": "rate_limit"}, True, RL_CODE),
        ({"failed": True, "failure_reason": "billing"}, True, RL_CODE),
        # Same quota failure OUTSIDE a kanban worker — must stay 1 so the plain
        # 0/1 contract that automation wrappers rely on is preserved.
        ({"failed": True, "failure_reason": "rate_limit"}, False, 1),
        ({"failed": True, "failure_reason": "billing"}, False, 1),
    ],
)
def test_kanban_exit_code(monkeypatch, result, kanban, expected):
    if kanban:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    assert cli_mod._kanban_exit_code(result) == expected


def test_quota_sentinel_is_ex_tempfail():
    """Guard the contract itself: 75 is BSD EX_TEMPFAIL, the conventional retry-later code."""
    assert RL_CODE == 75


# ── Production path: non-quiet `chat -q` exits with the sentinel ─────────────

def _run_main_non_quiet(monkeypatch, stashed_result):
    """Drive ``cli.main(query=..., quiet=False)`` with a stubbed CLI.

    Returns the SystemExit code raised, or None if main returned normally.
    """
    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = type("C", (), {"print": staticmethod(lambda *a, **k: None)})()
            self.session_id = "sq-quota-test"
            self.agent = type("A", (), {"session_id": "sq-quota-test"})()

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            return None

        def chat(self, query, images=None):
            # Mirrors the real mixin: chat() returns only text, so the full
            # result is stashed for the caller to inspect.
            self._last_conversation_result = stashed_result
            return "worker finished"

        def _print_exit_summary(self, clear_screen=True):
            return None

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda fake_cli: None)

    try:
        cli_mod.main(query="work kanban task t_deadbeef", quiet=False, toolsets="terminal")
    except SystemExit as exc:
        return exc.code
    return None


def test_non_quiet_path_exits_with_quota_sentinel(monkeypatch):
    """A quota wall in a kanban worker must exit 75, not 0."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    code = _run_main_non_quiet(
        monkeypatch, {"failed": True, "failure_reason": "rate_limit"}
    )
    assert code == RL_CODE, (
        "non-quiet chat -q in a kanban worker must exit with the EX_TEMPFAIL sentinel "
        "so the dispatcher requeues without counting a failure"
    )


def test_non_quiet_path_billing_also_sentinel(monkeypatch):
    """Credit exhaustion ('billing') is the same class of wall as a rate limit."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    code = _run_main_non_quiet(
        monkeypatch, {"failed": True, "failure_reason": "billing"}
    )
    assert code == RL_CODE


def test_non_quiet_path_success_still_exits_zero(monkeypatch):
    """A successful non-kanban one-shot must not gain a spurious non-zero exit."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    code = _run_main_non_quiet(monkeypatch, {"failed": False, "final_response": "ok"})
    assert code is None, "a clean run must return normally, not raise SystemExit"


def test_non_quiet_path_generic_failure_is_one(monkeypatch):
    """A genuine task failure in a worker still exits 1 (breaker must be able to trip)."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    code = _run_main_non_quiet(
        monkeypatch, {"failed": True, "failure_reason": "context_overflow"}
    )
    assert code == 1
