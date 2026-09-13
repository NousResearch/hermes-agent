"""A failed one-shot turn must never exit 0.

The kanban dispatcher books ``rc=0`` on a still-``running`` task as a *protocol
violation* (worker exited without ``kanban_complete``/``kanban_block``) — which is
wrong twice over: the work was usually in flight, and a provider/quota wall is not
a task failure at all. Only the ``-Q`` (quiet) one-shot branch propagated a failure
exit code, so every non-quiet worker (the dispatcher spawns ``hermes -p <profile>
--cli chat -q …``) reported a mid-flight provider death as "worker exited cleanly
(rc=0) without calling kanban_complete or kanban_block".

These pin the behaviour contract of the shared exit-code decision and of the
non-quiet route that must use it.
"""

from __future__ import annotations

import cli as cli_mod
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


class _StubCLI:
    """Minimum surface ``_run_single_query_mode`` touches on the non-quiet route."""

    def __init__(self, turn_result, response="done"):
        self._last_turn_result = turn_result
        self._response = response
        self.chatted = False
        self.exit_summary_printed = False

        class _Console:
            def print(self, *a, **k):
                pass

        self.console = _Console()

    def _claim_active_session(self, *a, **k):
        return True

    def _show_security_advisories(self):
        pass

    def chat(self, query, images=None):
        self.chatted = True
        return self._response

    def _print_exit_summary(self, **kwargs):
        self.exit_summary_printed = True


def _run_nonquiet(monkeypatch, turn_result, response="done"):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    stub = _StubCLI(turn_result, response=response)
    try:
        cli_mod._run_single_query_mode(stub, "work kanban task t_x", None, False, True)
    except SystemExit as exc:
        return stub, exc.code
    return stub, None


def test_failed_turn_is_nonzero_off_kanban(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert cli_mod._single_query_exit_code({"failed": True, "error": "boom"}) == 1


def test_clean_turn_is_zero(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert cli_mod._single_query_exit_code({"completed": True, "failed": False}) == 0
    assert cli_mod._single_query_exit_code(None) == 0


def test_kanban_quota_wall_keeps_tempfail_sentinel(monkeypatch):
    """A rate-limit/billing wall must not be booked as a task failure."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    for reason in ("rate_limit", "billing"):
        assert cli_mod._single_query_exit_code(
            {"failed": True, "failure_reason": reason}
        ) == KANBAN_RATE_LIMIT_EXIT_CODE
    assert cli_mod._single_query_exit_code(
        {"failed": True, "failure_reason": "api_error"}
    ) == 1


def test_nonquiet_one_shot_propagates_failed_turn(monkeypatch):
    """The route the kanban dispatcher spawns must exit nonzero on a failed turn."""
    stub, code = _run_nonquiet(monkeypatch, {"failed": True, "error": "provider down"})
    assert stub.chatted, "the non-quiet route must still run the turn"
    assert stub.exit_summary_printed, "the exit summary is part of the route's contract"
    assert code == 1, (
        f"a failed non-quiet one-shot turn exited {code!r}: rc=0 is read by the kanban "
        "dispatcher as a protocol violation, hiding provider crashes"
    )


def test_nonquiet_one_shot_still_exits_zero_on_success(monkeypatch):
    stub, code = _run_nonquiet(monkeypatch, {"completed": True, "failed": False})
    assert stub.chatted
    assert code == 0, f"a successful one-shot turn must exit 0, got {code!r}"
