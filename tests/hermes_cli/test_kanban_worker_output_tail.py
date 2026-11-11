"""A dead worker's quoted cause must come from ITS attempt, not the previous one's ending.

The per-task worker log is append-only across attempts, so it accumulates one CLI exit summary
(``Resume this session with:``) per attempt that reached the epilogue. ``_worker_final_output``
truncated at the LAST such marker, which threw away everything the newest attempt printed after
it — exactly what a fast-failing attempt prints. The board then quoted the previous attempt's
closing text as the crash cause (t_d9e8f954 runs 679/680: recorded cause was run 678's bullets,
while the log's own tail read ``Error: Unknown skill(s): sdlc-review``).
"""
from hermes_cli import kanban_db_dispatch as dispatch

PREVIOUS_ATTEMPT = """\
  Bitwarden Secrets Manager: applied 24 secrets
Query: work kanban task t_d9e8f954
╭─ ☤ Hermes ──────────────────────────────────────────────╮
Task t_d9e8f954 is implemented and handed to the review lane.
- I did not create a follow-up card for the harness leak.
- config_defaults.py also carries an unrelated in-flight hunk.
╰─────────────────────────────────────────────────────────╯

Resume this session with:
  hermes --resume 20261013_125944_ceea09 -p worker
  hermes -c "Dispatcher review-assignment" -p worker

Session:        20261013_125944_ceea09
Title:          Dispatcher review-assignment
Duration:       54m 23s
Messages:       315 (1 user, 313 tool calls)

[kanban-worker-exit] rc=0
"""

FAST_FAILURE = """\
  Bitwarden Secrets Manager: applied 25 secrets
Query: work kanban task t_d9e8f954
\x1b[2;3mInitializing agent...\x1b[0m
Error: Unknown skill(s): sdlc-review
"""


def _output(monkeypatch, text):
    monkeypatch.setattr(dispatch._kb, "read_worker_log", lambda *_a, **_k: text)
    return dispatch._worker_final_output("t_d9e8f954", board="phase1d-canary")


def test_fast_failure_after_a_completed_attempt_quotes_its_own_reason(monkeypatch):
    detail = _output(monkeypatch, PREVIOUS_ATTEMPT + FAST_FAILURE)

    assert "Unknown skill(s): sdlc-review" in detail
    # The previous attempt's ending is NOT this attempt's cause.
    assert "follow-up card" not in detail
    assert "Resume this session with" not in detail


def test_completed_attempt_still_quotes_its_own_final_text(monkeypatch):
    detail = _output(monkeypatch, PREVIOUS_ATTEMPT)

    assert "follow-up card" in detail
    assert "Resume this session with" not in detail
    # Exit-summary chrome never reaches the board.
    assert "20261013_125944_ceea09" not in detail
    assert "Duration:" not in detail


def test_managed_worker_refusal_keeps_its_session_pointer(monkeypatch):
    managed = (
        "Session: local-39089261ac6115bb8d1954980cc3a7b914a9986a7efba063ad1e5bad0f71bb24\n"
        "Kanban owner refused: Gateway disconnected; turn outcome is unknown. "
        "Resume the printed session ID to check whether it completed or was interrupted; "
        "do not resend the work.\n\n[kanban-worker-exit] rc=1\n"
    )

    detail = _output(monkeypatch, managed)

    assert detail.startswith("Session: local-39089261ac61")
    assert "Kanban owner refused: Gateway disconnected" in detail
    assert "[kanban-worker-exit]" not in detail


def test_missing_or_empty_log_returns_empty(monkeypatch):
    assert _output(monkeypatch, None) == ""
    assert _output(monkeypatch, "") == ""
