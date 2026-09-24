"""A failed quiet turn that already printed text must still show the failure reason (#121299)."""

from __future__ import annotations

from types import SimpleNamespace

import cli
from hermes_cli import quiet_single_query as qsq


def _run(monkeypatch, result):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(qsq, "continue_quiet_notify_completions", lambda *a, **k: None)

    def run_conversation(**kwargs):
        return result

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="s-1")
    try:
        cli._run_quiet_single_query(
            SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1"), "go",
        )
    except SystemExit as exc:
        return exc.code
    raise AssertionError("quiet single query did not exit")


def test_partial_answer_still_prints_the_failure_reason(monkeypatch, capsys):
    code = _run(monkeypatch, {
        "final_response": "PARTIAL-B",
        "error": "turn ended status=failed: FAIL-MARKER-89 stream disconnected",
        "partial": True,
        "completed": False,
    })
    out = capsys.readouterr()
    assert code == 1
    assert "PARTIAL-B" in out.out
    assert "FAIL-MARKER-89" in out.err


def test_failure_only_turn_prints_the_error_once(monkeypatch, capsys):
    code = _run(monkeypatch, {
        "final_response": "",
        "error": "turn ended status=failed: FAIL-MARKER-88 stream disconnected",
        "partial": True,
        "completed": False,
    })
    out = capsys.readouterr()
    assert code == 1
    assert "FAIL-MARKER-88" not in out.out
    assert out.err.count("FAIL-MARKER-88") == 1
