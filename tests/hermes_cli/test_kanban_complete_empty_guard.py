"""hermes kanban complete must reject an empty handoff (no summary, no result).

The kanban_complete tool handler (tools/kanban_tools.py) refuses completions
that carry neither a summary nor a result:

    if not (summary or result):
        return tool_error("provide at least one of: summary (preferred), result")

The CLI path (hermes_cli.kanban._cmd_complete) must apply the same guard, or
workers can mark cards done with a NULL result via the CLI even though the
tool path forbids it.

Uses mocks for kb.get_task / kb.complete_task (same idiom as the CLI judge
gate tests); the guard logic is the unit under test.
"""

import argparse
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

from hermes_cli.kanban import _cmd_complete


def _run(monkeypatch, *, task_ids, summary=None, result=None):
    fake_task = types.SimpleNamespace(
        goal_mode=False,
        title="Some task",
        body="",
    )
    fake_conn = MagicMock()
    complete_calls: list = []

    def fake_connect_closing():
        @contextmanager
        def _cm():
            yield fake_conn

        return _cm()

    def fake_complete_task(conn, tid, **kw):
        complete_calls.append(tid)
        return True

    monkeypatch.setattr("hermes_cli.kanban.kb.get_task", lambda conn, tid: fake_task)
    monkeypatch.setattr("hermes_cli.kanban.kb.complete_task", fake_complete_task)
    monkeypatch.setattr("hermes_cli.kanban.kb.connect_closing", fake_connect_closing)
    monkeypatch.setattr("hermes_cli.kanban._worker_run_id_for", lambda _: None)

    args = argparse.Namespace(
        task_ids=list(task_ids), summary=summary, result=result, metadata=None
    )
    return _cmd_complete(args), complete_calls


def test_complete_without_summary_or_result_is_rejected(monkeypatch, capsys):
    rc, complete_calls = _run(monkeypatch, task_ids=["t1"])
    assert rc != 0, "empty completion must produce a non-zero exit code"
    assert complete_calls == [], (
        "complete_task must NOT be invoked for an empty completion"
    )
    err = capsys.readouterr().err
    assert "provide at least one of" in err, (
        "error message must match the kanban_complete tool handler"
    )


def test_complete_multi_id_without_result_is_rejected(monkeypatch, capsys):
    # Multi-id complete only accepts --result; the empty guard must still fire.
    rc, complete_calls = _run(monkeypatch, task_ids=["t1", "t2"])
    assert rc != 0
    assert complete_calls == []
    assert "provide at least one of" in capsys.readouterr().err


def test_complete_with_result_only_is_accepted(monkeypatch):
    rc, complete_calls = _run(monkeypatch, task_ids=["t1"], result="PASS abc123")
    assert rc == 0
    assert complete_calls == ["t1"]


def test_complete_with_summary_only_is_accepted(monkeypatch):
    rc, complete_calls = _run(monkeypatch, task_ids=["t1"], summary="did the thing")
    assert rc == 0
    assert complete_calls == ["t1"]
