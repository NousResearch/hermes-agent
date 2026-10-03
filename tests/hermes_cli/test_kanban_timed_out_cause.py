"""A ``timed_out`` notice must name the cause its event payload records.

One defect, three placed facts (card t_1080ad83):

1. ``agent.turn_finalizer._record_kanban_budget_exhausted`` records the cause of an
   iteration-budget exhaustion as ``event_payload_extra={"budget_used": …, "budget_max": …}``
   and no ``limit_seconds`` — only ``enforce_max_runtime`` (the runtime cap) sets that key.
2. ``_record_task_failure``'s non-trip path dropped the extra, so the ``timed_out`` event it
   appended carried only the free-text ``error``/``failures``/``retry_status``. Measured on the
   ops board: ``t_82490443`` event 96801, and all 19 ``timed_out`` events in the 3 days to
   2026-09-27 carried no ``limit_seconds``.
3. Both notice formatters then read the absent key as ``0`` and printed a runtime cap the worker
   never hit — ``max_runtime=0s`` on the board, "its time limit" on Telegram — so the recorded
   cause was invisible exactly when the operator needed to act on it.

The recorder and its two formatters are one invariant, so they are pinned in one file: the
production caller is driven into a real (temp) board and the payload it actually recorded is
rendered by both formatters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# A runtime-cap stop: the ONLY cause that records ``limit_seconds``.
CAP_PAYLOAD = {"limit_seconds": 900, "failures": 1, "retry_status": "ready"}
# A timeout whose caller recorded no cause fields at all.
CAUSELESS_PAYLOAD = {"failures": 1, "retry_status": "ready"}


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated board the production caller reaches through ``kbc.connect()``.

    ``HERMES_KANBAN_DB`` pins the DB ahead of ``HERMES_HOME`` in ``kanban_db_path()``, so a
    worker shell that exports it (every dispatched worker does) would otherwise point these
    writes at the live board; the conftest ``kanban_write_guard`` turns that into a test error,
    but the pin is dropped here so the tests exercise the temp board they mean to.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exhaust_budget(conn, *, used: int = 200, maximum: int = 200):
    """Exhaust a claimed card's iteration budget through the production caller.

    Returns ``(task_id, event_row)`` for the ``timed_out`` event it recorded.
    """
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    tid = kb.create_task(conn, title="out of budget", assignee="platform-worker")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    _record_kanban_budget_exhausted(tid, used, maximum, logging.getLogger("test-budget"))
    event = conn.execute(
        "SELECT id, kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    return tid, event


def _run_metadata(conn, task_id: str) -> dict:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,)
    ).fetchone()
    return json.loads(row["metadata"]) if row and row["metadata"] else {}


def _board_text(task_id: str, payload: dict) -> str:
    from tui_gateway.session_notifications import _format_kanban_event_text

    task = SimpleNamespace(title="out of budget", assignee="platform-worker")
    return _format_kanban_event_text(
        {"task_id": task_id}, task, SimpleNamespace(kind="timed_out", payload=payload), "ops"
    )


def _telegram_text(payload: dict) -> str:
    from gateway.kanban_watchers_notifier import _EVENT_FORMATTERS

    msg, _, _ = _EVENT_FORMATTERS["timed_out"](
        SimpleNamespace(payload=payload), SimpleNamespace(head="t_abc123")
    )
    return msg


def test_budget_exhaustion_records_the_cause_on_the_event(kanban_home):
    """The extra the caller passes reaches the event payload, not the floor."""
    with kbc.connect() as conn:
        _tid, event = _exhaust_budget(conn)
    assert event["kind"] == "timed_out"
    payload = json.loads(event["payload"])
    assert payload["budget_used"] == 200
    assert payload["budget_max"] == 200
    # The runtime cap is a DIFFERENT cause; a budget exhaustion never has one.
    assert "limit_seconds" not in payload
    assert payload["retry_status"] == "ready"


def test_budget_exhaustion_keeps_the_cause_on_the_run_row(kanban_home):
    """``detail`` is the run row's metadata too, so the cause survives in run history."""
    with kbc.connect() as conn:
        tid, _event = _exhaust_budget(conn)
        metadata = _run_metadata(conn, tid)
    assert metadata["budget_used"] == 200
    assert metadata["budget_max"] == 200
    assert metadata["failures"] == 1


def test_board_notice_names_the_budget_it_recorded(kanban_home):
    """``max_runtime=0s`` was a cap that never existed; the notice names the real cause."""
    with kbc.connect() as conn:
        tid, event = _exhaust_budget(conn)
    text = _board_text(tid, json.loads(event["payload"]))
    assert "timed out (exhausted its turn budget (200/200)); will retry" in text
    assert "max_runtime=0s" not in text


def test_board_notice_still_names_a_recorded_runtime_cap():
    text = _board_text("t_abc123", dict(CAP_PAYLOAD))
    assert "timed out (max_runtime=900s); will retry" in text


def test_board_notice_says_so_when_no_cause_was_recorded():
    text = _board_text("t_abc123", dict(CAUSELESS_PAYLOAD))
    assert "timed out (cause not recorded); will retry" in text
    assert "max_runtime=" not in text


def test_telegram_notice_names_the_budget_it_recorded(kanban_home):
    with kbc.connect() as conn:
        _tid, event = _exhaust_budget(conn)
    msg = _telegram_text(json.loads(event["payload"]))
    assert "exhausted its turn budget (200/200) and was stopped" in msg
    assert "time limit" not in msg


def test_telegram_notice_still_names_the_minute_limit():
    msg = _telegram_text(dict(CAP_PAYLOAD))
    assert "ran past its 15-minute limit and was stopped" in msg


def test_telegram_notice_says_so_when_no_cause_was_recorded():
    msg = _telegram_text(dict(CAUSELESS_PAYLOAD))
    assert "was stopped (cause not recorded)" in msg
    assert "limit" not in msg
