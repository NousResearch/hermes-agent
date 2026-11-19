"""A task that already decomposed is neither offered to the decomposer nor sent to the model again.

A root keeps its ``decomposed`` event when it is re-triaged, and the graph layer refuses to
decompose it a second time, so offering it only spent one model call per sweep. The exclusion is
part of the triage query, ahead of the scan limit, so held cards cannot starve eligible ones.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _triage(title, *, priority=0, tenant=None, decomposed=False):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title=title, triage=True, priority=priority, tenant=tenant)
        if decomposed:
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "decomposed", {"child_ids": []})
        return tid
    finally:
        conn.close()


def test_decomposed_task_is_not_sent_to_the_model_again(kanban_home, monkeypatch):
    tid = _triage("already split", decomposed=True)
    call_aux = MagicMock(side_effect=AssertionError("the model must not be called"))
    monkeypatch.setattr(decomp, "_call_aux", call_aux)

    outcome = decomp.decompose_task(tid)

    assert outcome.ok is False
    assert outcome.reason == "task already decomposed"
    call_aux.assert_not_called()


def test_list_triage_ids_skips_decomposed_tasks(kanban_home):
    fresh = _triage("fresh")
    _triage("already split", decomposed=True)

    assert decomp.list_triage_ids() == [fresh]


def test_decomposed_cards_do_not_fill_the_scan_window(kanban_home, monkeypatch):
    """Held cards ahead of the window must not hide eligible work behind them."""
    monkeypatch.setattr(decomp, "TRIAGE_SCAN_LIMIT", 2)
    _triage("held one", priority=9, decomposed=True)
    _triage("held two", priority=8, decomposed=True)
    eligible = _triage("eligible", priority=1)

    assert decomp.list_triage_ids() == [eligible]


def test_list_triage_ids_keeps_priority_then_age_order(kanban_home):
    low = _triage("low", priority=1)
    high = _triage("high", priority=5)
    older_tie = _triage("tie a", priority=3)
    newer_tie = _triage("tie b", priority=3)

    assert decomp.list_triage_ids() == [high, older_tie, newer_tie, low]


def test_list_triage_ids_keeps_the_tenant_filter(kanban_home):
    mine = _triage("mine", tenant="a")
    _triage("theirs", tenant="b")
    _triage("held", tenant="a", decomposed=True)

    assert decomp.list_triage_ids(tenant="a") == [mine]
