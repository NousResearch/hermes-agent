"""Integration smoke: kanban complete/block wire into the CDP pruner.

Proves the dispatcher-adjacent wiring in ``kanban_db._maybe_prune_cdp_after_transition``
actually fires ``tools.cdp_prune.prune_after_transition`` on completion (and only
when enabled), without opening any socket — the pruner call itself is stubbed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tools import cdp_prune as cp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _clean_prune_env(monkeypatch):
    for var in (
        "HERMES_CDP_PRUNE_ENABLED",
        "HERMES_CDP_PRUNE_ON_BLOCK",
        "HERMES_CDP_PRUNE_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _capture_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cp, "prune_after_transition", lambda **kw: calls.append(kw) or {"ok": True}
    )
    return calls


def test_completion_fires_pruner_when_enabled(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")
    calls = _capture_calls(monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="prune me", assignee="claude-code")
        kb.complete_task(conn, tid, result="done")
    assert len(calls) == 1
    assert calls[0]["event"] == "completed"
    assert calls[0]["task_id"] == tid


def test_completion_inert_when_disabled(kanban_home, monkeypatch):
    # default: HERMES_CDP_PRUNE_ENABLED unset -> pruner never invoked
    calls = _capture_calls(monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="leave tabs alone")
        kb.complete_task(conn, tid, result="done")
    assert calls == []


def test_pruner_failure_never_breaks_completion(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_CDP_PRUNE_ENABLED", "1")

    def _boom(**kw):
        raise RuntimeError("pruner exploded")

    monkeypatch.setattr(cp, "prune_after_transition", _boom)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="must still complete")
        # completion must succeed despite the pruner raising
        assert kb.complete_task(conn, tid, result="done") is True
        assert kb.get_task(conn, tid).status == "done"
