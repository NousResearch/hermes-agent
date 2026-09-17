"""Regression: circuit-breaker trips (``gave_up`` -> status='blocked') must
fire the same ``kanban_task_blocked`` lifecycle hook as an explicit
``kanban_block()`` call.

Before this fix, ``_record_task_failure``'s trip branch flipped
``status='blocked'`` directly via SQL and never called ``_fire_task_hook``,
so any subscriber wired to ``kanban_task_blocked`` (e.g. the Decision HUD
escalation bridge, ``~/.hermes/scripts/kanban_escalation_bridge.py``) never
fired for a task that crashed/spawn-failed into the failure-limit trip —
only for a task a worker or reviewer explicitly blocked via ``kanban_block``.
Live symptom: several tasks landed in status=blocked with no Decision HUD
card and no classifier task spawned.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_gave_up_trip_fires_kanban_task_blocked_hook(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="flaky spawn")
        kb.claim_task(conn, tid)

        calls = []
        with mock.patch.object(
            kb, "_fire_task_hook", side_effect=lambda *a, **k: calls.append((a, k))
        ):
            tripped = kbd._record_task_failure(
                conn, tid, "boom",
                outcome="crashed", force_trip=True,
                release_claim=True, end_run=True,
            )

        assert tripped is True
        assert kb.get_task(conn, tid).status == "blocked"
        assert len(calls) == 1, "kanban_task_blocked hook must fire exactly once on trip"
        args, kwargs = calls[0]
        assert args[0] == "kanban_task_blocked"
        assert args[2] == tid
        assert kwargs.get("reason") == "boom"


def test_gave_up_non_trip_does_not_fire_blocked_hook(kanban_home: Path) -> None:
    """A failure that does NOT cross the threshold stays out of 'blocked'
    and must not fire the hook (would falsely escalate an in-progress retry)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="flaky spawn, still retrying")
        kb.claim_task(conn, tid)

        calls = []
        with mock.patch.object(
            kb, "_fire_task_hook", side_effect=lambda *a, **k: calls.append((a, k))
        ):
            tripped = kbd._record_task_failure(
                conn, tid, "transient hiccup",
                outcome="crashed", failure_limit=5,
                release_claim=True, end_run=True,
            )

        assert tripped is False
        assert kb.get_task(conn, tid).status != "blocked"
        assert calls == []
