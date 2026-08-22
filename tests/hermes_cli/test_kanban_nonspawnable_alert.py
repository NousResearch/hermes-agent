"""Silent-deadlock fix: skipped_nonspawnable must alert, not rot silently.

Regression coverage for #kanban-nonspawnable-deadlock (2026-08-19): a card
whose ``assignee`` names neither a live Hermes profile nor a registered
terminal lane used to be bucketed into ``skipped_nonspawnable`` with ZERO
``task_event`` and ZERO comment written to the card — indistinguishable,
from the card itself, from a healthy human-pulled terminal lane. A single
retired profile silently deadlocked a whole dependency chain because
nothing ever surfaced the skip.

Fix: ``_emit_nonspawnable_alert`` writes one ``nonspawnable_alerted``
task_event + one card comment the first time a given (task, assignee)
pair is skipped as nonspawnable, then stays silent on later ticks for the
same assignee (so long-lived, intentional terminal lanes don't get
repeat noise).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns):
    def _spawn(task, workspace, **kwargs):
        spawns.append(task.id)
        return 4242

    return _spawn


def test_nonspawnable_ready_task_gets_event_and_comment(kanban_home, monkeypatch):
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: False)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="orphaned card", assignee="upx-product-owner",
        )
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory([]), max_in_progress=2,
        )
        assert task_id in res.skipped_nonspawnable

        events = kb.list_events(conn, task_id)
        alert_events = [e for e in events if e.kind == "nonspawnable_alerted"]
        assert len(alert_events) == 1
        assert alert_events[0].payload["assignee"] == "upx-product-owner"

        comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1
        assert "upx-product-owner" in comments[0].body
        assert "does not resolve" in comments[0].body


def test_nonspawnable_alert_does_not_repeat_across_ticks(kanban_home, monkeypatch):
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: False)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="orphaned card", assignee="upx-devops",
        )
        for _ in range(3):
            res = kb.dispatch_once(
                conn, spawn_fn=_fake_spawn_factory([]), max_in_progress=2,
            )
            assert task_id in res.skipped_nonspawnable

        events = kb.list_events(conn, task_id)
        alert_events = [e for e in events if e.kind == "nonspawnable_alerted"]
        assert len(alert_events) == 1, (
            "alert must fire once per assignee, not once per dispatch tick "
            "(the whole point is not spamming a long-lived terminal lane)"
        )
        comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1


def test_nonspawnable_review_task_gets_event_and_comment(kanban_home, monkeypatch):
    """The review-lane skip site gets the same alert, independently."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    monkeypatch.setattr(profmod, "profile_exists", lambda name: False)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="review card", assignee="upx-product-owner",
        )
        conn.execute(
            "UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,)
        )
        conn.commit()
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory([]), max_in_progress=2,
        )
        assert task_id in res.skipped_nonspawnable

        events = kb.list_events(conn, task_id)
        alert_events = [e for e in events if e.kind == "nonspawnable_alerted"]
        assert len(alert_events) == 1

        comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1
