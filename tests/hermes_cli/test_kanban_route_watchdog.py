from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_route_watchdog import evaluate_route, load_watchdog_config


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _roster(*profiles: tuple[str, str]) -> list[dict]:
    return [
        {"name": name, "description": description, "description_auto": False}
        for name, description in profiles
    ]


def test_dispatch_watchdog_blocks_stranded_orchestrator(kanban_home, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: _roster(
            ("orchestrator", "route plan decompose review coordination"),
            ("sentinel", "runtime watchdog implementation tests patch audit monitoring"),
        ),
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Implement runtime watchdog overlay and tests",
            body="Patch dispatch logic, write tests, verify behavior.",
            assignee="orchestrator",
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id = ?", (task_id,))
        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: pytest.fail("spawn_fn should not run"),
            route_watchdog={
                "mode": "hold",
                "min_score": 0.10,
                "min_margin": 0.05,
                "orchestrator_profile": "orchestrator",
            },
        )
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)

    assert task is not None
    assert task.status == "blocked"
    assert res.spawned == []
    assert res.route_watchdog_hits == [(task_id, "stranded_orchestrator", "blocked")]
    assert comments
    assert comments[-1].author == "route-watchdog"
    assert "stranded_orchestrator" in comments[-1].body


def test_dispatch_watchdog_blocks_brand_mismatch(kanban_home, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: _roster(
            ("sentinel", "runtime watchdog monitoring ops audits"),
            ("mark", "marketing launch copy ads social growth content"),
        ),
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Write launch ad copy and social campaign",
            body="Draft campaign messaging, paid ads, and growth content.",
            assignee="sentinel",
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id = ?", (task_id,))
        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: pytest.fail("spawn_fn should not run"),
            route_watchdog={"mode": "hold", "min_score": 0.10, "min_margin": 0.05},
        )
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)

    assert task is not None
    assert task.status == "blocked"
    assert res.route_watchdog_hits == [(task_id, "brand_mismatch", "blocked")]
    assert comments
    assert comments[-1].author == "route-watchdog"
    assert "brand_mismatch" in comments[-1].body
    assert "mark" in comments[-1].body


def test_watchdog_low_confidence_for_ambiguous_route():
    config = load_watchdog_config({"mode": "hold", "min_score": 0.60, "min_margin": 0.20})
    decision = evaluate_route(
        {
            "id": "t_demo",
            "title": "Need help",
            "body": "Please handle this soon.",
            "assignee": "sentinel",
        },
        config=config,
        roster=_roster(
            ("sentinel", "ops monitoring audits"),
            ("mark", "marketing growth content"),
        ),
    )

    assert decision is not None
    assert decision.kind == "low_confidence"
    assert "threshold" in decision.reason or "ambiguous" in decision.detail
