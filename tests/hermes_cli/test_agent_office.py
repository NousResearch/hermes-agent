from __future__ import annotations

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


def test_event_aliases_map_native_kanban_events_to_office_vocabulary():
    from hermes_cli import agent_office

    assert agent_office.office_event_alias("created") == "card.created"
    assert agent_office.office_event_alias("specified") == "triaged"
    assert agent_office.office_event_alias("promoted") == "ready_for_assignment"
    assert agent_office.office_event_alias("claimed") == "started"
    assert agent_office.office_event_alias("spawned") == "started"
    assert agent_office.office_event_alias("completed") == "completed"


def test_agent_office_tick_specs_routes_and_dispatches_ready_work(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    from hermes_cli import agent_office
    from hermes_cli.kanban_specify import SpecifyOutcome

    spawned = []

    def fake_specify(task_id: str, *, author=None, timeout=None):
        with kb.connect() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title="Write README documentation",
                body="Create docs for the project.",
                author=author or "agent-office",
            )
        return SpecifyOutcome(task_id, ok, "specified", "Write README documentation")

    def fake_spawn(task, workspace, board=None):
        spawned.append((task.id, task.assignee, board))
        return 12345

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="make project docs", triage=True)
        result = agent_office.tick(conn, board="inbox", specify_fn=fake_specify, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)
        events = [e.kind for e in kb.list_events(conn, tid)]

    assert result.specified == [tid]
    assert result.routed == [(tid, "docs")]
    assert [(x[0], x[1]) for x in result.dispatched.spawned] == [(tid, "docs")]
    assert spawned == [(tid, "docs", "inbox")]
    assert task.assignee == "docs"
    assert task.status == "running"
    assert "specified" in events
    assert "assigned" in events
    assert "claimed" in events


def test_route_task_prefers_role_keywords_over_chief_fallback(kanban_home):
    from hermes_cli import agent_office

    with kb.connect() as conn:
        docs = kb.create_task(conn, title="Write user documentation")
        sec = kb.create_task(conn, title="Threat model auth token handling")
        broad = kb.create_task(conn, title="Make the product better")

        assert agent_office.route_task(kb.get_task(conn, docs)).profile == "docs"
        assert agent_office.route_task(kb.get_task(conn, sec)).profile == "security"
        assert agent_office.route_task(kb.get_task(conn, broad)).profile == "chief"


def test_tick_falls_back_when_llm_specifier_unavailable(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    from hermes_cli import agent_office
    from hermes_cli.kanban_specify import SpecifyOutcome

    spawned = []

    def unavailable(task_id: str, *, author=None, timeout=None):
        return SpecifyOutcome(task_id, False, "auxiliary client unavailable", None)

    def fake_spawn(task, workspace, board=None):
        spawned.append((task.id, task.assignee, board))
        return 12345

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Write README documentation",
            body="Create docs for the project.",
            triage=True,
        )
        result = agent_office.tick(conn, board="inbox", specify_fn=unavailable, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)

    assert result.specified == [tid]
    assert result.specify_failed == {}
    assert result.routed == [(tid, "docs")]
    assert spawned == [(tid, "docs", "inbox")]
    assert task.status == "running"


def test_tick_honors_disabled_config(kanban_home, monkeypatch, all_assignees_spawnable):
    from hermes_cli import agent_office
    from hermes_cli.kanban_specify import SpecifyOutcome

    monkeypatch.setattr(agent_office, "office_config", lambda: {
        "enabled": False,
        "board": "inbox",
        "auto_specify": True,
        "auto_route": True,
        "auto_supervise": True,
    })

    def fake_specify(task_id: str, *, author=None, timeout=None):
        raise AssertionError("disabled office tick must not specify")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Write README documentation", triage=True)
        result = agent_office.tick(conn, board="inbox", specify_fn=fake_specify)
        task = kb.get_task(conn, tid)

    assert result.specified == []
    assert result.routed == []
    assert result.dispatched.spawned == []
    assert task.status == "triage"


def test_tick_honors_auto_route_config(kanban_home, monkeypatch, all_assignees_spawnable):
    from hermes_cli import agent_office
    from hermes_cli.kanban_specify import SpecifyOutcome

    monkeypatch.setattr(agent_office, "office_config", lambda: {
        "enabled": True,
        "board": "inbox",
        "auto_specify": True,
        "auto_route": False,
        "auto_supervise": True,
    })

    def fake_specify(task_id: str, *, author=None, timeout=None):
        with kb.connect() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title="Write README documentation",
                body="Create docs for the project.",
                author=author or "agent-office",
            )
        return SpecifyOutcome(task_id, ok, "specified", "Write README documentation")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="make project docs", triage=True)
        result = agent_office.tick(conn, board="inbox", specify_fn=fake_specify)
        task = kb.get_task(conn, tid)

    assert result.specified == [tid]
    assert result.routed == []
    assert result.dispatched.spawned == []
    assert task.status == "ready"
    assert task.assignee is None


def test_route_ready_unassigned_balances_across_configured_role_seats(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    from hermes_cli import agent_office

    monkeypatch.setattr(agent_office, "office_config", lambda: {
        "enabled": True,
        "board": "inbox",
        "auto_specify": True,
        "auto_route": True,
        "auto_supervise": True,
        "default_mode": "yolo",
        "workspace_root": "/Users/akhilkinnera/Documents/My Workspace",
        "role_seats": {"coder": ["coder", "coder-1", "coder-2"]},
    })

    with kb.connect() as conn:
        first = kb.create_task(conn, title="implement feature alpha")
        second = kb.create_task(conn, title="implement feature beta")
        third = kb.create_task(conn, title="implement feature gamma")
        routed = agent_office.route_ready_unassigned(conn)
        tasks = [kb.get_task(conn, tid) for tid in (first, second, third)]
        comments = kb.list_comments(conn, first)

    assert routed == [(first, "coder"), (second, "coder-1"), (third, "coder-2")]
    assert [t.assignee for t in tasks] == ["coder", "coder-1", "coder-2"]
    assert "Strict route: role=coder" in comments[-1].body


def test_office_delegate_assigns_concrete_profile_seats(
    kanban_home, monkeypatch, all_assignees_spawnable
):
    from hermes_cli import agent_office
    from hermes_cli import office_delegate

    monkeypatch.setattr(agent_office, "office_config", lambda: {
        "enabled": True,
        "board": "inbox",
        "role_seats": {"pm": ["pm", "pm-1"], "coder": ["coder", "coder-1"]},
    })

    result = office_delegate.create_office_delegation(
        "pm then coder: build the roadmap exporter",
        created_by="test",
    )

    assert result.workflow == ("pm", "coder")
    assert [t.assignee for t in result.tasks] == ["pm", "coder"]
    with kb.connect() as conn:
        shown = [kb.get_task(conn, t.id) for t in result.tasks]
    assert "This step role: pm" in shown[0].body
    assert "This step concrete assignee: pm" in shown[0].body
    assert "This step concrete assignee: coder" in shown[1].body
    assert "Office quality gate for EVERY task" in shown[0].body
    assert "SCOPE_CHANGE_REQUEST" in shown[0].body
    assert "real produced artifacts" in shown[0].body


def test_office_status_exposes_every_task_quality_gate_policy(kanban_home, monkeypatch):
    from hermes_cli import agent_office

    status = agent_office.status(board="inbox")

    assert status["quality_gates"]["enabled"] is True
    assert status["quality_gates"]["apply_to_every_task"] is True
    assert status["quality_gates"]["do_not_trust_worker_done_text"] is True
    assert status["quality_gates"]["forbid_silent_scope_reduction"] is True
    assert status["quality_gates"]["scope_change_block"] == "SCOPE_CHANGE_REQUEST"
