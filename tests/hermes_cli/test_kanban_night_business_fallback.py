from agent.night_business_fallback import NightBusinessRouteDecision
from hermes_cli import kanban_db as kb


def test_dispatch_spawns_isolated_business_profile_without_reassigning_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()

    decision = NightBusinessRouteDecision(
        profile="night-business-claude",
        provider="anthropic",
        reason="personal_exhausted",
        timezone="Asia/Seoul",
        local_time="2026-08-03T21:00+09:00",
    )
    monkeypatch.setattr(kb, "_night_business_route_for_task", lambda task: decision)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda profile: True)
    spawned = []

    def spawn(task, workspace, board=None):
        spawned.append((task.assignee, workspace))
        return 4242

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="route me", assignee="char", workspace_kind="scratch")
        result = kb.dispatch_once(conn, spawn_fn=spawn)
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert result.spawned[0][0] == task_id
    assert spawned[0][0] == "night-business-claude"
    assert task is not None
    assert task.assignee == "char"
    route_events = [event for event in events if event.kind == "night_business_route"]
    assert len(route_events) == 1
    payload = route_events[0].payload
    assert payload is not None
    assert payload["provider"] == "anthropic"
    assert payload["profile"] == "night-business-claude"
