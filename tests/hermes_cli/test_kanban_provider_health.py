from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace


def _task(**overrides):
    values = {
        "id": "t-1",
        "assignee": "coder",
        "model_override": None,
        "provider_override": None,
        "reasoning_effort": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kanban_route_uses_sol_medium_when_profile_primary_is_open(tmp_path):
    from agent.provider_health import ProviderHealthStore, ProviderRoute
    from hermes_cli.kanban_provider_health import resolve_task_route

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    ProviderHealthStore(tmp_path).record_failure(
        ProviderRoute("zai", "glm-5.3-flash"),
        "quota exhausted; reset_at=2026-09-11T00:00:00Z",
        source="cron:eng-completion",
        now=now,
    )
    config = {
        "model": {"provider": "zai", "default": "glm-5.3-flash"},
        "agent": {"reasoning_effort": "medium"},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "medium"}
        ],
    }

    resolved = resolve_task_route(_task(), profile_home=tmp_path, config=config, now=now)

    assert resolved.route.provider == "openai-codex"
    assert resolved.route.model == "gpt-5.6-sol"
    assert resolved.route.reasoning_effort == "medium"
    assert resolved.deferred_until is None


def test_kanban_explicit_provider_restriction_never_uses_profile_fallback(tmp_path):
    from agent.provider_health import ProviderHealthStore, ProviderRoute
    from hermes_cli.kanban_provider_health import resolve_task_route

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    ProviderHealthStore(tmp_path).record_failure(
        ProviderRoute("zai", "glm-5.3-flash"),
        "quota exhausted; reset_at=2026-09-11T00:00:00Z",
        source="kanban:t-1",
        now=now,
    )
    config = {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6-sol"}
        ],
    }

    resolved = resolve_task_route(
        _task(model_override="glm-5.3-flash", provider_override="zai"),
        profile_home=tmp_path,
        config=config,
        now=now,
    )

    assert resolved.route is None
    assert resolved.deferred_until == datetime(2026, 9, 11, tzinfo=timezone.utc)


def test_kanban_provider_only_restriction_never_crosses_to_fallback_provider(tmp_path):
    from agent.provider_health import ProviderHealthStore, ProviderRoute
    from hermes_cli.kanban_provider_health import resolve_task_route

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    ProviderHealthStore(tmp_path).record_failure(
        ProviderRoute("zai", "glm-5.3-flash"),
        "quota exhausted; reset_at=2026-09-11T00:00:00Z",
        source="kanban:t-1",
        now=now,
    )
    config = {
        "model": {"provider": "openai-codex", "default": "glm-5.3-flash"},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6-sol"}
        ],
    }

    resolved = resolve_task_route(
        _task(provider_override="zai"),
        profile_home=tmp_path,
        config=config,
        now=now,
    )

    assert resolved.route is None


def test_dispatcher_parks_open_route_once_across_ticks(
    tmp_path, monkeypatch, all_assignees_spawnable
):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli import kanban_provider_health as health

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    deferred = SimpleNamespace(
        route=None,
        deferred_until=datetime(2026, 9, 11, tzinfo=timezone.utc),
        reason="weekly quota exhausted",
    )
    monkeypatch.setattr(health, "resolve_task_route", lambda task: deferred)
    starts = []

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="quota blocked", assignee="coder")
        first = dispatch.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: starts.append(args))
        second = dispatch.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: starts.append(args))
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? AND kind='provider_deferred'",
            (task_id,),
        ).fetchall()
        task = kb.get_task(conn, task_id)

    assert starts == []
    assert first.provider_deferred == second.provider_deferred == [
        (task_id, "2026-09-11T00:00:00+00:00")
    ]
    assert len(events) == 1
    assert task is not None and task.status == "ready"
