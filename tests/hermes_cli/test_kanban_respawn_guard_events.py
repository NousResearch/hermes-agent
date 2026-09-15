"""Coalesce identical dispatcher ``respawn_guarded`` events.

Regression for unbounded ``active_pr`` event growth: the guard must still
evaluate every tick (and still block a duplicate spawn), but an unchanged
decision must not insert a new ``task_events`` row every tick. A changed
reason or PR URL is recorded immediately. A None guard (the post-admission
shape used by explicit CI-repair) still spawns even if prior ticks were
coalesced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


PR_A = "https://github.com/NousResearch/hermes-agent/pull/123"
PR_B = "https://github.com/NousResearch/hermes-agent/pull/456"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _enable_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)


def _guarded_events(conn, task_id: str) -> list:
    return [e for e in kb.list_events(conn, task_id) if e.kind == "respawn_guarded"]


def _ready_task_with_pr(conn, *, url: str = PR_A, title: str = "already PRed") -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    kb.add_comment(conn, task_id, author="worker", body=f"Opened {url} for review.")
    return task_id


def _dispatch(conn, spawn_calls: list, **kwargs):
    def spawn(task, workspace, board=None):
        spawn_calls.append(task.id)
        return 4242

    return kbd.dispatch_once(conn, spawn_fn=spawn, **kwargs)


def test_identical_active_pr_guard_events_stay_bounded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five identical ticks persist one event; the in-memory result still records every skip."""
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn)
        results = [_dispatch(conn, spawn_calls) for _ in range(5)]

        events = _guarded_events(conn, task_id)
        task = kb.get_task(conn, task_id)

    assert spawn_calls == []
    assert task is not None and task.status == "ready"
    assert all(dict(r.respawn_guarded).get(task_id) == "active_pr" for r in results)
    assert len(events) == 1
    assert events[0].payload is not None
    assert events[0].payload["reason"] == "active_pr"
    assert PR_A in events[0].payload.get("pr_urls", [])
    assert events[0].payload.get("heartbeat") is not True


def test_changed_guard_reason_is_recorded_immediately(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn)
        first = _dispatch(conn, spawn_calls)
        assert dict(first.respawn_guarded).get(task_id) == "active_pr"

        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("401 unauthorized: invalid_api_key", task_id),
        )
        conn.commit()
        second = _dispatch(conn, spawn_calls)

        events = _guarded_events(conn, task_id)

    assert spawn_calls == []
    assert dict(second.respawn_guarded).get(task_id) == "blocker_auth"
    assert [e.payload["reason"] for e in events] == ["active_pr", "blocker_auth"]


def test_changed_pr_url_is_recorded_immediately(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn, url=PR_A)
        _dispatch(conn, spawn_calls)
        kb.add_comment(conn, task_id, author="worker", body=f"Superseded by {PR_B}")
        _dispatch(conn, spawn_calls)

        events = _guarded_events(conn, task_id)

    assert spawn_calls == []
    assert len(events) == 2
    assert events[0].payload["pr_urls"] == [PR_A]
    assert PR_B in events[1].payload["pr_urls"]
    assert events[1].payload["reason"] == "active_pr"


def test_heartbeat_after_cooldown_is_bounded(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_profiles(monkeypatch)
    monkeypatch.setattr(kbd, "DEFAULT_RESPAWN_GUARD_EVENT_COOLDOWN_SECONDS", 60)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn)
        _dispatch(conn, spawn_calls)
        first = _guarded_events(conn, task_id)
        assert len(first) == 1

        conn.execute(
            "UPDATE task_events SET created_at = created_at - 120 "
            "WHERE task_id = ? AND kind = 'respawn_guarded'",
            (task_id,),
        )
        conn.commit()
        _dispatch(conn, spawn_calls)
        _dispatch(conn, spawn_calls)

        events = _guarded_events(conn, task_id)

    assert spawn_calls == []
    assert len(events) == 2
    assert events[1].payload["reason"] == "active_pr"
    assert events[1].payload.get("heartbeat") is True
    assert events[1].payload.get("quiet_seconds", 0) >= 60


def test_open_pr_duplicate_protection_still_blocks_spawn(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn)
        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"
        result = _dispatch(conn, spawn_calls)
        task = kb.get_task(conn, task_id)

    assert spawn_calls == []
    assert task is not None and task.status == "ready"
    assert dict(result.respawn_guarded).get(task_id) == "active_pr"


def test_none_guard_still_spawns_after_coalesced_events(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later None guard (CI-repair admission / closed PR) must not stay suppressed."""
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn)
        _dispatch(conn, spawn_calls)
        _dispatch(conn, spawn_calls)
        assert _guarded_events(conn, task_id)

        monkeypatch.setattr(
            kbd, "check_respawn_guard", lambda _conn, _task_id, **_kw: None,
        )
        result = _dispatch(conn, spawn_calls)
        task = kb.get_task(conn, task_id)

    assert task_id in spawn_calls
    assert task_id in [s[0] for s in result.spawned]
    assert task is not None and task.status == "running"


def test_corrupt_last_event_fail_closed_persists(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        task_id = _ready_task_with_pr(conn)
        _dispatch(conn, spawn_calls)
        conn.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'respawn_guarded'",
            ("not-json", task_id),
        )
        conn.commit()
        _dispatch(conn, spawn_calls)
        events = _guarded_events(conn, task_id)

    assert spawn_calls == []
    assert len(events) == 2
    assert events[1].payload is not None
    assert events[1].payload["reason"] == "active_pr"


def test_sibling_tasks_do_not_share_coalesce_state(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_profiles(monkeypatch)
    spawn_calls: list[str] = []

    with kbc.connect() as conn:
        a = _ready_task_with_pr(conn, title="card-a")
        b = _ready_task_with_pr(conn, title="card-b")
        _dispatch(conn, spawn_calls)
        _dispatch(conn, spawn_calls)

        assert len(_guarded_events(conn, a)) == 1
        assert len(_guarded_events(conn, b)) == 1
        assert spawn_calls == []
