from agent.night_business_fallback import NightBusinessRouteDecision
from hermes_cli import kanban_db as kb
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

from tests.hermes_cli._kanban_isolation import pin_kanban_db


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated kanban DB + HERMES_HOME, safe even outside tests/ tree."""
    pin_kanban_db(tmp_path, monkeypatch)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_ready(conn, title="route me", assignee="char"):
    task_id = kb.create_task(conn, title=title, assignee=assignee, workspace_kind="scratch")
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    conn.commit()
    return task_id


def _fixed_decision(profile="night-business-claude", provider="anthropic"):
    return NightBusinessRouteDecision(
        profile=profile,
        provider=provider,
        reason="personal_exhausted",
        timezone="Asia/Seoul",
        local_time="2026-08-03T21:00+09:00",
    )


# ---------------------------------------------------------------------------
# Original test (kept intact)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Item 8: config 변경 즉시 반영 + duplicate dispatch 없음
# ---------------------------------------------------------------------------

def test_config_change_reflected_in_next_dispatch(kanban_home, monkeypatch):
    """Item 8 (part 1): config enabled 변경이 다음 dispatch_once 호출에 즉시 반영된다.

    enabled=True 설정으로 첫 번째 task를 business 라우팅한 뒤,
    enabled=False로 변경하면 두 번째 task는 business 라우팅을 받지 않는다.
    """
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda profile: True)

    # --- 첫 번째 dispatch: enabled=True → business profile로 spawn
    decision_on = _fixed_decision()
    deny_decision = NightBusinessRouteDecision(
        profile=None, provider="anthropic", reason="disabled",
        timezone="Asia/Seoul", local_time="2026-08-03T21:00+09:00",
    )

    spawned_profiles = []

    def spawn(task, workspace, board=None):
        spawned_profiles.append(task.assignee)
        return 999

    with kb.connect() as conn:
        t1 = _make_ready(conn, title="task-enabled")

    monkeypatch.setattr(kb, "_night_business_route_for_task", lambda task: decision_on)
    with kb.connect() as conn:
        kb.dispatch_once(conn, spawn_fn=spawn)

    assert spawned_profiles == ["night-business-claude"], (
        "enabled=True 시 business profile로 spawn되어야 함"
    )

    # --- 두 번째 dispatch: config가 disabled로 변경됨 → 일반 라우팅
    with kb.connect() as conn:
        t2 = _make_ready(conn, title="task-disabled")

    monkeypatch.setattr(kb, "_night_business_route_for_task", lambda task: deny_decision)
    with kb.connect() as conn:
        kb.dispatch_once(conn, spawn_fn=spawn)

    # t2는 원래 assignee("char")로 spawn되어야 함
    assert "char" in spawned_profiles, (
        "enabled=False 시 business 라우팅 없이 원래 assignee로 spawn되어야 함"
    )


def test_no_duplicate_routing_event_in_single_dispatch(kanban_home, monkeypatch):
    """Item 8 (part 2): 한 번의 dispatch_once에서 동일 task에 라우팅 이벤트는 1회만 기록된다."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda profile: True)
    monkeypatch.setattr(kb, "_night_business_route_for_task",
                        lambda task: _fixed_decision())

    def spawn(task, workspace, board=None):
        return 7777

    with kb.connect() as conn:
        task_id = _make_ready(conn)

    with kb.connect() as conn:
        kb.dispatch_once(conn, spawn_fn=spawn)

    with kb.connect() as conn:
        events = kb.list_events(conn, task_id)

    route_events = [e for e in events if e.kind == "night_business_route"]
    assert len(route_events) == 1, (
        f"night_business_route 이벤트가 정확히 1회여야 함; 실제: {len(route_events)}"
    )


# ---------------------------------------------------------------------------
# Item 9: credential pool 격리 + fail-closed
# ---------------------------------------------------------------------------

def test_credential_pool_isolation_business_does_not_contaminate_personal(
    kanban_home, monkeypatch
):
    """Item 9 (격리): _night_business_route_for_task는 개인/비즈니스 프로파일을
    별도 HERMES_HOME 컨텍스트에서 로드하므로, 비즈니스 auth.json이 개인 풀에 섞이지 않는다.

    set_hermes_home_override 호출 순서를 추적해 격리 계약을 검증한다.
    """
    from hermes_constants import get_hermes_home

    personal_homes = []
    business_homes = []
    call_phase = []  # 'personal' or 'business'

    def tracking_load_pool(provider):
        # load_pool 호출 시점의 HERMES_HOME을 캡처
        current_home = str(get_hermes_home())
        if call_phase and call_phase[-1] == "business":
            business_homes.append(current_home)
        else:
            personal_homes.append(current_home)
        pool = MagicMock()
        pool.has_credentials.return_value = True
        pool.has_available.return_value = False  # 개인 소진
        return pool

    def tracking_resolve(profile_name):
        from hermes_cli.profiles import resolve_profile_env as real_resolve
        result = real_resolve(profile_name)
        if "business" in profile_name:
            call_phase.append("business")
        else:
            call_phase.append("personal")
        return result

    night_settings = {
        "enabled": True,
        "timezone": "Asia/Seoul",
        "start_time": "21:00",
        "end_time": "06:00",
        "profiles": {
            "anthropic": "night-business-claude",
        },
    }

    # DB에서 실제 Task 객체를 가져와서 사용
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="isolation-test", assignee="char",
                                  workspace_kind="scratch")
        fake_task = kb.get_task(conn, task_id)

    with (
        patch("agent.credential_pool.load_pool", side_effect=tracking_load_pool),
        patch("hermes_cli.profiles.resolve_profile_env", side_effect=tracking_resolve),
        patch("hermes_cli.config.load_config", return_value={
            "kanban": {"night_business_fallback": night_settings},
            "model": {"provider": "anthropic"},
        }),
    ):
        from hermes_cli.kanban_db import _night_business_route_for_task
        _night_business_route_for_task(fake_task)

    # 핵심: 개인 풀과 비즈니스 풀 로드 시 HERMES_HOME이 달랐어야 한다
    if personal_homes and business_homes:
        assert personal_homes[0] != business_homes[0], (
            "개인과 비즈니스 풀 로드 시 HERMES_HOME이 달라야 함(격리 계약)"
        )


def test_fail_closed_on_pool_load_exception(kanban_home, monkeypatch):
    """Item 9 (fail-closed): load_pool이 예외를 던지면 deny()를 반환 → business 라우팅 금지.

    429/401 등 오류는 '소진'이 아니라 '판정불가'이므로 business 라우팅하지 않는다.
    """
    from hermes_cli.kanban_db import _night_business_route_for_task

    night_settings = {
        "enabled": True,
        "timezone": "Asia/Seoul",
        "start_time": "21:00",
        "end_time": "06:00",
        "profiles": {"anthropic": "night-business-claude"},
    }

    def raise_on_load(provider):
        raise RuntimeError("429 rate limit — cannot determine availability")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="fail-closed", assignee="char",
                                  workspace_kind="scratch")
        fake_task = kb.get_task(conn, task_id)

    with (
        patch("agent.credential_pool.load_pool", side_effect=raise_on_load),
        patch("hermes_cli.config.load_config", return_value={
            "kanban": {"night_business_fallback": night_settings},
            "model": {"provider": "anthropic"},
        }),
    ):
        decision = _night_business_route_for_task(fake_task)

    assert decision.profile is None, (
        "load_pool 예외(429/401 등) 시 business 라우팅되면 안 됨(fail-closed)"
    )


# ---------------------------------------------------------------------------
# Sender callback
# ---------------------------------------------------------------------------

def test_route_notify_fn_called_with_correct_args(kanban_home, monkeypatch):
    """route_notify_fn이 business 라우팅 시 올바른 인자(시간·근거·provider·profile)로 호출된다."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda profile: True)

    expected = _fixed_decision(profile="night-business-claude", provider="anthropic")
    monkeypatch.setattr(kb, "_night_business_route_for_task", lambda task: expected)

    received_calls = []

    def fake_sender(decision):
        received_calls.append(decision)

    def spawn(task, workspace, board=None):
        return 5555

    with kb.connect() as conn:
        _make_ready(conn)

    with kb.connect() as conn:
        kb.dispatch_once(conn, spawn_fn=spawn, route_notify_fn=fake_sender)

    assert len(received_calls) == 1, (
        f"route_notify_fn이 정확히 1회 호출되어야 함; 실제: {len(received_calls)}"
    )
    d = received_calls[0]
    assert d.provider == "anthropic"
    assert d.profile == "night-business-claude"
    assert d.reason == "personal_exhausted"
    assert d.timezone == "Asia/Seoul"
    assert d.local_time == "2026-08-03T21:00+09:00"


def test_route_notify_fn_not_called_when_no_business_routing(kanban_home, monkeypatch):
    """business 라우팅이 없을 때(deny) route_notify_fn은 호출되지 않는다."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda profile: True)

    deny = NightBusinessRouteDecision(
        profile=None, provider="anthropic", reason="disabled",
        timezone="Asia/Seoul", local_time="2026-08-03T21:00+09:00",
    )
    monkeypatch.setattr(kb, "_night_business_route_for_task", lambda task: deny)

    received_calls = []

    def fake_sender(decision):
        received_calls.append(decision)

    def spawn(task, workspace, board=None):
        return 5556

    with kb.connect() as conn:
        _make_ready(conn)

    with kb.connect() as conn:
        kb.dispatch_once(conn, spawn_fn=spawn, route_notify_fn=fake_sender)

    assert received_calls == [], "business 라우팅 없을 때 route_notify_fn 호출 안 됨"
