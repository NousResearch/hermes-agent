"""Exact route/resource admission for plugin-owned Telegram task surfaces."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.kanban_surfaces import register_task_cards, subscription_registration
from gateway.live_todo import TodoBinding, TodoSource, open_live_todo, register_live_todo
from gateway.surface_scope import parse_surface_scope
from gateway.task_read import TaskReadDenied, TaskReadService, register_task_detail


def _scope(*, chat_id="-1000000000001", thread_id="7", task_id="t_12345678"):
    return {
        "routes": [{
            "profile": "default",
            "platform": "telegram",
            "chat_id": chat_id,
            "thread_id": thread_id,
        }],
        "task_resources": [{"board": "synthetic", "task_id": task_id}],
    }


def test_scope_is_exact_immutable_and_null_thread_is_not_a_wildcard():
    raw = _scope(thread_id=None)
    parsed = parse_surface_scope(raw, require_tasks=True)
    raw["routes"][0]["chat_id"] = "-1000000000002"
    raw["task_resources"][0]["task_id"] = "t_87654321"

    assert parsed.allows_card(
        "default", Platform.TELEGRAM, "-1000000000001", None,
        "synthetic", "t_12345678",
    )
    assert not parsed.allows_route("default", "telegram", "-1000000000001", "7")
    assert not parsed.allows_task("default", "synthetic", "t_87654321")


@pytest.mark.parametrize("scope", [
    None,
    {},
    {"routes": [], "task_resources": []},
    {"routes": [{"profile": "default", "platform": "telegram",
                 "chat_id": "*", "thread_id": None}], "task_resources": []},
    {"routes": [{"profile": "default", "platform": "telegram",
                 "chat_id": "-1000000000001", "thread_id": "*"}], "task_resources": []},
    {"routes": [{"profile": "default", "platform": "discord",
                 "chat_id": "-1000000000001", "thread_id": None}], "task_resources": []},
    {"routes": [{"profile": "default", "platform": "telegram",
                 "chat_id": "-1000000000001", "thread_id": None, "wildcard": True}],
     "task_resources": []},
    {"routes": [{"profile": "default", "platform": "telegram",
                 "chat_id": "-1000000000001", "thread_id": None}],
     "task_resources": [{"board": "synthetic", "task_id": "*"}]},
])
def test_invalid_scope_denies_registration(scope):
    with pytest.raises(ValueError):
        parse_surface_scope(scope, require_tasks=True)


def test_partial_plugin_registration_is_revoked_by_existing_unload_ledger(tmp_path):
    callbacks = []
    manager = SimpleNamespace(home_path=tmp_path)
    ctx = SimpleNamespace(
        plugin_id="synthetic-plugin",
        _manager=manager,
        on_unload=lambda callback: callbacks.append(callback),
    )
    todo = register_live_todo(ctx, lambda handle: None, scope=_scope())

    bad_card_scope = _scope()
    bad_card_scope["task_resources"] = []
    with pytest.raises(ValueError):
        register_task_cards(ctx, lambda handle: None, scope=bad_card_scope)

    assert todo.active
    assert callbacks
    for callback in reversed(callbacks):
        callback()
    assert not todo.active


def test_live_todo_route_is_denied_before_adapter_binding(monkeypatch):
    parsed = parse_surface_scope(_scope(), require_tasks=False)
    registration = SimpleNamespace(active=True, scope=parsed)
    manager = SimpleNamespace(_live_todo_registration=registration)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter_lookups = []
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter_lookups.append(source),
    )
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, profile="default", chat_id="-1000000000002",
        thread_id="7",
    )
    turn = SimpleNamespace(
        source=source, scheduled_heartbeat=False, mute_notification_reply=False,
        session_id="synthetic-session", run_generation=1,
    )
    display = SimpleNamespace(_tool_progress_explicit=False, progress_mode="all")

    assert open_live_todo(runner, display, turn) is None
    assert adapter_lookups == []


def test_live_todo_transport_rechecks_exact_route():
    parsed = parse_surface_scope(_scope(), require_tasks=False)
    registration = SimpleNamespace(active=True, scope=parsed, lock=threading.RLock())
    adapter = SimpleNamespace(_bot=object(), _live_todo_epoch="epoch")
    source = object.__new__(TodoSource)
    source.active = True
    source.unknown = False
    source.registration = registration
    source.lock = threading.RLock()
    source.binding = TodoBinding(
        "default", "default", "/synthetic", "session", 1,
        "-1000000000002", "7", store_incarnation="incarnation",
    )
    source.is_current = lambda: True
    source.adapter = adapter
    source.client = adapter._bot
    source.epoch = "epoch"
    source.store = lambda: SimpleNamespace(incarnation="incarnation")

    assert not source.admitted()


def test_card_scope_denies_before_surface_collection(monkeypatch, tmp_path):
    parsed = parse_surface_scope(_scope(), require_tasks=True)
    registration = SimpleNamespace(active=True, scope=parsed)
    manager = SimpleNamespace(_task_card_registration=registration)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    runner = SimpleNamespace(_resolve_profile_home_for_source=lambda source: tmp_path)
    sub = {
        "platform": "telegram", "notifier_profile": "default", "delivery_mode": "notify",
        "chat_id": "-1000000000001", "thread_id": "7", "task_id": "t_87654321",
    }

    assert subscription_registration(runner, sub, "synthetic") is None
    sub["task_id"] = "t_12345678"
    assert subscription_registration(runner, sub, "synthetic") is registration


def test_task_read_is_independently_resource_scoped(tmp_path, monkeypatch):
    parsed = parse_surface_scope(_scope(), require_tasks=True)
    ctx = SimpleNamespace(plugin_id="synthetic-plugin")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service = TaskReadService(ctx, parsed)
    app = object()
    owner = SimpleNamespace(config=SimpleNamespace(token="synthetic-token"), _bot=object(),
                            _live_todo_epoch="epoch")
    runner = SimpleNamespace(
        _authorization_adapter=lambda platform, profile: owner,
        _primary_profile_name="default",
    )
    service.binding = (app, SimpleNamespace(gateway_runner=runner))
    service._policy = lambda: ({}, [], 300)
    monkeypatch.setattr(
        "gateway.task_read.verify_init_data",
        lambda raw, token, now, max_age: {"actor": 1, "bot_id": 2},
    )

    with pytest.raises(TaskReadDenied):
        service._authorize(app, "synthetic-init-data", ("default", "synthetic", "t_87654321", 1))


def test_task_read_registration_rejects_scope_for_different_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scope = _scope()
    scope["routes"][0]["profile"] = "other"
    callbacks = []
    manager = SimpleNamespace()
    ctx = SimpleNamespace(
        plugin_id="synthetic-plugin",
        _manager=manager,
        on_unload=lambda callback: callbacks.append(callback),
    )

    with pytest.raises(ValueError, match="must match its owning API-server profile"):
        register_task_detail(ctx, scope=scope)

    assert not hasattr(manager, "_task_read_registration")
    assert callbacks == []
