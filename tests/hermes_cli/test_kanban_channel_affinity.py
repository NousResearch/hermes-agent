"""Regression coverage for Kanban completion channel affinity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def create_task(kanban_home, monkeypatch):
    from tools import kanban_tools
    import hermes_cli.config as hermes_config

    cfg = {
        "kanban": {
            "auto_subscribe_on_create": True,
            "auto_subscribe_home_on_create": True,
            "auto_subscribe_home_platforms": ["telegram"],
        }
    }
    monkeypatch.setattr(kanban_tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hermes_config, "load_config", lambda: cfg)
    monkeypatch.setattr(
        kanban_tools,
        "_configured_home_channels",
        lambda: [{"platform": "telegram", "chat_id": "home-chat", "thread_id": "home-thread"}],
    )
    for name in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_KEY",
        "HERMES_KANBAN_SUB_PLATFORM",
        "HERMES_KANBAN_SUB_CHAT_ID",
        "HERMES_KANBAN_SUB_THREAD_ID",
        "HERMES_KANBAN_SUB_USER_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    def create(**kwargs):
        gateway_source = kwargs.pop("gateway_source", None)
        result = json.loads(kanban_tools._handle_create({
            "title": "Channel affinity task",
            "assignee": "worker",
            **kwargs,
        }, gateway_source=gateway_source))
        assert result["ok"] is True
        return result["task_id"]

    return create


def test_tui_create_keeps_completion_on_attached_tui_session(create_task):
    tokens = set_session_vars(session_key="desktop-session")
    try:
        task_id = create_task()
    finally:
        clear_session_vars(tokens)

    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, task_id=task_id)
    assert [(sub["platform"], sub["chat_id"], sub["thread_id"]) for sub in subs] == [
        ("tui", "desktop-session", ""),
    ]


def test_gateway_create_uses_originating_telegram_channel(create_task):
    task_id = create_task(gateway_source={
        "platform": "telegram",
        "chat_id": "telegram-chat",
        "thread_id": "topic-7",
    })

    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, task_id=task_id)
    assert [(sub["platform"], sub["chat_id"], sub["thread_id"]) for sub in subs] == [
        ("telegram", "telegram-chat", "topic-7"),
    ]


def test_unattached_create_preserves_home_channel_fallback(create_task):
    task_id = create_task()

    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, task_id=task_id)
    assert [(sub["platform"], sub["chat_id"], sub["thread_id"]) for sub in subs] == [
        ("telegram", "home-chat", "home-thread"),
    ]
