"""Slash auto-subscriptions read the owning profile, independent of launch config."""

import asyncio
from pathlib import Path

import pytest


@pytest.mark.parametrize("config, expected, warns", [
    ("kanban: {}\n", "notify+wake", False),
    ("kanban:\n  auto_subscribe_delivery_mode: notify+wake\n", "notify+wake", False),
    ("kanban:\n  auto_subscribe_delivery_mode: wake\n", "wake", False),
    ("kanban:\n  auto_subscribe_delivery_mode: notify\n", "notify", False),
    ("kanban:\n  auto_subscribe_delivery_mode: invalid\n", "notify+wake", True),
    ("kanban:\n  auto_subscribe_delivery_mode: null\n", "notify+wake", True),
    ("kanban:\n  auto_subscribe_delivery_mode: [wake]\n", "notify+wake", True),
    ("kanban: [\n", "notify+wake", True),
])
def test_slash_subscription_uses_owning_profile(tmp_path, monkeypatch, config, expected, warns, caplog):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
    from hermes_constants import get_hermes_home

    home = tmp_path / ".hermes"
    owner = home / "profiles" / "yuki"
    owner.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    (home / "config.yaml").write_text("kanban:\n  auto_subscribe_delivery_mode: notify\n")
    (owner / "config.yaml").write_text(config)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "default"
    source = SessionSource(platform=Platform.DISCORD, chat_id="post", chat_type="thread",
                           thread_id="post", profile="yuki")
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="slash-created")
    event = MessageEvent(text="/kanban create", source=source)
    assert asyncio.run(runner._kanban_auto_subscribe(event, task, None))
    with kbc.connect() as conn:
        sub = kbn.list_notify_subs(conn, task)[0]
    assert sub["delivery_mode"] == expected
    assert sub["notifier_profile"] == "yuki"
    assert get_hermes_home() == home
    assert ("kanban.auto_subscribe_delivery_mode" in caplog.text) == warns
