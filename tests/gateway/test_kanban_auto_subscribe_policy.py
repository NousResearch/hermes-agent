"""Regression for #127092: slash-created tasks must respect the wake opt-out."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_multiplex():
    from agent.secret_scope import set_multiplex_active
    set_multiplex_active(False)
    yield
    set_multiplex_active(False)


@pytest.mark.asyncio
async def test_slash_create_respects_profile_auto_subscribe_policy(tmp_path, monkeypatch):
    from agent import secret_scope
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session import SessionSource
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
    from hermes_cli.config import atomic_config_write

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "disabled"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    homes = {}
    for name, setting in (("disabled", False), ("enabled", True), ("default", None)):
        home = tmp_path / name
        home.mkdir()
        config = {} if setting is None else {"kanban": {"auto_subscribe_on_create": setting}}
        atomic_config_write(home / "config.yaml", config)
        homes[name] = home

    runner = GatewayRunner.__new__(GatewayRunner)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat1", chat_type="dm", user_id="u1")
    # A -> B -> A proves the worker-thread config read follows the current
    # profile, even though the process environment stays pinned to A.
    secret_scope.set_multiplex_active(True)
    for index, name in enumerate(("disabled", "enabled", "disabled", "default")):
        with _profile_runtime_scope(homes[name]):
            event = MessageEvent(text=f'/kanban create "task-{index}" --assignee alice', source=source)
            reply = await runner._handle_kanban_command(event)
            with kbc.connect() as conn:
                task = next(task for task in kb.list_tasks(conn) if task.title == f"task-{index}")
                subs = kbn.list_notify_subs(conn, task.id)
            enabled = name != "disabled"
            assert bool(subs) is enabled
            assert ("subscribed" in reply.lower()) is enabled
            if not enabled:
                # The automatic opt-out must still allow an explicit passive
                # subscription, without turning it into an agent wake.
                event.text = f'/kanban notify-subscribe {task.id} --platform telegram --chat-id chat1'
                await runner._handle_kanban_command(event)
                with kbc.connect() as conn:
                    subs = kbn.list_notify_subs(conn, task.id)
                assert len(subs) == 1
                assert subs[0]["delivery_mode"] == "notify"
