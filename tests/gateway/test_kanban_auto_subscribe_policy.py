"""Slash-created tasks respect wake opt-outs and existing delivery policies."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_multiplex():
    from agent.secret_scope import set_multiplex_active

    set_multiplex_active(False)
    yield
    set_multiplex_active(False)


def _source():
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat1", chat_type="dm", user_id="u1")


def _write_config(home: Path, enabled: bool | None) -> None:
    from hermes_cli.config import atomic_config_write

    home.mkdir(parents=True, exist_ok=True)
    config = {} if enabled is None else {"kanban": {"auto_subscribe_on_create": enabled}}
    atomic_config_write(home / "config.yaml", config)


@pytest.mark.asyncio
async def test_slash_create_respects_profile_auto_subscribe_policy(tmp_path, monkeypatch):
    from agent import secret_scope
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "disabled"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    homes = {}
    for name, setting in (("disabled", False), ("enabled", True), ("default", None)):
        home = tmp_path / name
        _write_config(home, setting)
        homes[name] = home

    runner = GatewayRunner.__new__(GatewayRunner)
    source = _source()
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
                event.text = f"/kanban notify-subscribe {task.id} --platform telegram --chat-id chat1"
                await runner._handle_kanban_command(event)
                with kbc.connect() as conn:
                    subs = kbn.list_notify_subs(conn, task.id)
                assert len(subs) == 1
                assert subs[0]["delivery_mode"] == "notify"


@pytest.mark.asyncio
async def test_idempotent_create_preserves_explicit_delivery_mode(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "home"
    _write_config(home, True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)

    runner = GatewayRunner.__new__(GatewayRunner)
    source = _source()
    command = '/kanban create "idem" --assignee alice --idempotency-key K1'
    with _profile_runtime_scope(home):
        await runner._handle_kanban_command(MessageEvent(text=command, source=source))
        with kbc.connect() as conn:
            task = next(task for task in kb.list_tasks(conn) if task.title == "idem")
            assert kbn.list_notify_subs(conn, task.id)[0]["delivery_mode"] == "notify+wake"
            kbn.add_notify_sub(
                conn,
                task_id=task.id,
                platform="telegram",
                chat_id="chat1",
                delivery_mode="notify",
            )

        # Repeating the same create returns the existing task. The automatic
        # subscription check must not replace an explicitly passive policy.
        await runner._handle_kanban_command(MessageEvent(text=command, source=source))
        with kbc.connect() as conn:
            assert len([task for task in kb.list_tasks(conn) if task.title == "idem"]) == 1
            subs = kbn.list_notify_subs(conn, task.id)
            assert len(subs) == 1
            assert subs[0]["delivery_mode"] == "notify"
