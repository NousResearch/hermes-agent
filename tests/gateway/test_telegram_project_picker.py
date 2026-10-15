"""Production gateway path tests for `/new -project`, including callback binding and replay."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_telegram_project_picker_creates_one_fresh_session_without_touching_opening_route(
    tmp_path,
    monkeypatch,
):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from hermes_state import SessionDB
    from hermes_cli import projects_db
    from plugins.platforms.telegram.adapter import TelegramAdapter

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    workspace = tmp_path / "registered-workspace"
    workspace.mkdir()
    second_workspace = tmp_path / "registered-workspace-alt"
    second_workspace.mkdir()
    other_workspace = tmp_path / "other-active-workspace"
    other_workspace.mkdir()
    with projects_db.connect_closing(home / "projects.db") as conn:
        project_id = projects_db.create_project(
            conn,
            name="Feature Workspace",
            slug="feature-workspace",
            folders=[str(workspace), str(second_workspace)],
        )
        active_id = projects_db.create_project(
            conn, name="Other Active", folders=[str(other_workspace)]
        )
        projects_db.set_active(conn, active_id)

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")
            }
        )
    )
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._send_control_message = AsyncMock(
        return_value=SimpleNamespace(message_id=701)
    )
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"
    )
    key = runner.session_store._generate_session_key(source)
    old = runner.session_store.get_or_create_session(source)
    db = SessionDB(home / "state.db")
    runner.session_store._db = db
    db.create_session(
        old.session_id, "telegram", session_key=key, chat_id="42", user_id="42"
    )
    db.append_message(old.session_id, role="user", content="prior conversation")

    response = await runner._handle_message(
        MessageEvent(text="/new -project", source=source)
    )
    assert response is None
    assert runner.session_store.peek_session_id(key) == old.session_id
    assert len(adapter._project_picker_state) == 1
    token, state = next(iter(adapter._project_picker_state.items()))
    choice = next(x for x in state["projects"] if x["id"] == project_id)
    index = state["projects"].index(choice)
    action = f"pj:{token}:select_{index}"

    def callback(*, user_id=42, chat_id=42, message_id=701):
        return SimpleNamespace(
            data=action,
            from_user=SimpleNamespace(id=user_id, first_name="tester"),
            message=SimpleNamespace(
                message_id=message_id,
                chat_id=chat_id,
                message_thread_id=None,
                chat=SimpleNamespace(type="private"),
            ),
            answer=AsyncMock(),
            edit_message_text=AsyncMock(),
        )

    query = callback()
    update = SimpleNamespace(callback_query=query)
    for invalid in (
        callback(user_id=43),
        callback(chat_id=99),
        callback(message_id=702),
    ):
        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=invalid), None
        )
        assert runner.session_store.peek_session_id(key) == old.session_id
        assert token in adapter._project_picker_state

    await adapter._handle_callback_query(update, None)
    assert state["view"] == "folders"
    assert runner.session_store.peek_session_id(key) == old.session_id
    action = f"pj:{token}:back"
    query.data = action
    await adapter._handle_callback_query(update, None)
    assert state["view"] == "projects"
    assert runner.session_store.peek_session_id(key) == old.session_id
    action = f"pj:{token}:select_{index}"
    query.data = action
    await adapter._handle_callback_query(update, None)
    assert state["view"] == "folders"

    entered, release = asyncio.Event(), asyncio.Event()
    original_handler = state["handler"]

    async def delayed(query_arg, action_arg, state_arg):
        entered.set()
        await release.wait()
        return await original_handler(query_arg, action_arg, state_arg)

    state["handler"] = delayed
    action = f"pj:{token}:folder_0"
    query.data = action
    first_tap = asyncio.create_task(adapter._handle_callback_query(update, None))
    await entered.wait()
    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=callback()), None
    )
    assert runner.session_store.peek_session_id(key) == old.session_id
    release.set()
    await first_tap
    current = runner.session_store.peek_session_id(key)
    assert current and current != old.session_id
    assert len(adapter._project_picker_state) == 0
    assert db.get_session(current)["cwd"] == str(workspace)
    assert db.get_messages(old.session_id)[0]["content"] == "prior conversation"
    assert db.get_messages(current) == []
    with projects_db.connect_closing(home / "projects.db") as conn:
        assert projects_db.get_active_id(conn) == active_id
    import json

    assert (
        json.loads(db.get_session(current)["model_config"])["_reset_from"]
        == old.session_id
    )

    from agent.runtime_cwd import resolve_agent_cwd, scoped_session_cwd
    from gateway.project_session import project_cwd_for_turn
    from gateway.session import build_session_context, SessionStore
    from tools.terminal_tool import get_session_cwd

    resumed_store = SessionStore(home / "sessions", GatewayConfig())
    resumed_store._db = db
    resumed = resumed_store.lookup_by_session_key(key)
    assert resumed and resumed.session_id == current
    context = build_session_context(source, runner.config, resumed)
    context.cwd = await project_cwd_for_turn(
        SimpleNamespace(session_store=resumed_store), resumed, source
    )
    tokens = runner._set_session_env(context)
    try:
        assert scoped_session_cwd() == str(workspace)
        assert resolve_agent_cwd() == workspace
        assert get_session_cwd(key) == str(workspace)
    finally:
        runner._clear_session_env(tokens)

    # Replaying the captured selection is stale and cannot create a second fresh session.
    await adapter._handle_callback_query(update, None)
    assert runner.session_store.peek_session_id(key) == current
    assert db.session_count() == 2
    db.close()
    runner._shutdown_executor()
