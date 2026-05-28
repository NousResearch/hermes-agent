"""RED tests for Kanban task-thread lifecycle and context injection.

These tests describe the desired Discord/Kanban thread behavior before the
implementation exists:

* A gateway thread subscribed to exactly one Kanban task can use lifecycle
  shorthand commands without repeating the task id.
* The dynamic session prompt for that gateway thread includes the bound Kanban
  task context so the agent knows which canonical task owns the conversation.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context, build_session_context_prompt
from hermes_cli import kanban_db as kb


@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture()
def discord_thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="1508752209679220948",
        chat_name="Olympus / olympus-kanban / Improve workflow",
        chat_type="group",
        user_id="sam-user-id",
        user_name="TheSamC",
        thread_id="1509093323816570890",
    )


@pytest.fixture()
def runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._kanban_notifier_profile = "default"
    runner._active_profile_name = lambda: "default"
    return runner


def _create_subscribed_task(source: SessionSource, *, board: str | None = None) -> str:
    if board is not None:
        kb.init_db(board=board)
    conn = kb.connect(board=board)
    try:
        task_id = kb.create_task(
            conn,
            title="Improve Athena and Kanban-in-Discord task lifecycle workflow",
            assignee="hermes",
            body="Implement task-thread lifecycle commands and task-context injection.",
        )
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform=source.platform.value,
            chat_id=source.chat_id,
            thread_id=source.thread_id,
            user_id=source.user_id,
            notifier_profile="default",
        )
        return task_id
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_kanban_block_in_subscribed_thread_uses_bound_task_id(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    task_id = _create_subscribed_task(discord_thread_source)

    result = await runner._handle_kanban_command(
        MessageEvent(
            text="/kanban block awaiting Sam approval",
            source=discord_thread_source,
        )
    )

    assert f"Blocked {task_id}" in result
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_kanban_done_in_subscribed_thread_uses_bound_task_id(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    task_id = _create_subscribed_task(discord_thread_source)

    result = await runner._handle_kanban_command(
        MessageEvent(
            text="/kanban done smoke test passed",
            source=discord_thread_source,
        )
    )

    assert f"Completed {task_id}" in result
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert "smoke test passed" in (task.result or "")
    finally:
        conn.close()


def test_session_prompt_includes_bound_kanban_task_context(
    kanban_home,
    discord_thread_source: SessionSource,
):
    task_id = _create_subscribed_task(discord_thread_source)

    ctx = build_session_context(discord_thread_source, GatewayConfig())
    prompt = build_session_context_prompt(ctx)

    assert f"Kanban task: `{task_id}`" in prompt
    assert "Improve Athena and Kanban-in-Discord task lifecycle workflow" in prompt
    assert "Assignee: `hermes`" in prompt
    assert "Status: `ready`" in prompt


@pytest.mark.asyncio
async def test_kanban_shorthand_is_not_inferred_for_whole_channel_subscription(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    channel_source = SessionSource(
        platform=discord_thread_source.platform,
        chat_id=discord_thread_source.chat_id,
        chat_name=discord_thread_source.chat_name,
        chat_type=discord_thread_source.chat_type,
        user_id=discord_thread_source.user_id,
        user_name=discord_thread_source.user_name,
        thread_id="",
    )
    task_id = _create_subscribed_task(channel_source)

    result = await runner._handle_kanban_command(
        MessageEvent(
            text="/kanban block awaiting Sam approval",
            source=channel_source,
        )
    )

    assert "unknown task" in result or "cannot block" in result
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_kanban_complete_shorthand_preserves_explicit_options(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    task_id = _create_subscribed_task(discord_thread_source)

    result = await runner._handle_kanban_command(
        MessageEvent(
            text="/kanban complete --result ok --summary reviewed",
            source=discord_thread_source,
        )
    )

    assert f"Completed {task_id}" in result
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result == "ok"
        runs = list(kb.list_runs(conn, task_id))
        assert runs
        assert runs[-1].summary == "reviewed"
    finally:
        conn.close()


def test_session_prompt_finds_bound_kanban_task_on_non_default_board(
    kanban_home,
    discord_thread_source: SessionSource,
):
    task_id = _create_subscribed_task(discord_thread_source, board="olympus")

    ctx = build_session_context(discord_thread_source, GatewayConfig())
    prompt = build_session_context_prompt(ctx)

    assert f"Kanban task: `{task_id}`" in prompt
    assert "Improve Athena and Kanban-in-Discord task lifecycle workflow" in prompt
    assert "Assignee: `hermes`" in prompt


@pytest.mark.asyncio
async def test_kanban_shorthand_routes_to_non_default_bound_board(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    task_id = _create_subscribed_task(discord_thread_source, board="olympus")

    result = await runner._handle_kanban_command(
        MessageEvent(
            text="/kanban block awaiting non-default board review",
            source=discord_thread_source,
        )
    )

    assert f"Blocked {task_id}" in result
    default_conn = kb.connect()
    olympus_conn = kb.connect(board="olympus")
    try:
        assert kb.get_task(default_conn, task_id) is None
        task = kb.get_task(olympus_conn, task_id)
        assert task is not None
        assert task.status == "blocked"
    finally:
        default_conn.close()
        olympus_conn.close()


@pytest.mark.asyncio
async def test_kanban_shorthand_missing_explicit_board_does_not_create_board(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    missing = "missing-board"
    assert not kb.board_exists(missing)

    result = await runner._handle_kanban_command(
        MessageEvent(
            text=f"/kanban --board {missing} block awaiting review",
            source=discord_thread_source,
        )
    )

    assert "unknown task" in result or "No such board" in result or "does not exist" in result
    assert not kb.board_exists(missing)


@pytest.mark.asyncio
async def test_kanban_shorthand_ambiguous_across_boards_does_not_infer(
    kanban_home,
    runner: GatewayRunner,
    discord_thread_source: SessionSource,
):
    task_id_default = _create_subscribed_task(discord_thread_source)
    task_id_olympus = _create_subscribed_task(discord_thread_source, board="olympus")

    result = await runner._handle_kanban_command(
        MessageEvent(
            text="/kanban block ambiguous thread binding",
            source=discord_thread_source,
        )
    )

    assert "unknown task" in result or "cannot block" in result
    default_conn = kb.connect()
    olympus_conn = kb.connect(board="olympus")
    try:
        default_task = kb.get_task(default_conn, task_id_default)
        olympus_task = kb.get_task(olympus_conn, task_id_olympus)
        assert default_task is not None
        assert olympus_task is not None
        assert default_task.status == "ready"
        assert olympus_task.status == "ready"
    finally:
        default_conn.close()
        olympus_conn.close()
