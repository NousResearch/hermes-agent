import contextvars
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionEntry, SessionSource, build_session_context_prompt
from gateway.topic_context import format_topic_context_prompt
from hermes_state import AsyncSessionDB, SessionDB


def _telegram_group_topic_source(thread_id: str = "205") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_name="Hermes",
        chat_type="group",
        user_id="208214988",
        user_name="tester",
        thread_id=thread_id,
    )


def _make_runner(session_db: SessionDB):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner._session_db = AsyncSessionDB(session_db)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-100123:205",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
        origin=_telegram_group_topic_source(),
    )
    runner._is_user_authorized = lambda _source: True
    return runner


def test_session_context_prompt_includes_persistent_topic_context():
    context = SessionContext(
        source=_telegram_group_topic_source(),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        shared_multi_user_session=True,
        topic_context_prompt=(
            "Telegram group topic context (persistent, compact metadata; not chat history):\n"
            "Topic: Hermes docs patch\n"
            "Purpose: Keep topic context after /new."
        ),
    )

    prompt = build_session_context_prompt(context)

    assert "## Current Session Context" in prompt
    assert "Telegram group topic context" in prompt
    assert "Topic: Hermes docs patch" in prompt
    assert "Purpose: Keep topic context after /new." in prompt


@pytest.mark.asyncio
async def test_group_topic_command_set_and_show_persistent_context(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(db)
    source = _telegram_group_topic_source()

    set_event = MessageEvent(
        text="/topic set Hermes docs patch :: Keep topic context after /new.",
        source=source,
        message_id="m1",
    )
    set_result = await runner._handle_topic_command(set_event)

    assert "Topic context saved" in set_result

    show_event = MessageEvent(text="/topic", source=source, message_id="m2")
    show_result = await runner._handle_topic_command(show_event)

    assert "Hermes docs patch" in show_result
    assert "Keep topic context after /new." in show_result

    ctx = db.get_gateway_topic_context(
        platform="telegram",
        chat_id="-100123",
        thread_id="205",
    )
    assert ctx["topic_name"] == "Hermes docs patch"
    assert ctx["purpose"] == "Keep topic context after /new."


@pytest.mark.asyncio
async def test_load_group_topic_context_is_thread_scoped(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.upsert_gateway_topic_context(
        platform="telegram",
        chat_id="-100123",
        thread_id="205",
        chat_name="Hermes",
        topic_name="Hermes docs patch",
        purpose="Keep topic context after /new.",
    )
    runner = _make_runner(db)

    ctx_205 = await runner._load_gateway_topic_context(_telegram_group_topic_source("205"))
    ctx_206 = await runner._load_gateway_topic_context(_telegram_group_topic_source("206"))

    prompt_205 = format_topic_context_prompt(ctx_205)
    assert "Hermes docs patch" in prompt_205
    assert "Keep topic context after /new." in prompt_205
    assert ctx_206 is None


@pytest.mark.asyncio
async def test_topic_workdir_set_show_and_clear(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(db)
    source = _telegram_group_topic_source()
    workdir = tmp_path / "project"
    workdir.mkdir()

    set_result = await runner._handle_topic_command(
        MessageEvent(text=f"/topic workdir {workdir}", source=source, message_id="m1")
    )
    assert str(workdir) in set_result
    assert "/new" in set_result

    ctx = db.get_gateway_topic_context(
        platform="telegram", chat_id="-100123", thread_id="205"
    )
    assert ctx["workdir"] == str(workdir)

    show_result = await runner._handle_topic_command(
        MessageEvent(text="/topic workdir show", source=source, message_id="m2")
    )
    assert str(workdir) in show_result

    clear_result = await runner._handle_topic_command(
        MessageEvent(text="/topic workdir clear", source=source, message_id="m3")
    )
    assert "cleared" in clear_result.lower()
    ctx = db.get_gateway_topic_context(
        platform="telegram", chat_id="-100123", thread_id="205"
    )
    assert ctx["workdir"] is None


@pytest.mark.asyncio
async def test_topic_workdir_validation_rejects_relative_and_missing(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(db)
    source = _telegram_group_topic_source()

    rel = await runner._handle_topic_command(
        MessageEvent(text="/topic workdir relative/path", source=source, message_id="m1")
    )
    assert "absolute" in rel.lower()

    missing = await runner._handle_topic_command(
        MessageEvent(
            text=f"/topic workdir {tmp_path / 'does-not-exist'}",
            source=source,
            message_id="m2",
        )
    )
    assert "does not exist" in missing.lower()

    # Nothing was persisted by the rejected attempts.
    assert db.get_gateway_topic_context(
        platform="telegram", chat_id="-100123", thread_id="205"
    ) is None


@pytest.mark.asyncio
async def test_topic_workdir_expands_home(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(db)
    source = _telegram_group_topic_source()
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    result = await runner._handle_topic_command(
        MessageEvent(text="/topic workdir ~/proj", source=source, message_id="m1")
    )
    assert str(home / "proj") in result
    ctx = db.get_gateway_topic_context(
        platform="telegram", chat_id="-100123", thread_id="205"
    )
    assert ctx["workdir"] == str(home / "proj")


@pytest.mark.asyncio
async def test_topic_show_includes_workdir_line(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    workdir = tmp_path / "proj"
    workdir.mkdir()
    db.upsert_gateway_topic_context(
        platform="telegram",
        chat_id="-100123",
        thread_id="205",
        topic_name="Notes",
        purpose="Project workspace.",
    )
    db.set_gateway_topic_context_workdir(
        platform="telegram",
        chat_id="-100123",
        thread_id="205",
        workdir=str(workdir),
    )
    runner = _make_runner(db)

    show_result = await runner._handle_topic_command(
        MessageEvent(text="/topic", source=_telegram_group_topic_source(), message_id="m1")
    )
    assert f"Workdir: {workdir}" in show_result


def test_set_session_env_pins_topic_workdir_cwd(tmp_path):
    from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd

    db = SessionDB(db_path=tmp_path / "state.db")
    runner = _make_runner(db)
    runner.adapters = {}
    workdir = tmp_path / "proj"
    workdir.mkdir()
    context = SessionContext(
        source=_telegram_group_topic_source(),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        topic_workdir=str(workdir),
        session_key="agent:main:telegram:group:-100123:205",
    )

    def _check():
        tokens = runner._set_session_env(context)
        try:
            assert resolve_agent_cwd() == workdir
            assert resolve_context_cwd() == workdir
        finally:
            runner._clear_session_env(tokens)
        # After clearing, the session override is gone.
        assert resolve_context_cwd() != workdir

    contextvars.copy_context().run(_check)


def test_topic_workdir_eager_loads_agents_md(tmp_path):
    from agent.prompt_builder import build_context_files_prompt
    from agent.runtime_cwd import resolve_context_cwd, set_session_cwd

    workdir = tmp_path / "proj"
    workdir.mkdir()
    (workdir / "AGENTS.md").write_text("Follow the project style guide.\n")
    (workdir / "CLAUDE.md").write_text("This should be shadowed by AGENTS.md.\n")

    def _check():
        set_session_cwd(str(workdir))
        cwd = resolve_context_cwd()
        prompt = build_context_files_prompt(cwd=str(cwd))
        assert "Follow the project style guide." in prompt
        assert "## AGENTS.md" in prompt
        assert "shadowed by AGENTS.md" not in prompt

    contextvars.copy_context().run(_check)
