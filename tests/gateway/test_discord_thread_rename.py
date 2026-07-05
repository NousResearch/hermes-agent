"""Tests for Discord auto-thread title renaming from generated session titles."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source(
    *,
    thread_id: str | None = "999",
    thread_initial_name: str | None = "Please investigate a very long first message",
) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="42",
        chat_id=thread_id or "123",
        chat_name="Hermes Server / #general / Hermes",
        chat_type="thread" if thread_id else "group",
        thread_id=thread_id,
        user_name="tester",
        thread_initial_name=thread_initial_name,
    )


def _make_runner(*, extra: dict | None = None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="***",
                extra=extra or {},
            )
        }
    )
    adapter = SimpleNamespace(rename_thread=AsyncMock(return_value=True))
    runner.adapters = {Platform.DISCORD: adapter}
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id="sess-discord"))),
    )
    return runner, adapter


class _StatefulTitleDB:
    """Tiny session-title store for tests that need real overwrite behavior."""

    def __init__(self, title: str | None = None):
        self.title = title
        self.set_calls: list[tuple[str, str]] = []

    async def get_session_title(self, session_id: str) -> str | None:
        return self.title

    def sanitize_title(self, title: str) -> str:
        cleaned = " ".join(str(title).split())
        if cleaned == "INVALID":
            raise ValueError("invalid title")
        return cleaned

    async def set_session_title(self, session_id: str, title: str) -> bool:
        self.title = title
        self.set_calls.append((session_id, title))
        return True

    async def set_session_title_if_empty(self, session_id: str, title: str) -> bool:
        if self.title:
            return False
        return await self.set_session_title(session_id, title)

    async def create_session(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_discord_thread_renames_placeholder_to_sanitized_generated_title():
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999"),
        "sess-discord",
        "  Investigate    Dashboard Build Timeout Across Workers  ",
    )

    adapter.rename_thread.assert_awaited_once()
    kwargs = adapter.rename_thread.await_args.kwargs
    assert kwargs["thread_id"] == "999"
    assert kwargs["name"] == "Investigate Dashboard Build Timeout Across Workers"
    assert kwargs["expected_current_name"] == "Please investigate a very long first message"
    assert len(kwargs["name"]) <= 70


@pytest.mark.asyncio
async def test_discord_thread_rename_guard_uses_raw_initial_name_for_attachment_prompts():
    """Attachment metadata injected into the agent prompt must not trip the manual-rename guard."""
    runner, adapter = _make_runner()
    raw_initial_name = "This FreeCAD file keeps crashing in the latest FreeCAD"
    enriched_agent_prompt = (
        "[The user sent a document: 'window-stop-v5.FCStd.zip'. The file is saved at: "
        "/home/pi/.hermes/cache/documents/doc_faa917d1c408_window-stop-v5.FCStd.zip. "
        "Ask the user what they'd like you to do with it.]\n\n"
        "[INeedAUsername] This FreeCAD file keeps crashing in the latest FreeCAD. "
        "Can you figure out what the problem is?"
    )

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999", thread_initial_name=raw_initial_name),
        "sess-discord",
        "Diagnosing FreeCAD File Crashes",
    )

    adapter.rename_thread.assert_awaited_once()
    kwargs = adapter.rename_thread.await_args.kwargs
    assert kwargs["name"] == "Diagnosing FreeCAD File Crashes"
    assert kwargs["expected_current_name"] == raw_initial_name
    assert kwargs["expected_current_name"] not in enriched_agent_prompt[:80]


@pytest.mark.asyncio
async def test_discord_auto_thread_name_mode_message_skips_summary_rename(monkeypatch):
    monkeypatch.delenv("DISCORD_AUTO_THREAD_NAME_MODE", raising=False)
    runner, adapter = _make_runner(extra={"auto_thread_name_mode": "message"})

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999"),
        "sess-discord",
        "Generated Summary Title",
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_auto_thread_summary_max_chars_is_configurable(monkeypatch):
    monkeypatch.delenv("DISCORD_AUTO_THREAD_SUMMARY_MAX_CHARS", raising=False)
    runner, adapter = _make_runner(extra={"auto_thread_summary_max_chars": 24})

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999"),
        "sess-discord",
        "Investigate Dashboard Build Timeout Across All Hosted Workers",
    )

    adapter.rename_thread.assert_awaited_once()
    renamed = adapter.rename_thread.await_args.kwargs["name"]
    assert renamed == "Investigate Dashboard..."
    assert len(renamed) <= 24


@pytest.mark.asyncio
async def test_discord_auto_title_does_not_rename_thread_to_generic_hermes():
    """A low-information generated title must not replace a useful initial thread name."""
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(
            thread_id="999",
            thread_initial_name="Can you try something with claude-rmux.py",
        ),
        "sess-discord",
        "Hermes",
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_discord_thread_does_not_auto_rename_thread():
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id=None),
        "sess-discord",
        "Generated Summary Title",
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_discord_thread_without_initial_name_does_not_auto_rename():
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999", thread_initial_name=None),
        "sess-discord",
        "Generated Summary Title",
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_title_renames_discord_thread_without_initial_name_match():
    runner, adapter = _make_runner()

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999", thread_initial_name=None),
        "sess-discord",
        "Manually Chosen Workflow Title",
        require_initial_name_match=False,
    )

    adapter.rename_thread.assert_awaited_once()
    kwargs = adapter.rename_thread.await_args.kwargs
    assert kwargs["thread_id"] == "999"
    assert kwargs["name"] == "Manually Chosen Workflow Title"
    assert kwargs["expected_current_name"] is None


@pytest.mark.asyncio
async def test_explicit_title_can_use_full_discord_thread_name_limit():
    runner, adapter = _make_runner(extra={"auto_thread_summary_max_chars": 70})
    title = "A" * 90

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999", thread_initial_name=None),
        "sess-discord",
        title,
        require_initial_name_match=False,
    )

    adapter.rename_thread.assert_awaited_once()
    assert adapter.rename_thread.await_args.kwargs["name"] == title


@pytest.mark.asyncio
async def test_explicit_title_longer_than_discord_limit_is_truncated_to_limit():
    runner, adapter = _make_runner(extra={"auto_thread_summary_max_chars": 70})
    title = "B" * 120

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999", thread_initial_name=None),
        "sess-discord",
        title,
        require_initial_name_match=False,
    )

    adapter.rename_thread.assert_awaited_once()
    renamed = adapter.rename_thread.await_args.kwargs["name"]
    assert renamed == "B" * 97 + "..."
    assert len(renamed) == 100


@pytest.mark.asyncio
async def test_interleaved_discord_bot_rename_echoes_do_not_revert_newer_title():
    runner, adapter = _make_runner()
    session_id = "sess-discord"
    source = _make_source(thread_id="999")
    session_db = _StatefulTitleDB(title="Auto Title")
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id=session_id))),
    )
    setattr(runner, "_session_db", session_db)

    await runner._rename_discord_thread_for_session_title(source, session_id, "Auto Title")
    session_db.title = "Manual Title"
    await runner._rename_discord_thread_for_session_title(
        source,
        session_id,
        "Manual Title",
        require_initial_name_match=False,
    )

    assert adapter.rename_thread.await_count == 2
    assert getattr(runner, "_discord_thread_rename_echoes") == {
        "999": [(session_id, "Auto Title"), (session_id, "Manual Title")]
    }

    await runner._handle_platform_thread_title_change(Platform.DISCORD, "999", "Auto Title")

    assert session_db.title == "Manual Title"
    assert session_db.set_calls == []
    assert getattr(runner, "_discord_thread_rename_echoes") == {
        "999": [(session_id, "Manual Title")]
    }

    await runner._handle_platform_thread_title_change(Platform.DISCORD, "999", "Manual Title")

    assert session_db.title == "Manual Title"
    assert session_db.set_calls == []
    assert getattr(runner, "_discord_thread_rename_echoes") == {}


@pytest.mark.asyncio
async def test_stale_discord_auto_title_does_not_rename_rebound_thread():
    runner, adapter = _make_runner()
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id="new-session"))),
    )

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999"),
        "old-session",
        "Generated Summary Title",
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_discord_title_task_does_not_rename_after_db_title_changes():
    runner, adapter = _make_runner()
    setattr(runner, "_session_db", _StatefulTitleDB(title="New Workflow Title"))

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999", thread_initial_name=None),
        "sess-discord",
        "Old Workflow Title",
        require_initial_name_match=False,
    )

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_discord_bot_rename_echo_does_not_overwrite_newer_session_title():
    runner, _adapter = _make_runner()
    session_id = "sess-discord"
    session_db = _StatefulTitleDB(title="New Workflow Title")
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id=session_id))),
    )
    setattr(runner, "_session_db", session_db)
    setattr(runner, "_discord_thread_rename_echoes", {"999": (session_id, "Old Workflow Title")})

    await runner._handle_platform_thread_title_change(
        Platform.DISCORD,
        "999",
        "Old Workflow Title",
    )

    assert session_db.title == "New Workflow Title"
    assert session_db.set_calls == []
    assert getattr(runner, "_discord_thread_rename_echoes") == {}


@pytest.mark.asyncio
async def test_stale_discord_bot_rename_echo_does_not_overwrite_rebound_session_title():
    runner, _adapter = _make_runner()
    session_db = _StatefulTitleDB(title="New Session Title")
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id="new-session"))),
    )
    setattr(runner, "_session_db", session_db)
    setattr(runner, "_discord_thread_rename_echoes", {"999": ("old-session", "Old Workflow Title")})

    await runner._handle_platform_thread_title_change(
        Platform.DISCORD,
        "999",
        "Old Workflow Title",
    )

    assert session_db.title == "New Session Title"
    assert session_db.set_calls == []
    assert getattr(runner, "_discord_thread_rename_echoes") == {}


@pytest.mark.asyncio
async def test_platform_thread_rename_syncs_back_to_session_title():
    runner, _adapter = _make_runner()
    entry = SimpleNamespace(session_id="sess-discord")
    finder = MagicMock(return_value=entry)
    set_session_title = AsyncMock(return_value=True)
    setattr(runner, "session_store", SimpleNamespace(find_session_by_thread=finder))
    setattr(
        runner,
        "_session_db",
        SimpleNamespace(
            set_session_title=set_session_title,
        ),
    )

    await runner._handle_platform_thread_title_change(
        Platform.DISCORD,
        "999",
        "  Renamed    Workflow Thread  ",
    )

    finder.assert_called_once_with(Platform.DISCORD, "999")
    set_session_title.assert_awaited_once_with(
        "sess-discord",
        "Renamed Workflow Thread",
    )


@pytest.mark.asyncio
async def test_manual_thread_rename_during_first_response_wins_over_auto_title(monkeypatch):
    """If the user/workflow names the thread first, generated auto-title must not override it."""
    from agent import title_generator

    runner, adapter = _make_runner()
    session_id = "sess-discord"
    source = _make_source(thread_id="999")
    entry = SimpleNamespace(session_id=session_id)
    session_db = _StatefulTitleDB()
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=entry)),
    )
    setattr(runner, "_session_db", session_db)

    await runner._handle_platform_thread_title_change(
        Platform.DISCORD,
        "999",
        "sample-repo ExampleOrg#1000001",
    )

    assert session_db.title == "sample-repo ExampleOrg#1000001"
    generate_title = MagicMock(return_value="Generated Summary Title")
    title_callback = MagicMock()
    monkeypatch.setattr(title_generator, "generate_title", generate_title)

    title_generator.auto_title_session(
        SimpleNamespace(get_session_title=lambda _session_id: session_db.title),
        session_id,
        "please investigate the workflow run",
        "I will check the failure",
        title_callback=title_callback,
    )

    generate_title.assert_not_called()
    title_callback.assert_not_called()
    adapter.rename_thread.assert_not_awaited()
    assert session_db.title == "sample-repo ExampleOrg#1000001"
    assert session_db.set_calls == [(session_id, "sample-repo ExampleOrg#1000001")]


@pytest.mark.asyncio
async def test_title_command_schedules_explicit_discord_thread_rename():
    """Explicit /title is user intent, so it syncs the Discord thread without initial-name guard."""
    runner, _adapter = _make_runner()
    source = _make_source(thread_id="999", thread_initial_name=None)
    session_id = "sess-discord"
    session_db = _StatefulTitleDB()
    schedule_rename = MagicMock()
    setattr(
        runner,
        "session_store",
        SimpleNamespace(get_or_create_session=MagicMock(return_value=SimpleNamespace(session_id=session_id))),
    )
    setattr(runner, "_session_db", session_db)
    setattr(runner, "_schedule_discord_thread_title_rename", schedule_rename)

    response = await runner._handle_title_command(
        MessageEvent(text="/title   My   Workflow   Title  ", source=source)
    )

    assert session_db.title == "My Workflow Title"
    schedule_rename.assert_called_once_with(
        source,
        session_id,
        "My Workflow Title",
        require_initial_name_match=False,
    )
    assert "My Workflow Title" in response


@pytest.mark.asyncio
async def test_discord_thread_rename_failure_is_non_fatal():
    runner, adapter = _make_runner()
    adapter.rename_thread = AsyncMock(side_effect=RuntimeError("missing manage threads"))

    await runner._rename_discord_thread_for_session_title(
        _make_source(thread_id="999"),
        "sess-discord",
        "Generated Summary Title",
    )

    adapter.rename_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_platform_thread_rename_does_not_corrupt_session_title():
    runner, _adapter = _make_runner()
    session_db = _StatefulTitleDB(title="Existing Title")
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id="sess-discord"))),
    )
    setattr(runner, "_session_db", session_db)

    await runner._handle_platform_thread_title_change(
        Platform.DISCORD,
        "999",
        "x" * 101,
    )

    assert session_db.title == "Existing Title"
    assert session_db.set_calls == []


@pytest.mark.asyncio
async def test_manual_thread_rename_after_auto_rename_updates_session_without_reediting_thread():
    runner, adapter = _make_runner()
    session_id = "sess-discord"
    source = _make_source(thread_id="999")
    await runner._rename_discord_thread_for_session_title(
        source,
        session_id,
        "Generated Summary Title",
    )
    adapter.rename_thread.assert_awaited_once()
    adapter.rename_thread.reset_mock()

    session_db = _StatefulTitleDB(title="Generated Summary Title")
    setattr(
        runner,
        "session_store",
        SimpleNamespace(find_session_by_thread=MagicMock(return_value=SimpleNamespace(session_id=session_id))),
    )
    setattr(runner, "_session_db", session_db)

    await runner._handle_platform_thread_title_change(
        Platform.DISCORD,
        "999",
        "Manual Workflow Title",
    )

    assert session_db.title == "Manual Workflow Title"
    assert session_db.set_calls == [(session_id, "Manual Workflow Title")]
    adapter.rename_thread.assert_not_awaited()
