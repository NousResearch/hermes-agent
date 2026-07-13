"""Gateway command tests for durable background handles and sidequests."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.sidequests import SidequestStore
from gateway.run import _normalize_sidequest_shortcut_text


def _make_event(
    text,
    platform=Platform.WHATSAPP,
    chat_id="chat-1",
    user_id="user-1",
    profile=None,
    thread_id=None,
):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            user_id=user_id,
            user_name="Tester",
            profile=profile,
            thread_id=thread_id,
        ),
    )


def _make_runner(store):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._sidequest_store = MagicMock(return_value=store)
    runner.session_store = MagicMock()
    return runner


def _close_created_task(coro, *args, **kwargs):
    coro.close()
    task = MagicMock()
    task.add_done_callback = MagicMock()
    return task


def test_sidequest_shortcut_normalizer_expands_compact_aliases_only_at_start():
    assert _normalize_sidequest_shortcut_text("sq2") == "/sq 2"
    assert _normalize_sidequest_shortcut_text("sq2 status") == "/sq 2 status"
    assert _normalize_sidequest_shortcut_text("/sq1 status") == "/sq 1 status"
    assert _normalize_sidequest_shortcut_text("/sidequest2 add docs") == "/sq 2 add docs"
    assert _normalize_sidequest_shortcut_text("#sq3 status") == "/sq 3 status"
    assert _normalize_sidequest_shortcut_text("/sq1") == "/sq 1"
    assert _normalize_sidequest_shortcut_text("/sq 1 status") == "/sq 1 status"
    assert _normalize_sidequest_shortcut_text("please check #sq1 later") == "please check #sq1 later"
    assert _normalize_sidequest_shortcut_text("please check sq1 later") == "please check sq1 later"
    assert _normalize_sidequest_shortcut_text("#sqlite note") == "#sqlite note"
    assert _normalize_sidequest_shortcut_text("/sqldb migrate") == "/sqldb migrate"


@pytest.mark.asyncio
async def test_background_start_persists_handle_and_status(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    runner = _make_runner(store)
    event = _make_event("/background build the thing")
    scope = runner._background_owner(event.source)[1]

    with patch("gateway.run.asyncio.create_task", side_effect=_close_created_task):
        result = await runner._handle_background_command(event)

    assert "Task ID:" in result
    bg_id = next(line.split("Task ID:", 1)[1].strip() for line in result.splitlines() if "Task ID:" in line)
    run = store.get_background_run(bg_id, platform="whatsapp", chat_id=scope)
    assert run is not None
    assert run["prompt"] == "build the thing"
    assert run["status"] == "queued"

    status = await runner._handle_background_command(_make_event(f"/bg {bg_id} status"))
    assert bg_id in status
    assert "Follow-up:" in status


@pytest.mark.asyncio
async def test_background_followup_starts_resume_run(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    runner = _make_runner(store)
    event = _make_event("/bg bg_120000_abcdef also check docs")
    scope = runner._background_owner(event.source)[1]
    store.create_background_run(
        bg_id="bg_120000_abcdef",
        prompt="initial research",
        platform="whatsapp",
        chat_id=scope,
        user_id="user-1",
        session_id="bg_120000_abcdef",
    )
    store.mark_completed("bg_120000_abcdef", summary="Initial summary")

    with patch("gateway.run.asyncio.create_task", side_effect=_close_created_task):
        result = await runner._handle_background_command(event)

    assert "Background task started" in result
    followups = store.list_followups("bg_120000_abcdef")
    assert followups[0]["message"] == "also check docs"
    runs = store.list_background_runs(platform="whatsapp", chat_id=scope, limit=5)
    assert any("Initial summary" in run["prompt"] for run in runs if run["bg_id"] != "bg_120000_abcdef")


@pytest.mark.asyncio
async def test_sidequest_create_and_short_alias_status(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    runner = _make_runner(store)

    with patch("gateway.run.asyncio.create_task", side_effect=_close_created_task):
        result = await runner._handle_sidequest_command(_make_event("/sidequest investigate background handles"))

    assert "Sidequest #1" in result
    assert "/sq 1 status" in result

    status = await runner._handle_sidequest_command(_make_event("/sq 1 status"))
    assert "Sidequest #1" in status
    assert "investigate background handles" in status

    bare_resume = await runner._handle_sidequest_command(
        _make_event(_normalize_sidequest_shortcut_text("sq1"))
    )
    assert "Sidequest #1" in bare_resume
    assert "investigate background handles" in bare_resume
    assert "Background task started" not in bare_resume


@pytest.mark.asyncio
async def test_sidequest_followup_relinks_completion_and_carries_artifacts(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    runner = _make_runner(store)
    event = _make_event("/sq 1 also check docs")
    scope = runner._background_owner(event.source)[1]
    quest = store.create_quest(
        title="investigate background handles",
        platform="whatsapp",
        chat_id=scope,
        user_id="user-1",
    )
    store.create_background_run(
        bg_id="bg_initial",
        prompt="initial research",
        platform="whatsapp",
        chat_id=scope,
        user_id="user-1",
        session_id="bg_initial",
    )
    store.attach_background_to_quest(quest_id=quest["quest_id"], bg_id="bg_initial")
    store.mark_completed(
        "bg_initial",
        summary="Initial summary",
        artifact_paths=["/tmp/initial-report.md"],
    )

    with patch("gateway.run.asyncio.create_task", side_effect=_close_created_task):
        result = await runner._handle_sidequest_command(event)

    new_bg_id = next(
        line.split("Task ID:", 1)[1].strip()
        for line in result.splitlines()
        if "Task ID:" in line
    )
    resumed = store.get_background_run(
        new_bg_id,
        platform="whatsapp",
        chat_id=scope,
    )
    assert resumed is not None
    assert "/tmp/initial-report.md" in resumed["prompt"]

    running_quest = store.resolve_quest("1", platform="whatsapp", chat_id=scope)
    assert running_quest is not None
    assert running_quest["source_bg_id"] == new_bg_id
    assert running_quest["status"] == "running"

    store.mark_completed(
        new_bg_id,
        summary="Follow-up complete",
        artifact_paths=["/tmp/followup-report.md"],
    )
    completed_quest = store.resolve_quest("1", platform="whatsapp", chat_id=scope)
    assert completed_quest is not None
    assert completed_quest["status"] == "waiting"
    assert completed_quest["latest_summary"] == "Follow-up complete"
    assert completed_quest["artifact_paths"] == ["/tmp/followup-report.md"]


@pytest.mark.asyncio
async def test_sidequest_scope_isolated_by_profile_and_thread(tmp_path):
    store = SidequestStore(tmp_path / "quests.sqlite")
    runner = _make_runner(store)
    profile_a_thread_1 = _make_event(
        "/sq task a",
        profile="profile-a",
        thread_id="topic-1",
    )
    profile_b_thread_1 = _make_event(
        "/sq list",
        profile="profile-b",
        thread_id="topic-1",
    )
    profile_a_thread_2 = _make_event(
        "/sq list",
        profile="profile-a",
        thread_id="topic-2",
    )

    with patch("gateway.run.asyncio.create_task", side_effect=_close_created_task):
        created = await runner._handle_sidequest_command(profile_a_thread_1)

    assert "Sidequest #1" in created
    assert "No sidequests yet" in await runner._handle_sidequest_command(profile_b_thread_1)
    assert "No sidequests yet" in await runner._handle_sidequest_command(profile_a_thread_2)
