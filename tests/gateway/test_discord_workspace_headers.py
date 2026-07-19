"""Behavior contracts for persistent Discord thread workspace headers."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.workspace_headers import (
    WorkspaceHeaderStore,
    WorkspaceHeaderStoreError,
    collect_workspace_header_candidates,
    revalidate_workspace_header_candidate,
)


def _source(*, scope_id="111", thread_id="222") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=thread_id,
        chat_name="MMG / launch-plan",
        chat_type="thread",
        user_id="333",
        user_name="Martin",
        thread_id=thread_id,
        scope_id=scope_id,
        parent_chat_id="444",
    )


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._build_operator_card_embed",
        lambda card: {
            "card_type": card.card_type,
            "state_ref": card.state_ref,
            "fields": {field.label: field.value for field in card.fields},
        },
    )
    return instance


@pytest.mark.asyncio
async def test_workspace_header_creates_once_then_edits_same_message(adapter, tmp_path):
    header_message = SimpleNamespace(id=7001, edit=AsyncMock())
    thread = SimpleNamespace(
        id=222,
        name="launch-plan",
        guild=SimpleNamespace(id=111),
        send=AsyncMock(return_value=header_message),
        fetch_message=AsyncMock(return_value=header_message),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 222 else None,
        fetch_channel=AsyncMock(),
    )

    created = await adapter.ensure_workspace_header(_source())
    updated = await adapter.ensure_workspace_header(_source())

    assert created.success is True
    assert created.action == "created"
    assert updated.success is True
    assert updated.action == "updated"
    assert thread.send.await_count == 1
    thread.fetch_message.assert_awaited_once_with(7001)
    assert header_message.edit.await_count == 1
    assert header_message.edit.await_args.kwargs["embed"]["card_type"] == "thread_header"
    assert header_message.edit.await_args.kwargs["content"].startswith("🔵")
    assert header_message.edit.await_args.kwargs["embed"]["state_ref"] == "discord-workspace:111:222"
    assert header_message.edit.await_args.kwargs["embed"]["fields"] == {
        "Owner": "Hermes",
        "Status": "Active",
        "Thread": "<#222>",
        "Linked issue / artifact": "Not linked",
        "Last decision": "No decision recorded",
        "Next action": "Awaiting next assistant turn",
    }

    persisted = WorkspaceHeaderStore().get("111", "222")
    assert persisted is not None
    assert persisted.message_id == "7001"
    assert (tmp_path / "gateway" / "discord_workspace_headers.json").exists()


@pytest.mark.asyncio
async def test_workspace_header_identity_survives_adapter_restart(adapter, tmp_path, monkeypatch):
    adapter._workspace_headers.put("111", "222", "7001")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    restarted = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    binding = restarted._workspace_headers.get("111", "222")
    assert binding is not None
    assert binding.message_id == "7001"


@pytest.mark.asyncio
async def test_workspace_header_persists_and_renders_thread_context(adapter, tmp_path, monkeypatch):
    adapter._workspace_headers.update_state(
        "111",
        "222",
        owner="Martin + Hermes",
        status="Needs review",
        linked_issue_or_artifact="OE-178 · docs/proofs/header.md",
        last_decision="Keep the human-renamed thread title",
        next_action="Review the backfill dry run",
    )
    header_message = SimpleNamespace(id=7001)
    thread = SimpleNamespace(
        id=222,
        name="launch-plan",
        guild=SimpleNamespace(id=111),
        send=AsyncMock(return_value=header_message),
    )
    adapter._client = SimpleNamespace(get_channel=lambda _id: thread, fetch_channel=AsyncMock())

    created = await adapter.ensure_workspace_header(_source())

    assert created.success is True
    fields = thread.send.await_args.kwargs["embed"]["fields"]
    assert fields["Owner"] == "Martin + Hermes"
    assert fields["Status"] == "Needs review"
    assert fields["Linked issue / artifact"] == "OE-178 · docs/proofs/header.md"
    assert fields["Last decision"] == "Keep the human-renamed thread title"
    assert fields["Next action"] == "Review the backfill dry run"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    restarted = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    state = restarted._workspace_headers.get_state("111", "222")
    assert state is not None
    assert state.linked_issue_or_artifact == "OE-178 · docs/proofs/header.md"
    assert state.last_decision == "Keep the human-renamed thread title"
    assert state.next_action == "Review the backfill dry run"


@pytest.mark.asyncio
async def test_workspace_header_requires_canonical_scope_and_matching_live_guild(adapter):
    thread = SimpleNamespace(
        id=222,
        name="launch-plan",
        guild=SimpleNamespace(id=999),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(get_channel=lambda _id: thread, fetch_channel=AsyncMock())

    missing_scope = await adapter.ensure_workspace_header(_source(scope_id=None))
    mismatched_scope = await adapter.ensure_workspace_header(_source(scope_id="111"))

    assert missing_scope.success is False
    assert missing_scope.action == "skipped"
    assert mismatched_scope.success is False
    assert mismatched_scope.action == "skipped"
    thread.send.assert_not_awaited()


class _UnknownMessage(RuntimeError):
    status = 404
    code = 10008


@pytest.mark.asyncio
async def test_deleted_header_is_recreated_but_transient_fetch_failure_never_duplicates(adapter):
    adapter._workspace_headers.put("111", "222", "7001")
    replacement = SimpleNamespace(id=7002)
    thread = SimpleNamespace(
        id=222,
        name="launch-plan",
        guild=SimpleNamespace(id=111),
        send=AsyncMock(return_value=replacement),
        fetch_message=AsyncMock(side_effect=_UnknownMessage("Unknown Message")),
    )
    adapter._client = SimpleNamespace(get_channel=lambda _id: thread, fetch_channel=AsyncMock())

    recreated = await adapter.ensure_workspace_header(_source())

    assert recreated.success is True
    assert recreated.action == "recreated"
    assert thread.send.await_count == 1
    assert adapter._workspace_headers.get("111", "222").message_id == "7002"

    adapter._workspace_headers.put("111", "222", "7003")
    thread.fetch_message = AsyncMock(side_effect=RuntimeError("temporary transport failure"))
    thread.send.reset_mock()

    failed = await adapter.ensure_workspace_header(_source())

    assert failed.success is False
    assert failed.action == "failed"
    thread.send.assert_not_awaited()
    assert adapter._workspace_headers.get("111", "222").message_id == "7003"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registry_payload",
    ["not json", '{"version": 999, "workspaces": {}}'],
)
async def test_corrupt_or_unknown_registry_fails_closed_without_duplicate(
    adapter, tmp_path, registry_payload
):
    registry = tmp_path / "gateway" / "discord_workspace_headers.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(registry_payload, encoding="utf-8")
    thread = SimpleNamespace(
        id=222,
        name="launch-plan",
        guild=SimpleNamespace(id=111),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(get_channel=lambda _id: thread, fetch_channel=AsyncMock())

    result = await adapter.ensure_workspace_header(_source())

    assert result.success is False
    assert result.action == "failed"
    assert "registry" in (result.error or "")
    thread.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_send_persistence_failure_leaves_reservation_and_never_duplicates(
    adapter, monkeypatch
):
    sent = SimpleNamespace(id=7001)
    thread = SimpleNamespace(
        id=222,
        name="launch-plan",
        guild=SimpleNamespace(id=111),
        send=AsyncMock(return_value=sent),
    )
    adapter._client = SimpleNamespace(get_channel=lambda _id: thread, fetch_channel=AsyncMock())
    monkeypatch.setattr(
        adapter._workspace_headers,
        "complete_creation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            WorkspaceHeaderStoreError("simulated persistence failure")
        ),
    )

    first = await adapter.ensure_workspace_header(_source())
    second = await adapter.ensure_workspace_header(_source())

    assert first.success is False
    assert first.message_id == "7001"
    assert second.success is False
    assert "pending" in (second.error or "")
    assert thread.send.await_count == 1


def test_two_store_instances_serialize_read_modify_write_without_lost_binding(tmp_path, monkeypatch):
    path = tmp_path / "headers.json"
    first = WorkspaceHeaderStore(path=path)
    second = WorkspaceHeaderStore(path=path)
    first_read = Event()
    allow_first_write = Event()
    original_read = first._read_unlocked

    def delayed_read():
        payload = original_read()
        first_read.set()
        assert allow_first_write.wait(timeout=2)
        return payload

    monkeypatch.setattr(first, "_read_unlocked", delayed_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_put = executor.submit(first.put, "111", "222", "7001")
        assert first_read.wait(timeout=2)
        second_put = executor.submit(second.put, "111", "333", "7002")
        assert second_put.done() is False
        allow_first_write.set()
        first_put.result(timeout=2)
        second_put.result(timeout=2)

    assert WorkspaceHeaderStore(path=path).get("111", "222").message_id == "7001"
    assert WorkspaceHeaderStore(path=path).get("111", "333").message_id == "7002"


@pytest.mark.asyncio
async def test_successful_turn_refreshes_header_without_changing_participation(adapter):
    adapter.ensure_workspace_header = AsyncMock()
    source = _source()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=SimpleNamespace(id=123),
    )

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter.ensure_workspace_header.assert_awaited_once_with(source)
    assert "222" not in adapter._threads

    adapter.ensure_workspace_header.reset_mock()
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    adapter.ensure_workspace_header.assert_not_awaited()


def test_backfill_candidates_are_known_workspaces_and_do_not_propose_title_changes(tmp_path):
    store = WorkspaceHeaderStore(path=tmp_path / "headers.json")
    store.put("111", "10", "9000")
    store.put("111", "13", "9003")
    threads = [
        SimpleNamespace(id=10, name="Hermes", guild=SimpleNamespace(id=111)),
        SimpleNamespace(id=11, name="Human-renamed launch", guild=SimpleNamespace(id=111)),
        SimpleNamespace(id=12, name="Hermes", guild=SimpleNamespace(id=111)),
        SimpleNamespace(id=13, name="Human-renamed retained", guild=SimpleNamespace(id=111)),
    ]

    candidates = collect_workspace_header_candidates(
        threads,
        participated_thread_ids={"10", "11", "13"},
        store=store,
    )

    assert [(item.thread_id, item.reasons) for item in candidates] == [
        ("10", ("placeholder_title",)),
        ("11", ("header_missing",)),
    ]
    assert all(item.proposed_title is None for item in candidates)


def test_backfill_apply_revalidates_title_participation_scope_and_missing_state(tmp_path):
    store = WorkspaceHeaderStore(path=tmp_path / "headers.json")
    planned_thread = SimpleNamespace(id=22, name="Hermes", guild=SimpleNamespace(id=111))
    candidate = collect_workspace_header_candidates(
        [planned_thread], participated_thread_ids={"22"}, store=store
    )[0]

    assert revalidate_workspace_header_candidate(
        candidate,
        live_thread=planned_thread,
        participated_thread_ids={"22"},
        store=store,
    ) is True
    assert revalidate_workspace_header_candidate(
        candidate,
        live_thread=SimpleNamespace(id=22, name="Human renamed", guild=SimpleNamespace(id=111)),
        participated_thread_ids={"22"},
        store=store,
    ) is False
    assert revalidate_workspace_header_candidate(
        candidate,
        live_thread=planned_thread,
        participated_thread_ids=set(),
        store=store,
    ) is False
    assert revalidate_workspace_header_candidate(
        candidate,
        live_thread=SimpleNamespace(id=22, name="Hermes", guild=SimpleNamespace(id=999)),
        participated_thread_ids={"22"},
        store=store,
    ) is False

    store.put("111", "22", "9000")
    assert revalidate_workspace_header_candidate(
        candidate,
        live_thread=planned_thread,
        participated_thread_ids={"22"},
        store=store,
    ) is False
