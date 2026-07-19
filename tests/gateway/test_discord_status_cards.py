"""Behavior contracts for keyed Discord task-run status cards (OE-180)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    instance.send = AsyncMock()
    instance.edit_message = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_status_card_first_send_caches_then_same_key_edits(adapter):
    adapter.send.return_value = SendResult(success=True, message_id="7001")
    adapter.edit_message.return_value = SendResult(success=True, message_id="7001")

    first = await adapter.send_or_update_status(
        "555", "run-1", "Starting", metadata={"thread_id": "777"}
    )
    second = await adapter.send_or_update_status(
        "555", "run-1", "Checking files", metadata={"thread_id": "777"}
    )

    assert first.message_id == second.message_id == "7001"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    assert adapter.send.await_args.kwargs["metadata"]["operator_card"]["card_type"] == "task_run"
    edit_kwargs = adapter.edit_message.await_args.kwargs
    assert edit_kwargs["chat_id"] == "555"
    assert edit_kwargs["message_id"] == "7001"
    assert edit_kwargs["content"] == "Checking files"
    assert edit_kwargs["finalize"] is False
    assert edit_kwargs["metadata"]["thread_id"] == "777"
    assert edit_kwargs["metadata"]["operator_card"]["card_type"] == "task_run"


@pytest.mark.asyncio
async def test_status_card_identity_isolated_by_thread_and_status_key(adapter):
    adapter.send.side_effect = [
        SendResult(success=True, message_id="7001"),
        SendResult(success=True, message_id="7002"),
        SendResult(success=True, message_id="7003"),
    ]

    await adapter.send_or_update_status(
        "555", "run-1", "Thread A", metadata={"thread_id": "777"}
    )
    await adapter.send_or_update_status(
        "555", "run-1", "Thread B", metadata={"thread_id": "778"}
    )
    await adapter.send_or_update_status(
        "555", "run-2", "Other run", metadata={"thread_id": "777"}
    )

    assert adapter.send.await_count == 3
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids == {
        ("555", "777", "run-1"): "7001",
        ("555", "778", "run-1"): "7002",
        ("555", "777", "run-2"): "7003",
    }


@pytest.mark.asyncio
async def test_sequential_runs_in_one_thread_keep_distinct_final_messages(adapter):
    adapter.send.side_effect = [
        SendResult(success=True, message_id="7001"),
        SendResult(success=True, message_id="7002"),
    ]

    async def edit_same_message(*, message_id, **_kwargs):
        return SendResult(success=True, message_id=message_id)

    adapter.edit_message.side_effect = edit_same_message

    await adapter.send_or_update_status(
        "555", "task_run:message:42", "Run one working", metadata={"thread_id": "777"}
    )
    first_final = await adapter.send_or_update_status(
        "555",
        "task_run:message:42",
        "Run one complete",
        metadata={"thread_id": "777", "status_terminal": True},
    )
    await adapter.send_or_update_status(
        "555", "task_run:message:43", "Run two working", metadata={"thread_id": "777"}
    )
    second_final = await adapter.send_or_update_status(
        "555",
        "task_run:message:43",
        "Run two complete",
        metadata={"thread_id": "777", "status_terminal": True},
    )

    assert first_final.message_id == "7001"
    assert second_final.message_id == "7002"
    assert adapter.send.await_count == 2
    assert adapter.edit_message.await_count == 2
    assert adapter._status_message_ids[("555", "777", "task_run:message:42")] == "7001"
    assert adapter._status_message_ids[("555", "777", "task_run:message:43")] == "7002"


@pytest.mark.asyncio
async def test_terminal_result_replaces_retained_card_as_plaintext(adapter):
    adapter.send.return_value = SendResult(success=True, message_id="7001")
    adapter.edit_message.return_value = SendResult(success=True, message_id="7001")

    await adapter.send_or_update_status(
        "555", "run-1", "Working", metadata={"thread_id": "777"}
    )
    result = await adapter.send_or_update_status(
        "555",
        "run-1",
        "Done. Here is the complete final result.",
        metadata={"thread_id": "777", "status_terminal": True},
    )

    assert result.success is True
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once_with(
        chat_id="555",
        message_id="7001",
        content="Done. Here is the complete final result.",
        finalize=True,
        metadata={"thread_id": "777", "status_terminal": True},
    )


@pytest.mark.asyncio
async def test_late_running_update_cannot_downgrade_terminal_card(adapter):
    adapter.send.return_value = SendResult(success=True, message_id="7001")
    adapter.edit_message.return_value = SendResult(success=True, message_id="7001")

    await adapter.send_or_update_status(
        "555", "run-1", "Working", metadata={"thread_id": "777"}
    )
    terminal = await adapter.send_or_update_status(
        "555",
        "run-1",
        "Complete final result",
        metadata={"thread_id": "777", "status_terminal": True},
    )
    late = await adapter.send_or_update_status(
        "555", "run-1", "Still working", metadata={"thread_id": "777"}
    )

    assert terminal.message_id == late.message_id == "7001"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    assert adapter._status_message_terminal == {
        ("555", "777", "run-1"): True,
    }


@pytest.mark.asyncio
async def test_edit_failure_sends_exactly_one_fresh_plaintext_final_and_recaches(adapter):
    adapter.send.side_effect = [
        SendResult(success=True, message_id="7001"),
        SendResult(success=True, message_id="7002"),
    ]
    adapter.edit_message.return_value = SendResult(success=False, error="Unknown Message")

    await adapter.send_or_update_status(
        "555", "run-1", "Working", metadata={"thread_id": "777"}
    )
    result = await adapter.send_or_update_status(
        "555",
        "run-1",
        "Final result",
        metadata={"thread_id": "777", "status_terminal": True},
    )

    assert result.message_id == "7002"
    assert adapter.send.await_count == 2
    assert adapter.send.await_args.kwargs == {
        "chat_id": "555",
        "content": "Final result",
        "metadata": {"thread_id": "777"},
    }
    assert adapter._status_message_ids[("555", "777", "run-1")] == "7002"


@pytest.mark.asyncio
async def test_partial_terminal_overflow_falls_back_once_before_latching(adapter):
    adapter.send.side_effect = [
        SendResult(success=True, message_id="7001"),
        SendResult(
            success=True,
            message_id="8001",
            raw_response={"message_ids": ["8001", "8002"]},
        ),
    ]
    adapter.edit_message.return_value = SendResult(
        success=True,
        message_id="7002",
        continuation_message_ids=("7002",),
        raw_response={
            "partial_overflow": True,
            "delivered_chunks": 2,
            "total_chunks": 3,
            "last_message_id": "7002",
            "continuation_message_ids": ("7002",),
        },
    )

    await adapter.send_or_update_status(
        "555", "run-1", "Working", metadata={"thread_id": "777"}
    )
    result = await adapter.send_or_update_status(
        "555",
        "run-1",
        "Complete final result " * 300,
        metadata={"thread_id": "777", "status_terminal": True},
    )

    assert result.message_id == "8001"
    assert adapter.edit_message.await_count == 1
    assert adapter.send.await_count == 2
    fallback = adapter.send.await_args_list[1].kwargs
    assert fallback["chat_id"] == "555"
    assert fallback["content"] == "Complete final result " * 300
    assert fallback["metadata"] == {"thread_id": "777"}
    key = ("555", "777", "run-1")
    assert adapter._status_message_ids[key] == "8001"
    assert adapter._status_message_terminal[key] is True
    assert adapter._last_self_message_id["777"] == "8002"


@pytest.mark.asyncio
async def test_embed_send_failure_falls_back_once_to_plaintext_and_recaches(adapter):
    adapter.send.side_effect = [
        SendResult(success=False, error="Invalid Form Body: embed"),
        SendResult(success=True, message_id="7002"),
    ]

    result = await adapter.send_or_update_status(
        "555", "run-1", "Working", metadata={"thread_id": "777"}
    )

    assert result.message_id == "7002"
    assert adapter.send.await_count == 2
    assert "operator_card" in adapter.send.await_args_list[0].kwargs["metadata"]
    assert adapter.send.await_args_list[1].kwargs == {
        "chat_id": "555",
        "content": "Working",
        "metadata": {"thread_id": "777"},
    }
    assert adapter._status_message_ids[("555", "777", "run-1")] == "7002"


@pytest.mark.asyncio
async def test_repeated_identical_heartbeat_does_not_send_or_edit_again(adapter):
    adapter.send.return_value = SendResult(success=True, message_id="7001")

    first = await adapter.send_or_update_status(
        "555", "run-1", "Still working", metadata={"thread_id": "777"}
    )
    second = await adapter.send_or_update_status(
        "555", "run-1", "Still working", metadata={"thread_id": "777"}
    )

    assert first.message_id == second.message_id == "7001"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_identical_first_updates_create_only_one_message(adapter):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_send(**_kwargs):
        entered.set()
        await release.wait()
        return SendResult(success=True, message_id="7001")

    adapter.send.side_effect = delayed_send
    first = asyncio.create_task(
        adapter.send_or_update_status(
            "555", "run-1", "Starting", metadata={"thread_id": "777"}
        )
    )
    await entered.wait()
    second = asyncio.create_task(
        adapter.send_or_update_status(
            "555", "run-1", "Starting", metadata={"thread_id": "777"}
        )
    )
    release.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.message_id == second_result.message_id == "7001"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_identity_cache_evicts_oldest_inactive_runs(adapter):
    adapter._STATUS_MESSAGE_CACHE_LIMIT = 3
    adapter.send.side_effect = [
        SendResult(success=True, message_id=str(7000 + index))
        for index in range(1, 6)
    ]

    for index in range(1, 6):
        await adapter.send_or_update_status(
            "555",
            f"run-{index}",
            f"Run {index}",
            metadata={"thread_id": "777", "status_terminal": True},
        )

    expected_keys = {
        ("555", "777", "run-3"),
        ("555", "777", "run-4"),
        ("555", "777", "run-5"),
    }
    assert set(adapter._status_message_ids) == expected_keys
    assert set(adapter._status_message_groups) == expected_keys
    assert set(adapter._status_message_fingerprints) == expected_keys
    assert set(adapter._status_message_locks) == expected_keys
    assert set(adapter._status_message_terminal) == expected_keys
    assert adapter._status_message_users == {}


@pytest.mark.asyncio
async def test_status_identity_cache_never_evicts_locked_inflight_run(adapter):
    adapter._STATUS_MESSAGE_CACHE_LIMIT = 2
    locked_key = ("555", "777", "run-1")
    locked = asyncio.Lock()
    await locked.acquire()
    adapter._status_message_ids[locked_key] = "7001"
    adapter._status_message_fingerprints[locked_key] = "old"
    adapter._status_message_locks[locked_key] = locked
    adapter._status_message_users[locked_key] = 1
    adapter._status_message_ids[("555", "777", "run-2")] = "7002"
    adapter._status_message_fingerprints[("555", "777", "run-2")] = "old"
    adapter._status_message_locks[("555", "777", "run-2")] = asyncio.Lock()
    adapter.send.return_value = SendResult(success=True, message_id="7003")

    await adapter.send_or_update_status(
        "555",
        "run-3",
        "Run 3",
        metadata={"thread_id": "777", "status_terminal": True},
    )

    assert locked_key in adapter._status_message_ids
    assert ("555", "777", "run-2") not in adapter._status_message_ids
    assert ("555", "777", "run-3") in adapter._status_message_ids
    locked.release()


@pytest.mark.asyncio
async def test_failed_new_status_does_not_retain_an_orphan_lock(adapter):
    adapter.send.return_value = SendResult(success=False, error="offline")

    result = await adapter.send_or_update_status(
        "555",
        "run-1",
        "Final",
        metadata={"thread_id": "777", "status_terminal": True},
    )

    assert result.success is False
    assert adapter._status_message_ids == {}
    assert adapter._status_message_groups == {}
    assert adapter._status_message_fingerprints == {}
    assert adapter._status_message_locks == {}
    assert adapter._status_message_terminal == {}
    assert adapter._status_message_users == {}


@pytest.mark.asyncio
async def test_failed_first_send_keeps_lock_while_same_key_caller_is_queued(adapter):
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    second_entered = asyncio.Event()
    release_second = asyncio.Event()
    third_started = asyncio.Event()
    send_count = 0

    async def controlled_send(**_kwargs):
        nonlocal send_count
        send_count += 1
        if send_count == 1:
            first_entered.set()
            await release_first.wait()
            return SendResult(success=False, error="offline")
        if send_count == 2:
            second_entered.set()
            await release_second.wait()
            return SendResult(success=True, message_id="7002")
        raise AssertionError("same-key caller escaped the retained lock")

    adapter.send.side_effect = controlled_send
    async def call(started=None):
        if started is not None:
            started.set()
        return await adapter.send_or_update_status(
            "555",
            "run-1",
            "Final",
            metadata={"thread_id": "777", "status_terminal": True},
        )
    key = ("555", "777", "run-1")

    first = asyncio.create_task(call())
    await first_entered.wait()
    retained_lock = adapter._status_message_locks[key]
    second = asyncio.create_task(call(second_started))
    await second_started.wait()
    release_first.set()
    await second_entered.wait()

    assert adapter._status_message_locks[key] is retained_lock
    third = asyncio.create_task(call(third_started))
    await third_started.wait()
    assert send_count == 2
    assert third.done() is False

    release_second.set()
    first_result, second_result, third_result = await asyncio.gather(
        first,
        second,
        third,
    )
    assert first_result.success is False
    assert second_result.message_id == third_result.message_id == "7002"
    assert send_count == 2


@pytest.mark.asyncio
async def test_stream_edit_and_base_final_share_one_real_discord_message(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._build_operator_card_embed",
        lambda card: {"kind": card.card_type, "summary": card.summary},
    )
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    retained = SimpleNamespace(id=7001, edit=AsyncMock())
    thread = SimpleNamespace(
        send=AsyncMock(return_value=retained),
        fetch_message=AsyncMock(return_value=retained),
    )
    instance._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 777 else None,
        fetch_channel=AsyncMock(),
    )

    running = await instance.send(
        "555",
        "Checking files",
        metadata={
            "thread_id": "777",
            "status_key": "task_run",
            "non_conversational": True,
        },
    )
    assert "7001" in instance._nonconversational_messages
    stream_final = await instance.edit_message(
        "555",
        "7001",
        "Complete final result",
        metadata={
            "thread_id": "777",
            "status_key": "task_run",
            "status_terminal": True,
            "expect_edits": True,
            "notify": True,
        },
    )
    base_final = await instance.send(
        "555",
        "Complete final result",
        metadata={
            "thread_id": "777",
            "status_key": "task_run",
            "status_terminal": True,
            "notify": True,
        },
    )

    assert (
        running.message_id
        == stream_final.message_id
        == base_final.message_id
        == "7001"
    )
    assert thread.send.await_count == 1
    assert "7001" not in instance._nonconversational_messages
    assert instance._last_self_message_id["777"] == "7001"
    assert thread.send.await_args.kwargs["embed"] == {
        "kind": "task_run",
        "summary": "Checking files",
    }
    thread.fetch_message.assert_awaited_once_with(7001)
    retained.edit.assert_awaited_once_with(
        content="Complete final result",
        embed=None,
    )


@pytest.mark.asyncio
async def test_gateway_style_running_edit_refreshes_embed_in_routed_thread(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._build_operator_card_embed",
        lambda card: {"kind": card.card_type, "summary": card.summary},
    )
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    retained = SimpleNamespace(id=7001, edit=AsyncMock())
    thread = SimpleNamespace(
        send=AsyncMock(return_value=retained),
        fetch_message=AsyncMock(return_value=retained),
    )
    instance._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 777 else None,
        fetch_channel=AsyncMock(),
    )

    await instance.send(
        "555",
        "Starting",
        metadata={"thread_id": "777", "status_key": "task_run"},
    )
    result = await instance.edit_message(
        "555",
        "7001",
        "Still working",
        metadata={"thread_id": "777", "status_key": "task_run"},
    )

    assert result.success is True
    retained.edit.assert_awaited_once()
    assert retained.edit.await_args.kwargs["embed"] == {
        "kind": "task_run",
        "summary": "Still working",
    }
    assert retained.edit.await_args.kwargs["content"].startswith(
        "🔵 **Info — Task running**"
    )


@pytest.mark.asyncio
async def test_keyed_consumer_fresh_oversized_final_preserves_all_chunks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = []

    async def send_message(**kwargs):
        message = SimpleNamespace(id=7001 + len(sent), edit=AsyncMock())
        sent.append((kwargs, message))
        return message

    thread = SimpleNamespace(
        id=777,
        send=AsyncMock(side_effect=send_message),
        fetch_message=AsyncMock(),
    )
    instance._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 777 else None,
        fetch_channel=AsyncMock(),
    )
    final_text = "A" * 2200 + "B" * 2200
    consumer = GatewayStreamConsumer(
        adapter=instance,
        chat_id="555",
        config=StreamConsumerConfig(cursor="", buffer_only=True),
        metadata={"thread_id": "777", "status_key": "task_run:message:42"},
    )
    consumer.on_delta(final_text)
    consumer.finish()

    await consumer.run()

    assert consumer.final_response_sent is True
    assert len(sent) >= 2
    assert sent[0][0]["content"].startswith("A" * 100)
    assert "B" * 100 in sent[-1][0]["content"]
    assert all(message.edit.await_count == 0 for _kwargs, message in sent)


@pytest.mark.asyncio
async def test_keyed_consumer_fallback_oversized_final_preserves_all_chunks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._build_operator_card_embed",
        lambda card: {"kind": card.card_type},
    )
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = []

    async def send_message(**kwargs):
        message = SimpleNamespace(id=7001 + len(sent), edit=AsyncMock())
        sent.append((kwargs, message))
        return message

    thread = SimpleNamespace(
        id=777,
        send=AsyncMock(side_effect=send_message),
        fetch_message=AsyncMock(),
    )
    instance._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 777 else None,
        fetch_channel=AsyncMock(),
    )
    running = await instance.send(
        "555",
        "Working",
        metadata={
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "non_conversational": True,
        },
    )
    retained = sent[0][1]
    thread.fetch_message.return_value = retained
    final_text = "A" * 2200 + "B" * 2200
    consumer = GatewayStreamConsumer(
        adapter=instance,
        chat_id="555",
        config=StreamConsumerConfig(cursor=""),
        metadata={"thread_id": "777", "status_key": "task_run:message:42"},
    )
    consumer._message_id = running.message_id
    consumer._fallback_final_send = True

    await consumer._send_fallback_final(final_text)

    assert consumer.final_response_sent is True
    assert retained.edit.await_count == 1
    assert retained.edit.await_args.kwargs["content"].startswith("A" * 100)
    assert len(sent) >= 2
    assert "B" * 100 in sent[-1][0]["content"]
    visible_ids = [str(message.id) for _kwargs, message in sent]
    assert all(
        message_id not in instance._nonconversational_messages
        for message_id in visible_ids
    )
    assert instance._last_self_message_id["777"] == visible_ids[-1]


@pytest.mark.asyncio
async def test_keyed_oversized_terminal_continuation_failure_falls_back_fresh_once(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._build_operator_card_embed",
        lambda card: {"kind": card.card_type},
    )
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    retained = SimpleNamespace(
        id=7001,
        edit=AsyncMock(),
        to_reference=lambda **_kwargs: object(),
    )
    send_calls = []

    async def send_message(**kwargs):
        send_calls.append(kwargs)
        call_number = len(send_calls)
        if call_number == 1:
            return retained
        if call_number == 2:
            return SimpleNamespace(
                id=7002,
                to_reference=lambda **_kwargs: object(),
            )
        if call_number in {3, 4}:
            raise RuntimeError("continuation send failed")
        return SimpleNamespace(
            id=8000 + call_number,
            to_reference=lambda **_kwargs: object(),
        )

    thread = SimpleNamespace(
        id=777,
        send=AsyncMock(side_effect=send_message),
        fetch_message=AsyncMock(return_value=retained),
    )
    instance._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 777 else None,
        fetch_channel=AsyncMock(),
    )

    await instance.send(
        "555",
        "Working",
        metadata={"thread_id": "777", "status_key": "task_run:message:42"},
    )
    final_text = "Complete final result " * 300
    result = await instance.send(
        "555",
        final_text,
        metadata={
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "status_terminal": True,
        },
    )

    fallback_ids = result.raw_response["message_ids"]
    assert result.success is True
    assert result.message_id == fallback_ids[0]
    assert fallback_ids[0].startswith("800")
    assert len(send_calls) == 4 + len(fallback_ids)
    assert retained.edit.await_count == 1
    key = ("555", "777", "task_run:message:42")
    assert instance._status_message_ids[key] == fallback_ids[0]
    assert instance._status_message_terminal[key] is True
    assert instance._last_self_message_id["777"] == fallback_ids[-1]


@pytest.mark.asyncio
async def test_oversized_streamed_terminal_transform_removes_prior_chunks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._build_operator_card_embed",
        lambda card: {"kind": card.card_type},
    )
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    retained = SimpleNamespace(
        id=7001,
        edit=AsyncMock(),
        delete=AsyncMock(),
        to_reference=lambda **_kwargs: object(),
    )
    messages = {7001: retained}
    next_message_id = 7002

    async def send_message(**_kwargs):
        nonlocal next_message_id
        message = SimpleNamespace(
            id=next_message_id,
            edit=AsyncMock(),
            delete=AsyncMock(),
            to_reference=lambda **_kwargs: object(),
        )
        messages[next_message_id] = message
        next_message_id += 1
        return message

    async def fetch_message(message_id):
        return messages[int(message_id)]

    thread = SimpleNamespace(
        id=777,
        send=AsyncMock(side_effect=send_message),
        fetch_message=AsyncMock(side_effect=fetch_message),
    )
    instance._client = SimpleNamespace(
        get_channel=lambda channel_id: thread if channel_id == 777 else None,
        fetch_channel=AsyncMock(),
    )

    # Seed the retained running card without consuming the continuation ID
    # allocator used by the oversized terminal edit below.
    instance._status_message_ids[("555", "777", "task_run:message:42")] = "7001"
    instance._status_message_groups[("555", "777", "task_run:message:42")] = (
        "7001",
    )
    raw_final = "Raw streamed answer " * 300
    first_terminal = await instance.send(
        "555",
        raw_final,
        metadata={
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "status_terminal": True,
        },
    )
    key = ("555", "777", "task_run:message:42")
    raw_group = instance._status_message_groups[key]

    assert first_terminal.success is True
    assert len(raw_group) > 1
    assert raw_group[0] == "7001"
    assert instance._status_message_ids[key] == "7001"

    transformed = "Plugin-transformed complete answer"
    transformed_terminal = await instance.send(
        "555",
        transformed,
        metadata={
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "status_terminal": True,
        },
    )

    assert transformed_terminal.success is True
    assert instance._status_message_ids[key] == "7001"
    assert instance._status_message_groups[key] == ("7001",)
    assert retained.edit.await_args.kwargs == {
        "content": transformed,
        "embed": None,
    }
    for stale_id in raw_group[1:]:
        messages[int(stale_id)].delete.assert_awaited_once()
    retained.delete.assert_not_awaited()
    assert instance._last_self_message_id["777"] == "7001"
