"""Matrix reply references and thread fallbacks."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import _profile_runtime_scope
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "first", "all"])
@pytest.mark.parametrize("thread", [None, "root", "known", "latest"])
@pytest.mark.parametrize("delivery", ["buffered", "live", "fallback"])
async def test_streamed_response_reply_policy_spans_consumer_chunks(mode, thread, delivery):
    room_id = "!room:example.org"
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test", reply_to_mode=mode,
        extra={
            "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
            "auto_thread": False, "max_message_length": 1000,
        },
    ))
    client = MagicMock()
    client.send_message_event = AsyncMock(side_effect=[f"$sent{i}" for i in range(20)])
    adapter._client = client
    metadata = {"thread_id": "$root"} if thread else {}
    if thread == "known":
        metadata["matrix_thread_fallback_event_id"] = "$known"
    if thread == "latest":
        adapter._thread_fallbacks.remember(room_id, "$root", "$latest")
    preview_sent = asyncio.Event()
    consumer = GatewayStreamConsumer(
        adapter, room_id, StreamConsumerConfig(edit_interval=0, cursor=""),
        metadata=metadata, initial_reply_to_id="$request", on_new_message=preview_sent.set,
    )
    task = asyncio.create_task(consumer.run())
    try:
        if delivery != "buffered":
            consumer.on_delta("preview " * 25)
            await asyncio.wait_for(preview_sent.wait(), timeout=5)
        if delivery == "fallback":
            adapter.edit_message = AsyncMock(return_value=SendResult(success=False, error="edit refused"))
        consumer.on_delta("answer " * 350)
        consumer.finish()
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
            await task

    messages = [(f"$sent{index}", call.args[2])
                for index, call in enumerate(client.send_message_event.await_args_list)
                if call.args[2].get("m.relates_to", {}).get("rel_type") != "m.replace"]
    assert len(messages) >= 3
    expected = []
    for index in range(len(messages)):
        rich_reply = mode == "all" or (mode == "first" and index == 0)
        relation = {"m.in_reply_to": {"event_id": "$request"}} if rich_reply else None
        if thread:
            fallback = "$known" if thread == "known" else messages[index - 1][0]
            if index == 0:
                fallback = {"root": "$root", "known": "$known", "latest": "$latest"}[thread]
            relation = {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request" if rich_reply else fallback},
                "is_falling_back": not rich_reply,
            }
        expected.append(relation)
    assert [message.get("m.relates_to") for _, message in messages] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "first", "all"])
@pytest.mark.parametrize("thread_id", [None, "$root"])
async def test_completed_commentary_reconciles_final_with_matrix_reply_policy(monkeypatch, mode, thread_id):
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", AsyncMock())
    room_id = "!room:example.org"
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test", reply_to_mode=mode,
        extra={
            "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
            "auto_thread": False, "max_message_length": 1000,
        },
    ))
    client = MagicMock()
    client.send_message_event = AsyncMock(side_effect=[f"$sent{i}" for i in range(3)])
    adapter._client = client
    metadata = {"thread_id": thread_id} if thread_id else {}
    final_text = "completed answer " * 150
    consumer = GatewayStreamConsumer(
        adapter, room_id, StreamConsumerConfig(cursor=""),
        metadata=metadata, initial_reply_to_id="$request",
    )
    consumer.on_commentary(final_text)
    consumer.finish(final_text)
    await consumer.run()

    source = SessionSource(platform=Platform.MATRIX, chat_id=room_id, thread_id=thread_id)
    turn_ctx = TurnContext(source=source, session_key="matrix-turn", stream_consumer_holder=[consumer])
    response = {"final_response": final_text}
    runner = GatewayTurnMixin()
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._deliver_media_from_response = AsyncMock()
    await runner._run_agent_mark_streamed_delivery(response, turn_ctx)
    event = SimpleNamespace()
    outgoing = await runner._hmwa_deliver_turn_response(
        event, source, None, turn_ctx.session_key, 1, response, [], final_text, "", False,
    )

    expected = []
    for index in range(3):
        rich_reply = mode == "all" or (mode == "first" and index == 0)
        relation = {"m.in_reply_to": {"event_id": "$request"}} if rich_reply else None
        if thread_id:
            relation = {
                "rel_type": "m.thread", "event_id": thread_id,
                "m.in_reply_to": {"event_id": "$request" if rich_reply else (
                    thread_id if index == 0 else f"$sent{index - 1}"
                )},
                "is_falling_back": not rich_reply,
            }
        expected.append(relation)

    assert {
        "response": response, "outgoing": outgoing, "delivered": event._streamed_final_response,
        "relations": [call.args[2].get("m.relates_to") for call in client.send_message_event.await_args_list],
        "metadata": consumer.metadata,
    } == {
        "response": {"final_response": final_text, "already_sent": True},
        "outgoing": None, "delivered": final_text, "relations": expected, "metadata": metadata,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "first", "all"])
@pytest.mark.parametrize("thread_id", [None, "$root"])
async def test_interim_commentary_quotes_the_request_once_per_turn_in_first_mode(monkeypatch, mode, thread_id):
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", AsyncMock())
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test", reply_to_mode=mode,
        extra={
            "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
            "auto_thread": False, "e2ee_mode": "off",
        },
    ))
    client = MagicMock()
    client.send_message_event = AsyncMock(side_effect=[f"$sent{i}" for i in range(10)])
    adapter._client = client
    metadata = {"thread_id": thread_id} if thread_id else {}
    consumer = GatewayStreamConsumer(
        adapter, "!room:example.org", StreamConsumerConfig(cursor=""),
        metadata=metadata, initial_reply_to_id="$request",
    )
    consumer.on_commentary("I'll check the logs first.")
    consumer.on_commentary("Now I'll read the config.")
    consumer.on_delta("The config sets the wrong port.")
    consumer.finish("The config sets the wrong port.")
    await consumer.run()

    expected = []
    for index in range(3):
        rich_reply = mode == "all" or (mode == "first" and index == 0)
        relation = {"m.in_reply_to": {"event_id": "$request"}} if rich_reply else None
        if thread_id:
            relation = {
                "rel_type": "m.thread", "event_id": thread_id,
                "m.in_reply_to": {"event_id": "$request" if rich_reply else (
                    thread_id if index == 0 else f"$sent{index - 1}"
                )},
                "is_falling_back": not rich_reply,
            }
        expected.append(relation)

    assert [
        call.args[2].get("m.relates_to") for call in client.send_message_event.await_args_list
        if call.args[2].get("m.relates_to", {}).get("rel_type") != "m.replace"
    ] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "first", "all"])
@pytest.mark.parametrize("thread_id", [None, "$root"])
async def test_segment_tail_retry_after_failed_first_send_preserves_matrix_reply_policy(monkeypatch, mode, thread_id):
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", AsyncMock())
    room_id = "!room:example.org"
    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test", reply_to_mode=mode,
        extra={
            "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
            "auto_thread": False, "e2ee_mode": "off",
        },
    ))
    client = MagicMock()
    client.send_message_event = AsyncMock(side_effect=[RuntimeError("send refused"), "$tail", "$final"])
    adapter._client = client
    metadata = {"thread_id": thread_id} if thread_id else {}
    consumer = GatewayStreamConsumer(
        adapter, room_id, StreamConsumerConfig(cursor=""),
        metadata=metadata, initial_reply_to_id="$request",
    )
    tail = "Completed text before the tool boundary."
    final_text = "The final answer after the tool boundary."
    consumer.on_delta(tail)
    consumer.on_segment_break()
    consumer.on_delta(final_text)
    consumer.finish(final_text)
    await consumer.run()

    expected = []
    for body, fallback in [(tail, thread_id), (tail, thread_id), (final_text, "$tail")]:
        rich_reply = mode == "all" or (mode == "first" and body == tail)
        relation = {"m.in_reply_to": {"event_id": "$request"}} if rich_reply else None
        if thread_id:
            relation = {
                "rel_type": "m.thread", "event_id": thread_id,
                "m.in_reply_to": {"event_id": "$request" if rich_reply else fallback},
                "is_falling_back": not rich_reply,
            }
        expected.append({"body": body, "relation": relation})

    assert [{"body": call.args[2]["body"], "relation": call.args[2].get("m.relates_to")}
            for call in client.send_message_event.await_args_list] == expected


@pytest.mark.asyncio
async def test_reply_modes_control_split_chunks_without_losing_thread_relations():
    room_id = "!room:example.org"
    expected = {
        "off": [None, None, None],
        "first": [
            {"m.in_reply_to": {"event_id": "$request"}},
            None,
            None,
        ],
        "all": [
            {"m.in_reply_to": {"event_id": "$request"}},
            {"m.in_reply_to": {"event_id": "$request"}},
            {"m.in_reply_to": {"event_id": "$request"}},
        ],
    }
    expected_thread = {
        "off": [
            {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$known"}, "is_falling_back": True,
            },
        ] * 3,
        "first": [
            {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request"}, "is_falling_back": False,
            },
            *[
                {
                    "rel_type": "m.thread", "event_id": "$root",
                    "m.in_reply_to": {"event_id": "$known"}, "is_falling_back": True,
                },
            ] * 2,
        ],
        "all": [
            {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$request"}, "is_falling_back": False,
            },
        ] * 3,
    }

    for mode in ("off", "first", "all"):
        config = PlatformConfig(
            enabled=True, token="syt_test", reply_to_mode=mode,
            extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
        )
        adapter = MatrixAdapter(config)
        client = MagicMock()
        client.send_message_event = AsyncMock(return_value="$sent")
        adapter._client = client

        with patch.object(adapter, "truncate_message", return_value=["one", "two", "three"]):
            await adapter.send(room_id, "split answer", reply_to="$request")
            plain = [call.args[2].get("m.relates_to") for call in client.send_message_event.await_args_list]

            client.send_message_event.reset_mock()
            await adapter.send(
                room_id, "split thread answer", reply_to="$request",
                metadata={"thread_id": "$root", "matrix_thread_fallback_event_id": "$known"},
            )
            threaded = [call.args[2]["m.relates_to"] for call in client.send_message_event.await_args_list]

        assert (plain, threaded) == (expected[mode], expected_thread[mode])


def test_off_mode_keeps_a_valid_fallback_on_threaded_media():
    config = PlatformConfig(
        enabled=True, token="syt_test", reply_to_mode="off",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"},
    )
    adapter = MatrixAdapter(config)
    content = {"msgtype": "m.image", "body": "photo"}

    adapter._apply_relation_metadata(
        "!room:example.org", content,
        reply_to="$specific", metadata={"thread_id": "$root"},
    )

    assert (adapter._reply_to_mode, content) == (
        "off",
        {
            "msgtype": "m.image", "body": "photo",
            "m.relates_to": {
                "rel_type": "m.thread", "event_id": "$root",
                "m.in_reply_to": {"event_id": "$root"}, "is_falling_back": True,
            },
        },
    )


def test_reply_mode_follows_each_profile_config_in_multiplex(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    secondary = root / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (root / "config.yaml").write_text("matrix:\n  enabled: true\n  reply_to_mode: all\n")
    (secondary / "config.yaml").write_text("matrix:\n  enabled: true\n  reply_to_mode: false\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    set_multiplex_active(True)
    try:
        from gateway.config import load_gateway_config

        def current_mode():
            config = load_gateway_config().platforms[Platform.MATRIX]
            return config.reply_to_mode, MatrixAdapter(config)._reply_to_mode

        default_before = current_mode()
        with _profile_runtime_scope(secondary, prepared_secret_scope={}):
            served = current_mode()
        default_after = current_mode()
    finally:
        set_multiplex_active(False)

    assert (default_before, served, default_after) == (
        ("all", "all"), ("off", "off"), ("all", "all"),
    )


@pytest.mark.parametrize(("env_override", "expected"), [(None, "off"), ("all", "all")])
def test_reply_mode_uses_highest_priority_yaml_block(monkeypatch, tmp_path, env_override, expected):
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(
        "gateway:\n  platforms:\n    matrix:\n      reply_to_mode: all\n"
        "platforms:\n  matrix:\n    reply_to_mode: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if env_override is None:
        monkeypatch.delenv("MATRIX_REPLY_TO_MODE", raising=False)
    else:
        monkeypatch.setenv("MATRIX_REPLY_TO_MODE", env_override)

    config = load_gateway_config().platforms[Platform.MATRIX]

    assert (config.reply_to_mode, MatrixAdapter(config)._reply_to_mode) == (expected, expected)
