"""Relayed Discord messages and interactions must share one prompt/session identity."""

import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import ProcessingOutcome
from gateway.relay.ws_transport import _event_from_wire
from gateway.session import SessionStore, build_session_context, build_session_key
from tests.gateway.relay.test_relay_interactive import _adapter


def _forward(**payload):
    body = {
        "type": 2,
        "id": "i1",
        "channel_id": "ch1",
        "guild_id": "g1",
        "data": {"name": "status"},
    }
    body.update(payload)

    class Forward:
        platform = "discord"
        method = "POST"
        path = "/interactions/bot1"

    Forward.body = json.dumps(body).encode()
    return Forward()


def _message(
    *, chat_name="Hermes / #ops", chat_topic="triage", thread=False,
    user_id="u1", user_name="ben", user_display_name="Ben D",
):
    chat = (
        {"chat_id": "th1", "chat_type": "thread", "thread_id": "th1", "parent_chat_id": "ch1"}
        if thread else {"chat_id": "ch1", "chat_type": "group"}
    )
    return _event_from_wire({
        "text": "hello",
        "message_type": "text",
        "source": {
            "platform": "discord",
            **chat,
            "scope_id": "g1",
            "user_id": user_id,
            "user_name": user_name,
            "user_display_name": user_display_name,
            "chat_name": chat_name,
            "chat_topic": chat_topic,
            "message_id": "m1",
        },
    })


def _prompt_sequence(*sources):
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    return [
        runner._pinned_session_context_prompt(build_session_context(source, config), False, "same-session")
        for source in sources
    ]


async def _passthrough_event(adapter, forward):
    adapter.handle_message = AsyncMock()
    await adapter._on_passthrough(forward)
    assert adapter.handle_message.await_count == 1
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("thread", [False, True], ids=["channel", "thread"])
async def test_message_interaction_message_keeps_prompt_and_session_identity(thread):
    adapter, _stub = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    message = _message(thread=thread)
    await adapter._on_inbound(message)

    interaction = {
        "member": {
            "nick": "Ben D",
            "user": {"id": "u1", "username": "ben", "global_name": "Ben D"},
        },
    }
    if thread:
        interaction.update(
            channel_id="th1",
            channel={"id": "th1", "type": 11, "parent_id": "ch1"},
        )
    slash = adapter._discord_interaction_to_event(_forward(**interaction))

    assert slash is not None
    assert build_session_key(slash.source) == build_session_key(message.source)
    assert (slash.source.chat_name, slash.source.chat_topic, slash.source.user_name) == (
        message.source.chat_name,
        message.source.chat_topic,
        message.source.user_name,
    )
    assert len(set(_prompt_sequence(message.source, slash.source, message.source))) == 1






@pytest.mark.asyncio
@pytest.mark.parametrize("modern", [False, True], ids=["legacy-action-row", "label-component"])
async def test_modal_submission_retains_unicode_and_empty_values(modern):
    adapter, _ = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    if modern:
        components = [
            {"type": 18, "id": 1, "component": {
                "type": 4, "id": 2, "custom_id": "answer", "value": "café 東京",
            }},
            {"type": 18, "id": 3, "component": {
                "type": 4, "id": 4, "custom_id": "optional", "value": "",
            }},
        ]
    else:
        components = [{
            "type": 1,
            "components": [
                {"type": 4, "custom_id": "answer", "value": "café 東京"},
                {"type": 4, "custom_id": "optional", "value": ""},
            ],
        }]
    forward = _forward(
        type=5,
        id=f"modal-{\'modern\' if modern else \'legacy\'}",
        member={"user": {"id": "u1", "username": "ben"}},
        data={"custom_id": "profile-form", "components": components},
    )

    await adapter._on_passthrough(forward)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "answer=café 東京\noptional="
    assert event.metadata["discord_modal_fields"] == [
        {"custom_id": "answer", "value": "café 東京"},
        {"custom_id": "optional", "value": ""},
    ]
    assert event.raw_message["data"]["components"] == components
    assert event.message_id.startswith("modal-")

@pytest.mark.asyncio
async def test_buffered_passthrough_replay_is_admitted_once_and_acked_each_time@pytest.mark.asyncio
async def test_buffered_passthrough_replay_is_admitted_once_and_acked_each_time():
    adapter, stub = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    forward = _forward(type=3, id="press-buffer-1", message={"id": "bot-buffer-1"},
                       member={"user": {"id": "u1", "username": "ben"}},
                       data={"custom_id": "inspect"})
    await adapter._on_passthrough(forward, "buffer-1")
    await adapter._on_passthrough(forward, "buffer-1")
    assert adapter.handle_message.await_count == 1
    assert stub.acked_buffer_ids == ["buffer-1", "buffer-1"]


@pytest.mark.asyncio
async def test_buffered_passthrough_cancel_before_admission_remains_retryable():
    adapter, stub = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    entered, release = asyncio.Event(), asyncio.Event()
    original = adapter._discord_context_for

    async def held(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    adapter._discord_context_for = held
    forward = _forward(type=3, id="press-buffer-cancel", message={"id": "bot-buffer-cancel"},
                       member={"user": {"id": "u1", "username": "ben"}},
                       data={"custom_id": "inspect"})
    task = asyncio.create_task(adapter._on_passthrough(forward, "buffer-cancel"))
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.handle_message.await_count == 0
    assert stub.acked_buffer_ids == []
    assert "passthrough_buffer:buffer-cancel" not in adapter._seen_inbound

    release.set()
    adapter._discord_context_for = original
    await adapter._on_passthrough(forward, "buffer-cancel")
    assert adapter.handle_message.await_count == 1
    assert stub.acked_buffer_ids == ["buffer-cancel"]


@pytest.mark.asyncio
async def test_buffered_passthrough_replay_finishes_ack_after_post_admission_failure():
    adapter, stub = _adapter(platform="discord")
    calls = 0

    async def failing(_event):
        nonlocal calls
        calls += 1
        raise RuntimeError("after admission")

    adapter.handle_message = failing
    forward = _forward(type=3, id="press-buffer-fail", message={"id": "bot-buffer-fail"},
                       member={"user": {"id": "u1", "username": "ben"}},
                       data={"custom_id": "inspect"})
    await adapter._on_passthrough(forward, "buffer-fail")
    assert calls == 1
    assert stub.acked_buffer_ids == []
    await adapter._on_passthrough(forward, "buffer-fail")
    assert calls == 1
    assert stub.acked_buffer_ids == ["buffer-fail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tools_enabled", [False, True], ids=["tools-off", "tools-on"])
async def test_model_reaching_slash_keeps_discord_prompt_presence_stable@pytest.mark.asyncio
@pytest.mark.parametrize("tools_enabled", [False, True], ids=["tools-off", "tools-on"])
async def test_model_reaching_slash_keeps_discord_prompt_presence_stable(monkeypatch, tools_enabled):
    """A truthful slash-without-message anchor must not flip cached Discord system guidance."""
    import gateway.session as gateway_session

    adapter, _ = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    message = _message()
    await adapter._on_inbound(message)

    member = {
        "nick": "Ben D",
        "user": {"id": "u1", "username": "ben", "global_name": "Ben D"},
    }
    slash = adapter._discord_interaction_to_event(_forward(
        type=2,
        id="slash-steer-1",
        member=member,
        data={"name": "steer", "options": [{"name": "text", "type": 3, "value": "inspect"}]},
    ))
    component = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-prompt-1",
        message={"id": "bot-message-prompt-1"},
        member=member,
        data={"custom_id": "inspect"},
    ))
    assert slash is not None and component is not None
    assert slash.source.message_id is None
    assert component.source.message_id == "bot-message-prompt-1"

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    runner.adapters = {}
    dispatched = await runner._hm_cmd_steer(
        slash, slash.source, build_session_key(slash.source)
    )
    assert dispatched == (False, None)
    assert slash.text == "inspect"

    monkeypatch.setattr(gateway_session, "_discord_tools_loaded", lambda: tools_enabled)
    prompts = _prompt_sequence(message.source, slash.source, component.source, message.source)
    assert len(set(prompts)) == 1


@pytest.mark.asyncio
async def test_discord_interaction_processing_reactions_use_message_anchor():
    """Processing reactions target the attached message, never the unique action id."""
    adapter, stub = _adapter(platform="discord")
    member = {"user": {"id": "u1", "username": "ben"}}

    component = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-react-1",
        message={"id": "bot-message-react-1"},
        member=member,
        data={"custom_id": "inspect"},
    ))
    assert component is not None
    await adapter.on_processing_start(component)
    await adapter.on_processing_complete(component, ProcessingOutcome.SUCCESS)
    assert component.message_id == "press-react-1"
    assert [
        item.get("message_id") for item in stub.sent if item.get("op") == "react"
    ] == ["bot-message-react-1"] * 3

    stub.sent.clear()
    slash = adapter._discord_interaction_to_event(_forward(
        type=2,
        id="slash-react-1",
        member=member,
        data={"name": "status"},
    ))
    assert slash is not None and slash.source.message_id is None
    await adapter.on_processing_start(slash)
    await adapter.on_processing_complete(slash, ProcessingOutcome.SUCCESS)
    assert slash.message_id == "slash-react-1"
    assert not [item for item in stub.sent if item.get("op") == "react"]


@pytest.mark.asyncio
async def test_interaction_triggering_note_round_trip_persists_authored_text(tmp_path, monkeypatch):
    """Producer and inverse share the attached-message anchor; action id remains persistence identity."""
    import gateway.session as gateway_session

    monkeypatch.setattr(gateway_session, "_discord_tools_loaded", lambda: True)
    adapter, _ = _adapter(platform="discord")
    event = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-note-1",
        message={"id": "bot-message-note-1"},
        member={"user": {"id": "u1", "username": "ben"}},
        data={"custom_id": "please summarize"},
    ))
    assert event is not None

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""

    model_text = await runner._prepare_inbound_message_text(
        event=event, source=event.source, history=[]
    )
    message_text, persisted, _ = runner._hmwa_apply_message_timestamp(event, model_text)
    assert "Triggering message id: `bot-message-note-1`" in message_text
    assert "press-note-1" not in message_text
    assert persisted == "please summarize"

    store = SessionStore(tmp_path, runner.config)
    entry = store.get_or_create_session(event.source)
    store.append_to_transcript(entry.session_id, {
        "role": "user",
        "content": persisted,
        "platform_message_id": event.message_id,
    })
    row = store.load_transcript(entry.session_id)[0]
    assert row["content"] == "please summarize"
    assert row["platform_message_id"] == "press-note-1"
    store.close_all_db_handles()

@pytest.mark.asyncio
async def test_channel_rename_survives_restart_with_latest_text_lane_labels(tmp_path):
    """A reused session's creation-time origin must not win after newer labels were observed."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    old = _message(chat_name="Hermes / #ops-old", chat_topic="old topic")
    store.get_or_create_session(old.source)

    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    await adapter._on_inbound(old)

    renamed = _message(chat_name="Hermes / #ops", chat_topic="new topic")
    await adapter._on_inbound(renamed)
    store.close_all_db_handles()

    restarted_store = SessionStore(tmp_path, config)
    restarted, _stub = _adapter(platform="discord")
    restarted.set_session_store(restarted_store)
    slash = await _passthrough_event(restarted, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))

    assert (slash.source.chat_name, slash.source.chat_topic) == (
        renamed.source.chat_name,
        renamed.source.chat_topic,
    )
    assert len(set(_prompt_sequence(renamed.source, slash.source, renamed.source))) == 1

    # /new/reset replaces the entry and drops generic metadata, but inherits the refreshed origin.
    key = build_session_key(renamed.source)
    reset_entry = restarted_store.reset_session(key)
    assert reset_entry is not None
    restarted_store.close_all_db_handles()
    after_reset = SessionStore(tmp_path, config)
    reset_adapter, _stub = _adapter(platform="discord")
    reset_adapter.set_session_store(after_reset)
    slash_after_reset = await _passthrough_event(reset_adapter, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))
    assert (slash_after_reset.source.chat_name, slash_after_reset.source.chat_topic) == (
        renamed.source.chat_name,
        renamed.source.chat_topic,
    )


@pytest.mark.asyncio
async def test_latest_channel_labels_win_across_per_user_sessions_after_restart(tmp_path):
    """One user's rename observation must supersede another user's older session origin."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    alice = _message(chat_name="Hermes / #before", user_id="u1", user_display_name="Alice")
    bob = _message(chat_name="Hermes / #before", user_id="u2", user_name="bob", user_display_name="Bob")
    store.get_or_create_session(alice.source)
    store.get_or_create_session(bob.source)

    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    await adapter._on_inbound(alice)
    await adapter._on_inbound(bob)

    renamed = _message(
        chat_name="Hermes / #after", chat_topic="renamed",
        user_id="u1", user_display_name="Alice",
    )
    await adapter._on_inbound(renamed)
    store.close_all_db_handles()

    restarted_store = SessionStore(tmp_path, config)
    restarted, _stub = _adapter(platform="discord")
    restarted.set_session_store(restarted_store)
    slash = await _passthrough_event(restarted, _forward(
        member={"user": {"id": "u2", "username": "bob", "global_name": "Bob"}},
    ))

    assert (slash.source.chat_name, slash.source.chat_topic) == (
        renamed.source.chat_name, renamed.source.chat_topic,
    )


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ({"nick": "Benny", "user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Benny"),
        ({"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Ben D"),
        ({"user": {"id": "u1", "username": "ben"}}, "ben"),
    ],
)
def test_cold_interaction_uses_discord_display_name_order(member, expected):
    adapter, _stub = _adapter(platform="discord")
    event = adapter._discord_interaction_to_event(_forward(member=member))
    assert event is not None
    assert event.source.user_name == expected


@pytest.mark.asyncio
async def test_peer_reset_cannot_outvote_newer_channel_observation(tmp_path):
    """A reset inherits routing state; it must never manufacture a newer channel observation."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    alice = _message(chat_name="A", user_id="u1", user_display_name="Alice")
    bob = _message(chat_name="A", user_id="u2", user_name="bob", user_display_name="Bob")
    for event in (alice, bob):
        store.get_or_create_session(event.source)
        await adapter._on_inbound(event)

    renamed = _message(chat_name="B", user_id="u1", user_display_name="Alice")
    await adapter._on_inbound(renamed)
    store.reset_session(build_session_key(bob.source))
    store.close_all_db_handles()

    restarted_store = SessionStore(tmp_path, config)
    restarted, _stub = _adapter(platform="discord")
    restarted.set_session_store(restarted_store)
    slash = await _passthrough_event(restarted, _forward(
        member={"user": {"id": "u2", "username": "bob", "global_name": "Bob"}},
    ))
    assert slash.source.chat_name == "B"


@pytest.mark.asyncio
async def test_latest_equal_peer_value_survives_restart(tmp_path):
    """A -> B -> A must persist the final A even when that observer's older value was also A."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    for uid in ("u1", "u2"):
        event = _message(chat_name="A", user_id=uid, user_display_name=uid)
        store.get_or_create_session(event.source)
        await adapter._on_inbound(event)
    await adapter._on_inbound(_message(chat_name="B", user_id="u1", user_display_name="u1"))
    await adapter._on_inbound(_message(chat_name="A", user_id="u2", user_display_name="u2"))
    store.close_all_db_handles()

    restarted_store = SessionStore(tmp_path, config)
    restarted, _stub = _adapter(platform="discord")
    restarted.set_session_store(restarted_store)
    slash = await _passthrough_event(restarted, _forward(
        member={"user": {"id": "u2", "username": "u2", "global_name": "u2"}},
    ))
    assert slash.source.chat_name == "A"


@pytest.mark.asyncio
async def test_label_observation_never_replaces_session_origin(tmp_path):
    """The context cache is independent of routing provenance stored on SessionEntry.origin."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    original = _message(chat_name="A")
    entry = store.get_or_create_session(original.source)
    origin = entry.origin
    marker = object()
    origin._transport_adapter_ref = marker

    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    await adapter._on_inbound(_message(chat_name="B"))

    current = store._entries[entry.session_key]
    assert current.origin is origin
    assert current.origin._transport_adapter_ref is marker


@pytest.mark.asyncio
async def test_missing_nick_uses_known_name_but_explicit_null_invalidates_it(tmp_path):
    """Optional nick omission preserves known identity; explicit null proves nickname removal."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    await adapter._on_inbound(_message(user_display_name="Benny"))
    store.close_all_db_handles()

    restarted_store = SessionStore(tmp_path, config)
    restarted, _stub = _adapter(platform="discord")
    restarted.set_session_store(restarted_store)

    missing = await _passthrough_event(restarted, _forward(member={
        "user": {"id": "u1", "username": "ben", "global_name": "Ben D"},
    }))
    assert missing.source.user_name == "Benny"

    removed = await _passthrough_event(restarted, _forward(member={
        "nick": None,
        "user": {"id": "u1", "username": "ben", "global_name": "Ben D"},
    }))
    assert removed.source.user_name == "Ben D"
    restarted_store.close_all_db_handles()

    after = SessionStore(tmp_path, config)
    after_adapter, _stub = _adapter(platform="discord")
    after_adapter.set_session_store(after)
    missing_after_removal = await _passthrough_event(after_adapter, _forward(member={
        "user": {"id": "u1", "username": "ben", "global_name": "Ben D"},
    }))
    assert missing_after_removal.source.user_name == "Ben D"


@pytest.mark.asyncio
async def test_shared_store_invalidates_peer_adapter_context_cache(tmp_path):
    """A successful miss or hit in one adapter must see later observations from a peer adapter."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    reader, _ = _adapter(platform="discord")
    writer, _ = _adapter(platform="discord")
    reader.set_session_store(store)
    writer.set_session_store(store)
    writer.handle_message = AsyncMock()

    first = await _passthrough_event(reader, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))
    assert first.source.chat_name is None

    await writer._on_inbound(_message(chat_name="A"))
    second = await _passthrough_event(reader, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))
    assert second.source.chat_name == "A"

    await writer._on_inbound(_message(chat_name="B"))
    third = await _passthrough_event(reader, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))
    assert third.source.chat_name == "B"


@pytest.mark.asyncio
async def test_cold_context_read_runs_off_event_loop(tmp_path):
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    seed, _ = _adapter(platform="discord")
    seed.set_session_store(store)
    seed.handle_message = AsyncMock()
    await seed._on_inbound(_message(chat_name="A"))
    store.close_all_db_handles()

    cold_store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(cold_store)
    loop_thread = threading.get_ident()
    read_threads = []
    original = cold_store.relay_discord_context

    def recording_read(*args):
        read_threads.append(threading.get_ident())
        return original(*args)

    cold_store.relay_discord_context = recording_read
    event = await _passthrough_event(adapter, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))
    assert event.source.chat_name == "A"
    assert read_threads and all(thread_id != loop_thread for thread_id in read_threads)


@pytest.mark.asyncio
async def test_partial_thread_channel_recovers_known_parent(tmp_path):
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    message = _message(thread=True)
    await adapter._on_inbound(message)

    interaction = await _passthrough_event(adapter, _forward(
        channel_id="th1",
        channel={"id": "th1", "type": 11},
        member={"nick": "Ben D", "user": {"id": "u1", "username": "ben"}},
    ))
    assert interaction.source.parent_chat_id == "ch1"
    assert build_session_key(interaction.source) == build_session_key(message.source)


def test_message_identity_is_normalized_at_each_ingress_boundary():
    text = _event_from_wire({
        "text": "hello",
        "message_type": "text",
        "message_id": "platform-message-1",
        "source": {
            "platform": "discord",
            "chat_id": "ch1",
            "chat_type": "group",
            "scope_id": "g1",
            "user_id": "u1",
        },
    })
    assert text.message_id == "platform-message-1"
    assert text.source.message_id == "platform-message-1"

    adapter, _ = _adapter(platform="discord")
    component = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="interaction-1",
        message={"id": "actual-message-77"},
        member={"user": {"id": "u1", "username": "ben"}},
        data={"custom_id": "foreign-button"},
    ))
    assert component is not None
    assert component.source.message_id == "actual-message-77"
    assert component.message_id == "interaction-1"
    assert component.metadata["discord_interaction_id"] == "interaction-1"
    from gateway.platforms.base import _reply_anchor_for_event
    assert _reply_anchor_for_event(component) == "actual-message-77"

    slash = adapter._discord_interaction_to_event(_forward(
        type=2,
        id="interaction-2",
        member={"user": {"id": "u1", "username": "ben"}},
        data={"name": "status"},
    ))
    assert slash is not None
    assert slash.source.message_id is None
    assert slash.message_id == "interaction-2"
    assert slash.metadata["discord_interaction_id"] == "interaction-2"
    assert _reply_anchor_for_event(slash) is None


@pytest.mark.asyncio
async def test_topic_removal_is_an_authoritative_observation(tmp_path):
    """A removed Discord topic must not revive from durable context on the next interaction."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    await adapter._on_inbound(_message(chat_topic="old topic"))
    await adapter._on_inbound(_message(chat_topic=None))
    store.close_all_db_handles()

    restarted_store = SessionStore(tmp_path, config)
    restarted, _ = _adapter(platform="discord")
    restarted.set_session_store(restarted_store)
    slash = await _passthrough_event(restarted, _forward(
        member={"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}},
    ))
    assert slash.source.chat_topic is None


@pytest.mark.asyncio
async def test_unchanged_text_context_skips_worker_round_trip(tmp_path, monkeypatch):
    """After publication, repeated identical text messages take the lock-free fast path."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    first = _message()
    await adapter._on_inbound(first)

    calls = 0
    real_to_thread = asyncio.to_thread

    async def recording_to_thread(func, *args, **kwargs):
        nonlocal calls
        if getattr(func, "__name__", "") == "observe_relay_discord_context":
            calls += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)
    repeat = _message()
    repeat.source.message_id = "m2"
    repeat.message_id = "m2"
    await adapter._on_inbound(repeat)
    assert calls == 0


def test_unavailable_context_db_stays_retryable(tmp_path, monkeypatch):
    """Unavailable reads/writes must not become authoritative cache state."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    source = _message().source
    original_method = store._routing_db_method
    blocked = {"set": True, "get": False}

    def routed_method(name):
        if name == "set_meta" and blocked["set"]:
            return None
        if name == "get_meta" and blocked["get"]:
            return None
        return original_method(name)

    monkeypatch.setattr(store, "_routing_db_method", routed_method)

    # A recoverable write outage must not publish the new value. The identical next
    # observation therefore still reaches persistence after the DB recovers.
    assert store.observe_relay_discord_context(source) is False
    assert store.cached_relay_discord_context("g1", "ch1", "u1") == {}

    blocked["set"] = False
    assert store.observe_relay_discord_context(source) is True
    cached = store.cached_relay_discord_context("g1", "ch1", "u1")
    assert cached["chat_name"] == source.chat_name
    assert cached["user_name"] == source.user_name

    # A recoverable cold-read outage is not the same thing as a durable miss either.
    store._relay_discord_context_cache_map = {}
    blocked["get"] = True
    assert store.relay_discord_context("g1", "ch1", "u1") == {}
    assert store.cached_relay_discord_context("g1", "ch1", "u1") == {}

    blocked["get"] = False
    restored = store.relay_discord_context("g1", "ch1", "u1")
    assert restored["chat_name"] == source.chat_name
    assert restored["user_name"] == source.user_name






@pytest.mark.asyncio
async def test_context_writer_runs_in_gateway_owned_executor_and_is_visible_to_shutdown(tmp_path):
    """A cancelled metadata await may outlive its coroutine, but not the gateway executor close gate."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._executor_lock = threading.Lock()
    runner._executor_closing = False
    runner._executor = None
    runner._housekeeping_executor = None
    adapter.gateway_runner = runner

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = store.observe_relay_discord_context

    def held(source):
        entered.set()
        try:
            assert release.wait(5)
            return original(source)
        finally:
            finished.set()

    store.observe_relay_discord_context = held
    task = asyncio.create_task(adapter._remember_discord_context(_message().source))
    assert await asyncio.to_thread(entered.wait, 3)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The coroutine is gone, but the lifecycle-owned worker is still visible to the exact
    # quiesce primitive used before SessionDB.close().
    assert gateway_run.GatewayRunner._shutdown_executor(runner, drain_timeout=0) == 1

    release.set()
    assert await asyncio.to_thread(finished.wait, 3)
    store.close_all_db_handles()


def test_relay_context_uses_short_observation_write_budget(tmp_path):
    """Discord label observations use the existing 0.5s activity/label patience, not generic 20s."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    db = store._routing_db
    assert db is not None

    observed = []
    original = db._write_sql

    def recording_write(sql, params=(), *, many=False, patience_s=None):
        observed.append(patience_s)
        return original(sql, params, many=many, patience_s=patience_s)

    db._write_sql = recording_write
    assert store.observe_relay_discord_context(_message().source) is True
    assert observed
    assert all(value == db._ACTIVITY_WRITE_PATIENCE_S for value in observed)
    store.close_all_db_handles()

@pytest.mark.asyncio
async def test_cancelled_context_write_keeps_inbound_replay_retryable(tmp_path):
    """Cancellation before admission must not turn a durable relay frame into a seen replay."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = store.observe_relay_discord_context

    def held(source):
        entered.set()
        try:
            assert release.wait(5)
            return original(source)
        finally:
            finished.set()

    store.observe_relay_discord_context = held
    first = _message()
    task = asyncio.create_task(adapter._on_inbound(first))
    assert await asyncio.to_thread(entered.wait, 3)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.handle_message.await_count == 0

    release.set()
    assert await asyncio.to_thread(finished.wait, 3)
    await adapter._on_inbound(_message())

    assert adapter.handle_message.await_count == 1
    key = adapter._inbound_dedupe_key(_message())
    assert key in adapter._seen_inbound
    assert key not in adapter._inflight_inbound
    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_overlapping_inbound_replays_share_one_admission_owner(tmp_path):
    """Concurrent duplicates wait on pre-admission work and only one reaches handle_message."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    entered = threading.Event()
    release = threading.Event()
    original = store.observe_relay_discord_context

    def held(source):
        entered.set()
        assert release.wait(5)
        return original(source)

    store.observe_relay_discord_context = held
    first = asyncio.create_task(adapter._on_inbound(_message()))
    assert await asyncio.to_thread(entered.wait, 3)
    duplicate = asyncio.create_task(adapter._on_inbound(_message()))
    await asyncio.sleep(0)
    assert not duplicate.done()

    release.set()
    await asyncio.gather(first, duplicate)
    assert adapter.handle_message.await_count == 1

    # A later replay is a normal seen hit and also stays suppressed.
    await adapter._on_inbound(_message())
    assert adapter.handle_message.await_count == 1
    assert not adapter._inflight_inbound
    store.close_all_db_handles()

@pytest.mark.asyncio
async def test_dm_interaction_uses_current_payload_identity(tmp_path):
    """A DM user object is current; cached text identity must not overwrite it."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    old = _event_from_wire({
        "text": "hello",
        "message_type": "text",
        "source": {
            "platform": "discord",
            "chat_id": "dm1",
            "chat_type": "dm",
            "user_id": "u1",
            "user_name": "ben",
            "user_display_name": "Old Name",
            "message_id": "dm-text-1",
        },
    })
    await adapter._on_inbound(old)

    current = await _passthrough_event(adapter, _forward(
        guild_id=None,
        channel_id="dm1",
        id="dm-interaction-1",
        member=None,
        user={"id": "u1", "username": "ben", "global_name": "New Name"},
    ))
    assert current.source.user_name == "New Name"
    assert store.cached_relay_discord_context("", "dm1", "u1")["user_name"] == "New Name"


@pytest.mark.asyncio
async def test_two_component_presses_keep_distinct_durable_turn_identity(tmp_path):
    """Two presses on one bot message share a reply anchor, never an inbound owner."""
    from agent.turn_failure_copy import FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE

    adapter, _ = _adapter(platform="discord")
    member = {"user": {"id": "u1", "username": "ben"}}
    first = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-1",
        message={"id": "bot-message-77"},
        member=member,
        data={"custom_id": "foreign-button-1"},
    ))
    second = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-2",
        message={"id": "bot-message-77"},
        member=member,
        data={"custom_id": "foreign-button-2"},
    ))
    assert first is not None and second is not None
    assert first.source.message_id == second.source.message_id == "bot-message-77"
    assert first.message_id == "press-1"
    assert second.message_id == "press-2"

    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    entry = store.get_or_create_session(first.source)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = store
    runner.config = config
    runner._session_db = None

    async def noop(*args, **kwargs):
        return None

    async def open_session(*args, **kwargs):
        return False, False

    async def keep_history(event, source, session_entry, session_key, history, *args, **kwargs):
        return history

    async def inbound_text(*, event, **kwargs):
        return event.text

    runner._hmwa_open_session = open_session
    runner._set_session_env = lambda *args, **kwargs: []
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = noop
    runner._mark_durable_active_turn = noop
    runner._hmwa_run_session_hygiene = keep_history
    runner._hmwa_first_contact_notes = noop
    runner._voice_channel_sidecar_note = lambda *args, **kwargs: None
    runner._prepare_profile_scoped_inbound_message_text = inbound_text
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, None)
    runner._delivery_adapter_for = lambda *args, **kwargs: None
    runner._bind_adapter_run_generation = lambda *args, **kwargs: None
    runner._hmwa_stop_typing_for_turn = noop
    runner._refresh_agent_cache_message_count = noop

    async def prepare(event, generation):
        prepared, _tokens = await runner._hmwa_prepare_turn(
            event, event.source, entry, entry.session_key, entry.session_key, generation,
        )
        assert isinstance(prepared, runner._PreparedTurn)
        return prepared

    first_prepared = await prepare(first, 1)
    db = store._db_for_session_id(entry.session_id)
    db.append_message(
        entry.session_id,
        "user",
        first.text,
        platform_message_id=first.message_id,
        display_metadata={"gateway_input_owner": first_prepared.persistence_owner},
    )
    db.append_message(entry.session_id, "assistant", "first reply")

    second_prepared = await prepare(second, 2)
    assert second_prepared.persistence_owner != first_prepared.persistence_owner
    before = db.message_count()
    reply = await runner._hmwa_agent_error_reply(
        RuntimeError("controlled second-press failure"),
        second,
        second.source,
        entry,
        entry.session_key,
        second_prepared,
    )
    assert db.message_count() == before + 2
    assert store.has_input_owner(entry.session_id, second_prepared.persistence_owner)
    assert store.has_platform_message_id(entry.session_id, second.message_id)
    assert PARTIAL_FAILED_TURN_NOTICE in reply
    assert db.get_messages(entry.session_id)[-1]["content"] == PARTIAL_FAILED_TURN_NOTICE

    # The transient-result writer uses platform_message_id dedupe instead of input-owner
    # dedupe. A third press on the same attached bot message must survive that path too.
    third = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-3",
        message={"id": "bot-message-77"},
        member=member,
        data={"custom_id": "foreign-button-3"},
    ))
    assert third is not None and third.message_id == "press-3"
    third_prepared = await prepare(third, 3)
    before = db.message_count()
    failed = {
        "failed": True,
        "final_response": "429",
        "error": "429",
        "messages": [],
        "history_offset": len(third_prepared.history),
        "last_prompt_tokens": 0,
        "agent_persisted": False,
    }
    await runner._hmwa_persist_turn_transcript(
        event=third,
        source=third.source,
        session_entry=entry,
        session_key=entry.session_key,
        agent_result=failed,
        agent_messages=[],
        prepared=third_prepared,
        response="rate limited",
        agent_failed_early=True,
        hidden_reasoning_incomplete=False,
        is_context_overflow_failure=False,
    )
    assert db.message_count() == before + 2
    assert store.has_platform_message_id(entry.session_id, third.message_id)
    assert db.get_messages(entry.session_id)[-1]["content"] == FAILED_TURN_NOTICE
    db.close()



def test_discord_interaction_triggering_note_uses_message_anchor(monkeypatch):
    """Tool guidance must name a real Discord message, never an interaction id."""
    import gateway.session as gateway_session

    monkeypatch.setattr(gateway_session, "_discord_tools_loaded", lambda: True)
    adapter, _ = _adapter(platform="discord")

    component = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="press-anchor-1",
        message={"id": "bot-message-88"},
        member={"user": {"id": "u1", "username": "ben"}},
        data={"custom_id": "foreign-button"},
    ))
    assert component is not None
    prepared = gateway_run.GatewayRunner._prepend_inbound_reply_context(
        component, component.source, "do it",
    )
    assert "Triggering message id: `bot-message-88`" in prepared
    assert "press-anchor-1" not in prepared

    slash = adapter._discord_interaction_to_event(_forward(
        type=2,
        id="slash-anchor-1",
        member={"user": {"id": "u1", "username": "ben"}},
        data={"name": "status"},
    ))
    assert slash is not None
    slash_prepared = gateway_run.GatewayRunner._prepend_inbound_reply_context(
        slash, slash.source, "status",
    )
    assert "Triggering message id:" not in slash_prepared



def test_discord_interaction_busy_reply_uses_attached_message_anchor():
    """Busy-path replies must not send a Discord interaction id as reply_to."""
    from gateway.platforms.base import _reply_anchor_for_event

    adapter, _ = _adapter(platform="discord")
    component = adapter._discord_interaction_to_event(_forward(
        type=3,
        id="busy-press-1",
        message={"id": "bot-message-99"},
        member={"user": {"id": "u1", "username": "ben"}},
        data={"custom_id": "foreign-button"},
    ))
    assert component is not None
    anchor = _reply_anchor_for_event(component)
    assert anchor == "bot-message-99"
    assert gateway_run.GatewayRunner._busy_reply_to(component, anchor) == "bot-message-99"

    slash = adapter._discord_interaction_to_event(_forward(
        type=2,
        id="busy-slash-1",
        member={"user": {"id": "u1", "username": "ben"}},
        data={"name": "status"},
    ))
    assert slash is not None
    assert gateway_run.GatewayRunner._busy_reply_to(
        slash, _reply_anchor_for_event(slash),
    ) is None
