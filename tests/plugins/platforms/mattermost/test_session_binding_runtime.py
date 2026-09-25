"""Mattermost binding runtime wiring stays entirely inside the plugin."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.mattermost.adapter import register
from plugins.platforms.mattermost.adapter import MattermostAdapter
from gateway.config import PlatformConfig
from plugins.platforms.mattermost.session_binding_runtime import MattermostSessionBindingRuntime
from plugins.platforms.mattermost.session_bindings import MattermostSessionBindingStore


@pytest.mark.asyncio
async def test_target_normalization_resolves_reply_to_canonical_root():
    runtime = MattermostSessionBindingRuntime()
    adapter = SimpleNamespace(
        is_connected=True,
        _api_get=AsyncMock(return_value={
            "id": "reply1", "root_id": "root1", "channel_id": "channel1"
        }),
    )
    runtime.wire_mattermost(None, adapter)

    assert await runtime.normalize_target("channel1", "reply1") == ("channel1", "root1")
    adapter._api_get.assert_awaited_once_with("posts/reply1")


@pytest.mark.asyncio
async def test_target_normalization_rejects_cross_channel_post():
    runtime = MattermostSessionBindingRuntime()
    runtime.wire_mattermost(None, SimpleNamespace(
        is_connected=True,
        _api_get=AsyncMock(return_value={
            "id": "root1", "root_id": "", "channel_id": "otherchannel"
        }),
    ))

    with pytest.raises(LookupError, match="does not belong"):
        await runtime.normalize_target("channel1", "root1")


@pytest.mark.asyncio
async def test_thread_creation_uses_existing_mattermost_adapter():
    runtime = MattermostSessionBindingRuntime()
    adapter = SimpleNamespace(
        is_connected=True,
        create_session_thread=AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="root1", error=None
        )),
    )
    runtime.wire_mattermost(None, adapter)

    assert await runtime.create_thread("session-1", "channel1", "Freunde") == (
        "channel1", "root1"
    )
    adapter.create_session_thread.assert_awaited_once_with(
        "channel1", "Freunde", session_id="session-1"
    )


def test_plugin_registers_both_platform_handlers_without_core_changes():
    ctx = MagicMock()
    register(ctx)

    registered = [call.args[0] for call in ctx.register_platform_handler.call_args_list]
    assert registered == ["mattermost", "api_server"]
    hooks = [call.args[0] for call in ctx.register_hook.call_args_list]
    assert hooks == ["post_llm_call", "on_session_end", "pre_gateway_dispatch"]
    ctx.register_platform.assert_called_once()


@pytest.mark.asyncio
async def test_api_turn_is_mirrored_once_in_user_then_assistant_order(tmp_path):
    path = tmp_path / "bindings.db"
    store_factory = lambda: MattermostSessionBindingStore(path)
    binding = store_factory().replace("session-1", "channel1", "root1")
    adapter = SimpleNamespace(
        send_session_mirror=AsyncMock(side_effect=[
            SimpleNamespace(success=True, message_id="post-user", error=None),
            SimpleNamespace(success=True, message_id="post-assistant", error=None),
        ])
    )
    runtime = MattermostSessionBindingRuntime(store_factory=store_factory)

    await runtime._mirror_api_turn(adapter, binding, "turn-1", "Hallo", "Hi zurück")
    await runtime._mirror_api_turn(adapter, binding, "turn-1", "Hallo", "Hi zurück")

    assert [call.kwargs["role"] for call in adapter.send_session_mirror.await_args_list] == [
        "user", "assistant"
    ]
    assert [call.args[2] for call in adapter.send_session_mirror.await_args_list] == [
        "Hallo", "Hi zurück"
    ]


@pytest.mark.asyncio
async def test_post_llm_hook_schedules_bound_api_turn_on_mattermost_loop(tmp_path):
    path = tmp_path / "bindings.db"
    store_factory = lambda: MattermostSessionBindingStore(path)
    store_factory().replace("session-1", "channel1", "root1")
    delivered = asyncio.Event()

    async def send(*_args, **_kwargs):
        role = _kwargs["role"]
        if role == "assistant":
            delivered.set()
        return SimpleNamespace(success=True, message_id=f"post-{role}", error=None)

    adapter = SimpleNamespace(is_connected=True, send_session_mirror=AsyncMock(side_effect=send))
    runtime = MattermostSessionBindingRuntime(store_factory=store_factory)
    runtime.wire_mattermost(None, adapter)

    runtime.post_llm_call(
        session_id="session-1",
        turn_id="turn-1",
        user_message="Hallo",
        assistant_response="Hi",
        platform="api_server",
    )
    runtime.on_session_end(
        session_id="session-1", turn_id="turn-1", completed=True
    )
    await asyncio.wait_for(delivered.wait(), timeout=2)

    assert adapter.send_session_mirror.await_count == 2


@pytest.mark.asyncio
async def test_failed_api_turn_is_not_mirrored(tmp_path):
    path = tmp_path / "bindings.db"
    store_factory = lambda: MattermostSessionBindingStore(path)
    store_factory().replace("session-1", "channel1", "root1")
    adapter = SimpleNamespace(is_connected=True, send_session_mirror=AsyncMock())
    runtime = MattermostSessionBindingRuntime(store_factory=store_factory)
    runtime.wire_mattermost(None, adapter)

    runtime.post_llm_call(
        session_id="session-1",
        turn_id="turn-1",
        user_message="Hallo",
        assistant_response="Fehler",
        platform="api_server",
    )
    runtime.on_session_end(
        session_id="session-1", turn_id="turn-1", completed=False, failed=True
    )
    await asyncio.sleep(0)

    adapter.send_session_mirror.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_mirror_role_can_retry(tmp_path):
    path = tmp_path / "bindings.db"
    store_factory = lambda: MattermostSessionBindingStore(path)
    binding = store_factory().replace("session-1", "channel1", "root1")
    adapter = SimpleNamespace(send_session_mirror=AsyncMock(side_effect=[
        SimpleNamespace(success=False, message_id=None, error="offline"),
        SimpleNamespace(success=True, message_id="post-user", error=None),
        SimpleNamespace(success=True, message_id="post-assistant", error=None),
    ]))
    runtime = MattermostSessionBindingRuntime(store_factory=store_factory)

    with pytest.raises(RuntimeError, match="offline"):
        await runtime._mirror_api_turn(adapter, binding, "turn-1", "Hallo", "Hi")
    await runtime._mirror_api_turn(adapter, binding, "turn-1", "Hallo", "Hi")

    assert adapter.send_session_mirror.await_count == 3


class _AsyncSessionStore:
    def __init__(self, current_session_id="mattermost-session"):
        self.current = SimpleNamespace(session_id=current_session_id)
        self.switches = []

    async def get_or_create_session(self, _source, touch_activity=False):
        assert touch_activity is False
        return self.current

    async def switch_session(self, session_key, session_id, expected_session_id=None):
        self.switches.append((session_key, session_id, expected_session_id))
        self.current = SimpleNamespace(session_id=session_id)
        return self.current


@pytest.mark.asyncio
async def test_authorized_mattermost_reply_continues_bound_hermes_session(tmp_path):
    path = tmp_path / "bindings.db"
    store_factory = lambda: MattermostSessionBindingStore(path)
    store_factory().replace("session-1", "channel1", "root1")
    runtime = MattermostSessionBindingRuntime(store_factory=store_factory)
    source = SimpleNamespace(
        platform=SimpleNamespace(value="mattermost"), chat_id="channel1"
    )
    event = SimpleNamespace(
        source=source,
        raw_message={"id": "reply1", "root_id": "root1", "channel_id": "channel1"},
        metadata={},
    )
    async_store = _AsyncSessionStore()
    gateway = SimpleNamespace(
        _is_user_authorized_for_source=lambda *_args, **_kwargs: True,
        _session_db=SimpleNamespace(get_session=AsyncMock(return_value={"id": "session-1"})),
        _session_key_for_source=lambda _source: "mattermost:channel1:root1",
        async_session_store=async_store,
    )

    result = await runtime.pre_gateway_dispatch(event=event, gateway=gateway)

    assert result == {"action": "allow"}
    assert async_store.switches == [
        ("mattermost:channel1:root1", "session-1", "mattermost-session")
    ]
    assert event.metadata["mattermost_bound_session_id"] == "session-1"


@pytest.mark.asyncio
async def test_bridge_origin_is_dropped_before_it_can_loop(tmp_path):
    runtime = MattermostSessionBindingRuntime(
        store_factory=lambda: MattermostSessionBindingStore(tmp_path / "bindings.db")
    )
    event = SimpleNamespace(
        source=SimpleNamespace(
            platform=SimpleNamespace(value="mattermost"), chat_id="channel1"
        ),
        raw_message={
            "id": "post1",
            "channel_id": "channel1",
            "props": {"hermes_origin": "api_server"},
        },
        metadata={},
    )

    result = await runtime.pre_gateway_dispatch(event=event, gateway=SimpleNamespace())

    assert result == {"action": "skip", "reason": "mattermost_bridge_echo"}


@pytest.mark.asyncio
async def test_unbound_mattermost_thread_keeps_existing_path(tmp_path):
    runtime = MattermostSessionBindingRuntime(
        store_factory=lambda: MattermostSessionBindingStore(tmp_path / "bindings.db")
    )
    event = SimpleNamespace(
        source=SimpleNamespace(
            platform=SimpleNamespace(value="mattermost"), chat_id="channel1"
        ),
        raw_message={"id": "root1", "channel_id": "channel1"},
        metadata={},
    )
    gateway = SimpleNamespace(async_session_store=MagicMock())

    assert await runtime.pre_gateway_dispatch(event=event, gateway=gateway) is None
    gateway.async_session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_bound_reply_does_not_repoint_session(tmp_path):
    path = tmp_path / "bindings.db"
    store_factory = lambda: MattermostSessionBindingStore(path)
    store_factory().replace("session-1", "channel1", "root1")
    runtime = MattermostSessionBindingRuntime(store_factory=store_factory)
    event = SimpleNamespace(
        source=SimpleNamespace(
            platform=SimpleNamespace(value="mattermost"), chat_id="channel1"
        ),
        raw_message={"id": "reply1", "root_id": "root1", "channel_id": "channel1"},
        metadata={},
    )
    gateway = SimpleNamespace(
        _is_user_authorized_for_source=lambda *_args, **_kwargs: False,
        async_session_store=MagicMock(),
    )

    assert await runtime.pre_gateway_dispatch(event=event, gateway=gateway) is None
    gateway.async_session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
async def test_adapter_posts_hidden_origin_metadata_on_mirrors():
    adapter = MattermostAdapter(PlatformConfig(
        enabled=True,
        token="token",
        extra={"url": "https://mattermost.example"},
    ))
    adapter._api_post = AsyncMock(return_value={"id": "post1"})

    result = await adapter.send_session_mirror(
        "channel1",
        "root1",
        "Hallo",
        session_id="session-1",
        turn_id="turn-1",
        role="user",
    )

    assert result.success is True
    payload = adapter._api_post.await_args.args[1]
    assert payload["root_id"] == "root1"
    assert payload["message"] == "**API client:** Hallo"
    assert payload["props"] == {
        "hermes_origin": "api_server",
        "hermes_session_id": "session-1",
        "hermes_turn_id": "turn-1",
        "hermes_role": "user",
        "disable_mentions": True,
    }


@pytest.mark.asyncio
async def test_adapter_creates_origin_marked_thread_root():
    adapter = MattermostAdapter(PlatformConfig(
        enabled=True,
        token="token",
        extra={"url": "https://mattermost.example"},
    ))
    adapter._api_post = AsyncMock(return_value={"id": "root1"})

    result = await adapter.create_session_thread(
        "channel1", "Freunde", session_id="session-1"
    )

    assert result.message_id == "root1"
    payload = adapter._api_post.await_args.args[1]
    assert payload["message"] == "### Freunde"
    assert "root_id" not in payload
    assert payload["props"]["hermes_role"] == "thread_root"
    assert payload["props"]["hermes_origin"] == "api_server"


@pytest.mark.asyncio
async def test_adapter_drops_bridge_echo_even_from_another_bot_identity():
    adapter = MattermostAdapter(PlatformConfig(
        enabled=True,
        token="token",
        extra={"url": "https://mattermost.example"},
    ))
    adapter._bot_user_id = "our-bot"
    adapter.handle_message = AsyncMock()
    post = {
        "id": "post1",
        "user_id": "bridge-bot",
        "channel_id": "channel1",
        "message": "echo",
        "props": {"hermes_origin": "api_server"},
    }

    await adapter._handle_ws_event({
        "event": "posted",
        "data": {"post": json.dumps(post), "channel_type": "O"},
    })

    adapter.handle_message.assert_not_awaited()
