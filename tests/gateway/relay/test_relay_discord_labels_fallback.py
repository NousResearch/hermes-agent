"""Discord prompt parity survives the routing store's degraded mode (#140259)."""

import json
from unittest.mock import AsyncMock

import pytest

import gateway.session as gateway_session
import hermes_state_registry
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionStore, build_session_key
from tests.gateway.relay.test_relay_discord_interaction_labels import (
    _message,
    _pinned_prompt,
    _slash,
)
from tests.gateway.relay.test_relay_interactive import _adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["restart", "thread", "rename", "peer-rename", "cleared", "late-session"])
@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("write_sessions_json", [False, True])
async def test_first_interaction_after_degraded_restart_keeps_prompt(
    tmp_path, monkeypatch, case, recovered, write_sessions_json,
):
    """Use real relay ingress and a failed SQLite open, then restart before any interaction.

    A recovered database deliberately contains older labels; fallback observations, including
    clearing both labels, must win. Session creation/reset order must not choose the labels.
    """
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")},
        write_sessions_json=write_sessions_json,
    )
    sessions = tmp_path / "sessions"
    chat = ({"chat_id": "th1", "chat_type": "thread", "thread_id": "th1", "parent_chat_id": "ch1"}
            if case == "thread" else {"chat_id": "ch1", "chat_type": "group"})
    stale = _message(chat, chat_name="Old channel", chat_topic="Old topic")
    seed = SessionStore(sessions, config)
    assert seed.record_chat_labels(stale.source)
    seed.close_all_db_handles()

    acquire = hermes_state_registry.acquire

    def unavailable(*args, **kwargs):
        raise OSError("state.db temporarily unavailable")

    monkeypatch.setattr(hermes_state_registry, "acquire", unavailable)
    store = SessionStore(sessions, config)
    assert store._routing_db is None
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()

    async def relay(event):
        await adapter._on_inbound(event)
        return store.get_or_create_session(event.source)

    message = _message(chat)
    if case == "late-session":
        late = _message(chat, "u2")
        await adapter._on_inbound(late)
        message = _message(chat, chat_name="New channel", chat_topic="New topic")
        await relay(message)
        store.get_or_create_session(late.source)
    else:
        entry = await relay(message)
        if case in ("rename", "peer-rename", "cleared"):
            labels = ((None, None) if case == "cleared" else ("New channel", "New topic"))
            await relay(_message(chat, "u2" if case == "peer-rename" else "u1",
                                 chat_name=labels[0], chat_topic=labels[1]))
            message = _message(chat, chat_name=labels[0], chat_topic=labels[1])
        if case == "peer-rename":
            store.reset_session(entry.session_key)

    store.close_all_db_handles()
    if recovered:
        monkeypatch.setattr(hermes_state_registry, "acquire", acquire)
    store = SessionStore(sessions, config)
    adapter, _ = _adapter(platform="discord")
    adapter.set_session_store(store)
    interaction = {"member": {"nick": "Ben D", "user": {"id": "u1", "username": "ben"}}}
    if case == "thread":
        interaction.update(channel_id="th1", channel={"id": "th1", "type": 11, "parent_id": "ch1"})
    slash = await _slash(adapter, **interaction)

    assert build_session_key(slash.source) == build_session_key(message.source)
    for tools_loaded in (False, True):
        monkeypatch.setattr(gateway_session, "_discord_tools_loaded", lambda: tools_loaded)
        assert _pinned_prompt(slash.source) == _pinned_prompt(message.source)
    if recovered:
        # The first interaction must repair SQLite without waiting for another text message.
        key = store._chat_labels_meta_key(Platform.DISCORD, "g1", chat["chat_id"])
        assert json.loads(store._routing_db.get_meta(key)) == [message.source.chat_name, message.source.chat_topic]
    store.close_all_db_handles()
