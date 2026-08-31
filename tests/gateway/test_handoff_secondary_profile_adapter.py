"""A secondary profile's handoff must deliver through ITS OWN adapter/config.

Regression guard for the third `/handoff` multi-profile bug. Even after the
watcher polls the right ``state.db`` and the session key carries the profile
namespace, ``_process_handoff`` still resolved delivery from ``self.adapters``
and ``self.config`` — which on a multiplexed gateway hold ONLY the primary
profile's adapters and home channel. A medicina handoff was therefore sent by
the default profile's bot, to the default profile's chat, while persisting a
``agent:medicina:...`` key and reporting ``handoff_state='completed'``: a
false positive that looks fine in the database and is wrong on the wire.

This was caught by an adversarial review reading gateway.log, not by the
end-to-end test — the log line showed ``hermes_plugins.telegram_platform``
(primary) instead of the secondary's ``..._home_<hash>`` adapter module.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionEntry


def _adapter(tag):
    """A platform adapter stand-in that records which one was used."""
    a = MagicMock()
    a.tag = tag
    a.send = AsyncMock(return_value=SimpleNamespace(success=True))
    a.create_handoff_thread = AsyncMock(return_value=None)
    a._bot = None
    return a


def _config(chat_id):
    cfg = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    cfg.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=chat_id, name=f"home-{chat_id}",
    )
    return cfg


def _make_multiplex_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = _config("1111")          # primary/default home
    runner.config.multiplex_profiles = True
    runner.adapters = {Platform.TELEGRAM: _adapter("primary")}
    runner._profile_adapters = {
        "medicina": {Platform.TELEGRAM: _adapter("medicina")},
    }
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False,
    )

    captured = {}

    store = MagicMock()
    store.get_or_create_session = AsyncMock(return_value=SessionEntry(
        session_key="k", session_id="s",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    ))

    async def _switch(key, sid):
        captured["session_key"] = key
        return SessionEntry(
            session_key=key, session_id=sid,
            created_at=datetime.now(), updated_at=datetime.now(),
            platform=Platform.TELEGRAM, chat_type="dm",
        )

    store.switch_session = AsyncMock(side_effect=_switch)
    # ``async_session_store`` is a derived property with no setter: it rebuilds
    # the facade whenever ``facade._store is not self.session_store``. Wiring
    # the mock as that ``_store`` is what makes the primed cache survive.
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=store.get_or_create_session,
        switch_session=store.switch_session,
    )
    runner._evict_cached_agent = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._session_db = None

    async def _handle_message(event):
        captured["source"] = event.source
        return "ok"

    runner._handle_message = AsyncMock(side_effect=_handle_message)
    return runner, captured


def _spy_transport_factory(used):
    """Build a resolve_delivery_transport stand-in that records what it got.

    The real transport exposes ``.adapter`` AND an awaitable ``.send``; the
    handoff uses both, so the stand-in must too.
    """
    def _spy(platform, config, adapters):
        adapter = adapters[platform]
        used["adapter_tag"] = adapter.tag
        used["home_chat_id"] = config.get_home_channel(platform).chat_id

        async def _send(_platform, _chat_id, _text, _metadata=None):
            used["sent_via"] = adapter.tag
            return SimpleNamespace(success=True)

        return SimpleNamespace(adapter=adapter, send=_send)

    return _spy


@pytest.mark.asyncio
async def test_secondary_profile_handoff_uses_its_own_adapter(monkeypatch):
    """medicina's handoff must NOT be delivered by the primary's adapter."""
    runner, captured = _make_multiplex_runner()

    used = {}
    monkeypatch.setattr(
        "gateway.run.resolve_delivery_transport", _spy_transport_factory(used),
    )
    # The watcher would already be inside _profile_runtime_scope here, so a
    # fresh load resolves the secondary's config.
    monkeypatch.setattr("gateway.run.load_gateway_config", lambda: _config("2222"))

    await runner._process_handoff(
        {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
        profile_name="medicina",
    )

    assert used["adapter_tag"] == "medicina", (
        "delivery must use the secondary profile's own adapter, not the primary's"
    )
    assert used["sent_via"] == "medicina", "the message went out on the wrong bot"
    assert used["home_chat_id"] == "2222", (
        "delivery must use the secondary profile's own home channel"
    )
    assert captured["session_key"].startswith("agent:medicina:"), (
        f"session key must carry the profile namespace, got {captured['session_key']}"
    )
    assert captured["source"].profile == "medicina"


@pytest.mark.asyncio
async def test_default_profile_handoff_keeps_primary_adapter(monkeypatch):
    """The default/root path must behave exactly as before the fix."""
    runner, captured = _make_multiplex_runner()

    used = {}
    monkeypatch.setattr(
        "gateway.run.resolve_delivery_transport", _spy_transport_factory(used),
    )

    await runner._process_handoff(
        {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
        profile_name=None,
    )

    assert used["adapter_tag"] == "primary"
    assert used["home_chat_id"] == "1111"


@pytest.mark.asyncio
async def test_secondary_profile_without_live_adapters_fails_loudly(monkeypatch):
    """Never silently fall back to the primary's bot — that ships to the wrong chat.

    Raising marks the row ``failed`` and the CLI reports it; delivering through
    another profile's bot would look like success.
    """
    runner, _ = _make_multiplex_runner()
    runner._profile_adapters = {}

    monkeypatch.setattr(
        "gateway.run.resolve_delivery_transport", _spy_transport_factory({}),
    )

    with pytest.raises(RuntimeError, match="no live adapters"):
        await runner._process_handoff(
            {"id": "cli-session", "title": "work", "handoff_platform": "telegram"},
            profile_name="medicina",
        )
