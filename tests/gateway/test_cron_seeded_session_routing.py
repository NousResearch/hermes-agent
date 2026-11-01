"""Cron-seeded and legacy lanes keep their owning profile across a restart."""

from pathlib import Path

import pytest

from cron.scheduler_delivery import _seed_cron_thread_session
from cron.scheduler_provider import _profile_cron_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.session_identity import identity_of
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from tests.gateway.test_background_process_notifications import AdmittingHandler


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "thread"}


@pytest.fixture
def mux(tmp_path, monkeypatch):
    import hermes_state

    # Undo the autouse fixture's fixed DB path so real profile scopes select their stores.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    primary = tmp_path / ".hermes"
    secondary = primary / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(primary))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, sessions_dir=primary / "sessions")
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    for profile, platform in (("default", Platform.TELEGRAM), ("secondary", Platform.DISCORD)):
        adapter = _Adapter(PlatformConfig(enabled=True), platform)
        adapter.gateway_runner = runner
        adapter.set_owner_profile(profile)
        adapter.handle_message = AdmittingHandler()
        if profile == "default":
            runner.adapters[platform] = adapter
        else:
            runner._profile_adapters[profile] = {platform: adapter}
    runner.session_store = SessionStore(primary / "sessions", runner.config)
    discord = runner._profile_adapters["secondary"][Platform.DISCORD]
    discord.set_session_store(runner.session_store)
    return runner, primary, secondary, discord


async def _restore_and_notify(runner, entry, discord):
    # Reload the routing index from SQLite: no live provenance survives a restart.
    runner.session_store = SessionStore(runner.session_store.sessions_dir, runner.config)
    event = {"type": "completion", "session_key": entry.session_key, "session_id": "background"}
    source = runner._build_process_event_source(event)
    assert await runner._completion_delivery_ready(event)
    assert await runner._inject_watch_notification("Background task completed", event) is True
    discord.handle_message.assert_awaited_once()
    accepted = discord.handle_message.await_args.args[0]
    assert accepted.source.profile == "secondary"
    assert accepted.metadata["gateway_session_key"] == entry.session_key
    assert runner._session_key_for_source(accepted.source) == entry.session_key
    return source


@pytest.mark.asyncio
async def test_cron_seed_notification_returns_to_owning_profile(mux):
    runner, primary, secondary, discord = mux
    assert get_hermes_home() == primary
    with _profile_cron_scope(secondary):
        _seed_cron_thread_session({"id": "brief"}, discord, "discord", "channel", "thread", "Brief")
        entry, = runner.session_store._entries.values()
        db = runner.session_store._db_for_key(entry.session_key)
        assert db.get_messages(entry.session_id)[0]["content"].endswith("Brief")
    source = await _restore_and_notify(runner, entry, discord)
    restored = identity_of(source)
    assert restored.runtime_home == secondary
    assert restored.runtime_profile == entry.origin.profile == "secondary"
    assert restored.transport_profile == entry.transport_profile == "secondary"
    assert restored.transport is None
    assert get_hermes_home() == primary
    with SessionDB(primary / "state.db", read_only=True) as db:
        assert db.get_session(entry.session_id) is None
        assert db.get_messages(entry.session_id) == []
    runner.adapters[Platform.TELEGRAM].handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_seed_notification_recovers_runtime_not_transport(mux):
    runner, primary, secondary, discord = mux
    with _profile_cron_scope(secondary):
        entry = runner.session_store.get_or_create_session(SessionSource(
            platform=Platform.DISCORD, chat_id="thread", chat_type="thread", thread_id="thread",
        ))
    assert entry.origin.profile is None and entry.transport_profile is None
    source = await _restore_and_notify(runner, entry, discord)
    assert source.profile == "secondary"
    assert identity_of(source) is None  # The key proves runtime, never which bot received it.
    assert runner.session_store._entries[entry.session_key].origin.profile is None
    assert get_hermes_home() == primary
    with SessionDB(primary / "state.db", read_only=True) as db:
        assert db.get_session(entry.session_id) is None
        assert db.get_messages(entry.session_id) == []
    runner.adapters[Platform.TELEGRAM].handle_message.assert_not_awaited()
