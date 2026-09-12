"""Secondary multiplex profiles' HOME CHANNELS must get a shutdown notice too.

``_notify_active_sessions_of_shutdown`` (gateway/run_shutdown.py) already resolves the
per-SESSION half of shutdown notices through each session's own profile adapter (fixed in
45a6101f36 — see test_multiplex_notice_egress_profile_adapter.py). But the second,
unconditional "home channel" broadcast loop in the same function only ever walked
``self.adapters``/``self.config`` — the DEFAULT profile's adapters and config — and never
``self._profile_adapters``, so a secondary profile's home channel (its own config.yaml, own
home directory, own bot) got no notification at all that the gateway was shutting down, even
though its bot was live and connected.

The fix resolves each secondary profile's OWN config (own home_channel) the same way the
handoff watcher already does for delivery (``_handoff_resolve_scope`` / ``_handoff_watcher``):
via ``_handoff_watch_scopes`` + ``_async_profile_runtime_scope`` + a fresh ``load_gateway_config()``
call, instead of guessing with the root config.
"""

from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner


class _Adapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content=None, metadata=None, **kw):
        self.sent.append(chat_id)
        return SimpleNamespace(success=True, message_id="m", error=None)


def _config(chat_id):
    cfg = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    cfg.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=chat_id, name=f"home-{chat_id}",
    )
    return cfg


def _runner():
    r = object.__new__(GatewayRunner)
    r.config = _config("1111")  # default/root profile's home channel
    r.adapters = {Platform.TELEGRAM: _Adapter()}
    r._profile_adapters = {"sec": {Platform.TELEGRAM: _Adapter()}}
    r._primary_profile_name = "default"
    r.session_store = None
    r._session_sources = None
    r._running_agents = {}
    r._restart_requested = False
    r._restart_command_source = None
    r._snapshot_running_agents = lambda: []  # isolate to the home-channel loops, no per-session noise
    return r


@asynccontextmanager
async def _null_profile_scope(_profile_home):
    yield


@pytest.mark.asyncio
async def test_secondary_profile_home_channel_gets_shutdown_notice(monkeypatch):
    import gateway.run as gateway_run

    r = _runner()
    secondary_config = _config("2222")  # secondary profile's OWN, DIFFERENT home channel

    monkeypatch.setattr(
        gateway_run, "_handoff_watch_scopes", lambda _runner: [(None, None), ("sec", "/tmp/fake-profile-home")],
    )
    monkeypatch.setattr(gateway_run, "load_gateway_config", lambda: secondary_config)
    monkeypatch.setattr(gateway_run, "_async_profile_runtime_scope", _null_profile_scope)

    await r._notify_active_sessions_of_shutdown()

    # Default/root home channel still gets its notice through the default bot.
    assert r.adapters[Platform.TELEGRAM].sent == ["1111"]
    # The secondary profile's OWN home channel gets a notice through ITS OWN bot.
    assert r._profile_adapters["sec"][Platform.TELEGRAM].sent == ["2222"]


@pytest.mark.asyncio
async def test_secondary_profile_without_config_is_skipped_not_fatal(monkeypatch):
    """A profile whose config fails to load must not blow up the whole shutdown notice pass."""
    import gateway.run as gateway_run

    r = _runner()

    def _boom():
        raise RuntimeError("config.yaml exploded")

    monkeypatch.setattr(
        gateway_run, "_handoff_watch_scopes", lambda _runner: [(None, None), ("sec", "/tmp/fake-profile-home")],
    )
    monkeypatch.setattr(gateway_run, "load_gateway_config", _boom)
    monkeypatch.setattr(gateway_run, "_async_profile_runtime_scope", _null_profile_scope)

    await r._notify_active_sessions_of_shutdown()  # must not raise

    assert r.adapters[Platform.TELEGRAM].sent == ["1111"]
    assert r._profile_adapters["sec"][Platform.TELEGRAM].sent == []
