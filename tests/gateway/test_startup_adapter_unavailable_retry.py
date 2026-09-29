"""A platform with no adapter at boot is queued for retry, not stranded (#126356).

A platform plugin whose load overran its deadline during a slow boot (unclean-reboot I/O contention)
left the enabled platform with no adapter and no reconnect queue entry: the gateway logged one
WARNING and never served the platform again. These tests pin the retry contract: the platform is
queued like any other retryable startup failure, a reconnect pass re-creates the adapter once the
plugin (re)registers it, and a pass that still finds no adapter keeps the platform queued instead of
dropping it.
"""
import asyncio
import hashlib
import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _HealthyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(monkeypatch, tmp_path, create_adapter, platforms=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms=platforms or {Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", create_adapter)

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    return runner


async def _wait_platform_status(key, predicate, timeout=5.0):
    """read_runtime_status() reads the FILE, while publish_runtime_status() persists through an
    async writer — poll until the platform's persisted status satisfies `predicate`."""
    deadline = time.monotonic() + timeout
    while True:
        plat = (read_runtime_status() or {}).get("platforms", {}).get(key)
        if plat is not None and predicate(plat):
            return plat
        assert time.monotonic() < deadline, f"status for {key} never satisfied predicate: {plat}"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_missing_adapter_at_boot_is_queued_for_retry(monkeypatch, tmp_path):
    """The #126356 outage shape: no adapter at boot must queue the platform with a visible
    ``retrying``/``adapter_unavailable`` status instead of one WARNING and silence."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        assert runner.adapters == {}
        info = runner._failed_platforms.get(Platform.TELEGRAM)
        assert info is not None, "platform with no adapter must be queued for reconnect"
        assert info["adapter_unavailable"] is True
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "retrying"
        assert state["platforms"]["telegram"]["error_code"] == "adapter_unavailable"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_queued_platform_heals_once_the_plugin_registers(monkeypatch, tmp_path):
    """Once adapter creation succeeds again (the loader's retry re-registered the platform), one
    reconnect pass installs the adapter and the platform leaves the queue connected."""
    available: list = [None]

    def _create(platform, cfg):
        return available[0]

    runner = _runner(monkeypatch, tmp_path, _create)
    try:
        assert await runner.start() is True
        assert Platform.TELEGRAM in runner._failed_platforms
        # The plugin finishes loading: adapter creation now works. One watcher pass heals.
        available[0] = _HealthyAdapter()
        info = runner._failed_platforms[Platform.TELEGRAM]
        info["next_retry"] = 0
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM not in runner._failed_platforms
        assert isinstance(runner.adapters.get(Platform.TELEGRAM), _HealthyAdapter)
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "connected"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_reconnect_without_adapter_stays_queued(monkeypatch, tmp_path):
    """A reconnect pass that still finds no adapter must NOT drop the platform (the pre-fix
    behaviour) — it backs off and keeps the entry for the plugin's eventual registration."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        info = runner._failed_platforms[Platform.TELEGRAM]
        info["next_retry"] = 0
        attempts_before = info["attempts"]
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM in runner._failed_platforms, "adapterless retry must not drop the platform"
        assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == attempts_before + 1
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "retrying"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_reconnect_without_flag_still_drops_unknown_platform(monkeypatch, tmp_path):
    """Protection: a platform queued WITHOUT the adapter_unavailable marker (it had an adapter that
    later vanished, e.g. plugin uninstalled mid-run) keeps the old drop-on-None semantics."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: _HealthyAdapter())
    try:
        assert await runner.start() is True
        assert isinstance(runner.adapters.get(Platform.TELEGRAM), _HealthyAdapter)
        # Simulate a runtime fatal queue entry, then the plugin disappearing.
        adapter = runner.adapters.pop(Platform.TELEGRAM)
        runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
            Platform.TELEGRAM, adapter, runner.config.platforms[Platform.TELEGRAM],
            attempts=0, delay=0.0,
        )
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM not in runner._failed_platforms
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_adapterless_queue_entry_reserves_credential_from_config(monkeypatch, tmp_path):
    """ehz0ah review: adapter=None recorded credential_claim=None, so a same-token secondary
    scanned before the plugin loaded could connect first and the later primary retry would
    collide with its own token's new owner. The queued entry must reserve the token from the
    CONFIG for the entry's whole lifetime."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        entry = runner._failed_platforms[Platform.TELEGRAM]
        claim = entry["credential_claim"]
        assert claim is not None, "adapterless entry must still reserve the primary's credential"
        assert claim[0] == Platform.TELEGRAM
        # The claim must be the config-derived fingerprint of the runner's effective token —
        # identical to what the eventual adapter would produce.
        effective = runner.config.platforms[Platform.TELEGRAM]
        assert claim == runner._config_credential_claim(Platform.TELEGRAM, effective)
        assert claim in runner._primary_resource_claims("default")
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_no_credential_drop_leaves_terminal_status(monkeypatch, tmp_path):
    """ehz0ah review: dropping a queued entry only deletes the queue item — without a terminal
    status gateway_state.json would keep saying 'retrying' forever with needs_attention never
    raised."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        assert Platform.TELEGRAM in runner._failed_platforms
        runner.config.platforms[Platform.TELEGRAM].token = None  # credential pulled from config
        info = runner._failed_platforms[Platform.TELEGRAM]
        info["next_retry"] = 0
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM not in runner._failed_platforms
        plat = await _wait_platform_status("telegram", lambda p: p["state"] != "retrying")
        assert plat["state"] == "fatal"
        assert plat["error_code"] == "no_credential"
        assert plat["needs_attention"] is True
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_adapterless_secondary_queued_in_profile_scope(monkeypatch, tmp_path):
    """ehz0ah review: a secondary whose platform plugin is missing at scan time was treated as
    success and stranded by the recorded signature. It must be queued in the profile's OWN
    reconnect scope (not the primary queue) instead."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        cfg_stub = SimpleNamespace(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok-secondary")})
        async def _load_cfg(name, home):
            return cfg_stub
        monkeypatch.setattr(runner, "_load_secondary_profile_config", _load_cfg)
        monkeypatch.setattr(runner, "_multiplex_on", lambda: False, raising=False)
        runner._running = True
        async def _still_missing(*args, **kwargs):
            return None, False  # plugin still not loaded: stay on the backoff loop
        monkeypatch.setattr(runner, "_secondary_reconnect_attempt", _still_missing)
        connected = await runner._start_one_profile_adapters("prof2", tmp_path, {})
        assert connected == 0
        queued = (runner._profile_failed_platforms.get("prof2") or {})
        assert Platform.TELEGRAM in queued, "adapterless secondary must be queued in its own scope"
        plat = await _wait_platform_status("prof2:telegram", lambda p: True)
        assert plat["state"] == "retrying"
        assert plat["error_code"] == "adapter_unavailable"
    finally:
        await runner.stop()


def _stub_adapter(**attrs):
    """The shape _adapter_credential_fingerprint() probes on a real adapter instance."""
    return SimpleNamespace(config=None, **attrs)


@pytest.mark.parametrize("platform,extra,adapter_attrs", [
    (Platform.FEISHU, {"app_id": "cli_feishu123"}, {"_app_id": "cli_feishu123"}),
    (Platform.DINGTALK, {"client_id": "ding_abc"}, {"_client_id": "ding_abc"}),
    (Platform.WECOM, {"bot_id": "bot_wecom1"}, {"_bot_id": "bot_wecom1"}),
])
def test_config_claim_mirrors_app_style_identities(platform, extra, adapter_attrs):
    """ehz0ah review round 2: the config-derived claim must cover every identity
    _adapter_credential_fingerprint() supports — app-style ids (Feishu app_id, DingTalk
    client_id, WeCom bot_id) live in PlatformConfig.extra, and a token-only shim left them
    unreserved."""
    from gateway.run import GatewayRunner
    config = PlatformConfig(enabled=True, extra=dict(extra))
    claim = GatewayRunner._config_credential_claim(platform, config)
    assert claim is not None, f"{platform.value}: config claim must reserve the app-style identity"
    assert claim[0] == platform
    # Identical to the fingerprint the eventual adapter instance produces.
    assert claim[1] == GatewayRunner._adapter_credential_fingerprint(_stub_adapter(**adapter_attrs))


@pytest.mark.asyncio
async def test_adapterless_queue_entry_reserves_app_id_credential(monkeypatch, tmp_path):
    """Startup-order regression for a NON-token platform: the Feishu plugin missing at boot must
    still reserve the app_id from config.extra for the queue entry's lifetime."""
    from gateway.run import GatewayRunner
    platforms = {Platform.FEISHU: PlatformConfig(enabled=True, extra={"app_id": "cli_feishu123"})}
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None, platforms=platforms)
    try:
        assert await runner.start() is True
        entry = runner._failed_platforms[Platform.FEISHU]
        claim = entry["credential_claim"]
        assert claim is not None, "feishu adapterless entry must reserve the app_id"
        assert claim[1] == GatewayRunner._adapter_credential_fingerprint(
            _stub_adapter(_app_id="cli_feishu123"))
        assert claim in runner._primary_resource_claims("default")
    finally:
        await runner.stop()
