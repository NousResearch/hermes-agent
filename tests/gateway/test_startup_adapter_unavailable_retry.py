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


@pytest.mark.asyncio
async def test_adapterless_secondary_with_claimed_credential_is_refused(monkeypatch, tmp_path):
    """ehz0ah round 3: the adapterless branch ran BEFORE the credential-conflict check, so a
    secondary sharing the queued primary's token was scheduled anyway — when the plugin
    registered, its reconnect would race the primary's retry for the same credential. The
    adapterless entry must be refused exactly like the adapter-present path."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        assert runner._failed_platforms[Platform.TELEGRAM]["credential_claim"] is not None
        cfg_stub = SimpleNamespace(
            # Same token as the queued primary ("***" in the fixture).
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
        async def _load_cfg(name, home):
            return cfg_stub
        monkeypatch.setattr(runner, "_load_secondary_profile_config", _load_cfg)
        monkeypatch.setattr(runner, "_multiplex_on", lambda: False, raising=False)
        runner._running = True
        claimed = runner._primary_resource_claims("default")
        connected = await runner._start_one_profile_adapters("prof2", tmp_path, claimed)
        assert connected == 0
        assert Platform.TELEGRAM not in (runner._profile_failed_platforms.get("prof2") or {}), \
            "a secondary claiming an already-owned credential must be refused, not queued"
        plat = await _wait_platform_status("prof2:telegram", lambda p: True)
        assert plat["state"] == "fatal"
        assert plat["error_code"] == "duplicate_credential"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_adapterless_secondary_terminal_exit_publishes_status(monkeypatch, tmp_path):
    """ehz0ah round 3: when the profile-scoped retry stops for good (profile disabled /
    credential removed / credential now owned elsewhere), the persisted state must not keep
    reporting 'retrying' — the queue-slot pop needs a terminal status beside it."""
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
        async def _terminal(*args, **kwargs):
            return None, None  # disabled / credential removed: give up for good
        monkeypatch.setattr(runner, "_secondary_reconnect_attempt", _terminal)
        await runner._start_one_profile_adapters("prof2", tmp_path, {})
        task = runner._profile_failed_platforms["prof2"][Platform.TELEGRAM]
        await task  # one pass, terminal exit; the finally pops the queue slot
        plat = await _wait_platform_status("prof2:telegram", lambda p: p["state"] != "retrying")
        assert plat["state"] == "disconnected"
        assert plat["error_code"] == "retry_stopped"
    finally:
        await runner.stop()


def _secondary_scan_harness(monkeypatch, runner, tmp_path, token, attempt_stub, hold_loop=True):
    """Wire a runner for direct ``_start_one_profile_adapters`` scans: every secondary profile
    loads a config whose telegram token is `token`, adapter creation fails (plugin missing),
    and the reconnect loop is parked on `attempt_stub`. ``hold_loop`` (default) replaces the
    loop coroutine itself with a parked wait so the reservation outlives the scan without a
    real profile home (the loop's own scope entry needs one); pass False to drive the real
    loop to a terminal exit."""
    cfg_stub = SimpleNamespace(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token=token)})
    async def _load_cfg(name, home):
        return cfg_stub
    monkeypatch.setattr(runner, "_load_secondary_profile_config", _load_cfg)
    monkeypatch.setattr(runner, "_multiplex_on", lambda: False, raising=False)
    monkeypatch.setattr(runner, "_secondary_reconnect_attempt", attempt_stub)
    if hold_loop:
        async def _held(*args, **kwargs):
            await asyncio.Event().wait()  # parked until cancelled at runner.stop()
        monkeypatch.setattr(runner, "_run_secondary_profile_reconnect", _held)
    runner._running = True
    return runner


@pytest.mark.asyncio
async def test_second_secondary_same_credential_refused_while_first_queued(monkeypatch, tmp_path):
    """ehz0ah round 4: two secondaries sharing one credential while the plugin is unavailable
    both saw no owner and both scheduled retries — the queued entry never reserved its config
    claim. One credential, one owner: the first secondary's reservation must cover the queue's
    whole lifetime and the second scan must be refused exactly like an adapter-present
    duplicate."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True

        async def _parked(*args, **kwargs):
            return None, False  # plugin still missing: hold the backoff loop

        _secondary_scan_harness(monkeypatch, runner, tmp_path, "shared-tok", _parked)
        claimed = runner._primary_resource_claims("default")
        await runner._start_one_profile_adapters("prof2", tmp_path, claimed)
        assert Platform.TELEGRAM in (runner._profile_failed_platforms.get("prof2") or {})
        claim = runner._config_credential_claim(
            Platform.TELEGRAM, PlatformConfig(enabled=True, token="shared-tok"))
        assert runner._secondary_queued_claims.get(claim) == "prof2", \
            "the queued secondary must reserve its credential for the queue's lifetime"

        await runner._start_one_profile_adapters("prof3", tmp_path, claimed)
        assert Platform.TELEGRAM not in (runner._profile_failed_platforms.get("prof3") or {}), \
            "a second secondary sharing the credential must be refused, not queued"
        plat = await _wait_platform_status("prof3:telegram", lambda p: True)
        assert plat["state"] == "fatal"
        assert plat["error_code"] == "duplicate_credential"
        # The refusal must not steal the reservation from the first owner.
        assert runner._secondary_queued_claims.get(claim) == "prof2"
    finally:
        await runner.stop()


def _install_reconnect_attempt_externals(monkeypatch, runner, tmp_path, token, adapter):
    """Externals the REAL ``_secondary_reconnect_attempt`` touches, so a test drives the true
    arbitration code with no profile home, scope, or plugin machinery. The plugin is "back":
    ``_create_adapter`` now returns `adapter` carrying `token` as its credential."""
    import contextlib

    import gateway.run as gateway_run

    @contextlib.contextmanager
    def _fake_scope(profile_home, *, hydrate_secrets=True):
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", _fake_scope)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: tmp_path)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: None)
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token=token)}),
    )
    monkeypatch.setattr(adapter, "token", token, raising=False)
    adapter.config.token = token
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: adapter)
    return adapter


@pytest.mark.asyncio
async def test_queued_secondary_credential_blocks_other_profile_reconnect(monkeypatch, tmp_path):
    """ehz0ah round 4 (reconnect ordering): prof2's adapterless retry is queued with token T.
    When the plugin registers, prof3's queued retry for the same T must lose at ATTEMPT-time
    arbitration too — ``_secondary_reconnect_attempt`` read only the PRIMARY claims, so a
    queued (or live) secondary owner was invisible and both retries raced the credential."""
    async def _parked(*args, **kwargs):
        return None, False

    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        real_attempt = runner._secondary_reconnect_attempt
        _secondary_scan_harness(monkeypatch, runner, tmp_path, "shared-tok", _parked)
        await runner._start_one_profile_adapters("prof2", tmp_path, {})
        claim = runner._config_credential_claim(
            Platform.TELEGRAM, PlatformConfig(enabled=True, token="shared-tok"))
        assert runner._secondary_queued_claims.get(claim) == "prof2"

        # The plugin is back: prof3's queued retry rebuilds an adapter for the SAME token.
        runner._secondary_reconnect_attempt = real_attempt
        adapter = _install_reconnect_attempt_externals(
            monkeypatch, runner, tmp_path, "shared-tok", _HealthyAdapter())
        rebuilt, success = await runner._secondary_reconnect_attempt("prof3", Platform.TELEGRAM)
        assert rebuilt is None and success is None, \
            "a credential reserved by another profile's queued retry must stop this attempt"
        await runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_live_secondary_credential_blocks_other_profile_reconnect(monkeypatch, tmp_path):
    """Reconnect ordering, live-owner arm: prof2's adapter CONNECTED first; prof3's pending
    retry for the same token must stop at attempt time (primary-only claims missed a live
    secondary owner)."""
    async def _parked(*args, **kwargs):
        return None, False

    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        _secondary_scan_harness(monkeypatch, runner, tmp_path, "shared-tok", _parked)
        live = _HealthyAdapter()
        live.token = "shared-tok"
        live.config.token = "shared-tok"
        runner._profile_adapters["prof2"] = {Platform.TELEGRAM: live}

        real_attempt = GatewayRunner._secondary_reconnect_attempt
        adapter = _install_reconnect_attempt_externals(
            monkeypatch, runner, tmp_path, "shared-tok", _HealthyAdapter())
        rebuilt, success = await real_attempt(runner, "prof3", Platform.TELEGRAM)
        assert rebuilt is None and success is None, \
            "a credential owned by a live secondary adapter must stop this attempt"
        await runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_reservation_released_when_retry_loop_exits(monkeypatch, tmp_path):
    """The reservation's lifetime is the queue slot's: once prof2's retry stops for good, the
    same credential is claimable again — a later same-credential profile must be QUEUED, not
    refused by a ghost reservation."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True

        async def _terminal(*args, **kwargs):
            return None, None

        _secondary_scan_harness(monkeypatch, runner, tmp_path, "shared-tok", _terminal, hold_loop=False)
        await runner._start_one_profile_adapters("prof2", tmp_path, {})
        claim = runner._config_credential_claim(
            Platform.TELEGRAM, PlatformConfig(enabled=True, token="shared-tok"))
        assert runner._secondary_queued_claims.get(claim) == "prof2"
        task = runner._profile_failed_platforms["prof2"][Platform.TELEGRAM]
        await task  # terminal exit; the finally releases slot AND reservation
        assert claim not in runner._secondary_queued_claims

        async def _parked(*args, **kwargs):
            return None, False

        async def _held(*args, **kwargs):
            await asyncio.Event().wait()  # parked until cancelled at runner.stop()

        monkeypatch.setattr(runner, "_secondary_reconnect_attempt", _parked)
        monkeypatch.setattr(runner, "_run_secondary_profile_reconnect", _held)
        await runner._start_one_profile_adapters("prof3", tmp_path, {})
        assert Platform.TELEGRAM in (runner._profile_failed_platforms.get("prof3") or {}), \
            "with the reservation released, a later same-credential profile may queue"
        assert runner._secondary_queued_claims.get(claim) == "prof3"
    finally:
        await runner.stop()
