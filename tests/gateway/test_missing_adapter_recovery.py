"""Failed plugin loads must not permanently disable a platform.

Regression for #126356: a transient plugin-load overrun (the 10s deadline blown by startup
I/O contention after an unclean reboot) left the platform with no adapter, no runtime status
and no retry — dead until a manual restart. The fix queues the missing adapter for the
reconnect watcher (``retrying`` status, so the normal ``needs_attention`` escalation and the
status indicator see a never-loaded platform exactly like a loaded-then-disconnected one),
revives failed deferred loads on each reconnect pass, and raises the load deadline while a
startup progress lease is held.
"""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from gateway.run import GatewayRunner


def _telegram_config(tmp_path) -> GatewayConfig:
    return GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")},
        sessions_dir=tmp_path / "sessions",
    )


@pytest.mark.asyncio
async def test_start_prefilter_queues_missing_adapter_with_retrying_status(
    tmp_path, monkeypatch
):
    """A missing adapter at startup is queued for retry with visible status, not dropped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = GatewayRunner(_telegram_config(tmp_path))
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)

    aborted, enabled, _skipped, pending = await runner._start_prefilter_platforms()

    assert aborted is False
    assert enabled == 1
    assert pending == []
    # Queued for the reconnect watcher with the missing-adapter marker ...
    assert Platform.TELEGRAM in runner._failed_platforms
    info = runner._failed_platforms[Platform.TELEGRAM]
    assert info.get("missing_adapter") is True
    assert info.get("credential_claim") is None
    # ... and stamped retrying so status surfaces carry it like a disconnect.
    from gateway.status import flush_runtime_status_async, read_runtime_status

    assert await flush_runtime_status_async(timeout=5.0)
    state = read_runtime_status()
    entry = state["platforms"]["telegram"]
    assert entry["state"] == "retrying"
    assert entry["error_code"] == "adapter_missing"


@pytest.mark.asyncio
async def test_reconnect_keeps_missing_adapter_queued_with_backoff(tmp_path, monkeypatch):
    """The watcher must not drop a still-missing adapter; it backs off and stays queued."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)
    runner.config = _telegram_config(tmp_path)
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    status_writes: list = []

    def _capture_status(name, **fields):
        status_writes.append((name, fields))

    runner._update_platform_runtime_status = _capture_status  # type: ignore[method-assign]
    cfg = runner.config.platforms[Platform.TELEGRAM]
    runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
        Platform.TELEGRAM, None, cfg, attempts=1, delay=0,
    )
    info = runner._failed_platforms[Platform.TELEGRAM]
    info["next_retry"] = time.monotonic() - 1  # due now
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)

    await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())

    # Still queued (the old code deleted it here), attempt bumped, retrying restamped.
    assert Platform.TELEGRAM in runner._failed_platforms
    assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == 2
    assert any(
        name == "telegram" and fields.get("platform_state") == "retrying"
        for name, fields in status_writes
    )


def test_failed_deferred_load_revives_on_retry(tmp_path, monkeypatch):
    """A failed deferred load parks its loader; retry re-queues it for one more attempt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = PlatformRegistry()
    calls: list = []

    def _loader():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated load overrun")
        registry.register(
            PlatformEntry(
                name="revive-target", label="Revive", adapter_factory=lambda cfg: object(),
                check_fn=lambda: True, source="plugin",
            ),
            scope=PlatformRegistry.current_scope_key(),
        )

    registry.register_deferred(
        "revive-target", _loader, scope=PlatformRegistry.current_scope_key()
    )
    assert registry.get("revive-target") is None  # first attempt fails
    assert registry.has_failed_load("revive-target") is True

    assert registry.retry_failed_load("revive-target") is True

    assert registry.get("revive-target") is not None  # second attempt heals
    assert registry.has_failed_load("revive-target") is False
    assert registry.retry_failed_load("revive-target") is False  # nothing left to revive


def test_plugin_deadline_extended_while_progress_lease_held(tmp_path, monkeypatch):
    """A live startup progress lease raises the plugin deadline; explicit 0 still disables."""
    import hermes_startup_watchdog as sw
    from hermes_cli import plugins_loader

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sw._reset_for_tests()
    try:
        assert plugins_loader._resolve_plugin_load_timeout() == pytest.approx(10.0)

        handle = sw.arm_startup_watchdog(timeout_s=300)
        assert handle is not None
        sw.report_startup_progress(600, phase="state_db_unclean_integrity_check")
        active, phase, _remaining = sw.startup_watchdog_lease_active()
        assert active is True
        assert phase == "state_db_unclean_integrity_check"

        assert plugins_loader._resolve_plugin_load_timeout() >= 60.0

        (tmp_path / "config.yaml").write_text("plugins:\n  load_timeout_seconds: 0\n")
        assert plugins_loader._resolve_plugin_load_timeout() == 0
    finally:
        sw._reset_for_tests()


def test_plugin_deadline_survives_disarm_inside_lease_window(tmp_path, monkeypatch):
    """The floor must survive ``record_startup() -> disarm`` ordering (#126356 review).

    Boot claims the progress lease (state.db integrity check / schema work during
    ``GatewayRunner.__init__``) and disarms the watchdog once the loop is live —
    but deferred plugin loads materialize afterwards, still inside the claimed
    window. The floor must apply to them, not just to loads while armed.
    """
    import time as _time

    import hermes_startup_watchdog as sw
    from hermes_cli import plugins_loader

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sw._reset_for_tests()
    try:
        handle = sw.arm_startup_watchdog(timeout_s=300)
        assert handle is not None
        # Models what record_startup()'s integrity check does on boot.
        sw.report_startup_progress(900, phase="state_db_unclean_integrity_check")
        assert plugins_loader._resolve_plugin_load_timeout() >= 60.0

        sw.disarm_startup_watchdog()
        active, phase, _remaining = sw.startup_watchdog_lease_active()
        assert active is True
        assert phase == "state_db_unclean_integrity_check"
        # The choke point every load passes through still grants the floor.
        assert plugins_loader._resolve_plugin_load_timeout() >= 60.0

        # Once the window lapses the deadline falls back to the default.
        base = _time.monotonic()
        monkeypatch.setattr(_time, "monotonic", lambda: base + 901.0)
        active, _, _ = sw.startup_watchdog_lease_active()
        assert active is False
        assert plugins_loader._resolve_plugin_load_timeout() == pytest.approx(10.0)
    finally:
        sw._reset_for_tests()


def test_gateway_boot_claims_lease_before_plugin_discovery(tmp_path, monkeypatch):
    """``GatewayRunner.__init__`` must hold the lease before the first sweep (#126356 review).

    The first ``discover_plugins()`` runs inside config load; the state.db leases
    only start after it. Without an early claim none of the eager loads ever see
    a live lease (reviewer: 53 loads, 0 saw one).
    """
    import hermes_startup_watchdog as sw
    from gateway import run as run_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sw._reset_for_tests()
    try:
        assert sw.arm_startup_watchdog(timeout_s=300) is not None
        order: list = []
        lease_seen_at_discover: list = []

        from hermes_cli import plugins as plugins_mod

        real_discover_and_load = plugins_mod.PluginManager.discover_and_load

        def _spy_discover(self, force: bool = False):
            active, _, _ = sw.startup_watchdog_lease_active()
            lease_seen_at_discover.append(bool(active))
            order.append("discover")
            return real_discover_and_load(self, force=force)

        real_report = sw.report_startup_progress

        def _spy_report(expected_s, phase: str = ""):
            order.append("lease")
            return real_report(expected_s, phase=phase)

        monkeypatch.setattr(
            plugins_mod.PluginManager, "discover_and_load", _spy_discover
        )
        monkeypatch.setattr(sw, "report_startup_progress", _spy_report)

        runner = run_mod.GatewayRunner.__new__(run_mod.GatewayRunner)
        # Run only the head of __init__ (early claim + config load) with the
        # remaining inits stubbed: full construction needs a live home/DB.
        for name in (
            "_init_runtime_settings", "_init_session_store", "_init_lifecycle_state",
            "_init_runtime_caches", "_init_startup_checks", "_init_session_db",
            "_init_registries_and_clocks",
        ):
            monkeypatch.setattr(runner, name, lambda: None)
        monkeypatch.setattr(runner, "_warn_if_docker_media_delivery_is_risky", lambda: None)
        run_mod.GatewayRunner.__init__(runner)
        assert "discover" in order, "expected a discovery sweep during config load"
        assert order.index("lease") < order.index("discover")
        assert lease_seen_at_discover and all(lease_seen_at_discover)
    finally:
        sw._reset_for_tests()


def _secondary_runner(tmp_path, monkeypatch):
    """Bare multiplex runner shaped like ``test_multiplex_hot_serve._runner``."""
    from gateway.run_profile_reconcile import profile_serve_signature

    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._profile_plugin_retry_pending = {}
    runner._profile_configs = {}
    runner._failed_platforms = {}
    runner._served_profile_homes = None
    runner._served_profile_signatures = None
    runner._profile_reconcile_lock = None
    runner._agent_cache = {}
    runner.pairing_store = MagicMock()
    runner.pairing_stores = {}
    runner._adapter_disconnect_timeout_secs = lambda: 0.5
    return runner, home


def _mk_secondary(home, name, env=""):
    d = home / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    (d / ".env").write_text(env, encoding="utf-8")
    return d


@pytest.mark.asyncio
async def test_secondary_missing_platform_retried_on_rescan_without_config_change(
    tmp_path, monkeypatch
):
    """A failed secondary platform load must retry on a later tick (#126356 review).

    The failure touches neither config.yaml nor .env, so the signature check
    alone leaves the profile in neither added nor changed and
    ``_apply_profile_changes`` early-returns without re-entering
    ``_start_one_profile_adapters`` — a permanently-``retrying`` platform.
    """
    from gateway.run_profile_reconcile import profile_serve_signature

    runner, home = _secondary_runner(tmp_path, monkeypatch)
    beta = _mk_secondary(home, "beta", "DISCORD_BOT_TOKEN=beta-token\n")
    monkeypatch.setattr(
        "gateway.status.live_gateway_pid_for_home", lambda h: None
    )
    runner._served_profile_homes = {"beta": beta}
    runner._served_profile_signatures = {"beta": profile_serve_signature(beta)}
    # Post-boot state: the boot sweep failed to materialize the platform and
    # recorded the pending entry (real code does this inside
    # ``_start_one_profile_adapters``).
    runner._profile_plugin_retry_pending = {"beta": {"discord"}}
    calls: list = []

    async def _start(profile_name, profile_home, claimed):
        calls.append(profile_name)
        runner._clear_secondary_missing_platform(profile_name, Platform.DISCORD)
        return 1

    runner._start_one_profile_adapters = _start  # type: ignore[method-assign]

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        first = await runner.reconcile_served_profiles(reason="watcher")
    assert first["rescanned"] == ["beta"], first
    assert calls == ["beta"]

    # Healed: no config change, no pending entry — the next tick stays quiet.
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        second = await runner.reconcile_served_profiles(reason="watcher")
    assert second["rescanned"] == []
    assert second["added"] == [] and second["removed"] == []
    assert calls == ["beta"]


@pytest.mark.asyncio
async def test_start_one_profile_adapters_notes_and_clears_missing(tmp_path, monkeypatch):
    """The real ``_start_one_profile_adapters`` records a missing platform for rescan
    and clears it once the retry heals (reviving a parked deferred loader first)."""
    runner, home = _secondary_runner(tmp_path, monkeypatch)
    beta = _mk_secondary(home, "beta", "DISCORD_BOT_TOKEN=beta-token\n")

    profile_cfg = SimpleNamespace(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="beta-token")}
    )

    async def _fake_load_config(profile_name, profile_home):
        return profile_cfg

    runner._load_secondary_profile_config = _fake_load_config  # type: ignore[method-assign]
    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *a: True)

    created = {"calls": 0}

    class _FakeAdapter:
        platform = Platform.DISCORD

    def _create(platform, cfg):
        created["calls"] += 1
        return None if created["calls"] == 1 else _FakeAdapter()

    runner._create_adapter = _create  # type: ignore[method-assign]
    runner._configure_profile_adapter = lambda *a, **k: None
    runner._connect_initial_adapter_with_timeout = (  # type: ignore[method-assign]
        lambda adapter, platform: asyncio.sleep(0, result=True)
    )
    runner._sync_voice_mode_state_to_adapter = lambda *a, **k: None
    runner._safe_adapter_disconnect = (  # type: ignore[method-assign]
        lambda adapter, platform: asyncio.sleep(0, result=None)
    )
    runner._schedule_secondary_profile_startup_reconnect = lambda *a, **k: None
    runner._adapter_credential_claim = staticmethod(lambda platform, adapter: None)
    runner._adapter_listener_claim = staticmethod(lambda platform, adapter: None)
    runner._refuse_duplicate_claim = lambda *a, **k: False
    statuses: list = []
    runner._update_platform_runtime_status = (  # type: ignore[method-assign]
        lambda name, **fields: statuses.append((name, fields))
    )

    revived: list = []
    from gateway import platform_registry as registry_mod

    real_retry = registry_mod.platform_registry.retry_failed_load

    def _spy_retry(name, *, scope=None):
        revived.append(name)
        return real_retry(name, scope=scope)

    monkeypatch.setattr(registry_mod.platform_registry, "retry_failed_load", _spy_retry)

    assert await runner._start_one_profile_adapters("beta", beta, {}) == 0
    assert runner._secondary_retry_pending_profiles() == {"beta"}
    assert any(
        name == "beta:discord" and fields.get("platform_state") == "retrying"
        for name, fields in statuses
    )

    assert await runner._start_one_profile_adapters("beta", beta, {}) == 1
    assert runner._secondary_retry_pending_profiles() == set()
    assert Platform.DISCORD in runner._profile_adapters["beta"]
    assert "discord" in revived
