"""Pairing WhatsApp adds one adapter without interrupting existing gateway work (LOCX-229)."""

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _runner(tmp_path, monkeypatch, *, connect):
    original = object()
    adapter = SimpleNamespace(
        platform=Platform.WHATSAPP, send_path_degraded=False, DEGRADED_STATUS_MESSAGE="",
        has_fatal_error=False, fatal_error_retryable=False, fatal_error_code=None,
        fatal_error_message=None,
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={}, multiplex_profiles=False)
    runner._running = True
    runner._draining = False
    runner._primary_profile_name = "default"
    runner.adapters = {Platform.TELEGRAM: original}
    runner._failed_platforms = {}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    runner._startup_parked_platforms = False
    runner._create_adapter = lambda platform, config: adapter
    runner._wire_adapter_handlers = lambda value: None
    runner._publish_primary_adapter = lambda platform, value: runner.adapters.__setitem__(platform, value)
    runner._update_platform_runtime_status = lambda *args, **kwargs: None
    runner._schedule_resume_pending_sessions = lambda **kwargs: None
    runner._connect_adapter_with_timeout = connect

    config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *args: True)
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: tmp_path)

    @asynccontextmanager
    async def scope(home):
        assert home == tmp_path
        yield

    monkeypatch.setattr("gateway.run._async_profile_runtime_scope", scope)
    from plugins.platforms.whatsapp.bridge_dependencies import whatsapp_activation_signature
    adapter._session_path = tmp_path / "whatsapp" / "session"
    adapter._activation_signature = whatsapp_activation_signature(tmp_path, adapter._session_path)
    async def directory(adapters):
        assert adapters[Platform.TELEGRAM] is original
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", directory)
    return runner, adapter, original


@pytest.mark.asyncio
async def test_pairing_activates_once_while_existing_adapter_stays_live(tmp_path, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def connect(adapter, platform):
        calls.append(platform)
        entered.set()
        await release.wait()
        return True

    runner, adapter, original = _runner(tmp_path, monkeypatch, connect=connect)
    first = asyncio.create_task(runner.activate_whatsapp("default"))
    await entered.wait()
    second = asyncio.create_task(runner.activate_whatsapp("default"))
    assert runner.adapters == {Platform.TELEGRAM: original}
    release.set()
    assert (await first)["active"] is True
    assert (await second)["already_active"] is True
    assert calls == [Platform.WHATSAPP]
    assert runner.adapters == {Platform.TELEGRAM: original, Platform.WHATSAPP: adapter}
    assert runner._running is True


@pytest.mark.asyncio
async def test_failed_whatsapp_activation_keeps_local_gateway_and_existing_adapter(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        adapter.has_fatal_error = True
        adapter.fatal_error_code = "whatsapp_npm_install_failed"
        adapter.fatal_error_message = "npm install failed"
        return False

    runner, _adapter, original = _runner(tmp_path, monkeypatch, connect=connect)
    runner._safe_adapter_disconnect = lambda *args: asyncio.sleep(0)
    runner._mark_platform_fatal = lambda *args: None
    states = []
    runner._update_runtime_status = states.append
    result = await runner.activate_whatsapp("default")
    assert result == {"active": False, "reason": "whatsapp_npm_install_failed"}
    assert runner.adapters == {Platform.TELEGRAM: original}
    assert runner._running is True
    assert states == ["degraded"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("other_fatal", "expected_state"), [
    (set(), "running"), ({"discord"}, "degraded"),
])
async def test_reactivation_clears_degraded_only_when_no_other_platform_is_fatal(
    tmp_path, monkeypatch, other_fatal, expected_state,
):
    async def connect(adapter, platform):
        return True

    runner, _adapter, _original = _runner(tmp_path, monkeypatch, connect=connect)
    runner._startup_parked_platforms = True
    monkeypatch.setattr("gateway.status_runtime_snapshot.primary_fatal_platforms_snapshot", lambda: other_fatal)
    states = []
    runner._update_runtime_status = states.append
    assert (await runner.activate_whatsapp("default"))["active"] is True
    assert runner._serving_state() == expected_state
    assert states == (["running"] if not other_fatal else [])


@pytest.mark.asyncio
async def test_pairing_during_startup_is_activated_after_startup(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        return True

    runner, adapter, original = _runner(tmp_path, monkeypatch, connect=connect)
    runner._running = False
    assert await runner.activate_whatsapp("default") == {"pending": True}
    assert runner.adapters == {Platform.TELEGRAM: original}
    runner._running = True
    runner._retain_background_task = lambda task: task
    runner._late_failure_callback = lambda label: lambda task: None
    tasks = runner._launch_pending_whatsapp_activations()
    assert len(tasks) == 1
    assert (await tasks[0])["active"] is True
    assert runner.adapters[Platform.WHATSAPP] is adapter
    assert runner._launch_pending_whatsapp_activations() == []


@pytest.mark.asyncio
async def test_secondary_pairing_during_drain_does_not_reconcile(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        pytest.fail("draining gateway must not connect")

    runner, _adapter, _original = _runner(tmp_path, monkeypatch, connect=connect)
    runner._draining = True
    runner.reconcile_served_profiles = lambda **kwargs: pytest.fail("must not rescan while draining")
    assert await runner.activate_whatsapp("work") == {
        "active": False, "reason": "gateway_stopping"}


@pytest.mark.asyncio
async def test_secondary_retry_is_reported_as_retrying(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        pytest.fail("reconcile stub owns connect")

    runner, _adapter, _original = _runner(tmp_path, monkeypatch, connect=connect)
    runner._profile_adapters = {"work": {}}
    runner._profile_failed_platforms = {"work": {Platform.WHATSAPP: object()}}
    runner._secondary_held_claims = {}
    runner._multiplex_on = lambda: True
    runner.served_profile_names = lambda: ["default", "work"]
    async def reconcile(**kwargs):
        assert kwargs["refresh_whatsapp"] == "work"
        return {}
    runner.reconcile_served_profiles = reconcile
    assert await runner.activate_whatsapp("work") == {
        "active": False, "pending": False, "retrying": True}


@pytest.mark.asyncio
async def test_repair_refreshes_queued_whatsapp_policy(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        pytest.fail("queued reconnect should own the connect")

    runner, _adapter, original = _runner(tmp_path, monkeypatch, connect=connect)
    old = PlatformConfig(enabled=True, extra={"dm_policy": "open"})
    new = PlatformConfig(enabled=True, extra={"dm_policy": "pairing"})
    runner.config.platforms[Platform.WHATSAPP] = old
    runner._failed_platforms[Platform.WHATSAPP] = {
        "config": old, "attempts": 1, "next_retry": float("inf"), "paused": False,
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig(
        platforms={Platform.WHATSAPP: new}))
    assert await runner.activate_whatsapp("default") == {"active": False, "retrying": True}
    queued = runner._failed_platforms[Platform.WHATSAPP]
    assert queued["config"] is new
    assert queued["next_retry"] < float("inf")
    assert queued["retry_now"] is True
    assert runner.adapters == {Platform.TELEGRAM: original}


@pytest.mark.asyncio
async def test_changed_env_policy_replaces_only_whatsapp(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        return True

    runner, first, telegram = _runner(tmp_path, monkeypatch, connect=connect)
    assert (await runner.activate_whatsapp("default"))["active"]
    (tmp_path / ".env").write_text("WHATSAPP_ALLOWED_USERS=12345\n", encoding="utf-8")
    replacement = SimpleNamespace(**vars(first))
    runner._create_adapter = lambda platform, config: replacement
    torn_down = []
    async def teardown(adapter, platform):
        torn_down.append(adapter)
    runner._bounded_adapter_teardown = teardown
    assert (await runner.activate_whatsapp("default"))["active"]
    assert torn_down == [first]
    assert runner.adapters == {Platform.TELEGRAM: telegram, Platform.WHATSAPP: replacement}


@pytest.mark.asyncio
async def test_repeated_save_with_same_env_content_keeps_adapter(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("WHATSAPP_ALLOWED_USERS=12345\n", encoding="utf-8")
    async def connect(adapter, platform):
        return True

    runner, adapter, _telegram = _runner(tmp_path, monkeypatch, connect=connect)
    assert (await runner.activate_whatsapp("default"))["active"]
    env.write_text("WHATSAPP_ALLOWED_USERS=12345\n", encoding="utf-8")
    runner._create_adapter = lambda *args: pytest.fail("unchanged Save should not reconnect")
    assert await runner.activate_whatsapp("default") == {"active": True, "already_active": True}
    assert runner.adapters[Platform.WHATSAPP] is adapter


@pytest.mark.asyncio
async def test_retryable_activation_restarts_dead_reconnect_watcher(tmp_path, monkeypatch):
    async def connect(adapter, platform):
        return False

    runner, adapter, telegram = _runner(tmp_path, monkeypatch, connect=connect)
    adapter.fatal_error_retryable = True
    runner._safe_adapter_disconnect = lambda *args: asyncio.sleep(0)
    runner._startup_retry_entry = lambda *args: {"config": args[-1]}
    calls = []
    runner._ensure_reconnect_watcher_running = lambda: calls.append(True)
    assert await runner.activate_whatsapp("default") == {"active": False, "retrying": True}
    assert calls == [True]
    assert runner.adapters == {Platform.TELEGRAM: telegram}


@pytest.mark.asyncio
async def test_secondary_repair_releases_old_pairing_only(tmp_path, monkeypatch):
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
    from plugins.platforms.whatsapp.bridge_dependencies import (
        stamp_whatsapp_pairing, whatsapp_activation_signature)

    runner_config = GatewayConfig(platforms={})
    config = PlatformConfig(enabled=True, extra={
        "group_sessions_per_user": runner_config.group_sessions_per_user,
        "thread_sessions_per_user": getattr(runner_config, "thread_sessions_per_user", False),
    })
    old = SimpleNamespace(_session_path=tmp_path / "session", config=config)
    old._activation_signature = whatsapp_activation_signature(tmp_path, old._session_path)
    telegram = object()
    runner = object.__new__(GatewayRunner)
    runner.config = runner_config
    runner._profile_adapters = {"work": {Platform.WHATSAPP: old, Platform.TELEGRAM: telegram}}
    runner._profile_failed_platforms = {}
    torn_down = []
    async def teardown(adapter, platform, **kwargs):
        torn_down.append((adapter, platform))
    runner._bounded_adapter_teardown = teardown
    from contextlib import contextmanager
    @contextmanager
    def scope(home, hydrate_secrets=False):
        yield
    monkeypatch.setattr("gateway.run._profile_runtime_scope", scope)
    (old._session_path).mkdir()
    (old._session_path / "creds.json").write_text('{"id":"new"}', encoding="utf-8")
    await GatewayProfileReconcileMixin._refresh_secondary_whatsapp(
        runner, "work", tmp_path, GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}))
    assert torn_down == []  # routine Baileys credential writes are not a new pairing
    stamp_whatsapp_pairing(old._session_path)
    await GatewayProfileReconcileMixin._refresh_secondary_whatsapp(
        runner, "work", tmp_path, GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}))
    assert torn_down == [(old, Platform.WHATSAPP)]
    assert runner._profile_adapters["work"] == {Platform.TELEGRAM: telegram}


@pytest.mark.asyncio
async def test_timed_out_install_worker_does_not_overlap_retry(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp import adapter as whatsapp
    from plugins.platforms.whatsapp.bridge_dependencies import stamp_bridge_dependencies

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "node_modules").mkdir()
    (bridge / "package.json").write_text('{"dependencies":{}}', encoding="utf-8")
    first = whatsapp.WhatsAppAdapter(PlatformConfig(extra={"session_path": str(tmp_path / "s1")}))
    second = whatsapp.WhatsAppAdapter(PlatformConfig(extra={"session_path": str(tmp_path / "s2")}))
    entered, release = threading.Event(), threading.Event()
    calls = []
    def install(_directory):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        stamp_bridge_dependencies(bridge)
        return True
    monkeypatch.setattr(first, "_install_bridge_deps", install)
    monkeypatch.setattr(second, "_install_bridge_deps", install)
    worker = asyncio.create_task(asyncio.to_thread(first._ensure_bridge_deps, bridge))
    assert await asyncio.to_thread(entered.wait, 5)
    worker.cancel()  # gateway connect timeout cancels the coroutine, not the npm thread
    with pytest.raises(asyncio.CancelledError):
        await worker
    retry_started = threading.Event()
    def retry_install():
        retry_started.set()
        return second._ensure_bridge_deps(bridge)
    retry = asyncio.create_task(asyncio.to_thread(retry_install))
    assert await asyncio.to_thread(retry_started.wait, 5)
    assert len(calls) == 1
    release.set()
    assert await retry is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_bridge_dependency_install_runs_off_gateway_loop(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp import adapter as whatsapp

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    instance = whatsapp.WhatsAppAdapter(PlatformConfig(extra={
        "bridge_script": str(bridge / "bridge.js"), "session_path": str(tmp_path / "session"),
    }))
    monkeypatch.setattr(whatsapp, "find_node_executable", lambda name: "node")
    monkeypatch.setattr(instance, "_preflight", lambda: True)
    monkeypatch.setattr(instance, "_acquire_platform_lock", lambda *args: True)
    monkeypatch.setattr(instance, "_release_platform_lock", lambda: None)
    on_loop = threading.get_ident()
    called_from = []

    def install(_directory):
        called_from.append(threading.get_ident())
        return False

    monkeypatch.setattr(instance, "_ensure_bridge_deps", install)
    assert await instance.connect() is False
    assert len(called_from) == 1
    assert called_from[0] != on_loop
