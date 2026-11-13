"""Secondary-profile platforms that boot with no adapter (a plugin load that failed or overran) and heal later,
and the bot-claim holds every secondary takes while it is not connected (boot, teardown, queued reconnect, unserve)."""
import logging
import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_agent.gateway.run as gateway_run
from hermes_agent.gateway.run_adapters_recovery import _adapter_unavailable_message
from tests.waiting import async_wait_until
from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from tests.gateway.test_multiplex_adapter_registry import (
    _install_secondary_reconnect_context,
    _SecondaryRecoveryAdapter,
    _secondary_recovery_runner,
)


class _PluginPlatformRecoveryAdapter(_SecondaryRecoveryAdapter):
    """A secondary adapter on a platform that is NOT a builtin: the one shape whose creation can
    return None and still heal (a bundled plugin whose deferred load died transiently)."""

    platform = Platform("irc")


def _install_plugin_platform_context(monkeypatch, runner, adapter):
    """Point the secondary-restart fixtures at that platform + credential."""
    _install_secondary_reconnect_context(monkeypatch, runner, adapter)
    monkeypatch.setattr(
        "hermes_agent.gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform("irc"): PlatformConfig(enabled=True, token="plugin-token")
            },
        ),
    )


def _dispose_platform_registration(monkeypatch, name: str) -> None:
    """Model the state #137268(d) rules in: the plugin's load overran its deadline, so the plugin
    manager disposed the lease and the registry holds NO registration for the platform at boot."""
    from hermes_agent.gateway.platform_registry import platform_registry

    real_is_registered = platform_registry.is_registered
    monkeypatch.setattr(
        platform_registry,
        "is_registered",
        lambda candidate: False if candidate == name else real_is_registered(candidate),
    )


def _install_none_buildable_context(monkeypatch, runner, platform):
    """Secondary restart for a platform whose creation returns None and cannot heal without a
    config change (a builtin whose probe fails).

    The platform must be a real builtin (``_BUILTIN_ADAPTERS``: API_SERVER, WEBHOOK,
    MSGRAPH_WEBHOOK): a bundled plugin platform such as SIGNAL is exactly the case that now
    classifies as healable once its registration is gone.
    """
    _install_secondary_reconnect_context(monkeypatch, runner, _SecondaryRecoveryAdapter())
    monkeypatch.setattr(
        "hermes_agent.gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={
                platform: PlatformConfig(enabled=True, token="probe-token")
            },
        ),
    )


class TestSecondaryAdapterUnavailableRecovery:
    """A secondary platform whose adapter creation returns None reaches the same reconnect slot a
    connect failure gets, when the platform can still heal. The startup sweep runs once per process,
    so dropping the platform there left it dead until the gateway was restarted by hand (#137268)."""

    @pytest.mark.asyncio
    async def test_healable_none_is_queued_and_installed_on_a_later_tick(self, monkeypatch):
        runner = _secondary_recovery_runner()
        replacement = _PluginPlatformRecoveryAdapter()
        _install_plugin_platform_context(monkeypatch, runner, replacement)
        _dispose_platform_registration(monkeypatch, "irc")
        built = []

        def create(platform, config):
            built.append(platform)
            # Unusable on the startup sweep AND on the first reconnect attempt; importable after.
            return replacement if len(built) > 2 else None

        monkeypatch.setattr(runner, "_create_adapter", create)
        rearm = AsyncMock(return_value=None)
        monkeypatch.setattr(runner, "_rearm_failed_plugin_load", rearm)
        monkeypatch.setattr(
            runner, "_connect_adapter_with_timeout", AsyncMock(return_value=True)
        )
        real_sleep = asyncio.sleep
        ticks = []

        async def fast_backoff(delay, *args, **kwargs):
            ticks.append(delay)
            if len(ticks) >= 3:
                runner._running = False
            await real_sleep(0 if delay > 0.05 else delay)

        monkeypatch.setattr(asyncio, "sleep", fast_backoff)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/reviewer", {}
        )

        assert connected == 0
        assert runner._profile_adapters.get("reviewer", {}) == {}
        # The platform is parked for reconnect instead of being dropped for the process lifetime.
        bridge = list(runner._background_tasks)
        assert len(bridge) == 1
        await asyncio.wait_for(bridge[0], timeout=5.0)
        task = runner._profile_failed_platforms["reviewer"][Platform("irc")]
        await asyncio.wait_for(task, timeout=5.0)
        # The failed load is re-armed, the next attempt builds the plugin, and the slot clears.
        assert rearm.await_count == 1
        assert len(built) == 3
        assert runner._profile_adapters["reviewer"][Platform("irc")] is replacement
        assert Platform("irc") not in runner._profile_failed_platforms.get("reviewer", {})

    @pytest.mark.asyncio
    async def test_unhealable_none_is_not_queued(self, monkeypatch):
        """A builtin whose probe fails needs a config change: no slot, no retry storm."""
        runner = _secondary_recovery_runner()
        built = []
        _install_none_buildable_context(monkeypatch, runner, Platform.MSGRAPH_WEBHOOK)
        monkeypatch.setattr(
            runner, "_create_adapter", lambda platform, config: built.append(platform) or None
        )

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/reviewer", {}
        )

        assert connected == 0
        assert built == [Platform.MSGRAPH_WEBHOOK]
        assert list(runner._background_tasks) == []
        assert runner._profile_failed_platforms == {}

    @pytest.mark.asyncio
    async def test_attempt_classifies_healable_none_as_retryable(self, monkeypatch):
        """``(None, False)`` keeps the slot; ``(None, None)`` still gives up."""
        runner = _secondary_recovery_runner()
        _install_none_buildable_context(monkeypatch, runner, Platform.MSGRAPH_WEBHOOK)
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: None)

        assert await runner._secondary_reconnect_attempt(
            "reviewer", Platform.MSGRAPH_WEBHOOK
        ) == (None, None)

        _install_plugin_platform_context(
            monkeypatch, runner, _PluginPlatformRecoveryAdapter()
        )
        _dispose_platform_registration(monkeypatch, "irc")
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: None)
        assert await runner._secondary_reconnect_attempt(
            "reviewer", Platform("irc")
        ) == (None, False)

    @pytest.mark.asyncio
    async def test_queued_slot_stops_a_second_build(self, monkeypatch):
        """The queued entry is the claim the runtime re-scan defers to: one adapter, never two."""
        runner = _secondary_recovery_runner(running=False)
        _install_plugin_platform_context(
            monkeypatch, runner, _PluginPlatformRecoveryAdapter()
        )
        built = []
        monkeypatch.setattr(
            runner, "_create_adapter", lambda platform, config: built.append(platform) or None
        )
        runner._profile_failed_platforms = {
            "reviewer": {Platform("irc"): MagicMock(name="reconnect-task")}
        }

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/reviewer", {}
        )

        assert connected == 0
        assert built == []

    @pytest.mark.asyncio
    async def test_plugin_load_rearm_is_capped_and_off_loop(self, monkeypatch):
        import hermes_cli.plugins as plugins_mod

        runner = _secondary_recovery_runner()
        _install_plugin_platform_context(
            monkeypatch, runner, _PluginPlatformRecoveryAdapter()
        )
        rearmed, on_loop = [], []
        manager = MagicMock()
        manager.rearm_failed_platform.side_effect = (
            lambda name: rearmed.append(name)
            or on_loop.append(threading.current_thread() is threading.main_thread())
            or True
        )
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
        queue_info = {}

        for _ in range(5):
            await runner._rearm_failed_plugin_load(Platform("irc"), queue_info)

        assert rearmed == ["irc"] * 3  # a permanently broken plugin stops being re-imported
        assert queue_info["load_rearms"] == 3
        assert on_loop and not any(on_loop)  # the re-arm never blocks the event loop

    @pytest.mark.asyncio
    async def test_classification_load_and_rearm_run_in_the_secondary_scope(self, monkeypatch):
        """Plugin registrations and plugin managers are per profile home, and the launch profile can hold
        a healthy registration of the platform whose load failed only in the secondary. The startup
        classification, the attempt's registry load (off the loop) and the re-arm all run in the
        secondary's scope."""
        import contextvars

        import hermes_cli.plugins as plugins_mod
        from hermes_agent.gateway.platform_registry import platform_registry

        scope = contextvars.ContextVar("probe_scope", default="launch")

        @contextmanager
        def recording_scope(profile_home, *, hydrate_secrets=True):
            token = scope.set(str(profile_home))
            try:
                yield
            finally:
                scope.reset(token)

        runner = _secondary_recovery_runner()
        _install_plugin_platform_context(monkeypatch, runner, _PluginPlatformRecoveryAdapter())
        monkeypatch.setattr(gateway_run, "_profile_runtime_scope", recording_scope)
        monkeypatch.setattr(runner, "_routed_profile_home", lambda name: Path("/profiles") / name)
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: None)
        seen = []
        real_is_registered, real_get = platform_registry.is_registered, platform_registry.get

        def is_registered(name):
            if name != "irc":
                return real_is_registered(name)
            seen.append(("classify", scope.get()))
            return scope.get() == "launch"  # only the launch profile's load of this plugin succeeded

        def get(name):
            if name != "irc":
                return real_get(name)
            seen.append(("load", scope.get(), threading.current_thread() is threading.main_thread()))

        monkeypatch.setattr(platform_registry, "is_registered", is_registered)
        monkeypatch.setattr(platform_registry, "get", get)
        manager = MagicMock()
        manager.rearm_failed_platform.side_effect = lambda name: seen.append(("rearm", scope.get())) or True
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

        await runner._start_one_profile_adapters("reviewer", Path("/profiles/reviewer"), {})
        for bridge in list(runner._background_tasks):
            bridge.cancel()
        # Healable in ITS scope (the launch scope holds a registration), so it was queued.
        assert [e for e in seen if e[0] == "classify"] == [("classify", "/profiles/reviewer")]
        assert len(runner._background_tasks) == 1

        async def stop_after_backoff(_delay):
            runner._running = False

        monkeypatch.setattr(asyncio, "sleep", stop_after_backoff)
        seen.clear()
        await runner._run_secondary_profile_reconnect("reviewer", Platform("irc"))

        assert ("load", "/profiles/reviewer", False) in seen  # the import a re-arm enables runs off the loop
        assert [e for e in seen if e[0] in ("classify", "rearm")] == [
            ("classify", "/profiles/reviewer"),
            ("rearm", "/profiles/reviewer"),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("owned_elsewhere", [False, True])
    async def test_healed_platform_passes_the_startup_ownership_check(self, monkeypatch, owned_elsewhere):
        """A platform queued without an adapter never met the startup duplicate-credential check, so the
        attempt that finally builds it runs that check before connect() and records the bot's owner."""
        runner = _secondary_recovery_runner()
        healed = _PluginPlatformRecoveryAdapter()
        healed.token = "shared-bot-token"
        _install_plugin_platform_context(monkeypatch, runner, healed)
        if owned_elsewhere:
            owner = _PluginPlatformRecoveryAdapter()
            owner.token = "shared-bot-token"
            runner._profile_adapters["owner"] = {Platform("irc"): owner}
        connect = AsyncMock(return_value=True)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)

        adapter, success = await runner._secondary_reconnect_attempt("reviewer", Platform("irc"))

        owned = (getattr(runner, "_profile_bot_platforms", None) or {}).get("reviewer", set())
        if owned_elsewhere:
            assert (adapter, success) == (None, None)
            connect.assert_not_awaited()
            assert owned == set()
        else:
            assert (adapter, success) == (healed, True)
            assert owned == {Platform("irc")}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("heals", [True, False])
    async def test_unbuilt_platform_status_names_the_cause(self, monkeypatch, heals):
        """``hermes status`` shows an unbuilt secondary platform as adapter_unavailable: retrying when its plugin
        can still load, fatal when only a config change helps."""
        runner = _secondary_recovery_runner()
        _install_plugin_platform_context(monkeypatch, runner, _PluginPlatformRecoveryAdapter())
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: None)
        monkeypatch.setattr(runner, "_adapter_may_heal", lambda platform, config: heals)
        writes = _record_status_writes(monkeypatch, runner)

        await runner._start_one_profile_adapters("reviewer", Path("/profiles/reviewer"), {})
        for bridge in list(runner._background_tasks):
            bridge.cancel()

        assert writes == [("reviewer:irc", {
            "platform_state": "retrying" if heals else "fatal", "error_code": "adapter_unavailable",
            "error_message": _adapter_unavailable_message(Platform("irc"), retrying=heals), "needs_attention": True,
        })]

    @pytest.mark.asyncio
    async def test_attention_flag_keeps_the_unavailable_hint(self, monkeypatch):
        """The attention flag rewrites the status entry without an error code; the tick it fires on puts the
        adapter_unavailable hint back, and later ticks write nothing more."""
        runner = _secondary_recovery_runner()
        _install_plugin_platform_context(monkeypatch, runner, _PluginPlatformRecoveryAdapter())
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: None)
        monkeypatch.setattr(runner, "_adapter_may_heal", lambda platform, config: True)
        monkeypatch.setattr(runner, "_rearm_failed_plugin_load", AsyncMock())
        monkeypatch.setattr(gateway_run, "_reconnect_needs_attention", lambda info, now: True)
        writes = _record_status_writes(monkeypatch, runner)
        ticks = []

        async def stop_after_two_ticks(_delay):
            ticks.append(_delay)
            runner._running = len(ticks) < 2

        monkeypatch.setattr(asyncio, "sleep", stop_after_two_ticks)
        await runner._run_secondary_profile_reconnect("reviewer", Platform("irc"))

        assert len(ticks) == 2
        assert [kw.get("error_code") for _key, kw in writes] == [None, "adapter_unavailable"]  # flag, then hint
        assert writes[1][1]["platform_state"] == "retrying"
        assert writes[1][1]["error_message"] == _adapter_unavailable_message(Platform("irc"), retrying=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cause", ["disabled", "no credential", "duplicate", "fatal"])
    async def test_never_built_platform_that_gives_up_drops_its_bot_ownership(self, monkeypatch, cause):
        """A never-built platform that gives up for good stops claiming the bot (cron and the delivery drain
        read _profile_bot_platforms), and a config reason replaces its retrying status with the reason."""
        runner, _create = _shared_token_runner(monkeypatch)
        runner._profile_bot_platforms = {"reviewer": {Platform("irc")}}  # an earlier tick's claim
        config = PlatformConfig(enabled=cause != "disabled", token="plugin-token")
        monkeypatch.setattr("hermes_agent.gateway.config.load_gateway_config",
                            lambda: GatewayConfig(multiplex_profiles=True, platforms={Platform("irc"): config}))
        monkeypatch.setattr(gateway_run, "_platform_has_bot_credential",
                            lambda platform, cfg: cause != "no credential")
        if cause == "duplicate":
            runner._profile_adapters = {"owner": {Platform("irc"): _create(Platform("irc"), None)}}
        if cause == "fatal":  # claimed, then a non-retryable fatal on connect
            async def fatal_connect(adapter, platform, **kwargs):
                adapter.has_fatal_error, adapter.fatal_error_retryable = True, False
                return False

            monkeypatch.setattr(runner, "_connect_adapter_with_timeout", fatal_connect)
        writes = _record_status_writes(monkeypatch, runner)

        async def no_retry(_delay):  # a regression that keeps retrying ends here instead of hanging
            runner._running = False

        monkeypatch.setattr(asyncio, "sleep", no_retry)

        await runner._run_secondary_profile_reconnect("reviewer", Platform("irc"))

        assert runner._profile_bot_platforms["reviewer"] == set()
        reason = {"disabled": "disabled in config", "no credential": "no bot credential configured"}.get(cause)
        if reason:
            assert writes[-1] == ("reviewer:irc", {"platform_state": "disabled", "error_code": None,
                                                   "error_message": reason, "needs_attention": False})
        elif cause == "duplicate":
            assert (writes[-1][1]["platform_state"], writes[-1][1]["error_code"]) == ("fatal", "duplicate_credential")

    @pytest.mark.asyncio
    async def test_concurrent_heals_and_retry_ticks_share_one_bot(self, monkeypatch):
        """A healing attempt holds its bot from the ownership check across the connect await: a second
        never-built profile with the same token is refused meanwhile, while the same task's next tick is not
        refused by its own hold."""
        runner, _create = _shared_token_runner(monkeypatch)
        outcomes = iter([False, True])
        between_ticks = []

        async def flaky_connect(adapter, platform, **kwargs):
            return next(outcomes)

        async def two_ticks():
            await runner._secondary_reconnect_attempt("c", Platform("irc"))
            between_ticks.append(set(runner._held_claim_owners().values()))
            return await runner._secondary_reconnect_attempt("c", Platform("irc"))

        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", flaky_connect)
        assert (await asyncio.ensure_future(two_ticks()))[1] is True
        assert between_ticks == [{"c"}]
        await async_wait_until(lambda: not runner._held_claim_owners())  # released with its task

        release, connects = asyncio.Event(), []

        async def slow_connect(adapter, platform, **kwargs):
            connects.append(adapter)
            await release.wait()
            return True

        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", slow_connect)
        first = asyncio.ensure_future(runner._secondary_reconnect_attempt("a", Platform("irc")))
        await async_wait_until(lambda: connects)
        # Bounded: without the hold "b" would also block in connect().
        assert await asyncio.wait_for(runner._secondary_reconnect_attempt("b", Platform("irc")), 5) == (None, None)
        release.set()
        assert (await first)[1] is True
        assert len(connects) == 1

    @pytest.mark.asyncio
    async def test_queued_reconnect_holds_its_bot_until_it_ends(self, monkeypatch):
        """A reconnect queued with its failed predecessor still owns the bot: a healing profile and runtime
        admission both see it. Cancelled before it ever ran (profile unserved), it lets go."""
        runner, create = _shared_token_runner(monkeypatch)
        connects = _record_connects(monkeypatch, runner)
        parked = _park_reconnects(monkeypatch, runner)
        runner._schedule_secondary_profile_reconnect("a", Platform("irc"), create(Platform("irc"), None))

        assert await runner._secondary_reconnect_attempt("b", Platform("irc")) == (None, None)
        assert connects == []
        assert set(runner._current_claim_owners().values()) == {"a"}

        queued = runner._profile_failed_platforms["a"].pop(Platform("irc"))
        queued.cancel()
        await asyncio.gather(queued, return_exceptions=True)
        await async_wait_until(lambda: not runner._held_claim_owners())
        parked.set()

    @pytest.mark.asyncio
    async def test_reconnect_holds_the_token_it_has_now(self, monkeypatch):
        """A reconnect with a predecessor re-holds what its rebuilt adapter claims: a token changed since it was
        queued must not keep the old token reserved."""
        runner, create = _shared_token_runner(monkeypatch)
        _record_connects(monkeypatch, runner, result=False)
        runner._hold_secondary_claims("a", Platform("irc"), None, (("credential", "irc", "old-token"), None))
        rebuilt = create(Platform("irc"), None)
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: rebuilt)

        async def one_tick():  # read the hold before its task ends (the end releases it)
            await runner._secondary_reconnect_attempt("a", Platform("irc"), create(Platform("irc"), None))
            return runner._secondary_held_claims[("a", Platform("irc"))][1]

        assert await asyncio.ensure_future(one_tick()) == runner._adapter_claims(Platform("irc"), rebuilt)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("unserved", [False, True])
    async def test_fatal_teardown_holds_its_bot(self, monkeypatch, unserved):
        """A fatal secondary holds its bot across the disconnect await (a healing profile is refused, and a
        re-scan of its own profile does not rebuild it), then hands the hold to its queued reconnect. A profile
        unserved meanwhile gets no reconnect and lets go."""
        runner, create = _shared_token_runner(monkeypatch)
        connects = _record_connects(monkeypatch, runner)
        parked = _park_reconnects(monkeypatch, runner)
        writes = _record_status_writes(monkeypatch, runner)
        torn_down = asyncio.Event()

        async def slow_disconnect(adapter, platform):
            await torn_down.wait()

        monkeypatch.setattr(runner, "_safe_adapter_disconnect", slow_disconnect)
        failed = create(Platform("irc"), None)
        runner._profile_adapters = {"a": {Platform("irc"): failed}}
        teardown = asyncio.ensure_future(runner._handle_profile_adapter_fatal_error("a", Platform("irc"), failed))
        await async_wait_until(lambda: not runner._profile_adapters["a"])  # popped; disconnect pending

        assert set(runner._held_claim_owners().values()) == {"a"}
        assert await runner._secondary_reconnect_attempt("b", Platform("irc")) == (None, None)
        assert await runner._start_one_profile_adapters("a", Path("/profiles/a"), {}) == 0
        assert not any(kw.get("error_code") == "duplicate_credential" and key == "a:irc" for key, kw in writes)
        if unserved:
            runner._profile_adapters.pop("a")
        torn_down.set()
        await teardown

        assert connects == []
        if unserved:
            assert runner._profile_failed_platforms == {} and runner._held_claim_owners() == {}
        else:
            assert set(runner._held_claim_owners().values()) == {"a"}  # now the queued reconnect's
            assert Platform("irc") in runner._profile_failed_platforms["a"]
        parked.set()
        await async_wait_until(lambda: not runner._held_claim_owners())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resume", ["during_unserve", "during_reserve_connect", "unserve_cancelled"]
    )
    async def test_unserve_retires_a_tearing_down_platform(self, monkeypatch, tmp_path, caplog, resume):
        """While unserve awaits its reconnects' cancellation the profile's adapters stay live owners and a fatal
        teardown finishing meanwhile queues nothing; each adapter then holds its bot until its own teardown. A
        later teardown spares a re-serve's hold. An unserve cancelled mid-wait leaves the profile served."""
        from hermes_agent.gateway.runtime_ownership import process_ownership

        runner, create = _shared_token_runner(monkeypatch)
        torn_down, reserve_connecting, owners_during_cancel, finished_in_wait = asyncio.Event(), asyncio.Event(), [], []

        async def slow_disconnect(adapter, platform):
            if adapter is failed:
                await torn_down.wait()

        async def reconnect_until_cancelled(profile_name, platform, predecessor=None):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                owners_during_cancel.append(runner._current_claim_owners())
                runner._profile_adapters.setdefault("a", {})[Platform("telegram")] = late  # installed as cancelled
                if resume == "during_unserve":
                    torn_down.set()  # the teardown finishes while unserve awaits this cancellation
                    await asyncio.sleep(0)
                    finished_in_wait.append(teardown.done())
                if resume == "unserve_cancelled" and platform == Platform("telegram"):
                    await asyncio.Event().wait()  # never finishes cancelling: the unserve itself is cancelled
                raise

        owners_during_reserve = []

        async def reserve_connect(adapter, platform):
            reserve_connecting.set()
            await torn_down.wait()  # the old teardown finishes while this connect is pending
            await teardown
            owners_during_reserve.append((runner._secondary_held_claims.get(("a", Platform("irc"))) or (None,))[0])
            return True

        monkeypatch.setattr(runner, "_safe_adapter_disconnect", slow_disconnect)
        torn = []

        async def bounded_teardown(adapter, platform, profile=None):
            owners = runner._current_claim_owners()
            torn.append((adapter, owners.get(runner._adapter_claims(platform, adapter)[0]),
                         owners.get(runner._adapter_claims(Platform("slack"), live)[0])))

        monkeypatch.setattr(runner, "_bounded_adapter_teardown", bounded_teardown)
        monkeypatch.setattr(runner, "_run_secondary_profile_reconnect", reconnect_until_cancelled)
        monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", reserve_connect)
        monkeypatch.setattr("hermes_agent.hermes_state_registry.close_all_under", lambda directory: 0)
        home = tmp_path / "a"
        process_ownership.reserve([home])
        runner._served_profile_homes, runner._served_profile_signatures = {"a": home}, {}
        runner._agent_cache, runner._evict_cached_agent = {}, lambda key: None
        runner.config._runtime_profile_homes = (("a", str(home)),)
        runner._adapter_disconnect_timeout_secs = lambda: 5
        runner._schedule_secondary_profile_reconnect("a", Platform("telegram"))  # another platform, queued
        failed, live, late = (create(Platform("irc"), None) for _ in range(3))
        runner._profile_adapters = {"a": {Platform("irc"): failed, Platform("slack"): live}}
        teardown = asyncio.ensure_future(runner._handle_profile_adapter_fatal_error("a", Platform("irc"), failed))
        await async_wait_until(lambda: Platform("irc") not in runner._profile_adapters["a"])
        assert set(runner._held_claim_owners().values()) == {"a"}
        try:
            if resume == "unserve_cancelled":
                unserve = asyncio.ensure_future(runner._unserve_profile("a", home))
                await async_wait_until(lambda: owners_during_cancel)
                unserve.cancel()
                await asyncio.gather(unserve, return_exceptions=True)
                assert runner._profile_adapters["a"] == {Platform("slack"): live, Platform("telegram"): late}
                assert set(runner._secondary_held_claims) == {("a", Platform("irc"))}  # the fatal teardown's own
                assert torn == [] and not runner._retiring_profiles
                torn_down.set()
                await teardown  # the profile is still served: its fatal platform is queued again
            else:
                with caplog.at_level(logging.INFO, logger="hermes_agent.gateway.run_profile_reconcile"):
                    await runner._unserve_profile("a", home)
                assert "2 adapter(s) stopped" in caplog.text
                live_claim = runner._adapter_claims(Platform("slack"), live)[0]
                assert [owners.get(live_claim) for owners in owners_during_cancel] == ["a"]  # still held
                assert torn == [(live, "a", "a"), (late, "a", None)]  # a late install too; each holds until its own
                assert finished_in_wait == ([True] if resume == "during_unserve" else [])
                if resume == "during_reserve_connect":
                    reserve = asyncio.ensure_future(runner._start_one_profile_adapters("a", Path("/profiles/a"), {}))
                    await asyncio.wait_for(reserve_connecting.wait(), 5)  # skipped for a stale hold: fails, not hangs
                    torn_down.set()
                    assert await reserve == 1
                    assert owners_during_reserve == [runner._profile_adapters["a"][Platform("irc")]]  # new hold kept
        finally:
            torn_down.set()
            await teardown
            leftover = runner._profile_failed_platforms.pop("a", None) or {}
            stray = {*leftover.values(), *runner._background_tasks}
            for task in stray:
                task.cancel()
            if stray:
                await asyncio.wait(stray, timeout=5)
            process_ownership.release(home)

        # no reconnect queued for the retired profile; a still-served one gets its fatal platform back
        assert set(leftover) == ({Platform("irc")} if resume == "unserve_cancelled" else set())
        assert runner._held_claim_owners() == {}

    @pytest.mark.asyncio
    async def test_a_retiring_profile_is_queued_no_reconnect(self, monkeypatch):
        """Whoever asks (a fatal teardown, a startup handoff), a profile being unserved gets no reconnect."""
        runner, create = _shared_token_runner(monkeypatch)
        runner._retiring_profiles = {"a"}
        runner._schedule_secondary_profile_reconnect("a", Platform("irc"), create(Platform("irc"), None))
        assert not runner._profile_failed_platforms.get("a") and runner._held_claim_owners() == {}

    @pytest.mark.asyncio
    async def test_runtime_admission_reads_current_owners(self, monkeypatch):
        """A hot-add checks owners as of its check, not the caller's snapshot (taken before the healed "a" was
        installed), and holds its own claim across the connect await so a heal meanwhile is refused."""
        runner, create = _shared_token_runner(monkeypatch)
        runner._profile_adapters = {"a": {Platform("irc"): create(Platform("irc"), None)}}
        initial = _record_connects(monkeypatch, runner, initial=True)
        assert await runner._start_one_profile_adapters("b", Path("/profiles/b"), {}) == 0
        assert initial == []

        runner._profile_adapters = {}
        release = asyncio.Event()

        async def slow_initial_connect(adapter, platform):
            initial.append(adapter)
            await release.wait()
            return True

        monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", slow_initial_connect)
        hot_add = asyncio.ensure_future(runner._start_one_profile_adapters("b", Path("/profiles/b"), {}))
        await async_wait_until(lambda: initial)
        assert set(runner._held_claim_owners().values()) == {"b"}
        assert await runner._secondary_reconnect_attempt("c", Platform("irc")) == (None, None)
        release.set()
        assert await hot_add == 1
        assert runner._held_claim_owners() == {}

    @pytest.mark.asyncio
    async def test_boot_sweep_respects_a_failed_connects_hold(self, monkeypatch):
        """At boot a connect that failed holds its bot before the runner is live: a later profile in the same
        sweep is refused. A shutdown before the handoff lets go."""
        runner, create = _shared_token_runner(monkeypatch)
        runner._running, runner._shutdown_event = False, asyncio.Event()
        runner._schedule_secondary_profile_startup_reconnect("a", Platform("irc"), create(Platform("irc"), None))
        built_before = len(create.built)
        assert set(runner._held_claim_owners().values()) == {"a"}
        initial = _record_connects(monkeypatch, runner, initial=True)

        assert await runner._start_one_profile_adapters("c", Path("/profiles/c"), {}) == 0
        assert len(create.built) == built_before + 1  # c's adapter was built, then refused
        assert initial == []

        runner._shutdown_event.set()
        await async_wait_until(lambda: not runner._held_claim_owners())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("served_again", [False, True])
    async def test_startup_bridge_hands_nothing_over_once_its_profile_is_unserved(self, monkeypatch, served_again):
        """A boot bridge still waiting for startup when its profile is unserved (or unserved and served again,
        which brings its own bridge) queues no reconnect after, and lets go of its bot."""
        runner, create = _shared_token_runner(monkeypatch)
        runner._running, runner._shutdown_event = False, asyncio.Event()
        runner._profile_adapters = {"a": {}}
        runner._schedule_secondary_profile_startup_reconnect("a", Platform("irc"), create(Platform("irc"), None))
        bridge = next(t for t in runner._background_tasks if t.get_name().startswith("secondary-startup"))
        runner._profile_adapters.pop("a")  # unserved while startup is still running
        if served_again:
            runner._profile_adapters["a"] = {}
        runner._running = True
        await asyncio.wait_for(bridge, 5)
        await asyncio.sleep(0)  # the bridge's done callback releases its hold
        assert not runner._profile_failed_platforms.get("a") and runner._held_claim_owners() == {}


def _shared_token_runner(monkeypatch):
    """A live runner whose every irc adapter claims one bot token (``create.built`` lists them)."""
    runner = _secondary_recovery_runner()
    _install_plugin_platform_context(monkeypatch, runner, None)
    built = []

    def create(platform, config):
        adapter = _PluginPlatformRecoveryAdapter()
        adapter.token = "shared-bot-token"
        built.append(adapter)
        return adapter

    create.built = built
    monkeypatch.setattr(runner, "_create_adapter", create)
    return runner, create


def _record_status_writes(monkeypatch, runner):
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))
    return writes


def _record_connects(monkeypatch, runner, *, result=True, initial=False):
    connects = []

    async def connect(adapter, platform, **kwargs):
        connects.append(adapter)
        return result

    name = "_connect_initial_adapter_with_timeout" if initial else "_connect_adapter_with_timeout"
    monkeypatch.setattr(runner, name, connect)
    return connects


def _park_reconnects(monkeypatch, runner):
    """Scheduled reconnect tasks wait (holding their slot) until the returned event is set."""
    parked = asyncio.Event()

    async def waiting_reconnect(profile_name, platform, predecessor=None):
        await parked.wait()

    monkeypatch.setattr(runner, "_run_secondary_profile_reconnect", waiting_reconnect)
    return parked
