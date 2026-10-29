"""Background service registry + plugin registration hook.

Covers the register_background_service() plugin surface: registry CRUD,
the create_service() gate chain (deps -> config validation -> factory),
and the PluginContext hook that plugins call from register().
"""

import pytest

from gateway.service_registry import (
    BackgroundServiceEntry,
    BackgroundServiceRegistry,
    service_registry,
)


def _entry(name="svc_test", source="plugin", **overrides) -> BackgroundServiceEntry:
    defaults = dict(
        name=name,
        label="Service Test",
        service_factory=lambda cfg, gateway: ("instance", cfg, gateway),
        check_fn=lambda: True,
    )
    defaults.update(overrides)
    return BackgroundServiceEntry(source=source, **defaults)


class TestRegistryCrud:
    def test_register_get_is_registered(self):
        reg = BackgroundServiceRegistry()
        entry = _entry()
        reg.register(entry)
        assert reg.is_registered("svc_test")
        assert reg.get("svc_test") is entry
        assert reg.all_entries() == [entry]

    def test_reregister_replaces_last_writer_wins(self):
        reg = BackgroundServiceRegistry()
        reg.register(_entry(source="builtin"))
        replacement = _entry(source="plugin")
        reg.register(replacement)
        assert reg.get("svc_test") is replacement
        assert len(reg.all_entries()) == 1

    def test_unregister(self):
        reg = BackgroundServiceRegistry()
        reg.register(_entry())
        assert reg.unregister("svc_test") is True
        assert reg.unregister("svc_test") is False
        assert not reg.is_registered("svc_test")

    def test_plugin_entries_filters_source(self):
        reg = BackgroundServiceRegistry()
        reg.register(_entry(name="a", source="builtin"))
        reg.register(_entry(name="b", source="plugin"))
        assert [e.name for e in reg.plugin_entries()] == ["b"]


class TestCreateService:
    def test_happy_path_passes_config_and_gateway(self):
        reg = BackgroundServiceRegistry()
        reg.register(_entry())
        cfg = {"enabled": True}
        gateway = object()
        result = reg.create_service("svc_test", cfg, gateway)
        assert result == ("instance", cfg, gateway)

    def test_unknown_name_returns_none(self):
        reg = BackgroundServiceRegistry()
        assert reg.create_service("nope", {}, None) is None

    def test_failed_check_fn_returns_none(self):
        reg = BackgroundServiceRegistry()
        reg.register(_entry(check_fn=lambda: False, install_hint="pip install x"))
        assert reg.create_service("svc_test", {}, None) is None

    def test_raising_check_fn_returns_none(self):
        # A raising dependency check must be contained per entry — the
        # gateway iterates all configured services, and an escaped exception
        # would skip every service after this one.
        reg = BackgroundServiceRegistry()

        def boom():
            raise ImportError("optional dep missing")

        reg.register(_entry(check_fn=boom))
        assert reg.create_service("svc_test", {}, None) is None

    def test_failed_validate_config_returns_none(self):
        reg = BackgroundServiceRegistry()
        reg.register(_entry(validate_config=lambda cfg: False))
        assert reg.create_service("svc_test", {}, None) is None

    def test_raising_validate_config_returns_none(self):
        reg = BackgroundServiceRegistry()

        def boom(cfg):
            raise ValueError("bad config")

        reg.register(_entry(validate_config=boom))
        assert reg.create_service("svc_test", {}, None) is None

    def test_raising_factory_returns_none(self):
        reg = BackgroundServiceRegistry()

        def factory(cfg, gateway):
            raise RuntimeError("init failed")

        reg.register(_entry(service_factory=factory))
        assert reg.create_service("svc_test", {}, None) is None


class TestPluginContextHook:
    @pytest.fixture
    def ctx(self):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(
            name="svc-test-plugin", source="test", key="svc-test-plugin"
        )
        return PluginContext(manifest, PluginManager())

    def test_register_background_service_posts_to_registry(self, ctx):
        handle = None
        try:
            handle = ctx.register_background_service(
                name="ctx_svc_test",
                label="Ctx Service",
                service_factory=lambda cfg, gateway: "svc",
                check_fn=lambda: True,
                required_env=["CTX_SVC_TOKEN"],
                install_hint="pip install ctx-svc",
            )
            # Scoped-ownership contract: registration returns a live handle.
            assert handle is not None
            assert handle.active
            assert handle.kind == "background_service"
            entry = service_registry.get("ctx_svc_test")
            assert entry is not None
            assert entry.label == "Ctx Service"
            assert entry.source == "plugin"
            assert entry.plugin_name == "svc-test-plugin"
            assert entry.required_env == ["CTX_SVC_TOKEN"]
            assert service_registry.create_service("ctx_svc_test", {}, None) == "svc"
        finally:
            if handle is not None:
                handle.dispose()
        # Disposing the handle removes the registration (no predecessor).
        assert service_registry.get("ctx_svc_test") is None
        assert not service_registry.is_registered("ctx_svc_test")


class _RecordingService:
    """Fake service that records lifecycle transitions."""

    def __init__(self, events, name):
        self._events = events
        self._name = name

    async def start(self) -> bool:
        self._events.append(f"start:{self._name}")
        return True

    async def stop(self) -> None:
        self._events.append(f"stop:{self._name}")


class TestGatewayLifecycle:
    """Config load -> enabled service startup -> stop, over the real
    ``gateway.run_services`` owners (not a reimplementation)."""

    @pytest.mark.asyncio
    async def test_config_load_start_and_stop(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from gateway.config import load_gateway_config
        from gateway.run_services import (
            start_plugin_background_services,
            stop_plugin_background_services,
        )

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "services:\n"
            "  lifecycle_svc:\n"
            "    enabled: true\n"
            "    extra:\n"
            "      poll_interval: 5\n"
            "  disabled_svc:\n"
            "    enabled: false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Register AFTER the home switch: a scope-less plugin registration
        # lands in the CURRENT profile scope (fail-closed scoping), so it has
        # to happen under the scope the gateway will resolve against.
        events = []
        entry = _entry(
            name="lifecycle_svc",
            service_factory=lambda cfg, gateway: _RecordingService(
                events, "lifecycle_svc"
            ),
        )
        service_registry.register(entry)
        try:
            config = load_gateway_config()
            assert config.services["lifecycle_svc"]["enabled"] is True
            assert config.services["lifecycle_svc"]["extra"] == {"poll_interval": 5}

            # Minimal runner stand-in: the lifecycle owners only touch
            # ``runner.config`` and ``runner.services``.
            runner = SimpleNamespace(config=config, services={})

            await start_plugin_background_services(runner)
            assert list(runner.services) == ["lifecycle_svc"]
            assert events == ["start:lifecycle_svc"]

            await stop_plugin_background_services(runner)
            assert events == ["start:lifecycle_svc", "stop:lifecycle_svc"]
            assert runner.services == {}
        finally:
            service_registry.unregister("lifecycle_svc")

    @pytest.mark.asyncio
    async def test_unregistered_enabled_service_is_skipped(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        from gateway.config import load_gateway_config
        from gateway.run_services import start_plugin_background_services

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "services:\n"
            "  ghost_svc:\n"
            "    enabled: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        runner = SimpleNamespace(config=load_gateway_config(), services={})
        await start_plugin_background_services(runner)
        assert runner.services == {}


class TestScopedOwnership:
    """The registry participates in the profile-scoped replacement protocol:
    per-scope entries, snapshot + CAS restore, and plugin-owned handles —
    two profile scopes never overwrite each other, and no factory crosses
    scope or survives its owner."""

    def test_profile_scopes_do_not_overwrite_each_other(self):
        reg = BackgroundServiceRegistry()
        entry_a = _entry(label="from A")
        entry_b = _entry(label="from B")
        reg.register(entry_a, scope="scope-A")
        reg.register(entry_b, scope="scope-B")
        assert reg.snapshot_registration("svc_test", scope="scope-A")[0] is entry_a
        assert reg.snapshot_registration("svc_test", scope="scope-B")[0] is entry_b

    def test_get_resolves_current_scope_not_foreign(self):
        reg = BackgroundServiceRegistry()
        entry_a = _entry(label="from A")
        entry_b = _entry(label="from B")
        reg.register(entry_a, scope="scope-A")
        reg.register(entry_b, scope="scope-B")
        reg.current_scope_key = lambda: "scope-A"  # shadow the staticmethod
        assert reg.get("svc_test") is entry_a
        reg.current_scope_key = lambda: "scope-B"
        assert reg.get("svc_test") is entry_b
        # A third profile with no registration of its own sees nothing —
        # scoped factories never leak across profiles.
        reg.current_scope_key = lambda: "scope-C"
        assert reg.get("svc_test") is None

    def test_snapshot_and_cas_restore(self):
        reg = BackgroundServiceRegistry()
        entry_1 = _entry(label="gen 1")
        reg.register(entry_1, scope="S")
        previous = reg.snapshot_registration("svc_test", scope="S")
        entry_2 = _entry(label="gen 2")
        reg.register(entry_2, scope="S")
        current = reg.snapshot_registration("svc_test", scope="S")
        assert current[0] is entry_2
        # CAS: restore succeeds while gen 2 is still current...
        assert reg.restore_registration("svc_test", current, previous, scope="S") is True
        assert reg.snapshot_registration("svc_test", scope="S")[0] is entry_1
        # ...and is refused once the state moved on (stale current).
        assert reg.restore_registration("svc_test", current, previous, scope="S") is False

    def _ctx(self, name, scope_key):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name=name, source="test", key=name)
        return PluginContext(manifest, PluginManager(scope_key=scope_key))

    def test_ab_profiles_same_name_are_isolated(self, tmp_path):
        """Deterministic A/B profile case: the same service name registered
        from two PluginManagers with different scope keys — neither handle
        displaces the other, and disposal only removes its own scope."""
        scope_a, scope_b = str(tmp_path / "homeA"), str(tmp_path / "homeB")
        ctx_a = self._ctx("plugin-a", scope_a)
        ctx_b = self._ctx("plugin-b", scope_b)
        handle_a = ctx_a.register_background_service(
            name="ab_scoped_svc", label="A", check_fn=lambda: True,
            service_factory=lambda cfg, gw: "svc-a",
        )
        handle_b = ctx_b.register_background_service(
            name="ab_scoped_svc", label="B", check_fn=lambda: True,
            service_factory=lambda cfg, gw: "svc-b",
        )
        try:
            assert handle_a is not None and handle_a.active
            assert handle_b is not None and handle_b.active
            snap_a = service_registry.snapshot_registration("ab_scoped_svc", scope=scope_a)
            snap_b = service_registry.snapshot_registration("ab_scoped_svc", scope=scope_b)
            assert snap_a[0] is not None and snap_a[0].label == "A"
            assert snap_b[0] is not None and snap_b[0].label == "B"
        finally:
            handle_b.dispose()
            handle_a.dispose()
        assert service_registry.snapshot_registration("ab_scoped_svc", scope=scope_a)[0] is None
        assert service_registry.snapshot_registration("ab_scoped_svc", scope=scope_b)[0] is None

    def test_unload_restores_displaced_predecessor(self, tmp_path):
        """Reload generation: a re-registration under the same scope displaces
        the predecessor; disposing the newer handle CAS-restores it, disposing
        the older one then removes it — the factory never survives its owner."""
        scope = str(tmp_path / "home")
        ctx_gen1 = self._ctx("svc-plugin", scope)
        handle_1 = ctx_gen1.register_background_service(
            name="reload_scoped_svc", label="gen 1", check_fn=lambda: True,
            service_factory=lambda cfg, gw: "svc-gen1",
        )
        assert handle_1 is not None
        ctx_gen2 = self._ctx("svc-plugin", scope)
        handle_2 = ctx_gen2.register_background_service(
            name="reload_scoped_svc", label="gen 2", check_fn=lambda: True,
            service_factory=lambda cfg, gw: "svc-gen2",
        )
        assert handle_2 is not None
        assert service_registry.snapshot_registration("reload_scoped_svc", scope=scope)[0].label == "gen 2"
        handle_2.dispose()
        restored = service_registry.snapshot_registration("reload_scoped_svc", scope=scope)[0]
        assert restored is not None and restored.label == "gen 1"
        handle_1.dispose()
        assert service_registry.snapshot_registration("reload_scoped_svc", scope=scope)[0] is None


class _LifecycleProbe:
    """Configurable fake service for the run_services lifecycle contract."""

    def __init__(self, events, name, *, start_result="none", stop_hangs=False):
        self.events = events
        self.name = name
        self.start_result = start_result
        self.stop_hangs = stop_hangs

    async def start(self):
        import asyncio

        self.events.append(f"start:{self.name}")
        if self.start_result == "none":
            return None
        if self.start_result == "false":
            return False
        if self.start_result == "raise":
            # partial allocation then failure — the runtime owns cleanup
            raise RuntimeError("boom after partial alloc")
        if self.start_result == "hang":
            await asyncio.sleep(3600)
        if self.start_result == "resist-cancel":
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                # swallow the cancel, finish on our own — the caller must
                # already have detached instead of waiting on us
                await asyncio.sleep(0)
                self.events.append(f"survived-cancel:{self.name}")
        if self.start_result == "resist-cancel-hang":
            # The adversarial shape: swallow the runtime's cancel AND keep
            # running — this generation never settles within the detach
            # grace. A SECOND cancel (test cleanup) is honored so the task
            # cannot outlive the test's event loop.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.events.append(f"swallowed-cancel:{self.name}")
                await asyncio.sleep(3600)
        return True

    async def stop(self):
        import asyncio

        self.events.append(f"stop:{self.name}")
        if self.stop_hangs:
            await asyncio.sleep(3600)


class TestLifecycleContract:
    """P1 lifecycle contract (gateway/run_services.py): completion counts as
    started (``None`` included), only ``False``/raise fail; failed starts get
    bounded cleanup; hanging start/stop cannot wedge the gateway; one bad
    service never blocks the others."""

    def _register(self, events, name, **probe_kwargs):
        entry = _entry(
            name=name,
            service_factory=lambda cfg, gw: _LifecycleProbe(events, name, **probe_kwargs),
        )
        service_registry.register(entry)
        return entry

    def _runner(self, *names):
        from types import SimpleNamespace

        return SimpleNamespace(
            config=SimpleNamespace(services={n: {"enabled": True} for n in names}),
            services={},
        )

    @pytest.mark.asyncio
    async def test_start_returning_none_counts_as_started(self):
        from gateway.run_services import (
            start_plugin_background_services,
            stop_plugin_background_services,
        )

        events = []
        self._register(events, "none_svc", start_result="none")
        try:
            runner = self._runner("none_svc")
            await start_plugin_background_services(runner)
            # The common untyped ``-> None`` shape is a successful start...
            assert list(runner.services) == ["none_svc"]
            # ...and the service is owned: it receives stop() at shutdown.
            await stop_plugin_background_services(runner)
            assert events == ["start:none_svc", "stop:none_svc"]
            assert runner.services == {}
        finally:
            service_registry.unregister("none_svc")

    @pytest.mark.asyncio
    async def test_start_returning_false_gets_bounded_cleanup(self):
        from gateway.run_services import start_plugin_background_services

        events = []
        self._register(events, "false_svc", start_result="false")
        try:
            runner = self._runner("false_svc")
            await start_plugin_background_services(runner)
            assert runner.services == {}
            # Transactional start: the failed instance was cleaned up.
            assert events == ["start:false_svc", "stop:false_svc"]
        finally:
            service_registry.unregister("false_svc")

    @pytest.mark.asyncio
    async def test_raising_start_gets_cleanup_and_later_services_run(self):
        from gateway.run_services import start_plugin_background_services

        events = []
        self._register(events, "raise_svc", start_result="raise")
        self._register(events, "good_svc", start_result="none")
        try:
            runner = self._runner("raise_svc", "good_svc")
            await start_plugin_background_services(runner)
            # raise-after-partial-alloc: instance discarded but cleaned up
            assert "stop:raise_svc" in events
            assert "raise_svc" not in runner.services
            # containment: the bad service did not block the next one
            assert list(runner.services) == ["good_svc"]
        finally:
            service_registry.unregister("raise_svc")
            service_registry.unregister("good_svc")

    @pytest.mark.asyncio
    async def test_hanging_start_is_bounded(self, monkeypatch):
        from gateway import run_services

        events = []
        self._register(events, "hang_svc", start_result="hang")
        self._register(events, "after_svc", start_result="none")
        monkeypatch.setattr(run_services, "_SERVICE_START_TIMEOUT_S", 0.05)
        monkeypatch.setattr(run_services, "_SERVICE_STOP_TIMEOUT_S", 0.05)
        try:
            runner = self._runner("hang_svc", "after_svc")
            await run_services.start_plugin_background_services(runner)
            # The wedged service was bounded away and the gateway moved on.
            assert "hang_svc" not in runner.services
            assert list(runner.services) == ["after_svc"]
        finally:
            service_registry.unregister("hang_svc")
            service_registry.unregister("after_svc")

    @pytest.mark.asyncio
    async def test_cancellation_resistant_start_is_detached_not_awaited(self, monkeypatch):
        import asyncio

        from gateway import run_services

        events = []
        self._register(events, "resist_svc", start_result="resist-cancel")
        monkeypatch.setattr(run_services, "_SERVICE_START_TIMEOUT_S", 0.05)
        monkeypatch.setattr(run_services, "_SERVICE_STOP_TIMEOUT_S", 0.05)
        try:
            runner = self._runner("resist_svc")
            await run_services.start_plugin_background_services(runner)
            # Returned promptly despite the cancel-swallowing service...
            assert runner.services == {}
            # ...which finishes detached, on its own, without a waiting caller.
            await asyncio.sleep(0.05)
            assert "survived-cancel:resist_svc" in events
        finally:
            service_registry.unregister("resist_svc")

    @pytest.mark.asyncio
    async def test_hanging_stop_does_not_block_shutdown(self, monkeypatch):
        from types import SimpleNamespace

        from gateway import run_services

        events = []
        bad = _LifecycleProbe(events, "bad_stop", stop_hangs=True)
        good = _LifecycleProbe(events, "good_stop")
        monkeypatch.setattr(run_services, "_SERVICE_STOP_TIMEOUT_S", 0.05)
        runner = SimpleNamespace(config=SimpleNamespace(services={}), services={
            "bad_stop": bad,
            "good_stop": good,
        })
        await run_services.stop_plugin_background_services(runner)
        # Shutdown settled: the hanging stop was bounded, the next service
        # still stopped, and the dict cleared.
        assert "stop:good_stop" in events
        assert runner.services == {}


class TestLifecycleRaceRepairs:
    """October-review P1 repairs: the detached-start/cleanup race and
    outer-cancellation shutdown settlement."""

    def _register(self, events, name, **probe_kwargs):
        entry = _entry(
            name=name,
            service_factory=lambda cfg, gw: _LifecycleProbe(events, name, **probe_kwargs),
        )
        service_registry.register(entry)
        return entry

    def _runner(self, *names):
        from types import SimpleNamespace

        return SimpleNamespace(
            config=SimpleNamespace(services={n: {"enabled": True} for n in names}),
            services={},
        )

    @pytest.mark.asyncio
    async def test_cleanup_fences_on_detached_start_settlement(self, monkeypatch):
        """stop() must never run concurrently with a cancel-swallowing
        start() that is still executing: cleanup waits for the detached
        generation to settle first, so the stop event strictly follows the
        survived-cancel event."""
        from gateway import run_services

        events = []
        self._register(events, "fence_svc", start_result="resist-cancel")
        monkeypatch.setattr(run_services, "_SERVICE_START_TIMEOUT_S", 0.05)
        monkeypatch.setattr(run_services, "_SERVICE_STOP_TIMEOUT_S", 0.5)
        try:
            runner = self._runner("fence_svc")
            await run_services.start_plugin_background_services(runner)
            assert runner.services == {}
            # Cleanup ran — but only AFTER the detached generation settled.
            assert "survived-cancel:fence_svc" in events
            assert "stop:fence_svc" in events
            assert events.index("survived-cancel:fence_svc") < events.index("stop:fence_svc")
        finally:
            service_registry.unregister("fence_svc")

    @pytest.mark.asyncio
    async def test_never_settling_generation_keeps_ownership_no_stop(self, monkeypatch, caplog):
        """A generation that swallows the cancel AND keeps running past the
        grace is not raced: stop() is skipped with a warning (the detached
        task keeps ownership of its partial allocations)."""
        import logging

        from gateway import run_services

        events = []
        self._register(events, "never_svc", start_result="resist-cancel-hang")
        monkeypatch.setattr(run_services, "_SERVICE_START_TIMEOUT_S", 0.05)
        monkeypatch.setattr(run_services, "_SERVICE_DETACH_GRACE_S", 0.05)
        try:
            runner = self._runner("never_svc")
            with caplog.at_level(logging.WARNING, logger="gateway.run_services"):
                await run_services.start_plugin_background_services(runner)
            assert runner.services == {}
            # No teardown raced the still-running setup.
            assert "stop:never_svc" not in events
            assert any("still running" in r.message for r in caplog.records)
        finally:
            service_registry.unregister("never_svc")
            # Settle the deliberately-detached generation so it cannot
            # outlive this test's event loop (second cancel is honored).
            import asyncio

            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_outer_cancel_mid_stop_still_settles_everything(self, monkeypatch):
        """Cancelling the shutdown task while one stop() hangs must not skip
        the remaining services or the ownership clear; the cancellation is
        re-raised only after settlement."""
        import asyncio
        from types import SimpleNamespace

        from gateway import run_services

        events = []
        bad = _LifecycleProbe(events, "hang", stop_hangs=True)
        good = _LifecycleProbe(events, "good")
        # Long per-service bound so the outer cancel (not the bound) is what
        # interrupts the first stop.
        monkeypatch.setattr(run_services, "_SERVICE_STOP_TIMEOUT_S", 30.0)
        runner = SimpleNamespace(config=SimpleNamespace(services={}), services={
            "hang": bad,
            "good": good,
        })

        task = asyncio.ensure_future(run_services.stop_plugin_background_services(runner))
        await asyncio.sleep(0.05)  # let it enter the hanging stop
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Settlement was unconditional: the second service stopped and the
        # ownership dict cleared despite the cancellation.
        assert "stop:good" in events
        assert runner.services == {}


class TestScopeFailClosed:
    """October-review P2 repair: a scope-less direct registration never lands
    process-global."""

    def test_unresolvable_plugin_scope_falls_back_to_current_scope(self):
        reg = BackgroundServiceRegistry()
        reg.current_scope_key = lambda: "profile-A"  # shadow the staticmethod
        entry_a = _entry(label="from A")
        reg.register(entry_a)  # plain lambdas: inference yields None
        # Landed in the CURRENT profile scope, not the global map.
        assert reg.snapshot_registration("svc_test", scope="profile-A")[0] is entry_a
        assert reg._entries == {}
        # A second profile registering the same name cannot overwrite it.
        reg.current_scope_key = lambda: "profile-B"
        entry_b = _entry(label="from B")
        reg.register(entry_b)
        assert reg.snapshot_registration("svc_test", scope="profile-A")[0] is entry_a
        assert reg.snapshot_registration("svc_test", scope="profile-B")[0] is entry_b
        assert reg._entries == {}
        # Each profile resolves its own entry.
        reg.current_scope_key = lambda: "profile-A"
        assert reg.get("svc_test") is entry_a

    def test_builtin_source_stays_process_global(self):
        reg = BackgroundServiceRegistry()
        reg.current_scope_key = lambda: "profile-A"
        entry = _entry(source="builtin")
        reg.register(entry)
        assert reg._entries.get("svc_test") is entry


class TestRegistrationMissIsLogged:
    """October-review P3 repair: an exact-replacement miss returns None WITH
    a warning naming the service and scope."""

    def test_displaced_registration_warns(self, monkeypatch, caplog):
        import logging

        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="miss-plugin", source="test", key="miss-plugin")
        ctx = PluginContext(manifest, PluginManager())
        # Force the miss path deterministically: registration is swallowed,
        # so the post-register snapshot cannot be the new entry.
        monkeypatch.setattr(service_registry, "register", lambda *a, **k: None)
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            handle = ctx.register_background_service(
                name="miss_svc",
                label="Miss Service",
                service_factory=lambda cfg, gw: "svc",
                check_fn=lambda: True,
            )
        assert handle is None
        hits = [r for r in caplog.records if "did not land exactly" in r.message]
        assert hits, "expected a warning on the exact-replacement miss path"
        assert "miss_svc" in hits[0].getMessage()
