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
        try:
            ctx.register_background_service(
                name="ctx_svc_test",
                label="Ctx Service",
                service_factory=lambda cfg, gateway: "svc",
                check_fn=lambda: True,
                required_env=["CTX_SVC_TOKEN"],
                install_hint="pip install ctx-svc",
            )
            entry = service_registry.get("ctx_svc_test")
            assert entry is not None
            assert entry.label == "Ctx Service"
            assert entry.source == "plugin"
            assert entry.plugin_name == "svc-test-plugin"
            assert entry.required_env == ["CTX_SVC_TOKEN"]
            assert service_registry.create_service("ctx_svc_test", {}, None) == "svc"
        finally:
            service_registry.unregister("ctx_svc_test")


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
    GatewayRunner methods (not a reimplementation)."""

    @pytest.fixture
    def lifecycle_registry(self):
        events = []
        entry = _entry(
            name="lifecycle_svc",
            service_factory=lambda cfg, gateway: _RecordingService(
                events, "lifecycle_svc"
            ),
        )
        service_registry.register(entry)
        try:
            yield events
        finally:
            service_registry.unregister("lifecycle_svc")

    @pytest.mark.asyncio
    async def test_config_load_start_and_stop(
        self, tmp_path, monkeypatch, lifecycle_registry
    ):
        from types import SimpleNamespace

        from gateway.config import load_gateway_config
        from gateway.run import GatewayRunner

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

        config = load_gateway_config()
        assert config.services["lifecycle_svc"]["enabled"] is True
        assert config.services["lifecycle_svc"]["extra"] == {"poll_interval": 5}

        # Minimal runner stand-in: the real methods only touch
        # ``self.config`` and ``self.services``.
        runner = SimpleNamespace(config=config, services={})

        await GatewayRunner._start_plugin_background_services(runner)
        assert list(runner.services) == ["lifecycle_svc"]
        assert lifecycle_registry == ["start:lifecycle_svc"]

        await GatewayRunner._stop_plugin_background_services(runner)
        assert lifecycle_registry == ["start:lifecycle_svc", "stop:lifecycle_svc"]
        assert runner.services == {}

    @pytest.mark.asyncio
    async def test_unregistered_enabled_service_is_skipped(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        from gateway.config import load_gateway_config
        from gateway.run import GatewayRunner

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
        await GatewayRunner._start_plugin_background_services(runner)
        assert runner.services == {}
