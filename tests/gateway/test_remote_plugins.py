"""Remote gateway plugins: registry, ABC, ctx registration, CLI consumer."""

from __future__ import annotations

import asyncio
import logging

import pytest

from gateway.remote_plugins import (
    BUILTIN_SCHEMES,
    RemotePlugin,
    RemoteTarget,
    ServeHandle,
    canonical_scheme,
    get_remote_plugin,
    list_remote_plugins,
    register_remote_plugin,
    scheme_of,
    unregister_remote_plugin,
)
from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager


class FakeRemotePlugin(RemotePlugin):
    scheme = "FAKE"
    hint = "paste a fake ticket"
    paste_pattern = r"[a-z2-7]{40,}"

    def __init__(self):
        self.served = []
        self.torn_down = []

    async def serve(self, gateway_url, config=None):
        self.served.append((gateway_url, config))

        return ServeHandle(ticket="fake://ticket-123", target_url=gateway_url,
                           teardown=self._record("serve"))

    async def connect(self, saved_url):
        return RemoteTarget(base_url="http://127.0.0.1:59999", host_label="fake:ticket",
                            teardown=self._record("connect"))

    def _record(self, what):
        async def _stop():
            self.torn_down.append(what)

        return _stop


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    for scheme in list(list_remote_plugins()):
        unregister_remote_plugin(scheme)


class TestSchemeHelpers:
    def test_canonical_scheme(self):
        assert canonical_scheme(" EXample ") == "example"
        assert canonical_scheme("acme+ssh") == "acme+ssh"
        assert canonical_scheme("bad scheme") == ""
        assert canonical_scheme("1bad") == ""
        assert canonical_scheme(None) == ""

    def test_scheme_of(self):
        assert scheme_of("example://abc123") == "example"
        assert scheme_of("HTTPS://x.test") == "https"
        assert scheme_of("homelab.local:9119") == ""
        assert scheme_of(None) == ""


class TestRegistry:
    def test_register_lowercases_and_dispatches(self):
        plugin = FakeRemotePlugin()
        register_remote_plugin(plugin)

        assert plugin.scheme == "fake"
        assert get_remote_plugin("fake://ticket") is plugin
        assert get_remote_plugin("fake") is plugin
        assert get_remote_plugin("http://plain.remote") is None
        assert get_remote_plugin("not-a-scheme-url-noise") is None

    def test_builtin_schemes_not_registrable(self):
        async def _noop_serve(self, gateway_url, config=None):
            return ServeHandle(target_url=gateway_url)

        for scheme in sorted(BUILTIN_SCHEMES):
            probe = type("P", (RemotePlugin,), {"scheme": scheme, "serve": _noop_serve})()

            with pytest.raises(ValueError, match="built-in"):
                register_remote_plugin(probe)

    def test_duplicate_scheme_raises(self):
        register_remote_plugin(FakeRemotePlugin())

        with pytest.raises(ValueError, match="already registered"):
            register_remote_plugin(FakeRemotePlugin())

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            register_remote_plugin({"scheme": "fake", "serve": None})

    def test_invalid_scheme_raises(self):
        async def _noop_serve(self, gateway_url, config=None):
            return ServeHandle(target_url=gateway_url)

        probe = type("P", (RemotePlugin,), {"scheme": "not valid!", "serve": _noop_serve})()

        with pytest.raises(ValueError, match="invalid scheme"):
            register_remote_plugin(probe)

    def test_unregister_idempotent_then_reusable(self):
        register_remote_plugin(FakeRemotePlugin())
        unregister_remote_plugin("fake")
        unregister_remote_plugin("fake")  # idempotent
        assert get_remote_plugin("fake://x") is None

        second = FakeRemotePlugin()
        register_remote_plugin(second)
        assert get_remote_plugin("fake://x") is second

    def test_list_returns_snapshot(self):
        register_remote_plugin(FakeRemotePlugin())
        snapshot = list_remote_plugins()
        snapshot.clear()
        assert "fake" in list_remote_plugins()


class TestABC:
    def test_connect_optional_raises_not_implemented(self):
        class ServeOnly(RemotePlugin):
            scheme = "onlyserve"

            async def serve(self, gateway_url, config=None):
                return ServeHandle(target_url=gateway_url)

        async def run():
            with pytest.raises(NotImplementedError):
                await ServeOnly().connect("onlyserve://x")

        asyncio.run(run())

    def test_claims_and_normalize(self):
        plugin = FakeRemotePlugin()

        assert plugin.claims("aoeu" * 11) is True
        assert plugin.claims("homelab.local:9119") is False
        assert plugin.claims("") is False
        assert plugin.normalize("  fake://abc123  ") == "fake://abc123"

    def test_bad_paste_pattern_never_claims(self):
        plugin = FakeRemotePlugin()
        plugin.paste_pattern = "([unclosed"
        assert plugin.claims("anything") is False


class TestHandles:
    def test_serve_handle_stop_once_only_and_swallows(self):
        calls = []

        async def boom():
            calls.append(1)
            raise RuntimeError("already closed")

        async def run():
            handle = ServeHandle(ticket="t", teardown=boom)
            await handle.stop()
            await handle.stop()  # idempotent: teardown consumed once

            assert calls == [1]

        asyncio.run(run())

    def test_remote_target_stop_without_teardown(self):
        async def run():
            await RemoteTarget(base_url="http://127.0.0.1:1").stop()

        asyncio.run(run())


class TestCtxRegistration:
    def _ctx(self):
        return PluginContext(PluginManifest(name="test-plugin", source="user"), PluginManager())

    def test_registers_and_persists(self):
        ctx = self._ctx()
        handle = ctx.register_remote_plugin(FakeRemotePlugin())

        assert handle is not None
        assert "fake" in list_remote_plugins()

    def test_duplicate_warns_and_ignores(self, caplog):
        ctx = self._ctx()

        ctx.register_remote_plugin(FakeRemotePlugin())

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            result = ctx.register_remote_plugin(FakeRemotePlugin())

        assert result is None
        assert "already registered" in caplog.text

    def test_builtin_scheme_warns_and_ignores(self, caplog):
        ctx = self._ctx()
        https = type("P", (RemotePlugin,), {
            "scheme": "https",
            "serve": lambda self, gateway_url, config=None: ServeHandle(target_url=gateway_url),
        })()

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            assert ctx.register_remote_plugin(https) is None

        assert "built-in" in caplog.text

    def test_wrong_type_warns_and_ignores(self, caplog):
        ctx = self._ctx()

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            assert ctx.register_remote_plugin(object()) is None

        assert "RemotePlugin" in caplog.text

    def test_noop_context_accepts_the_call(self):
        from plugins.plugin_loader import NoopPluginContext

        assert NoopPluginContext().register_remote_plugin(FakeRemotePlugin()) is None


class TestCliConsumer:
    def _parse(self, argv):
        import argparse

        from hermes_cli.subcommands.remote import build_remote_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_remote_parser(subparsers)

        return parser.parse_args(["remote"] + argv)

    def test_list_empty_and_populated(self, capsys, monkeypatch):
        from hermes_cli.subcommands import remote as remote_cmd

        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)

        assert remote_cmd.cmd_remote(self._parse(["list"])) == 0
        assert "No remote gateway plugins" in capsys.readouterr().out

        register_remote_plugin(FakeRemotePlugin())

        assert remote_cmd.cmd_remote(self._parse(["list"])) == 0
        out = capsys.readouterr().out

        assert "fake://" in out and "serves+dials" in out

    def test_unknown_scheme_fails_closed(self, capsys, monkeypatch):
        from hermes_cli.subcommands import remote as remote_cmd

        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)

        assert remote_cmd.cmd_remote(self._parse(["serve", "--scheme", "ghost"])) == 2
        assert remote_cmd.cmd_remote(self._parse(["connect", "ghost://x"])) == 2
        assert "No plugin registered" in capsys.readouterr().err

    def test_https_needs_no_remote_plugin(self, capsys, monkeypatch):
        from hermes_cli.subcommands import remote as remote_cmd

        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)

        assert remote_cmd.cmd_remote(self._parse(["connect", "https://plain.remote"])) == 2
        assert "need no remote plugin" in capsys.readouterr().err

    def test_serve_passes_the_remote_plugin_config_block(self, monkeypatch):
        from hermes_cli.subcommands import remote as remote_cmd

        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"plugins": {"entries": {"fake-remote-plugin": {"relays": ["r"]}}}},
        )

        plugin = FakeRemotePlugin()
        plugin.plugin_id = "fake-remote-plugin"
        register_remote_plugin(plugin)

        def fake_run_forever(coro, *, mode):
            assert mode == "serve"
            asyncio.run(coro)

            return 0

        monkeypatch.setattr(remote_cmd, "_run_forever", fake_run_forever)

        assert remote_cmd.cmd_remote(self._parse(["serve", "--scheme", "fake"])) == 0
        assert plugin.served == [("http://127.0.0.1:9119", {"relays": ["r"]})]
