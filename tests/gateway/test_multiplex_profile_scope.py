"""Multiplex profile scoping for gateway config reads (#57492).

``GatewayRunner.__init__`` snapshots provider routing and the fallback chain
before any per-turn profile scope exists, so those snapshots can only ever
reflect the DEFAULT profile. During a secondary profile's turn the gateway
enters that profile's scope via ``set_hermes_home_override``; per-turn
consumers must resolve provider routing / fallback chains against THAT home,
and per-profile state must never leak across profiles.

These tests pin the scope-aware readers without driving a full gateway.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write_cfg(home, text):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _stub_runner(GatewayRunner, **attrs):
    runner = SimpleNamespace(**attrs)
    runner._load_provider_routing = GatewayRunner._load_provider_routing
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    runner._resolve_provider_routing = GatewayRunner._resolve_provider_routing.__get__(runner)
    runner._refresh_fallback_model = GatewayRunner._refresh_fallback_model.__get__(runner)
    return runner


class TestProviderRoutingScope:
    def test_load_provider_routing_reads_active_profile_scope(self, tmp_path, monkeypatch):
        from gateway.run import GatewayRunner

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "travel"
        _write_cfg(default_home, "provider_routing:\n  sort: price\n")
        _write_cfg(profile_home, "provider_routing:\n  sort: throughput\n")
        monkeypatch.setattr("gateway.run._hermes_home", default_home)

        assert GatewayRunner._load_provider_routing() == {"sort": "price"}

        token = set_hermes_home_override(profile_home)
        try:
            assert GatewayRunner._load_provider_routing() == {"sort": "throughput"}
        finally:
            reset_hermes_home_override(token)

    def test_resolve_provider_routing_default_uses_init_snapshot(self, tmp_path, monkeypatch):
        from gateway.run import GatewayRunner

        default_home = tmp_path / "default"
        _write_cfg(default_home, "provider_routing:\n  sort: price\n")
        monkeypatch.setattr("gateway.run._hermes_home", default_home)

        snapshot = {"sort": "init-snapshot"}
        runner = _stub_runner(GatewayRunner, _provider_routing=snapshot)

        # No profile scope active → the init-time snapshot, not a disk read.
        assert runner._resolve_provider_routing() is snapshot

    def test_resolve_provider_routing_secondary_profile_reads_own_config(
        self, tmp_path, monkeypatch,
    ):
        from gateway.run import GatewayRunner

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "travel"
        _write_cfg(default_home, "provider_routing:\n  sort: price\n")
        _write_cfg(profile_home, "provider_routing:\n  sort: latency\n")
        monkeypatch.setattr("gateway.run._hermes_home", default_home)

        runner = _stub_runner(GatewayRunner, _provider_routing={"sort": "price"})

        token = set_hermes_home_override(profile_home)
        try:
            assert runner._resolve_provider_routing() == {"sort": "latency"}
        finally:
            reset_hermes_home_override(token)

        # Default snapshot untouched, per-profile value cached.
        assert runner._provider_routing == {"sort": "price"}
        assert runner._provider_routing_by_home[profile_home] == {"sort": "latency"}


class TestFallbackChainScope:
    def test_refresh_reads_active_profile_chain_without_clobbering_default(
        self, tmp_path, monkeypatch,
    ):
        from gateway.run import GatewayRunner

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "travel"
        _write_cfg(
            default_home,
            "fallback_providers:\n  - provider: deepseek\n    model: deepseek-v4-flash\n",
        )
        _write_cfg(
            profile_home,
            "fallback_providers:\n  - provider: gemini\n    model: gemini-2.5-flash\n",
        )
        monkeypatch.setattr("gateway.run._hermes_home", default_home)

        runner = _stub_runner(GatewayRunner, _fallback_model=None)
        default_chain = runner._refresh_fallback_model()
        assert default_chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
        assert runner._fallback_model == default_chain

        token = set_hermes_home_override(profile_home)
        try:
            profile_chain = runner._refresh_fallback_model()
        finally:
            reset_hermes_home_override(token)

        assert profile_chain == [{"provider": "gemini", "model": "gemini-2.5-flash"}]
        # The default profile's slot must NOT have been clobbered by the
        # secondary profile's refresh.
        assert runner._fallback_model == default_chain
        # And refreshing outside the scope again returns the default chain.
        assert runner._refresh_fallback_model() == default_chain

    def test_transient_read_failure_keeps_per_profile_last_known_good(
        self, tmp_path, monkeypatch,
    ):
        from gateway.run import GatewayRunner

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "travel"
        _write_cfg(
            default_home,
            "fallback_providers:\n  - provider: deepseek\n    model: deepseek-v4-flash\n",
        )
        _write_cfg(
            profile_home,
            "fallback_providers:\n  - provider: gemini\n    model: gemini-2.5-flash\n",
        )
        monkeypatch.setattr("gateway.run._hermes_home", default_home)

        runner = _stub_runner(GatewayRunner, _fallback_model=None)
        runner._refresh_fallback_model()

        token = set_hermes_home_override(profile_home)
        try:
            good = runner._refresh_fallback_model()
            assert good == [{"provider": "gemini", "model": "gemini-2.5-flash"}]

            # Simulate a mid-edit config: unparseable YAML → transient failure.
            (profile_home / "config.yaml").write_text(
                "fallback_providers: [unclosed", encoding="utf-8",
            )
            assert runner._refresh_fallback_model() == good
        finally:
            reset_hermes_home_override(token)

        # The transient failure in the secondary profile's scope must not
        # have touched the default profile's chain either.
        assert runner._fallback_model == [
            {"provider": "deepseek", "model": "deepseek-v4-flash"}
        ]

    def test_missing_profile_config_clears_only_that_profile(self, tmp_path, monkeypatch):
        from gateway.run import GatewayRunner

        default_home = tmp_path / "default"
        profile_home = tmp_path / "profiles" / "travel"  # no config.yaml
        profile_home.mkdir(parents=True)
        _write_cfg(
            default_home,
            "fallback_providers:\n  - provider: deepseek\n    model: deepseek-v4-flash\n",
        )
        monkeypatch.setattr("gateway.run._hermes_home", default_home)

        runner = _stub_runner(GatewayRunner, _fallback_model=None)
        default_chain = runner._refresh_fallback_model()
        assert default_chain

        token = set_hermes_home_override(profile_home)
        try:
            assert runner._refresh_fallback_model() is None
        finally:
            reset_hermes_home_override(token)

        assert runner._fallback_model == default_chain
