"""Routing-only multiplexing: one shared connection, many personas.

``gateway.multiplex_routing_only`` — with ``multiplex_profiles`` on, the
gateway starts no secondary-profile adapters (all profiles share the default
profile's credentials) but still REGISTERS every served profile:
``_profile_adapters[profile]`` points at the shared adapter map and each
profile gets its own PairingStore. That registration is what makes a source
stamped by ``gateway.profile_routes`` resolve to a live adapter for
authorization and egress — ``_authorization_adapter`` deliberately fails
closed on unregistered stamped profiles.
"""
import pytest
from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Config: multiplex_routing_only parsing
# ---------------------------------------------------------------------------

class TestRoutingOnlyConfig:
    def test_defaults_off(self):
        assert GatewayConfig().multiplex_routing_only is False
        assert GatewayConfig.from_dict({}).multiplex_routing_only is False

    def test_from_dict_top_level(self):
        cfg = GatewayConfig.from_dict(
            {"multiplex_profiles": True, "multiplex_routing_only": True}
        )
        assert cfg.multiplex_routing_only is True

    def test_from_dict_nested_gateway_section(self):
        cfg = GatewayConfig.from_dict(
            {"gateway": {"multiplex_profiles": True, "multiplex_routing_only": True}}
        )
        assert cfg.multiplex_routing_only is True

    def test_to_dict_roundtrip(self):
        cfg = GatewayConfig(multiplex_profiles=True, multiplex_routing_only=True)
        restored = GatewayConfig.from_dict(cfg.to_dict())
        assert restored.multiplex_routing_only is True

    def test_load_gateway_config_nested_form(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "gateway:\n"
            "  multiplex_profiles: true\n"
            "  multiplex_routing_only: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from gateway.config import load_gateway_config

        cfg = load_gateway_config()
        assert cfg.multiplex_profiles is True
        assert cfg.multiplex_routing_only is True


# ---------------------------------------------------------------------------
# Gateway: routing-only shared-adapter registration
# ---------------------------------------------------------------------------

def _bare_runner(routing_only=True, shared_adapter=None, profile_routes=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig.from_dict(
        {
            "multiplex_profiles": True,
            "multiplex_routing_only": routing_only,
            "profile_routes": profile_routes or [],
        }
    )
    runner.adapters = {Platform.MATRIX: shared_adapter or MagicMock(name="shared-matrix")}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock(name="global-pairing-store")
    runner.pairing_stores = {}
    return runner


def _serve_default_and_milo(monkeypatch, tmp_path):
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(
        profiles_mod,
        "profiles_to_serve",
        lambda multiplex: [
            ("default", tmp_path / "default"),
            ("milo", tmp_path / "profiles" / "milo"),
        ],
    )
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    # PairingStore resolves its directory from HERMES_HOME at construction.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class TestRoutingOnlyRegistration:
    @pytest.mark.asyncio
    async def test_registers_shared_adapters_and_pairing_store(
        self, monkeypatch, tmp_path
    ):
        shared = MagicMock(name="shared-matrix")
        runner = _bare_runner(shared_adapter=shared)
        _serve_default_and_milo(monkeypatch, tmp_path)

        async def _boom(*a, **k):  # noqa: ANN002, ANN003
            raise AssertionError(
                "routing-only must not start secondary profile adapters"
            )

        monkeypatch.setattr(runner, "_start_one_profile_adapters", _boom)

        connected = await runner._start_secondary_profile_adapters()

        assert connected == 0
        assert runner._profile_adapters["milo"][Platform.MATRIX] is shared
        milo_store = runner.pairing_stores.get("milo")
        assert milo_store is not None
        assert milo_store.profile == "milo"

    @pytest.mark.asyncio
    async def test_stamped_source_resolves_to_shared_adapter(
        self, monkeypatch, tmp_path
    ):
        """The failure mode routing-only exists to fix: a stamped profile with
        no registry entry resolves to NO adapter (authorization fails closed
        and egress has nothing to send through). Routing-only registration
        must make the stamped source resolve to the shared adapter instead."""
        shared = MagicMock(name="shared-matrix")
        runner = _bare_runner(shared_adapter=shared)
        _serve_default_and_milo(monkeypatch, tmp_path)

        # Before registration: fail closed, exactly as authz_mixin documents.
        assert runner._authorization_adapter(Platform.MATRIX, "milo") is None

        await runner._start_secondary_profile_adapters()

        assert runner._authorization_adapter(Platform.MATRIX, "milo") is shared
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!milo:example.org",
            user_id="@alice:example.org",
            profile="milo",
        )
        assert runner._adapter_for_source(source) is shared

    @pytest.mark.asyncio
    async def test_profile_routes_stamp_resolves_shared_adapter(
        self, monkeypatch, tmp_path
    ):
        """End-to-end with the merged route matcher: a gateway.profile_routes
        rule keyed on a Matrix room resolves the profile for an inbound
        source, and routing-only registration gives that profile a live
        (shared) adapter."""
        shared = MagicMock(name="shared-matrix")
        runner = _bare_runner(
            shared_adapter=shared,
            profile_routes=[
                {
                    "name": "milo-room",
                    "platform": "matrix",
                    "chat_id": "!milo:example.org",
                    "profile": "milo",
                }
            ],
        )
        _serve_default_and_milo(monkeypatch, tmp_path)
        await runner._start_secondary_profile_adapters()

        routed = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!milo:example.org",
            user_id="@alice:example.org",
        )
        assert runner._profile_name_for_source(routed) == "milo"
        routed.profile = "milo"
        assert runner._adapter_for_source(routed) is shared

        unrouted = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!other:example.org",
            user_id="@alice:example.org",
        )
        assert runner._profile_name_for_source(unrouted) is None

    @pytest.mark.asyncio
    async def test_unserved_profile_still_fails_closed(self, monkeypatch, tmp_path):
        runner = _bare_runner()
        _serve_default_and_milo(monkeypatch, tmp_path)

        await runner._start_secondary_profile_adapters()

        assert runner._authorization_adapter(Platform.MATRIX, "ghost") is None

    @pytest.mark.asyncio
    async def test_pairing_checks_use_per_profile_store(self, monkeypatch, tmp_path):
        runner = _bare_runner()
        _serve_default_and_milo(monkeypatch, tmp_path)

        await runner._start_secondary_profile_adapters()

        stamped = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!milo:example.org",
            user_id="@alice:example.org",
            profile="milo",
        )
        unstamped = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!main:example.org",
            user_id="@alice:example.org",
        )
        assert runner._pairing_store_for(stamped) is runner.pairing_stores["milo"]
        assert runner._pairing_store_for(unstamped) is runner.pairing_store

    @pytest.mark.asyncio
    async def test_full_multiplex_path_unchanged_when_flag_off(
        self, monkeypatch, tmp_path
    ):
        runner = _bare_runner(routing_only=False)
        _serve_default_and_milo(monkeypatch, tmp_path)

        started = []

        async def _fake_start(profile_name, profile_home, claimed):
            started.append(profile_name)
            return 1

        monkeypatch.setattr(runner, "_start_one_profile_adapters", _fake_start)

        connected = await runner._start_secondary_profile_adapters()

        assert started == ["milo"]
        assert connected == 1

    @pytest.mark.asyncio
    async def test_served_profiles_recorded(self, monkeypatch, tmp_path):
        runner = _bare_runner()
        _serve_default_and_milo(monkeypatch, tmp_path)

        recorded = {}

        def _fake_write(**kwargs):
            recorded.update(kwargs)

        import gateway.status as status_mod

        monkeypatch.setattr(status_mod, "write_runtime_status", _fake_write)

        await runner._start_secondary_profile_adapters()

        assert recorded.get("served_profiles") == ["default", "milo"]
