"""Regression tests for t_40b1741b: a secondary profile sharing the SAME WAHA session/listener as
another profile under ``gateway.multiplex_profiles`` must be refused BEFORE ``connect()`` attempts a
socket bind, instead of hitting ``[Errno 98] address already in use`` and looping an exponential
reconnect backoff forever. Mirrors the existing Photon sidecar coverage in
``test_multiplex_adapter_registry.py`` — same claim mechanism (``_adapter_listener_claim`` /
``_refuse_duplicate_claim``), now also wired for WAHA's webhook listener.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


class _FakeWahaAdapter:
    """Stands in for ``plugins.platforms.waha.adapter.WahaAdapter`` — only the attributes the
    listener-claim logic reads, so these tests stay independent of aiohttp being installed."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self.platform = Platform("waha")
        self.config = PlatformConfig(enabled=True)
        self.connect_called = False
        self.disconnected = False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self.connect_called = True  # the exact behavior that must NEVER happen on a refused duplicate
        return True

    async def disconnect(self) -> None:
        self.disconnected = True

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


class TestWahaListenerClaim:
    def test_claims_same_host_port(self):
        waha = Platform("waha")
        a = _FakeWahaAdapter("127.0.0.1", 8666)
        b = _FakeWahaAdapter("127.0.0.1", 8666)
        claim_a = GatewayRunner._adapter_listener_claim(waha, a)
        claim_b = GatewayRunner._adapter_listener_claim(waha, b)
        assert claim_a is not None
        assert claim_a == claim_b

    def test_distinct_ports_distinct_claims(self):
        waha = Platform("waha")
        a = _FakeWahaAdapter("127.0.0.1", 8666)
        b = _FakeWahaAdapter("127.0.0.1", 8667)
        assert GatewayRunner._adapter_listener_claim(waha, a) != GatewayRunner._adapter_listener_claim(waha, b)

    def test_photon_and_waha_never_collide_on_the_same_port_number(self):
        """The claim tuple carries the platform name, so two unrelated platforms coincidentally
        bound to the same port number (8666) never cross-refuse each other."""
        photon = Platform("photon")
        waha = Platform("waha")

        class _FakePhoton:
            _sidecar_bind = "127.0.0.1"
            _sidecar_port = 8666

        assert (
            GatewayRunner._adapter_listener_claim(photon, _FakePhoton())
            != GatewayRunner._adapter_listener_claim(waha, _FakeWahaAdapter("127.0.0.1", 8666))
        )


@pytest.mark.asyncio
async def test_secondary_same_waha_session_is_refused_without_ever_calling_connect(monkeypatch):
    """The reported symptom: profile 'andrew' shares the default profile's WAHA session/port.
    Before the fix this adapter's own ``connect()`` was reached, which tries ``TCPSite.start()``
    and raises ``OSError(98, 'address already in use')`` — logged every 30s/60s/120s forever. The
    claim must refuse it BEFORE ``connect()`` is ever awaited."""
    waha = Platform("waha")
    primary = _FakeWahaAdapter("127.0.0.1", 8666)
    secondary = _FakeWahaAdapter("127.0.0.1", 8666)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {}
    runner.session_store = None
    runner._busy_text_mode = "queue"

    profile_cfg = GatewayConfig(multiplex_profiles=True)
    profile_cfg.platforms = {waha: PlatformConfig(enabled=True)}
    writes = []

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)
    monkeypatch.setattr(runner, "_create_adapter", lambda p, c: secondary)
    monkeypatch.setattr(runner, "_make_adapter_auth_check", lambda p, **kwargs: None)
    monkeypatch.setattr(
        runner, "_update_platform_runtime_status",
        lambda key, **kwargs: writes.append((key, kwargs)),
    )

    claimed = {GatewayRunner._adapter_listener_claim(waha, primary): "default"}
    connected = await runner._start_one_profile_adapters("andrew", "/tmp/andrew", claimed)

    assert connected == 0
    assert secondary.connect_called is False  # the actual bug: this must never fire
    assert waha not in runner._profile_adapters.get("andrew", {})
    assert writes and writes[0][0] == "andrew:waha"
    assert writes[0][1]["error_code"] == "duplicate_listener"
    # The remedy must name the real fix (profile_routes) and not merely "pick a different port",
    # which is wrong advice when the two profiles deliberately share one WhatsApp number.
    assert "profile_routes" in writes[0][1]["error_message"]
    # Still no public bind-detail leak (the convention the Photon test already locks in).
    assert "127.0.0.1" not in writes[0][1]["error_message"]
    assert "8666" not in writes[0][1]["error_message"]


@pytest.mark.asyncio
async def test_distinct_waha_sessions_on_distinct_ports_both_connect(monkeypatch):
    """Two profiles genuinely running separate WAHA instances (different numbers, different
    ports) are NOT the same resource and must both connect — this is not a blanket ban on a
    second WAHA adapter under multiplex, only on a literal socket collision."""
    waha = Platform("waha")
    primary = _FakeWahaAdapter("127.0.0.1", 8666)
    secondary = _FakeWahaAdapter("127.0.0.1", 8667)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {}
    runner.session_store = None
    runner._busy_text_mode = "queue"

    profile_cfg = GatewayConfig(multiplex_profiles=True)
    profile_cfg.platforms = {waha: PlatformConfig(enabled=True)}

    async def _connect(adapter, platform, **_kw):
        adapter.connect_called = True
        return True

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)
    monkeypatch.setattr(runner, "_create_adapter", lambda p, c: secondary)
    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", _connect)
    monkeypatch.setattr(runner, "_configure_profile_adapter", lambda *a: None)
    monkeypatch.setattr(runner, "_sync_voice_mode_state_to_adapter", lambda *a: None)
    monkeypatch.setattr(runner, "_make_adapter_auth_check", lambda p, **kwargs: None)

    claimed = {GatewayRunner._adapter_listener_claim(waha, primary): "default"}
    connected = await runner._start_one_profile_adapters("reviewer", "/tmp/reviewer", claimed)

    assert connected == 1
    assert secondary.connect_called is True
    assert runner._profile_adapters["reviewer"][waha] is secondary
