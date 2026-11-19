"""A2A Agent Card ``version`` is the running Hermes release, not a constant."""
from gateway.config import PlatformConfig
from hermes_cli import version_info
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def _seed_version(monkeypatch, base_version):
    monkeypatch.setattr(version_info, "_cached_version_info", version_info.VersionInfo(
        base_version, f"{base_version}+3.gabc1234.dirty", 3, "a" * 40, "main", "git", dirty=True))


def test_card_version_is_the_running_release(monkeypatch):
    _seed_version(monkeypatch, "0.23.2")
    card = A2AAdapter(PlatformConfig(enabled=True))._build_card()
    assert card["version"] == "0.23.2"
    # The protocol version stays on the interface.
    assert card["supportedInterfaces"][0]["protocolVersion"] == protocol.PROTOCOL_VERSION


def test_card_version_falls_back_when_release_is_unknown(monkeypatch):
    _seed_version(monkeypatch, "unknown")
    card = A2AAdapter(PlatformConfig(enabled=True))._build_card()
    assert card["version"] == protocol.FALLBACK_AGENT_VERSION


def test_build_agent_card_defaults_to_the_fallback():
    card = protocol.build_agent_card(name="x", url="http://localhost:9900/", description="d")
    assert card["version"] == protocol.FALLBACK_AGENT_VERSION
    assert protocol.build_agent_card(name="x", url="u", description="d", version="0.24.0")["version"] == "0.24.0"
