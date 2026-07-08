"""ha_conversation config defaults + lazy-dep wiring tests."""


def test_default_config_has_ha_conversation_section():
    from hermes_cli.config import DEFAULT_CONFIG

    section = DEFAULT_CONFIG["ha_conversation"]
    assert section["bind_host"] == "127.0.0.1"  # safe default: no LAN exposure
    assert section["announce_mode"] == "off"
    assert section["ack_after_seconds"] > 0
    assert section["max_transcript_chars"] > 0
    # port must avoid the well-known Wyoming service ports
    assert section["port"] not in (10700, 10400, 10300, 10200)
    # supports_home_control is derived from HA credentials, never config
    assert "supports_home_control" not in section


def test_lazy_dep_pin_matches_satellite_extra():
    from tools.lazy_deps import LAZY_DEPS

    assert LAZY_DEPS["platform.ha_conversation"] == ("wyoming==1.10.0",)
