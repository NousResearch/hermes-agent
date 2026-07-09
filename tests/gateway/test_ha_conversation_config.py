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


import textwrap


class _RegistryCtx:
    def register_platform(self, **kwargs):
        from gateway.platform_registry import PlatformEntry, platform_registry

        allowed = {f.name for f in PlatformEntry.__dataclass_fields__.values()}
        platform_registry.register(
            PlatformEntry(**{k: v for k, v in kwargs.items() if k in allowed})
        )


def _load_with(tmp_path, monkeypatch, yaml_text):
    from gateway import config as gw_config
    from tests.gateway._plugin_adapter_loader import load_plugin_adapter

    load_plugin_adapter("ha_conversation").register(_RegistryCtx())
    (tmp_path / "config.yaml").write_text(
        textwrap.dedent(yaml_text), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = gw_config.load_gateway_config()
    return config.platforms.get(gw_config.Platform("ha_conversation"))


def test_gateway_config_chain_top_level(tmp_path, monkeypatch):
    pc = _load_with(tmp_path, monkeypatch, """
        ha_conversation:
          enabled: true
          port: 10611
          announce_mode: broadcast
    """)
    assert pc is not None and pc.enabled is True
    assert pc.extra["port"] == 10611
    assert pc.extra["announce_mode"] == "broadcast"


def test_gateway_config_chain_nested_platforms(tmp_path, monkeypatch):
    pc = _load_with(tmp_path, monkeypatch, """
        platforms:
          ha_conversation:
            enabled: true
            port: 10612
    """)
    assert pc is not None and pc.enabled is True
    assert pc.extra["port"] == 10612
