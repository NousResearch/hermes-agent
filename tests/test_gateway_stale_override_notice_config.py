from gateway.config import GatewayConfig, load_gateway_config


def test_gateway_config_from_dict_parses_stale_override_notice():
    cfg = GatewayConfig.from_dict({
        "stale_override_notice": {
            "mode": "confirm",
            "idle_minutes": 90,
            "model": "off",
            "reasoning": "non_default",
            "channels": ["home", "discord:*"],
        }
    })
    notice = cfg.stale_override_notice
    assert notice.mode == "confirm"
    assert notice.idle_minutes == 90
    assert notice.model == "off"
    assert notice.reasoning == "non_default"
    assert notice.channels == ("home", "discord:*")
    assert cfg.to_dict()["stale_override_notice"]["mode"] == "confirm"


def test_loader_accepts_nested_gateway_stale_override_notice(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        """
gateway:
  stale_override_notice:
    mode: info_only
    idle_minutes: 60
    channels: [home]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    cfg = load_gateway_config()

    assert cfg.stale_override_notice.mode == "info_only"
    assert cfg.stale_override_notice.idle_minutes == 60
    assert cfg.stale_override_notice.channels == ("home",)


def test_top_level_notice_takes_precedence_over_nested(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        """
stale_override_notice:
  mode: confirm
gateway:
  stale_override_notice:
    mode: info_only
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    cfg = load_gateway_config()

    assert cfg.stale_override_notice.mode == "confirm"
