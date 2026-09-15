"""Regression coverage for platform setup values managed through ``.env``."""

import yaml


def test_platform_env_key_round_trips_without_a_config_yaml_copy(tmp_path, monkeypatch, capsys):
    """``config set/get/unset`` shares the platform setup flow's .env storage."""
    from hermes_cli import config as cfg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  default: test/model\n", encoding="utf-8")

    cfg.set_config_value("FEISHU_HOME_CHANNEL", "oc_ROUTING_TEST")

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "model": {"default": "test/model"}
    }
    assert (tmp_path / ".env").read_text(encoding="utf-8") == (
        "FEISHU_HOME_CHANNEL=oc_ROUTING_TEST\n"
    )

    cfg.get_config_value("FEISHU_HOME_CHANNEL")
    assert capsys.readouterr().out.strip().endswith("oc_ROUTING_TEST")

    cfg.unset_config_value("FEISHU_HOME_CHANNEL")
    assert "FEISHU_HOME_CHANNEL" not in (tmp_path / ".env").read_text(encoding="utf-8")
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "model": {"default": "test/model"}
    }
