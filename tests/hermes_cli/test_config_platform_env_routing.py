"""``hermes config set/get/unset`` route every name Hermes registers as an environment variable to
``.env`` — the file the platform setup flows and ``/sethome`` already write (#111848)."""

import pytest
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


def test_registered_env_setting_converges_stale_config_yaml_copy(tmp_path, monkeypatch, capsys):
    """A non-suffix adapter key (``*_ALLOWED_USERS``) routes to ``.env``; a top-level ``config.yaml``
    copy left by an older ``config set`` is dropped on ``set`` and ``unset`` so one reader can't see a
    value the other doesn't. Credentials keep their own ``.env`` lifecycle (control)."""
    from hermes_cli import config as cfg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  default: test/model\nDISCORD_ALLOWED_USERS: '111'\nWHATSAPP_MODE: web\n",
        encoding="utf-8")

    cfg.set_config_value("DISCORD_ALLOWED_USERS", "222")
    cfg.set_config_value("TAVILY_API_KEY", "tvly-control")
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "DISCORD_ALLOWED_USERS=222" in env_text and "TAVILY_API_KEY=tvly-control" in env_text
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "model": {"default": "test/model"}, "WHATSAPP_MODE": "web"}

    cfg.unset_config_value("WHATSAPP_MODE")  # only a stale yaml copy existed
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {"model": {"default": "test/model"}}
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cfg.get_config_value("WHATSAPP_MODE")
