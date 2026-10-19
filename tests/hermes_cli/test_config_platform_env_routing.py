"""``hermes config set/get/unset`` route every name Hermes registers as an environment variable to
``.env`` — the file the platform setup flows and ``/sethome`` already write (#111848)."""

import pytest
import hermes_yaml as yaml


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


def test_unregistered_upper_snake_name_routes_to_env_by_shape(tmp_path, monkeypatch, capsys):
    """Any ``UPPER_SNAKE`` key is an environment setting even when Hermes never registered it
    (``TELEGRAM_GROUP_ALLOWED_USERS`` is read straight from ``os.getenv``): it lands in ``.env``,
    the stale ``config.yaml`` copy converges, and ``get`` reads the ``.env`` value. A lowercase bare
    key keeps the open top-level namespace (control)."""
    from hermes_cli import config as cfg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  default: test/model\nTELEGRAM_GROUP_ALLOWED_USERS: '111'\n", encoding="utf-8")

    cfg.set_config_value("TELEGRAM_GROUP_ALLOWED_USERS", "222,333")
    cfg.set_config_value("HERMES_TIMEZONE", "Europe/Berlin")
    cfg.set_config_value("my_custom_flag", "hello")
    capsys.readouterr()

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "TELEGRAM_GROUP_ALLOWED_USERS=222,333" in env_text and "HERMES_TIMEZONE=Europe/Berlin" in env_text
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "model": {"default": "test/model"}, "my_custom_flag": "hello"}

    cfg.get_config_value("TELEGRAM_GROUP_ALLOWED_USERS")
    assert capsys.readouterr().out.strip() == "222,333"


def test_env_writer_denylist_guards_upper_snake_names_and_unknown_names_get_a_note(
        tmp_path, monkeypatch, capsys):
    """Routing by shape closes the config.yaml detour around the env writer's denylist: a
    denylisted name is refused and written nowhere. A name neither registered nor documented is
    still stored in ``.env`` with a one-line note."""
    from hermes_cli import config as cfg

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  default: test/model\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        cfg.set_config_value("HERMES_YOLO_MODE", "true", force=True)
    assert "denylist" in capsys.readouterr().err
    assert not (tmp_path / ".env").exists()
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {"model": {"default": "test/model"}}

    cfg.set_config_value("SOME_PLUGIN_ONLY_KNOB", "xyz")
    assert "SOME_PLUGIN_ONLY_KNOB=xyz" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {"model": {"default": "test/model"}}
