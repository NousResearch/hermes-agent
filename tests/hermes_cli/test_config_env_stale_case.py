"""Registered environment names retain their legacy YAML fallback in every spelling."""

import json

import pytest

from hermes_cli import config as cfg


@pytest.mark.parametrize("key", ["FEISHU_HOME_CHANNEL", "feishu_home_channel", "Feishu_Home_Channel"])
@pytest.mark.parametrize("value", ["oc_CASE_TEST", "", 0, False])
def test_config_get_finds_canonical_yaml_copy_in_every_case(tmp_path, monkeypatch, capsys, key, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)
    path = tmp_path / "config.yaml"
    original = "FEISHU_HOME_CHANNEL: " + json.dumps(value) + "\n"
    path.write_text(original, encoding="utf-8")

    cfg.get_config_value(key, as_json=True, raw=True)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == value
    assert "stale top-level config.yaml copy" in captured.err
    assert path.read_text(encoding="utf-8-sig") == original
    assert not (tmp_path / ".env").exists()


def test_env_and_as_typed_values_keep_their_precedence(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)
    path = tmp_path / "config.yaml"
    original = "FEISHU_HOME_CHANNEL: canonical\nfeishu_home_channel: as-typed\n"
    path.write_text(original, encoding="utf-8")
    cfg.get_config_value("feishu_home_channel", raw=True)
    assert capsys.readouterr().out.strip() == "as-typed"

    env_path = tmp_path / ".env"
    env_path.write_text("FEISHU_HOME_CHANNEL=dotenv\n", encoding="utf-8")
    for expected in ("dotenv", "process"):
        if expected == "process":
            monkeypatch.setenv("FEISHU_HOME_CHANNEL", expected)
        for key in ("FEISHU_HOME_CHANNEL", "feishu_home_channel", "Feishu_Home_Channel"):
            cfg.get_config_value(key, raw=True)
            captured = capsys.readouterr()
            assert captured.out.strip() == expected
            assert "stale" not in captured.err
    assert path.read_text(encoding="utf-8-sig") == original
    assert env_path.read_text(encoding="utf-8-sig") == "FEISHU_HOME_CHANNEL=dotenv\n"
