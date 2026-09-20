"""Config-set must not turn model routing identifiers into scalars (#117345)."""

import argparse

import pytest
import yaml

from hermes_cli.config import config_command, read_raw_config, set_config_value


@pytest.mark.parametrize("key", [
    "model.provider", "model.default", "model.name", "model.model",
    "model.base_url", "model.api_base", "model.api_mode", "model",
])
@pytest.mark.parametrize("value", ["2", "007", "0", "1.5", "off", "null", "", "[local]"])
def test_model_route_strings_survive_config_set(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "original", "context_length": 4096}}),
        encoding="utf-8",
    )
    config_command(argparse.Namespace(config_command="set", key=key, value=value, force=False))
    stored = read_raw_config()["model"]
    leaf = "default" if key == "model" else key.split(".")[1]
    if leaf == "api_base":
        leaf = "base_url"
        if not value:
            # The existing alias normalizer drops empty aliases; do not change it.
            assert leaf not in stored
            return
    assert stored[leaf] == value
    assert isinstance(stored[leaf], str)
    assert stored["context_length"] == 4096


def test_non_string_settings_keep_coercion_and_container_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key, value in [
        ("model.context_length", "8192"), ("agent.max_turns", "7"),
        ("compression.threshold", "0.5"), ("checkpoints.enabled", "false"),
    ]:
        set_config_value(key, value)
    stored = read_raw_config()
    assert stored["model"]["context_length"] == 8192
    assert stored["agent"]["max_turns"] == 7
    assert stored["compression"]["threshold"] == 0.5
    assert stored["checkpoints"]["enabled"] is False
    before = (tmp_path / "config.yaml").read_bytes()
    with pytest.raises(SystemExit):
        set_config_value("model.aliases", "not-a-mapping")
    assert (tmp_path / "config.yaml").read_bytes() == before
