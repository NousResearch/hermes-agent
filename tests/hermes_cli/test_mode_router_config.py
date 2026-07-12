"""Configuration contract tests for the automatic mode router flag."""

from unittest.mock import patch

from hermes_cli.config import DEFAULT_CONFIG, load_config


def test_mode_router_is_disabled_by_default():
    assert DEFAULT_CONFIG["agent"]["mode_router"] == {"enabled": False}


def test_mode_router_default_is_merged_without_discarding_unknown_fields(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "agent:\n"
        "  mode_router:\n"
        "    future_option: keep-me\n",
        encoding="utf-8",
    )

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}):
        config = load_config()

    assert config["agent"]["mode_router"] == {
        "enabled": False,
        "future_option": "keep-me",
    }
