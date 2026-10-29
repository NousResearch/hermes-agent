"""The model cap must be configurable through the supported CLI, without warnings."""

import yaml

from hermes_cli.config import DEFAULT_CONFIG, _validate_config_key, set_config_value


def test_model_token_caps_are_registered_and_model_names_are_dynamic():
    assert DEFAULT_CONFIG["compression"].get("model_token_caps") == {}
    for key in ("compression.model_token_caps", "compression.model_token_caps.claude",
                "compression.model_token_caps.some-new-model"):
        assert _validate_config_key(key)[0], key
    assert not _validate_config_key("compression.model_token_cap")[0]


def test_cli_sets_numeric_model_cap_without_unknown_key_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    set_config_value("compression.model_token_caps.claude", "150000")
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved["compression"]["model_token_caps"] == {"claude": 150000}
    assert "not a recognized config key" not in capsys.readouterr().out
