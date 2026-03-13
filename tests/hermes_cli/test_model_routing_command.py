import argparse

from hermes_cli.config import load_config
from hermes_cli.main import cmd_configure_model_routing


def test_configure_model_routing_reset(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = load_config()
    cfg["model_profiles"]["ops"] = {"model": "x", "provider": "custom", "base_url": "", "api_key_env": "", "api_key": ""}

    cmd_configure_model_routing(argparse.Namespace(reset=True))

    updated = load_config()
    assert "ops" not in updated["model_profiles"]
    assert updated["model_routing"] == {"rules": []}
    assert "reset to defaults" in capsys.readouterr().out.lower()
