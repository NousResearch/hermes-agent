"""Saving an effective configuration must not materialize include secrets."""
import os
from unittest.mock import patch


def test_include_env_secret_stays_in_include_after_unrelated_save(tmp_path):
    import hermes_cli.config as c
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "TEST_INCLUDE_SECRET": "FAKE_SECRET_ONLY"}):
        inc = tmp_path / "included.yaml"
        source = "model:\n  api_key: ${TEST_INCLUDE_SECRET}\n"
        inc.write_text(source)
        main = tmp_path / "config.yaml"
        main.write_text("include: included.yaml\n")
        c._LOAD_CONFIG_CACHE.clear()
        c._RAW_CONFIG_CACHE.clear()
        config = c.load_config()
        assert config["model"]["api_key"] == "FAKE_SECRET_ONLY"
        config["agent"]["max_turns"] = 47
        c.save_config(config)
        assert "FAKE_SECRET_ONLY" not in main.read_text()
        assert "api_key" not in main.read_text()
        assert inc.read_text() == source
        c._LOAD_CONFIG_CACHE.clear()
        c._RAW_CONFIG_CACHE.clear()
        assert c.load_config()["model"]["api_key"] == "FAKE_SECRET_ONLY"
