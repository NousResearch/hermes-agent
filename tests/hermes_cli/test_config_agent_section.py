"""A non-mapping ``agent:`` section in config.yaml must not take down load_config()."""

import logging
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import DEFAULT_CONFIG, _normalize_max_turns_config, load_config


@pytest.mark.parametrize("raw", ["5", "true", "fast", "[1, 2]"])
def test_non_mapping_agent_section_falls_back_to_defaults(tmp_path, caplog, raw):
    (tmp_path / "config.yaml").write_text(f"agent: {raw}\nterminal:\n  timeout: 77\n", encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}), caplog.at_level(logging.WARNING):
        config = load_config()

    assert config["agent"] == DEFAULT_CONFIG["agent"]
    assert config["terminal"]["timeout"] == 77  # the rest of the file still applies
    assert "'agent' must be a mapping" in caplog.text


def test_non_mapping_agent_section_keeps_legacy_root_max_turns(tmp_path):
    (tmp_path / "config.yaml").write_text("agent: 5\nmax_turns: 42\n", encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config = load_config()

    assert config["agent"]["max_turns"] == 42
    assert "max_turns" not in config


@pytest.mark.parametrize("agent", [5, True, "fast", [1]])
def test_normalize_max_turns_tolerates_non_mapping_agent(agent):
    assert _normalize_max_turns_config({"agent": agent}) == {"agent": {}}
    assert _normalize_max_turns_config({"agent": agent, "max_turns": 7}) == {"agent": {"max_turns": 7}}
