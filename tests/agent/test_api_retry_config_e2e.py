"""Retry budgets are resolved through real configuration and agent initialization."""
import json
from unittest.mock import patch

import pytest

from agent.retry_utils import zai_coding_overload_retry_ceiling
from hermes_cli.config_defaults import DEFAULT_CONFIG


@pytest.mark.parametrize("key,attribute,floor", [
    ("api_max_retries", "_api_max_retries", 1),
    ("auto_recovery_cycles", "_auto_recovery_cycles", 0),
])
@pytest.mark.parametrize("setting", [None, 1, 5, "invalid", 0])
def test_real_config_retry_budget_reaches_constructor(tmp_path, monkeypatch, key, attribute, floor, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"agent": {"environment_probe": False}, "model": {"context_length": 128000}}
    if setting is not None:
        config["agent"][key] = setting
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    from hermes_cli.config import load_config_readonly
    from run_agent import AIAgent

    loaded = load_config_readonly()
    default = (zai_coding_overload_retry_ceiling() if key == "api_max_retries"
               else DEFAULT_CONFIG["agent"][key])
    expected = default if setting in (None, "invalid") else max(setting, floor)
    if setting is None:
        assert loaded["agent"][key] == expected
    with (patch("agent.process_bootstrap.OpenAI"),
          patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("hermes_cli.plugins.discover_plugins")):
        agent = AIAgent(api_key="test-key", base_url="https://api.z.ai/api/coding/paas/v4",
                        model="glm-5.2", provider="zai", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    assert getattr(agent, attribute) == expected
