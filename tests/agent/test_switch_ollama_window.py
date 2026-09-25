"""Ollama request windows must follow a deliberate in-session model switch."""
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import ContextCompressor
from run_agent import AIAgent


@pytest.mark.parametrize("detected,override,expected", [(262144, None, 130000), (None, None, None), (RuntimeError("offline"), None, None), (262144, 80000, 80000)])
def test_switch_refreshes_ollama_request_window(tmp_path, monkeypatch, detected, override, expected):
    import os
    from pathlib import Path
    import yaml

    url = "http://127.0.0.1:11434/v1"
    cfg = {
        "model": {"default": "model-a", "provider": "custom", "base_url": url},
        "providers": {"ollama-local": {
            "base_url": url,
            "models": {"model-b": {"context_length": 130000}},
        }},
    }
    if override is not None:
        cfg["model"]["ollama_num_ctx"] = override
    Path(os.environ["HERMES_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(yaml.safe_dump(cfg))
    agent = AIAgent.__new__(AIAgent)
    agent.model = "model-a"
    agent.provider = "custom"
    agent.base_url = url
    agent.api_key = "fixture"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = 40000
    agent._ollama_num_ctx = 40000
    agent._primary_runtime = {}
    agent.context_compressor = ContextCompressor(
        model="model-a", threshold_percent=0.5, base_url=url, api_key="fixture",
        provider="custom", quiet_mode=True, config_context_length=40000,
    )
    agent._create_openai_client = lambda *a, **kw: MagicMock()
    def probe(*args, **kwargs):
        if isinstance(detected, Exception):
            raise detected
        return detected

    monkeypatch.setattr("agent.agent_init.query_ollama_num_ctx", probe)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: kw.get("config_context_length") or 262144)
    agent.switch_model("model-b", "custom", api_key="fixture", base_url=url)
    assert agent._config_context_length == 130000
    assert agent._ollama_num_ctx == expected
    assert agent.context_compressor.context_length == (expected or 130000)

    # A later failed switch restores the request window along with the identity.
    def fail(*args, **kwargs):
        raise RuntimeError("compressor unavailable")

    monkeypatch.setattr(agent.context_compressor, "update_model", fail)
    with pytest.raises(RuntimeError, match="compressor unavailable"):
        agent.switch_model("model-c", "custom", api_key="fixture", base_url=url)
    assert agent.model == "model-b"
    assert agent._ollama_num_ctx == expected
