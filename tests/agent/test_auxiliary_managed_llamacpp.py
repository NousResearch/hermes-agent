"""A bare llama.cpp fallback resolves the managed local server, not a generic custom endpoint."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest
from openai import OpenAI


@pytest.fixture
def managed_llamacpp(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model:\n  provider: nous\n  default: test-model\n", encoding="utf-8")

    from hermes_cli.local_runtime.supervisor import state_path
    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

    state = state_path()
    state.parent.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    parent = proc.parent()
    assert parent is not None
    state.write_text(json.dumps({
        "base_url": "http://127.0.0.1:19991/v1", "api_key": "sk-managed-test",
        "pid": proc.pid, "create_time": proc.create_time(), "executable": proc.exe(),
        "owner_pid": parent.pid, "owner_create_time": parent.create_time(),
    }), encoding="utf-8")
    assert resolve_llamacpp_endpoint(wait_for_boot_s=0) == {
        "base_url": "http://127.0.0.1:19991/v1", "api_key": "sk-managed-test",
    }


@pytest.mark.parametrize("provider", ("llamacpp", "llama.cpp", "llama-cpp"))
def test_managed_llamacpp_alias_resolves_without_inline_endpoint(managed_llamacpp, provider):
    from agent.auxiliary_client import resolve_provider_client

    client, model = resolve_provider_client(provider, model="local-model", raw_codex=True)

    assert isinstance(client, OpenAI)
    assert str(client.base_url).rstrip("/") == "http://127.0.0.1:19991/v1"
    assert client.api_key == "sk-managed-test"
    assert model == "local-model"


def test_rate_limited_turn_activates_managed_llamacpp_fallback(managed_llamacpp):
    from agent.error_classifier import FailoverReason
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="primary-key", base_url="https://openrouter.ai/api/v1", provider="openrouter",
            model="primary-model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "llamacpp", "model": "local-model"}],
        )
    agent.client = MagicMock()

    assert agent._try_activate_fallback(reason=FailoverReason.rate_limit)
    assert agent.provider == "llamacpp"
    assert agent.model == "local-model"
    assert agent.api_key == "sk-managed-test"
