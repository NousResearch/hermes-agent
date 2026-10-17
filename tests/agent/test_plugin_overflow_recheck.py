"""Provider-overflow recovery for uncalibrated plugin engines (regression for #134321)."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import hermes_yaml as yaml

from agent.context_engine import ContextEngine
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from run_agent import AIAgent


class UncalibratedEngine(ContextEngine):
    name = "uncalibrated"

    def update_model(self, *args, **kwargs):
        pass

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, **kwargs):
        return messages


def _agent(home, host_window, *, engine_window=0, threshold=0, percent=0.75):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "test/model", "provider": "openrouter", "context_length": host_window},
        "compression": {"enabled": True},
    }))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(model="test/model", provider="openrouter", api_key="test-key",
                        base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, save_trajectories=False)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.context_compressor = UncalibratedEngine()
    agent.context_compressor.context_length = engine_window
    agent.context_compressor.threshold_tokens = threshold
    agent.context_compressor.threshold_percent = percent
    assert agent._config_context_length == host_window
    return agent


def _recover(agent, after_tokens, reported_window, *, unknown_host=False):
    error = Exception(
        f"request (300000 tokens) exceeds the available context size ({reported_window} tokens)"
        if reported_window else "Your input exceeds the context window of this model."
    )
    error.status_code = 400
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered", tool_calls=None,
            reasoning_content=None, reasoning=None), finish_reason="stop")],
        model=agent.model, usage=None,
    )
    agent.client.chat.completions.create.side_effect = [error, response]

    def pressure(*args, **kwargs):
        return 300_000 if agent.client.chat.completions.create.call_count == 0 else after_tokens

    compacted = ([{"role": "user", "content": "summary"},
                  {"role": "assistant", "content": "ack"}], "You are helpful.")
    with ExitStack() as stack:
        stack.enter_context(patch("agent.turn_overflow.time.sleep"))
        stack.enter_context(patch("agent.retry_utils.jittered_backoff", return_value=0))
        stack.enter_context(patch("agent.turn_context.estimate_request_tokens_rough", return_value=after_tokens))
        stack.enter_context(patch("agent.conversation_loop._midturn_request_pressure_tokens", side_effect=pressure))
        stack.enter_context(patch.object(agent, "_compress_context", return_value=compacted))
        if unknown_host:
            stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=0))
        return agent.run_conversation("continue", conversation_history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier answer"},
        ])


@pytest.mark.parametrize("host_window,engine_window,threshold,percent,reported,after_tokens,unknown_host,completed", [
    (272_000, 0, 0, 0.75, 272_000, 52_500, False, True),
    (272_000, 0, 0, 0.75, None, 52_500, False, True),
    (272_000, 0, 0, float("nan"), None, 52_500, False, True),
    (272_000, 128_000, 0, 0.5, None, 70_000, False, False),
    (272_000, 0, 0, 0.75, 32_000, 52_500, False, False),
    (272_000, 0, 0, 0.75, 32_000, 20_000, False, True),
    (272_000, 0, 0, 0.75, None, 260_000, False, False),
    (272_000, 0, 0, 0.75, None, 52_500, True, False),
    (272_000, 128_000, 64_000, 0.75, None, 70_000, False, False),
    (272_000, 128_000, 64_000, 0.75, None, 52_500, False, True),
])
def test_recovered_request_must_fit_a_known_window(
    host_window, engine_window, threshold, percent, reported, after_tokens, unknown_host, completed,
):
    agent = _agent(get_hermes_home(), host_window, engine_window=engine_window,
                   threshold=threshold, percent=percent)
    result = _recover(agent, after_tokens, reported, unknown_host=unknown_host)
    assert result["completed"] is completed
    assert agent.client.chat.completions.create.call_count == (2 if completed else 1)
    if completed:
        assert result["final_response"] == "Recovered"
    else:
        assert result["compression_exhausted"] is True


@pytest.mark.parametrize("launch", ["default", "named"])
def test_overflow_limits_remain_in_the_owning_profile(tmp_path, monkeypatch, launch):
    homes = {"a": tmp_path / "profiles" / "a", "b": tmp_path / "profiles" / "b"}
    launch_home = tmp_path / "default" if launch == "default" else homes["b"]
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    agents = {}
    for name, reported, after, completed in [("a", 32_000, 20_000, True),
                                              ("b", 128_000, 52_500, True),
                                              ("a", 32_000, 52_500, False)]:
        token = set_hermes_home_override(homes[name])
        try:
            if name not in agents:
                agents[name] = _agent(homes[name], 272_000)
            agent = agents[name]
            agent.client.chat.completions.create.reset_mock()
            result = _recover(agent, after, reported)
            assert result["completed"] is completed
            cache = yaml.safe_load((homes[name] / "context_length_cache.yaml").read_text())
            assert cache["context_lengths"][f"{agent.model}@{agent.base_url.rstrip('/')}"] == reported
        finally:
            reset_hermes_home_override(token)
    if launch == "default":
        assert not (launch_home / "context_length_cache.yaml").exists()
