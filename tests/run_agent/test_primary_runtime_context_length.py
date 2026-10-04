"""The explicit model.context_length pin survives fallback + restore.

Fallback clears the live ``_config_context_length`` (it describes the old
runtime); restore must bring the snapshot's pin back, or the primary runs
auto-detected for the rest of the session.
"""

from types import SimpleNamespace

from agent.agent_init import _snapshot_primary_runtime
from agent.agent_runtime_helpers import _apply_primary_runtime_fields


def _agent(context_length=204800):
    return SimpleNamespace(
        model="m",
        provider="p",
        requested_provider="p",
        base_url="https://x.example.com/v1",
        api_mode="chat_completions",
        api_key="k",
        request_overrides={},
        _client_kwargs={},
        _use_prompt_caching=True,
        _use_native_cache_layout=False,
        _reasoning_echo_flag=False,
        _config_context_length=context_length,
        context_compressor=SimpleNamespace(
            model="m", base_url="https://x.example.com/v1", api_key="k",
            provider="p", context_length=1, threshold_tokens=2,
        ),
    )


class TestContextLengthPinRestore:
    def test_snapshot_carries_pin(self):
        agent = _agent()
        _snapshot_primary_runtime(agent)
        assert agent._primary_runtime["config_context_length"] == 204800

    def test_snapshot_carries_absent_pin_as_none(self):
        agent = _agent(context_length=None)
        _snapshot_primary_runtime(agent)
        assert agent._primary_runtime["config_context_length"] is None

    def test_apply_reinstates_pin(self):
        agent = _agent()
        _snapshot_primary_runtime(agent)
        agent._config_context_length = None  # fallback cleared it
        _apply_primary_runtime_fields(agent, agent._primary_runtime)
        assert agent._config_context_length == 204800

    def test_apply_without_key_leaves_live_value(self):
        agent = _agent()
        rt = {
            "model": "m", "provider": "p", "base_url": "https://x.example.com/v1",
            "api_mode": "chat_completions", "api_key": "k", "request_overrides": {},
            "client_kwargs": {},
        }
        agent._config_context_length = 123
        _apply_primary_runtime_fields(agent, rt)
        assert agent._config_context_length == 123
