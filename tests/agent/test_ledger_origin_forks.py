"""The two review forks carry the markers the skill ledger reads for ``origin`` and ``route``.

``build_cache_parity_fork`` stores its routing decision on the fork (``_review_routed``); the
curator's consolidation fork is tagged ``_ledger_origin = "curator"``. Without these markers the
ledger cannot say which fork, or which routing, made a write.
"""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest


def _parent(agent_cls):
    """Minimal parent agent: just what build_cache_parity_fork reads on both paths."""
    import datetime as _dt

    agent = object.__new__(agent_cls)
    agent.model, agent.provider, agent.platform = "parent-model", "anthropic", "test"
    agent.session_id = "sess-1"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = agent._user_profile_enabled = False
    agent._memory_nudge_interval = agent._skill_nudge_interval = 5
    agent._cached_system_prompt = "SYSTEM"
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent.enabled_toolsets = agent.disabled_toolsets = None
    agent.reasoning_config = None
    agent.prefill_messages = []
    agent.providers_allowed = agent.providers_ignored = agent.providers_order = None
    agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    return agent


class _Recorder:
    """Stands in for the fork AIAgent; accepts the attributes build_cache_parity_fork assigns."""

    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.tools = None
        self.valid_tool_names = set()


@pytest.mark.parametrize("routed", [True, False])
def test_cache_parity_fork_records_its_routing_decision(routed):
    import run_agent
    from agent.background_review import build_cache_parity_fork

    parent = _parent(run_agent.AIAgent)  # built from the real class, before the fork class is patched
    runtime = {"routed": routed, "model": "other-model" if routed else None}
    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch("agent.background_review._resolve_review_runtime", lambda *a, **k: runtime):
        fork, _rt, returned = build_cache_parity_fork(parent, max_iterations=5)

    assert returned is routed
    assert getattr(fork, "_review_routed", None) is routed


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import tools.skill_usage as usage
    importlib.reload(usage)
    from agent import curator
    importlib.reload(curator)
    return curator


def test_curator_consolidation_fork_is_tagged_for_the_ledger(curator_env, monkeypatch):
    seen = {}

    class _StubAgent:
        def __init__(self, *args, **kwargs):
            self._memory_write_origin = "assistant_tool"
            self._memory_nudge_interval = self._skill_nudge_interval = 0
            self._session_messages = []

        def run_conversation(self, user_message=None, **kwargs):
            seen["write_origin"] = self._memory_write_origin
            seen["ledger_origin"] = getattr(self, "_ledger_origin", None)
            return {"final_response": "ok"}

        def close(self):
            pass

    cfg = {"model": {"provider": "openai-api", "default": "gpt-4o-mini"}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kw: {"provider": "openai-api", "api_key": "k", "base_url": "https://x.test/v1"})
    monkeypatch.setattr("run_agent.AIAgent", _StubAgent)

    meta = curator_env._run_llm_review("review prompt")

    assert meta.get("error") is None, meta.get("error")
    assert seen == {"write_origin": "background_review", "ledger_origin": "curator"}
