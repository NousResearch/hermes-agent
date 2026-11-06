"""``pre_llm_call`` and ``post_llm_call`` tell a plugin where the turn came from (``turn_origin``).

A gateway agent's ``sender_id`` is the session principal, not always the turn's author, so the origin
is read per turn: a cached agent's next turn without one gets None, never the previous value. Fork
turns pass their own origin so the per-turn reset keeps their ``origin=`` log tag (#118693)."""
import logging
from types import SimpleNamespace

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def hooks(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    ctx = PluginContext(PluginManifest(name="memory-attribution", source="user"), mgr)
    seen = {"pre_llm_call": [], "post_llm_call": [], "narrow": []}
    ctx.register_hook("pre_llm_call", lambda **kw: seen["pre_llm_call"].append(kw))
    ctx.register_hook("post_llm_call", lambda **kw: seen["post_llm_call"].append(kw))

    def narrow(session_id, user_message, conversation_history, is_first_turn, model, platform):
        seen["narrow"].append(platform)

    ctx.register_hook("pre_llm_call", narrow)  # a fixed signature never receives the new kwarg
    return seen


@pytest.fixture
def agent(monkeypatch):
    from run_agent import AIAgent

    class _Completions:
        def create(self, **_kw):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, reasoning=None, reasoning_content=None),
                finish_reason="stop")], usage=None, model="test-model")

    monkeypatch.setattr("agent.process_bootstrap.OpenAI",
                        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    built = AIAgent(model="test-model", api_key="k", base_url="http://localhost:1/v1", platform="telegram",
                    user_id="owner-42", max_iterations=2, quiet_mode=True, skip_memory=True)
    built._disable_streaming = True
    return built


def test_both_hooks_receive_the_turn_origin_through_plugin_dispatch(hooks, agent):
    agent.run_conversation("typed in Desktop", turn_origin="viewer")

    [pre], [post] = hooks["pre_llm_call"], hooks["post_llm_call"]
    assert (pre["turn_origin"], pre["sender_id"]) == ("viewer", "owner-42")
    assert post["turn_origin"] == "viewer"
    assert hooks["narrow"] == ["telegram"]


def test_a_cached_agents_next_turn_without_an_origin_gets_none(hooks, agent):
    agent.run_conversation("a heartbeat", turn_origin="automation")
    agent.run_conversation("an older host passes nothing")

    assert [kw["turn_origin"] for kw in hooks["pre_llm_call"]] == ["automation", None]
    assert [kw["turn_origin"] for kw in hooks["post_llm_call"]] == ["automation", None]
    assert agent._turn_origin is None


@pytest.mark.parametrize("origin", ["background_review", "side_question"])
def test_a_fork_turn_keeps_its_origin_log_tag(agent, caplog, origin):
    agent._turn_origin = origin  # build_cache_parity_fork's stamp; the run then passes the same value
    with caplog.at_level(logging.INFO, logger="run_agent"):
        agent.run_conversation("review", turn_origin=origin)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(("conversation turn:", "Turn ended:"))]
    assert lines and all(f"origin={origin}" in line for line in lines), lines


def test_the_background_review_run_passes_the_forks_origin(monkeypatch):
    """The real review spawn: the fork's own ``run_conversation`` call carries its origin."""
    import run_agent as run_agent_module
    from run_agent import AIAgent
    from tests.agent.test_background_review import ImmediateThread, _bare_agent

    runs = []

    class _ReviewFork:
        def __init__(self, **_kw):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            runs.append(kwargs)

        def release_clients(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", _ReviewFork)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    AIAgent._spawn_background_review(_bare_agent(), messages_snapshot=[{"role": "user", "content": "hi"}],
                                     review_memory=True)

    assert [run.get("turn_origin") for run in runs] == ["background_review"]
