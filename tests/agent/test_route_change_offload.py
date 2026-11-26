"""A mid-turn fallback moves the turn to a route that has never seen this prefix: its first request gets the
same cold offload a turn start on a changed route gets, before the old tool output is written there whole."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_init import _init_prompt_cache_config
from agent.context_compressor import ContextCompressor
from agent.prompt_cache_clock import CacheEvent, _route_key
from agent.prompt_cache_lifecycle import get_prompt_cache_warmer
from run_agent import AIAgent

_PRIMARY = ("openrouter", "https://openrouter.ai/api/v1", "test/primary")


def _history(n_pairs, big, chars=20_000):
    msgs = []
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "web_search", "arguments": '{"query":"x"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": chr(65 + i % 26) * chars if i in big else "ok"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def _stop_response():
    msg = SimpleNamespace(content="done", reasoning_content=None, reasoning=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="m", usage=None)


def _agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url=_PRIMARY[1], model=_PRIMARY[2], provider=_PRIMARY[0],
                    quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=6)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.save_trajectories = False
    a.compression_enabled = True
    a._session_db = None
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        a.context_compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=4)
    a.context_compressor._session_db = MagicMock()  # the prune only runs where a store archives originals
    a.context_compressor._session_id = a.session_id
    # The primary route answered this prefix moments ago: its cache is warm, so turn start prunes nothing.
    route = _route_key(*_PRIMARY)
    a._prompt_cache_clock = CacheEvent(
        route=route, destination=route, route_class="openai_spec", requested_ttl=None, sent_at=0.0,
        read_tokens=50_000, write_tokens=0, tier="5m", confidence="proven", reason="test",
        guaranteed_until=9e18, estimated_until=9e18, session_id=a.session_id)
    a._prompt_cache_compression_mark = 0
    a._prompt_cache_clock_restored = True
    return a


def test_fallback_first_request_offloads_old_tool_output():
    agent = _agent()
    history = _history(8, big=set(range(5)))
    sent = []

    def create(**kwargs):
        sent.append([dict(m) for m in kwargs["messages"]])
        if len(sent) == 1:
            return SimpleNamespace(choices=[], model="m", usage=None)  # the primary fails: fallback
        return _stop_response()

    def activate_fallback(**_kw):
        agent.provider, agent.base_url, agent.model = "nous", "https://inference.nous.example/v1", "test/fallback"
        return True

    agent.client.chat.completions.create.side_effect = create
    agent._fallback_chain = [object()]
    agent._try_activate_fallback = MagicMock(side_effect=activate_fallback)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.compaction_events.publish_prune"),
    ):
        agent.run_conversation("next question", conversation_history=history)

    assert agent._try_activate_fallback.called and len(sent) >= 2
    big = "A" * 20_000
    assert any(m.get("content") == big for m in sent[0])  # the warm primary was sent the history as is
    fallback = sent[1]
    assert not any(m.get("role") == "tool" and len(str(m.get("content"))) >= 20_000 for m in fallback)
    assert any("full result: " in str(m.get("content")) for m in fallback if m.get("role") == "tool")
    assert fallback[-1]["role"] == "user" and "next question" in str(fallback[-1]["content"])


def test_release_and_close_cancel_the_live_warmer():
    agent = _agent()
    warmer = get_prompt_cache_warmer(agent)
    for release in (agent.release_clients, agent.close):
        timer = MagicMock()
        run = SimpleNamespace(cancelled=False, lock=threading.RLock(), timer=timer, warm_count=0)
        warmer._run = run
        release()
        release()  # idempotent cleanup does not resurrect the timer
        assert run.cancelled and warmer._run is None
        timer.cancel.assert_called_once()


def test_early_turn_exit_ends_machine_warming():
    agent = _agent()
    warmer = get_prompt_cache_warmer(agent)
    result = {"final_response": "done", "messages": [], "completed": True}
    for idle_ok in (False, True):
        timer = MagicMock()
        run = SimpleNamespace(cancelled=False, lock=threading.RLock(), timer=timer, warm_count=0, idle_ok=idle_ok)
        warmer._run = run
        with patch("agent.conversation_loop._run_conversation_turn", return_value=result):
            agent.run_conversation("hello")
        assert run.cancelled is not idle_ok
        assert (warmer._run is run) is idle_ok
        assert timer.cancel.call_count == (0 if idle_ok else 1)
        warmer.cancel("test cleanup")


def test_list_cache_ttl_falls_back_to_short_tier():
    agent = SimpleNamespace(_anthropic_prompt_cache_policy=lambda: (True, False), platform="cli")
    with patch("hermes_cli.config.load_config_readonly", return_value={"prompt_caching": {"cache_ttl": []}}):
        _init_prompt_cache_config(agent)
    assert agent._cache_ttl == "5m" and not agent._cache_disabled
