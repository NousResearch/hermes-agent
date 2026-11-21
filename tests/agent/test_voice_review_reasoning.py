"""Review forks must reuse a same-route voice turn's effort (#156822)."""

import pytest

from agent.background_review import build_cache_parity_fork, release_cache_parity_fork
from run_agent import AIAgent
from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home


@pytest.mark.parametrize("separate_voice_model", [False, True])
def test_review_reasoning_tracks_voice_without_changing_session(tmp_path, monkeypatch, separate_voice_model):
    with FakeLLMServer(default_text="done") as server:
        extra = "auxiliary:\n  voice_chat:\n    reasoning_effort: none\n"
        if separate_voice_model:
            extra += ("    provider: custom\n    model: voice-model\n"
                      f"    base_url: {server.base_url}\n    api_key: sk-test-voice\n")
        home = write_hermes_home(tmp_path / "home", server.base_url, extra_config=extra)
        monkeypatch.setenv("HERMES_HOME", str(home))
        parent = AIAgent(model="fake-model", provider="custom", base_url=server.base_url,
                         api_key="sk-test-main", quiet_mode=True, skip_context_files=True,
                         skip_memory=True, enabled_toolsets=["memory"],
                         reasoning_config={"enabled": True, "effort": "high"})
        parent._voice_turn_pending = True
        voice = parent.run_conversation("spoken question")
        voice_request = server.main_requests()[-1]
        assert voice_request["reasoning_effort"] == "none"
        assert parent.reasoning_config == {"enabled": True, "effort": "high"}

        fork, _, routed = build_cache_parity_fork(parent, max_iterations=2)
        try:
            assert not routed
            fork.run_conversation("review the turn", conversation_history=voice["messages"])
            expected = "high" if separate_voice_model else "none"
            assert server.main_requests()[-1]["reasoning_effort"] == expected
        finally:
            release_cache_parity_fork(parent, fork, task="background_review")

        typed = parent.run_conversation("typed question", conversation_history=voice["messages"])
        assert server.main_requests()[-1]["reasoning_effort"] == "high"
        fork, _, _ = build_cache_parity_fork(parent, max_iterations=2)
        try:
            fork.run_conversation("review typed turn", conversation_history=typed["messages"])
            assert server.main_requests()[-1]["reasoning_effort"] == "high"
        finally:
            release_cache_parity_fork(parent, fork, task="background_review")
            parent.release_clients()


@pytest.mark.parametrize("changed", [None, "model", "provider", "base_url", "api_mode"])
def test_anthropic_review_parity_is_bound_to_the_voice_runtime(changed):
    from types import SimpleNamespace

    from agent.anthropic_adapter import build_anthropic_kwargs
    from agent.background_review import _same_model_parity_kwargs
    from agent.voice_turn_route import end_voice_turn_route

    parent = SimpleNamespace(
        provider="anthropic", model="claude-sonnet-5-5", base_url="https://api.anthropic.com",
        api_mode="anthropic_messages", reasoning_config={"enabled": False},
        _voice_route_state={"reasoning_config": {"enabled": True, "effort": "high"}},
        _turn_received_provider_response=True,
    )
    voice = build_anthropic_kwargs(parent.model, [{"role": "user", "content": "voice"}], None,
                                   4096, parent.reasoning_config)
    end_voice_turn_route(parent)
    if changed:
        setattr(parent, changed, getattr(parent, changed) + "-different")
    config = _same_model_parity_kwargs(parent)["reasoning_config"]
    review = build_anthropic_kwargs("claude-sonnet-5-5", [{"role": "user", "content": "review"}],
                                    None, 4096, config)
    if changed:
        assert review["thinking"]["type"] == "adaptive"
        assert review["output_config"] == {"effort": "high"}
    else:
        assert review["thinking"] == voice["thinking"] == {"type": "between_tools"}
        assert "output_config" not in review
    assert parent.reasoning_config == {"enabled": True, "effort": "high"}


@pytest.mark.parametrize("deferred", [False, True])
def test_scheduled_review_keeps_effort_after_next_turn(tmp_path, monkeypatch, deferred):
    import run_agent
    from agent import background_review as bg
    from agent import review_idle_queue as idle
    from agent.background_review import _same_model_parity_kwargs

    with FakeLLMServer(default_text="done") as server:
        home = write_hermes_home(tmp_path / "home", server.base_url)
        monkeypatch.setenv("HERMES_HOME", str(home))
        parent = AIAgent(model="fake-model", provider="custom", base_url=server.base_url,
                         api_key="sk-test-main", quiet_mode=True, skip_context_files=True,
                         skip_memory=True, enabled_toolsets=["memory"],
                         reasoning_config={"enabled": True, "effort": "high"})
        parent._voice_turn_pending = True
        voice = parent.run_conversation("voice")
        targets, observed = [], []
        queue = idle.ReviewIdleQueue()
        monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
        monkeypatch.setattr(idle, "QUEUE", queue)
        monkeypatch.setattr(run_agent, "_review_should_defer", lambda *_: deferred)
        monkeypatch.setattr(bg, "load_background_review_settings", lambda: (True, {}))

        class HeldThread:
            def __init__(self, *, target, **kwargs):
                targets.append(target)

            def start(self):
                pass

        with monkeypatch.context() as scheduling:
            scheduling.setattr(run_agent.threading, "Thread", HeldThread)
            parent._spawn_background_review(voice["messages"], review_memory=True)
        # The next real entry clears the parent's marker before the held worker runs.
        parent.run_conversation("typed", conversation_history=voice["messages"])
        assert parent._voice_review_reasoning is None

        def record(agent, *args, review_run=None, **kwargs):
            observed.append(_same_model_parity_kwargs(agent)["reasoning_config"])
            bg.finish_background_review_run(agent, review_run)

        monkeypatch.setattr(bg, "_run_review_in_thread", record)
        if deferred:
            item = next(iter(queue._pending.values()))
            with monkeypatch.context() as scheduling:
                scheduling.setattr(run_agent.threading, "Thread", HeldThread)
                item.context.run(queue._dispatch, item)
        assert len(targets) == 1
        targets[0]()
        assert observed == [{"enabled": False}]
        assert parent.reasoning_config == {"enabled": True, "effort": "high"}
        parent.release_clients()
