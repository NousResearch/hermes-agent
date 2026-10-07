"""An opt-in managed observer sees every accounted response before the next model call."""
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="test-model", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                   enabled_toolsets=["file"])


def _record(agent, usage, served_model="test-model"):
    from agent.turn_usage import record_response_usage
    record_response_usage(agent, SimpleNamespace(usage=usage, id=None, model=served_model),
                          messages=[{"role": "user", "content": "hello"}], api_call_count=1,
                          api_duration=0.1, compression_attempts=0, max_compression_attempts=3)


def test_malformed_cache_and_mismatched_raw_total_are_not_verified(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    events = []
    agent._managed_turn_usage_callback = lambda usage, **meta: events.append(meta)
    try:
        for raw in (
            SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=-3), completion_tokens_details=None),
            SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=999,
                            prompt_tokens_details=None, completion_tokens_details=None),
            SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=None,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=20), completion_tokens_details=None),
        ):
            _record(agent, raw)
        assert [item["raw_usage_complete"] for item in events] == [False, False, False]
    finally:
        agent.close()


def test_malformed_or_excessive_reasoning_is_not_verified(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    events = []
    agent._managed_turn_usage_callback = lambda usage, **meta: events.append(meta)
    try:
        for detail in (-3, "bad", 20):
            _record(agent, SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12,
                                          completion_tokens_details=SimpleNamespace(reasoning_tokens=detail)))
        assert [item["raw_usage_complete"] for item in events] == [False, False, False]
    finally:
        agent.close()


def test_partial_raw_usage_and_served_model_mismatch_are_reported_as_unverified(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    events = []
    agent._managed_turn_usage_callback = lambda usage, **meta: events.append(meta)
    try:
        _record(agent, SimpleNamespace(prompt_tokens=10, completion_tokens=None,
                                       prompt_tokens_details=None), served_model="test-model")
        _record(agent, SimpleNamespace(prompt_tokens=10, completion_tokens=2,
                                       prompt_tokens_details=None, completion_tokens_details=None),
                served_model="different-model")
        assert events[0]["raw_usage_complete"] is False
        assert events[0]["served_model"] == "test-model"
        assert events[1]["raw_usage_complete"] is True
        assert events[1]["served_model"] == "different-model"
    finally:
        agent.close()


def test_opt_in_usage_hook_sees_measured_then_missing_without_session_baseline(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    events = []
    agent.session_total_tokens = 700  # an older turn must not enter this callback's per-call delta
    agent._managed_turn_usage_callback = lambda usage, **meta: events.append((usage, meta))
    try:
        _record(agent, SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14,
                                       prompt_tokens_details=None, completion_tokens_details=None))
        _record(agent, None)
        assert len(events) == 2
        assert events[0][0].total_tokens == 14
        assert events[0][1] == {"model": "test-model", "served_model": "test-model", "raw_usage_complete": True}
        assert events[1] == (None, {"model": "test-model", "served_model": "test-model", "raw_usage_complete": False})
        del agent._managed_turn_usage_callback
        _record(agent, None)  # ordinary turns have no observer
        assert len(events) == 2
    finally:
        agent.close()
