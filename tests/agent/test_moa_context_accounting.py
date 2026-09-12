"""MoA context pressure uses one request; billing includes every model attempt."""

import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("finish_reason", ["stop", "length", "content_filter"])
def test_moa_response_accounts_spend_without_inflating_context(finish_reason, monkeypatch):
    from agent.moa_loop import MoAClient
    from agent.turn_response_check import check_api_response
    from agent.turn_retry_state import TurnRetryState
    from agent.usage_pricing import CanonicalUsage
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", provider="custom", model="test/model",
        base_url="http://localhost:12345/v1", api_mode="chat_completions",
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False,
    )
    agent.provider = "moa"
    agent.client = MoAClient("test")
    advisor_usage = CanonicalUsage(input_tokens=230_000, output_tokens=2_000)
    agent.client.chat.completions._pending_reference_usage = advisor_usage
    compressor = agent.context_compressor
    compressor.context_length = 262_144
    compressor.threshold_tokens = 180_000
    compressor._verify_compaction_cleared_threshold = True
    messages = [{"role": "user", "content": "Continue the task."}]
    response = SimpleNamespace(
        id="test-response", model="test/model",
        choices=[SimpleNamespace(
            index=0, finish_reason=finish_reason,
            message=SimpleNamespace(content="Partial answer.", tool_calls=[], refusal=None),
        )],
        usage=SimpleNamespace(prompt_tokens=8_000, completion_tokens=100, total_tokens=8_100),
    )
    monkeypatch.setattr(agent, "_persist_session", lambda *args: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *args: None)
    try:
        verdict = check_api_response(
            agent, response=response, _retry=TurnRetryState(), thinking_spinner=None,
            messages=messages, api_messages=list(messages), api_kwargs={},
            active_system_prompt="", conversation_history=[], finish_reason=None,
            retry_count=0, max_retries=3, compression_attempts=3,
            max_compression_attempts=3, length_continue_retries=0,
            truncated_response_parts=[], truncated_tool_call_retries=0,
            current_turn_user_idx=0, api_call_count=1, api_request_id="test-request",
            api_start_time=time.time(), effective_task_id="test-task", turn_id="test-turn",
            _preflight_compression_blocked=True, _last_preflight_pressure=238_000,
        )
        assert compressor.last_prompt_tokens == 8_000
        assert compressor.last_completion_tokens == 100
        assert agent._last_turn_usage["total_tokens"] == 8_100
        assert agent.session_prompt_tokens == 8_000 + advisor_usage.prompt_tokens
        assert agent.session_output_tokens == 100 + advisor_usage.output_tokens
        assert agent.session_api_calls == 1
        assert agent.client.consume_reference_usage()[0].total_tokens == 0
        assert verdict.compression_attempts == 0
        assert verdict._preflight_compression_blocked is False
        assert verdict._last_preflight_pressure is None
    finally:
        agent.close()


@pytest.mark.parametrize("task", ["moa_reference", "moa_aggregator"])
@pytest.mark.parametrize("model, cap_key", [
    ("custom-model", "max_tokens"), ("gpt-5", "max_completion_tokens"),
])
def test_moa_request_builder_preserves_explicit_output_budget(task, model, cap_key):
    from agent.auxiliary_client import _build_call_kwargs

    messages = [{"role": "user", "content": "Continue the answer."}]
    for cap in (None, 8_192, 16_384, 32_768):
        request = _build_call_kwargs(
            "custom", model, messages, max_tokens=cap,
            base_url="https://model.example/v1", task=task,
        )
        assert request.get(cap_key) == cap
        other_key = "max_completion_tokens" if cap_key == "max_tokens" else "max_tokens"
        assert other_key not in request


@pytest.fixture
def accounting_attempt(monkeypatch):
    from unittest.mock import Mock
    from agent.moa_loop import MoAClient
    from agent.turn_retry_state import TurnRetryState
    from agent.usage_pricing import CanonicalUsage
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", provider="custom", model="test/model",
                    base_url="http://localhost:12345/v1", api_mode="chat_completions",
                    enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                    skip_memory=True, save_trajectories=False)
    agent.provider = "moa"
    client = MoAClient("test")
    agent.client = client
    facade = client.chat.completions
    facade._pending_reference_usage = CanonicalUsage(input_tokens=230_000, output_tokens=2_000)
    facade._pending_reference_cost = 0.25
    facade._pending_trace = {"preset": "test"}
    monkeypatch.setattr(agent, "_persist_session", lambda *args: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *args: None)
    database = Mock()
    agent._session_db = database
    agent._session_db_created = True
    agent.session_id = "accounting-test"
    messages = [{"role": "user", "content": "Continue the task."}]
    kwargs = dict(_retry=TurnRetryState(), thinking_spinner=None, messages=messages,
                  api_messages=list(messages), api_kwargs={}, active_system_prompt="",
                  conversation_history=[], finish_reason=None, retry_count=0, max_retries=3,
                  compression_attempts=3, max_compression_attempts=3, length_continue_retries=0,
                  truncated_response_parts=[], truncated_tool_call_retries=0, current_turn_user_idx=0,
                  api_call_count=1, api_request_id="attempt-1", api_start_time=time.time(),
                  effective_task_id="test-task", turn_id="test-turn",
                  _preflight_compression_blocked=True, _last_preflight_pressure=238_000)
    try:
        yield agent, client, database, kwargs
    finally:
        agent._session_db = None
        agent.close()


@pytest.mark.parametrize("finish", ["stop", "content_filter", "length"])
def test_missing_acting_usage_still_records_advisors(accounting_attempt, finish):
    from agent.turn_response_check import check_api_response

    agent, client, database, kwargs = accounting_attempt
    response = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content="Partial answer.", tool_calls=[], refusal=None))])
    check_api_response(agent, response=response, **kwargs)
    assert agent.session_prompt_tokens == 230_000
    assert agent.session_output_tokens == 2_000
    assert agent.session_estimated_cost_usd == pytest.approx(0.25)
    assert agent.session_api_calls == 1
    assert client.consume_reference_usage()[0].total_tokens == 0
    assert client.chat.completions._pending_trace is None
    assert not agent._last_turn_usage
    assert agent.context_compressor.last_prompt_tokens != 230_000
    database.queue_token_counts.assert_called_once()
    persisted = database.queue_token_counts.call_args.kwargs
    assert persisted["input_tokens"] == 230_000
    assert persisted["estimated_cost_usd"] == pytest.approx(0.25)


@pytest.mark.parametrize("recovery, as_dict", [
    ("retry", False), ("fallback", False), ("exhaustion", False), ("fallthrough", False),
    ("retry", True), ("exhaustion", True),
])
def test_malformed_attempt_is_counted_before_recovery(accounting_attempt, monkeypatch, recovery, as_dict):
    from agent import turn_response_check as check

    agent, client, database, kwargs = accounting_attempt
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=8000, completion_tokens=100), choices=[])
    if as_dict:
        response = {"usage": {"prompt_tokens": 8000, "completion_tokens": 100}, "choices": []}
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda: recovery == "fallback")
    monkeypatch.setattr("agent.conversation_loop._arm_fallback_restart", lambda *a, **kw: "")
    if recovery == "exhaustion":
        kwargs["retry_count"] = 2
    if recovery == "fallthrough":
        def recover(*args, **kw):
            assert agent.session_api_calls == 1
            assert agent.session_prompt_tokens == 238_000
            response.choices = [SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content="Recovered.", tool_calls=[], refusal=None))]
            return check.InvalidResponseVerdict("fallthrough", None, "", 0, 3)
        monkeypatch.setattr(check, "retry_invalid_response", recover)
    verdict = check.check_api_response(agent, response=response, **kwargs)
    assert verdict.action == {"retry": "continue", "fallback": "break", "exhaustion": "return",
                              "fallthrough": "break"}[recovery]
    assert agent.session_prompt_tokens == 238_000
    assert agent.session_output_tokens == 2100
    assert agent.session_api_calls == 1
    assert client.consume_reference_usage()[0].total_tokens == 0
    assert client.chat.completions._pending_trace is None
    database.queue_token_counts.assert_called_once()


@pytest.mark.parametrize("cap_key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("cap", [8192, 16384, 32768])
@pytest.mark.parametrize("stream", [False, True])
def test_moa_facade_sends_acting_cap_to_http_transport(tmp_path, monkeypatch, cap_key, cap, stream):
    import json
    import httpx
    import openai
    from agent import auxiliary_client
    from agent.moa_loop import MoAClient

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("""moa:
  presets:
    test:
      reference_models:
        - provider: custom
          model: advisor
      aggregator:
        provider: custom
        model: actor
""", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    requests = []
    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0,
                     "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                         "delta": {"role": "assistant", "content": "Done."}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Done."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
    with httpx.Client(transport=httpx.MockTransport(transport)) as http_client:
        sdk = openai.OpenAI(api_key="test-key", base_url="http://model.test/v1", http_client=http_client)
        monkeypatch.setattr(auxiliary_client, "_get_cached_client", lambda provider, model, **kw: (sdk, model))
        client = MoAClient("test")
        response = client.chat.completions.create(
            model="test", messages=[{"role": "user", "content": "Proceed."}],
            stream=stream, **{cap_key: cap})
        if stream:
            assert list(response)
    acting = [item for item in requests if item["model"] == "actor"]
    advisors = [item for item in requests if item["model"] == "advisor"]
    assert len(acting) == len(advisors) == 1
    assert acting[0]["max_tokens"] == cap
    assert "max_completion_tokens" not in acting[0]


@pytest.mark.parametrize("has_usage", [True, False])
def test_context_callback_failure_keeps_attempt_accounting(accounting_attempt, monkeypatch, has_usage):
    from agent.turn_usage import record_response_usage

    agent, client, database, kwargs = accounting_attempt
    compressor = agent.context_compressor
    compressor.awaiting_real_usage_after_compression = True
    def broken_context(usage):
        raise RuntimeError("Context plug-in failed")
    monkeypatch.setattr(compressor, "update_from_response", broken_context)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=8000, completion_tokens=100) if has_usage else None)
    with pytest.raises(RuntimeError, match="Context plug-in failed"):
        record_response_usage(agent, response, messages=kwargs["messages"], api_call_count=1,
                              api_duration=1.0, compression_attempts=1, max_compression_attempts=3)
    assert agent.session_prompt_tokens == 230_000 + (8000 if has_usage else 0)
    assert agent.session_estimated_cost_usd == pytest.approx(0.25)
    assert client.consume_reference_usage()[0].total_tokens == 0
    assert client.chat.completions._pending_trace is None
    database.queue_token_counts.assert_called_once()
