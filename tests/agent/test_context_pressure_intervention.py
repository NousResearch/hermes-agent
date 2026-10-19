"""Provider-confirmed context-pressure intervention regressions."""
import threading
from types import SimpleNamespace

from agent.usage_pricing import CanonicalUsage


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="k", base_url="https://inference-api.nousresearch.com/v1",
        provider="nous", api_mode="chat_completions", model="anthropic/claude-fable-5.1",
        session_id="pressure", platform="cli", quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False, enabled_toolsets=["file"],
    )
    agent.context_compressor.context_length = 100
    return agent


def _response(prompt_tokens):
    return SimpleNamespace(usage={
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 1,
        "total_tokens": prompt_tokens + 1,
    })


def _record(agent, response):
    from agent.turn_usage import record_response_usage

    return record_response_usage(
        agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=0.1, compression_attempts=0, max_compression_attempts=3,
    )


def test_provider_usage_warns_once_at_first_sixty_percent_crossing(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    warnings = []
    agent.status_callback = lambda kind, text: warnings.append((kind, text))
    try:
        _record(agent, _response(59))
        _record(agent, _response(60))
        _record(agent, _response(80))
        _record(agent, _response(20))
        _record(agent, _response(60))
    finally:
        agent.close()

    assert len(warnings) == 1
    kind, text = warnings[0]
    assert kind == "context_pressure"
    assert "60%" in text
    assert "Compress now" in text
    assert "Start new session" in text
    assert "Continue knowingly" in text


def test_throwing_status_callback_does_not_fail_turn_or_spend_warning_without_fallback(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    delivered = []

    def broken_callback(_kind, _text):
        raise RuntimeError("status sink disconnected")

    def broken_printer(_text):
        raise OSError("stdout closed")

    agent.status_callback = broken_callback
    agent._safe_print = broken_printer
    try:
        _record(agent, _response(60))
        assert not getattr(agent, "_context_pressure_intervention_emitted", False)

        agent._safe_print = delivered.append
        _record(agent, _response(60))
    finally:
        agent.close()

    assert agent._context_pressure_intervention_emitted
    assert len(delivered) == 1
    assert "Compress now" in delivered[0]


def test_failed_status_delivery_uses_cli_fallback_before_spending_warning(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    delivered = []
    agent.status_callback = lambda _kind, _text: False
    agent._safe_print = delivered.append
    try:
        _record(agent, _response(60))
    finally:
        agent.close()

    assert agent._context_pressure_intervention_emitted
    assert len(delivered) == 1
    assert "Compress now" in delivered[0]


def test_api_status_wrapper_preserves_failed_receipt_for_cli_fallback_retry(tmp_path, monkeypatch):
    """The API observer wrapper must not turn a failed durable receipt into success."""
    from gateway.session_api_turn import wire_api_observers

    agent = _agent(tmp_path, monkeypatch)
    fallback_attempts = []

    def failed_fallback(text):
        fallback_attempts.append(text)
        raise OSError("stdout closed")

    authority = SimpleNamespace(
        sessions={"pressure": SimpleNamespace(event_stream=SimpleNamespace(lock=threading.RLock(), execution={}))},
        check_approval_generation=lambda _session_id, _generation: None,
    )
    agent.status_callback = lambda _kind, _text: False
    agent._safe_print = failed_fallback
    wire_api_observers(agent, (authority, "pressure", 1), want_interim=False)
    try:
        _record(agent, _response(60))
    finally:
        agent.close()

    assert len(fallback_attempts) == 1
    assert not getattr(agent, "_context_pressure_intervention_emitted", False)


def test_provider_usage_rearms_only_at_session_boundary(tmp_path, monkeypatch):
    """A /new's reset lifecycle re-arms the provider-confirmed one-shot warning."""
    agent = _agent(tmp_path, monkeypatch)
    warnings = []
    agent.status_callback = lambda kind, text: warnings.append((kind, text))
    try:
        _record(agent, _response(60))
        _record(agent, _response(20))
        _record(agent, _response(60))

        # ``HermesCLI.new_session()`` invokes this exact live-agent boundary seam
        # after assigning the fresh session id. A lower occupancy or compression
        # must not re-arm the latch by itself.
        agent.session_id = "pressure-after-new"
        agent.reset_session_state()
        _record(agent, _response(60))
    finally:
        agent.close()

    assert [kind for kind, _text in warnings] == ["context_pressure", "context_pressure"]


def test_missing_or_invalid_provider_usage_never_warns(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    warnings = []
    agent.status_callback = lambda kind, text: warnings.append((kind, text))
    try:
        _record(agent, SimpleNamespace(usage=None))
        _record(agent, _response(0))
        agent.context_compressor.context_length = 0
        _record(agent, _response(100))
    finally:
        agent.close()

    assert warnings == []


def test_moa_advisor_usage_does_not_raise_context_pressure(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    warnings = []
    agent.status_callback = lambda kind, text: warnings.append((kind, text))

    class MoAClient:
        def consume_reference_usage(self):
            return CanonicalUsage(input_tokens=10_000, output_tokens=1), 0.0

    agent.client = MoAClient()
    try:
        _record(agent, _response(59))
    finally:
        agent.close()

    assert warnings == []
    assert agent.context_compressor.last_prompt_tokens == 59
    assert agent._last_prompt_size_tokens == 59
    assert agent.session_prompt_tokens > 10_000
