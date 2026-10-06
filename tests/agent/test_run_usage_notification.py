"""Completed provider-call usage must notify a run owner before the turn ends."""

from types import SimpleNamespace

from tests.agent.test_turn_usage_log_line import _agent, _usage
from tests.agent.test_usage_anchor import TestCodexAppServerAnchor as _CodexHarness


def test_codex_usage_notifies_only_after_counters_advance():
    from agent.codex_runtime import _record_codex_app_server_usage
    harness = _CodexHarness()
    agent = harness._agent()
    seen = []
    agent._run_usage_callback = lambda: seen.append(agent.session_total_tokens)
    _record_codex_app_server_usage(agent, harness._turn(harness._usage(input_tokens=12, output_tokens=7)))
    assert seen == [19]


def test_provider_call_notifies_only_after_counters_advance(tmp_path, monkeypatch):
    from agent.turn_usage import record_response_usage
    agent = _agent(tmp_path, monkeypatch)
    seen = []
    agent._run_usage_callback = lambda: seen.append(agent.session_total_tokens)
    try:
        record_response_usage(
            agent, SimpleNamespace(usage=_usage(0, 0, 12), id="resp", model=agent.model),
            messages=[{"role": "user", "content": "hi"}], api_call_count=1,
            api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
        )
        assert seen == [19]
    finally:
        agent.close()
