"""Real plugin delivery produces useful, content-free run diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import lifecycle, plugins
from agent.conversation_compression import _notify_context_engine_compression_complete
from agent.turn_context import PreflightCompressionTimedOut
from plugins.observability import run_metrics as metrics
from scripts.run_metrics_report import aggregate
from run_agent import AIAgent
from tests.fakes.fake_llm_provider import DropMidStream, Error, FakeLLMServer, Text
from tests.agent.test_run_agent import _make_tool_defs, _mock_response


@pytest.fixture
def enabled(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: [observability/run_metrics]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    metrics._RUNS.clear()
    metrics._CHILD_PARENTS.clear()
    plugins._reset_plugin_managers_for_tests()
    plugins.discover_plugins()
    loaded = plugins.get_plugin_manager()._plugins["observability/run_metrics"]
    assert loaded.enabled is True
    yield home
    plugins._reset_plugin_managers_for_tests()
    metrics._RUNS.clear()
    metrics._CHILD_PARENTS.clear()


def _event(turn="session:metrics-smoke:turn", **extra):
    return {"session_id": "session", "task_id": "metrics-smoke", "turn_id": turn, "platform": "cli", **extra}


def _reports(home):
    return [json.loads(path.read_text(encoding="utf-8")) for path in (home / "logs" / "run-metrics").glob("*.json")]


def test_real_lifecycle_retries_timeout_and_redacts_content(enabled):
    secret = "super-secret-canary"
    lifecycle.invoke_hook("pre_llm_call", **_event(user_message=secret))
    start = 1_800_000_000.0
    pre = _event(api_request_id="session:api:0", retry_count=0, started_at=start,
                 model="local-test-model", provider="local", context_limit_tokens=32768,
                 max_tokens=4096, approx_input_tokens=20000,
                 request={"body": {"messages": [{"content": secret}]}})
    lifecycle.invoke_hook("pre_api_request", **pre)
    running = _reports(enabled)[0]
    assert running["status"] == running["attempts"][0]["status"] == "running"
    lifecycle.invoke_hook("api_request_error", **_event(
        api_request_id="session:api:0", retry_count=0, started_at=start, ended_at=start + 40,
        api_duration=40, reason="timeout", error={"type": "ReadTimeout", "message": secret},
    ))
    lifecycle.invoke_hook("pre_api_request", **{**pre, "retry_count": 1, "started_at": start + 41})
    lifecycle.invoke_hook("post_api_request", **_event(
        api_request_id="session:api:0", started_at=start + 41, ended_at=start + 45,
        api_duration=4, first_chunk_at=start + 42, first_delta_at=start + 43,
        response_model="local-test-model", finish_reason="stop",
        usage={"input_tokens": 20200, "output_tokens": 500, "reasoning_tokens": 200},
        response={"assistant_message": {"content": secret}},
    ))
    lifecycle.invoke_hook("on_session_end", **_event(completed=True, failed=False, interrupted=False,
                                                    turn_exit_reason="text_response(finish_reason=stop)"))
    report = _reports(enabled)[0]
    assert report["status"] == "completed"
    assert report["summary"]["logical_model_calls"] == 1
    assert report["summary"]["provider_attempts"] == 2
    assert report["summary"]["retry_attempts"] == 1
    assert report["summary"]["provider_reported_input_tokens"] == 20200
    assert report["attempts"][0]["token_source"] == "unavailable"
    assert report["attempts"][1]["time_to_first_chunk_s"] == 1
    assert report["attempts"][1]["time_to_first_delta_s"] == 2
    assert report["attempts"][1]["context_limit_tokens"] == 32768
    assert aggregate(enabled / "logs" / "run-metrics")["stop_causes"] == {"completed": 1}
    assert secret not in json.dumps(report)
    assert (enabled / "logs" / "run-metrics" / f"{report['run_id']}.json").stat().st_mode & 0o777 == 0o600


def test_output_cap_compaction_tool_and_child_are_distinct(enabled):
    parent = _event()
    lifecycle.invoke_hook("pre_llm_call", **parent)
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=100,
                                                     retry_count=0, approx_input_tokens=20000))
    lifecycle.invoke_hook("post_api_request", **_event(api_request_id="session:api:0",
                                                      started_at=100, ended_at=109, api_duration=9,
                                                      finish_reason="length", usage=None))
    lifecycle.invoke_hook("pre_tool_call", **_event(tool_call_id="tool-1", tool_name="terminal", args={"secret": "canary"}))
    lifecycle.invoke_hook("post_tool_call", **_event(tool_call_id="tool-1", tool_name="terminal", status="ok",
                                                    duration_ms=2500, result="canary"))
    compressor = SimpleNamespace(on_session_start=MagicMock())
    compression_agent = SimpleNamespace(context_compressor=compressor, _current_task_id="metrics-smoke",
                                        _current_turn_id=parent["turn_id"], platform="cli")
    with patch("agent.relay_runtime.SESSION_COORDINATOR.notify_session_compacted"):
        assert _notify_context_engine_compression_complete(
            compression_agent, new_session_id="session-new", old_session_id="session")
    lifecycle.invoke_hook("subagent_start", parent_turn_id=parent["turn_id"], child_session_id="child",
                          child_role="reviewer", child_goal="secret")
    lifecycle.invoke_hook("pre_llm_call", **_event(turn="child:metrics-smoke:turn", session_id="child"))
    lifecycle.invoke_hook("on_session_end", **_event(turn="child:metrics-smoke:turn", session_id="child",
                                                    completed=True, turn_exit_reason="text_response(finish_reason=stop)"))
    lifecycle.invoke_hook("subagent_stop", parent_turn_id=parent["turn_id"], child_session_id="child",
                          child_status="completed", duration_ms=1000)
    lifecycle.invoke_hook("on_session_end", **_event(completed=True,
                                                    turn_exit_reason="text_response(finish_reason=length)"))
    reports = _reports(enabled)
    root = next(report for report in reports if not report["parent_run_id"])
    child = next(report for report in reports if report["parent_run_id"])
    assert child["parent_run_id"] == root["run_id"]
    assert root["status"] == "output_capped"
    assert root["summary"]["output_capped_attempts"] == 1
    assert root["summary"]["tool_time_s"] == 2.5
    assert root["summary"]["compactions"] == 1
    assert aggregate(enabled / "logs" / "run-metrics")["root_runs"] == 1
    assert aggregate(enabled / "logs" / "run-metrics")["child_runs"] == 1


def test_interrupted_run_has_truthful_terminal_state(enabled):
    lifecycle.invoke_hook("pre_llm_call", **_event())
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=100))
    lifecycle.invoke_hook("on_session_end", **_event(completed=False, interrupted=True,
                                                    turn_exit_reason="interrupted_by_user"))
    report = _reports(enabled)[0]
    assert report["status"] == "interrupted"
    assert report["attempts"][0]["status"] == "aborted"
    assert report["attempts"][0]["input_tokens"] is None
    assert aggregate(enabled / "logs" / "run-metrics")["stop_causes"] == {"interrupted": 1}


def test_same_retry_counter_still_records_each_physical_attempt(enabled):
    lifecycle.invoke_hook("pre_llm_call", **_event())
    now = time.time()
    request = _event(api_request_id="session:api:0", retry_count=0, started_at=now,
                     approx_input_tokens=24000, context_limit_tokens=32768)
    lifecycle.invoke_hook("pre_api_request", **request)
    # An early recovery path can restart before emitting api_request_error.
    lifecycle.invoke_hook("pre_api_request", **{**request, "started_at": now + 2})
    lifecycle.invoke_hook("post_api_request", **_event(
        api_request_id="session:api:0", ended_at=now + 4, api_duration=2,
        first_delta_at=now + 3, finish_reason="stop",
        usage={"input_tokens": 24050, "output_tokens": 40},
    ))
    lifecycle.invoke_hook("on_session_end", **_event(completed=True,
                                                    turn_exit_reason="text_response(finish_reason=stop)"))
    report = _reports(enabled)[0]
    assert report["summary"]["provider_attempts"] == 2
    assert report["summary"]["retry_attempts"] == 1
    assert report["attempts"][0]["status"] == "outcome_unobserved"
    assert report["attempts"][0]["token_source"] == "unavailable"
    assert report["attempts"][0]["estimated_context_utilization"] == 0.7324
    assert report["summary"]["measured_decode_tokens_per_s"] == 40


def test_terminal_timeout_and_fail_open_storage(enabled):
    with patch.object(metrics, "_persist", side_effect=OSError("disk full")):
        lifecycle.invoke_hook("pre_llm_call", **_event())
    now = time.time()
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=now))
    lifecycle.invoke_hook("api_request_error", **_event(
        api_request_id="session:api:0", ended_at=now + 1, reason="timeout",
        error={"type": "ReadTimeout", "message": "provider secret"},
    ))
    lifecycle.invoke_hook("on_session_end", **_event(completed=False, failed=True,
                                                    turn_exit_reason="api_error"))
    report = _reports(enabled)[0]
    assert report["status"] == "failed"
    assert report["attempts"][0]["status"] == "error"
    assert report["attempts"][0]["input_tokens"] is None
    assert aggregate(enabled / "logs" / "run-metrics")["stop_causes"] == {"timeout": 1}
    assert "provider secret" not in json.dumps(report)


def test_actual_agent_turn_writes_correlated_report(enabled):
    """Exercise Hermes's real loop and plugin discovery, not manually fabricated hook calls."""
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://example.invalid/v1",
                       model="test/model", provider="openai", session_id="run-metrics-smoke",
                       quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response(
        "Implemented and checked", usage={"prompt_tokens": 120, "completion_tokens": 18},
    )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    with (
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("small coding task", task_id="metrics-smoke")
    assert result["final_response"] == "Implemented and checked"
    report = _reports(enabled)[0]
    assert report["task_id"] == "metrics-smoke"
    assert report["session_id"] == "run-metrics-smoke"
    assert report["status"] == "completed"
    assert report["summary"]["provider_reported_input_tokens"] == 120
    assert report["summary"]["provider_reported_output_tokens"] == 18


def test_real_provider_wire_records_output_cap_before_recovery(enabled):
    """The first real HTTP/SSE response is truncated; later success must not hide it."""
    with FakeLLMServer([Text("incomplete", finish_reason="length", prompt_tokens=90,
                             completion_tokens=8), Text("complete", prompt_tokens=105,
                                                        completion_tokens=9)], api_key="test-key") as server:
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("model_tools.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(api_key="test-key", base_url=server.base_url, model="fake-model",
                            provider="custom", session_id="run-metrics-wire", max_iterations=3,
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        with (
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("small coding task", task_id="metrics-smoke")
    report = _reports(enabled)[0]
    assert report["summary"]["output_capped_attempts"] >= 1
    assert report["attempts"][0]["finish_reason"] == "length"
    assert report["attempts"][0]["input_tokens"] == 90
    assert report["attempts"][0]["output_tokens"] == 8
    assert report["attempts"][0]["time_to_first_delta_s"] is not None


def test_content_filter_is_one_physical_attempt_and_early_turn_is_terminal(enabled):
    with FakeLLMServer([Text("refused", finish_reason="content_filter")], api_key="test-key") as server:
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("model_tools.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(api_key="test-key", base_url=server.base_url, model="fake-model",
                            provider="custom", session_id="run-metrics-filter", max_iterations=1,
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        with (
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("small coding task", task_id="metrics-smoke")
    report = _reports(enabled)[0]
    assert result["completed"] is False
    assert report["status"] == "failed"
    assert report["ended_at"] is not None
    assert report["summary"]["provider_attempts"] == 1
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["finish_reason"] == "content_filter"
    assert report["attempts"][0]["input_tokens"] == 100


def test_cached_tokens_are_not_mistaken_for_total_input(enabled):
    lifecycle.invoke_hook("pre_llm_call", **_event())
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=100))
    lifecycle.invoke_hook("post_api_request", **_event(
        api_request_id="session:api:0", ended_at=101, finish_reason="stop",
        usage={"prompt_tokens": 1000, "input_tokens": 200, "cache_read_tokens": 800,
               "cache_write_tokens": 0, "output_tokens": 50},
    ))
    lifecycle.invoke_hook("on_turn_result", **_event(completed=True, turn_exit_reason="text_response(finish_reason=stop)"))
    attempt = _reports(enabled)[0]["attempts"][0]
    assert attempt["input_tokens"] == 1000
    assert attempt["uncached_input_tokens"] == 200
    assert attempt["cache_read_tokens"] == 800


def test_outer_retry_time_excludes_backoff(enabled):
    lifecycle.invoke_hook("pre_llm_call", **_event())
    for retry, start, end, error in ((0, 100, 102, True), (1, 120, 123, False)):
        lifecycle.invoke_hook("pre_api_request", **_event(
            api_request_id="session:api:0", retry_count=retry,
            started_at=100, attempt_started_at=start,
        ))
        lifecycle.invoke_hook("api_request_error" if error else "post_api_request", **_event(
            api_request_id="session:api:0", retry_count=retry,
            started_at=100, ended_at=end, api_duration=end - 100,
            reason="timeout" if error else None,
            error={"type": "ReadTimeout"} if error else None,
            finish_reason="stop" if not error else None,
            usage={"prompt_tokens": 12, "output_tokens": 3} if not error else None,
        ))
    lifecycle.invoke_hook("on_turn_result", **_event(completed=True))
    report = _reports(enabled)[0]
    assert [a["duration_s"] for a in report["attempts"]] == [2, 3]
    assert report["summary"]["model_time_s"] == 5
    assert report["summary"]["retry_attempts"] == 1


def test_stream_retry_counts_wire_attempts_not_only_outer_hooks(enabled):
    with FakeLLMServer([DropMidStream(after_chars=0), Text("recovered")], api_key="test-key") as server:
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("model_tools.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(api_key="test-key", base_url=server.base_url, model="fake-model",
                            provider="custom", session_id="run-metrics-retry", max_iterations=1,
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        with (
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("small coding task", task_id="metrics-smoke")
    report = _reports(enabled)[0]
    assert result["completed"] is True
    assert report["summary"]["logical_model_calls"] == 1
    assert report["summary"]["provider_attempts"] == 2
    assert report["summary"]["retry_attempts"] == 1
    assert [t["status"] for t in report["attempts"][0]["transport_attempts"]] == ["error", "completed"]


def test_stream_5xx_nonstream_probe_is_a_physical_request(enabled, monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    with FakeLLMServer([Error(status=503), Text("probe recovery")], api_key="test-key") as server:
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("model_tools.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(api_key="test-key", base_url=server.base_url, model="fake-model",
                            provider="custom", session_id="run-metrics-probe", max_iterations=1,
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        with (
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("small coding task", task_id="metrics-smoke")
    report = _reports(enabled)[0]
    assert result["completed"] is True
    assert len(server.main_requests()) == 2
    assert report["summary"]["provider_attempts"] == 2
    assert report["summary"]["retry_attempts"] == 1
    assert [t["status"] for t in report["attempts"][0]["transport_attempts"]] == ["error", "completed"]


def test_storage_failure_cannot_block_tools(enabled):
    lifecycle.invoke_hook("on_turn_start", **_event())
    with patch.object(metrics, "_persist", side_effect=OSError("disk full")):
        # A pre_tool_call exception becomes a policy block in Hermes, so this
        # observer never registers for that hook at all.
        assert lifecycle.invoke_hook("pre_tool_call", **_event(
            tool_call_id="tool-1", tool_name="terminal")) == []
        assert lifecycle.invoke_hook("post_tool_call", **_event(
            tool_call_id="tool-1", tool_name="terminal", status="ok", duration_ms=100)) == []


def test_raw_usage_presence_keeps_unknown_cache_buckets_null(enabled):
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://example.invalid/v1",
                        model="test/model", provider="openai", session_id="cache-presence",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    raw = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    summary = agent._usage_summary_for_api_request_hook(SimpleNamespace(usage=raw))
    assert summary["reported_usage_fields"]["cache_read_tokens"] is False
    assert summary["reported_usage_fields"]["cache_write_tokens"] is False
    lifecycle.invoke_hook("on_turn_start", **_event())
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=100))
    lifecycle.invoke_hook("post_api_request", **_event(
        api_request_id="session:api:0", ended_at=101, finish_reason="stop", usage=summary,
    ))
    lifecycle.invoke_hook("on_turn_result", **_event(completed=True))
    attempt = _reports(enabled)[0]["attempts"][0]
    assert attempt["input_tokens"] == 100
    assert attempt["output_tokens"] == 20
    assert attempt["uncached_input_tokens"] is None
    assert attempt["cache_read_tokens"] is None
    assert attempt["cache_write_tokens"] is None
    assert attempt["reasoning_tokens"] is None


def test_synthetic_length_is_not_provider_output_cap(enabled):
    lifecycle.invoke_hook("on_turn_start", **_event())
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=100))
    lifecycle.invoke_hook("post_api_request", **_event(
        api_request_id="session:api:0", ended_at=101, finish_reason="length",
        synthetic_response=True,
    ))
    lifecycle.invoke_hook("on_turn_result", **_event(completed=False, failed=True,
                                                     failure_reason="truncated"))
    report = _reports(enabled)[0]
    assert report["attempts"][0]["status"] == "incomplete_stream"
    assert report["status"] == "failed"
    assert report["summary"]["output_capped_attempts"] == 0
    assert aggregate(enabled / "logs" / "run-metrics")["stop_causes"] == {"truncated": 1}


def test_real_partial_stream_does_not_look_like_output_cap(enabled):
    with FakeLLMServer([DropMidStream(after_chars=4), Text("recovered")], api_key="test-key") as server:
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("model_tools.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(api_key="test-key", base_url=server.base_url, model="fake-model",
                            provider="custom", session_id="partial-stream", max_iterations=3,
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        with (
            patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            agent.run_conversation("small coding task", task_id="metrics-smoke")
    report = _reports(enabled)[0]
    assert report["attempts"][0]["synthetic_response"] is True
    assert report["attempts"][0]["status"] == "incomplete_stream"
    assert report["summary"]["output_capped_attempts"] == 0


def test_uninstrumented_transport_count_is_marked_incomplete(enabled):
    lifecycle.invoke_hook("on_turn_start", **_event())
    lifecycle.invoke_hook("pre_api_request", **_event(
        api_request_id="session:api:0", api_mode="codex_responses", started_at=100,
    ))
    lifecycle.invoke_hook("post_api_request", **_event(
        api_request_id="session:api:0", ended_at=101, finish_reason="stop",
    ))
    lifecycle.invoke_hook("on_turn_result", **_event(completed=True))
    report = _reports(enabled)[0]
    assert report["summary"]["provider_attempts"] == 1
    assert report["summary"]["provider_attempts_complete"] is False


def test_preflight_timeout_writes_terminal_report_before_pre_llm(enabled):
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://example.invalid/v1",
                        model="test/model", provider="openai", session_id="preflight-failure",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    with (
        patch("agent.turn_context_compaction.run_turn_start_compaction",
              side_effect=PreflightCompressionTimedOut("preflight timeout")),
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("small coding task", task_id="metrics-smoke")
    report = _reports(enabled)[0]
    assert result["failed"] is True
    assert report["status"] == "failed"
    assert report["turn_exit_reason"] == "context_compression_timeout"
    assert report["attempts"] == []
    assert aggregate(enabled / "logs" / "run-metrics")["stop_causes"] == {"timeout": 1}


def test_historical_timeout_is_not_terminal_stop_cause(enabled):
    lifecycle.invoke_hook("on_turn_start", **_event())
    lifecycle.invoke_hook("pre_api_request", **_event(api_request_id="session:api:0", started_at=100))
    lifecycle.invoke_hook("api_request_error", **_event(
        api_request_id="session:api:0", ended_at=101, reason="timeout",
        error={"type": "ReadTimeout"},
    ))
    lifecycle.invoke_hook("on_turn_result", **_event(completed=False, failed=True,
                                                     failure_reason="max_iterations"))
    report = _reports(enabled)[0]
    assert aggregate(enabled / "logs" / "run-metrics")["stop_causes"] == {"max_iterations": 1}
    assert report["attempts"][0]["error_reason"] == "timeout"
