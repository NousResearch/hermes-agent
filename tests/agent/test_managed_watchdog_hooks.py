"""Wiring tests for the wedged-child watchdog hooks (issue #104050).

The retry path reports managed-endpoint outcomes to
``hermes_cli.local_runtime.supervisor.report_inference_result``; behavior
lives in the supervisor tests — here only proves the two call sites fire with
the right arguments, and stay silent for non-managed backends.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from agent.error_classifier import FailoverReason

MANAGED_BASE = "http://127.0.0.1:18434/v1"


def _classified(reason=FailoverReason.server_error):
    return SimpleNamespace(reason=reason, is_auth=False, retryable=True,
                           should_compress=False)


def _retry_state():
    return SimpleNamespace(auth_failover_attempted=True,
                           copilot_stale_cred_retry_attempted=True)


def test_failure_hook_reports_managed_server_error(monkeypatch):
    from agent import turn_recovery
    import hermes_cli.local_runtime.supervisor as sup_mod

    calls = []
    monkeypatch.setattr(sup_mod, "report_inference_result",
                        lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(turn_recovery, "is_output_cap_error", lambda *a, **k: False)
    monkeypatch.setattr(turn_recovery, "parse_available_output_tokens_from_error",
                        lambda *a, **k: None)
    monkeypatch.setattr(turn_recovery, "is_zai_coding_overload_error", lambda **k: False)

    agent = SimpleNamespace(provider="openai", compression_enabled=True,
                            _credential_pool=None, _fallback_chain=[],
                            _fallback_index=0)
    turn_recovery.route_classified_error(
        agent, SimpleNamespace(status_code=500), _classified(),
        _retry_state(), error_msg="Compute error", error_context=None,
        recovered_with_pool=False, base_url=MANAGED_BASE, model="m",
        messages=[], api_messages=[], system_message=None,
        active_system_prompt=None, conversation_history=[], retry_count=0,
        max_retries=3, compression_attempts=0, max_compression_attempts=3,
        api_call_count=1, effective_task_id=None)
    assert calls == [((MANAGED_BASE, "m"), {"ok": False, "reason": "server_error"})]


def test_failure_hook_silent_for_cloud_backends(monkeypatch):
    from agent import turn_recovery
    import hermes_cli.local_runtime.supervisor as sup_mod

    calls = []
    monkeypatch.setattr(sup_mod, "report_inference_result",
                        lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(turn_recovery, "is_output_cap_error", lambda *a, **k: False)
    monkeypatch.setattr(turn_recovery, "parse_available_output_tokens_from_error",
                        lambda *a, **k: None)
    monkeypatch.setattr(turn_recovery, "is_zai_coding_overload_error", lambda **k: False)

    agent = SimpleNamespace(provider="openai", compression_enabled=True,
                            _credential_pool=None, _fallback_chain=[],
                            _fallback_index=0)
    turn_recovery.route_classified_error(
        agent, SimpleNamespace(status_code=500), _classified(),
        _retry_state(), error_msg="Compute error", error_context=None,
        recovered_with_pool=False, base_url="https://api.openai.com/v1",
        model="gpt-x", messages=[], api_messages=[], system_message=None,
        active_system_prompt=None, conversation_history=[], retry_count=0,
        max_retries=3, compression_attempts=0, max_compression_attempts=3,
        api_call_count=1, effective_task_id=None)
    assert calls == []


def test_success_hook_resets_managed_streak(monkeypatch):
    from agent import turn_response_check
    import hermes_cli.local_runtime.supervisor as sup_mod

    calls = []
    monkeypatch.setattr(sup_mod, "report_inference_result",
                        lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(turn_response_check, "stop_thinking_spinner",
                        lambda agent, spinner: None)
    monkeypatch.setattr(turn_response_check, "record_response_usage",
                        lambda agent, response, **k: SimpleNamespace(
                            compression_attempts=0, rearmed=False))
    monkeypatch.setattr("agent.turn_recovery.validate_response_shape",
                        lambda agent, response: (False, []))
    monkeypatch.setattr("agent.relay_llm.complete_logical_call",
                        lambda *a, **k: None)

    transport = SimpleNamespace(
        normalize_response=lambda r: SimpleNamespace(finish_reason="stop"))
    agent = SimpleNamespace(
        api_mode="chat_completions", quiet_mode=True, verbose_logging=False,
        provider="custom", base_url=MANAGED_BASE, model="m", log_prefix="",
        _get_transport=lambda: transport,
        _should_treat_stop_as_truncated=lambda *a: False,
        _touch_activity=lambda *a, **k: None)
    verdict = turn_response_check.check_api_response(
        agent, response=SimpleNamespace(), _retry=SimpleNamespace(has_retried_429=False),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt=None, conversation_history=[], finish_reason="stop",
        retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=3, length_continue_retries=0,
        truncated_response_parts=[], truncated_tool_call_retries=0,
        current_turn_user_idx=0, api_call_count=1, api_request_id="t",
        api_start_time=time.time(), effective_task_id=None, turn_id="t",
        _preflight_compression_blocked=False, _last_preflight_pressure=None)
    assert verdict.action == "break"
    assert calls == [((MANAGED_BASE, "m"), {"ok": True})]
