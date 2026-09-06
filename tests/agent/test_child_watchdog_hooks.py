"""Hook contracts for the wedged-child watchdog (issue #104050).

The agent turn loop feeds the supervisor one outcome per inference: failures
in ``handle_api_error`` (server_error only), successes in
``check_api_response``. Per the agent rubric: patch the binding the phase
actually reads, assert observable behavior (counter moves / doesn't), never
snapshot prompt text.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.error_classifier import FailoverReason


@pytest.fixture
def managed_agent(monkeypatch, tmp_path):
    """Agent double on the managed endpoint with a live recording supervisor.

    Real imports, temp HERMES_HOME, real state file (ownership guard needs a
    live pid) — the only doubles are the agent itself and the supervisor
    object (no router process to spawn).
    """
    import json
    import os

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.local_runtime.child_watchdog import ChildWatchdog
    from hermes_cli.local_runtime.supervisor import state_path

    port = 18431
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{port}/v1",
        "api_key": "sk-managed", "pid": os.getpid(),
    }), encoding="utf-8")

    sup = SimpleNamespace(
        _wd=ChildWatchdog(cooldown_s=0),
        note_inference_result=lambda model_id, ok: None,
    )

    def _note(model_id, ok):
        sup._wd.note_result(model_id, ok=ok)

    sup.note_inference_result = _note
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", sup)
    agent = SimpleNamespace(
        provider="llamacpp", base_url=f"http://127.0.0.1:{port}/v1",
        model="m", log_prefix="",
    )
    return agent, sup


def test_failure_hook_records_managed_server_error(managed_agent, monkeypatch):
    """handle_api_error with a classified server_error feeds one failure to
    THIS process's supervisor for the managed model — and nothing else."""
    from agent import turn_api_error

    agent, sup = managed_agent
    agent.thinking_callback = None
    agent._extract_api_error_context = lambda e: {}
    calls = []
    agent._invoke_api_request_error_hook = lambda **k: calls.append(k)

    classified = SimpleNamespace(
        reason=FailoverReason.server_error, status_code=500, retryable=True,
        should_compress=False, should_rotate_credential=False,
        should_fallback=False,
    )
    monkeypatch.setattr(turn_api_error, "classify_api_error",
                        lambda *a, **k: classified)
    # Patch the bindings handle_api_error actually reads (top-level imports
    # in agent/turn_api_error), not the source module.
    monkeypatch.setattr(turn_api_error, "recover_before_classification",
                        lambda *a, **k: (False, "sys"))
    monkeypatch.setattr(turn_api_error, "recover_after_classification",
                        lambda *a, **k: (True, None))

    verdict = turn_api_error.handle_api_error(
        agent, api_error=RuntimeError("Compute error"), _retry=SimpleNamespace(),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        system_message="sys", active_system_prompt="sys", conversation_history=[],
        approx_tokens=1, retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=1, api_call_count=1, api_request_id="r1",
        api_start_time=0.0, effective_task_id="t", turn_id="u",
    )
    assert verdict.action == "continue"
    assert sup._wd.consecutive_failures("m") == 1


def test_failure_hook_ignores_non_server_error(managed_agent, monkeypatch):
    """Rate-limit (and friends) never touch the watchdog streak."""
    from agent import turn_api_error

    agent, sup = managed_agent
    agent.thinking_callback = None
    agent._extract_api_error_context = lambda e: {}
    agent._invoke_api_request_error_hook = lambda **k: None

    classified = SimpleNamespace(
        reason=FailoverReason.rate_limit, status_code=429, retryable=True,
        should_compress=False, should_rotate_credential=False,
        should_fallback=True,
    )
    monkeypatch.setattr(turn_api_error, "classify_api_error",
                        lambda *a, **k: classified)
    # Same binding rule as above: patch what the phase reads.
    monkeypatch.setattr(turn_api_error, "recover_before_classification",
                        lambda *a, **k: (False, "sys"))
    monkeypatch.setattr(turn_api_error, "recover_after_classification",
                        lambda *a, **k: (True, None))

    turn_api_error.handle_api_error(
        agent, api_error=RuntimeError("limited"), _retry=SimpleNamespace(),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        system_message="sys", active_system_prompt="sys", conversation_history=[],
        approx_tokens=1, retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=1, api_call_count=1, api_request_id="r1",
        api_start_time=0.0, effective_task_id="t", turn_id="u",
    )
    assert sup._wd.consecutive_failures("m") == 0


def test_failure_hook_ignores_foreign_endpoints(monkeypatch, tmp_path):
    """Cloud provider, or llamacpp pointed at someone else's server: silent."""
    from agent.conversation_loop import _note_managed_inference_result

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cloud = SimpleNamespace(provider="openai", base_url="https://x",
                            model="m", log_prefix="")
    _note_managed_inference_result(cloud, False)  # must not raise, records nothing

    foreign = SimpleNamespace(provider="llamacpp",
                              base_url="http://127.0.0.1:9/v1",
                              model="m", log_prefix="")
    _note_managed_inference_result(foreign, False)
    _note_managed_inference_result(foreign, True)


def test_success_hook_resets_streak(managed_agent, monkeypatch):
    """check_api_response on a good response resets the model's streak."""
    from agent import turn_response_check

    agent, sup = managed_agent
    sup._wd.note_result("m", ok=False)
    sup._wd.note_result("m", ok=False)
    assert sup._wd.consecutive_failures("m") == 2

    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.thinking_callback = None
    agent.api_mode = "chat_completions"
    agent._turn_received_provider_response = False
    transport = SimpleNamespace(
        normalize_response=lambda r: SimpleNamespace(finish_reason="stop"))
    agent._get_transport = lambda: transport
    agent._should_treat_stop_as_truncated = lambda *a: False

    monkeypatch.setattr("agent.turn_recovery.validate_response_shape",
                        lambda *a, **k: (False, []))
    monkeypatch.setattr(turn_response_check, "record_response_usage",
                        lambda *a, **k: SimpleNamespace(
                            compression_attempts=0, rearmed=False))
    monkeypatch.setattr("agent.relay_llm.complete_logical_call",
                        lambda *a, **k: None)
    agent._touch_activity = lambda *a, **k: None

    verdict = turn_response_check.check_api_response(
        agent, response=SimpleNamespace(), _retry=SimpleNamespace(),
        thinking_spinner=None, messages=[], api_messages=[], api_kwargs={},
        active_system_prompt="sys", conversation_history=[],
        finish_reason="stop", retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=1, length_continue_retries=0,
        truncated_response_parts=[], truncated_tool_call_retries=0,
        current_turn_user_idx=0, api_call_count=1, api_request_id="r1",
        api_start_time=0.0, effective_task_id="t", turn_id="u",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )
    assert verdict.action == "break"
    assert sup._wd.consecutive_failures("m") == 0
