"""Contracts for the provider-neutral, shadow-only decision layer."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from agent.secret_scope import reset_secret_scope, set_secret_scope
from agent.decision_provider import (
    CONSUMERS,
    ConstraintLedger,
    DecisionRuntime,
    SubprocessDecisionProvider,
    build_context_sieve_payload,
    calibrate_thresholds,
    privacy_gate,
    record_completion,
    record_notification,
    record_pre_action,
    record_specialist,
    record_tool_result,
)


class _Provider:
    name = "test-provider"

    def __init__(self, confidence=0.9, delay=0):
        self.confidence = confidence
        self.delay = delay
        self.calls = []

    def decide(self, request, *, timeout_seconds):
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        return {"decision": "allow", "confidence": self.confidence}


def _runtime(tmp_path: Path, provider: _Provider, **extra):
    config = {
        "logging": {"enabled": True, "path": str(tmp_path / "replay.jsonl")},
        "context_sieve": {"enabled": True, "min_chars": 20},
        **extra,
    }
    return DecisionRuntime(config, provider=provider)


def test_constraint_ledger_is_deterministic_and_non_verbatim():
    ledger = ConstraintLedger.from_messages([
        {"role": "user", "content": "Do not publish. Ask before destructive changes."},
    ])

    metadata = ledger.to_metadata()
    serialized = json.dumps(metadata)
    assert metadata["count"] == 2
    assert "Do not publish" not in serialized
    assert "destructive" in metadata["safety_boundaries"]
    assert all(len(item["text_digest"]) == 64 for item in metadata["constraints"])


def test_privacy_gate_fails_closed_for_secret_content_and_secret_paths():
    assert not privacy_gate({"command": "cat .env"}).allowed
    assert not privacy_gate({"text": "OPENAI_API_KEY=super-secret-value"}).allowed
    assert not privacy_gate({"text": "```python\nprint('private')\n```"}).allowed
    assert privacy_gate({"text": "ordinary test output", "path": "src/app.py"}).allowed


def test_thresholds_are_per_consumer_and_shadow_never_enforces(tmp_path):
    provider = _Provider(confidence=0.8)
    runtime = _runtime(
        tmp_path, provider,
        thresholds={"pre_action": 0.95, "completion": 0.75},
    )

    pre_action = runtime.evaluate("pre_action", {"tool": "write_file"})
    completion = runtime.evaluate("completion", {"claims_done": True})

    assert not pre_action.threshold_met
    assert completion.threshold_met
    assert pre_action.shadow and completion.shadow
    assert pre_action.candidate_action == "unknown"
    assert completion.candidate_action == "allow"
    assert len(provider.calls) == 2


def test_secret_tool_output_is_not_sent_to_provider_or_replay_log(tmp_path):
    provider = _Provider()
    runtime = _runtime(tmp_path, provider)
    observation = runtime.evaluate(
        "context_sieve",
        {"sample": "token=should-not-leave"},
        privacy_input="token=should-not-leave",
    )

    assert not observation.privacy_allowed
    assert not provider.calls
    log_text = (tmp_path / "replay.jsonl").read_text()
    assert "should-not-leave" not in log_text


def test_private_code_is_metadata_only_and_not_sent_to_provider(tmp_path):
    provider = _Provider()
    runtime = _runtime(tmp_path, provider)
    observation = runtime.evaluate(
        "pre_action",
        {"tool": "patch", "args": {"patch": "private source code"}},
        privacy_input="private source code",
        private_content=True,
    )

    assert not observation.privacy_allowed
    assert observation.skipped_reason == "private_content"
    assert not provider.calls

    agent = SimpleNamespace(_decision_runtime=runtime, _session_messages=[])
    projected = record_pre_action(agent, "patch", {"patch": "private source code"}, "default")
    assert projected.privacy_allowed
    assert provider.calls
    assert "private source code" not in json.dumps(provider.calls[-1].payload)


def test_timeout_uses_consumer_policy_without_changing_execution(tmp_path):
    provider = _Provider(delay=0.2)
    runtime = _runtime(
        tmp_path, provider,
        policies={"loop": {"timeout_seconds": 0.05, "fail_mode": "open"}},
    )

    observation = runtime.evaluate("loop", {"repeat_count": 3})

    assert observation.result.timed_out
    assert observation.policy.fail_mode == "open"
    assert observation.shadow


def test_context_sieve_preserves_error_paths_approval_and_non_repro_evidence():
    provider = _Provider()
    runtime = DecisionRuntime(
        {
            "logging": {"enabled": False},
            "context_sieve": {"enabled": True, "min_chars": 20},
        },
        provider=provider,
    )
    agent = SimpleNamespace(
        _decision_runtime=runtime,
        _turn_file_mutation_paths={"src/app.py"},
        _turn_failed_file_mutations={},
    )
    output = (
        "prefix\n" * 8
        + "ERROR: test failed for src/app.py; approval pending\n"
        + "The failure is flaky and non-reproducible.\n"
        + "suffix\n" * 8
    )

    payload = build_context_sieve_payload(agent, "terminal", {}, output, is_error=True)

    assert payload is not None
    assert payload["preserve_exact_error"]
    assert payload["active_file_paths"]
    assert payload["unresolved_failure"]
    assert payload["approval_evidence"]
    assert payload["non_reproducible_evidence"]
    assert payload["candidate_line_ranges"]


def test_calibration_returns_independent_thresholds_per_consumer():
    records = [
        {"consumer": "pre_action", "confidence": 0.9, "outcome": False},
        {"consumer": "pre_action", "confidence": 0.6, "outcome": True},
        {"consumer": "completion", "confidence": 0.8, "outcome": True},
        {"consumer": "completion", "confidence": 0.6, "outcome": False},
    ]

    calibrated = calibrate_thresholds(records)

    assert set(calibrated) == {"pre_action", "completion"}
    assert calibrated["pre_action"]["samples"] == 2
    assert calibrated["completion"]["samples"] == 2


def test_one_provider_gateway_routes_every_consumer(tmp_path):
    provider = _Provider()
    runtime = _runtime(tmp_path, provider)

    for consumer in CONSUMERS:
        runtime.evaluate(consumer, {"consumer": consumer})

    assert len(provider.calls) == len(CONSUMERS)
    assert [request.consumer for request in provider.calls] == list(CONSUMERS)


def test_subprocess_gateway_invocation_and_malformed_response(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import json,sys; json.load(sys.stdin); print(json.dumps({'decision':'review','confidence':0.91}))",
    ]
    runtime = _runtime(
        tmp_path,
        SubprocessDecisionProvider(command, name="test-gateway"),
    )
    observation = runtime.evaluate("pre_action", {"tool": "terminal"})

    assert observation.result.decision == "review"
    assert observation.result.provider == "test-gateway"

    bad_runtime = _runtime(
        tmp_path,
        SubprocessDecisionProvider([sys.executable, "-c", "print('not-json')"]),
    )
    malformed = bad_runtime.evaluate("loop", {"repeat_count": 2})
    assert malformed.result.error == "invalid_json"
    assert malformed.skipped_reason == "invalid_json"

    slow_runtime = _runtime(
        tmp_path,
        SubprocessDecisionProvider([sys.executable, "-c", "import time; time.sleep(1)"]),
        policies={"loop": {"timeout_seconds": 0.05, "fail_mode": "open"}},
    )
    timed_out = slow_runtime.evaluate("loop", {"repeat_count": 2})
    assert timed_out.result.timed_out
    assert timed_out.skipped_reason == "timeout"


def test_subprocess_gateway_uses_the_active_secret_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-test-value")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-test-value")
    scope_token = set_secret_scope({"OPENROUTER_API_KEY": "scoped-test-value"})
    try:
        command = [
            sys.executable,
            "-c",
            (
                "import json,os,sys; json.load(sys.stdin); "
                "ok=(os.getenv('OPENROUTER_API_KEY')=='scoped-test-value' "
                "and 'OPENAI_API_KEY' not in os.environ); "
                "print(json.dumps({'decision':'review' if ok else 'unknown','confidence':0.91}))"
            ),
        ]
        runtime = _runtime(tmp_path, _Provider())
        runtime.provider = SubprocessDecisionProvider(command, name="test-gateway")

        observation = runtime.evaluate("pre_action", {"tool": "terminal"})

        assert observation.result.decision == "review"
    finally:
        reset_secret_scope(scope_token)


def test_null_provider_is_a_safe_fallback(tmp_path):
    runtime = DecisionRuntime(
        {
            "provider": "none",
            "adapter_path": str(tmp_path / "missing-adapter.cjs"),
            "logging": {"enabled": False},
        }
    )

    observation = runtime.evaluate("completion", {"claims_done": True})

    assert observation.result.provider == "none"
    assert observation.skipped_reason == "no_provider"


def test_error_round_trips_failure_retry_loop_and_context_observations(tmp_path):
    provider = _Provider()
    runtime = _runtime(tmp_path, provider)
    agent = SimpleNamespace(
        _decision_runtime=runtime,
        _session_messages=[],
        _tool_guardrails=None,
        _turn_file_mutation_paths=set(),
        _turn_failed_file_mutations={},
        _incident_failed_repair_attempts=1,
    )

    record_tool_result(agent, "terminal", {}, "ERROR: test failed; approval pending", is_error=True)

    consumers = [request.consumer for request in provider.calls]
    assert consumers[:3] == ["loop", "failure_classification", "retry"]
    assert "context_sieve" in consumers


def test_context_sieve_and_completion_do_not_send_raw_private_text(tmp_path):
    provider = _Provider()
    runtime = _runtime(tmp_path, provider)
    agent = SimpleNamespace(
        _decision_runtime=runtime,
        _session_messages=[],
        _turn_file_mutation_paths={"src/app.py"},
        _turn_failed_file_mutations={},
        _incident_state=None,
        _verification_stop_nudges=0,
        session_id="test",
        runtime_cwd=str(tmp_path),
    )
    private_output = "private source code\n" * 20

    record_tool_result(agent, "read_file", {}, private_output, is_error=False)
    record_completion(agent, "Done with private source code")
    record_specialist(agent, "Review the app safely")
    record_notification(agent, "Done")

    serialized_payloads = [json.dumps(request.payload) for request in provider.calls]
    assert all("private source code" not in payload for payload in serialized_payloads)
