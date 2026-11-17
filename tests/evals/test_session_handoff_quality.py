"""Offline controls for the opt-in probe; scripted responses prove no model quality."""
import json
import sys

import httpx
import openai
import pytest
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from pydantic import TypeAdapter

from evals import session_handoff_quality as probe
from tests.fakes.providers.openai_responses import FunctionCall, Message, Turn, turn_events


def scripted(checkpoint=None, readback=None):
    responses = []
    if checkpoint is not None:
        responses.append({"role": "assistant", "tool_calls": [{"id": "choice", "function": {
            "name": "fresh_session", "arguments": json.dumps({"handoff": checkpoint})}}]})
    responses.append({"role": "assistant", "content": "Phase complete."})
    if checkpoint is not None:
        responses.append({"role": "assistant", "content": readback or checkpoint})
    requests = []

    def complete(messages, tools):
        requests.append((list(messages), tools))
        return responses.pop(0)

    return complete, requests


@pytest.mark.parametrize("case", probe.CASES, ids=lambda case: case.name)
def test_expected_decision_and_fact_retention_controls(case):
    checkpoint = "; ".join(case.facts) if case.should_handoff else None
    complete, requests = scripted(checkpoint)
    result = probe.run_case(complete, case)
    assert result["decision_pass"]
    assert result["missing_checkpoint_facts"] == result["missing_readback_facts"] == []
    if case.should_handoff:
        successor_messages, successor_tools = requests[-1]
        assert successor_tools is None
        assert [m["role"] for m in successor_messages] == ["system", "assistant", "user"]
        assert successor_messages[1]["content"] == checkpoint


def test_lossy_checkpoint_and_unhelpful_successor_are_reported_separately():
    case = probe.CASES[0]
    complete, _ = scripted("Inspect billing/totals.py next.", "I will continue.")
    result = probe.run_case(complete, case)
    assert result["decision_pass"]
    assert "test_refund_tax" in result["missing_checkpoint_facts"]
    assert "billing/totals.py" not in result["missing_checkpoint_facts"]
    assert "billing/totals.py" in result["missing_readback_facts"]


def test_unnecessary_handoff_is_not_a_passing_decision():
    complete, _ = scripted("Continue the tiny variable rename.")
    result = probe.run_case(complete, probe.CASES[1])
    assert result["eligible"]
    assert not result["decision_pass"]


@pytest.mark.parametrize("api_mode,status", [
    ("chat_completions", "completed"), ("codex_responses", "completed"),
    ("codex_responses", "incomplete"), ("codex_responses", "cancelled"),
])
def test_explicit_cli_probe_uses_only_synthetic_transport(tmp_path, monkeypatch, api_mode, status):
    case = probe.CASES[0]
    checkpoint = "; ".join(case.facts)
    monkeypatch.setattr(probe, "CASES", (case,))
    monkeypatch.setenv("SYNTHETIC_EVAL_KEY", "synthetic-not-a-credential")
    output = tmp_path / "evaluation.json"
    turns = iter([
        Turn([FunctionCall("fresh_session", {"handoff": checkpoint})]),
        Turn([Message("Phase complete.")]), Turn([Message(checkpoint)]),
    ])
    chat, _ = scripted(checkpoint)
    captured = []

    def respond(request):
        body = json.loads(request.content)
        captured.append(body)
        assert body["stream"] is False
        if api_mode == "codex_responses":
            TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(body)
            payload = turn_events(next(turns))[-1]["response"]
            payload["status"] = status
        else:
            payload = {"id": "synthetic", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "message": chat([], None), "finish_reason": "stop"}]}
        return httpx.Response(200, json=payload)

    sdk = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: sdk(**kw,
        http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    monkeypatch.setattr(sys, "argv", ["probe", "--base-url", "https://synthetic.invalid/v1", "--model", "gpt-5.4",
        "--api-key-env", "SYNTHETIC_EVAL_KEY", "--api-mode", api_mode,
        "--output", str(output), "--repetitions", "1"])
    if status != "completed":
        with pytest.raises(RuntimeError, match="Incomplete provider response"):
            probe.main()
        assert not output.exists()
        return
    assert probe.main() == 0
    report = json.loads(output.read_text())
    assert report["synthetic_inputs"]
    assert report["results"][0]["decision_pass"]
    assert len(captured) == 3

