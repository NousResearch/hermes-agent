"""Assistant phase must survive ordinary replay and both preflight passes."""
from copy import deepcopy

import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_api_kwargs,
    _preflight_codex_input_items,
)


@pytest.mark.parametrize("phase", ["commentary", "final_answer", "analysis", " Commentary "])
@pytest.mark.parametrize("content", ["Checking the next file.", [{"type": "output_text", "text": "Checking the next file."}]])
def test_untyped_preflight_keeps_explicit_assistant_phase(phase, content):
    raw = [{"role": "assistant", "phase": phase, "content": content, "id": "must-not-be-replayed", "extra": "omit"}]
    original = deepcopy(raw)
    wire = _preflight_codex_input_items(raw)
    assert wire[0]["phase"] == phase.strip()
    assert set(wire[0]) == {"role", "content", "phase"}
    assert raw == original
    assert _preflight_codex_input_items(wire) == wire


@pytest.mark.parametrize("issuer", [None, "codex_backend", "github_responses", "xai_responses"])
@pytest.mark.parametrize("structured", [False, True])
def test_history_fallback_and_final_request_preflight_retain_phase(issuer, structured):
    text = "Продолжаю проверку."
    content = [{"type": "text", "text": text}] if structured else text
    history = [{"role": "user", "content": "audit"}, {"role": "assistant", "content": content, "phase": "commentary"}]
    original = deepcopy(history)
    converted = _chat_messages_to_responses_input(history, current_issuer_kind=issuer, is_github_responses=issuer == "github_responses")
    assert converted[-1]["phase"] == "commentary"
    request = {"model": "test-model", "instructions": "test", "input": converted, "store": False}
    first = _preflight_codex_api_kwargs(request, is_github_responses=issuer == "github_responses")
    second = _preflight_codex_api_kwargs(first, is_github_responses=issuer == "github_responses")
    assert second["input"][-1]["phase"] == "commentary"
    assert second["input"][-1]["content"] == converted[-1]["content"]
    assert first == second
    assert history == original


@pytest.mark.parametrize("phase", [None, "", "   ", 42, False, {}, []])
def test_absent_or_invalid_phase_is_not_invented(phase):
    raw = [{"role": "assistant", "content": "Answer", "phase": phase}]
    assert "phase" not in _preflight_codex_input_items(raw)[0]
    assert "phase" not in _chat_messages_to_responses_input(raw)[0]


def test_user_phase_is_not_forwarded_and_typed_sidecar_remains_authoritative():
    history = [
        {"role": "user", "content": "audit", "phase": "commentary"},
        {"role": "assistant", "content": "stale fallback", "phase": "final_answer", "codex_message_items": [
            {"type": "message", "role": "assistant", "status": "completed", "id": "msg_original", "phase": "commentary", "content": [{"type": "output_text", "text": "Checking."}]},
        ]},
    ]
    original = deepcopy(history)
    wire = _preflight_codex_input_items(_chat_messages_to_responses_input(history))
    assert "phase" not in wire[0]
    assert wire[1]["phase"] == "commentary"
    assert wire[1]["id"] == "msg_original"
    assert wire[1]["content"][0]["text"] == "Checking."
    assert history == original


def test_phase_and_text_sanitization_are_independent():
    raw = [{"role": "assistant", "phase": "commentary", "content": "Reading <|start|> safely."}]
    wire = _preflight_codex_input_items(raw, sanitize_harmony_tokens=True)
    assert wire[0]["phase"] == "commentary"
    assert "<|start|>" not in wire[0]["content"]
    assert raw[0]["content"] == "Reading <|start|> safely."
