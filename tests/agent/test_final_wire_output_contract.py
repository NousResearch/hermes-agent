"""Final output fields and window invalid-accounting dispositions."""
import pytest
from agent.final_wire_admission import (
    COVERED_MAIN, FinalAttemptIdentity, ProviderBoundInvalidAccounting,
    bind_attempt_identity, admit_final_json, project_final_body, KNOWN_R,
)
from agent.conversation_compression import ProviderBoundRequestOverLimit

CASES = [
    ("chat_completions", {"messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
    ("chat_completions", {"messages": [{"role": "user", "content": "hi"}]}, "max_completion_tokens"),
    ("responses", {"input": "hi"}, "max_output_tokens"),
    ("anthropic_messages", {"messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
    ("anthropic_bedrock", {"messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
    ("bedrock_converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}, "inferenceConfig.maxTokens"),
    ("gemini_native", {"contents": [{"parts": [{"text": "hi"}]}]}, "generationConfig.maxOutputTokens"),
]


def put(body, field, value):
    body = dict(body)
    if "." in field:
        container, key = field.split(".")
        body[container] = {key: value}
    else:
        body[field] = value
    return body


def test_real_dispatch_does_not_normalize_invalid_window_before_admission():
    from types import SimpleNamespace
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    for window in (True, 3.5, "100000", float("nan"), None):
        calls = []
        agent = SimpleNamespace(model="inert", provider="openai", api_mode="chat_completions", base_url="https://inert.invalid", session_id="window", context_compressor=SimpleNamespace(context_length=window))
        with pytest.raises(ProviderBoundInvalidAccounting):
            _dispatch_nonstreaming_api_request(agent, {"model": "inert", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, make_client=lambda *a, **kw: calls.append(True))
        assert calls == []


def ident(family, window=1000):
    return FinalAttemptIdentity(COVERED_MAIN, family, "inert", "https://inert.invalid", window, "output")


@pytest.mark.parametrize("family,body,field", CASES)
@pytest.mark.parametrize("bad", [True, 1.5, "2", None, 0, -1, float("inf")])
def test_final_cap_invalid_values_are_typed(family, body, field, bad):
    with bind_attempt_identity(ident(family)), pytest.raises(ProviderBoundInvalidAccounting):
        admit_final_json(put(body, field, bad))


@pytest.mark.parametrize("bad", [None, "1000", True, 1000.5, float("inf"), 0, -1])
def test_invalid_window_never_coerces_or_leaks_raw_exception(bad):
    with bind_attempt_identity(ident("chat_completions", bad)), pytest.raises(ProviderBoundInvalidAccounting):
        admit_final_json({"messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize("family,body,field", CASES)
def test_known_final_cap_shared_window_and_equality(family, body, field):
    body = put(body, field, 200)
    snap = project_final_body(body, ident(family))
    assert snap.reservation_state == KNOWN_R
    assert snap.resolved_r == 200
    with bind_attempt_identity(ident(family, snap.estimated_input + 200)):
        admit_final_json(body)
    with bind_attempt_identity(ident(family, snap.estimated_input + 199)), pytest.raises(ProviderBoundRequestOverLimit):
        admit_final_json(body)
