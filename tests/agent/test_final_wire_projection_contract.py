"""Physical family projections with valid identity and anti-vacuity controls."""
from copy import deepcopy
import json

import httpx
import pytest

import agent.final_wire_admission as admission
from agent.conversation_compression import ProviderBoundRequestOverLimit


# Native model identifiers are carried by serialized operation URLs, not JSON.
ROUTES = {
    "chat_completions": "/v1/chat/completions",
    "codex_responses": "/v1/responses",
    "anthropic_messages": "/v1/messages",
    "gemini_native": "/v1beta/models/inert:generateContent",
    "bedrock_converse": "/model/inert/converse",
}
PRESSURE_CASES = [
    pytest.param("chat_completions", {"messages": [{"role": "assistant", "content": "", "reasoning_details": [{"text": "x" * 4400}]}]}, ("messages", 0, "reasoning_details", 0, "text"), "messages", id="reasoning_details"),
    pytest.param("codex_responses", {"input": [{"type": "function_call", "name": "inert", "arguments": "x" * 4400}]}, ("input", 0, "arguments"), "input", id="function_call_arguments"),
    pytest.param("gemini_native", {"contents": [{"role": "user", "parts": [{"text": "x" * 4400}]}], "generationConfig": {"maxOutputTokens": 1}}, ("contents", 0, "parts", 0, "text"), "contents", id="native_contents"),
    pytest.param("chat_completions", {"messages": [{"role": "user", "content": "hi"}], "stop": ["x" * 4400]}, ("stop", 0), "stop", id="stop"),
    pytest.param("bedrock_converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}], "promptVariables": {"x": {"text": "x" * 4400}}}, ("promptVariables", "x", "text"), "promptVariables", id="prompt_variables"),
]
BLOCK_CASES = [
    pytest.param("chat_completions", {"messages": [{"role": "user", "content": [{"type": "unknown_block", "payload": "hi"}]}]}, ("messages", 0, "content", 0), {"type": "text", "text": "hi"}, "unsupported context block", id="unknown_block"),
    pytest.param("anthropic_messages", {"messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "base64", "data": "abc"}}]}], "max_tokens": 1}, ("messages", 0, "content", 0), {"type": "text", "text": "hi"}, "unsupported context block", id="document_block"),
    pytest.param("gemini_native", {"contents": [{"parts": [{"inlineData": {"mimeType": "audio/wav", "data": "abc"}}]}], "generationConfig": {"maxOutputTokens": 1}}, ("contents", 0, "parts", 0), {"text": "hi"}, "unsupported inline media", id="audio_inline"),
]


def ident(family, window=1000):
    return admission.FinalAttemptIdentity(admission.COVERED_MAIN, family, "inert", "https://inert.invalid", window, "projection")


def final_body(family, body):
    body = deepcopy(body)
    if family not in {"gemini_native", "bedrock_converse"}:
        body["model"] = "inert"
    return body


def replace_at(body, path, value):
    body = deepcopy(body)
    parent = body
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return body


@pytest.fixture
def physical_trace(monkeypatch):
    """Observe successful reconciliation and real projection, never bypass them."""
    reconciled, projected = [], []
    original_reconcile = admission._reconcile_final_identity
    original_project = admission.project_final_body

    def reconcile(body, identity, url):
        original_reconcile(body, identity, url)
        reconciled.append((deepcopy(body), identity, str(url)))

    def project(body, identity):
        snapshot = original_project(body, identity)
        projected.append((deepcopy(body), identity, snapshot))
        return snapshot

    monkeypatch.setattr(admission, "_reconcile_final_identity", reconcile)
    monkeypatch.setattr(admission, "project_final_body", project)
    return reconciled, projected


def assert_reached(trace, bodies, identity, url):
    reconciled, projected = trace
    assert reconciled == [(body, identity, url) for body in bodies]
    assert [(body, bound) for body, bound, _ in projected] == [(body, identity) for body in bodies]
    return [snapshot for _, _, snapshot in projected]


@pytest.mark.parametrize("family,body,path,bucket", PRESSURE_CASES)
def test_transmitted_context_is_counted_at_physical_boundary(family, body, path, bucket, physical_trace):
    identity = ident(family)
    large = final_body(family, body)
    small = replace_at(large, path, "hi")
    url = identity.endpoint + ROUTES[family]
    sent = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(json.loads(request.content)) or httpx.Response(200))) as client:
        admission.wrap_httpx_client_transports(client)
        with admission.bind_attempt_identity(identity):
            assert client.post(url, json=small).status_code == 200
        with admission.bind_attempt_identity(identity), pytest.raises(ProviderBoundRequestOverLimit) as refusal:
            client.post(url, json=large)
    # Unsupported/invalid accounting are subclasses, not evidence of pressure.
    assert type(refusal.value) is ProviderBoundRequestOverLimit
    small_snapshot, large_snapshot = assert_reached(physical_trace, [small, large], identity, url)
    assert small_snapshot.coverage == large_snapshot.coverage == admission.COMPLETE_LOCAL
    assert small_snapshot.estimated_input < identity.window <= large_snapshot.estimated_input
    assert dict(large_snapshot.bucket_totals)[bucket] > dict(small_snapshot.bucket_totals)[bucket]
    assert refusal.value.pressure == large_snapshot.estimated_input
    assert refusal.value.limit == identity.window
    assert sent == [small]


@pytest.mark.parametrize("family,body,path,control,reason", BLOCK_CASES)
def test_unknown_or_unsupported_block_is_not_zero(family, body, path, control, reason, physical_trace):
    identity = ident(family)
    unsupported = final_body(family, body)
    small = replace_at(unsupported, path, control)
    url = identity.endpoint + ROUTES[family]
    sent = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: sent.append(json.loads(request.content)) or httpx.Response(200))) as client:
        admission.wrap_httpx_client_transports(client)
        with admission.bind_attempt_identity(identity):
            assert client.post(url, json=small).status_code == 200
        with admission.bind_attempt_identity(identity), pytest.raises(admission.ProviderBoundUnsupportedAccounting) as refusal:
            client.post(url, json=unsupported)
    assert type(refusal.value) is admission.ProviderBoundUnsupportedAccounting
    assert refusal.value.reason == reason
    small_snapshot, unsupported_snapshot = assert_reached(physical_trace, [small, unsupported], identity, url)
    assert small_snapshot.coverage == admission.COMPLETE_LOCAL
    assert 0 < small_snapshot.estimated_input < identity.window
    assert unsupported_snapshot.coverage == admission.UNSUPPORTED
    assert unsupported_snapshot.coverage_reason == reason
    assert sent == [small]


@pytest.mark.parametrize("family,body,path,bucket", PRESSURE_CASES)
def test_field_omission_breaks_pressure_regression(monkeypatch, family, body, path, bucket, physical_trace):
    """A bounded test-local lost-field fault must break the corresponding test."""
    original = admission._classified_pressure

    def omit_field(final, current_family):
        # Keep identity, routes and every other field; lose only this sentinel.
        return original(replace_at(final, path, ""), current_family)

    monkeypatch.setattr(admission, "_classified_pressure", omit_field)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_transmitted_context_is_counted_at_physical_boundary(family, body, path, bucket, physical_trace)


@pytest.mark.parametrize("family,body,path,control,reason", BLOCK_CASES)
def test_block_check_omission_breaks_unsupported_regression(monkeypatch, family, body, path, control, reason, physical_trace):
    """Simulate only the relevant unsupported check being lost, never a route fault."""
    target = body
    for key in path:
        target = target[key]
    original = admission._context_projection

    def ignore_block(value):
        if value == target:
            return {}, 0
        return original(value)

    monkeypatch.setattr(admission, "_context_projection", ignore_block)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_unknown_or_unsupported_block_is_not_zero(family, body, path, control, reason, physical_trace)
