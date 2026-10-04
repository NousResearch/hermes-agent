"""Independent inert contract probes; canonical runner only, no sockets/auth stores.

This is Reviewer evidence, NOT a candidate test/source correction. All assertions
encode the frozen contracts; failures preserve the observed defect. HTTPTransport
keeps its real request-to-httpcore conversion; only the pool is inert.
"""
import json
from types import SimpleNamespace

import httpcore
import httpx
import pytest
from openai import OpenAI

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.conversation_compression import ProviderBoundRequestOverLimit
import agent.final_wire_admission as admission


def identity(window=1000):
    return admission.FinalAttemptIdentity(
        admission.COVERED_MAIN, "chat_completions", "inert-original-model",
        "https://inert.invalid/v1", window, "reviewer-inert-attempt",
    )


def response_body():
    return {"id": "inert", "object": "chat.completion", "created": 0,
            "model": "inert-original-model", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "inert"}, "finish_reason": "stop"}]}


class InertPool:
    def __init__(self):
        self.bodies = []
    def handle_request(self, request):
        # Exactly the stream consumed by ordinary HTTPTransport/httpcore.
        self.bodies.append(json.loads(b"".join(request.stream)))
        return httpcore.Response(200, headers=[(b"content-type", b"application/json")],
                                 content=json.dumps(response_body()).encode())
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def test_hook_stream_replacement_cannot_send_unmeasured_body(monkeypatch):
    pool = InertPool()
    measured = []
    original_projection = admission.project_final_body
    def record_projection(body, attempt):
        snap = original_projection(body, attempt)
        measured.append(snap.estimated_input)
        return snap
    monkeypatch.setattr(admission, "project_final_body", record_projection)
    def hook(request):
        body = json.loads(request.content)
        body["messages"] = [{"role": "user", "content": "x" * 4400}]
        encoded = json.dumps(body).encode()
        request.stream = httpx.ByteStream(encoded)
        request.headers["Content-Length"] = str(len(encoded))
        # Deliberately do NOT repair private _content. Public stream is final.
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(
            api_key="inert-not-a-secret", base_url="https://inert.invalid/v1",
            http_client=http, max_retries=0,
        )
        caught = None
        with admission.bind_attempt_identity(identity()):
            try:
                sdk.chat.completions.create(model="inert-original-model",
                    messages=[{"role": "user", "content": "hi"}], max_tokens=1)
            except ProviderBoundRequestOverLimit as exc:
                caught = exc
        actual = [original_projection(body, identity()).estimated_input for body in pool.bodies]
        print("HOOK_STREAM measured_pressures=", measured,
              "physical_pressures=", actual, "delegate_count=", len(pool.bodies))
        assert caught is not None, "final over-limit stream was delegated while cached content was measured"
        assert pool.bodies == []


def test_sdk_model_override_cannot_reuse_original_identity_window():
    pool = InertPool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    seen = []
    def hook(request):
        seen.append((admission.current_attempt_identity(), json.loads(request.content)["model"]))
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(
            api_key="inert-not-a-secret", base_url="https://inert.invalid/v1",
            http_client=http, max_retries=0,
        )
        agent = SimpleNamespace(model="inert-original-model", provider="openai",
            api_mode="chat_completions", base_url="https://inert.invalid/v1",
            session_id="reviewer-inert", context_compressor=SimpleNamespace(context_length=10000))
        caught = None
        try:
            _dispatch_nonstreaming_api_request(agent,
                {"model": agent.model, "messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": 1, "extra_body": {"model": "inert-overridden-model"}},
                make_client=lambda *args, **kw: sdk)
        except ProviderBoundRequestOverLimit as exc:
            caught = exc
        print("MODEL_OVERRIDE bound_model=", seen[0][0].model,
              "final_model=", seen[0][1], "bound_window=", seen[0][0].window,
              "delegate_count=", len(pool.bodies))
        assert caught is not None, "unknown changed final model reused original W without rebinding/refusal"
        assert pool.bodies == []


@pytest.mark.parametrize("bad_alias", [True, 1.0])
def test_both_cap_aliases_are_individually_type_validated(bad_alias):
    body = {"model": "inert-original-model", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "max_completion_tokens": bad_alias}
    snap = admission.project_final_body(body, identity())
    print("ALIASES second_type=", type(bad_alias).__name__, "reservation_state=", snap.reservation_state)
    assert snap.reservation_state == admission.INVALID_RESERVATION
    with pytest.raises(admission.ProviderBoundInvalidAccounting):
        admission.admit_final_json(body, identity())


@pytest.mark.parametrize("bad_alias", [True, 1.0])
def test_invalid_equal_aliases_do_not_delegate_through_real_sdk(bad_alias):
    pool = InertPool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    with httpx.Client(transport=transport) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert-not-a-secret",
            base_url="https://inert.invalid/v1", http_client=http, max_retries=0)
        caught = None
        with admission.bind_attempt_identity(identity()):
            try:
                sdk.chat.completions.create(model="inert-original-model",
                    messages=[{"role": "user", "content": "hi"}], max_tokens=1,
                    extra_body={"max_completion_tokens": bad_alias})
            except admission.ProviderBoundInvalidAccounting as exc:
                caught = exc
        print("INVALID_ALIAS_PHYSICAL second_type=", type(bad_alias).__name__,
              "delegate_count=", len(pool.bodies))
        assert caught is not None
        assert pool.bodies == []
        sdk.close()


@pytest.mark.parametrize("oversized", [False, True])
def test_inert_httptransport_pool_control(oversized):
    pool = InertPool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    with httpx.Client(transport=transport) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert-not-a-secret",
            base_url="https://inert.invalid/v1", http_client=http, max_retries=0)
        caught = None
        with admission.bind_attempt_identity(identity()):
            try:
                sdk.chat.completions.create(model="inert-original-model",
                    messages=[{"role": "user", "content": "x" * 4400 if oversized else "hi"}],
                    max_tokens=1)
            except ProviderBoundRequestOverLimit as exc:
                caught = exc
        print("CONTROL oversized=", oversized, "delegate_count=", len(pool.bodies))
        assert (caught is not None) is oversized
        assert len(pool.bodies) == (0 if oversized else 1)
        sdk.close()
