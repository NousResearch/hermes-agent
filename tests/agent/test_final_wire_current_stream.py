"""R1: real SDK -> ordinary HTTPTransport -> inert httpcore stream consumer.

No private content-cache repair. The final public stream is the authority.
"""
import json

import httpcore
import httpx
import pytest
from openai import OpenAI, AsyncOpenAI

import agent.final_wire_admission as admission
from tests.agent.test_final_wire_review_regressions import identity, response_body


class Pool:
    def __init__(self, statuses=(200,)):
        self.statuses = list(statuses)
        self.sent = []

    def reply(self, content):
        self.sent.append(json.loads(content))
        status = self.statuses.pop(0)
        return httpcore.Response(status, headers=[(b"content-type", b"application/json")],
                                 content=json.dumps(response_body()).encode())

    def handle_request(self, request):
        return self.reply(b"".join(request.stream))

    async def handle_async_request(self, request):
        return self.reply(b"".join([part async for part in request.stream]))

    def close(self):
        pass

    async def aclose(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


def replace_stream(request, final):
    encoded = json.dumps(final).encode()
    request.stream = httpx.ByteStream(encoded)
    request.headers["Content-Length"] = str(len(encoded))


def body(text):
    return {"model": identity().model, "messages": [{"role": "user", "content": text}], "max_tokens": 1}


def record_measurement(monkeypatch):
    measured = []
    original = admission.project_final_body
    def project(value, attempt):
        measured.append((json.loads(json.dumps(value)), attempt))
        return original(value, attempt)
    monkeypatch.setattr(admission, "project_final_body", project)
    return measured


@pytest.mark.parametrize("stage", ["auth", "hook"])
@pytest.mark.parametrize("change", ["growth", "shrink", "replacement"])
def test_current_sync_stream_measured_equals_sent(monkeypatch, stage, change):
    final = body("x" * 4400 if change == "growth" else "replacement")
    source = body("x" * 4400 if change == "shrink" else "hi")
    measured = record_measurement(monkeypatch)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    def hook(request):
        replace_stream(request, final)
    class ReplaceAuth(httpx.Auth):
        def auth_flow(self, request):
            hook(request)
            yield request
    with httpx.Client(transport=transport, auth=ReplaceAuth() if stage == "auth" else None,
                      event_hooks={"request": [hook]} if stage == "hook" else None) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()):
            if change == "growth":
                with pytest.raises(admission.ProviderBoundRequestOverLimit) as caught:
                    sdk.chat.completions.create(**source)
                assert type(caught.value) is admission.ProviderBoundRequestOverLimit
            else:
                sdk.chat.completions.create(**source)
        assert measured == [(final, identity())]
        assert pool.sent == ([] if change == "growth" else [final])
        assert sleeps == []
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["auth", "hook"])
@pytest.mark.parametrize("change", ["growth", "shrink", "replacement"])
async def test_current_async_stream_measured_equals_sent(monkeypatch, stage, change):
    final = body("x" * 4400 if change == "growth" else "replacement")
    source = body("x" * 4400 if change == "shrink" else "hi")
    measured = record_measurement(monkeypatch)
    sleeps = []
    async def sleep(*a):
        sleeps.append(a)
    monkeypatch.setattr("asyncio.sleep", sleep)
    pool = Pool()
    transport = httpx.AsyncHTTPTransport(trust_env=False)
    transport._pool = pool
    async def hook(request):
        replace_stream(request, final)
    class ReplaceAuth(httpx.Auth):
        async def async_auth_flow(self, request):
            replace_stream(request, final)
            yield request
    async with httpx.AsyncClient(transport=transport, auth=ReplaceAuth() if stage == "auth" else None,
                                event_hooks={"request": [hook]} if stage == "hook" else None) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(AsyncOpenAI)(api_key="inert", base_url=identity().endpoint,
                                                             http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()):
            if change == "growth":
                with pytest.raises(admission.ProviderBoundRequestOverLimit) as caught:
                    await sdk.chat.completions.create(**source)
                assert type(caught.value) is admission.ProviderBoundRequestOverLimit
            else:
                await sdk.chat.completions.create(**source)
        assert measured == [(final, identity())]
        assert pool.sent == ([] if change == "growth" else [final])
        assert sleeps == []
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("route", ["default", "mount", "proxy"])
@pytest.mark.parametrize("resend", ["none", "auth_challenge", "sdk_retry"])
def test_final_stream_selected_route_resend_refuses_only_current_send(monkeypatch, route, resend):
    pool = Pool((401 if resend == "auth_challenge" else 500,))
    transport = httpx.HTTPTransport(proxy="http://proxy.invalid" if route == "proxy" else None, trust_env=False)
    transport._pool = pool
    hooks, sleeps = [], []
    def hook(request):
        hooks.append(request)
        if resend == "none" or len(hooks) == 2:
            replace_stream(request, body("x" * 4400))
    class Challenge(httpx.Auth):
        def auth_flow(self, request):
            response = yield request
            if response.status_code == 401:
                yield request
    monkeypatch.setattr("time.sleep", lambda *a: sleeps.append(a))
    kwargs = {"mounts": {"all://inert.invalid": transport}} if route == "mount" else {"transport": transport}
    with httpx.Client(**kwargs, trust_env=False, auth=Challenge() if resend == "auth_challenge" else None,
                      event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(OpenAI)(api_key="inert", base_url=identity().endpoint,
                                                        http_client=http, max_retries=2)
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundRequestOverLimit):
            sdk.chat.completions.create(**body("hi"))
        assert pool.sent == ([] if resend == "none" else [body("hi")])
        assert len(hooks) == (1 if resend == "none" else 2)
        # Only the genuine remote 500 can trigger one SDK backoff.
        assert len(sleeps) == (1 if resend == "sdk_retry" else 0)
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None


@pytest.mark.parametrize("payload", [b"not-json", b"", b"null", b"[]"])
def test_malformed_or_missing_current_stream_typed_refusal(payload):
    pool = Pool()
    transport = httpx.HTTPTransport(trust_env=False)
    transport._pool = pool
    def hook(request):
        request.stream = httpx.ByteStream(payload)
        request.headers["Content-Length"] = str(len(payload))
    with httpx.Client(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundRequestOverLimit):
            http.post(identity().endpoint + "/chat/completions", json=body("hi"))
        assert pool.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unprovable_stream_refuses_without_consumption(asynchronous):
    class Unknown(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            raise AssertionError("unprovable stream must not be consumed")
        async def __aiter__(self):
            raise AssertionError("unprovable stream must not be consumed")
            yield b""
    request = httpx.Request("POST", identity().endpoint + "/chat/completions", json=body("hi"))
    request.stream = Unknown()
    calls = []
    delegate = httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200))
    with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundUnsupportedAccounting):
        if asynchronous:
            await admission.GuardedAsyncHTTPXTransport(delegate).handle_async_request(request)
        else:
            admission.GuardedHTTPXTransport(delegate).handle_request(request)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("defect", ["stream", "model", "cap"])
async def test_async_sdk_local_refusal_restored_with_zero_or_nonzero_retry(monkeypatch, retries, defect):
    pool = Pool()
    transport = httpx.AsyncHTTPTransport(trust_env=False)
    transport._pool = pool
    sleeps = []
    async def sleep(*a):
        sleeps.append(a)
    monkeypatch.setattr("asyncio.sleep", sleep)
    async def hook(request):
        final = body("x" * 4400 if defect == "stream" else "hi")
        if defect == "model":
            final["model"] = "inert-smaller-model"
        elif defect == "cap":
            final["max_completion_tokens"] = True
        replace_stream(request, final)
    async with httpx.AsyncClient(transport=transport, event_hooks={"request": [hook]}) as http:
        admission.wrap_httpx_client_transports(http)
        sdk = admission.intercepted_openai_class(AsyncOpenAI)(api_key="inert", base_url=identity().endpoint,
                                                             http_client=http, max_retries=retries)
        with admission.bind_attempt_identity(identity()), pytest.raises(admission.ProviderBoundRequestOverLimit) as caught:
            await sdk.chat.completions.create(**body("hi"))
        expected = admission.ProviderBoundRequestOverLimit if defect == "stream" else admission.ProviderBoundInvalidAccounting
        assert type(caught.value) is expected
    assert pool.sent == []
    assert sleeps == []
    assert admission.current_attempt_identity() is None
    assert admission.current_local_refusal() is None
