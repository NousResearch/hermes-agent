"""Native streaming and IAM recovery at real SDK physical boundaries, inert only."""
import json
from unittest.mock import patch

import httpx
import pytest
from anthropic import AnthropicBedrock
from botocore.awsrequest import AWSResponse

from agent.chat_completion_helpers import interruptible_streaming_api_call
from agent.bedrock_adapter import build_converse_kwargs
from agent.conversation_compression import ProviderBoundRequestOverLimit
from agent.final_wire_admission import (
    current_attempt_identity, current_local_refusal, intercepted_anthropic_class,
    wrap_httpx_client_transports,
)
from tests.agent.test_final_wire_native_egress import converse_client, RawResponse
from tests.run_agent.test_413_compression import agent  # noqa: F401


@pytest.mark.parametrize("family", ["converse", "anthropic_bedrock"])
@pytest.mark.parametrize("iam_first", [False, True])
def test_real_native_stream_and_iam_recovery_refusal(agent, monkeypatch, family, iam_first):
    a = agent
    a.model, a.provider = "claude-sonnet-4-5", "bedrock"
    a.api_mode = "bedrock_converse" if family == "converse" else "anthropic_messages"
    a.base_url = "https://inert.invalid"
    a.context_compressor.context_length = 1000
    a.tools = []
    a._disable_streaming = False
    sends, events, sleeps, identities = [], [], [], []
    # threading's inert polling/lock acquisition uses sub-100ms sleeps;
    # SDK/botocore and outer backoff are the delays under test.
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds) if seconds >= 0.1 else None)
    if family == "converse":
        client, delegate = converse_client()
        def receive(request):
            sends.append(request.url)
            return AWSResponse(request.url, 403, {"content-type": "application/json", "x-amzn-errortype": "AccessDeniedException"}, RawResponse(json.dumps({"message": "inert denied bedrock:InvokeModelWithResponseStream"}).encode()))
        delegate.send = receive
        def mutate(request, **kwargs):
            identities.append(current_attempt_identity())
            events.append(request.url)
            if not iam_first or not request.url.endswith("converse-stream"):
                body = json.loads(request.body)
                body["system"] = [{"text": "x" * 4400}]
                request.body = json.dumps(body).encode()
        client.meta.events.register_last("before-send.bedrock-runtime", mutate)
        monkeypatch.setattr("agent.bedrock_adapter._get_bedrock_runtime_client", lambda _: client)
        kwargs = build_converse_kwargs(a.model, [{"role": "user", "content": "hi"}], max_tokens=1)
        kwargs.update(__bedrock_converse__=True, __bedrock_region__="us-east-1")
    else:
        def receive(request):
            sends.append(request.url.path)
            return httpx.Response(403, json={"message": "inert denied bedrock:InvokeModelWithResponseStream", "__type": "AccessDeniedException"}, headers={"x-amzn-errortype": "AccessDeniedException"})
        def mutate(request):
            identities.append(current_attempt_identity())
            events.append(request.url.path)
            if not iam_first or not request.url.path.endswith("invoke-with-response-stream"):
                body = json.loads(request.content)
                body["system"] = "x" * 4400
                request.stream = httpx.ByteStream(json.dumps(body).encode())
                if hasattr(request, "_content"):
                    del request._content
                request.read()
        http = httpx.Client(transport=httpx.MockTransport(receive), event_hooks={"request": [mutate]})
        wrap_httpx_client_transports(http)
        client = intercepted_anthropic_class(AnthropicBedrock)(aws_access_key="inert", aws_secret_key="inert", aws_region="us-east-1", base_url=a.base_url, http_client=http, max_retries=2)
        monkeypatch.setattr(a, "_create_request_anthropic_client", lambda **kw: client)
        monkeypatch.setattr(a, "_close_request_anthropic_client", lambda *args, **kw: None)
        kwargs = {"model": a.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    try:
        with patch.object(a, "_try_activate_fallback") as fallback:
            if family == "anthropic_bedrock" and iam_first:
                # This branch returns the genuine IAM error to the outer loop;
                # the subsequent nonstream dispatch must retain coverage.
                from anthropic import PermissionDeniedError
                from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
                with pytest.raises(PermissionDeniedError):
                    interruptible_streaming_api_call(a, kwargs)
                assert a._disable_streaming is True
                with pytest.raises(ProviderBoundRequestOverLimit) as caught:
                    _dispatch_nonstreaming_api_request(a, kwargs, make_client=lambda *args, **kw: client)
            else:
                with pytest.raises(ProviderBoundRequestOverLimit) as caught:
                    interruptible_streaming_api_call(a, kwargs)
            assert type(caught.value) is ProviderBoundRequestOverLimit
            assert len(sends) == (1 if iam_first else 0)
            assert len(events) == (2 if iam_first else 1)
            assert sleeps == []
            assert fallback.call_count == 0
            assert all(i is not None and i.window == 1000 and i.model == a.model for i in identities)
            assert all(i.family == ("bedrock_converse" if family == "converse" else "anthropic_bedrock") for i in identities)
            assert current_attempt_identity() is None
            assert current_local_refusal() is None
    finally:
        client.close()
