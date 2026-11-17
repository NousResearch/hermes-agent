"""A Bedrock model call makes ONE HTTP attempt; Hermes's retry loop owns retries and backoff.

botocore's default retry policy (5 attempts) re-sent a Converse read timeout, so a hung call took
~5x the configured request timeout to surface, and re-sent throttles under Hermes's own rate-limit
backoff. These tests drive the real boto3 client the adapter builds against a local fake endpoint
and count the requests that reach it.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("boto3")
botocore_exceptions = pytest.importorskip("botocore.exceptions")

from agent import bedrock_adapter
from agent.bedrock_adapter import _get_bedrock_runtime_client
from agent.error_classifier import FailoverReason, classify_api_error

REGION = "us-east-1"
MODEL = "us.amazon.nova-pro-v1:0"
MESSAGES = [{"role": "user", "content": [{"text": "hi"}]}]
CONVERSE_OK = {"output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}}, "stopReason": "end_turn",
               "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}


@pytest.fixture
def fake_bedrock(monkeypatch):
    """Local bedrock-runtime endpoint; ``mode`` is ``hang`` (never answers until teardown) or
    ``throttle`` (400 ThrottlingException) / ``unavailable`` (503) for the first ``fail_first``
    requests, or until ``fail_for`` seconds after the first request, then a Converse answer.
    Yields the state; ``hits`` lists request paths, ``times`` their monotonic arrival."""
    state = {"mode": "hang", "hits": [], "times": [], "fail_first": 10**6, "fail_for": None}
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["hits"].append(self.path)
            state["times"].append(time.monotonic())
            self.rfile.read(int(self.headers.get("content-length", 0)))
            if state["mode"] == "hang":
                release.wait(30)
                return
            status, error_type, payload = {
                "throttle": (400, "ThrottlingException", {"message": "Too many requests, please wait before trying again."}),
                "unavailable": (503, "ServiceUnavailableException", {"message": "Service unavailable, try later."}),
            }[state["mode"]]
            window = state["fail_for"]
            if (len(state["hits"]) > state["fail_first"] if window is None
                    else state["times"][-1] - state["times"][0] >= window):
                status, error_type, payload = 200, None, CONVERSE_OK
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            if error_type:
                self.send_header("x-amzn-ErrorType", error_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-FAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for var in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK", "AWS_MAX_ATTEMPTS",
                "AWS_RETRY_MODE", "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS"):
        monkeypatch.delenv(var, raising=False)
    bedrock_adapter.reset_client_cache()
    try:
        yield state
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        bedrock_adapter.reset_client_cache()


def _use_config(monkeypatch, cfg):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)


def test_hung_converse_fails_after_one_read_timeout_not_five(fake_bedrock, monkeypatch):
    _use_config(monkeypatch, {"providers": {"bedrock": {"request_timeout_seconds": 0.5}}})
    client = _get_bedrock_runtime_client(REGION, MODEL)

    with pytest.raises(botocore_exceptions.ReadTimeoutError) as excinfo:
        client.converse(modelId=MODEL, messages=MESSAGES)

    assert len(fake_bedrock["hits"]) == 1, "botocore re-sent the timed-out call"
    assert classify_api_error(excinfo.value, provider="bedrock", model=MODEL).retryable


def test_throttle_reaches_hermes_retry_loop_on_first_refusal_with_default_timeouts(fake_bedrock, monkeypatch):
    _use_config(monkeypatch, {})
    fake_bedrock["mode"] = "throttle"
    client = _get_bedrock_runtime_client(REGION, MODEL)

    with pytest.raises(botocore_exceptions.ClientError) as excinfo:
        client.converse(modelId=MODEL, messages=MESSAGES)

    assert len(fake_bedrock["hits"]) == 1, "botocore retried the throttle under Hermes's own backoff"
    assert client.meta.config.read_timeout == 60, "unset config keeps botocore's own timeouts"
    classified = classify_api_error(excinfo.value, provider="bedrock", model=MODEL)
    assert (classified.reason, classified.retryable) == (FailoverReason.rate_limit, True)


@pytest.mark.parametrize("mode", ["unavailable", "throttle"])
def test_auxiliary_call_recovers_from_one_transient_bedrock_refusal(fake_bedrock, monkeypatch, mode):
    """With botocore no longer retrying, the aux recovery ladder must read the botocore error
    (status in ResponseMetadata, ThrottlingException code) or a single 503/throttle fails the task."""
    from agent import auxiliary_client
    from agent.auxiliary_client import BedrockAuxiliaryClient, call_llm

    _use_config(monkeypatch, {})
    monkeypatch.setattr(auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
    fake_bedrock.update(mode=mode, fail_first=1)
    client = BedrockAuxiliaryClient(REGION, MODEL)
    monkeypatch.setattr(auxiliary_client, "_get_auxiliary_task_config",
                        lambda task: {"provider": "bedrock", "model": MODEL, "max_output_tokens": 0})
    monkeypatch.setattr(auxiliary_client, "_get_cached_client", lambda *a, **k: (client, MODEL))

    response = call_llm(task="session_search", messages=[{"role": "user", "content": "ping"}])

    assert response.choices[0].message.content == "pong"
    assert len(fake_bedrock["hits"]) == 2


@pytest.mark.real_retry_backoff
@pytest.mark.parametrize("caller", ["sync", "async"])
def test_auxiliary_call_waits_out_a_bedrock_throttle_window(fake_bedrock, monkeypatch, caller):
    """A throttle lasting ~300ms is backed off, not re-sent at once: botocore no longer waits
    between attempts, so an immediate re-send lands in the same window and the task fails."""
    from agent import auxiliary_client
    from agent.auxiliary_client import (
        AsyncBedrockAuxiliaryClient, BedrockAuxiliaryClient, async_call_llm, call_llm,
    )

    _use_config(monkeypatch, {})
    fake_bedrock.update(mode="throttle", fail_for=0.3)
    client = BedrockAuxiliaryClient(REGION, MODEL)
    monkeypatch.setattr(auxiliary_client, "_get_auxiliary_task_config",
                        lambda task: {"provider": "bedrock", "model": MODEL, "max_output_tokens": 0})
    monkeypatch.setattr(
        auxiliary_client, "_get_cached_client",
        lambda *a, async_mode=False, **k: (AsyncBedrockAuxiliaryClient(client) if async_mode else client, MODEL))
    messages = [{"role": "user", "content": "ping"}]

    if caller == "sync":
        response = call_llm(task="session_search", messages=messages)
    else:
        response = asyncio.run(async_call_llm(task="session_search", messages=messages))

    assert response.choices[0].message.content == "pong"
    times = fake_bedrock["times"]
    assert times[1] - times[0] >= 0.25, "the throttled call was re-sent without a backoff"
