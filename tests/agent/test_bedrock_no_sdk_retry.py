"""A Bedrock model call makes ONE HTTP attempt; Hermes's retry loop owns retries and backoff.

botocore's default retry policy (5 attempts) re-sent a Converse read timeout, so a hung call took
~5x the configured request timeout to surface, and re-sent throttles under Hermes's own rate-limit
backoff. These tests drive the real boto3 client the adapter builds against a local fake endpoint
and count the requests that reach it.
"""

import json
import threading
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


@pytest.fixture
def fake_bedrock(monkeypatch):
    """Local bedrock-runtime endpoint; ``mode`` is ``hang`` (never answers until teardown) or
    ``throttle`` (400 ThrottlingException). Yields the list of request paths it received."""
    state = {"mode": "hang", "hits": []}
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["hits"].append(self.path)
            self.rfile.read(int(self.headers.get("content-length", 0)))
            if state["mode"] == "hang":
                release.wait(30)
                return
            body = json.dumps({"message": "Too many requests, please wait before trying again."}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("x-amzn-ErrorType", "ThrottlingException")
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
