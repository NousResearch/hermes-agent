"""A refused connection to a local endpoint is classified as down, not just slow."""
import httpx
import pytest

from agent.error_classifier import FailoverReason, classify_api_error

LOCAL = "http://127.0.0.1:11334/v1"


def _wrapped(inner):
    outer = httpx.ConnectError("All connection attempts failed")
    outer.__cause__ = inner
    return outer


def _classify(error, base_url=LOCAL):
    return classify_api_error(error, provider="custom", model="m", base_url=base_url)


@pytest.mark.parametrize("error, refused", [
    (_wrapped(ConnectionRefusedError(61, "Connection refused")), True),
    (ConnectionRefusedError(61, "Connection refused"), True),
    (httpx.ConnectError("[Errno 61] Connection refused"), True),
    (httpx.ReadTimeout("timed out"), False),
    (httpx.ConnectTimeout("timed out"), False),
])
def test_a_local_refused_connect_is_a_timeout_verdict_marked_refused(error, refused):
    classified = _classify(error)
    assert classified.reason is FailoverReason.timeout
    assert classified.local_connection_refused is refused


@pytest.mark.parametrize("base_url, refused", [
    ("http://localhost:8080/v1", True), ("http://192.168.1.20:1234/v1", True), ("http://studio.local:11434", True),
    ("https://api.openai.com/v1", False), ("https://openrouter.ai/api/v1", False), ("", False),
])
def test_only_a_local_endpoint_refusal_is_marked(base_url, refused):
    error = _wrapped(ConnectionRefusedError(61, "Connection refused"))
    assert _classify(error, base_url).local_connection_refused is refused


def test_a_status_error_is_never_marked_refused():
    err = RuntimeError("Error code: 503 - overloaded")
    err.status_code = 503
    assert _classify(err).local_connection_refused is False


def test_a_cause_cycle_terminates():
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__context__ = b, a
    assert _classify(a).local_connection_refused is False
