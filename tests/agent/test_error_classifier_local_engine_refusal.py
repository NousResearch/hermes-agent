"""A self-hosted engine's refusal is deterministic: retrying burns backoff AND the host's own
startup wait (a sleeping GPU box answers "no respondio tras el Wake-on-LAN" after a full
wake timeout), while the configured fallback chain could answer immediately. These refusals
must skip the retry loop and hand over at once — without changing how a third-party
provider's transient 503 is handled."""

from agent.error_classifier import FailoverReason, classify_api_error

_LOCAL_ROUTE = "http://127.0.0.1:12345/v1"


class _UnavailableError(Exception):
    """OpenAI-compatible 503 with a structured body, as local engines relay it."""

    def __init__(self, message, code):
        super().__init__(f"Error code: 503 - {message}")
        self.status_code = 503
        self.body = {"error": {"message": message, "type": code, "code": code}}
        self.code = code


def _local(message: str, code: str, base_url: str = _LOCAL_ROUTE):
    return classify_api_error(_UnavailableError(message, code), provider="custom",
                              model="local-engine", base_url=base_url)


def test_engine_busy_refusal_falls_back_without_retrying():
    r = _local("The engine is busy (a game holds the GPU). Assistants return when you are done.",
               "engine_in_use")
    assert r.reason == FailoverReason.overloaded
    assert (r.retryable, r.should_fallback) == (False, True)


def test_sleeping_host_that_missed_wake_falls_back_without_retrying():
    r = _local("Host did not answer after Wake-on-LAN. Try again in a moment.", "host_asleep")
    assert r.reason == FailoverReason.overloaded
    assert (r.retryable, r.should_fallback) == (False, True)


def test_engine_pausa_refusal_falls_back_without_retrying():
    r = _local("The engine is paused (offline batch translation running).", "engine_paused")
    assert r.reason == FailoverReason.overloaded
    assert (r.retryable, r.should_fallback) == (False, True)


def test_a_foreign_503_still_uses_the_retry_budget():
    """A capacity 503 from a real provider clears by itself: keep today's retry-then-fallback."""
    r = classify_api_error(_UnavailableError("model temporarily unavailable due to capacity limits",
                                             "server_error"),
                           provider="openai", model="gpt-x", base_url="https://api.openai.com/v1")
    assert r.reason == FailoverReason.overloaded
    assert (r.retryable, r.should_fallback) == (True, False)


def test_a_route_declared_local_falls_back_even_on_an_unrecognised_503(monkeypatch):
    """`agent.local_refusal_hosts` covers engines whose refusal wording we cannot enumerate."""
    import agent.error_classifier as ec

    monkeypatch.setattr(ec, "load_config_readonly",
                        lambda: {"agent": {"local_refusal_hosts": ["127.0.0.1:12345"]}})
    r = _local("Something unrecognised happened behind the proxy.", "server_error")
    assert (r.retryable, r.should_fallback) == (False, True)


def test_an_undeclared_route_is_not_treated_as_local(monkeypatch):
    import agent.error_classifier as ec

    monkeypatch.setattr(ec, "load_config_readonly", lambda: {"agent": {"local_refusal_hosts": []}})
    r = _local("Something unrecognised happened behind the proxy.", "server_error")
    assert (r.retryable, r.should_fallback) == (True, False)


def test_a_dead_socket_still_retries_the_same_route():
    """Connection refused on the local route is transient (tunnel restarting), not a refusal."""
    import httpx

    r = classify_api_error(httpx.ConnectError("Connection refused"), provider="custom",
                           model="local-engine", base_url=_LOCAL_ROUTE)
    assert r.reason == FailoverReason.timeout
    assert (r.retryable, r.should_fallback) == (True, False)
