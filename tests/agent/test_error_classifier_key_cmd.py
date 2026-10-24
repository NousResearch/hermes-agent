"""A ``key_cmd`` mint failure fails fast instead of retrying as a transport error (#91108).

``CommandTokenError`` is raised while the provider SDK is inside its own request path, so the SDK
re-raises it as ``APIConnectionError("Connection error.")``. The generic message defeats pattern
matching and the type name matches the transport heuristics, so the real cause survives only in the
exception chain.
"""

from agent.command_token_source import CommandTokenError
from agent.error_classifier import FailoverReason, classify_api_error


class APIConnectionError(Exception):
    """Stands in for the SDK class; the classifier matches it by type name."""


def _classify(error):
    return classify_api_error(error, provider="custom", model="some-model")


def _assert_fails_fast(result):
    assert result.reason == FailoverReason.auth_permanent
    assert result.retryable is False
    assert result.should_fallback is True  # another configured provider can still serve the turn


def test_sdk_wrapped_mint_failure_fails_fast():
    exc = APIConnectionError("Connection error.")
    exc.__cause__ = CommandTokenError("key_cmd for provider 'gateway' exited 1.")
    _assert_fails_fast(_classify(exc))


def test_bare_mint_failure_fails_fast():
    _assert_fails_fast(_classify(CommandTokenError("key_cmd for provider 'gateway' produced no output")))


def test_implicit_context_chain_is_inspected():
    exc = APIConnectionError("Connection error.")
    exc.__context__ = CommandTokenError("key_cmd for provider 'g' exited 1.")
    _assert_fails_fast(_classify(exc))


def test_mint_failure_below_an_intermediate_wrapper_is_found():
    inner = RuntimeError("transport wrapper")
    inner.__cause__ = CommandTokenError("key_cmd for provider 'g' timed out after 15s")
    exc = APIConnectionError("Connection error.")
    exc.__cause__ = inner
    _assert_fails_fast(_classify(exc))


def test_genuine_connection_error_still_retries():
    result = _classify(APIConnectionError("Connection error."))
    assert result.reason == FailoverReason.timeout
    assert result.retryable is True


def test_cyclic_cause_chain_terminates():
    first, second = APIConnectionError("Connection error."), RuntimeError("loop")
    first.__cause__, second.__cause__ = second, first
    assert _classify(first).retryable is True
