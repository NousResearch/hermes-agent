"""Regression tests for #86570: gateway provider error connection messaging."""

import pytest

from gateway.config import Platform
from gateway.run import (
    _GATEWAY_CONNECTION_ERROR_RE,
    _gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
    _sanitize_gateway_final_response,
)

# Terminal-path envelopes for an ESTABLISHED connection that died mid-transfer. Whether the
# endpoint is up is unknowable from these (the same turn may already have been answered by it).
INTERRUPTED_ENVELOPES = (
    "API call failed after 3 retries: httpx.ReadError: [Errno 104] Connection reset by peer",
    "API call failed after 3 retries: ConnectionResetError: [Errno 104] Connection reset by peer",
    "API call failed after 3 retries: httpx.RemoteProtocolError: peer closed connection "
    "without sending complete message body",
    "API call failed after 3 retries: httpx.ReadError: server disconnected without sending a response",
)

# Nothing accepted the connection / no path to the host: "the endpoint is not up" IS the diagnosis
# here, and it is the case the #86570 wording was written for.
UNREACHABLE_ENVELOPES = (
    "API call failed after 3 retries: httpx.ConnectError: [Errno 111] Connection refused",
    "API call failed after 3 retries: ConnectionError: [WinError 10061] No connection could "
    "be made because the target machine actively refused it",
    "API call failed after 3 retries: httpx.ConnectError: [Errno 113] No route to host",
)

# Connection-shaped, but the SDK flattened the cause away; neither diagnosis is supported.
AMBIGUOUS_ENVELOPES = (
    "API call failed after 3 retries: openai.APIConnectionError: Connection error.",
    "❌ API failed after 3 retries — openai.APIConnectionError: Connection error.",
)

# Claims that the configured endpoint stopped/never came up. Asserting one of these about a
# mid-transfer reset sends the user to debug a server that is answering.
_ENDPOINT_DOWN_CLAIMS = ("not running", "not responding", "unreachable", "not started", "is down")


def _claims_the_endpoint_is_down(reply: str) -> bool:
    return any(claim in reply.lower() for claim in _ENDPOINT_DOWN_CLAIMS)


class TestGatewayConnectionErrorReply:
    def test_connection_error_strings_produce_specific_reply(self):
        """A connect that was REFUSED/unroutable is the local-endpoint-down case of #86570.

        A bare ``openai.APIConnectionError`` is not: the SDK kept no cause, so it is equally a
        dropped response from a live endpoint. It stays a recognised provider envelope, but the
        endpoint-down diagnosis is no longer asserted for it (see the distinct-categories test).
        """
        samples = [
            "httpx.ConnectError: connection refused",
            "ConnectionError: [WinError 10061] No connection could be made",
            "Errno 111 Connection refused",
            "All connection attempts failed: Connection refused",
        ]
        for text in samples:
            assert _looks_like_gateway_provider_error(text), text
            reply = _gateway_provider_error_reply(text)
            assert "not responding" in reply.lower(), text
            assert "not running or is unreachable" in reply, text

        assert _looks_like_gateway_provider_error("openai.APIConnectionError")

    def test_broad_connection_phrases_still_map_once_classified(self):
        """Reply selector keeps the full phrase set; the gate does not."""
        for text in (
            "cannot connect to http://127.0.0.1:8033/v1",
            "failed to establish a new connection",
        ):
            reply = _gateway_provider_error_reply(text)
            assert "not running or is unreachable" in reply, text

    def test_prose_cannot_connect_is_not_a_provider_error(self):
        text = (
            "cannot connect to the office VPN from this cafe, "
            "so I used the backup notes instead"
        )
        assert not _looks_like_gateway_provider_error(text)

    def test_other_errors_keep_generic_reply(self):
        for text in (
            "RuntimeError: model returned empty content",
            "Exception: unknown provider",
            "HTTP 500 internal server error",
        ):
            if _looks_like_gateway_provider_error(text):
                reply = _gateway_provider_error_reply(text)
                assert "not running or is unreachable" not in reply, text

    def test_connection_regex_does_not_match_non_connection_error(self):
        assert not _GATEWAY_CONNECTION_ERROR_RE.search("Rate limited after 3 retries")
        assert not _GATEWAY_CONNECTION_ERROR_RE.search("Provider authentication failed")

    def test_auth_and_rate_limit_preserved(self):
        assert "authentication" in _gateway_provider_error_reply(
            "provider authentication failed"
        ).lower()
        assert "rate-limiting" in _gateway_provider_error_reply(
            "rate limited after 3 retries"
        ).lower()

    def test_three_connection_causes_are_three_distinct_categories(self):
        """A dropped response, a refused connect and a cause-free error are different failures.

        Contract, not wording: each cause gets ONE reply, the three replies differ, and only the
        refused/unroutable one may claim the endpoint is down (that claim is pinned to the
        refusal samples by ``test_connection_error_strings_produce_specific_reply`` above).
        """
        interrupted = {_gateway_provider_error_reply(e) for e in INTERRUPTED_ENVELOPES}
        unreachable = {_gateway_provider_error_reply(e) for e in UNREACHABLE_ENVELOPES}
        ambiguous = {_gateway_provider_error_reply(e) for e in AMBIGUOUS_ENVELOPES}

        assert len(interrupted) == 1, interrupted
        assert len(unreachable) == 1, unreachable
        assert len(ambiguous) == 1, ambiguous
        assert len(interrupted | unreachable | ambiguous) == 3

        for reply in interrupted | ambiguous:
            assert reply.strip()
            assert not _claims_the_endpoint_is_down(reply), reply

    @pytest.mark.parametrize("platform", [Platform.TELEGRAM, "slack", "feishu"])
    def test_interrupted_connection_delivery_keeps_precedence_and_redaction(self, platform):
        """The new category rides the real chat path, and takes nothing from the other rows."""
        raw_reset = (
            "API call failed after 3 retries: httpx.ReadError: [Errno 104] Connection reset "
            "by peer (Authorization: Bearer sk-ABCDEF0123456789abcdef0123)"
        )

        sanitized = _sanitize_gateway_final_response(platform, raw_reset)

        assert sanitized.strip()
        assert "sk-ABCDEF" not in sanitized
        assert "Errno 104" not in sanitized
        assert not _claims_the_endpoint_is_down(sanitized), sanitized
        assert sanitized != _gateway_provider_error_reply(UNREACHABLE_ENVELOPES[0])

        # Auth beats policy beats rate-limit beats connection: an envelope carrying BOTH its own
        # marker and connection wording keeps the category it had before the connection split.
        for tainted, clean in (
            ("API call failed after 3 retries: HTTP 401 incorrect api key provided; "
             "connection reset by peer on the retry", "provider authentication failed"),
            ("API call failed after 3 retries: HTTP 400 request was blocked under the provider "
             "safety policy; connection reset by peer", "request was blocked under the safety policy"),
            ("API call failed after 3 retries: HTTP 429 rate limit exceeded for this model; "
             "connection reset by peer", "rate limited after 3 retries"),
        ):
            assert _sanitize_gateway_final_response(platform, tainted) == (
                _gateway_provider_error_reply(clean)), tainted

        # Programmatic consumers still get the bottom exception, byte for byte.
        assert _sanitize_gateway_final_response("local", raw_reset) == raw_reset
