"""Regression tests for #86570: gateway provider error connection messaging."""

import pytest

from gateway.run import (
    _GATEWAY_CONNECTION_ERROR_RE,
    _gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
)


class TestGatewayConnectionErrorReply:
    def test_connection_error_strings_produce_specific_reply(self):
        samples = [
            "openai.APIConnectionError",
            "httpx.ConnectError: connection refused",
            "ConnectionError: [WinError 10061] No connection could be made",
            "Errno 111 Connection refused",
            "All connection attempts failed: Connection refused",
        ]
        for text in samples:
            assert _looks_like_gateway_provider_error(text), text
            reply = _gateway_provider_error_reply(text)
            assert "not running or is unreachable" in reply, text
            assert "/retry" in reply, text

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
        auth_reply = _gateway_provider_error_reply("provider authentication failed")
        assert "sign-in" in auth_reply.lower() and "/login" in auth_reply
        assert "rate-limiting" in _gateway_provider_error_reply(
            "rate limited after 3 retries"
        ).lower()

    def test_every_reply_names_a_slash_command_and_no_jargon(self):
        """Each shaped reply must give the chat user something they can run; 'provider' and
        'gateway logs' are operator words (the log pointer is the `hermes logs` command)."""
        from gateway.run import _PROVIDER_ERROR_REPLIES
        replies = [reply for _, reply in _PROVIDER_ERROR_REPLIES] + [_gateway_provider_error_reply("zzz")]
        for reply in replies:
            assert any(cmd in reply for cmd in ("/login", "/retry", "/model")), reply
            assert "provider" not in reply.lower(), reply
