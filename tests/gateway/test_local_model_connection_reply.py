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


class TestQuotaExhaustedIsNotAnAuthFailure:
    """A 429/quota envelope must never send the user to re-authenticate valid credentials, and a
    long reset window must be named instead of "wait a moment" (#89401)."""

    def test_quota_envelope_with_auth_preamble_names_reset_window(self):
        reply = _gateway_provider_error_reply(
            "⚠️ Provider authentication failed: Codex provider quota exhausted (429); "
            "retry after 116168s. Credentials are still valid.")
        assert "/login" not in reply and "resets in ~33h" in reply
        body_reply = _gateway_provider_error_reply(
            "API call failed after 3 retries: Error code: 429 - "
            "{'error': {'type': 'usage_limit_reached', 'resets_in_seconds': 30995}}")
        assert "resets in ~9h" in body_reply
        assert "resets in ~5h" in _gateway_provider_error_reply("429 weekly limit reached. Resets in 4hr 5min")
        # A bare 401 inside a timestamp is not a sign-in failure; every real 401 envelope still is,
        # including the ``HTTP 401: Unauthorized`` shape _summarize_api_error emits (control).
        assert "rate-limiting" in _gateway_provider_error_reply("API call failed at 05:14:15,401 status 429")
        assert "/login" not in _gateway_provider_error_reply("request 05:14:15,401 failed")
        for text in ("HTTP 401: Unauthorized", "returned 401.", "HTTP 401, re-authenticate", "Error code: 401 - token rejected"):
            assert "/login" in _gateway_provider_error_reply(text), text

    def test_turn_runner_resolution_quota_failure_names_reset_window_not_login(self, monkeypatch):
        """Production entry: TurnRunner.run_sync with credential resolution raising the quota
        RuntimeError the gateway wraps around a ``codex_rate_limited`` AuthError."""
        from types import SimpleNamespace
        import gateway.run as gateway_run
        from gateway.config import Platform
        from gateway.run_turn_runner import TurnRunner
        from gateway.session import SessionSource
        from gateway.turn_context import TurnContext
        from hermes_cli.auth import AuthError
        from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE

        def _resolve(**_kwargs):
            try:
                raise AuthError("Codex provider quota exhausted (429); retry after 116168s. Credentials are still valid.",
                                code=CODEX_RATE_LIMITED_CODE, relogin_required=False)
            except AuthError as exc:
                raise RuntimeError(str(exc)) from exc

        monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 30)
        runner = SimpleNamespace(_resolve_session_agent_runtime=_resolve,
                                 _get_system_prompt_for_channel=lambda *a, **k: "",
                                 _ephemeral_system_prompt="")
        ctx = TurnContext(source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm"),
                          session_key="slack:C1", user_config={}, message="Hi")
        result = TurnRunner(runner, ctx).run_sync()
        reply = result["final_response"]
        assert "/login" not in reply and "resets in ~33h" in reply and result["api_calls"] == 0
