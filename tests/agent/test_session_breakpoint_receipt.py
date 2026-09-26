"""CLI session breakpoint receipts for network-class turn failures.

Zero-network criteria: feed constructed network-class
exceptions to the receipt path and assert the typed breakpoint is appended
with the last successful turn; non-network errors must NOT append. No real
network is touched (classification is string/type based on the exception
object, matching error_classifier's zero-probe design).
"""

from __future__ import annotations

import pytest

from agent.session_breakpoint_receipt import (
    SESSION_SUSPENDED_NETWORK,
    _last_successful_user_turn,
    maybe_append_network_breakpoint,
)


class TestLastSuccessfulTurn:
    def test_completed_exchange_counts(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2 (suspended turn)"},
        ]
        assert _last_successful_user_turn(history) == 0

    def test_empty_history(self):
        assert _last_successful_user_turn([]) == -1

    def test_no_completed_exchange(self):
        history = [{"role": "user", "content": "only"}]
        assert _last_successful_user_turn(history) == -1


class TestNetworkBreakpointReceipt:
    def test_dns_failure_appends_typed_receipt(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        exc = OSError(8, "nodename nor servname provided, or not known")
        message = maybe_append_network_breakpoint(
            history, exc, provider="custom", model="m1",
            error_summary="API Error: 502 [Errno 8] nodename nor servname provided",
        )
        assert message is not None, "DNS-class failure must produce a receipt"
        assert history[-1] is message
        assert message["role"] == "assistant"
        assert message["display_kind"] == "session_breakpoint_receipt"
        assert SESSION_SUSPENDED_NETWORK in message["content"]
        assert "last successful exchange is user turn #0" in message["content"]
        meta = message["display_metadata"]
        assert meta["receipt_type"] == SESSION_SUSPENDED_NETWORK
        assert meta["last_successful_turn"] == 0

    def test_connection_refused_appends_receipt(self):
        history = [{"role": "user", "content": "q"}]
        exc = ConnectionError("Connection refused by 127.0.0.1:17654")
        message = maybe_append_network_breakpoint(history, exc)
        assert message is not None
        assert history[-1] is message

    def test_upstream_5xx_does_not_append(self):
        history = [{"role": "user", "content": "q"}]
        exc = Exception("HTTP 502 internal server error from provider")
        message = maybe_append_network_breakpoint(history, exc)
        assert message is None, "upstream 5xx is provider-side, not a local suspension"
        assert len(history) == 1

    def test_auth_failure_does_not_append(self):
        history = [{"role": "user", "content": "q"}]
        exc = Exception("HTTP 401 unauthorized")
        message = maybe_append_network_breakpoint(history, exc)
        assert message is None
        assert len(history) == 1

    def test_malformed_history_is_ignored_not_raised(self):
        message = maybe_append_network_breakpoint(None, OSError(8, "getaddrinfo failed"))
        assert message is None

    def test_receipt_never_raises_on_classifier_breakage(self, monkeypatch):
        history = [{"role": "user", "content": "q"}]
        import agent.error_classifier as classifier_mod
        monkeypatch.setattr(
            classifier_mod, "classify_api_error",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("classifier exploded")),
        )
        message = maybe_append_network_breakpoint(history, OSError(8, "getaddrinfo failed"))
        assert message is None, "receipt path must swallow internal failures"
        assert len(history) == 1
