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


class TestClassifiedReasonIsPinned:
    """The receipt is NAMED for the network boundary; pin that ``meta["reason"]``
    carries the classifier's verdict (not a hardcoded literal) across the exact
    transport types this feature exists for."""

    @pytest.mark.parametrize(
        "exc",
        [
            OSError(8, "nodename nor servname provided, or not known"),
            ConnectionError("Connection refused by 127.0.0.1:17654"),
            TimeoutError("read timed out"),
        ],
    )
    def test_transport_types_pin_timeout_reason(self, exc):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        message = maybe_append_network_breakpoint(history, exc, provider="custom", model="m1")
        assert message is not None, f"{type(exc).__name__} must be network-class"
        assert message["display_metadata"]["reason"] == "timeout"

    def test_httpx_connect_error_pins_timeout_reason(self):
        """``httpx.ConnectError`` is the one type spanning transient target-unreachable
        and a deterministic local DNS/proxy misconfig — the boundary this feature names.
        It must classify as ``timeout`` and get a receipt."""
        httpx = pytest.importorskip("httpx")
        history = [{"role": "user", "content": "q"}]
        exc = httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
        message = maybe_append_network_breakpoint(history, exc, provider="custom", model="m1")
        assert message is not None, "httpx.ConnectError is a local network suspension"
        assert message["display_metadata"]["reason"] == "timeout"


class TestReceiptSurvivesSettleWipeAndReachesDB:
    """Blocking-review regression: the receipt must live in the AUTHORITATIVE
    ``_session_messages`` list so (1) ``_chat_settle_turn`` reassigning
    ``conversation_history`` from ``turn.result["messages"]`` (empty on error)
    cannot wipe it, and (2) the SQLite flush — which writes rows from
    ``_session_messages`` and uses ``conversation_history`` only as a skip-set —
    actually persists it across a restart."""

    def _session_messages(self):
        return [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2 (suspended)"},
        ]

    def test_receipt_survives_conversation_history_reassignment(self):
        session_messages = self._session_messages()
        exc = OSError(8, "nodename nor servname provided, or not known")
        message = maybe_append_network_breakpoint(session_messages, exc, provider="custom", model="m1")
        assert message is not None
        assert session_messages[-1] is message

        # Reproduce the exact _chat_settle_turn wipe: conversation_history is
        # replaced by turn.result["messages"], which the error path sets to [].
        turn_result = {"messages": [], "failed": True}
        conversation_history = turn_result.get("messages", session_messages)
        assert conversation_history == [], "settle reassigns conversation_history to the empty list"
        # The receipt is unharmed because it lives in _session_messages.
        assert session_messages[-1] is message

    def test_receipt_is_collected_by_real_db_flush(self):
        """Drive the real ``_db_flush_collect``: the receipt row must be emitted
        for the SQLite write and NOT skipped as an already-durable history copy."""
        from agent.session_persistence import _db_flush_collect

        session_messages = self._session_messages()
        exc = OSError(8, "getaddrinfo failed")
        message = maybe_append_network_breakpoint(session_messages, exc, provider="custom", model="m1")
        assert message is not None

        class _FakeAgent:
            session_id = "sess-under-test"
            _flushed_db_message_session_id = None
            _flushed_db_message_ids = None
            _last_flushed_db_idx = 0
            _db_flush_scan_prefix = ()
            _persist_user_message_idx = None
            _pending_cli_user_message = None
            _mute_notification_reply = False
            _persist_user_message_override = None
            _persist_user_message_timestamp = None

        agent = _FakeAgent()
        # conversation_history is the wiped empty list (settle already ran); it is
        # the skip-set, so it must NOT suppress the receipt row.
        batch_rows, batch_msgs = _db_flush_collect(agent, session_messages, conversation_history=[])
        assert message in batch_msgs, "receipt must be written to SQLite, not skipped"
        receipt_rows = [r for r in batch_rows if r.get("display_kind") == "session_breakpoint_receipt"]
        assert len(receipt_rows) == 1
        assert SESSION_SUSPENDED_NETWORK in (receipt_rows[0].get("content") or "")
