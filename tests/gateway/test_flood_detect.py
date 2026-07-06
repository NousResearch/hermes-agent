"""Unit tests for the canonical flood-control predicate (gateway.flood_detect).

The behavior contract: a flood/rate reject is known-not-delivered (callers may
re-seed/retry), an ambiguous failure is maybe-delivered (callers must latch).
The key regression these lock: the old bare ``"rate"`` substring false-matched
ordinary words ("accurate", "moderate", "separate", "generate"), which on a
dup-sensitive seed path could have wrongly classified an ambiguous failure as a
flood and re-seeded into a duplicate bubble.
"""

from gateway.flood_detect import is_flood_error


class _R:
    def __init__(self, success=False, error=None, message_id=None, retryable=False):
        self.success = success
        self.error = error
        self.message_id = message_id
        self.retryable = retryable


class TestFloodDetectPositive:
    def test_retryable_flag_short_circuits(self):
        # Telegram short floods (<=5s): retryable=True, no error string.
        assert is_flood_error(_R(retryable=True)) is True

    def test_telegram_long_flood_control(self):
        assert is_flood_error(_R(error="flood_control:18")) is True

    def test_flood_control_exceeded_phrasing(self):
        assert is_flood_error(_R(error="Flood control exceeded. Retry in 19 seconds")) is True

    def test_retry_after(self):
        assert is_flood_error(_R(error="Too Many Requests: retry after 12")) is True

    def test_too_many_requests(self):
        assert is_flood_error(_R(error="Too Many Requests")) is True

    def test_http_429(self):
        assert is_flood_error(_R(error="[429] slow down")) is True

    def test_rate_limit_variants(self):
        assert is_flood_error(_R(error="rate limit exceeded")) is True
        assert is_flood_error(_R(error="RateLimit hit")) is True
        assert is_flood_error(_R(error="request was rate_limited")) is True
        assert is_flood_error(_R(error="rate-limited by upstream")) is True

    def test_case_insensitive(self):
        assert is_flood_error(_R(error="FLOOD CONTROL EXCEEDED")) is True


class TestFloodDetectNegative:
    def test_none_result(self):
        assert is_flood_error(None) is False

    def test_no_error_string(self):
        assert is_flood_error(_R(error=None)) is False
        assert is_flood_error(_R(error="")) is False

    def test_ambiguous_failures_are_not_flood(self):
        # Might have delivered → must NOT be treated as flood (caller latches).
        for msg in ("Bad Gateway", "connection reset", "Timed out",
                    "chat not found", "bot was blocked by the user",
                    "message is too long"):
            assert is_flood_error(_R(error=msg)) is False, msg

    def test_bare_rate_substring_no_longer_false_matches(self):
        # THE regression this change fixes: the old predicate used a bare
        # "rate" substring, so these ordinary words wrongly classified as flood.
        for msg in ("could not generate response", "accurate result expected",
                    "moderate confidence", "separate the inputs",
                    "operation rated invalid"):
            assert is_flood_error(_R(error=msg)) is False, msg

    def test_success_result_never_flood(self):
        assert is_flood_error(_R(success=True, message_id="42")) is False


class TestDelegationParity:
    """Both legacy entry points must return identical results to the canonical."""

    def test_subagent_roster_reexport_matches(self):
        from gateway.subagent_roster import is_flood_error as roster_pred
        for r in (None, _R(retryable=True), _R(error="flood_control:9"),
                  _R(error="accurate"), _R(error="Bad Gateway"), _R(error="429")):
            assert roster_pred(r) == is_flood_error(r)

    def test_stream_consumer_method_matches(self):
        from gateway.stream_consumer import GatewayStreamConsumer
        pred = GatewayStreamConsumer._is_flood_error
        for r in (None, _R(retryable=True), _R(error="retry after 5"),
                  _R(error="moderate"), _R(error="connection reset"), _R(error="rate limit")):
            # _is_flood_error is a plain method using only `result`; call unbound.
            assert pred(None, r) == is_flood_error(r)
