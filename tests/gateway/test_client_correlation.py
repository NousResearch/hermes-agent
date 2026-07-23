"""Tests for gateway.client_correlation."""

from gateway.client_correlation import format_correlation_log_suffix, parse_correlation_headers


def test_parse_correlation_headers_standard_request_id():
    corr = parse_correlation_headers(
        {"X-Request-Id": "abc-123", "X-Stream-Token": "tok-1"}
    )
    assert corr == {"request_id": "abc-123", "stream_token": "tok-1"}


def test_parse_correlation_headers_request_id_alias():
    corr = parse_correlation_headers({"X-Request-ID": "upper-case"})
    assert corr == {"request_id": "upper-case"}


def test_format_correlation_log_suffix():
    suffix = format_correlation_log_suffix(
        {"request_id": "r1", "stream_token": "s1"},
        session_id="sess-9",
    )
    assert "request_id=r1" in suffix
    assert "stream_token=s1" in suffix
    assert "session=sess-9" in suffix


def test_parse_correlation_headers_empty():
    assert parse_correlation_headers({}) == {}


def test_parse_correlation_headers_truncates_long_values():
    long_id = "x" * 300
    corr = parse_correlation_headers({"X-Request-Id": long_id})
    assert len(corr["request_id"]) == 128
    assert corr["request_id"] == long_id[:128]


def test_parse_correlation_headers_strips_newlines():
    corr = parse_correlation_headers({
        "X-Request-Id": "abc\r\n123",
        "X-Stream-Token": "tok\n1",
    })
    assert "\r" not in corr["request_id"]
    assert "\n" not in corr["request_id"]
    assert "\r" not in corr["stream_token"]
    assert "\n" not in corr["stream_token"]
