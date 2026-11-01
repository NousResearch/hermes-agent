"""A 429's rate-limit headers are kept and logged, so the window that refused the call
(5-hour, weekly, per-minute) is on record after the fact."""

import logging
from types import SimpleNamespace

from agent.agent_runtime_helpers import extract_api_error_context, ratelimit_headers


def _err(headers, status=429):
    err = Exception("Error code: 429 - rate_limit_error")
    err.status_code = status
    err.response = SimpleNamespace(headers=headers)
    return err


def test_unified_window_headers_are_kept_and_logged(caplog):
    headers = {
        "Anthropic-Ratelimit-Unified-Status": "rejected",
        "anthropic-ratelimit-unified-5h-utilization": "1.0",
        "anthropic-ratelimit-unified-5h-reset": "1793602800",
        "anthropic-ratelimit-unified-7d-utilization": "0.61",
        "retry-after": "836",
        "request-id": "req_x",
        "content-type": "application/json",
    }
    with caplog.at_level(logging.INFO, logger="agent.agent_runtime_helpers"):
        ctx = extract_api_error_context(_err(headers))
    kept = ctx["ratelimit_headers"]
    assert kept["anthropic-ratelimit-unified-status"] == "rejected"
    assert kept["anthropic-ratelimit-unified-5h-utilization"] == "1.0"
    assert kept["retry-after"] == "836"
    assert "request-id" not in kept and "content-type" not in kept
    line = next(r.getMessage() for r in caplog.records if "Rate-limit headers" in r.getMessage())
    assert "anthropic-ratelimit-unified-5h-utilization=1.0" in line
    assert "on 429" in line


def test_no_rate_limit_headers_logs_nothing(caplog):
    with caplog.at_level(logging.INFO, logger="agent.agent_runtime_helpers"):
        ctx = extract_api_error_context(_err({"content-type": "application/json"}))
    assert "ratelimit_headers" not in ctx
    assert not any("Rate-limit headers" in r.getMessage() for r in caplog.records)


def test_values_are_capped_and_unreadable_headers_are_skipped():
    assert ratelimit_headers({"x-ratelimit-remaining-tokens": "9" * 500})["x-ratelimit-remaining-tokens"] == "9" * 120
    assert ratelimit_headers(object()) == {}
