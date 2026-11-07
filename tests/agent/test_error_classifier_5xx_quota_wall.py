"""A 503/529 that reports spent quota falls back instead of backing off on the same model.

Credential-pooling gateways (e.g. CLIProxyAPI) answer 503 once every upstream credential for the
model is spent. Same-model backoff cannot succeed, and the key to the gateway is healthy (other
models still work), so the verdict falls back without benching it. Capacity 503/529s keep backing off.
"""
import dataclasses
import time
from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error

_POOL_WALL = (
    "auth_unavailable: no auth available (providers=codex, model=gpt-6-astra; "
    "last upstream error: usage_limit_reached: The usage limit has been reached)"
)


class _APIError(Exception):
    def __init__(self, message, status_code, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


def _classify(message, status, **kwargs):
    return classify_api_error(_APIError(message, status, **kwargs), provider="custom", model="gpt-6-astra")


def _route(monkeypatch, error, error_context, model="m"):
    """Drive ``route_classified_error`` for ``error`` with one fallback configured; return the kwargs
    ``_try_activate_fallback`` received."""
    from agent import conversation_loop
    from agent.turn_recovery import route_classified_error

    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda *args: "prompt")
    seen = {}
    agent = SimpleNamespace(
        _fallback_index=0, _fallback_chain=[{"provider": "fallback"}], _credential_pool=None,
        _buffer_diagnostic_status=lambda *_: None,
        _try_activate_fallback=lambda **kwargs: seen.update(kwargs) or True,
    )
    verdict = route_classified_error(
        agent, error, classify_api_error(error, provider="custom", model=model), SimpleNamespace(),
        error_msg=str(error), error_context=error_context, recovered_with_pool=False,
        base_url="http://127.0.0.1:8317/v1", model=model, messages=[], api_messages=[], system_message="",
        active_system_prompt="", conversation_history=[], retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id="task",
    )
    assert verdict.action == "break"
    return seen


def _assert_falls_back_without_benching(result):
    assert result.reason == FailoverReason.upstream_rate_limit
    assert result.should_fallback is True
    assert result.should_rotate_credential is False
    assert result.retryable is True


@pytest.mark.parametrize("status", [503, 529])
def test_pool_quota_wall_falls_back_without_benching_key(status):
    result = _classify(_POOL_WALL, status, body={"error": {"message": _POOL_WALL, "type": "server_error"}})
    _assert_falls_back_without_benching(result)
    assert result.error_context == {}


@pytest.mark.parametrize("status", [503, 529])
@pytest.mark.parametrize("suffix", [". Please try again in 2 hours", ". Please retry after a while"])
def test_quota_wall_with_retry_wording_still_falls_back(status, suffix):
    _assert_falls_back_without_benching(_classify(_POOL_WALL + suffix, status))


@pytest.mark.parametrize("status", [503, 529])
def test_quota_wall_carries_reset_window_from_message(status):
    before = time.time()
    result = _classify(_POOL_WALL + "; resets in 1 hour", status)
    _assert_falls_back_without_benching(result)
    assert before + 3590 <= result.error_context["reset_at"] <= time.time() + 3610


def test_quota_wall_carries_retry_after_header_window():
    before = time.time()
    result = _classify(_POOL_WALL, 503, headers={"retry-after": "120"})
    _assert_falls_back_without_benching(result)
    assert before + 110 <= result.error_context["reset_at"] <= time.time() + 130


@pytest.mark.parametrize("code", ["usage_limit_reached", "insufficient_quota", "terminal_quota_exhausted"])
def test_structured_exhaustion_code_falls_back(code):
    body = {"error": {"code": code, "message": "Service Unavailable"}}
    _assert_falls_back_without_benching(_classify("Service Unavailable", 503, body=body))


@pytest.mark.parametrize("message", [
    "Quota exceeded for model gpt-6-astra",
    "error: quota_exceeded",
    "Monthly quota exhausted",
    "You exceeded your current quota, please check your plan and billing details",
    "insufficient_quota",
    "Key limit exceeded",
])
def test_exhaustion_wording_falls_back(message):
    _assert_falls_back_without_benching(_classify(message, 503))


def test_openrouter_wrapped_quota_wall_names_the_upstream():
    body = {"error": {"code": 503, "message": "Provider returned error",
                      "metadata": {"provider_name": "Codex", "raw": "usage_limit_reached"}}}
    result = _classify("Provider returned error: usage_limit_reached", 503, body=body)
    _assert_falls_back_without_benching(result)
    assert result.error_context["upstream_provider"] == "Codex"


@pytest.mark.parametrize("status, message", [
    (503, "Service temporarily overloaded, please retry"),
    (529, "Service temporarily overloaded, please retry"),
    # Capacity, not exhaustion: overload wording wins, and bare "quota"/"limit exceeded" is not enough.
    (503, "Service temporarily overloaded: concurrency limit exceeded"),
    (503, "concurrency limit exceeded"),
    (529, "Overloaded: quota service temporarily unavailable"),
    (529, "overloaded_error: Overloaded (usage limit scheduler busy)"),
])
def test_capacity_errors_still_back_off(status, message):
    result = classify_api_error(_APIError(message, status))
    assert result.reason == FailoverReason.overloaded
    assert result.retryable is True
    assert result.should_fallback is False


_FUTURE = time.time() + 86400  # stays in the future for the whole test session


@pytest.mark.parametrize("extracted, expected", [({}, "classifier"), ({"reset_at": _FUTURE}, _FUTURE)],
                         ids=["classifier", "extracted"])
def test_eager_fallback_gets_the_reset_window(monkeypatch, extracted, expected):
    """The turn's separately extracted context may miss a reset the classifier found (nested
    bodies, long messages); the fallback cooldown must still get it."""
    error = _APIError(_POOL_WALL + "; resets in 1 hour", 503)
    before = time.time()
    seen = _route(monkeypatch, error, extracted, model="gpt-6-astra")
    assert seen["reason"] == FailoverReason.upstream_rate_limit
    if expected == "classifier":
        assert before + 3590 <= seen["reset_at"] <= time.time() + 3610
    else:
        assert seen["reset_at"] == expected


def test_eager_fallback_without_any_reset_passes_none(monkeypatch):
    seen = _route(monkeypatch, _APIError(_POOL_WALL, 503), {"reset_at": "not a time"})
    assert seen["reason"] == FailoverReason.upstream_rate_limit
    assert seen["reset_at"] is None


def _absolute_reset_forms(at):
    return [
        str(int(at)), int(at * 1000), str(int(at * 1000)),
        datetime.fromtimestamp(at, UTC).isoformat(),
        format_datetime(datetime.fromtimestamp(at, UTC), usegmt=True),
    ]


@pytest.mark.parametrize("form", range(5))
def test_absolute_reset_fields_are_timestamps_not_durations(form):
    """``resets_at``/``reset_at`` are absolute (epoch s/ms or ISO, numeric or string, or an HTTP-date):
    a string epoch read as a duration would bench the primary for decades."""
    at = time.time() + 3600
    body = {"error": {"code": "usage_limit_reached", "message": "Service Unavailable"},
            "resets_at": _absolute_reset_forms(at)[form]}
    result = _classify("Service Unavailable", 503, body=body)
    _assert_falls_back_without_benching(result)
    assert at - 5 <= result.error_context["reset_at"] <= at + 5


def test_eager_fallback_gets_a_top_level_string_epoch_reset(monkeypatch):
    """Real turn flow: the extractor reads only ``body["error"]``, so this reset comes from the
    classifier alone; the cooldown must get the instant, not ~1.7e9 s."""
    from agent.agent_runtime_helpers import extract_api_error_context

    at = time.time() + 3600
    error = _APIError("Service Unavailable", 503, body={
        "error": {"code": "usage_limit_reached", "message": "Service Unavailable"}, "resets_at": str(int(at)),
    })
    seen = _route(monkeypatch, error, extract_api_error_context(error))
    assert at - 5 <= seen["reset_at"] <= at + 5


def test_past_absolute_reset_defers_to_retry_after():
    body = {"error": {"code": "usage_limit_reached", "message": "Service Unavailable"},
            "resets_at": str(int(time.time() - 600))}
    before = time.time()
    result = _classify("Service Unavailable", 503, body=body, headers={"retry-after": "120"})
    assert before + 110 <= result.error_context["reset_at"] <= time.time() + 130


@pytest.mark.parametrize("field, value", [
    # .NET's ``DateTime.MinValue``/``MaxValue`` sentinels serialise naive. Windows cannot convert them to
    # an epoch at all; elsewhere MaxValue parses, centuries out.
    ("resets_at", "0001-01-01T00:00:00"), ("resets_at", "9999-12-31T23:59:59"),
    ("resets_at", "2999-01-01T00:00:00+00:00"), ("resets_in_seconds", 40 * 86400),
    ("resets_at", 10**400), ("resets_in_seconds", 10**400),
    ("resets_at", "inf"), ("resets_in_seconds", "inf"), ("retry_after", float("inf")),
    ("resets_at", "Fri, 01 Jan 100000000000000000000000 00:00:00 GMT"),
], ids=["min-iso", "max-iso", "far-future-at", "far-future-duration", "huge-int-at", "huge-int-duration",
        "inf-at", "inf-duration", "inf-retry-after", "overflowing-http-date-at"])
def test_unusable_reset_is_skipped_not_raised(field, value):
    body = {"error": {"code": "usage_limit_reached", "message": "Service Unavailable"}, field: value}
    before = time.time()
    result = _classify("Service Unavailable", 503, body=body, headers={"retry-after": "120"})
    assert before + 110 <= result.error_context["reset_at"] <= time.time() + 130


@pytest.mark.parametrize("headers, suffix", [
    ({"retry-after": "inf"}, ""), ({"retry-after": str(40 * 86400)}, ""),
    ({}, "; retry after " + "9" * 400 + " seconds"), ({}, "; resets in 1000 hours"),
], ids=["inf-header", "far-future-header", "inf-message", "far-future-message"])
def test_unusable_header_or_message_reset_is_skipped(headers, suffix):
    result = _classify(_POOL_WALL + suffix, 503, headers=headers)
    _assert_falls_back_without_benching(result)
    assert result.error_context == {}


@pytest.mark.parametrize("make_reset", [lambda at: int(at * 1000), lambda at: str(int(at))],
                         ids=["epoch-ms", "string-epoch"])
def test_429_reads_absolute_reset_as_an_instant(make_reset):
    """The 429 path shares the parser: epoch ms was read as seconds, a string epoch as a duration."""
    at = time.time() + 3600
    body = {"error": {"message": "Rate limit exceeded"}, "resets_at": make_reset(at)}
    result = _classify("Rate limit exceeded", 429, body=body)
    assert result.reason == FailoverReason.rate_limit
    assert at - 5 <= result.error_context["reset_at"] <= at + 5


@pytest.mark.parametrize("extracted, expected", [
    ({"reset_at": _FUTURE}, _FUTURE), ({"reset_at": "garbage"}, "classifier"),
    ({"reset_at": "2999-01-01T00:00:00+00:00"}, "classifier"), (None, "classifier"),
], ids=["valid-extracted", "invalid-extracted", "sentinel-extracted", "no-context"])
def test_effective_reset_at_prefers_a_valid_extracted_reset(extracted, expected):
    """Shared by every fallback site (eager, non-retryable, max-retries exhausted)."""
    from agent.fallback_cooldown import effective_reset_at

    classified = SimpleNamespace(error_context={"reset_at": _FUTURE + 1})
    assert effective_reset_at(extracted, classified) == (_FUTURE + 1 if expected == "classifier" else expected)


def test_effective_reset_at_without_any_reset_is_none():
    from agent.fallback_cooldown import effective_reset_at

    assert effective_reset_at({"reset_at": "garbage"}, SimpleNamespace(error_context=None)) is None


@pytest.mark.parametrize("site", ["non-retryable", "max-retries"])
def test_unrecovered_error_fallback_gets_the_classifier_reset(monkeypatch, site):
    """The non-retryable and max-retries-exhausted fallback sites hand off the same reset as the eager
    one, e.g. after ``fallback.min_switch_reset_seconds`` or pool rotation skipped the eager switch."""
    from agent import conversation_loop
    from agent.turn_api_error import settle_unrecovered_error

    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda *args: "prompt")
    monkeypatch.setattr(conversation_loop, "_is_copilot_provider", lambda *args: False)
    seen, statuses = {}, []
    agent = SimpleNamespace(
        _has_pending_fallback=lambda: True, _buffer_diagnostic_status=statuses.append,
        _try_activate_fallback=lambda **kwargs: seen.update(kwargs) or True,
    )
    error = _APIError(_POOL_WALL + "; resets in 1 hour", 503)
    classified = classify_api_error(error, provider="custom", model="m")
    if site == "non-retryable":
        # upstream_rate_limit is a retryable client reason, so it never reaches this site: drive it
        # with a billing verdict carrying the same classifier reset.
        classified = dataclasses.replace(classified, reason=FailoverReason.billing, retryable=False)
    retry = SimpleNamespace(copilot_stale_cred_retry_attempted=False, primary_recovery_attempted=True)
    before = time.time()
    verdict = settle_unrecovered_error(
        agent, api_error=error, classified=classified, _retry=retry, status_code=503, error_msg=str(error),
        is_context_length_error=False, is_rate_limited=True, _is_zai_coding_overload=False,
        _provider="custom", _base="http://127.0.0.1:8317/v1", _model="m", messages=[], api_messages=[],
        api_kwargs=None, active_system_prompt="", conversation_history=None, approx_tokens=10,
        retry_count=0 if site == "non-retryable" else 3, max_retries=3, compression_attempts=0,
        api_call_count=3, error_context={"reset_at": time.time() - 600},
    )
    assert verdict.action == "break"
    assert ("Max retries" in statuses[-1]) == (site == "max-retries")
    assert seen["reason"] == classified.reason
    assert before + 3590 <= seen["reset_at"] <= time.time() + 3610


@pytest.mark.parametrize("make_reset, headers, window", [
    (lambda now: int(now - 600), {"retry-after": "120"}, 120),
    (lambda now: str(int(now - 600)), {"retry-after": "120"}, 120),
    (lambda now: format_datetime(datetime.fromtimestamp(now + 3600, UTC), usegmt=True), {}, 3600),
    (lambda now: "2999-01-01T00:00:00+00:00", {"retry-after": "120"}, 120),
], ids=["past-epoch", "past-string-epoch", "future-http-date", "far-future-sentinel"])
def test_eager_fallback_ignores_an_extracted_reset_that_is_not_a_future_instant(
    monkeypatch, make_reset, headers, window
):
    """The extractor reads ``body["error"]["resets_at"]`` as-is; an expired or unparseable value must
    not override the classifier's reset and leave the cooldown at its 60 s default."""
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.fallback_cooldown import _provider_reset_delay

    error = _APIError("Service Unavailable", 503, headers=headers, body={
        "error": {"code": "usage_limit_reached", "message": "Service Unavailable", "resets_at": make_reset(time.time())},
    })
    seen = _route(monkeypatch, error, extract_api_error_context(error))
    assert window - 15 <= _provider_reset_delay(seen["reset_at"]) <= window + 5
