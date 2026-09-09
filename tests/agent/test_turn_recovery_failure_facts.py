"""Terminal results carry the narrow machine discriminators (status code, OS errno,
provider error code) alongside the classifier facts (failure_reason /
failure_retryable), so the one-shot usage report can distinguish the observed
terminal provider-failure classes (broken pipe, overload, usage-limit wall)
without parsing message text."""

import errno
import os

from agent.turn_recovery import (
    _errno_of, _failure_discriminators, max_retries_exhausted_result, nonretryable_client_error_result,
)
from agent.error_classifier import ClassifiedError, FailoverReason


class _FakeAgent:
    """Bare stand-in: only the helpers the terminal path calls."""

    log_prefix = ""

    def _flush_status_buffer(self):
        pass

    def _summarize_api_error(self, error):
        return "summary"

    def _emit_status(self, *_a, **_k):
        pass

    def _buffer_vprint(self, *_a, **_k):
        pass

    def _vprint(self, *_a, **_k):
        pass

    def _vlines(self, *_a, **_k):
        pass

    def _print_nonretryable_auth_guidance(self, *a, **k):
        pass

    def _dump_api_request_debug(self, *a, **k):
        pass

    def _persist_session(self, messages, history):
        pass


class _BodyError(Exception):
    """API error shaped like the SDK's: carries a structured ``body`` and ``status_code``."""

    def __init__(self, message, body=None):
        super().__init__(message)
        self.body: dict = body or {}
        self.status_code = None


class _NoBodyWrapper(Exception):
    """SDK-style wrapper: no body of its own; the provider error is the cause."""


def _classified(reason=FailoverReason.timeout, status_code=None, retryable=True, message=""):
    return ClassifiedError(
        reason=reason, status_code=status_code, provider="p", model="m", message=message,
        retryable=retryable,
    )


class TestErrnoOf:
    def test_reads_errno_directly(self):
        err = OSError(errno.EPIPE, "Broken pipe")
        assert _errno_of(err) == errno.EPIPE

    def test_walks_cause_chain_through_sdk_wrapper(self):
        inner = OSError(errno.EPIPE, "Broken pipe")
        outer = type("WrappedAPIError", (Exception,), {})("upstream killed")
        outer.__cause__ = inner
        assert _errno_of(outer) == errno.EPIPE

    def test_no_errno_returns_none(self):
        assert _errno_of(ValueError("no errno here")) is None


class TestFailureDiscriminators:
    def test_status_only_when_classified_carries_it(self):
        out = _failure_discriminators(None, _classified(status_code=503))
        assert out == {"failure_status_code": 503}

    def test_errno_from_transport_error(self):
        out = _failure_discriminators(OSError(errno.EPIPE, "Broken pipe"), _classified())
        assert out == {"failure_errno": errno.EPIPE}

    def test_provider_code_extracted_from_error_body(self):
        err = _BodyError("limit reached", body={"error": {"type": "usage_limit_reached", "message": "limit reached"}})
        out = _failure_discriminators(err, _classified(FailoverReason.billing, 429, retryable=False))
        assert out == {
            "failure_status_code": 429,
            "failure_provider_code": "usage_limit_reached",
        }

    def test_provider_code_prefers_code_over_type(self):
        err = _BodyError("quota", body={"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}})
        out = _failure_discriminators(err, _classified(FailoverReason.billing, 429, retryable=False))
        assert out["failure_provider_code"] == "insufficient_quota"

    def test_provider_code_resolved_through_wrapped_exception(self):
        """The classifier resolves the error body through the cause chain; the
        discriminator must yield the same token when the provider error is
        wrapped and the wrapper itself carries no body."""
        inner = _BodyError(
            "Your account has reached its usage limit.",
            body={"error": {"type": "usage_limit_reached", "message": "limit reached"}},
        )
        wrapped = _NoBodyWrapper("API call failed")
        wrapped.__cause__ = inner
        out = _failure_discriminators(wrapped, _classified(FailoverReason.billing, 429, retryable=False))
        assert out == {
            "failure_status_code": 429,
            "failure_provider_code": "usage_limit_reached",
        }

    def test_no_provider_code_without_body(self):
        out = _failure_discriminators(_BodyError("credits exhausted"), _classified(FailoverReason.billing, 429, retryable=False))
        assert "failure_provider_code" not in out

    def test_empty_when_nothing_to_say(self):
        assert _failure_discriminators(ValueError("x"), _classified()) == {}


def _run_max_retries(api_error, classified):
    agent = _FakeAgent()
    return max_retries_exhausted_result(
        agent, api_error, classified, max_retries=3, is_rate_limited=False,
        error_msg="e", api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=3, approx_tokens=0,
        provider="p", base_url="https://x", model="m",
    )


def _run_nonretryable(api_error, classified):
    agent = _FakeAgent()
    return nonretryable_client_error_result(
        agent, api_error, classified, status_code=classified.status_code,
        api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=2, approx_tokens=0,
        provider="p", base_url="https://x", model="m",
    )


class TestTerminalResultCarriesDiscriminators:
    def test_broken_pipe_max_retries_result(self):
        result = _run_max_retries(OSError(errno.EPIPE, "Broken pipe"), _classified())
        assert result["failed"] is True
        assert result["failure_reason"] == "timeout"
        assert result["failure_errno"] == errno.EPIPE

    def test_overload_max_retries_result(self):
        result = _run_max_retries(Exception("overloaded"), _classified(FailoverReason.overloaded, 503))
        assert result["failure_reason"] == "overloaded"
        assert result["failure_status_code"] == 503

    def test_usage_limit_reached_max_retries_result(self):
        err = _BodyError(
            "Your account has reached its usage limit.",
            body={"error": {"type": "usage_limit_reached", "message": "Your account has reached its usage limit."}},
        )
        result = _run_max_retries(err, _classified(FailoverReason.billing, 429, retryable=False))
        assert result["failure_reason"] == "billing"
        assert result["failure_status_code"] == 429
        assert result["failure_provider_code"] == "usage_limit_reached"


class TestClassifierFactContractConsistency:
    """Every touched terminal path that emits discriminators must also carry the
    classifier facts (failure_reason / failure_retryable) they are read with."""

    def test_content_policy_arm_carries_classifier_facts(self):
        err = _BodyError("blocked", body={"error": {"code": "content_policy_violation", "message": "blocked"}})
        result = _run_nonretryable(err, _classified(FailoverReason.content_policy_blocked, 400, retryable=False))
        assert result["failed"] is True
        assert result["failure_reason"] == "content_policy_blocked"
        assert result["failure_retryable"] is False
        assert result["failure_status_code"] == 400
        assert result["failure_provider_code"] == "content_policy_violation"

    def test_generic_nonretryable_arm_carries_classifier_facts(self):
        result = _run_nonretryable(
            _BodyError("nope", body={"error": {"code": "model_not_found", "message": "nope"}}),
            _classified(FailoverReason.model_not_found, 404, retryable=False),
        )
        assert result["failure_reason"] == "model_not_found"
        assert result["failure_retryable"] is False
        assert result["failure_provider_code"] == "model_not_found"

    def test_billing_nonretryable_arm_carries_classifier_facts(self):
        result = _run_nonretryable(
            _BodyError(
                "Your account has reached its usage limit.",
                body={"error": {"type": "usage_limit_reached", "message": "limit"}},
            ),
            _classified(FailoverReason.billing, 429, retryable=False),
        )
        assert result["failure_reason"] == "billing"
        assert result["failure_retryable"] is False
        assert result["failure_status_code"] == 429
        assert result["failure_provider_code"] == "usage_limit_reached"


class TestLiveCodexWallShape:
    """The exact wire shape observed on the real Sol/Codex reviewer-route outage
    (retained request dump, 2026-09-06): the SDK parses the 429 body with the
    narrow error-type token at the TOP level, no ``error`` envelope, and reset
    fields present. Acceptance: the token survives every layer into the report."""

    WIRE_BODY = {
        "type": "usage_limit_reached", "message": "The usage limit has been reached",
        "plan_type": "plus", "resets_at": 1788748901, "eligible_promo": None,
        "resets_in_seconds": 52454,
    }
    WIRE_MSG = (
        "Error code: 429 - {'error': {'type': 'usage_limit_reached', 'message': "
        "'The usage limit has been reached', 'plan_type': 'plus', 'resets_at': 1788748901, "
        "'eligible_promo': None, 'resets_in_seconds': 52454}}"
    )

    def _wire_error(self):
        err = _BodyError(self.WIRE_MSG, body=dict(self.WIRE_BODY))
        err.status_code = 429
        return err

    def test_wire_shape_classifies_rate_limit_with_reset(self):
        from agent.error_classifier import classify_api_error
        c = classify_api_error(self._wire_error(), provider="openai-codex", model="gpt-5.6-sol")
        assert (c.reason.value, c.retryable, c.status_code) == ("rate_limit", True, 429)

    def test_wire_shape_token_reaches_terminal_result_and_report(self):
        from agent.error_classifier import classify_api_error
        import json
        import tempfile
        from hermes_cli.oneshot import _write_usage_file
        err = self._wire_error()
        c = classify_api_error(err, provider="openai-codex", model="gpt-5.6-sol")
        result = _run_max_retries(err, c)
        assert result["failure_provider_code"] == "usage_limit_reached"
        assert result["failure_reason"] == "rate_limit" and result["failure_retryable"] is True
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        _write_usage_file(path, result)
        report = json.load(open(path))
        os.unlink(path)
        assert report["failure_provider_code"] == "usage_limit_reached"
        assert report["failure_status_code"] == 429

    def test_wire_shape_report_leak_guard(self):
        from agent.error_classifier import classify_api_error
        import json
        import tempfile
        from hermes_cli.oneshot import _write_usage_file
        result = _run_max_retries(self._wire_error(), classify_api_error(self._wire_error(), provider="openai-codex", model="m"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        _write_usage_file(path, result)
        blob = open(path).read()
        os.unlink(path)
        for leaked in ("The usage limit has been reached", "plan_type", "plus", "resets_at", "52454"):
            assert leaked not in blob, leaked

    def test_generic_rate_limit_429_has_no_narrow_token(self):
        from agent.error_classifier import classify_api_error
        err = _BodyError("Rate limit exceeded", body={"error": {"message": "Rate limit exceeded", "type": "requests"}})
        err.status_code = 429
        c = classify_api_error(err, provider="openai-codex", model="m")
        d = _failure_discriminators(err, c)
        assert d.get("failure_status_code") == 429
        assert d.get("failure_provider_code") != "usage_limit_reached"

    def test_top_level_type_token_extracted_when_no_code(self):
        d = _failure_discriminators(self._wire_error(), _classified())
        assert d["failure_provider_code"] == "usage_limit_reached"

    def test_sse_frame_nested_error_type_preserved(self):
        from agent.codex_runtime import _raise_stream_error
        frame = {"type": "error", "error": {"type": "usage_limit_reached", "message": "The usage limit has been reached"}}
        try:
            _raise_stream_error(frame)
            raise AssertionError("expected _raise_stream_error to raise")
        except Exception as exc:
            body = getattr(exc, "body", {})
            assert body["error"]["type"] == "usage_limit_reached", body
            d = _failure_discriminators(exc, _classified())
            assert d["failure_provider_code"] == "usage_limit_reached"

    def test_sse_generic_frame_keeps_placeholder_and_code_preference(self):
        from agent.codex_runtime import _raise_stream_error
        frame = {"type": "error", "message": "boom", "code": "server_error"}
        try:
            _raise_stream_error(frame)
            raise AssertionError("expected _raise_stream_error to raise")
        except Exception as exc:
            assert getattr(exc, "body", {})["error"]["type"] == "error"
            d = _failure_discriminators(exc, _classified())
            assert d["failure_provider_code"] == "server_error"
