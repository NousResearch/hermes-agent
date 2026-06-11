"""Regression guard: Anthropic Bedrock streaming transient faults must not be
classified as local validation errors by the main agent loop.

The Anthropic SDK's Bedrock event-stream decoder
(``anthropic/lib/bedrock/_stream_decoder.py``) raises a *bare* ``ValueError``
when a streamed event frame carries a non-200 status:

    raise ValueError(f"Bad response code, expected 200: {response_dict}")

The embedded ``:exception-type`` is frequently a TRANSIENT server fault
(``internalServerException``, ``modelStreamErrorException``,
``throttlingException``, ``serviceUnavailableException``,
``modelTimeoutException``) that AWS explicitly tells callers to retry
("The system encountered an unexpected error during processing. Try your
request again.").  Because the SDK raises a bare ValueError, the HTTP status
is lost (the agent sees ``HTTP None``), and the agent loop's non-retryable
classifier treats every ``ValueError`` / ``TypeError`` as a local programming
bug — aborting the turn instead of retrying.

This test mirrors the exact predicate shape used in
``agent/conversation_loop.py`` so any future refactor must preserve:

    internalServerException ValueError  → NOT local validation (retryable)
    throttlingException ValueError      → NOT local validation (retryable)
    validationException ValueError      → IS local validation (real client bug)
    bare ValueError                     → IS local validation (programming bug)
"""
from __future__ import annotations

import json


def _mirror_agent_predicate(err: BaseException) -> bool:
    """Exact shape of agent/conversation_loop.py's is_local_validation_error.

    Kept in lock-step with the source. If you change one, change both.
    """
    import ssl

    return (
        isinstance(err, (ValueError, TypeError))
        and not isinstance(err, (UnicodeEncodeError, json.JSONDecodeError))
        and not isinstance(err, ssl.SSLError)
        and not (
            isinstance(err, TypeError)
            and "nonetype" in str(err).lower()
            and "not iterable" in str(err).lower()
        )
        # Bedrock streaming transient-fault carve-out.
        and not (
            isinstance(err, ValueError)
            and "bad response code, expected 200" in str(err).lower()
            and any(
                _t in str(err).lower()
                for _t in (
                    "internalserverexception",
                    "modelstreamerrorexception",
                    "throttlingexception",
                    "serviceunavailableexception",
                    "modeltimeoutexception",
                )
            )
        )
    )


def _stream_decoder_error(exception_type: str) -> ValueError:
    """Build a ValueError matching the exact shape raised by
    anthropic/lib/bedrock/_stream_decoder.py for a non-200 event frame."""
    response_dict = {
        "status_code": 400,
        "headers": {
            ":exception-type": exception_type,
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        "body": b'{"message":"The system encountered an unexpected error during processing. Try your request again."}',
    }
    return ValueError(f"Bad response code, expected 200: {response_dict}")


class TestBedrockStreamTransientIsRetryable:

    def test_internal_server_exception_is_not_local_validation(self):
        """internalServerException is a transient AWS fault — must retry."""
        assert not _mirror_agent_predicate(
            _stream_decoder_error("internalServerException")
        )

    def test_throttling_exception_is_not_local_validation(self):
        assert not _mirror_agent_predicate(
            _stream_decoder_error("throttlingException")
        )

    def test_model_stream_error_exception_is_not_local_validation(self):
        assert not _mirror_agent_predicate(
            _stream_decoder_error("modelStreamErrorException")
        )

    def test_service_unavailable_exception_is_not_local_validation(self):
        assert not _mirror_agent_predicate(
            _stream_decoder_error("serviceUnavailableException")
        )

    def test_model_timeout_exception_is_not_local_validation(self):
        assert not _mirror_agent_predicate(
            _stream_decoder_error("modelTimeoutException")
        )

    def test_validation_exception_is_still_local_validation(self):
        """A genuine validationException is a real client bug — retrying
        reproduces it. Must remain classified as non-retryable."""
        assert _mirror_agent_predicate(
            _stream_decoder_error("validationException")
        )

    def test_bare_value_error_is_still_local_validation(self):
        """The carve-out must be narrow: an unrelated bare ValueError is
        still a programming bug."""
        assert _mirror_agent_predicate(ValueError("bad arg"))


class TestAgentLoopSourceStillHasCarveOut:
    """Belt-and-suspenders: the production source must actually include the
    Bedrock transient-stream carve-out. Protects against an accidental revert
    that happens to leave the test file intact."""

    def test_conversation_loop_excludes_bedrock_transient_stream_faults(self):
        import inspect
        from agent import conversation_loop

        src = inspect.getsource(conversation_loop)
        assert "is_local_validation_error" in src
        assert "bad response code, expected 200" in src.lower(), (
            "agent/conversation_loop.py must carve out the Anthropic Bedrock "
            "stream decoder's transient-fault ValueError from the "
            "is_local_validation_error classification."
        )
        assert "internalserverexception" in src.lower()
