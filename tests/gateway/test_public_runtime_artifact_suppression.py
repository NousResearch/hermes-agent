from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _should_suppress_public_runtime_artifact,
)


def _source(chat_type: str):
    return SimpleNamespace(chat_type=chat_type)


def test_suppresses_empty_response_fallback_in_public_group():
    assert _should_suppress_public_runtime_artifact(
        "⚠️ The model returned no response after processing tool results. This can happen with some models — try again or rephrase your question.",
        _source("group"),
    )


def test_suppresses_operation_interrupted_in_public_thread():
    assert _should_suppress_public_runtime_artifact(
        "Operation interrupted: waiting for model response (2.3s elapsed).",
        _source("thread"),
    )


def test_suppresses_retry_boilerplate_in_public_group():
    assert _should_suppress_public_runtime_artifact(
        "⚠️ Empty response from model — retrying (1/3)",
        _source("supergroup"),
    )


def test_suppresses_processing_completed_no_response_fallback_in_public_group():
    assert _should_suppress_public_runtime_artifact(
        "⚠️ Processing completed but no response was generated. This may be a transient error — try sending your message again.",
        _source("group"),
    )


def test_keeps_same_runtime_notice_visible_in_dm():
    assert not _should_suppress_public_runtime_artifact(
        "⚠️ The model returned no response after processing tool results. This can happen with some models — try again or rephrase your question.",
        _source("dm"),
    )


def test_does_not_suppress_normal_public_answer():
    assert not _should_suppress_public_runtime_artifact(
        "Here is the requested proof summary.",
        _source("group"),
    )


def test_does_not_suppress_public_answer_that_mentions_runtime_phrase():
    assert not _should_suppress_public_runtime_artifact(
        "If your process logs say Operation interrupted, check the gateway log before retrying.",
        _source("group"),
    )


def test_does_not_suppress_public_answer_starting_with_runtime_phrase():
    assert not _should_suppress_public_runtime_artifact(
        "Operation interrupted means the OS delivered SIGINT; here is how to debug it.",
        _source("group"),
    )


def test_status_callback_suppresses_public_retry_artifact():
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM,
        "warn",
        "⚠️ Empty response from model — retrying (1/3)",
        source=_source("supergroup"),
    ) is None


def test_status_callback_keeps_dm_retry_artifact():
    message = "⚠️ Empty response from model — retrying (1/3)"
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM,
        "warn",
        message,
        source=_source("dm"),
    ) == message
