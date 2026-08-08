"""Regression tests for iteration-limit summary failures."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers
from agent.chat_completion_helpers import handle_max_iterations


class _UsageLimitError(Exception):
    status_code = 429
    body = {
        "error": {
            "type": "usage_limit_reached",
            "message": "The usage limit has been reached",
            "plan_type": "plus",
            "resets_at": 1_785_376_302,
            "resets_in_seconds": 558_947,
        }
    }


class _ProviderError(Exception):
    def __init__(self, status_code: int, error_type: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"type": error_type, "message": message}}


class _SummaryAgent:
    max_iterations = 90
    model = "test-model"
    provider = "openai-codex"
    base_url = "https://example.invalid/v1"
    _base_url_lower = base_url
    api_mode = "openai"
    max_tokens = None
    reasoning_config = None
    ephemeral_system_prompt = None
    prefill_messages = []
    openrouter_min_coding_score = None
    providers_allowed = []
    providers_ignored = []
    providers_order = []
    provider_sort = None
    provider_require_parameters = False
    provider_data_collection = None
    session_id = "test-session"

    def __init__(self, error: Exception):
        self._error = error
        self._cached_system_prompt = ""

    def _should_sanitize_tool_calls(self):
        return False

    def _copy_reasoning_content_for_api(self, source, target):
        return None

    def _sanitize_api_messages(self, messages):
        return messages

    def _drop_thinking_only_and_merge_users(self, messages):
        return messages

    def _supports_reasoning_extra_body(self):
        return False

    def _is_openrouter_url(self):
        return False

    def _ensure_primary_openai_client(self, *, reason):
        assert reason == "iteration_limit_summary"
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=self._raise_summary_error),
            )
        )

    def _raise_summary_error(self, **kwargs):
        raise self._error


def test_usage_limit_summary_failure_is_actionable_and_hides_provider_payload(caplog):
    messages = [{"role": "user", "content": "do the work"}]

    result = handle_max_iterations(
        _SummaryAgent(_UsageLimitError("raw provider payload")), messages, 90
    )

    assert "maximum number of tool-calling iterations (90)" in result
    assert "usage limit" in result.lower()
    assert "retry" in result.lower()
    assert "switch" in result.lower()
    assert (
        "work completed before the limit remains in the conversation" in result.lower()
    )
    assert "raw provider payload" not in result
    assert "plan_type" not in result
    assert "resets_at" not in result
    assert "558947" not in result
    assert "raw provider payload" not in caplog.text
    assert "reason=rate_limit" in caplog.text


def test_generic_summary_failure_does_not_echo_exception_or_secrets(caplog):
    private_detail = "connection failed api_key=private-test-token"
    messages = [{"role": "user", "content": "do the work"}]

    result = handle_max_iterations(
        _SummaryAgent(RuntimeError(private_detail)), messages, 90
    )

    assert "an error prevented the final summary request" in result.lower()
    assert (
        "work completed before the limit remains in the conversation" in result.lower()
    )
    assert private_detail not in result
    assert "private-test-token" not in result
    assert private_detail not in caplog.text
    assert "private-test-token" not in caplog.text


@pytest.mark.parametrize(
    ("error", "expected_cause", "expected_action"),
    [
        (
            _ProviderError(
                402, "insufficient_quota", "account has insufficient credits"
            ),
            "billing or credit limit",
            "check the provider account",
        ),
        (
            _ProviderError(401, "authentication_error", "invalid bearer token"),
            "provider authentication failed",
            "check the configured credentials",
        ),
    ],
)
def test_summary_failure_uses_classified_action_without_echoing_error(
    error,
    expected_cause,
    expected_action,
):
    messages = [{"role": "user", "content": "do the work"}]

    result = handle_max_iterations(_SummaryAgent(error), messages, 90)

    assert expected_cause in result.lower()
    assert expected_action in result.lower()
    assert str(error) not in result


def test_classifier_failure_does_not_log_untrusted_status_object(monkeypatch, caplog):
    private_detail = "private-status-object"

    class _UnsafeStatus:
        def __str__(self):
            return private_detail

    class _UnsafeStatusError(RuntimeError):
        def __init__(self):
            super().__init__("summary failed")
            self.status_code = _UnsafeStatus()

    error = _UnsafeStatusError()

    def fail_classifier(*args, **kwargs):
        raise RuntimeError("classifier failed")

    monkeypatch.setattr(chat_completion_helpers, "classify_api_error", fail_classifier)

    result = handle_max_iterations(
        _SummaryAgent(error),
        [{"role": "user", "content": "do the work"}],
        90,
    )

    assert "an error prevented the final summary request" in result.lower()
    assert private_detail not in result
    assert private_detail not in caplog.text
    assert "status=None" in caplog.text
