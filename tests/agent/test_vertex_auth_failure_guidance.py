"""Regression coverage for Vertex service-account auth guidance (#121295)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason
from agent.turn_failure_copy import nonretryable_copy
from agent.turn_recovery import _print_nonretryable_auth_guidance


class _Agent:
    log_prefix = ""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _vprint(self, line: str, **_kwargs: object) -> None:
        self.lines.append(line)


_AUTH = SimpleNamespace(reason=FailoverReason.auth, is_auth=True)


@pytest.mark.parametrize("status_code", [401, 403])
def test_vertex_auth_failures_name_service_account_and_iam(status_code: int):
    agent = _Agent()

    _print_nonretryable_auth_guidance(
        agent, _AUTH, status_code=status_code, provider="vertex",
        base_url="https://aiplatform.googleapis.com", model="gemini-2.5-pro",
    )

    guidance = "\n".join(agent.lines)
    assert "service-account" in guidance
    assert "Application Default Credentials" in guidance
    assert "IAM" in guidance
    assert "API key" not in guidance


def test_vertex_chat_copy_names_service_account_instead_of_api_key():
    copy = nonretryable_copy(_AUTH, provider="vertex", model="gemini-2.5-pro", summary="denied")

    assert "service-account" in copy
    assert "API key" not in copy


def test_other_api_key_provider_keeps_existing_key_guidance():
    copy = nonretryable_copy(_AUTH, provider="openrouter", model="openai/gpt-5", summary="denied")

    assert "rejected your API key" in copy

    agent = _Agent()
    _print_nonretryable_auth_guidance(
        agent, _AUTH, status_code=401, provider="openrouter",
        base_url="https://openrouter.ai", model="openai/gpt-5",
    )

    assert "API key" in "\n".join(agent.lines)
