"""Vertex auth guidance names the service account, not an API key (#121295)."""

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_failure_copy import nonretryable_copy


def _copy(status: int, provider: str = "vertex") -> str:
    classified = ClassifiedError(
        reason=FailoverReason.auth, status_code=status, provider=provider, message="denied", retryable=False,
    )
    return nonretryable_copy(classified, provider=provider, model="gemini-2.5-pro", summary="vendor text")


def _guidance(text: str) -> str:
    return text.split("Provider said:")[0].lower()


def test_vertex_403_names_the_project_role_not_an_api_key():
    guidance = _guidance(_copy(403))
    assert "api key" not in guidance
    assert "vertex ai user" in guidance
    assert "project_id" in guidance


def test_vertex_401_names_the_service_account_not_an_api_key():
    guidance = _guidance(_copy(401))
    assert "api key" not in guidance
    assert "vertex_credentials_path" in guidance
    assert "application-default login" in guidance


def test_gemini_api_key_auth_copy_is_unchanged():
    guidance = _guidance(_copy(401, provider="gemini"))
    assert "api key" in guidance
