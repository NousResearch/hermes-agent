import pytest
from agent.google_code_assist import CodeAssistError, ProjectIdRequiredError

def test_code_assist_error_init():
    """Test CodeAssistError initialization and default values."""
    err = CodeAssistError("test message")
    assert str(err) == "test message"
    assert err.code == "code_assist_error"
    assert err.status_code is None
    assert err.response is None
    assert err.retry_after is None
    assert err.details == {}

def test_code_assist_error_init_with_kwargs():
    """Test CodeAssistError initialization with custom kwargs."""
    details = {"reason": "MODEL_CAPACITY_EXHAUSTED"}
    response = object()

    err = CodeAssistError(
        "test message",
        code="custom_code",
        status_code=429,
        response=response,
        retry_after=60.0,
        details=details
    )

    assert str(err) == "test message"
    assert err.code == "custom_code"
    assert err.status_code == 429
    assert err.response is response
    assert err.retry_after == 60.0
    assert err.details == details

def test_project_id_required_error_init():
    """Test ProjectIdRequiredError initialization."""
    err = ProjectIdRequiredError()
    assert str(err) == "GCP project id required for this tier"
    assert err.code == "code_assist_project_id_required"

def test_project_id_required_error_init_with_custom_message():
    """Test ProjectIdRequiredError initialization with a custom message."""
    err = ProjectIdRequiredError("Custom message")
    assert str(err) == "Custom message"
    assert err.code == "code_assist_project_id_required"
