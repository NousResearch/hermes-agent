"""Regression tests for Vertex provider resolution in /model switching."""

from hermes_cli.model_switch import switch_model
from hermes_cli.providers import get_label, normalize_provider, resolve_provider_full


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_vertex_is_known_to_shared_provider_registry():
    """The shared /model pipeline resolves providers through hermes_cli.providers."""
    assert normalize_provider("vertex") == "vertex"
    assert normalize_provider("vertex-ai") == "vertex"
    assert normalize_provider("google-vertex") == "vertex"

    resolved = resolve_provider_full("vertex")
    assert resolved is not None
    assert resolved.id == "vertex"
    assert resolved.name == "Google Vertex AI"
    assert resolved.auth_type == "vertex"
    assert get_label("vertex") == "Google Vertex AI"


def test_switch_model_accepts_explicit_vertex_provider(monkeypatch):
    """`/model <model> --provider vertex` should not fail as an unknown provider."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "vertex-access-token",
            "base_url": "https://aiplatform.googleapis.com/v1beta1/projects/test/locations/global/endpoints/openapi",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_cli.models.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="gemini-3-flash-preview",
        current_provider="openai-codex",
        current_model="gpt-5.5",
        current_base_url="https://chatgpt.com/backend-api/codex",
        current_api_key="",
        explicit_provider="vertex",
    )

    assert result.success is True
    assert result.target_provider == "vertex"
    assert result.provider_label == "Google Vertex AI"
    assert result.new_model == "google/gemini-3-flash-preview"
    assert result.api_key == "vertex-access-token"
    assert result.api_mode == "chat_completions"
