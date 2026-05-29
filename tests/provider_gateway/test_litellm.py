import sys
from unittest.mock import MagicMock, patch
import pytest


def test_litellm_backend_is_available_returns_bool() -> None:
    """Test is_available returns a boolean in real environment."""
    import provider_gateway.litellm_backend as backend

    res = backend.is_available()
    assert isinstance(res, bool)


def test_complete_raises_import_error_when_unavailable() -> None:
    """Test that wrapper throws ImportError and fallback to 0.0/empty when litellm is absent."""
    # Force reload backend with litellm absent from sys.modules
    if "provider_gateway.litellm_backend" in sys.modules:
        del sys.modules["provider_gateway.litellm_backend"]

    # We mock 'litellm' import as failing
    with patch.dict("sys.modules", {"litellm": None}):
        import provider_gateway.litellm_backend as backend

        assert backend.is_available() is False

        with pytest.raises(ImportError, match="The 'litellm' package is not installed"):
            backend.complete("openai/gpt-4o", [])

        assert backend.estimate_cost("openai/gpt-4o", 100, 200) == 0.0
        assert backend.list_models() == []


def test_complete_calls_litellm_completion_mocked() -> None:
    """Test dynamic wrapper behavior when litellm is mock-present in sys.modules."""
    mock_litellm = MagicMock()
    mock_resp = MagicMock()
    mock_litellm.completion.return_value = mock_resp
    mock_litellm.completion_cost.return_value = 0.015
    mock_litellm.model_list = ["openai/gpt-4o", "anthropic/claude-3"]

    # Force reload backend with mock litellm present
    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        if "provider_gateway.litellm_backend" in sys.modules:
            del sys.modules["provider_gateway.litellm_backend"]

        import provider_gateway.litellm_backend as backend

        assert backend.is_available() is True

        messages = [{"role": "user", "content": "hi"}]
        res = backend.complete(
            "openai/gpt-4o",
            messages,
            api_key="my-key",
            api_base="https://my-base.com",
            temperature=0.5,
        )

        assert res is mock_resp
        mock_litellm.completion.assert_called_once_with(
            model="openai/gpt-4o",
            messages=messages,
            stream=False,
            api_key="my-key",
            api_base="https://my-base.com",
            temperature=0.5,
        )

        cost = backend.estimate_cost("openai/gpt-4o", 1000, 500)
        assert cost == 0.015
        mock_litellm.completion_cost.assert_called_once_with(
            model="openai/gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        models = backend.list_models()
        assert models == ["openai/gpt-4o", "anthropic/claude-3"]

    # Cleanup backend to restore real environment after test finishes
    if "provider_gateway.litellm_backend" in sys.modules:
        del sys.modules["provider_gateway.litellm_backend"]
