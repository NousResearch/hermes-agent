from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


def test_google_gemini_cli_appears_when_runtime_oauth_resolves():
    """Model picker discovery should mirror Gemini CLI runtime auth.

    Hermes can use google-gemini-cli via the official ~/.gemini OAuth store
    even when ~/.hermes/auth.json has no provider or credential-pool row for
    google-gemini-cli.  The picker should therefore ask the runtime resolver
    before hiding the provider.
    """
    with patch(
        "hermes_cli.auth.resolve_gemini_oauth_runtime_credentials",
        return_value={"api_key": "google-oauth-token", "provider": "google-gemini-cli"},
    ), patch("agent.credential_pool.load_pool") as load_pool:
        load_pool.return_value.has_credentials.return_value = False
        load_pool.return_value.has_available.return_value = False
        providers = list_authenticated_providers(
            current_provider="google-gemini-cli",
            current_model="gemini-3.1-pro-preview",
            max_models=50,
        )

    gemini_cli = next((p for p in providers if p["slug"] == "google-gemini-cli"), None)
    assert gemini_cli is not None
    assert gemini_cli["is_current"] is True
    assert "gemini-3.1-pro-preview" in gemini_cli["models"]


def test_google_gemini_cli_hidden_when_runtime_oauth_missing():
    with patch(
        "hermes_cli.auth.resolve_gemini_oauth_runtime_credentials",
        side_effect=RuntimeError("not logged in"),
    ), patch("agent.credential_pool.load_pool") as load_pool:
        load_pool.return_value.has_credentials.return_value = False
        load_pool.return_value.has_available.return_value = False
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    assert all(p["slug"] != "google-gemini-cli" for p in providers)
