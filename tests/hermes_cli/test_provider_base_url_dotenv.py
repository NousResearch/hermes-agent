"""Regression tests for #18757.

Provider base URLs declared via ``base_url_env_var`` must resolve from
``~/.hermes/.env`` (via :func:`hermes_cli.config.get_env_value`) — not only
from the process environment (``os.getenv``). API keys already used the
``.env``-aware lookup; base URLs were previously inconsistent.
"""

from unittest.mock import patch

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


# Use Arcee as a representative API-key provider with a base_url_env_var.
# The fix is provider-agnostic — we just need any provider whose registry
# entry has base_url_env_var set.
_PROVIDER = "arcee"
_BASE_URL_ENV = "ARCEE_BASE_URL"
_API_KEY_ENV = "ARCEEAI_API_KEY"
_DOTENV_BASE_URL = "https://custom-arcee.example.com/v1"
_API_KEY_VALUE = "arc-test-from-dotenv"


def _stub_get_env_value(values):
    """Build a get_env_value replacement that reads from ``values`` only."""
    def _impl(key, default=""):
        return values.get(key, default)
    return _impl


@pytest.fixture(autouse=True)
def _registry_has_base_url_env_var():
    """Sanity-guard: the test only makes sense if the provider declares a
    base_url_env_var. If the registry changes, fail loudly."""
    assert PROVIDER_REGISTRY[_PROVIDER].base_url_env_var == _BASE_URL_ENV


class TestResolveApiKeyProviderCredentialsReadsDotenvBaseUrl:
    """``resolve_api_key_provider_credentials()`` must consult ``.env`` for
    the base URL — not only the shell environment."""

    def test_dotenv_base_url_used_when_shell_env_unset(self, monkeypatch):
        # Clear the shell env so ``os.getenv`` returns nothing for the base URL.
        monkeypatch.delenv(_BASE_URL_ENV, raising=False)

        env_values = {
            _BASE_URL_ENV: _DOTENV_BASE_URL,
            _API_KEY_ENV: _API_KEY_VALUE,
        }
        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=_stub_get_env_value(env_values),
        ):
            result = resolve_api_key_provider_credentials(_PROVIDER)

        # Pre-fix this would be the registry default ("https://api.arcee.ai/api/v1")
        # because os.getenv could not see the .env value.
        assert result["base_url"] == _DOTENV_BASE_URL.rstrip("/")

    def test_shell_env_still_works_when_dotenv_empty(self, monkeypatch):
        # Confirm we did not regress the shell-env path.
        monkeypatch.setenv(_BASE_URL_ENV, _DOTENV_BASE_URL)

        env_values = {_API_KEY_ENV: _API_KEY_VALUE}  # no BASE_URL in .env
        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=_stub_get_env_value(env_values),
        ):
            result = resolve_api_key_provider_credentials(_PROVIDER)

        assert result["base_url"] == _DOTENV_BASE_URL.rstrip("/")


class TestApiKeyProviderStatusReadsDotenvBaseUrl:
    """``get_api_key_provider_status()`` (used by ``hermes doctor``) must also
    surface the .env base URL — otherwise users get inconsistent reporting:
    runtime works but ``doctor`` shows the wrong endpoint."""

    def test_doctor_status_reflects_dotenv_base_url(self, monkeypatch):
        monkeypatch.delenv(_BASE_URL_ENV, raising=False)

        env_values = {
            _BASE_URL_ENV: _DOTENV_BASE_URL,
            _API_KEY_ENV: _API_KEY_VALUE,
        }
        with patch(
            "hermes_cli.config.get_env_value",
            side_effect=_stub_get_env_value(env_values),
        ):
            status = get_api_key_provider_status(_PROVIDER)

        # The status should reflect the .env base URL — pre-fix it returned
        # the registry default.
        assert status.get("base_url") == _DOTENV_BASE_URL or status.get("base_url", "").rstrip("/") == _DOTENV_BASE_URL.rstrip("/")
