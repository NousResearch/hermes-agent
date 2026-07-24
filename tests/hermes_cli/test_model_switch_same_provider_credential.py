"""Same-provider switch must preserve working credentials when the resolver
returns empty api_key/base_url (issue #44490 / opencode-go 401).

During a same-provider model switch, ``switch_model`` re-resolves credentials
via ``resolve_runtime_provider``. For some providers (e.g. opencode-go) that
resolution can return empty key/URL even though the active session already has
working credentials. Overwriting the session values with those empties causes
the next request to 401.

These tests mock an empty (and non-empty) resolver result and assert the
fallback / override behaviour in the ``provider_changed=False`` branch.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _run_same_provider_switch(runtime: dict, **switch_kwargs):
    """Run a same-provider switch_model with network/catalog deps mocked out."""
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=runtime,
        ),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value=_MOCK_VALIDATION,
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=switch_kwargs.get("raw_input", "kimi-k2.5"),
            current_provider=switch_kwargs.get("current_provider", "opencode-go"),
            current_model=switch_kwargs.get("current_model", "mimo-v2.5"),
            current_api_key=switch_kwargs.get("current_api_key", "sk-current-key"),
            current_base_url=switch_kwargs.get(
                "current_base_url", "https://api.opencode-go.com/v1"
            ),
        )


class TestSameProviderCredentialFallback:
    """Same-provider switch should preserve current credentials when resolution
    returns empty values."""

    def test_empty_resolved_key_falls_back_to_current(self):
        """When resolve_runtime_provider returns empty api_key, keep current key."""
        result = _run_same_provider_switch(
            {
                "api_key": "",
                "base_url": "https://api.opencode-go.com/v1",
                "api_mode": "chat",
            }
        )

        assert result.success is True, f"switch_model failed: {result.error_message}"
        assert result.api_key == "sk-current-key"

    def test_empty_resolved_base_falls_back_to_current(self):
        """When resolve_runtime_provider returns empty base_url, keep current."""
        result = _run_same_provider_switch(
            {
                "api_key": "sk-new-key",
                "base_url": "",
                "api_mode": "chat",
            }
        )

        assert result.success is True, f"switch_model failed: {result.error_message}"
        assert result.base_url == "https://api.opencode-go.com/v1"

    def test_non_empty_resolved_values_preferred_over_current(self):
        """When resolve returns non-empty values, use them (credential rotation)."""
        result = _run_same_provider_switch(
            {
                "api_key": "sk-rotated-key",
                "base_url": "https://api.opencode-go.com/v2",
                "api_mode": "chat",
            },
            current_api_key="sk-old-key",
            current_base_url="https://api.opencode-go.com/v1",
        )

        assert result.success is True, f"switch_model failed: {result.error_message}"
        assert result.api_key == "sk-rotated-key"
        assert result.base_url == "https://api.opencode-go.com/v2"

    def test_both_empty_falls_back_to_both_current(self):
        """When resolve returns both api_key and base_url as empty, keep both current."""
        result = _run_same_provider_switch(
            {
                "api_key": "",
                "base_url": "",
                "api_mode": "",
            }
        )

        assert result.success is True, f"switch_model failed: {result.error_message}"
        assert result.api_key == "sk-current-key"
        assert result.base_url == "https://api.opencode-go.com/v1"
