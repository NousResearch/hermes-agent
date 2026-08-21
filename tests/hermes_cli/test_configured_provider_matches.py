"""Regression tests for the false "declared by multiple configured providers"
error.

A single ``providers.ollama-cte700air`` entry (with ``name: Ollama cte700air``)
was surfaced twice by :func:`_configured_provider_matches`:

- the ``user_providers`` branch built the slug from the raw dict key
  (``ollama-cte700air``), and
- the ``custom_providers`` branch (the merged legacy list produced by
  ``get_compatible_custom_providers``) built it from the display name
  (``custom:Ollama cte700air``).

Both slugs pointed at the *same* provider, so ``switch_model`` falsely reported
``'qwen3.8:27b' is declared by multiple configured providers``.

The fix aligns both branches on the canonical ``custom:<provider_key>`` identity
(via :func:`custom_provider_slug`), so one config entry yields exactly one slug.
"""

from unittest.mock import patch

from hermes_cli.config import get_compatible_custom_providers
from hermes_cli.model_switch import _configured_provider_matches, switch_model
from hermes_cli.providers import custom_provider_slug

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _user_providers_with_cte700air():
    """The ``providers:`` dict that reproduces the reported bug."""
    return {
        "ollama-cte700air": {
            "name": "Ollama cte700air",
            "base_url": "http://192.168.10.143:11434/v1",
            "models": {"qwen3.8:27b": {}},
        }
    }


def test_single_keyed_provider_yields_single_slug():
    """A keyed ``providers:`` entry must produce exactly one canonical slug.

    Before the fix this returned two slugs (``ollama-cte700air`` from the
    ``user_providers`` branch and ``custom:Ollama cte700air`` from the merged
    ``custom_providers`` branch), which is the reported bug.
    """
    user_providers = _user_providers_with_cte700air()
    custom_providers = get_compatible_custom_providers(
        {"providers": user_providers}
    )

    matches = _configured_provider_matches(
        "qwen3.8:27b", user_providers, custom_providers
    )

    assert matches == {"custom:ollama-cte700air": "qwen3.8:27b"}


def test_slug_matches_custom_provider_slug():
    """The slug for a keyed entry equals ``custom_provider_slug(name, key)``."""
    user_providers = _user_providers_with_cte700air()
    custom_providers = get_compatible_custom_providers(
        {"providers": user_providers}
    )

    matches = _configured_provider_matches(
        "qwen3.8:27b", user_providers, custom_providers
    )

    expected = custom_provider_slug("Ollama cte700air", "ollama-cte700air")
    assert expected == "custom:ollama-cte700air"
    assert set(matches) == {expected}


def test_legacy_custom_providers_without_provider_key_keeps_name_slug():
    """A real legacy ``custom_providers:`` entry (no ``provider_key``) keeps its
    slug derived from the display name — no regression."""
    custom_providers = [
        {"name": "Legacy Endpoint", "models": ["legacy-model"]}
    ]

    matches = _configured_provider_matches(
        "legacy-model", {}, custom_providers
    )

    assert matches == {"custom:legacy-endpoint": "legacy-model"}


def test_switch_model_no_multiple_provider_error():
    """After the fix, switching to a model declared in a single keyed provider
    succeeds instead of raising the "multiple configured providers" error."""
    user_providers = _user_providers_with_cte700air()
    custom_providers = get_compatible_custom_providers(
        {"providers": user_providers}
    )

    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "***",
                 "base_url": "http://192.168.10.143:11434/v1",
                 "api_mode": "",
             },
         ):
        result = switch_model(
            raw_input="qwen3.8:27b",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            user_providers=user_providers,
            custom_providers=custom_providers,
        )

    assert result.success is True, result.error_message
    assert "multiple configured providers" not in (result.error_message or "").lower()
    assert result.target_provider == "custom:ollama-cte700air"


def test_true_duplicate_still_reports_multiple():
    """A model genuinely declared in two *different* providers must still report
    the "multiple configured providers" error."""
    user_providers = {
        "provider-a": {"name": "Provider A", "models": {"shared-model": {}}},
        "provider-b": {"name": "Provider B", "models": {"shared-model": {}}},
    }
    custom_providers = get_compatible_custom_providers(
        {"providers": user_providers}
    )

    matches = _configured_provider_matches(
        "shared-model", user_providers, custom_providers
    )

    assert set(matches) == {"custom:provider-a", "custom:provider-b"}
