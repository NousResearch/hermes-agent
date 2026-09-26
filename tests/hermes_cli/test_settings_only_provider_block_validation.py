"""Regression tests for #120020: a settings-only ``providers.<slug>`` block must not
reroute model validation into the custom-endpoint branch.

Repro: ``providers.openai-codex: {request_timeout_seconds: 3600}`` (no
``base_url``/``url``/``api``) made ``_validate_switch`` remap ``openai-codex``
to ``custom:openai-codex`` purely on ``pdef.source == "user-config"``. The
custom branch probes ``GET {base_url}/models`` on the Codex backend (no listing
available to clients), gets ``models=None``, and hard-rejects because
``codex_responses`` is not in its soft-accept set — every assignment path
(``/model --provider``, desktop Settings, onboarding confirm) failed.

Hermetic: credentials, aliasing, and (where noted) the network probe are
mocked; no network, no credentials needed.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_SETTINGS_ONLY = {"request_timeout_seconds": 3600, "stale_timeout_seconds": 300}
_WITH_ENDPOINT = {"base_url": "https://example.com/v1", "request_timeout_seconds": 3600}

_RUNTIME_CODEX = {
    "api_key": "test-key",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_mode": "codex_responses",
}

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run(raw_input, user_providers, validation=_ACCEPTED):
    """Drive PATH A (explicit ``--provider``) with everything hermetic except,
    by default, the real ``validate_requested_model`` (overridden per test)."""
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider",
               side_effect=lambda model, provider: model), \
         patch("hermes_cli.models_validate.validate_requested_model",
               return_value=validation) as validate, \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value=dict(_RUNTIME_CODEX)):
        result = switch_model(
            raw_input=raw_input,
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            current_api_key="",
            explicit_provider="openai-codex",
            user_providers=user_providers,
            custom_providers=[],
        )
    return result, validate


def test_settings_only_block_keeps_builtin_validation_routing():
    """Settings-only block: validation must run as ``openai-codex``, not
    ``custom:openai-codex``."""
    result, validate = _run("gpt-5.6-luna", {"openai-codex": dict(_SETTINGS_ONLY)})

    assert result.success is True
    assert validate.call_count == 1
    assert validate.call_args[0][1] == "openai-codex"


def test_endpoint_block_still_routes_custom_validation():
    """A block declaring its own endpoint stays on the custom branch (pins the
    legit case the guard must not break)."""
    result, validate = _run("gpt-5.6-luna", {"openai-codex": dict(_WITH_ENDPOINT)})

    assert result.success is True
    assert validate.call_count == 1
    assert validate.call_args[0][1] == "custom:openai-codex"


def test_settings_only_block_codex_switch_succeeds_without_listing():
    """End-to-end shape of the issue: listing-less Codex backend (probe yields
    no ``/models``) + settings-only block must soft-accept via the built-in
    static catalog path (#16172 / #19729), not hard-reject via custom."""
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider",
               side_effect=lambda model, provider: model), \
         patch("hermes_cli.models.provider_model_ids",
               return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"]), \
         patch("hermes_cli.models.probe_api_models",
               return_value={"models": None,
                             "probed_url": "https://chatgpt.com/backend-api/codex/models",
                             "suggested_base_url": None}), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value=dict(_RUNTIME_CODEX)):
        result = switch_model(
            raw_input="gpt-5.6-luna",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            current_api_key="",
            explicit_provider="openai-codex",
            user_providers={"openai-codex": dict(_SETTINGS_ONLY)},
            custom_providers=[],
        )

    assert result.success is True
    assert result.new_model == "gpt-5.6-luna"
    assert result.target_provider == "openai-codex"
