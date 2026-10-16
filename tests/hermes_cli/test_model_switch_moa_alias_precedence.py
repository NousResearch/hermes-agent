"""Regression tests for #134162: a user-configured alias must win over a
same-named MoA preset on the no-``--provider`` switch path.

Repro: the built-in MoA config ships one enabled preset named ``default``
(``DEFAULT_MOA_PRESET_NAME``), and ``_route_from_model_input`` checked MoA
preset names BEFORE the profile's aliases. An alias named ``default`` — the
most natural name for "my everyday model" — was therefore unreachable:
``/model default`` silently switched the session to MoA, and every turn then
ran the reference fan-out plus the pay-per-token aggregator with the whole
conversation as input, with nothing telling the user their alias was not used.

The fix: a bare name that matches an enabled MoA preset AND is explicitly
configured as a user alias (``model_aliases:`` / ``model.aliases:``) routes to
the alias. MoA stays reachable via ``--provider moa`` and the /moa picker, as
#55187 already provides.

Hermetic: config and the model-resolution chain are fully mocked (no network),
mirroring ``tests/hermes_cli/test_model_switch_configured_provider_routing.py``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run_switch(
    *, raw_input, config, alias_resolution, current_provider="openai-codex"
):
    """Drive ``switch_model`` with config and the resolution chain mocked out.

    ``alias_resolution`` maps a typed name to the tuple ``resolve_alias`` would
    return for it (missing key = no alias). The precedence decision under test
    reads the mocked ``load_config()`` directly via ``_load_user_direct_aliases``.
    """

    def _resolve_alias(raw, provider, user_providers=None, custom_providers=None):
        return alias_resolution.get(str(raw).strip().lower())

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.model_switch.DIRECT_ALIASES", {}),
        patch("hermes_cli.model_switch.resolve_alias", side_effect=_resolve_alias),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.model_switch.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
        patch(
            "hermes_cli.models_validate.validate_requested_model",
            return_value=_ACCEPTED,
        ),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "***",
                "base_url": "http://resolved/v1",
                "api_mode": "",
            },
        ),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="old-model",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        )


def test_user_alias_named_default_wins_over_stock_moa_preset():
    """/model default with a `model_aliases` entry named `default` routes to the alias, not MoA."""
    config = {
        "model_aliases": {
            "default": {"provider": "openai-codex", "model": "gpt-6.1-sol"}
        }
    }
    result = _run_switch(
        raw_input="default",
        config=config,
        alias_resolution={"default": ("openai-codex", "gpt-6.1-sol", "default")},
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-6.1-sol"


def test_bare_preset_name_without_conflicting_alias_still_routes_to_moa():
    """No same-named alias: the stock `default` preset keeps matching a bare /model default."""
    result = _run_switch(
        raw_input="default",
        config={
            "model_aliases": {
                "standard": {"provider": "openai-codex", "model": "gpt-6.1-sol"}
            }
        },
        alias_resolution={},
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "moa"
    assert result.new_model == "default"


def test_simple_model_dot_aliases_entry_also_wins():
    """`model.aliases: {default: provider/model}` is explicit user config too."""
    config = {
        "model": {
            "provider": "openai-codex",
            "aliases": {"default": "openai-codex/gpt-6.1-sol"},
        }
    }
    result = _run_switch(
        raw_input="default",
        config=config,
        alias_resolution={"default": ("openai-codex", "gpt-6.1-sol", "default")},
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-6.1-sol"


def test_custom_enabled_preset_name_collision_also_prefers_user_alias():
    """The precedence is not `default`-specific: any enabled preset name colliding with a
    user alias routes to the alias."""
    config = {
        "moa": {"presets": {"fast": {"reference_models": ["openai/gpt-5.4"]}}},
        "model_aliases": {"fast": {"provider": "openai-codex", "model": "gpt-6.1-sol"}},
    }
    result = _run_switch(
        raw_input="fast",
        config=config,
        alias_resolution={"fast": ("openai-codex", "gpt-6.1-sol", "fast")},
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-6.1-sol"
