from hermes_cli.moa_config import (
    DEFAULT_MOA_AGGREGATOR,
    DEFAULT_MOA_PRESET_NAME,
    DEFAULT_MOA_REFERENCE_MODELS,
    build_moa_turn_prompt,
    decode_moa_turn,
    exact_moa_preset_name,
    normalize_moa_config,
    resolve_moa_preset,
    set_active_moa_preset,
)


def test_normalize_moa_config_uses_default_named_preset():
    cfg = normalize_moa_config({})

    assert cfg["default_preset"] == DEFAULT_MOA_PRESET_NAME
    assert list(cfg["presets"]) == [DEFAULT_MOA_PRESET_NAME]
    assert cfg["reference_models"] == DEFAULT_MOA_REFERENCE_MODELS
    assert cfg["aggregator"] == DEFAULT_MOA_AGGREGATOR


def test_normalize_moa_config_preserves_named_presets():
    cfg = normalize_moa_config(
        {
            "default_preset": "coding",
            "presets": {
                "coding": {
                    "reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                },
                "review": {
                    "reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                },
            },
        }
    )

    assert cfg["default_preset"] == "coding"
    assert set(cfg["presets"]) == {"coding", "review"}
    assert cfg["reference_models"] == [{"provider": "openai-codex", "model": "gpt-5.5"}]


def test_legacy_flat_config_becomes_default_preset():
    cfg = normalize_moa_config(
        {
            "reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}],
            "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        }
    )

    assert cfg["presets"][DEFAULT_MOA_PRESET_NAME]["reference_models"] == [
        {"provider": "openai-codex", "model": "gpt-5.5"}
    ]


def test_normalize_moa_config_tolerates_non_numeric_values():
    """Non-numeric strings in hand-edited config.yaml must degrade to defaults
    instead of crashing normalize_moa_config with ValueError."""
    cfg = normalize_moa_config(
        {
            "presets": {
                "broken": {
                    "max_tokens": "notanumber",
                    "reference_temperature": "hot",
                    "aggregator_temperature": "",
                }
            }
        }
    )

    preset = cfg["presets"]["broken"]
    assert preset["max_tokens"] == 4096
    assert preset["reference_temperature"] == 0.6
    assert preset["aggregator_temperature"] == 0.4


def test_normalize_moa_config_tolerates_non_list_reference_models():
    """A hand-edited scalar reference_models must degrade to defaults instead of
    crashing normalize_moa_config with TypeError (symmetric with the non-numeric
    scalar-field tolerance)."""
    cfg = normalize_moa_config(
        {"presets": {"broken": {"reference_models": 2}}}
    )
    assert cfg["presets"]["broken"]["reference_models"] == DEFAULT_MOA_REFERENCE_MODELS


def test_normalize_moa_config_wraps_bare_dict_reference_models():
    """A single reference slot written without the list wrapper is rescued."""
    cfg = normalize_moa_config(
        {"presets": {"p": {"reference_models": {"provider": "openai", "model": "gpt-4o"}}}}
    )
    assert cfg["presets"]["p"]["reference_models"] == [{"provider": "openai", "model": "gpt-4o"}]


def test_normalize_moa_config_coerces_numeric_strings():
    """Valid numeric strings (e.g. from YAML round-trip) must coerce correctly."""
    cfg = normalize_moa_config({"max_tokens": "8192", "reference_temperature": "0.9"})

    preset = cfg["presets"][DEFAULT_MOA_PRESET_NAME]
    assert preset["max_tokens"] == 8192
    assert preset["reference_temperature"] == 0.9


def test_normalize_moa_config_coerces_float_max_tokens():
    """max_tokens: 4096.0 (float from YAML) must coerce to int."""
    cfg = normalize_moa_config({"max_tokens": 4096.0})
    assert cfg["presets"][DEFAULT_MOA_PRESET_NAME]["max_tokens"] == 4096

    cfg2 = normalize_moa_config({"max_tokens": "4096.5"})
    assert cfg2["presets"][DEFAULT_MOA_PRESET_NAME]["max_tokens"] == 4096


def test_exact_preset_matching_is_not_fuzzy():
    config = {"presets": {"coding": {}, "review": {}}}

    assert exact_moa_preset_name(config, "coding") == "coding"
    assert exact_moa_preset_name(config, "cod") is None
    assert exact_moa_preset_name(config, "coding please fix this") is None


def test_exact_preset_skips_disabled_presets():
    """Disabled presets must not be matched by name.

    Prevents implicit MoA switches when the user has opted out of a
    preset via ``enabled: false`` (issue #55187).
    """
    config = {
        "presets": {
            "default": {"enabled": False},
            "review": {"enabled": True},
        },
    }

    assert exact_moa_preset_name(config, "default") is None
    assert exact_moa_preset_name(config, "review") == "review"
    # Enabled defaults to True when key is absent.
    config_missing = {"presets": {"coding": {}}}
    assert exact_moa_preset_name(config_missing, "coding") == "coding"


def test_active_preset_toggle_validation():
    config = {"default_preset": "coding", "presets": {"coding": {}, "review": {}}}

    active = set_active_moa_preset(config, "review")
    assert active["active_preset"] == "review"

    inactive = set_active_moa_preset(active, "")
    assert inactive["active_preset"] == ""


def test_resolve_moa_preset_returns_requested_model_set():
    cfg = normalize_moa_config(
        {
            "presets": {
                "coding": {"reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}]},
                "review": {"reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]},
            }
        }
    )

    assert resolve_moa_preset(cfg, "review")["reference_models"] == [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
    ]


def test_build_moa_turn_prompt_encodes_one_shot_default_preset():
    prompt = build_moa_turn_prompt("write a file then inspect it")

    decoded_prompt, cfg = decode_moa_turn(prompt)
    assert decoded_prompt == "write a file then inspect it"
    assert cfg is not None
    assert cfg["reference_models"] == DEFAULT_MOA_REFERENCE_MODELS


def test_moa_provider_rejected_as_reference_slot():
    """A reference slot pointing at the moa virtual provider is dropped, so a
    preset cannot recursively reference another MoA run."""
    cfg = normalize_moa_config(
        {
            "presets": {
                "p": {
                    "reference_models": [
                        {"provider": "moa", "model": "default"},
                        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                    ],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                }
            }
        }
    )

    refs = cfg["presets"]["p"]["reference_models"]
    assert {"provider": "moa", "model": "default"} not in refs
    assert refs == [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]


def test_moa_provider_rejected_as_aggregator_slot():
    """An aggregator slot pointing at the moa virtual provider is dropped and
    falls back to the default aggregator, never a recursive MoA aggregator."""
    cfg = normalize_moa_config(
        {
            "presets": {
                "p": {
                    "reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
                    "aggregator": {"provider": "moa", "model": "default"},
                }
            }
        }
    )

    agg = cfg["presets"]["p"]["aggregator"]
    assert agg["provider"] != "moa"
    assert agg == DEFAULT_MOA_AGGREGATOR


def test_moa_provider_rejected_case_insensitive():
    """Case variants like ``MoA`` are also blocked."""
    cfg = normalize_moa_config(
        {"presets": {"p": {"aggregator": {"provider": "MoA", "model": "default"}}}}
    )

    assert cfg["presets"]["p"]["aggregator"]["provider"] != "moa"
    assert cfg["presets"]["p"]["aggregator"] == DEFAULT_MOA_AGGREGATOR
