from hermes_cli.moa_config import (
    DEFAULT_MOA_AGGREGATOR,
    DEFAULT_MOA_PRESET_NAME,
    DEFAULT_MOA_REFERENCE_CONTEXT,
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
    assert cfg["reference_context"] == DEFAULT_MOA_REFERENCE_CONTEXT


def test_normalize_moa_config_preserves_reference_context_per_preset():
    cfg = normalize_moa_config(
        {
            "default_preset": "persona",
            "presets": {
                "persona": {
                    "reference_models": [{"provider": "openai-codex", "model": "gpt-5.5"}],
                    "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                    "reference_context": {
                        "system": "full",
                        "files": {
                            "enabled": True,
                            "names": ["SOUL.md", "AGENTS.md", "SOUL.md", "../outside"],
                        },
                    },
                }
            },
        }
    )

    assert cfg["reference_context"] == {
        "system": "full",
        "files": {"enabled": True, "names": ["SOUL.md", "AGENTS.md"]},
    }
    assert cfg["presets"]["persona"]["reference_context"] == cfg["reference_context"]




def test_reference_context_dict_can_disable_preselected_files():
    cfg = normalize_moa_config(
        {
            "presets": {
                "review": {
                    "reference_context": {
                        "system": "none",
                        "files": {"enabled": False, "names": ["SOUL.md"]},
                    },
                }
            }
        }
    )

    assert cfg["presets"]["review"]["reference_context"] == {
        "system": "none",
        "files": {"enabled": False, "names": ["SOUL.md"]},
    }


def test_hidden_moa_turn_does_not_trust_reference_context_from_payload():
    marker = build_moa_turn_prompt(
        "review this",
        {
            "default_preset": "rich",
            "presets": {
                "rich": {
                    "reference_context": {
                        "system": "full",
                        "files": {"enabled": True, "names": ["SOUL.md", "AGENTS.md"]},
                    }
                }
            },
        },
        preset="rich",
    )

    decoded_prompt, cfg = decode_moa_turn(marker)

    assert decoded_prompt == "review this"
    assert cfg is not None
    assert cfg["reference_context"] == DEFAULT_MOA_REFERENCE_CONTEXT


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


def test_exact_preset_matching_is_not_fuzzy():
    config = {"presets": {"coding": {}, "review": {}}}

    assert exact_moa_preset_name(config, "coding") == "coding"
    assert exact_moa_preset_name(config, "cod") is None
    assert exact_moa_preset_name(config, "coding please fix this") is None


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
