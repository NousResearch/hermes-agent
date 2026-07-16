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
    # Unparseable/blank temperatures degrade to None = "don't send the
    # parameter; provider default applies" (matching single-model behavior),
    # not to a hardcoded sampling value.
    assert preset["reference_temperature"] is None
    assert preset["aggregator_temperature"] is None


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


def test_normalize_moa_config_preserves_slot_reasoning_effort():
    cfg = normalize_moa_config(
        {
            "presets": {
                "p": {
                    "reference_models": [
                        {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "LOW"},
                        {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": False},
                        {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "nonsense"},
                        {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "ultra"},
                    ],
                    "aggregator": {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "xhigh"},
                }
            }
        }
    )

    preset = cfg["presets"]["p"]
    assert preset["reference_models"][0]["reasoning_effort"] == "low"
    assert preset["reference_models"][1]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in preset["reference_models"][2]
    assert preset["reference_models"][3]["reasoning_effort"] == "ultra"
    assert preset["aggregator"]["reasoning_effort"] == "xhigh"


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


def test_exact_preset_matching_skips_disabled_presets():
    """A disabled preset must not match the implicit bare-name switch path.

    Regression for #55187: with ``enabled: false`` presets, a plain model
    switch whose name collides with a preset key (e.g. ``default``) silently
    pivoted the session onto the MoA virtual provider. The per-preset
    ``enabled`` opt-out must gate this implicit match.
    """
    config = {
        "presets": {
            "default": {"enabled": False},
            "klo": {"enabled": False},
        },
    }
    assert exact_moa_preset_name(config, "default") is None
    assert exact_moa_preset_name(config, "klo") is None


def test_exact_preset_matching_allows_enabled_presets():
    """An explicitly enabled preset still matches the bare-name switch path."""
    config = {
        "presets": {
            "fast": {"enabled": True},
            "slow": {"enabled": False},
        },
    }
    assert exact_moa_preset_name(config, "fast") == "fast"
    assert exact_moa_preset_name(config, "slow") is None
    # Default (no explicit enabled key) is enabled and still matches.
    assert exact_moa_preset_name({"presets": {"x": {}}}, "x") == "x"


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


def _preset(**extra):
    base = {
        "reference_models": [{"provider": "openrouter", "model": "anthropic/claude-opus-4.8"}],
        "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    }
    base.update(extra)
    return {"default_preset": "p", "presets": {"p": base}}


def test_reference_max_tokens_defaults_to_none_uncapped():
    """Unset reference_max_tokens resolves to None (no cap) so existing presets
    keep their prior uncapped advisor behavior — no silent regression."""
    p = resolve_moa_preset(_preset(), "p")
    assert p["reference_max_tokens"] is None


def test_reference_max_tokens_positive_value_preserved():
    """A positive cap flows through resolve_moa_preset to the runtime path."""
    p = resolve_moa_preset(_preset(reference_max_tokens=600), "p")
    assert p["reference_max_tokens"] == 600


def test_reference_max_tokens_invalid_falls_back_to_none():
    """Non-positive / non-numeric caps degrade to None (uncapped) rather than
    clamping advisors to a nonsense value or crashing."""
    for bad in (0, -5, "abc", "", None):
        p = resolve_moa_preset(_preset(reference_max_tokens=bad), "p")
        assert p["reference_max_tokens"] is None, bad


def test_reference_max_tokens_string_number_coerced():
    """A hand-edited config.yaml string like '600' coerces to int."""
    p = resolve_moa_preset(_preset(reference_max_tokens="600"), "p")
    assert p["reference_max_tokens"] == 600


def test_reference_max_tokens_in_flattened_view():
    """The flattened compatibility view (dashboard/desktop callers) exposes the
    active preset's reference_max_tokens."""
    cfg = normalize_moa_config(_preset(reference_max_tokens=750))
    assert cfg["reference_max_tokens"] == 750


# ── validate_moa_payload (write-boundary validation, #64156) ─────────────────
#
# normalize_moa_config is deliberately tolerant at READ time (hand-edited
# configs degrade to defaults). validate_moa_payload is the strict WRITE-time
# counterpart: it must flag exactly the payloads normalize would silently
# repair, so API save paths reject them instead of corrupting user config.


def _valid_preset_payload():
    return {
        "reference_models": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
        "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    }


def test_validate_moa_payload_accepts_complete_presets():
    from hermes_cli.moa_config import validate_moa_payload

    assert validate_moa_payload({"presets": {"default": _valid_preset_payload()}}) == []


def test_validate_moa_payload_accepts_legacy_flat_payload():
    from hermes_cli.moa_config import validate_moa_payload

    assert validate_moa_payload(_valid_preset_payload()) == []


def test_validate_moa_payload_flags_half_filled_reference_slot():
    """The #64156 shape: provider picked, model still empty (mid-edit autosave)."""
    from hermes_cli.moa_config import validate_moa_payload

    preset = _valid_preset_payload()
    preset["reference_models"].append({"provider": "kilo", "model": ""})
    problems = validate_moa_payload({"presets": {"default": preset}})

    assert problems
    assert any("reference 2" in p and "model is required" in p for p in problems)


def test_validate_moa_payload_flags_half_filled_aggregator():
    from hermes_cli.moa_config import validate_moa_payload

    preset = _valid_preset_payload()
    preset["aggregator"] = {"provider": "openrouter", "model": ""}
    problems = validate_moa_payload({"presets": {"default": preset}})

    assert any("aggregator" in p and "model is required" in p for p in problems)


def test_validate_moa_payload_flags_empty_references():
    from hermes_cli.moa_config import validate_moa_payload

    preset = _valid_preset_payload()
    preset["reference_models"] = []
    problems = validate_moa_payload({"presets": {"default": preset}})

    assert any("at least one complete reference model" in p for p in problems)


def test_validate_moa_payload_flags_recursive_moa_slot():
    from hermes_cli.moa_config import validate_moa_payload

    preset = _valid_preset_payload()
    preset["aggregator"] = {"provider": "MoA", "model": "default"}
    problems = validate_moa_payload({"presets": {"default": preset}})

    assert any("recursive MoA" in p for p in problems)


def test_validate_moa_payload_names_the_broken_preset():
    """Multi-preset payloads must say WHICH preset is broken."""
    from hermes_cli.moa_config import validate_moa_payload

    problems = validate_moa_payload(
        {
            "presets": {
                "good": _valid_preset_payload(),
                "broken": {
                    "reference_models": [{"provider": "", "model": ""}],
                    "aggregator": {"provider": "a", "model": "b"},
                },
            }
        }
    )

    assert problems
    assert all("'broken'" in p for p in problems)
    assert not any("'good'" in p for p in problems)


def test_validate_moa_payload_agrees_with_clean_slot():
    """Contract: a payload validate accepts must survive normalize UNCHANGED in
    its slots — validate and _clean_slot can never disagree (else a payload
    could pass validation and still be swapped for defaults)."""
    from hermes_cli.moa_config import validate_moa_payload

    payload = {"presets": {"p": _valid_preset_payload()}}
    assert validate_moa_payload(payload) == []

    cfg = normalize_moa_config(payload)
    assert cfg["presets"]["p"]["reference_models"] == payload["presets"]["p"]["reference_models"]
    assert cfg["presets"]["p"]["aggregator"] == payload["presets"]["p"]["aggregator"]


def test_validate_moa_payload_rejects_non_dict():
    from hermes_cli.moa_config import validate_moa_payload

    assert validate_moa_payload(None)
    assert validate_moa_payload([1, 2])
    assert validate_moa_payload({"presets": {"p": "not-a-dict"}})
