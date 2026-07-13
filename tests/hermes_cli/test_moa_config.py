from hermes_cli.moa_config import (
    DEFAULT_MOA_AGGREGATOR,
    DEFAULT_MOA_PRESET_NAME,
    DEFAULT_MOA_REFERENCE_MODELS,
    _slot_runtime_available,
    build_moa_turn_prompt,
    classify_moa_auto_route,
    decode_moa_turn,
    exact_moa_preset_name,
    moa_config_revision,
    normalize_moa_config,
    resolve_available_moa_preset,
    resolve_moa_preset,
    resolve_moa_preset_for_messages,
    set_active_moa_preset,
    validate_moa_config,
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


def test_normalize_moa_config_preserves_explicit_empty_reference_models():
    """Явный пустой список создаёт одиночный маршрут без скрытого fan-out."""
    cfg = normalize_moa_config(
        {
            "presets": {
                "fast": {
                    "reference_models": [],
                    "aggregator": {"provider": "openai-codex", "model": "gpt-5.6-luna"},
                }
            }
        }
    )
    assert cfg["presets"]["fast"]["reference_models"] == []


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


def test_classify_moa_auto_route_is_conservative_and_task_aware():
    cases = {
        "Переведи эту фразу на английский": "fast",
        "Проверь актуальные официальные источники и дай ссылки": "research",
        "Исправь баг в parser.py и запусти тесты": "code_heavy",
        "Используй режим max, нужна максимальная глубина": "max",
        "Подготовь план запуска продукта на новый рынок": "balanced",
    }
    for prompt, expected in cases.items():
        assert classify_moa_auto_route([{"role": "user", "content": prompt}]) == expected


def test_classify_moa_auto_route_continuation_keeps_previous_task_kind():
    messages = [
        {"role": "user", "content": "Исправь архитектурную ошибку в репозитории"},
        {"role": "assistant", "content": "Начинаю."},
        {"role": "user", "content": "Продолжай"},
    ]
    assert classify_moa_auto_route(messages) == "code_heavy"


def test_resolve_moa_preset_for_messages_uses_configured_auto_routes():
    cfg = {
        "default_preset": "auto",
        "presets": {
            "auto": {
                "reference_models": [{"provider": "xai-oauth", "model": "grok-4.5"}],
                "auto_routes": {
                    "fast": "quick",
                    "balanced": "normal",
                    "research": "sources",
                    "code_heavy": "coding",
                    "max": "deep",
                },
            },
            "quick": {"reference_models": []},
            "normal": {"reference_models": []},
            "sources": {"reference_models": []},
            "coding": {"reference_models": []},
            "deep": {"reference_models": []},
        },
    }
    name, preset = resolve_moa_preset_for_messages(
        cfg,
        "auto",
        [{"role": "user", "content": "Реализуй поддержку нового API и тесты"}],
    )
    assert name == "coding"
    assert preset["reference_models"] == []


def test_resolve_moa_preset_for_messages_leaves_regular_preset_unchanged():
    cfg = {
        "presets": {
            "balanced": {
                "reference_models": [{"provider": "xai-oauth", "model": "grok-4.5"}],
            }
        }
    }
    name, preset = resolve_moa_preset_for_messages(
        cfg,
        "balanced",
        [{"role": "user", "content": "Проверь свежие новости"}],
    )
    assert name == "balanced"
    assert preset["reference_models"] == [{"provider": "xai-oauth", "model": "grok-4.5"}]


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


def test_context_length_is_normalized_and_exposed_in_flattened_view():
    cfg = normalize_moa_config(_preset(context_length="450000"))
    assert cfg["presets"]["p"]["context_length"] == 450000
    assert cfg["context_length"] == 450000


def test_invalid_context_length_falls_back_to_auto_detection():
    for bad in (0, -1, "invalid", "", None):
        preset = resolve_moa_preset(_preset(context_length=bad), "p")
        assert preset["context_length"] is None, bad


def test_excluded_provider_is_removed_without_restoring_default_fanout():
    cfg = normalize_moa_config({
        "excluded_providers": ["antigravity_cli"],
        "presets": {
            "max": {
                "reference_models": [
                    {"provider": "antigravity_cli", "model": "legacy"},
                    {"provider": "openrouter", "model": "z-ai/glm-5.2"},
                ],
                "aggregator": {"provider": "openai-codex", "model": "gpt-5.6-sol"},
            },
        },
    })

    assert cfg["excluded_providers"] == ["antigravity_cli"]
    assert cfg["presets"]["max"]["reference_models"] == [
        {"provider": "openrouter", "model": "z-ai/glm-5.2"},
    ]


def test_runtime_resolution_uses_available_fallback_and_deduplicates_reference():
    preset = {
        "reference_models": [
            {"provider": "xai-oauth", "model": "grok-4.5"},
            {"provider": "openrouter", "model": "z-ai/glm-5.2"},
        ],
        "aggregator": {"provider": "openai-codex", "model": "gpt-5.6-sol"},
        "fallback_aggregators": [
            {"provider": "xai-oauth", "model": "grok-4.5"},
        ],
    }

    resolved, status = resolve_available_moa_preset(
        preset,
        availability_check=lambda slot: slot["provider"] == "xai-oauth",
    )

    assert resolved["aggregator"] == {"provider": "xai-oauth", "model": "grok-4.5"}
    assert resolved["reference_models"] == []
    assert status["degraded"] is True
    assert status["aggregator_available"] is True
    assert status["aggregator"] == "xai-oauth:grok-4.5"
    assert status["configured_aggregator"] == "openai-codex:gpt-5.6-sol"
    assert status["fallback_used"] is True
    assert all("xai-oauth:grok-4.5" not in item for item in status["unavailable"])


def test_validator_rejects_excluded_aggregator_and_broken_auto_route():
    issues = validate_moa_config({
        "excluded_providers": ["antigravity_cli"],
        "presets": {
            "auto": {
                "reference_models": [],
                "aggregator": {"provider": "antigravity_cli", "model": "legacy"},
                "auto_routes": {"research": "missing"},
            },
        },
    })

    error_codes = {
        item["code"] for item in issues if item["severity"] == "error"
    }
    assert error_codes == {"excluded_aggregator_provider", "invalid_auto_route"}


def test_runtime_availability_rejects_endpoint_without_credentials(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
        },
    )

    assert _slot_runtime_available({
        "provider": "openrouter",
        "model": "z-ai/glm-5.2",
    }) is False


def test_budget_and_circuit_fields_are_normalized():
    cfg = normalize_moa_config({
        "presets": {
            "bounded": {
                "reference_models": [],
                "aggregator": {"provider": "openai-codex", "model": "gpt-5.6-terra"},
                "max_advisors": "2",
                "max_reference_cost_usd": "0.25",
                "max_fanout_latency_seconds": "15",
                "circuit_breaker_seconds": "45",
                "quota_cooldown_seconds": "600",
            },
        },
    })
    preset = cfg["presets"]["bounded"]

    assert preset["max_advisors"] == 2
    assert preset["max_reference_cost_usd"] == 0.25
    assert preset["max_fanout_latency_seconds"] == 15.0
    assert preset["circuit_breaker_seconds"] == 45.0
    assert preset["quota_cooldown_seconds"] == 600.0


def test_revision_is_stable_and_changes_with_config():
    base = {
        "presets": {
            "p": {
                "reference_models": [],
                "aggregator": {"provider": "openai-codex", "model": "gpt-5.6-terra"},
            },
        },
    }
    same = normalize_moa_config(base)
    changed = normalize_moa_config(base)
    changed["presets"]["p"]["max_advisors"] = 1

    assert moa_config_revision(base) == moa_config_revision(same)
    assert moa_config_revision(base) != moa_config_revision(changed)
