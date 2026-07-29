import pytest

from agent.errors import MoAPresetNotFoundError
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


def test_moa_slot_picker_excludes_unconfigured_providers(monkeypatch):
    from hermes_cli import moa_cmd

    captured = {}
    monkeypatch.setattr(moa_cmd, "load_picker_context", lambda: object())

    def fake_build(_context, **kwargs):
        captured.update(kwargs)
        return {
            "providers": [
                {"slug": "moa", "models": ["default"]},
                {"slug": "opencode-go", "models": ["deepseek-v4-pro"]},
            ]
        }

    monkeypatch.setattr(moa_cmd, "build_models_payload", fake_build)

    assert [row["slug"] for row in moa_cmd._model_options()] == ["opencode-go"]
    assert captured["include_unconfigured"] is False


def _enabled_refs(refs):
    return [{**slot, "enabled": True} for slot in refs]


def test_normalize_moa_config_uses_default_named_preset():
    cfg = normalize_moa_config({})

    assert cfg["default_preset"] == DEFAULT_MOA_PRESET_NAME
    assert list(cfg["presets"]) == [DEFAULT_MOA_PRESET_NAME]
    assert cfg["reference_models"] == _enabled_refs(DEFAULT_MOA_REFERENCE_MODELS)
    assert cfg["aggregator"] == DEFAULT_MOA_AGGREGATOR


def test_normalize_moa_config_round_trips_reasoning_effort_and_enabled():
    """Regression: a client that GETs the config and PUTs it straight back must
    not strip per-slot keys. reasoning_effort AND enabled have to survive a
    normalize → normalize round trip together (a save path that re-normalizes
    the previously normalized payload is the exact client round-trip shape)."""
    cfg = normalize_moa_config(
        {
            "presets": {
                "p": {
                    "reference_models": [
                        {"provider": "openai-codex", "model": "gpt-5.5", "reasoning_effort": "high", "enabled": False},
                        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "enabled": True},
                    ],
                    "aggregator": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-opus-4.8",
                        "reasoning_effort": "xhigh",
                    },
                }
            }
        }
    )

    round_tripped = normalize_moa_config(cfg)

    refs = round_tripped["presets"]["p"]["reference_models"]
    assert refs[0] == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "enabled": False,
    }
    assert refs[1] == {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "enabled": True}
    assert round_tripped["presets"]["p"]["aggregator"]["reasoning_effort"] == "xhigh"


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
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "enabled": True}
    ]


def test_resolve_missing_moa_preset_has_actionable_error():
    cfg = {
        "default_preset": "日常对话-高峰",
        "presets": {"日常对话-高峰": {}, "日常对话-非高峰": {}},
    }

    with pytest.raises(MoAPresetNotFoundError) as exc_info:
        resolve_moa_preset(cfg, "日常对话-高峰期")

    message = str(exc_info.value)
    assert "日常对话-高峰期" in message
    assert "日常对话-高峰" in message
    assert "日常对话-非高峰" in message
    assert "hermes moa list" in message


def test_missing_moa_preset_is_non_retryable():
    from agent.error_classifier import FailoverReason, classify_api_error

    result = classify_api_error(
        MoAPresetNotFoundError("MoA preset 'old' was not found"),
        provider="moa",
        model="old",
    )

    assert result.reason == FailoverReason.model_not_found
    assert result.retryable is False
    assert result.should_fallback is False


def test_build_moa_turn_prompt_encodes_one_shot_default_preset():
    prompt = build_moa_turn_prompt("write a file then inspect it")

    decoded_prompt, cfg = decode_moa_turn(prompt)
    assert decoded_prompt == "write a file then inspect it"
    assert cfg is not None
    assert cfg["reference_models"] == _enabled_refs(DEFAULT_MOA_REFERENCE_MODELS)


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
    assert refs == [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "enabled": True}]


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


def test_validate_moa_payload_agrees_with_clean_slot():
    """Contract: a payload validate accepts must survive normalize UNCHANGED in
    its slots — validate and _clean_slot can never disagree (else a payload
    could pass validation and still be swapped for defaults)."""
    from hermes_cli.moa_config import validate_moa_payload

    payload = {"presets": {"p": _valid_preset_payload()}}
    assert validate_moa_payload(payload) == []

    cfg = normalize_moa_config(payload)
    # Slots survive with only the canonical enabled=True default added — no
    # provider/model swap, no defaults substitution.
    assert cfg["presets"]["p"]["reference_models"] == _enabled_refs(payload["presets"]["p"]["reference_models"])
    assert cfg["presets"]["p"]["aggregator"] == payload["presets"]["p"]["aggregator"]


# ── Per-slot max_tokens ────────────────────────────────────────────────────


def test_slot_max_tokens_invalid_dropped():
    """Non-positive / non-numeric slot max_tokens is dropped (slot kept)."""
    for bad in (0, -5, "abc", "", None):
        cfg = normalize_moa_config(
            {
                "presets": {
                    "p": {
                        "reference_models": [
                            {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro", "max_tokens": bad},
                        ],
                    }
                }
            }
        )
        ref = cfg["presets"]["p"]["reference_models"][0]
        assert "max_tokens" not in ref, bad
        assert ref["provider"] == "openrouter"


def test_slot_max_tokens_absent_by_default():
    """Slots without max_tokens don't get the field — backward compat."""
    cfg = normalize_moa_config(
        {
            "presets": {
                "p": {
                    "reference_models": [
                        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                    ],
                }
            }
        }
    )
    ref = cfg["presets"]["p"]["reference_models"][0]
    assert "max_tokens" not in ref


# --- fanout cadence normalization (every_n) ---


def test_fanout_defaults_to_user_turn():
    # Default is the cheapest cadence (#67199): advisors once per user turn.
    cfg = normalize_moa_config({})
    assert cfg["fanout"] == "user_turn"


def test_fanout_every_n_string_form_normalized():
    cfg = normalize_moa_config({"fanout": "every_n:3"})
    assert cfg["fanout"] == "every_n:3"
    assert cfg["presets"][DEFAULT_MOA_PRESET_NAME]["fanout"] == "every_n:3"


def test_fanout_every_n_degenerate_n_falls_back():
    # n=1 means "every iteration" — that semantically IS per_iteration;
    # n=0 / negative / garbage is unparseable and falls to the default
    # cadence (user_turn, the cheapest — #67199).
    assert normalize_moa_config({"fanout": "every_n:1"})["fanout"] == "per_iteration"
    assert normalize_moa_config({"fanout": "every_n:0"})["fanout"] == "user_turn"
    assert normalize_moa_config({"fanout": "every_n:-2"})["fanout"] == "user_turn"
    assert normalize_moa_config({"fanout": "every_n:x"})["fanout"] == "user_turn"
    assert normalize_moa_config({"fanout": "every_n"})["fanout"] == "user_turn"
    assert normalize_moa_config({"fanout": {"mode": "every_n"}})["fanout"] == "user_turn"


# --- privacy_filter normalization ---


def test_privacy_filter_defaults_off():
    cfg = normalize_moa_config({})
    assert cfg["privacy_filter"] == ""


def test_privacy_filter_round_trips_through_normalize():
    once = normalize_moa_config({"privacy_filter": "display"})
    assert once["privacy_filter"] == "display"
    assert normalize_moa_config(once)["privacy_filter"] == "display"
    full = normalize_moa_config({"privacy_filter": "full"})
    assert normalize_moa_config(full)["privacy_filter"] == "full"


def test_reference_timeout_is_uncapped_and_unknown_policy_is_loud():
    preset = resolve_moa_preset(
        _preset(reference_timeout=9999, degraded_reference_policy="wat"), "p"
    )

    # Explicit per-preset values are honored as-is — long-thinking advisor
    # models legitimately run beyond any fixed cap.
    assert preset["reference_timeout"] == 9999.0
    assert preset["degraded_reference_policy"] == "loud"
