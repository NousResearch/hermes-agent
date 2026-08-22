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








def _preset(**extra):
    base = {
        "reference_models": [{"provider": "openrouter", "model": "anthropic/claude-opus-4.8"}],
        "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    }
    base.update(extra)
    return {"default_preset": "p", "presets": {"p": base}}






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






# --- fanout cadence normalization (every_n) ---








# --- privacy_filter normalization ---








# --- reference_view / reference_detail_tools normalization ---


def test_reference_view_defaults_to_digest():
    cfg = normalize_moa_config({})
    preset = cfg["presets"][DEFAULT_MOA_PRESET_NAME]
    assert preset["reference_view"] == "digest"
    assert preset["reference_detail_tools"] is None
    # Flattened compat view carries the fields too.
    assert cfg["reference_view"] == "digest"
    assert cfg["reference_detail_tools"] is None


def test_reference_view_coercion():
    from hermes_cli.moa_config import _coerce_reference_view

    assert _coerce_reference_view(None) == "digest"
    assert _coerce_reference_view("") == "digest"
    assert _coerce_reference_view("Transcript") == "transcript"
    assert _coerce_reference_view("digest") == "digest"
    # Unknown values degrade to the default, matching other scalar fields.
    assert _coerce_reference_view("verbose") == "digest"
    assert _coerce_reference_view(123) == "digest"


def test_reference_detail_tools_coercion():
    from hermes_cli.moa_config import _coerce_reference_detail_tools

    assert _coerce_reference_detail_tools(None) is None
    assert _coerce_reference_detail_tools("") is None
    assert _coerce_reference_detail_tools([]) is None
    assert _coerce_reference_detail_tools(["Terminal", " read_file "]) == ["terminal", "read_file"]
    # Comma-separated string form for hand-edited configs.
    assert _coerce_reference_detail_tools("terminal, read_file") == ["terminal", "read_file"]
    # Bad types degrade to None (built-in substance set).
    assert _coerce_reference_detail_tools(42) is None


def test_reference_view_survives_normalize():
    cfg = normalize_moa_config(
        {
            "presets": {
                "review": {
                    "reference_view": "transcript",
                    "reference_detail_tools": ["terminal"],
                }
            },
            "default_preset": "review",
        }
    )
    preset = cfg["presets"]["review"]
    assert preset["reference_view"] == "transcript"
    assert preset["reference_detail_tools"] == ["terminal"]


def test_reference_prose_budget_coercion():
    cfg = normalize_moa_config(
        {"presets": {"prose": {"reference_prose_budget": 16000}}, "default_preset": "prose"}
    )
    assert cfg["presets"]["prose"]["reference_prose_budget"] == 16000
    assert cfg["reference_prose_budget"] == 16000
    # Default: None (built-in cap).
    assert normalize_moa_config({})["presets"][DEFAULT_MOA_PRESET_NAME]["reference_prose_budget"] is None
