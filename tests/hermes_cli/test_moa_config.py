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


# ── G8 V2 oracle: reference_input_scope / reference_input_filter (RED) ─────
#
# Contract for the MoA reference input boundary (feature NOT yet implemented):
# read-time defaults degrade to the legacy behavior (conversation / none), and
# write-time validation rejects anything else. The runtime seam consumes these
# fields from the RESOLVED PRESET (facade create() reads presets via
# resolve_moa_preset), so the contract is asserted at preset level — no new
# top-level flattened keys are required.
#
# SUT call reconciliation (parametrization runs both combos):
#   normalize_moa_config x4  (default / preset w/ fields / invalid read / flat)
#   validate_moa_payload x2  (invalid write / valid accepted)
#   decode_moa_turn      x1          = 7 per combo -> 14 total
#   (encode_moa_turn is a fixture builder and is not counted as SUT.)


@pytest.mark.parametrize(
    "scope, input_filter",
    [("conversation", "none"), ("current_turn", "redact")],
)
def test_reference_input_scope_filter_config_contract(scope, input_filter):
    """F1: reference_input_scope/reference_input_filter read+write contract.

    - default legacy: absent config falls back to conversation / none;
    - preset roundtrip + validate-accepted => normalize preserves valid values;
    - invalid write-time values are rejected loudly (validate_moa_payload),
      never silently repaired;
    - invalid read-time values degrade to the legacy defaults;
    - the legacy flat shape still becomes the default preset, fields included;
    - the /moa one-shot marker round-trips the fields through encode/decode.

    RED baseline: neither field exists in the config model yet, so every
    value-bearing assertion fails on the absent key (None != expected).
    """
    from hermes_cli.moa_config import (
        decode_moa_turn,
        encode_moa_turn,
        normalize_moa_config,
        validate_moa_payload,
    )

    # Exercise every contract seam before asserting RED, so the baseline run
    # mechanically reconciles all 7 calls per parameterized case.
    default_cfg = normalize_moa_config({})

    valid_preset = {
        **_valid_preset_payload(),
        "reference_input_scope": scope,
        "reference_input_filter": input_filter,
    }
    payload = {"default_preset": "p", "presets": {"p": valid_preset}}
    valid_cfg = normalize_moa_config(payload)

    bad_payload = {
        "presets": {
            "p": {
                **_valid_preset_payload(),
                "reference_input_scope": f"{scope}-bogus",
                "reference_input_filter": f"{input_filter}-bogus",
            }
        }
    }
    problems = validate_moa_payload(bad_payload)
    valid_problems = validate_moa_payload(payload)

    invalid_cfg = normalize_moa_config(
        {
            "default_preset": "p",
            "presets": {
                "p": {
                    **valid_preset,
                    "reference_input_scope": f"{scope}-bogus",
                    "reference_input_filter": f"{input_filter}-bogus",
                }
            },
        }
    )

    flat = {
        **_valid_preset_payload(),
        "reference_input_scope": scope,
        "reference_input_filter": input_filter,
    }
    flat_cfg = normalize_moa_config(flat)
    prompt, decoded = decode_moa_turn(encode_moa_turn("do the thing", config=payload))

    # 1. Default legacy: both fields fall back to conversation / none.
    assert default_cfg["presets"]["default"].get("reference_input_scope") == "conversation"
    assert default_cfg["presets"]["default"].get("reference_input_filter") == "none"

    # 2. Preset roundtrip: valid values survive normalize unchanged.
    assert valid_cfg["presets"]["p"].get("reference_input_scope") == scope
    assert valid_cfg["presets"]["p"].get("reference_input_filter") == input_filter
    assert valid_cfg["presets"]["p"]["aggregator"] == valid_preset["aggregator"]

    # 3. Invalid write-time values are rejected loudly, not silently repaired.
    assert any(
        "reference_input_scope" in p or "reference_input_filter" in p
        for p in problems
    )

    # 4. Valid write-time payload passes validation (the "accepted" half).
    assert valid_problems == []

    # 5. Invalid named-preset values degrade to the legacy defaults at read time.
    assert invalid_cfg["presets"]["p"].get("reference_input_scope") == "conversation"
    assert invalid_cfg["presets"]["p"].get("reference_input_filter") == "none"

    # 6. Legacy flat shape still becomes the default preset, fields included.
    assert flat_cfg["presets"]["default"]["reference_models"] == _enabled_refs(
        flat["reference_models"]
    )
    assert flat_cfg["presets"]["default"]["aggregator"] == flat["aggregator"]
    assert flat_cfg["presets"]["default"].get("reference_input_scope") == scope
    assert flat_cfg["presets"]["default"].get("reference_input_filter") == input_filter

    # 7. /moa one-shot marker round-trips the fields through encode/decode.
    assert prompt == "do the thing"
    assert decoded.get("reference_input_scope") == scope
    assert decoded.get("reference_input_filter") == input_filter






