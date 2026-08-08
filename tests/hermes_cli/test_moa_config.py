import pytest

from agent.errors import MoAPresetNotFoundError
from hermes_cli.moa_config import (
    DEFAULT_MOA_AGGREGATOR,
    DEFAULT_MOA_PRESET_NAME,
    DEFAULT_MOA_REFERENCE_MODELS,
    build_moa_turn_prompt,
    decode_moa_turn,
    exact_moa_preset_name,
    list_moa_presets,
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








# ── Unknown-key diagnostic (#65110) ─────────────────────────────────────────


@pytest.fixture
def moa_warnings(caplog):
    """Capture moa_config warnings with the process-lifetime dedup set cleared.

    ``_MOA_NORMALIZE_WARNED`` persists for the process, so without an explicit
    reset the first test to warn would suppress every later one.
    """
    import logging

    from hermes_cli import moa_config

    moa_config._MOA_NORMALIZE_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.moa_config"):
        yield caplog
    moa_config._MOA_NORMALIZE_WARNED.clear()


def _warning_texts(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "hermes_cli.moa_config"
    ]


def test_unknown_preset_key_warns_with_preset_name_and_key(moa_warnings):
    """#65110: `reasoning_effort` at the preset level is dropped — say so."""
    normalize_moa_config(_preset(reasoning_effort="high"))

    messages = _warning_texts(moa_warnings)
    assert messages == [
        "moa.presets.p: unknown config keys ignored: reasoning_effort"
    ]


def test_unknown_reference_slot_key_warns_with_index(moa_warnings):
    """A slot warning must name WHICH reference model carries the bad key."""
    cfg = {
        "default_preset": "p",
        "presets": {
            "p": {
                "reference_models": [
                    {"provider": "openrouter", "model": "a"},
                    {"provider": "openrouter", "model": "b"},
                    {"provider": "openrouter", "model": "c", "temperature": 0.4},
                ],
                "aggregator": {"provider": "openrouter", "model": "agg"},
            },
        },
    }

    normalize_moa_config(cfg)

    assert _warning_texts(moa_warnings) == [
        "moa.presets.p.reference_models[2]: unknown config keys ignored: temperature"
    ]


def test_unknown_aggregator_slot_key_warns(moa_warnings):
    normalize_moa_config(
        _preset(aggregator={"provider": "openrouter", "model": "agg", "effort": "high"})
    )

    assert _warning_texts(moa_warnings) == [
        "moa.presets.p.aggregator: unknown config keys ignored: effort"
    ]


def test_unknown_key_warning_is_emitted_once_per_process(moa_warnings):
    """normalize_moa_config runs on every model switch — one warning, not N."""
    cfg = _preset(reasoning_effort="high")
    for _ in range(5):
        normalize_moa_config(cfg)
    resolve_moa_preset(cfg, "p")
    list_moa_presets(cfg)

    assert len(_warning_texts(moa_warnings)) == 1


def test_legacy_flat_moa_block_does_not_warn(moa_warnings):
    """Negative control — the legacy flat shape must stay silent.

    In the flat shape the whole ``moa`` block IS the default preset, so its own
    legitimate top-level keys (save_traces, trace_dir, default_preset,
    active_preset, privacy_filter) would every one of them look "unknown" to a
    preset allow-list. Warning here would fire on valid configs.
    """
    flat = {
        "save_traces": True,
        "trace_dir": "/custom/traces",
        "default_preset": "default",
        "active_preset": "",
        "privacy_filter": "display",
        "reference_models": [{"provider": "openrouter", "model": "a"}],
        "aggregator": {"provider": "openrouter", "model": "agg"},
        "max_tokens": 4096,
        "enabled": True,
    }

    cfg = normalize_moa_config(flat)

    assert list(cfg["presets"]) == [DEFAULT_MOA_PRESET_NAME]
    assert _warning_texts(moa_warnings) == []


def test_fully_populated_valid_preset_does_not_warn(moa_warnings):
    """Negative control — every documented knob, plus a GUI-written config.

    ``enabled`` on the AGGREGATOR slot is the load-bearing case: _clean_slot
    only honors it for reference slots, but web_server's MoaModelSlot writes it
    into the aggregator on every dashboard save, so it must not warn.
    """
    cfg = {
        "default_preset": "p",
        "active_preset": "p",
        "privacy_filter": "full",
        "presets": {
            "p": {
                "enabled": True,
                "reference_models": [
                    {
                        "provider": "openrouter",
                        "model": "a",
                        "reasoning_effort": "high",
                        "max_tokens": 600,
                        "enabled": True,
                    },
                ],
                "aggregator": {
                    "provider": "openrouter",
                    "model": "agg",
                    "reasoning_effort": "high",
                    "enabled": True,
                },
                "reference_temperature": 0.6,
                "aggregator_temperature": 0.4,
                "reference_timeout": 120,
                "degraded_reference_policy": "silent",
                "max_tokens": 4096,
                "reference_max_tokens": 600,
                "fanout": "per_iteration",
            },
        },
    }

    normalize_moa_config(cfg)

    assert _warning_texts(moa_warnings) == []


def test_unknown_keys_do_not_change_normalized_output(moa_warnings):
    """The diagnostic is a warning only — normalization is byte-for-byte equal."""
    with_unknown = normalize_moa_config(
        _preset(
            reasoning_effort="high",
            reference_models=[
                {"provider": "openrouter", "model": "a", "temperature": 0.4},
            ],
            aggregator={"provider": "openrouter", "model": "agg", "effort": "high"},
        )
    )
    expected = normalize_moa_config(
        _preset(
            reference_models=[{"provider": "openrouter", "model": "a"}],
            aggregator={"provider": "openrouter", "model": "agg"},
        )
    )

    assert with_unknown == expected
    # ...and it did warn (preset + reference slot + aggregator), so the
    # equality above is not vacuously comparing two silent runs.
    assert len(_warning_texts(moa_warnings)) == 3
