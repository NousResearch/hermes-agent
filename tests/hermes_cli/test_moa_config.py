import pytest
import yaml

from hermes_cli import inventory, moa_cmd

from agent.errors import MoAPresetNotFoundError
from hermes_cli.moa_config import (
    DEFAULT_MOA_AGGREGATOR,
    DEFAULT_MOA_PRESET_NAME,
    DEFAULT_MOA_REFERENCE_MODELS,
    decode_moa_turn,
    exact_moa_preset_name,
    normalize_moa_config,
    resolve_moa_preset,
)


def test_moa_slot_picker_excludes_unconfigured_providers(monkeypatch):
    from hermes_cli import moa_cmd

    captured = {}
    monkeypatch.setattr(moa_cmd, "load_picker_context", lambda: inventory.ConfigContext("", "", "", {}, []))

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


@pytest.fixture
def picker_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "model": {
            "provider": "main",
            "default": "main-model",
            "base_url": "https://main.example/v1",
            "picker": {"hide": ["main", "hidden", "slot"], "order": ["second", "first"]},
        },
        "model_catalog": {"excluded_providers": ["excluded"]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    def discover(**kwargs):
        return [
            {"slug": slug, "name": slug, "models": [f"{slug}-model"],
             "is_current": slug == kwargs["current_provider"]}
            for slug in ("main", "first", "hidden", "slot", "second", "moa")
        ]

    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", discover)
    # Managed local rows are injected after authenticated-provider exclusions.
    monkeypatch.setattr(inventory, "_local_runtime_row", lambda ctx: {
        "slug": "excluded", "name": "excluded", "models": ["excluded-model"],
        "is_current": ctx.current_provider == "excluded",
    })
    monkeypatch.setattr(inventory, "_moa_provider_row", lambda current: None)
    for name in ("_apply_picker_hints", "_apply_pricing", "_apply_capabilities", "_apply_custom_aliases"):
        monkeypatch.setattr(inventory, name, lambda rows, **kwargs: None)


def test_moa_options_filter_excluded_hidden_and_order(picker_config):
    assert [row["slug"] for row in moa_cmd._model_options()] == ["second", "first"]


def test_moa_slot_retains_only_its_current_hidden_provider(picker_config, monkeypatch):
    prompts = []

    def choose(title, rows, default=0):
        prompts.append((rows, default))
        return default

    monkeypatch.setattr(moa_cmd, "_prompt_choice", choose)
    slot = {"provider": "slot", "model": "slot-model"}
    assert moa_cmd._pick_slot(slot) == slot
    assert prompts[0] == (["second  (second)", "first  (first)", "slot  (slot)"], 2)
    prompts.clear()
    assert moa_cmd._pick_slot({"provider": "excluded", "model": "excluded-model"}) == {
        "provider": "second", "model": "second-model",
    }
    assert prompts[0] == (["second  (second)", "first  (first)"], 0)


def test_custom_slot_order_and_exclusions_do_not_depend_on_base_url(tmp_path, monkeypatch):
    from dataclasses import replace
    from hermes_cli import model_switch_providers as providers

    # Keep real custom-endpoint discovery, config loading and preference handling;
    # isolate unrelated built-in auth/network discovery and presentation metadata.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers, "_build_curated_lists", lambda *args: {})
    monkeypatch.setattr(providers, "_collect_authed_provider_slugs", lambda *args: [])
    for name in ("_lap_builtin_rows", "_lap_overlay_rows", "_lap_canonical_rows"):
        monkeypatch.setattr(providers, name, lambda *args: None)
    for name in ("_local_runtime_row", "_moa_provider_row"):
        monkeypatch.setattr(inventory, name, lambda *args: None)
    for name in ("_apply_picker_hints", "_apply_pricing", "_apply_capabilities", "_apply_custom_aliases"):
        monkeypatch.setattr(inventory, name, lambda *args, **kwargs: None)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoints = {
        name: {"base_url": f"https://{name}.example/v1", "models": [f"{name}-model"],
               "discover_models": False}
        for name in ("slot", "other", "excluded")
    }
    config = {
        "model": {"provider": "other", "default": "other-model",
                  "base_url": endpoints["other"]["base_url"]},
        "providers": endpoints,
        "model_catalog": {"excluded_providers": ["excluded"]},
    }
    slot = {"provider": "slot", "model": "slot-model"}
    for prefs, expected in (({}, ["slot", "other"]),
                            ({"hide": ["slot"], "order": ["other", "slot"]}, ["other", "slot"])):
        config["model"]["picker"] = prefs
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
        ctx = replace(inventory.load_picker_context(), current_provider=slot["provider"],
                      current_model=slot["model"], current_base_url=endpoints["slot"]["base_url"])
        for apply_prefs in (False, True):
            with_url = inventory.build_models_payload(ctx, canonical_order=True, apply_picker_prefs=apply_prefs)
            without_url = inventory.build_models_payload(
                replace(ctx, current_base_url=""), canonical_order=True, apply_picker_prefs=apply_prefs)
            assert with_url == without_url
            slugs = [row["slug"] for row in with_url["providers"]]
            if apply_prefs:
                assert slugs == expected
            else:
                # Raw inventory can retain excluded custom rows; compare the
                # visible order without pinning that separate discovery gap.
                assert [slug for slug in slugs if slug != "excluded"] == ["slot", "other"]
        assert [row["slug"] for row in moa_cmd._model_options(slot)] == expected


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


def test_print_config_marks_aggregator_as_billed_and_warns_on_provider_mismatch(capsys):
    """#112359: the aggregator is the acting model billed for the run; when it sits on a
    different provider than the main model, ``hermes moa list``/``configure`` say so."""
    from hermes_cli import moa_cmd

    moa_cmd._print_config({"model": {"provider": "openai-codex"}})

    out = capsys.readouterr().out
    assert "acting model — runs every step" in out
    assert "advise once per user turn" in out
    # Default preset's aggregator is on openrouter → the notice names both providers.
    assert "Aggregator is on openrouter; the whole tool loop will be billed there, not to openai-codex." in out


@pytest.mark.parametrize("cfg", [{"model": {"provider": "openrouter"}}, {}])
def test_billing_notice_silent_when_providers_match_or_main_unknown(cfg, capsys):
    from hermes_cli import moa_cmd

    moa_cmd._print_config(cfg)

    out = capsys.readouterr().out
    assert "acting model — runs every step" in out
    assert "Aggregator is on" not in out


# ── Per-slot max_tokens ────────────────────────────────────────────────────






# --- fanout cadence normalization (every_n) ---








# --- privacy_filter normalization ---






