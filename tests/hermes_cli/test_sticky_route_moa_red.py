"""RED: sticky MoA route must not silently become factory GPT defaults.

Product outcomes under test:
1) Selected MoA preset identity (provider=moa + preset name) must survive
   session override sanitize/persist.
2) A *named user preset* whose slots are temporarily incomplete must not be
   rewritten into factory DEFAULT_MOA_REFERENCE_MODELS (gpt-5.5) when the user
   still owns that preset name — silent factory inject is the corruption class
   described in moa_config.validate_moa_payload (#64156) extended to sticky
   route selection.

These tests document desired sticky behavior. On clean main some may FAIL (RED)
until class fixes land. No Billy/daily-grok literals in production assertions
beyond opaque fixture names.
"""

from __future__ import annotations

from gateway.session import PERSISTABLE_MODEL_OVERRIDE_KEYS, sanitize_model_override
from hermes_cli.moa_config import (
    DEFAULT_MOA_REFERENCE_MODELS,
    normalize_moa_config,
    resolve_moa_preset,
)


def test_sanitize_keeps_moa_provider_and_opaque_preset_name():
    """Session override for MoA is provider=moa + model=<preset name>."""
    cleaned = sanitize_model_override(
        {
            "provider": "moa",
            "model": "user-preset-alpha",
            "api_key": "sk-should-never-persist",
            "base_url": "",
        }
    )
    assert cleaned == {"provider": "moa", "model": "user-preset-alpha"}


def test_persistable_keys_are_documented_minimum_for_route_identity():
    """Route identity needs at least provider+model; expand only with proof."""
    assert "provider" in PERSISTABLE_MODEL_OVERRIDE_KEYS
    assert "model" in PERSISTABLE_MODEL_OVERRIDE_KEYS
    # Secrets must never be persistable via this path.
    assert "api_key" not in PERSISTABLE_MODEL_OVERRIDE_KEYS


def test_named_user_preset_empty_refs_must_not_silently_become_factory_gpt():
    """Sticky integrity: named preset with empty refs must not look healthy.

    Current main normalize_moa_config injects DEFAULT_MOA_REFERENCE_MODELS
    (includes openai-codex/gpt-5.5) when cleaned refs are empty. For sticky
    route product integrity, a *named user preset* should either:
      - preserve last-known-good slots, or
      - fail closed / mark invalid,
    not silently present as a healthy preset whose refs are factory GPT.

    RED until the class fix lands: assert factory GPT is NOT injected under
    the user preset name.
    """
    factory_models = {(s["provider"], s["model"]) for s in DEFAULT_MOA_REFERENCE_MODELS}
    assert ("openai-codex", "gpt-5.5") in factory_models  # fixture sanity on current main

    raw = {
        "default_preset": "user-preset-alpha",
        "presets": {
            "user-preset-alpha": {
                "reference_models": [],  # incomplete / wiped
                "aggregator": {
                    "provider": "xai-oauth",
                    "model": "grok-4.5",
                },
            }
        },
    }
    cfg = normalize_moa_config(raw)
    preset = resolve_moa_preset(cfg, "user-preset-alpha")
    refs = preset.get("reference_models") or []
    ref_pairs = {(r.get("provider"), r.get("model")) for r in refs if isinstance(r, dict)}

    # Desired sticky behavior: do not silently substitute factory GPT stack.
    assert ("openai-codex", "gpt-5.5") not in ref_pairs, (
        "named user preset silently received factory GPT reference defaults — "
        "this is the silent-route-corruption class"
    )


def test_standalone_override_survives_sanitize_without_secret_bleed():
    cleaned = sanitize_model_override(
        {
            "provider": "xai-oauth",
            "model": "grok-4.5",
            "api_key": "sk-nope",
            "api_mode": "chat",
        }
    )
    assert cleaned == {"provider": "xai-oauth", "model": "grok-4.5"}
    assert "api_key" not in (cleaned or {})


def test_empty_moa_config_still_seeds_factory_default_preset():
    """Factory empty config may still receive product seed defaults."""
    cfg = normalize_moa_config({})
    refs = cfg.get("reference_models") or []
    pairs = {(r.get("provider"), r.get("model")) for r in refs}
    # Seed path intentionally uses factory defaults — not a sticky bug.
    assert ("openai-codex", "gpt-5.5") in pairs or len(pairs) >= 1
    assert cfg.get("default_preset")


def test_named_preset_keeps_valid_user_slots_without_factory_bleed():
    """A healthy opaque preset must round-trip without factory GPT injection."""
    raw = {
        "default_preset": "user-preset-beta",
        "presets": {
            "user-preset-beta": {
                "reference_models": [
                    {"provider": "provider-alpha", "model": "opaque-model-a"},
                ],
                "aggregator": {
                    "provider": "provider-beta",
                    "model": "opaque-model-b",
                },
            }
        },
    }
    cfg = normalize_moa_config(raw)
    preset = resolve_moa_preset(cfg, "user-preset-beta")
    refs = preset["reference_models"]
    assert len(refs) == 1
    assert refs[0]["provider"] == "provider-alpha"
    assert refs[0]["model"] == "opaque-model-a"
    assert preset["aggregator"]["provider"] == "provider-beta"
    assert preset["aggregator"]["model"] == "opaque-model-b"
    factory = {(s["provider"], s["model"]) for s in DEFAULT_MOA_REFERENCE_MODELS}
    assert (refs[0]["provider"], refs[0]["model"]) not in factory


def test_explicit_empty_reference_models_on_flat_seed_does_not_inject_gpt():
    """Even the factory seed path must respect an explicit empty list."""
    cfg = normalize_moa_config({"reference_models": [], "aggregator": {"provider": "provider-beta", "model": "opaque-model-b"}})
    refs = cfg.get("reference_models") or []
    pairs = {(r.get("provider"), r.get("model")) for r in refs if isinstance(r, dict)}
    assert ("openai-codex", "gpt-5.5") not in pairs
    assert pairs == set()
    assert cfg["aggregator"]["provider"] == "provider-beta"


def test_named_preset_invalid_ref_slots_do_not_become_factory_stack():
    """Wiped/invalid slots under a named preset stay empty (fail closed)."""
    raw = {
        "default_preset": "user-preset-gamma",
        "presets": {
            "user-preset-gamma": {
                "reference_models": [
                    {"provider": "moa", "model": "recursive"},  # invalid recursive
                    {"provider": "", "model": ""},
                ],
                "aggregator": {
                    "provider": "provider-beta",
                    "model": "opaque-model-b",
                    "reasoning_effort": "Ultra",
                },
            }
        },
    }
    cfg = normalize_moa_config(raw)
    preset = resolve_moa_preset(cfg, "user-preset-gamma")
    assert preset["reference_models"] == []
    assert preset["aggregator"]["provider"] == "provider-beta"
    assert preset["aggregator"]["model"] == "opaque-model-b"
    # Product intent effort must survive clean when known on slot
    assert preset["aggregator"].get("reasoning_effort") in (None, "Ultra", "ultra")


def test_decode_moa_turn_does_not_seed_factory_into_empty_encoded_config():
    from hermes_cli.moa_config import decode_moa_turn, encode_moa_turn

    # Encode a healthy opaque preset, then decode — slots must match.
    cfg = {
        "default_preset": "user-preset-delta",
        "presets": {
            "user-preset-delta": {
                "reference_models": [
                    {"provider": "provider-alpha", "model": "opaque-model-a"},
                ],
                "aggregator": {"provider": "provider-beta", "model": "opaque-model-b"},
            }
        },
    }
    encoded = encode_moa_turn("hello", cfg, preset="user-preset-delta")
    prompt, decoded = decode_moa_turn(encoded)
    assert prompt == "hello"
    assert decoded is not None
    refs = decoded.get("reference_models") or []
    assert len(refs) == 1
    assert refs[0]["provider"] == "provider-alpha"
    assert ("openai-codex", "gpt-5.5") not in {
        (r.get("provider"), r.get("model")) for r in refs
    }


def test_build_moa_facade_missing_named_preset_fails_closed(tmp_path, monkeypatch):
    """Selected MoA preset missing from config must not silently become default."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "moa:\n"
        "  default_preset: factory-default\n"
        "  presets:\n"
        "    factory-default:\n"
        "      reference_models:\n"
        "        - {provider: provider-alpha, model: opaque-a}\n"
        "      aggregator: {provider: provider-beta, model: opaque-b}\n"
        "    user-preset-alive:\n"
        "      reference_models:\n"
        "        - {provider: provider-alpha, model: opaque-a}\n"
        "      aggregator: {provider: provider-beta, model: opaque-b}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from types import SimpleNamespace

    import pytest

    from agent.errors import MoAPresetNotFoundError
    from agent.moa_loop import build_moa_facade

    agent = SimpleNamespace(provider="moa", model="user-preset-missing", tool_progress_callback=None)
    with pytest.raises(MoAPresetNotFoundError) as ei:
        build_moa_facade(agent, "user-preset-missing")
    assert "user-preset-missing" in str(ei.value)
    assert "factory-default" not in str(ei.value).split("was not found")[0]


def test_moa_preset_not_found_never_triggers_provider_fallback():
    from agent.error_classifier import classify_api_error
    from agent.errors import MoAPresetNotFoundError

    result = classify_api_error(
        MoAPresetNotFoundError("MoA preset 'gone' was not found"),
        provider="moa",
        model="gone",
    )
    assert result.retryable is False
    assert result.should_fallback is False

def test_resolve_moa_preset_name_realigns_drift_to_default(tmp_path, monkeypatch):
    """Construction helper must re-align non-preset model ids under provider=moa."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "moa:\n"
        "  default_preset: factory-default\n"
        "  presets:\n"
        "    factory-default:\n"
        "      reference_models:\n"
        "        - {provider: provider-alpha, model: opaque-a}\n"
        "      aggregator: {provider: provider-beta, model: opaque-b}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from types import SimpleNamespace

    from agent.moa_loop import build_moa_facade, resolve_moa_preset_name

    agent = SimpleNamespace(provider="moa", model="deepseek-v4-flash", tool_progress_callback=None)
    preset = resolve_moa_preset_name(agent, agent.model)
    assert preset == "factory-default"
    # Helper must not mutate agent.model; caller adopts.
    assert agent.model == "deepseek-v4-flash"
    agent.model = preset
    client = build_moa_facade(agent, preset)
    assert client.chat.completions.preset_name == "factory-default"

