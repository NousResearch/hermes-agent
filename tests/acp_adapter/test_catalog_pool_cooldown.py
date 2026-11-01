"""ACP pickers must retain models usable beside a sibling entitlement bench (LS-3671)."""

import json
from pathlib import Path

import pytest

from acp_adapter.model_catalog import build_model_state
from hermes_cli.inventory import build_models_payload as native_build_models_payload
from hermes_cli.inventory import load_picker_context as native_load_picker_context


@pytest.mark.parametrize("current_provider,current_model", [
    ("custom", "deepseek-v4-flash"),
    ("claude-subscription-directsdk-experimental", "claude-opus-5-5[1m]"),
])
def test_sibling_model_bench_does_not_remove_codex_picker(
    tmp_path, monkeypatch, current_provider, current_model,
):
    from agent import credential_pool, models_dev
    from hermes_cli import inventory, model_switch_providers, providers

    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "physical-home")
    root = tmp_path / "hermes-global"
    root.mkdir()
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-6.1-sol\n",
    )
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "credential_pool": {"openai-codex": [{
            "id": "synthetic-fixture", "label": "fixture", "auth_type": "oauth",
            "source": "manual:device_code", "priority": 0,
            "access_token": "synthetic-not-a-real-token",
            "model_cooldowns": {"gpt-5.5": 4102444800.0},
        }]},
    }))
    # Keep the actual inventory, auth fallback, pool and overlay gates. Only model
    # discovery's external registry/catalog boundary is replaced with offline data.
    monkeypatch.setattr(inventory, "build_models_payload", native_build_models_payload)
    monkeypatch.setattr(inventory, "load_picker_context", native_load_picker_context)
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers, "HERMES_OVERLAYS", {
        "openai-codex": providers.HERMES_OVERLAYS["openai-codex"],
    })
    models = ["gpt-6.1-sol", "gpt-6.1-sol-900k", "gpt-5.5"]
    monkeypatch.setattr(model_switch_providers, "_build_curated_lists", lambda *_a, **_kw: {
        "openai-codex": models,
    })
    monkeypatch.setattr(model_switch_providers, "_live_or_curated_ids", lambda *_a, **_kw: models)
    monkeypatch.setattr(model_switch_providers, "_prefetch_provider_models_parallel", lambda *_a: None)

    for profile in (root, root / "profiles" / "pike", root):
        profile.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        # Deliberate route selection in each profile; credentials remain root-only.
        (profile / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n  default: gpt-6.1-sol\n",
        )
        pool = credential_pool.load_pool("openai-codex")
        assert not pool.has_available()
        assert pool.has_available(model="gpt-6.1-sol")
        assert not pool.has_available(model="gpt-5.5")
        state = build_model_state(current_model, current_provider, "")
        choices = {item.model_id for item in state.available_models}
        assert "openai-codex:gpt-6.1-sol" in choices
        assert "openai-codex:gpt-6.1-sol-900k" in choices
        assert state.current_model_id == f"{current_provider}:{current_model}"
