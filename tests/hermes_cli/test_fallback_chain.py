"""Tests for NVIDIA/Nous dynamic fallback chain expansion."""

from __future__ import annotations

from hermes_cli.fallback_chain import (
    is_nous_free_fallback_entry,
    is_nvidia_auto_fallback_entry,
    normalize_fallback_entries,
)
from hermes_cli.fallback_config import get_fallback_chain, resolve_fallback_chain


def test_nvidia_auto_expands_and_skips_primary_model():
    chain = normalize_fallback_entries(
        [{"provider": "nvidia", "model": "auto"}],
        exclude_model="nvidia/nemotron-3-super-120b-a12b",
    )
    providers = {entry["provider"] for entry in chain}
    models = [entry["model"] for entry in chain]

    assert providers == {"nvidia"}
    assert "nvidia/nemotron-3-super-120b-a12b" not in models
    assert len(models) >= 2


def test_nous_auto_free_expands_to_free_models():
    chain = normalize_fallback_entries(
        [{"provider": "nous", "model": "auto-free"}],
    )
    assert chain
    assert all(entry["provider"] == "nous" for entry in chain)
    assert all(entry["model"].endswith(":free") or entry["model"].endswith("-free") for entry in chain)


def test_resolve_fallback_chain_injects_nvidia_rotation_when_only_duplicate_present():
    config = {
        "model": {
            "provider": "nvidia",
            "default": "nvidia/nemotron-3-super-120b-a12b",
        },
        "fallback_providers": [
            {"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b"},
            {"provider": "nous", "model": "auto-free"},
        ],
    }
    raw = get_fallback_chain(config)
    assert len(raw) == 2

    resolved = resolve_fallback_chain(config)
    nvidia_models = [
        entry["model"]
        for entry in resolved
        if entry["provider"] == "nvidia"
    ]
    assert "nvidia/nemotron-3-super-120b-a12b" not in nvidia_models
    assert len(nvidia_models) >= 2
    assert any(entry["provider"] == "nous" for entry in resolved)


def test_resolve_fallback_chain_defaults_for_bare_nvidia_primary():
    config = {
        "model": {
            "provider": "nvidia",
            "default": "nvidia/nemotron-3-super-120b-a12b",
        },
        "fallback_providers": [],
    }
    resolved = resolve_fallback_chain(config)
    assert any(entry["provider"] == "nvidia" for entry in resolved)
    assert any(entry["provider"] == "nous" for entry in resolved)
    assert len(resolved) >= 3


def test_dynamic_entry_detectors():
    assert is_nvidia_auto_fallback_entry({"provider": "nvidia", "model": "auto"})
    assert is_nous_free_fallback_entry({"provider": "nous", "model": "auto-free"})
