"""A cron creation snapshot must pin the durable provider identity, not the
resolved billing class.

``_compute_provider_model_snapshots`` stored the resolved runtime's provider
tag. Every named custom entry resolves to the literal ``"custom"``, so a job
created without an explicit provider pinned ``"custom"``; on replay the
scheduler resolves that bare string through the OpenRouter fallback instead of
the configured entry (#109765). The snapshot must heal bare ``custom`` to the
``custom:<name>`` identity — the same lookup session persistence uses.
"""

from __future__ import annotations

import pytest

from cron import jobs
from hermes_cli import runtime_provider as rp

PROVIDER_KEY = "omniroute"
BASE_URL = "https://omniroute.invalid/v1"
CANONICAL = f"custom:{PROVIDER_KEY}"


@pytest.fixture
def named_custom_config(monkeypatch):
    config = {
        "model": {"default": "hermes-smart-stack", "provider": CANONICAL},
        "custom_providers": [
            {
                "name": PROVIDER_KEY,
                "base_url": BASE_URL,
                "api_key": "sk-test",
                "model": "hermes-smart-stack",
            },
        ],
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    return config


def _snapshot(monkeypatch, runtime):
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **k: dict(runtime))
    return jobs._compute_provider_model_snapshots(
        provider=None, model=None, base_url=None, no_agent=False
    )


def test_bare_custom_snapshot_heals_to_named_identity(named_custom_config, monkeypatch):
    """The regression (#109765): the resolved tag `custom` must not be pinned as-is."""
    provider_snapshot, _ = _snapshot(
        monkeypatch,
        {"provider": "custom", "base_url": BASE_URL, "requested_provider": CANONICAL},
    )
    assert provider_snapshot == CANONICAL


def test_unresolvable_bare_custom_keeps_current_behaviour(monkeypatch):
    """No configured entry to heal to: keep the bare tag instead of inventing an identity."""
    config = {"model": {}}
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    provider_snapshot, _ = _snapshot(
        monkeypatch,
        {
            "provider": "custom",
            "base_url": "https://unconfigured.invalid/v1",
            "requested_provider": "custom",
        },
    )
    assert provider_snapshot == "custom"


def test_non_custom_tag_is_stored_unchanged(monkeypatch):
    """Providers that are their own identity (openrouter, anthropic, ...) need no healing."""
    provider_snapshot, _ = _snapshot(
        monkeypatch,
        {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "requested_provider": "auto",
        },
    )
    assert provider_snapshot == "openrouter"


PROVIDERS_KEY = "tokenrhythm"
PROVIDERS_BASE_URL = "https://api.tokenrhythm.invalid/v1"
PROVIDERS_CANONICAL = f"custom:{PROVIDERS_KEY}"


@pytest.fixture
def providers_only_config(monkeypatch):
    config = {
        "model": {"default": "hermes-smart-stack", "provider": PROVIDERS_CANONICAL},
        "providers": {
            PROVIDERS_KEY: {
                "name": "TokenRhythm",
                "api": PROVIDERS_BASE_URL,
                "key_env": "TOKENRHYTHM_API_KEY",
                "model": "hermes-smart-stack",
            },
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    return config


def test_providers_only_named_entry_heals(providers_only_config, monkeypatch):
    """A keyed ``providers:`` entry with no ``custom_providers`` at all (review ask on this PR):
    ``_find_custom_identity`` scans ``providers:`` first, so the snapshot still heals."""
    provider_snapshot, _ = _snapshot(
        monkeypatch,
        {
            "provider": "custom",
            "base_url": PROVIDERS_BASE_URL,
            "requested_provider": "custom",
        },
    )
    assert provider_snapshot == PROVIDERS_CANONICAL


BARE_NAME_KEY = "gx10-8888"
BARE_NAME_BASE_URL = "http://gx10:8888/v1"


def test_bare_name_provider_spelling_heals(monkeypatch):
    """A bare-name ``model.provider`` spelling (no ``custom:`` prefix), as independently
    reproduced on #109765: the resolve layer still reports tag ``custom`` with the bare
    name as ``requested_provider`` — the endpoint match must heal it regardless."""
    config = {
        "model": {"default": "qwen3.8-flash-next", "provider": BARE_NAME_KEY},
        "providers": {
            BARE_NAME_KEY: {
                "base_url": BARE_NAME_BASE_URL,
                "model": "qwen3.8-flash-next",
                "key_env": "HERMES_CUSTOM_GX10_8888_API_KEY",
            },
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    provider_snapshot, _ = _snapshot(
        monkeypatch,
        {
            "provider": "custom",
            "base_url": BARE_NAME_BASE_URL,
            "requested_provider": BARE_NAME_KEY,
        },
    )
    assert provider_snapshot == f"custom:{BARE_NAME_KEY}"
