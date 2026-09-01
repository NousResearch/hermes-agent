"""Keyed ``providers.<key>`` entries must use the durable pool slug.

``hermes auth add b-ai`` stores keys under ``credential_pool.b-ai``. Runtime
used to look up ``custom:<display-name>`` (e.g. ``custom:b.ai`` from
``name: B.AI``), miss the pool, and send the ``no-key-required`` placeholder
to an auth-required endpoint (HTTP 401 Invalid api_key format).
"""

from __future__ import annotations

import json

import yaml


POOL_KEY = "sk-real-b-ai-pool-key-12345"
LEGACY_KEY = "sk-legacy-custom-b-ai-pool-key"
ENDPOINT = "https://api.b.ai/v1"


def _write_keyed_provider_home(tmp_path, monkeypatch, *, pool_id="b-ai", extra_config=None):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = {
        "model": {"default": "b-ai-model", "provider": "b-ai"},
        "providers": {
            "b-ai": {
                "name": "B.AI",
                "base_url": ENDPOINT,
            }
        },
    }
    if extra_config:
        config["providers"]["b-ai"].update(extra_config)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    pool_id: [
                        {
                            "id": "k1",
                            "label": "primary",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": POOL_KEY if pool_id == "b-ai" else LEGACY_KEY,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return hermes_home


def test_get_named_custom_provider_exposes_provider_key_and_key_env(
    tmp_path, monkeypatch
):
    _write_keyed_provider_home(
        tmp_path, monkeypatch, extra_config={"key_env": "B_AI_API_KEY"}
    )
    monkeypatch.setenv("B_AI_API_KEY", "sk-from-env-not-the-pool")

    from hermes_cli.runtime_provider import _get_named_custom_provider

    entry = _get_named_custom_provider("b-ai")
    assert entry is not None
    assert entry.get("provider_key") == "b-ai"
    assert entry.get("key_env") == "B_AI_API_KEY"
    assert entry.get("name") == "B.AI"
    assert entry.get("base_url") == ENDPOINT


def test_keyed_provider_runtime_uses_durable_pool_slug(tmp_path, monkeypatch):
    """Main turns must send the pooled key, not the no-key-required placeholder."""
    _write_keyed_provider_home(tmp_path, monkeypatch)

    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="b-ai")
    assert resolved["base_url"] == ENDPOINT
    assert resolved["api_key"] == POOL_KEY
    assert resolved["api_key"] != "no-key-required"
    assert str(resolved.get("source") or "").startswith("pool:")


def test_keyed_provider_runtime_falls_back_to_legacy_custom_namespace(
    tmp_path, monkeypatch
):
    """Older auth.json rows stored under custom:<display-name> must still work."""
    _write_keyed_provider_home(tmp_path, monkeypatch, pool_id="custom:b.ai")

    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="b-ai")
    assert resolved["api_key"] == LEGACY_KEY
    assert resolved["api_key"] != "no-key-required"


def _model_config_entry(entry_id, token):
    return {
        "id": entry_id,
        "label": "model_config",
        "auth_type": "api_key",
        "priority": 0,
        "source": "model_config",
        "access_token": token,
    }


def test_prune_keeps_active_legacy_pool_for_keyed_provider(tmp_path, monkeypatch):
    """A keyed provider's own legacy-named pool must not be false-pruned.

    Regression: with keys stored under ``custom:b.ai`` while the provider is
    configured as ``providers.b-ai``, the active pool key resolves to the
    durable slug ``b-ai``; comparing with ``==`` let the prune strip the
    provider's own current credential from ``custom:b.ai``.
    """
    config = {
        "model": {"default": "b-ai-model", "provider": "b-ai"},
        "providers": {
            "b-ai": {
                "name": "B.AI",
                "base_url": "https://api.b.ai/v1",
            }
        },
    }
    pools = {
        # legacy-named pool for the (now keyed) b.ai provider holding the
        # credential seeded from model.api_key — this is the ACTIVE pool
        "custom:b.ai": [_model_config_entry("mc1", "sk-current-b-ai-key")],
        # an unrelated stale pool that SHOULD be pruned
        "custom:old-endpoint": [_model_config_entry("mc2", "sk-stale-key")],
    }
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": pools,
            }
        ),
        encoding="utf-8",
    )

    from hermes_cli.model_setup_flows import (
        _prune_replaced_custom_model_config_credentials,
    )

    _prune_replaced_custom_model_config_credentials(
        "https://api.b.ai/v1", provider_name="B.AI"
    )

    after = json.loads(
        (tmp_path / ".hermes" / "auth.json").read_text(encoding="utf-8")
    )
    kept = (after.get("credential_pool") or {}).get("custom:b.ai")
    pruned = (after.get("credential_pool") or {}).get("custom:old-endpoint")

    assert kept, "active provider's own legacy-named pool must keep its credential"
    assert kept[0]["access_token"] == "sk-current-b-ai-key"
    assert pruned == [], "stale pool for a different endpoint must be pruned"


def test_seed_custom_pool_matches_legacy_named_pool(tmp_path, monkeypatch):
    """A legacy-named pool must still seed from model.api_key.

    Regression: with model.provider 'custom' pointing at a keyed provider's
    base_url, the pool key ``custom:b.ai`` no longer equals the preferred
    candidate (the slug ``b-ai``), silently skipping the model_config seed.
    """
    config = {
        "model": {
            "default": "b-ai-model",
            "provider": "custom",
            "base_url": "https://api.b.ai/v1",
            "api_key": "sk-model-config-key",
        },
        "providers": {
            "b-ai": {
                "name": "B.AI",
                "base_url": "https://api.b.ai/v1",
            }
        },
    }
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {"custom:b.ai": []},
            }
        ),
        encoding="utf-8",
    )

    from agent.credential_pool import load_pool

    pool = load_pool("custom:b.ai")
    entries = pool.entries() if hasattr(pool, "entries") else []
    seeded = [e for e in entries if getattr(e, "source", "") == "model_config"]
    assert seeded, "legacy-named pool must still seed model_config from model.api_key"
    assert getattr(seeded[0], "access_token", "") == "sk-model-config-key"
