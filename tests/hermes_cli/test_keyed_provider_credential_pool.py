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
