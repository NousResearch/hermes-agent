"""`model.aliases:` (config.yaml `model:` section) `key_env`/`api_key` silently dropped.

Regression test for the remaining half of #83847: the `model_aliases:` top-level
section was fixed by PR #99625, but the `model:` → `aliases:` section path inside
`_load_direct_aliases()` still builds `DirectAlias` without `api_key`/`key_env`,
so `key_env` is silently discarded for that config layout.
"""

import pytest


def _install_model_aliases_config(monkeypatch, alias_entry):
    """Point every config reader at a config with `model.aliases:` only."""
    cfg = {
        "model": {
            "default": "gpt-4",
            "provider": "openrouter",
            "aliases": {"theta": alias_entry},
        },
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda *a, **k: cfg)
    return cfg


def test_model_section_alias_key_env_is_loaded(monkeypatch):
    """`model.aliases: <name>.key_env` must survive into DirectAlias."""
    import hermes_cli.model_switch as ms

    _install_model_aliases_config(monkeypatch, {
        "model": "glm-4.7-flash",
        "provider": "custom",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_VELOS_KEY",
    })
    monkeypatch.setenv("ZHIPU_VELOS_KEY", "sk-real-zhipu-key")

    ms._load_direct_aliases.cache_clear() if hasattr(ms._load_direct_aliases, "cache_clear") else None
    from hermes_cli.model_switch import _load_direct_aliases
    merged = _load_direct_aliases()
    alias = merged.get("theta")

    assert alias is not None, "alias was not registered"
    assert alias.key_env == "ZHIPU_VELOS_KEY", (
        f"key_env was dropped! got {alias.key_env!r}"
    )


def test_model_section_alias_api_key_is_loaded(monkeypatch):
    """`model.aliases: <name>.api_key` must survive into DirectAlias."""
    import hermes_cli.model_switch as ms

    _install_model_aliases_config(monkeypatch, {
        "model": "glm-4.7-flash",
        "provider": "custom",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "sk-literal-test",
    })

    ms._load_direct_aliases.cache_clear() if hasattr(ms._load_direct_aliases, "cache_clear") else None
    from hermes_cli.model_switch import _load_direct_aliases
    merged = _load_direct_aliases()
    alias = merged.get("theta")

    assert alias is not None, "alias was not registered"
    assert alias.api_key == "sk-literal-test", (
        f"api_key was dropped! got {alias.api_key!r}"
    )


def test_model_section_alias_credentials_default_to_empty(monkeypatch):
    """When neither api_key nor key_env is set, both default to '' (no silent leak)."""
    import hermes_cli.model_switch as ms

    _install_model_aliases_config(monkeypatch, {
        "model": "glm-4.7-flash",
        "provider": "custom",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    })

    ms._load_direct_aliases.cache_clear() if hasattr(ms._load_direct_aliases, "cache_clear") else None
    from hermes_cli.model_switch import _load_direct_aliases
    merged = _load_direct_aliases()
    alias = merged.get("theta")

    assert alias is not None, "alias was not registered"
    assert alias.api_key == ""
    assert alias.key_env == ""
