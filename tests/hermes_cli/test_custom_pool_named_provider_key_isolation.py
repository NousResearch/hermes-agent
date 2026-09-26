"""A bare-``custom`` main model and a named provider on the SAME ``base_url`` each resolve
with their OWN credential.

Regressions, both from matching a custom pool by ``base_url`` alone:

* ``agent.credential_pool._seed_custom_pool`` seeded the main model's ``api_key`` into the
  first same-URL named provider's pool, so that provider resolved with the main model's key.
* ``_resolve_openrouter_runtime`` resolved the main model from that same-URL provider's pool,
  so the main model resolved with the provider's key.

The ``hermes model`` shape (``model.api_key: ${VAR}`` beside a ``custom_providers`` entry with
``key_env: VAR``) is the model's own entry and keeps sharing its pool.
"""

from __future__ import annotations

import json

import hermes_yaml as yaml
import pytest

OPENROUTER = "https://openrouter.ai/api/v1"
GENERIC = "https://llm.example.test/v1"
ENDPOINTS = pytest.mark.parametrize("endpoint", [OPENROUTER, GENERIC], ids=["openrouter", "generic"])


def _write_home(tmp_path, monkeypatch, config, *, env=None, pools=None):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "CUSTOM_BASE_URL", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    for var, value in (env or {}).items():
        monkeypatch.setenv(var, value)
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": pools or {}}))
    return home


def _main_model(endpoint, api_key="main-key"):
    return {
        "default": "openai/gpt-4o-mini",
        "provider": "custom",
        "base_url": endpoint,
        "api_mode": "chat_completions",
        "api_key": api_key,
    }


def _sibling_config(endpoint, **credential):
    return {
        "model": _main_model(endpoint),
        "providers": {
            "second": {
                "name": "Second account",
                "api": endpoint,
                "default_model": "google/gemini-2.5-flash",
                **credential,
            }
        },
    }


def _pool_rows(pool_key):
    from agent.credential_pool import load_pool

    return [(entry.source, entry.access_token) for entry in load_pool(pool_key).entries()]


# ── the hermes-model setup shape keeps its pool ──────────────────────────────────────────


@ENDPOINTS
def test_setup_flow_shape_keeps_the_models_own_pool(tmp_path, monkeypatch, endpoint):
    """``hermes model`` writes ``model.api_key: ${VAR}`` and a ``custom_providers`` entry with
    ``key_env: VAR`` for the same endpoint: that entry is the model's own, so its pool keeps the
    ``model_config`` row and the main model resolves (on openrouter.ai only through it or its key)."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(tmp_path, monkeypatch, {
        "model": _main_model(endpoint, api_key="${HERMES_CUSTOM_SETUP_API_KEY}"),
        "custom_providers": [
            {"name": "Setup endpoint", "base_url": endpoint, "key_env": "HERMES_CUSTOM_SETUP_API_KEY",
             "model": "openai/gpt-4o-mini"},
        ],
    }, env={"HERMES_CUSTOM_SETUP_API_KEY": "setup-key"})

    assert ("model_config", "setup-key") in _pool_rows("custom:setup-endpoint")

    main = resolve_runtime_provider(requested="custom")
    assert main["api_key"] == "setup-key"
    assert main["source"] == "pool:custom:setup-endpoint"


# ── key_env sibling: each side keeps its own key ─────────────────────────────────────────


@ENDPOINTS
def test_sibling_with_its_own_key_env_resolves_with_its_own_key(tmp_path, monkeypatch, endpoint):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(tmp_path, monkeypatch, _sibling_config(endpoint, key_env="SECOND_KEY"), env={"SECOND_KEY": "second-key"})

    assert resolve_runtime_provider(requested="second")["api_key"] == "second-key"
    assert all(token != "main-key" for _source, token in _pool_rows("custom:second-account"))


@ENDPOINTS
def test_main_model_beside_a_key_env_sibling_resolves_with_its_own_key(tmp_path, monkeypatch, endpoint):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(tmp_path, monkeypatch, _sibling_config(endpoint, key_env="SECOND_KEY"), env={"SECOND_KEY": "second-key"})

    main = resolve_runtime_provider(requested="custom")
    assert main["api_key"] == "main-key"
    assert main["source"] != "pool:custom:second-account"


def test_switch_model_to_the_sibling_returns_its_own_key(tmp_path, monkeypatch):
    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.model_switch import switch_model

    _write_home(tmp_path, monkeypatch, _sibling_config(OPENROUTER, key_env="SECOND_KEY"), env={"SECOND_KEY": "second-key"})

    cfg = load_config()
    result = switch_model(
        raw_input="google/gemini-2.5-flash",
        explicit_provider="second",
        current_provider="custom",
        current_model="openai/gpt-4o-mini",
        current_base_url=OPENROUTER,
        current_api_key="main-key",
        user_providers=cfg.get("providers"),
        custom_providers=get_compatible_custom_providers(cfg),
    )

    assert result.target_provider == "second"
    assert result.api_key == "second-key"


# ── literal api_key sibling: the inverse leak ────────────────────────────────────────────


@ENDPOINTS
def test_main_model_beside_a_literal_key_sibling_keeps_its_own_key(tmp_path, monkeypatch, endpoint):
    """The main model must not resolve from the sibling's pool, which holds the sibling's key."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(tmp_path, monkeypatch, _sibling_config(endpoint, api_key="second-key"))

    main = resolve_runtime_provider(requested="custom")
    assert main["api_key"] == "main-key"
    assert resolve_runtime_provider(requested="second")["api_key"] == "second-key"


def test_main_model_does_not_take_a_manual_row_from_a_siblings_pool(tmp_path, monkeypatch):
    """``hermes auth add`` rows in a sibling's pool are the sibling's credentials."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(
        tmp_path, monkeypatch, _sibling_config(GENERIC, key_env="SECOND_KEY"), env={"SECOND_KEY": "second-key"},
        pools={"custom:second-account": [
            {"id": "m1", "source": "manual", "auth_type": "api_key", "access_token": "second-manual-key",
             "base_url": GENERIC, "label": "manual", "priority": 0},
        ]},
    )

    main = resolve_runtime_provider(requested="custom")
    assert main["api_key"] == "main-key"


def test_sibling_with_key_cmd_is_not_given_the_main_models_key(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(tmp_path, monkeypatch, _sibling_config(GENERIC, key_cmd="printf second-key"))

    assert resolve_runtime_provider(requested="second")["api_key"] != "main-key"
    assert resolve_runtime_provider(requested="custom")["api_key"] == "main-key"


def test_sibling_whose_key_env_is_unset_is_not_given_the_main_models_key(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.delenv("SECOND_KEY", raising=False)
    _write_home(tmp_path, monkeypatch, _sibling_config(GENERIC, key_env="SECOND_KEY"))

    assert resolve_runtime_provider(requested="second")["api_key"] != "main-key"
    assert resolve_runtime_provider(requested="custom")["api_key"] == "main-key"


def test_credential_less_sibling_still_shares_the_models_key(tmp_path, monkeypatch):
    """#100413: a same-URL entry with no credential of its own keeps inheriting the model key."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_home(tmp_path, monkeypatch, _sibling_config(GENERIC))

    assert ("model_config", "main-key") in _pool_rows("custom:second-account")
    assert resolve_runtime_provider(requested="second")["api_key"] == "main-key"
    assert resolve_runtime_provider(requested="custom")["api_key"] == "main-key"


# ── a model with no key of its own still borrows its sole same-URL entry's key ──────────


@ENDPOINTS
@pytest.mark.parametrize("manual_row", [False, True], ids=["entry-key", "manual-row"])
def test_model_without_its_own_key_still_uses_its_sole_same_url_entry(tmp_path, monkeypatch, endpoint, manual_row):
    """With no model key and no env key there is no own key to tell the entry apart by, so the
    lookup stays URL-only, as before: the model resolves with that entry's credential."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    model = _main_model(endpoint)
    del model["api_key"]
    pools = None
    if manual_row:
        pools = {"custom:foo": [
            {"id": "m1", "source": "manual", "auth_type": "api_key", "access_token": "manual-key",
             "base_url": endpoint, "label": "manual", "priority": 0},
        ]}
    _write_home(
        tmp_path, monkeypatch,
        {"model": model, "custom_providers": [{"name": "Foo", "base_url": endpoint, "api_key": "entry-key", "model": "m1"}]},
        pools=pools,
    )

    main = resolve_runtime_provider(requested="custom")
    assert main["api_key"] in (("manual-key", "entry-key") if manual_row else ("entry-key",))
    assert main["api_key"] != "no-key-required"
