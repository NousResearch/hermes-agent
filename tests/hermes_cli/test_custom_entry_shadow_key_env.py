"""A ``providers:`` entry keyed like a built-in sends its own ``key_env``, never the built-in's key.

The entry's durable pool slug equals the built-in's pool id, and ``load_pool`` seeds that pool
from the built-in's env vars (``env:DEEPSEEK_API_KEY``). The named-custom rung consulted the pool
before ``key_env``, so every turn to the custom endpoint carried the built-in provider's secret.
"""

from __future__ import annotations

import json

import hermes_yaml as yaml
import pytest

ENDPOINT = "http://127.0.0.1:18080/v1"
ENTRY_KEY = "entry-key-A-0123456789"
BUILTIN_KEY = "builtin-key-B-0123456789"
MANUAL_KEY = "manual-pool-key-0123456789"

# (entry key == built-in pool id, the built-in's api-key env var)
SHADOWED_BUILTINS = [
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("xai", "XAI_API_KEY"),
]


def _home(tmp_path, monkeypatch, name, builtin_env, *, entry_extra=None, pool=None):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    entry = {"name": name, "api": ENDPOINT, "transport": "chat_completions", "default_model": "m"}
    entry.update({"key_env": "ENTRY_KEY_VAR"} if entry_extra is None else entry_extra)
    config = {"model": {"default": "m", "provider": f"custom:{name}"}, "providers": {name: entry}}
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": pool or {}}), encoding="utf-8")
    monkeypatch.setenv(builtin_env, BUILTIN_KEY)
    monkeypatch.setenv("ENTRY_KEY_VAR", ENTRY_KEY)
    return hermes_home


@pytest.mark.parametrize("name,builtin_env", SHADOWED_BUILTINS)
def test_shadowing_entry_main_runtime_sends_its_key_env(tmp_path, monkeypatch, name, builtin_env):
    """``model.provider: custom:<name>`` (which pick a bare ``<name>`` routes to is #140107's scope)."""
    _home(tmp_path, monkeypatch, name, builtin_env)
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested=f"custom:{name}")
    assert resolved["base_url"] == ENDPOINT
    assert resolved["api_key"] == ENTRY_KEY


@pytest.mark.parametrize("name,builtin_env", SHADOWED_BUILTINS)
def test_shadowing_entry_frozen_policy_sends_its_key_env(tmp_path, monkeypatch, name, builtin_env):
    """The session's frozen route ``{provider: custom:<name>, base_url, key_env}``."""
    _home(tmp_path, monkeypatch, name, builtin_env)
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_custom import _resolve_named_custom_runtime

    frozen = rp.load_config()
    resolved = _resolve_named_custom_runtime(requested_provider=f"custom:{name}", config=frozen)
    assert (resolved["base_url"], resolved["api_key"]) == (ENDPOINT, ENTRY_KEY)
    with rp.frozen_runtime_config(frozen):
        resolved = rp.resolve_runtime_provider(requested=f"custom:{name}")
    assert (resolved["base_url"], resolved["api_key"]) == (ENDPOINT, ENTRY_KEY)


@pytest.mark.parametrize("name,builtin_env", SHADOWED_BUILTINS)
@pytest.mark.parametrize("entry_key_set", [True, False])
def test_shadowing_entry_auxiliary_client_never_borrows_builtin_key(tmp_path, monkeypatch, name, builtin_env,
                                                                    entry_key_set):
    """Aux reads key_env first; with key_env declared but unset it must not fall back to the
    built-in's env-seeded pool row (the placeholder + warning is the documented outcome)."""
    _home(tmp_path, monkeypatch, name, builtin_env)
    if not entry_key_set:
        monkeypatch.delenv("ENTRY_KEY_VAR")
    from agent.auxiliary_client import _named_custom_api_key
    from hermes_cli.runtime_provider import _get_named_custom_provider

    entry = _get_named_custom_provider(f"custom:{name}")
    expected = ENTRY_KEY if entry_key_set else "no-key-required"
    assert _named_custom_api_key(entry, name, ENDPOINT) == expected


def test_shadowing_entry_unset_key_env_never_borrows_builtin_key(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    monkeypatch.delenv("ENTRY_KEY_VAR")
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="custom:deepseek")
    assert resolved["base_url"] == ENDPOINT
    assert resolved["api_key"] == "no-key-required"


def test_shadowing_entry_inline_api_key_beats_builtin_env(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY", entry_extra={"api_key": ENTRY_KEY})
    from hermes_cli import runtime_provider as rp

    assert rp.resolve_runtime_provider(requested="custom:deepseek")["api_key"] == ENTRY_KEY


def test_shadowing_entry_manual_pool_row_still_wins(tmp_path, monkeypatch):
    """``hermes auth add deepseek`` rows are the operator's credential for the slug and keep
    their precedence over key_env, exactly like a non-shadowing keyed entry."""
    manual = {"id": "m1", "label": "manual", "auth_type": "api_key", "priority": 0, "source": "manual",
              "access_token": MANUAL_KEY}
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY", pool={"deepseek": [manual]})
    from hermes_cli import runtime_provider as rp

    for _ in range(3):
        resolved = rp.resolve_runtime_provider(requested="custom:deepseek")
        assert resolved["api_key"] == MANUAL_KEY
        pool = resolved["credential_pool"]
        # The rotation pool must not reach the built-in's env-seeded row either.
        assert {e.access_token for e in pool.entries()} == {MANUAL_KEY}
    # The built-in's own pool keeps its env-seeded row on disk.
    from agent.credential_pool import load_pool

    assert BUILTIN_KEY in {e.access_token for e in load_pool("deepseek").entries()}


def test_entry_without_key_env_keeps_existing_resolution(tmp_path, monkeypatch):
    """No key_env / api_key / key_cmd: resolution is unchanged by this fix (pool first, then
    host-gated env, then the keyless placeholder)."""
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY", entry_extra={})
    from hermes_cli import runtime_provider as rp

    assert rp.resolve_runtime_provider(requested="custom:deepseek")["api_key"] == BUILTIN_KEY


def test_non_shadowing_entry_sends_its_key_env(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "mydeep", "DEEPSEEK_API_KEY")
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="custom:mydeep")
    assert (resolved["base_url"], resolved["api_key"]) == (ENDPOINT, ENTRY_KEY)


# ── every later pool acquisition keeps the entry's ownership ───────────────────────────────────

BUILTIN_HOST = "https://api.deepseek.com/v1"


def _edit_config(home, *, model=None, entry=None):
    cfg = yaml.safe_load((home / "config.yaml").read_text())
    cfg["model"].update(model or {})
    cfg["providers"]["deepseek"].update(entry or {})
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.mark.parametrize("explicit", [False, True], ids=["model_base_url", "explicit_base_url"])
def test_url_matched_bare_custom_sends_the_entry_key(tmp_path, monkeypatch, explicit):
    """Bare ``provider: custom`` whose URL matches ``providers.deepseek`` gets the entry's key_env."""
    home = _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    _edit_config(home, model={"provider": "custom", "base_url": ENDPOINT})
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="custom", **({"explicit_base_url": ENDPOINT} if explicit else {}))
    assert (resolved["base_url"], resolved["api_key"]) == (ENDPOINT, ENTRY_KEY)


@pytest.mark.parametrize("explicit", [False, True], ids=["model_base_url", "explicit_base_url"])
def test_url_matched_bare_custom_frozen_route_sends_the_entry_key(tmp_path, monkeypatch, explicit):
    home = _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    _edit_config(home, model={"provider": "custom", "base_url": ENDPOINT})
    from hermes_cli import runtime_provider as rp

    frozen = rp.load_config()
    # A live edit after the session froze its route must not change who owns the endpoint.
    _edit_config(home, entry={"key_env": "OTHER_UNSET_VAR"})
    with rp.frozen_runtime_config(frozen):
        resolved = rp.resolve_runtime_provider(requested="custom", **({"explicit_base_url": ENDPOINT} if explicit else {}))
    assert (resolved["base_url"], resolved["api_key"]) == (ENDPOINT, ENTRY_KEY)


def test_url_matched_entry_without_declared_credential_keeps_existing_resolution(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY", entry_extra={})
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="custom", explicit_base_url=ENDPOINT)
    assert resolved["api_key"] == BUILTIN_KEY


@pytest.mark.parametrize("requested", ["custom:deepseek", "custom"])
def test_declared_key_env_unset_never_falls_back_to_the_builtin_host_key(tmp_path, monkeypatch, caplog, requested):
    """On the built-in's own host the host-derived ``DEEPSEEK_API_KEY`` is another account's key."""
    home = _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    _edit_config(home, entry={"api": BUILTIN_HOST})
    monkeypatch.delenv("ENTRY_KEY_VAR")
    from hermes_cli import runtime_provider as rp

    with caplog.at_level("WARNING"):
        resolved = rp.resolve_runtime_provider(requested=requested, explicit_base_url=BUILTIN_HOST if requested == "custom" else None)
    assert resolved["api_key"] == "no-key-required"
    warning = next(r.getMessage() for r in caplog.records if "ENTRY_KEY_VAR" in r.getMessage())
    assert "no-key-required" in warning


def test_entry_owns_pool_rows_seeded_from_its_own_key_env(tmp_path, monkeypatch):
    """Ownership is by provenance: an ``env:<the entry's key_env>`` row (#125447 seeds those into the
    slug pool) is the entry's own and keeps its rotation state; the built-in's env row is not."""
    row = {"id": "e1", "label": "ENTRY_KEY_VAR", "auth_type": "api_key", "priority": 0,
           "source": "env:ENTRY_KEY_VAR", "access_token": ENTRY_KEY, "base_url": ENDPOINT}
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY", pool={"deepseek": [row]})
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested="custom:deepseek")
    assert resolved["api_key"] == ENTRY_KEY
    pool = resolved.get("credential_pool")
    assert pool is not None and [e.source for e in pool.entries()] == ["env:ENTRY_KEY_VAR"]


def _restorable_agent(primary: dict):
    """An agent on a cross-provider fallback (OpenRouter pool attached) whose snapshot is ``primary``."""
    from unittest.mock import MagicMock

    from agent.credential_pool import CredentialPool
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider, agent.base_url = "fallback-model", "openrouter", "https://openrouter.ai/api/v1"
    agent.api_mode, agent.api_key = "chat_completions", "fallback-key-0123456789"
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    agent._credential_pool = CredentialPool("openrouter", [])
    agent._fallback_activated, agent._fallback_index, agent._rate_limited_until = True, 1, 0
    agent._use_prompt_caching = agent._use_native_cache_layout = False
    agent.context_compressor = MagicMock()
    agent._primary_runtime = {
        "model": "m", "api_mode": "chat_completions", **primary,
        "client_kwargs": {"api_key": primary["api_key"], "base_url": primary["base_url"]},
        "use_prompt_caching": False, "use_native_cache_layout": False, "compressor_model": "m",
        "compressor_base_url": primary["base_url"], "compressor_api_key": primary["api_key"],
        "compressor_provider": primary["provider"], "compressor_context_length": 128000,
        "compressor_threshold_tokens": 0.8,
    }
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._apply_client_headers_for_base_url = MagicMock()
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    return agent


def test_restoring_the_primary_after_fallback_keeps_the_entry_endpoint_and_key(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    from hermes_cli import runtime_provider as rp

    initial = rp.resolve_runtime_provider(requested="custom:deepseek")
    assert initial["api_key"] == ENTRY_KEY
    agent = _restorable_agent({"provider": initial["provider"], "requested_provider": "custom:deepseek",
                               "base_url": ENDPOINT, "api_key": ENTRY_KEY})
    assert agent._restore_primary_runtime() is True
    assert (agent.api_key, agent.base_url) == (ENTRY_KEY, ENDPOINT)
    assert agent._credential_pool is None


def test_restoring_a_builtin_primary_still_reselects_its_own_pool(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    agent = _restorable_agent({"provider": "deepseek", "requested_provider": "deepseek",
                               "base_url": BUILTIN_HOST, "api_key": "stale-snapshot-key-0123"})
    assert agent._restore_primary_runtime() is True
    assert agent.api_key == BUILTIN_KEY


def test_delegated_child_never_leases_the_builtin_account(tmp_path, monkeypatch):
    """Entry on DeepSeek's own endpoint with a separately declared account key."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    home = _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY")
    _edit_config(home, entry={"api": BUILTIN_HOST})
    from hermes_cli import runtime_provider as rp
    from tools.delegate_tool_child_run import _lease_child_credential
    from tools.delegate_tool_config import _resolve_child_credential_pool

    resolved = rp.resolve_runtime_provider(requested="custom:deepseek")
    assert resolved["api_key"] == ENTRY_KEY
    parent = SimpleNamespace(provider="custom", requested_provider="custom:deepseek", base_url=BUILTIN_HOST,
                             _credential_pool=resolved.get("credential_pool"))
    child = SimpleNamespace(base_url=BUILTIN_HOST, _swap_credential=MagicMock(),
                            _credential_pool=_resolve_child_credential_pool("custom", parent, BUILTIN_HOST, "custom:deepseek"))
    pool, lease_id = _lease_child_credential(child)
    if lease_id is not None:
        pool.release_lease(lease_id)
    assert [c.args[0].runtime_api_key for c in child._swap_credential.call_args_list] == []


@pytest.mark.parametrize("name,builtin_env", SHADOWED_BUILTINS)
def test_explicit_builtin_pick_keeps_its_env_key(tmp_path, monkeypatch, name, builtin_env):
    _home(tmp_path, monkeypatch, name, builtin_env)
    from hermes_cli import runtime_provider as rp

    resolved = rp.resolve_runtime_provider(requested=name)
    assert resolved["api_key"] == BUILTIN_KEY
    assert resolved["base_url"] != ENDPOINT


def test_persisting_the_entry_view_keeps_the_builtin_rows_and_writes_no_secret(tmp_path, monkeypatch):
    manual = {"id": "m1", "label": "manual", "auth_type": "api_key", "priority": 0, "source": "manual",
              "access_token": MANUAL_KEY}
    home = _home(tmp_path, monkeypatch, "deepseek", "DEEPSEEK_API_KEY", pool={"deepseek": [manual]})
    from agent.credential_pool import load_pool
    from hermes_cli import runtime_provider as rp

    load_pool("deepseek")
    before = {r["id"] for r in json.loads((home / "auth.json").read_text())["credential_pool"]["deepseek"]}
    rp.resolve_runtime_provider(requested="custom:deepseek")["credential_pool"]._persist()
    text = (home / "auth.json").read_text()
    after = json.loads(text)["credential_pool"]["deepseek"]
    assert {r["id"] for r in after} == before
    assert any(r["source"] == "env:DEEPSEEK_API_KEY" for r in after)
    assert BUILTIN_KEY not in text and ENTRY_KEY not in text


# ── review round 2 ──────────────────────────────────────────────────────────────────────────────

SIBLING_KEY = "sibling-key-C-0123456789"


def _two_entries_on_one_url(tmp_path, monkeypatch):
    """``alpha`` (with a ``hermes auth add alpha`` row) and ``beta`` share ENDPOINT; beta is the primary."""
    home = _home(tmp_path, monkeypatch, "alpha", "ALPHA_KEY_VAR", entry_extra={"key_env": "ALPHA_KEY_VAR"},
                 pool={"alpha": [{"id": "m1", "label": "manual", "auth_type": "api_key", "priority": 0,
                                  "source": "manual", "access_token": MANUAL_KEY}]})
    cfg = yaml.safe_load((home / "config.yaml").read_text())
    cfg["providers"]["beta"] = {**cfg["providers"]["alpha"], "name": "beta", "key_env": "ENTRY_KEY_VAR"}
    cfg["model"]["provider"] = "custom:beta"
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("ALPHA_KEY_VAR", SIBLING_KEY)
    return home


def test_restoring_a_named_entry_never_resumes_on_a_sibling_sharing_its_url(tmp_path, monkeypatch):
    _two_entries_on_one_url(tmp_path, monkeypatch)
    from agent.custom_entry_pool import load_runtime_pool
    from hermes_cli import runtime_provider as rp

    pool = load_runtime_pool("custom", ENDPOINT, "custom:beta")
    assert pool is None or not {e.access_token for e in pool.entries()} & {MANUAL_KEY, SIBLING_KEY}
    initial = rp.resolve_runtime_provider(requested="custom:beta")
    assert initial["api_key"] == ENTRY_KEY
    agent = _restorable_agent({"provider": initial["provider"], "requested_provider": "custom:beta",
                               "base_url": ENDPOINT, "api_key": ENTRY_KEY})
    assert agent._restore_primary_runtime() is True
    assert (agent.api_key, agent.base_url) == (ENTRY_KEY, ENDPOINT)


def test_a_genuinely_bare_custom_snapshot_still_takes_the_first_entry_on_its_url(tmp_path, monkeypatch):
    _two_entries_on_one_url(tmp_path, monkeypatch)
    from agent.custom_entry_pool import load_runtime_pool

    assert MANUAL_KEY in {e.access_token for e in load_runtime_pool("custom", ENDPOINT).entries()}
