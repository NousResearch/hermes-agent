"""A pooled key for a custom endpoint is bound to the origin it was entered for (#130410).

Every acquisition of an entry's pool (main runtime, auxiliary clients, a delegated child, the
primary restored after a fallback) takes the same origin-bound view, so a key added through
``hermes auth add`` or the dashboard, or a built-in's seeded account, never follows the entry to a
new host.
"""

import json

import pytest

OLD, NEW = "https://llm.old.example/v1", "https://llm.new.example/v1"
POOLED = "sk-pooled-old-host-0123456789"
BUILTIN = "sk-deepseek-builtin-0123456789"


@pytest.fixture
def client(_isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _entry(name="myep", **fields):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg["providers"] = {name: {"name": name, "base_url": OLD, "model": "m", "models": {"m": {}}, **fields}}
    save_config(cfg)


def _move(client, name="myep", **extra):
    return client.post("/api/providers/custom-endpoints", json={
        "id": name, "name": name, "base_url": NEW, "model": "m", "make_default": False, **extra})


def _main_key(requested="custom:myep"):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    return str(resolve_runtime_provider(requested=requested).get("api_key"))


def _bare_key_at_new():
    from hermes_cli.config import load_config, save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    cfg = load_config()
    cfg["model"] = {"provider": "custom", "base_url": NEW, "default": "m"}
    save_config(cfg)
    rt = resolve_runtime_provider()
    assert rt["base_url"] == NEW
    return str(rt.get("api_key"))


def _aux_key(provider="custom:myep"):
    from agent import auxiliary_client as ac
    ac._client_cache.clear()
    client, _model = ac.resolve_provider_client(provider, "m")
    return str(client.api_key)


def _child_pool_keys(base_url, requested="custom:myep"):
    from types import SimpleNamespace

    from tools.delegate_tool_config import _resolve_child_credential_pool
    parent = SimpleNamespace(provider="openrouter", requested_provider="openrouter", base_url="", _credential_pool=None)
    pool = _resolve_child_credential_pool("custom", parent, base_url, requested)
    return {e.access_token for e in pool.entries()} if pool is not None else set()


def _restore_pool_keys(base_url, requested="custom:myep"):
    from agent.custom_entry_pool import load_runtime_pool
    pool = load_runtime_pool("custom", base_url, requested)
    return {e.access_token for e in pool.entries()} if pool is not None else set()


def _stored_rows(key):
    from hermes_constants import get_hermes_home
    return json.loads((get_hermes_home() / "auth.json").read_text())["credential_pool"][key]


def test_dashboard_added_key_stays_on_its_origin_after_a_desktop_move(client):
    _entry()
    assert client.post("/api/credentials/pool", json={"provider": "myep", "api_key": POOLED}).status_code == 200
    assert [r.get("base_url") for r in _stored_rows("myep")] == [OLD]
    assert _main_key() == POOLED

    # The pooled key counts as the entry's credential: an untouched key field refuses the move.
    assert _move(client).status_code == 400
    assert _move(client, api_key="").status_code == 200

    assert POOLED not in {_main_key(), _aux_key()}
    assert POOLED not in _child_pool_keys(NEW) | _restore_pool_keys(NEW)
    assert _bare_key_at_new() != POOLED


@pytest.mark.parametrize("name", ["openrouter", "deepseek"])
def test_key_added_for_a_builtin_named_proxy_entry_records_the_proxy(client, name):
    """The migration note's remedy: a key added while ``providers.<built-in>`` points at a proxy is
    the proxy's, for OpenRouter too."""
    _entry(name)
    assert client.post("/api/credentials/pool", json={"provider": name, "api_key": POOLED}).status_code == 200
    assert [r.get("base_url") for r in _stored_rows(name) if r.get("source") == "manual"] == [OLD]
    assert _main_key(f"custom:{name}") == POOLED


def test_anthropic_oauth_token_is_bound_to_anthropic_even_with_a_proxy_entry(_isolate_hermes_home, monkeypatch):
    """The Anthropic browser login mints the token at Anthropic: a providers.anthropic proxy never gets it."""
    from types import SimpleNamespace

    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_commands import auth_add_command
    _entry("anthropic")
    monkeypatch.setattr("agent.anthropic_credentials.run_hermes_oauth_login_pure",
                        lambda *_a, **_k: {"access_token": POOLED, "refresh_token": "r", "expires_at_ms": 4102444800000})
    auth_add_command(SimpleNamespace(provider="anthropic", auth_type="oauth", api_key=None,
                                     label="browser", timeout=None, no_browser=True))
    assert [r.get("base_url") for r in _stored_rows("anthropic") if r.get("source") == "manual:hermes_pkce"] == [
        PROVIDER_REGISTRY["anthropic"].inference_base_url]
    assert _main_key("custom:anthropic") != POOLED


def test_openrouter_pkce_key_is_bound_to_openrouter_even_with_a_proxy_entry(_isolate_hermes_home, monkeypatch):
    """The browser login mints the key at openrouter.ai: it is never the proxy's key."""
    from types import SimpleNamespace

    from hermes_cli.auth_commands import auth_add_command
    from hermes_constants import OPENROUTER_BASE_URL
    _entry("openrouter")
    monkeypatch.setattr("hermes_cli.auth._openrouter_pkce_login", lambda **_kw: {"api_key": POOLED})
    auth_add_command(SimpleNamespace(provider="openrouter", auth_type="oauth", api_key=None,
                                     label="browser", timeout=None, no_browser=True))
    assert [r.get("base_url") for r in _stored_rows("openrouter")
            if r.get("source") == "manual:openrouter_pkce"] == [OPENROUTER_BASE_URL]


def test_cli_added_key_reaches_no_aux_delegate_or_restore_path_after_a_move(client):
    from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool
    _entry()
    load_pool("myep").add_entry(PooledCredential(
        provider="myep", id="cli001", label="k", auth_type=AUTH_TYPE_API_KEY, priority=0,
        source=SOURCE_MANUAL, access_token=POOLED, base_url=OLD))
    # Positive control: on its own origin every path still uses the pooled key.
    assert _main_key() == _aux_key() == POOLED
    assert POOLED in _child_pool_keys(OLD) & _restore_pool_keys(OLD)

    assert _move(client, api_key="").status_code == 200
    assert POOLED not in {_main_key(), _aux_key()}
    assert POOLED not in _child_pool_keys(NEW) | _restore_pool_keys(NEW)


def test_unrecorded_legacy_row_is_bound_to_the_entry_address_and_left_behind_by_a_move(client):
    """A manual row written before writers recorded an address is bound to the entry's address on
    first use (persisted once), and the move guard treats it as the old host's credential."""
    from hermes_constants import get_hermes_home
    _entry()
    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "myep": [{"id": "leg001", "label": "old", "auth_type": "api_key", "priority": 0,
                  "source": "manual", "access_token": POOLED}]}}), encoding="utf-8")

    assert _main_key() == POOLED
    assert [r.get("base_url") for r in _stored_rows("myep")] == [OLD]
    assert _move(client).status_code == 400
    assert _move(client, api_key="").status_code == 200
    assert POOLED not in {_main_key(), _aux_key(), _bare_key_at_new()}


@pytest.mark.parametrize("stored", [{}, {"extra_headers": {"X-Org": "a"}}, {"key_env": "MYEP_OLD_KEY"}])
def test_unrecorded_legacy_row_is_bound_by_the_move_guard_even_if_never_used(client, stored):
    """Config credentials on the entry must not skip the binding, or the row follows the move."""
    from hermes_constants import get_hermes_home
    _entry(**stored)
    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "myep": [{"id": "leg002", "label": "old", "auth_type": "api_key", "priority": 0,
                  "source": "manual", "access_token": POOLED}]}}), encoding="utf-8")
    assert _move(client, api_key="").status_code == 200
    assert [r.get("base_url") for r in _stored_rows("myep")] == [OLD]
    assert POOLED not in {_main_key(), _aux_key()}


def test_builtin_seeded_row_never_follows_a_credential_less_moved_entry(client, monkeypatch):
    from hermes_cli.config import save_env_value
    _entry("deepseek", key_env="DEEPSEEK_API_KEY")
    save_env_value("DEEPSEEK_API_KEY", BUILTIN)
    monkeypatch.setenv("DEEPSEEK_API_KEY", BUILTIN)
    assert _main_key("custom:deepseek") == BUILTIN  # its own key_env, on its own address

    assert _move(client, "deepseek", api_key="").status_code == 200
    assert BUILTIN not in {_main_key("custom:deepseek"), _aux_key("custom:deepseek")}
    assert BUILTIN not in _child_pool_keys(NEW, "custom:deepseek") | _restore_pool_keys(NEW, "custom:deepseek")


def _add_row(key="myep", base_url=OLD):
    from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool
    load_pool(key).add_entry(PooledCredential(
        provider=key, id="row001", label="k", auth_type=AUTH_TYPE_API_KEY, priority=0,
        source=SOURCE_MANUAL, access_token=POOLED, base_url=base_url))


def _set(**sections):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.update(sections)
    save_config(cfg)


def _cached_aux_client(task="compression"):
    from agent import auxiliary_client as ac
    ac._client_cache.clear()
    provider, model, base_url, api_key, api_mode = ac._resolve_task_provider_model(task)
    client, _ = ac._get_cached_client(provider, model, base_url=base_url, api_key=api_key, api_mode=api_mode)
    return str(client.base_url).rstrip("/"), str(client.api_key)


@pytest.mark.parametrize("main_provider,task_provider", [("myep", "main"), ("custom", "myep"), ("myep", "myep")])
def test_bare_slug_identity_never_takes_the_old_origin_key_after_a_move(client, main_provider, task_provider):
    """Desktop's activate writes ``model.provider: <key>``; an aux task or a fallback may name the
    slug too. Every one of them is the entry's identity, bound like ``custom:<name>``."""
    _entry()
    _add_row()
    assert _move(client, api_key="").status_code == 200
    _set(model={"provider": main_provider, "default": "m", "base_url": NEW},
         auxiliary={"compression": {"provider": task_provider, "model": "m"}})

    assert _cached_aux_client() == (NEW, "no-key-required")


def test_bare_slug_fallback_switch_and_cron_pools_drop_the_old_origin_row(client):
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _rebind_fallback_credential_pool
    from agent.custom_entry_pool import load_route_pool, pool_for_aux_provider
    from agent.credential_pool import load_pool
    _entry()
    _add_row()
    # Positive control: on its own origin the slug still rotates through the pooled key.
    assert POOLED in {e.access_token for e in load_route_pool("myep", OLD).entries()}
    from cron.scheduler import _load_credential_pool
    assert POOLED in {e.access_token for e in _load_credential_pool({"provider": "myep", "base_url": OLD}, "job").entries()}
    assert _move(client, api_key="").status_code == 200

    agent = SimpleNamespace(_credential_pool=None, _credential_pool_entry_id=None, base_url=NEW, provider="myep")
    _rebind_fallback_credential_pool(agent, "myep", "m")
    assert agent._credential_pool is None
    assert load_route_pool("myep", NEW) is None
    assert _load_credential_pool({"provider": "myep", "base_url": NEW, "requested_provider": "myep"}, "job") is None
    assert pool_for_aux_provider("myep", "myep", load_pool) is None
    # A delegated child on ``delegation.provider: myep`` (provider id kept as the bare slug).
    from tools.delegate_tool_config import _resolve_child_credential_pool
    parent = SimpleNamespace(provider="openrouter", requested_provider="openrouter", base_url="", _credential_pool=None)
    child = _resolve_child_credential_pool("myep", parent, NEW, "myep")
    assert child is None or POOLED not in {e.access_token for e in child.entries()}


def test_switch_model_onto_a_moved_bare_slug_reloads_no_old_origin_row(client, monkeypatch):
    """``/model`` mid-session reloads the pool for the new route: a bare slug is the entry's identity."""
    from types import SimpleNamespace

    from agent import agent_runtime_helpers as arh
    monkeypatch.setattr(arh, "_build_switched_client", lambda *_a, **_k: None)
    monkeypatch.setattr(arh, "sync_credential_pool_entry_id", lambda *_a, **_k: None)
    _entry()
    _add_row()

    def _switched_pool(base_url):
        agent = SimpleNamespace(_credential_pool=None, _credential_pool_entry_id=None,
                                _read_reasoning_echo_from_config=lambda: False)
        arh._swap_switch_runtime(agent, "m", "myep", "", base_url, "chat_completions", "openrouter", "openrouter", "myep")
        return {e.access_token for e in agent._credential_pool.entries()} if agent._credential_pool else set()

    assert POOLED in _switched_pool(OLD)
    assert _move(client, api_key="").status_code == 200
    assert POOLED not in _switched_pool(NEW)


@pytest.mark.parametrize("provider", ["custom:myep", "myep"])
def test_aux_task_base_url_override_never_receives_the_entry_origin_key(_isolate_hermes_home, provider):
    from agent.credential_pool import load_pool
    from agent.custom_entry_pool import pool_for_aux_provider
    _entry()
    _add_row()
    _set(auxiliary={"compression": {"provider": provider, "model": "m", "base_url": NEW}})

    assert _cached_aux_client() == (NEW, "no-key-required")
    # Credential recovery for a request sent to NEW rotates nothing bound to OLD.
    assert pool_for_aux_provider(provider, "myep", load_pool, NEW) is None
    assert POOLED in {e.access_token for e in pool_for_aux_provider(provider, "myep", load_pool, OLD).entries()}


def test_resolving_a_same_named_entry_never_moves_the_builtin_pool_to_the_entry_host(_isolate_hermes_home, monkeypatch):
    """A built-in's manual row with no recorded address belongs to the built-in's own address: a
    ``providers.deepseek`` proxy entry neither stamps it with the proxy URL nor sends it there."""
    from hermes_constants import get_hermes_home
    from hermes_cli.runtime_provider import resolve_runtime_provider
    proxy = "https://proxy.example/v1"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "deepseek": [{"id": "dash01", "label": "dash", "auth_type": "api_key", "priority": 0,
                      "source": "manual", "access_token": BUILTIN}]}}), encoding="utf-8")
    builtin = resolve_runtime_provider(requested="deepseek")
    assert (builtin["base_url"], builtin["api_key"]) == ("https://api.deepseek.com/v1", BUILTIN)

    _entry("deepseek", base_url=proxy)
    entry = resolve_runtime_provider(requested="custom:deepseek")
    assert (entry["base_url"], entry["api_key"]) == (proxy, "no-key-required")
    assert [r.get("base_url") for r in _stored_rows("deepseek")] == [None]

    _set(providers={})
    after = resolve_runtime_provider(requested="deepseek")
    assert (after["base_url"], after["api_key"]) == ("https://api.deepseek.com/v1", BUILTIN)


@pytest.mark.parametrize("key_env", [None, "ENTRY_KEY_VAR"])
def test_delegated_child_on_a_builtin_named_entry_leases_no_builtin_key(_isolate_hermes_home, monkeypatch, key_env):
    """``delegation.provider: deepseek`` with a ``providers.deepseek`` proxy: the child keeps the bare
    slug, whose pool is also the built-in's. Its startup lease must not bind the built-in's
    dashboard row or env key to the proxy, nor replace the entry's own key_env credential."""
    from types import SimpleNamespace

    from hermes_constants import get_hermes_home
    from tools.delegate_tool_child_run import _lease_child_credential
    from tools.delegate_tool_config import _resolve_child_credential_pool, _resolve_delegation_credentials
    proxy = "https://proxy.example/v1"
    monkeypatch.setenv("DEEPSEEK_API_KEY", BUILTIN)
    monkeypatch.setenv("ENTRY_KEY_VAR", "sk-entry-0123456789")
    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "deepseek": [{"id": "dash01", "label": "dash", "auth_type": "api_key", "priority": 0,
                      "source": "manual", "access_token": POOLED}]}}), encoding="utf-8")
    _entry("deepseek", base_url=proxy, **({"key_env": key_env} if key_env else {}))
    _set(delegation={"provider": "deepseek", "model": "m"})
    from hermes_cli.config import load_config
    parent = SimpleNamespace(provider="openrouter", requested_provider="openrouter", base_url="", _credential_pool=None, model="m")

    creds = _resolve_delegation_credentials(load_config()["delegation"], parent)
    assert (creds["provider"], creds["base_url"]) == ("deepseek", proxy)
    assert creds["api_key"] == ("sk-entry-0123456789" if key_env else "no-key-required")
    pool = _resolve_child_credential_pool(creds["provider"], parent, creds["base_url"], creds.get("requested_provider"))
    swapped = []
    child = SimpleNamespace(_credential_pool=pool, base_url=proxy, _swap_credential=lambda e: swapped.append(e.access_token))
    for _ in range(3):
        leased_pool, lease_id = _lease_child_credential(child)
        if leased_pool is not None and lease_id:
            leased_pool.release_lease(lease_id)
    assert not {POOLED, BUILTIN} & set(swapped)
