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


def test_unrecorded_legacy_row_is_bound_by_the_move_guard_even_if_never_used(client):
    from hermes_constants import get_hermes_home
    _entry()
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
