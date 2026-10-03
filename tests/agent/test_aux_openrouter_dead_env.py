"""OpenRouter auxiliary env fallback must not revive its terminal pool row."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-runtime")
    (home / "config.yaml").write_text("auxiliary:\n  free_only: false\n", encoding="utf-8")
    return home


def write_pool(home, *, status="dead", token="sk-or-test-runtime", source="env:OPENROUTER_API_KEY"):
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "openrouter": [{"id": "env", "source": source, "auth_type": "api_key",
                        "access_token": token, "last_status": status,
                        "last_error_reset_at": 4102444800 if status == "exhausted" else None}]
    }}), encoding="utf-8")


def test_dead_env_key_is_not_returned(store):
    from agent.auxiliary_client import _try_openrouter
    write_pool(store)
    client, model = _try_openrouter(model="test/model")
    try:
        assert client is None
        assert model is None
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize("status", [None, "ok", "exhausted", "DEAD", "unknown"])
def test_nonterminal_env_fallback_is_preserved(store, status):
    from agent.auxiliary_client import _try_openrouter
    write_pool(store, status=status)
    client, model = _try_openrouter(model="test/model")
    try:
        assert client.api_key == "sk-or-test-runtime"
        assert model == "test/model"
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize("token,source", [("sk-or-old", "env:OPENROUTER_API_KEY"),
                                          ("sk-or-test-runtime", "manual")])
def test_different_key_or_source_does_not_block_env(store, token, source):
    from agent.auxiliary_client import _try_openrouter
    write_pool(store, token=token, source=source)
    client, _ = _try_openrouter(model="test/model")
    try:
        assert client.api_key == "sk-or-test-runtime"
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize("failure", ["selection", "read"])
def test_lookup_exceptions_preserve_env_fallback(store, monkeypatch, failure):
    from agent import auxiliary_client as aux
    from hermes_cli import auth

    write_pool(store, status="exhausted")

    def fail(*args, **kwargs):
        raise OSError("fixture lookup failure")

    if failure == "selection":
        from agent.credential_pool import CredentialPool
        monkeypatch.setattr(CredentialPool, "select", fail)
    else:
        monkeypatch.setattr(auth, "read_credential_pool", fail)
    client, _ = aux._try_openrouter(model="test/model")
    try:
        assert client.api_key == "sk-or-test-runtime"
    finally:
        if client is not None:
            client.close()


def test_healthy_pool_selection_wins_over_dead_env(store):
    from agent.auxiliary_client import _try_openrouter
    write_pool(store)
    path = store / "auth.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["credential_pool"]["openrouter"].append({
        "id": "healthy", "source": "manual", "auth_type": "api_key",
        "access_token": "healthy-pool", "last_status": "ok", "priority": 1,
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    client, _ = _try_openrouter(model="test/model")
    try:
        assert client.api_key == "healthy-pool"
    finally:
        if client is not None:
            client.close()


def test_explicit_key_and_endpoint_remain_authoritative(store):
    from agent.auxiliary_client import _try_openrouter
    write_pool(store)
    client, _ = _try_openrouter(explicit_api_key="explicit", model="test/model",
                              explicit_base_url="https://example.invalid/v1")
    try:
        assert client.api_key == "explicit"
        assert str(client.base_url) == "https://example.invalid/v1/"
    finally:
        if client is not None:
            client.close()
