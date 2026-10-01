"""Pools consume explicit application inputs and retain their profile owner."""
from __future__ import annotations

from contextlib import contextmanager
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from auth.context import CredentialScope
from auth.credential_pool import CredentialPool, PooledCredential, load_pool
from auth.errors import AuthError
from auth.pool_environment import PoolEnvironment, PoolProviderHooks
from auth.pool_persistence import read_credential_pool
from agent.secret_scope import (
    is_multiplex_active, reset_secret_scope, set_multiplex_active, set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def multiplex_scope():
    previous = is_multiplex_active()
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        yield
    finally:
        reset_secret_scope(token)
        set_multiplex_active(previous)


@contextmanager
def profile_scope(home):
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({}, profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def environment(home, *, config=None, env=None, hooks=None):
    return PoolEnvironment(
        scope=CredentialScope(home),
        read_config=lambda: config or {},
        custom_providers=lambda cfg: cfg.get("custom_providers", []),
        read_env=lambda: env or {},
        secret_source=lambda name: "provided-source",
        provider_config=lambda name: SimpleNamespace(
            auth_type="api_key", api_key_env_vars=("SYNTHETIC_KEY",),
            inference_base_url="https://synthetic.invalid/v1", base_url_env_var=""),
        provider_configured=lambda provider: False,
        key_endpoint=lambda provider, token, fallback, override: override or fallback,
        normalize_endpoint=lambda value: str(value or "").rstrip("/"),
        provider_hooks=lambda provider: hooks or PoolProviderHooks(),
    )


def entry(provider="synthetic"):
    return PooledCredential(
        provider=provider, id="owned", label="owned", auth_type="api_key",
        priority=0, source="manual", access_token="owned-secret",
        base_url="https://synthetic.invalid/v1",
    )


def test_settings_and_secret_sources_are_supplied_by_the_application(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    supplied = environment(
        tmp_path, config={"credential_pool_strategies": {"synthetic": "least_used"}},
        env={"SYNTHETIC_KEY": "supplied-secret"},
    )
    pool = load_pool("synthetic", environment=supplied)
    selected = pool.select()
    assert pool._strategy == "least_used"
    assert selected.runtime_api_key == "supplied-secret"
    assert selected.extra["secret_source"] == "provided-source"
    disk = read_credential_pool("synthetic")
    assert disk[0].get("access_token") in (None, "")
    assert "supplied-secret" not in (tmp_path / "auth.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("operation", [
    lambda pool: pool.select(),
    lambda pool: pool.remove_index(1),
    lambda pool: pool.reset_statuses(),
    lambda pool: pool.move_entry("owned", 0),
    lambda pool: pool.acquire_lease(),
    lambda pool: pool.token_is_blocked("owned-secret"),
])
def test_retained_pool_refuses_another_profile_then_recovers(tmp_path, operation):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with profile_scope(a):
        pool = CredentialPool("synthetic", [entry()], environment=environment(a))
        pool._persist()
        before_a = (a / "auth.json").read_bytes()
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            operation(pool)
        assert not (b / "auth.json").exists()
        assert (a / "auth.json").read_bytes() == before_a
    with profile_scope(a):
        assert pool.select().runtime_api_key == "owned-secret"


def test_load_rejects_wrong_scope_before_any_source_callback(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    supplied = replace(environment(a), read_env=lambda: pytest.fail("source was read"))
    with profile_scope(b):
        with pytest.raises(ValueError, match="different profile"):
            load_pool("openai-codex", environment=supplied)
    assert not (a / "auth.json").exists()
    assert not (b / "auth.json").exists()


def test_pool_invokes_injected_refresh_and_persists_the_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []
    def refresh(access, refresh_token):
        calls.append((access, refresh_token))
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh"}
    hooks = PoolProviderHooks(
        refresh_tokens=refresh, terminal_error=lambda error: False,
        token_expiring=lambda token: False,
    )
    original = replace(entry("openai-codex"), auth_type="oauth", refresh_token="original-refresh")
    pool = CredentialPool("openai-codex", [original], environment=environment(tmp_path, hooks=hooks))
    pool._persist()
    rotated = pool.try_refresh_matching(credential_id="owned")
    assert calls == [("owned-secret", "original-refresh")]
    assert rotated.access_token == "rotated-access"
    assert read_credential_pool("openai-codex")[0]["refresh_token"] == "rotated-refresh"


def test_injected_terminal_failure_keeps_a_manual_grant_out_of_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def refresh(*args):
        raise AuthError("revoked", provider="openai-codex", code="invalid_grant", relogin_required=True)
    hooks = PoolProviderHooks(
        refresh_tokens=refresh, terminal_error=lambda error: error.relogin_required,
        token_expiring=lambda token: False,
    )
    original = replace(entry("openai-codex"), auth_type="oauth", refresh_token="revoked-refresh")
    pool = CredentialPool("openai-codex", [original], environment=environment(tmp_path, hooks=hooks))
    pool._persist()
    assert pool.try_refresh_matching(credential_id="owned") is None
    assert pool.entries()[0].last_status == "dead"
    assert pool.select() is None


def test_pool_import_and_manual_selection_do_not_load_cli_or_discover_providers(tmp_path):
    code = r'''
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"hermes_cli", "nous_cli", "providers"}:
            raise AssertionError("unexpected dependency: " + fullname)
sys.meta_path.insert(0, Block())
from auth.credential_pool import CredentialPool, PooledCredential
from auth.context import CredentialScope
from auth.pool_environment import PoolEnvironment, PoolProviderHooks
from hermes_constants import get_hermes_home
env = PoolEnvironment(CredentialScope(get_hermes_home()), lambda: {}, lambda c: [],
    lambda: {}, lambda n: None, lambda p: None, lambda p: False,
    lambda p,t,f,o: o or f, lambda u: str(u), lambda p: PoolProviderHooks())
row = PooledCredential("synthetic","owned","owned","api_key",0,"manual","secret")
assert CredentialPool("synthetic", [row], environment=env).select().id == "owned"
'''
    import os
    child_env = dict(os.environ, HERMES_HOME=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code], cwd=Path(__file__).resolve().parents[2],
        env=child_env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_documented_external_plugin_imports_remain_usable(tmp_path, monkeypatch):
    import importlib
    from auth.credential_pool import AUTH_TYPE_OAUTH
    from hermes_cli.plugin_compat import load_manifest, scan_source

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pool = CredentialPool("synthetic", [entry()], environment=environment(tmp_path))
    pool._persist()
    legacy = importlib.import_module("agent.credential_pool")
    assert legacy.PooledCredential is PooledCredential
    assert legacy.AUTH_TYPE_OAUTH == AUTH_TYPE_OAUTH
    assert legacy.load_pool("synthetic").select().id == "owned"
    assert importlib.import_module("hermes_cli.auth_constants").AuthError is AuthError

    plugin_source = (
        "from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool\n"
        "from hermes_cli.auth_constants import AuthError\n"
    )
    assert scan_source(plugin_source, "provider.py", load_manifest()) == []
