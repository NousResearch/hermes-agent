"""A runtime fallback must not resurrect terminally dead pool grants."""

import json
from pathlib import Path

import pytest

from hermes_cli.auth import AuthError, resolve_xai_oauth_runtime_credentials


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def write_store(home, entries, providers=None):
    path = home / "auth.json"
    path.write_text(json.dumps({
        "version": 1, "providers": providers or {},
        "credential_pool": {"xai-oauth": entries},
    }), encoding="utf-8")
    return path


def entry(token, status=None):
    return {"id": token, "source": "manual:oauth", "auth_type": "oauth",
            "access_token": token, "refresh_token": token + "-refresh",
            "last_status": status}


@pytest.mark.parametrize("reset_at", [None, 1, 4102444800])
def test_dead_only_pool_fails_closed(home, reset_at):
    dead = dict(entry("revoked", "dead"), last_error_reset_at=reset_at)
    path = write_store(home, [dead])
    before = path.read_bytes()
    with pytest.raises(AuthError) as exc:
        resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
    assert exc.value.provider == "xai-oauth"
    assert path.read_bytes() == before


def test_dead_entry_does_not_shadow_healthy_entry(home):
    path = write_store(home, [entry("revoked", "dead"), entry("healthy", "ok")])
    before = path.read_bytes()
    result = resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
    assert result["api_key"] == "healthy"
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", [None, "ok", "exhausted", "DEAD", " dead ", "unknown"])
def test_other_statuses_retain_existing_fallback_behavior(home, status):
    write_store(home, [entry("available", status)])
    assert resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "available"


def test_singleton_precedence_is_unchanged(home):
    write_store(home, [entry("revoked", "dead")], {
        "xai-oauth": {"tokens": {"access_token": "singleton", "refresh_token": "refresh"}},
    })
    assert resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)["api_key"] == "singleton"


def test_empty_pool_still_raises_auth_error(home):
    write_store(home, [])
    with pytest.raises(AuthError):
        resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False)
