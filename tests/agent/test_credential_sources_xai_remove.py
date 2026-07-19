"""R8: hermes auth remove must surface shared-mode disable failures."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import auth
from hermes_cli.auth import AuthError


@pytest.fixture
def shared_env(tmp_path, monkeypatch):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    monkeypatch.setenv("HERMES_XAI_SHARED_AUTH", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    store = shared_dir / "xai_oauth.json"
    store.write_text(
        json.dumps(
            {
                "_schema": 1,
                "generation": 1,
                "access_token": "at",
                "refresh_token": "rt",
                "token_type": "Bearer",
                "auth_mode": "oauth_device_code",
            }
        ),
        encoding="utf-8",
    )
    return {"store": store, "profile_auth": hermes_home / "auth.json"}


def test_r8_auth_remove_propagates_disable_failure(shared_env, monkeypatch):
    """R8: disable failure must raise — not silently fall through to legacy clear."""
    from agent.credential_sources import _remove_xai_oauth_device_code

    def boom():
        raise AuthError(
            "marker write failed",
            provider="xai-oauth",
            code="xai_shared_disable_failed",
        )

    monkeypatch.setattr(auth, "disable_profile_xai_shared_auth", boom)
    removed = SimpleNamespace(source="device_code", label="xai")
    with pytest.raises(AuthError) as exc:
        _remove_xai_oauth_device_code("xai-oauth", removed)
    assert exc.value.code == "xai_shared_disable_failed"
