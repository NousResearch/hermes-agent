"""Unit tests for the xAI shared OAuth token store.

Mirrors the Nous shared-store tests in
``tests/hermes_cli/test_auth_nous_provider.py`` — same fixture conventions,
same seat-belt expectations. The shared store is what stops two profiles
that both pick ``model.provider: xai-oauth`` from racing on xAI's
single-use rotating refresh_token (which otherwise revokes the entire
token family the moment the second profile's refresh attempt hits the
already-consumed token).
"""
# pylint: disable=protected-access
from __future__ import annotations

import json
import os
import stat

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_store_env(tmp_path, monkeypatch):
    """Redirect HERMES_SHARED_AUTH_DIR to a tmp_path.

    Required for every test that touches the shared xAI store — the
    in-auth.py seat belt refuses to touch the real user's shared store
    under pytest, so tests that forget this fixture fail loudly instead
    of corrupting real state.
    """
    shared_dir = tmp_path / "shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_dir))
    return shared_dir


def _shared_payload(
    *,
    refresh_token: str = "rt-current",
    access_token: str = "at-current",
) -> dict:
    """A complete shared-store payload as the writer would emit it."""
    return {
        "_schema": 1,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "id_token": "id-tok",
        "expires_in": 3600,
        "discovery": {"token_endpoint": "https://api.x.ai/oauth2/token"},
        "redirect_uri": "http://127.0.0.1:8765/callback",
        "last_refresh": "2026-05-27T00:00:00Z",
        "updated_at": "2026-05-27T00:00:00+00:00",
    }


def _local_state(*, refresh_token: str = "rt-local") -> dict:
    """A profile-local xAI OAuth state as :func:`_read_xai_oauth_tokens` returns."""
    return {
        "tokens": {
            "access_token": "at-local",
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "id_token": "id-local",
            "expires_in": 3600,
        },
        "last_refresh": "2026-05-20T00:00:00Z",
        "discovery": {"token_endpoint": "https://api.x.ai/oauth2/token"},
        "redirect_uri": "http://127.0.0.1:8765/callback",
    }


# ---------------------------------------------------------------------------
# Seat belt + path resolution
# ---------------------------------------------------------------------------


def test_shared_store_seat_belt_refuses_real_home_under_pytest(monkeypatch):
    """Without HERMES_SHARED_AUTH_DIR override, the seat belt must trip.

    Mirrors the Nous seat belt: forgetting to redirect this store in a
    test must fail loudly instead of silently writing to the user's real
    ``~/.hermes/shared/`` across CI runs.
    """
    from hermes_cli.auth import _xai_shared_store_path

    monkeypatch.delenv("HERMES_SHARED_AUTH_DIR", raising=False)

    with pytest.raises(RuntimeError, match="shared xAI auth store"):
        _xai_shared_store_path()


def test_shared_store_honors_env_override(tmp_path, monkeypatch):
    """HERMES_SHARED_AUTH_DIR must redirect the path."""
    from hermes_cli.auth import XAI_SHARED_STORE_FILENAME, _xai_shared_store_path

    custom_dir = tmp_path / "custom_shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(custom_dir))

    path = _xai_shared_store_path()
    assert path == custom_dir / XAI_SHARED_STORE_FILENAME


# ---------------------------------------------------------------------------
# Read: missing / malformed / incomplete
# ---------------------------------------------------------------------------


def test_shared_store_read_missing_returns_none(shared_store_env):
    """Missing file → ``_read_shared_xai_state()`` returns None."""
    from hermes_cli.auth import _read_shared_xai_state

    assert _read_shared_xai_state() is None


def test_shared_store_read_malformed_returns_none(shared_store_env):
    """Non-JSON content → None, not an exception."""
    from hermes_cli.auth import _read_shared_xai_state, _xai_shared_store_path

    path = _xai_shared_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")

    assert _read_shared_xai_state() is None


def test_shared_store_read_missing_required_fields_returns_none(shared_store_env):
    """Payload without refresh_token → None (nothing worth importing)."""
    from hermes_cli.auth import _read_shared_xai_state, _xai_shared_store_path

    path = _xai_shared_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_schema": 1, "access_token": "abc"}))

    assert _read_shared_xai_state() is None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def test_shared_store_write_and_read_roundtrip(shared_store_env):
    """Write → read must preserve refresh_token + the OAuth metadata
    we'll need to skip discovery on the next refresh."""
    from hermes_cli.auth import (
        _read_shared_xai_state,
        _write_shared_xai_state,
        _xai_shared_store_path,
    )

    tokens = {
        "access_token": "at-rolled",
        "refresh_token": "rt-rolled",
        "token_type": "Bearer",
        "id_token": "id-rolled",
        "expires_in": 3600,
    }
    _write_shared_xai_state(
        tokens,
        discovery={"token_endpoint": "https://api.x.ai/oauth2/token"},
        redirect_uri="http://127.0.0.1:8765/callback",
        last_refresh="2026-05-27T07:00:00Z",
    )

    path = _xai_shared_store_path()
    assert path.is_file()

    # 0o600 where the platform supports it.
    mode = path.stat().st_mode & 0o777
    assert mode in {stat.S_IRUSR | stat.S_IWUSR, 0o644}

    loaded = _read_shared_xai_state()
    assert loaded is not None
    assert loaded["refresh_token"] == "rt-rolled"
    assert loaded["access_token"] == "at-rolled"
    assert loaded["token_type"] == "Bearer"
    assert loaded["id_token"] == "id-rolled"
    assert loaded["expires_in"] == 3600
    assert loaded["discovery"] == {
        "token_endpoint": "https://api.x.ai/oauth2/token",
    }
    assert loaded["last_refresh"] == "2026-05-27T07:00:00Z"


def test_shared_store_write_skips_when_refresh_token_missing(shared_store_env):
    """Write is a no-op when refresh_token is absent (nothing to share)."""
    from hermes_cli.auth import _write_shared_xai_state, _xai_shared_store_path

    _write_shared_xai_state(
        {"access_token": "at", "refresh_token": ""},
        last_refresh="2026-05-27T00:00:00Z",
    )

    assert not _xai_shared_store_path().is_file()


def test_shared_store_write_skips_when_access_token_missing(shared_store_env):
    """Write is a no-op when access_token is absent (incomplete pair)."""
    from hermes_cli.auth import _write_shared_xai_state, _xai_shared_store_path

    _write_shared_xai_state(
        {"access_token": "", "refresh_token": "rt"},
        last_refresh="2026-05-27T00:00:00Z",
    )

    assert not _xai_shared_store_path().is_file()


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


def test_shared_store_clear_removes_file(shared_store_env):
    """``_clear_shared_xai_state`` deletes the file (idempotent if absent)."""
    from hermes_cli.auth import (
        _clear_shared_xai_state,
        _write_shared_xai_state,
        _xai_shared_store_path,
    )

    _write_shared_xai_state(
        {"access_token": "at", "refresh_token": "rt"},
        last_refresh="2026-05-27T00:00:00Z",
    )
    assert _xai_shared_store_path().is_file()

    _clear_shared_xai_state(reason="test")
    assert not _xai_shared_store_path().is_file()

    # Idempotent — a second clear on an already-absent file must not raise.
    _clear_shared_xai_state(reason="test_again")


# ---------------------------------------------------------------------------
# Merge — the cross-profile import path
# ---------------------------------------------------------------------------


def test_merge_no_op_when_refresh_token_matches(shared_store_env):
    """Shared and local agree → merge is a no-op, returns False."""
    from hermes_cli.auth import (
        _merge_shared_xai_oauth_state,
        _xai_shared_store_path,
    )

    path = _xai_shared_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_shared_payload(refresh_token="rt-same")))

    state = _local_state(refresh_token="rt-same")
    snapshot = json.loads(json.dumps(state))

    changed = _merge_shared_xai_oauth_state(state)

    assert changed is False
    assert state == snapshot


def test_merge_copies_fresher_shared_tokens_into_state(shared_store_env):
    """Shared has a different (rotated) refresh_token → merge copies it
    plus the matching access_token into the local state and returns True.
    """
    from hermes_cli.auth import (
        _merge_shared_xai_oauth_state,
        _xai_shared_store_path,
    )

    path = _xai_shared_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _shared_payload(
                refresh_token="rt-rolled",
                access_token="at-rolled",
            )
        )
    )

    state = _local_state(refresh_token="rt-stale")

    changed = _merge_shared_xai_oauth_state(state)

    assert changed is True
    assert state["tokens"]["refresh_token"] == "rt-rolled"
    assert state["tokens"]["access_token"] == "at-rolled"
    # Other token fields propagate too.
    assert state["tokens"]["token_type"] == "Bearer"
    assert state["tokens"]["expires_in"] == 3600
    # last_refresh follows the shared store's clock.
    assert state["last_refresh"] == "2026-05-27T00:00:00Z"


def test_merge_returns_false_when_shared_missing(shared_store_env):
    """No shared file → merge is a no-op, returns False."""
    from hermes_cli.auth import _merge_shared_xai_oauth_state

    state = _local_state(refresh_token="rt-local")
    snapshot = json.loads(json.dumps(state))

    assert _merge_shared_xai_oauth_state(state) is False
    assert state == snapshot


# ---------------------------------------------------------------------------
# Integration: _save_xai_oauth_tokens mirrors to shared
# ---------------------------------------------------------------------------


def test_save_xai_oauth_tokens_mirrors_to_shared_store(
    tmp_path, monkeypatch, shared_store_env,
):
    """``_save_xai_oauth_tokens`` must populate BOTH per-profile auth.json
    AND the shared store so a sibling profile picking ``xai-oauth`` next
    sees the rotated tokens instead of re-using its stale refresh_token.
    This is the cross-profile race fix.
    """
    from hermes_cli.auth import (
        _read_shared_xai_state,
        _save_xai_oauth_tokens,
        _xai_shared_store_path,
    )

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_xai_oauth_tokens(
        {
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "token_type": "Bearer",
            "id_token": "id-1",
            "expires_in": 3600,
        },
        discovery={"token_endpoint": "https://api.x.ai/oauth2/token"},
        redirect_uri="http://127.0.0.1:8765/callback",
        last_refresh="2026-05-27T07:00:00Z",
    )

    # Local auth.json should exist with the tokens.
    auth_path = hermes_home / "auth.json"
    assert auth_path.is_file()
    local = json.loads(auth_path.read_text())
    assert (
        local["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "rt-1"
    )

    # Shared store should mirror them.
    shared_path = _xai_shared_store_path()
    assert shared_path.is_file()
    shared = _read_shared_xai_state()
    assert shared is not None
    assert shared["refresh_token"] == "rt-1"
    assert shared["access_token"] == "at-1"
    assert shared["last_refresh"] == "2026-05-27T07:00:00Z"
