"""Regression tests for the stale-writer fence on the shared xAI OAuth lineage.

Issue #77553: xAI rotates refresh tokens on every refresh, and replaying a
consumed refresh token revokes the *entire* token family. A stale writer —
a gateway that cached credentials before another writer rotated the shared
grant, or a profile shadow key left behind by a pre-fix runtime — can spend
a superseded refresh token and revoke the shared lineage for every profile
plus the external refresh coordinator (fleet-wide ``invalid_grant``).

The fence (``_fence_xai_oauth_refresh_spend``) runs at the single spend
choke point (``_refresh_xai_oauth_tokens``) under the auth-store lock and
fails CLOSED: if the token about to be spent is no longer the current head
of the persisted lineage, it raises without ever calling the token endpoint,
so no token is burned and no family-wide revocation can be triggered.

These tests drive the real ``_refresh_xai_oauth_tokens`` / ``_save_xai_oauth_tokens``
against real on-disk auth stores (profile + root under ``tmp_path``) and spy
on the token endpoint so we can assert it is never called on a fenced spend.
"""

import json

import pytest

from hermes_cli import auth


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    Returns (profile_path, root_path). HOME is pointed away from the tmp
    root so the pytest seat belt in
    ``_write_through_xai_oauth_to_global_root`` does not trip (mirrors
    ``test_xai_oauth_writethrough.py``).
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


def _refresh_spy(spent, result=None):
    """Token-endpoint spy: records every refresh token it is asked to spend."""
    def _fake_pure(access_token, refresh_token, **kwargs):
        spent.append(refresh_token)
        if result is not None:
            return result
        return {
            "access_token": "a-rotated",
            "refresh_token": "r-rotated",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": "2026-08-03T00:00:00Z",
        }
    return _fake_pure


def test_stale_writer_is_fenced_before_spending_shared_token(profile_and_root, monkeypatch):
    """#77553: a writer holding a superseded refresh token fails closed and
    the token endpoint is NEVER called (nothing is burned)."""
    profile_path, root_path = profile_and_root
    # Root already holds the rotated head (gen 2) written by another writer.
    _write_store(root_path, {"version": 1, "providers": {"xai-oauth": {
        "tokens": {"access_token": "a-new", "refresh_token": "r-new"},
        "lineage_generation": 2,
        "lineage_write_contract": auth.XAI_OAUTH_LINEAGE_CONTRACT,
    }}})
    _write_store(profile_path, {"version": 1, "providers": {}})

    spent = []
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh_spy(spent))

    with pytest.raises(auth.AuthError) as excinfo:
        auth._refresh_xai_oauth_tokens(
            {"access_token": "a-old", "refresh_token": "r-old"},
            token_endpoint="https://token.x.ai/oauth2/token",
            timeout_seconds=5.0,
        )
    assert excinfo.value.code == "xai_stale_refresh_fenced"
    assert spent == []  # token endpoint never called -> no token burned
    # Stores untouched by the fenced spend.
    assert _read_store(root_path)["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "r-new"
    assert "providers" not in _read_store(profile_path) or "xai-oauth" not in _read_store(profile_path)["providers"]


def test_current_head_writer_passes_fence_and_bumps_generation(profile_and_root, monkeypatch):
    """A writer spending the CURRENT head passes the fence, rotates, and the
    write-through lands back in root with the generation bumped and the
    write-through contract stamped — without shadowing the profile store."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {"xai-oauth": {
        "tokens": {"access_token": "a-new", "refresh_token": "r-new"},
        "lineage_generation": 2,
        "lineage_write_contract": auth.XAI_OAUTH_LINEAGE_CONTRACT,
    }}})
    _write_store(profile_path, {"version": 1, "providers": {}})

    spent = []
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh_spy(spent))

    rotated = auth._refresh_xai_oauth_tokens(
        {"access_token": "a-new", "refresh_token": "r-new"},
        token_endpoint="https://token.x.ai/oauth2/token",
        timeout_seconds=5.0,
    )
    assert spent == ["r-new"]  # the head was spent exactly once
    assert rotated["refresh_token"] == "r-rotated"

    root = _read_store(root_path)
    state = root["providers"]["xai-oauth"]
    assert state["tokens"]["refresh_token"] == "r-rotated"
    assert state["lineage_generation"] == 3
    assert state["lineage_write_contract"] == auth.XAI_OAUTH_LINEAGE_CONTRACT
    # Source-path write-through (#74339): no shadowing key in the profile.
    assert "xai-oauth" not in _read_store(profile_path).get("providers", {})


def test_detached_pool_writer_is_fenced(profile_and_root, monkeypatch):
    """A writer spending a refresh token that matches NO persisted lineage
    head (provider state or credential pool) fails closed."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {}})
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {
        "xai-oauth": [{"access_token": "a-new", "refresh_token": "r-new"}]
    }})

    spent = []
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh_spy(spent))

    with pytest.raises(auth.AuthError) as excinfo:
        auth._refresh_xai_oauth_tokens(
            {"access_token": "a-old", "refresh_token": "r-detached"},
            token_endpoint="https://token.x.ai/oauth2/token",
            timeout_seconds=5.0,
        )
    assert excinfo.value.code == "xai_unverified_lineage"
    assert excinfo.value.relogin_required is True
    assert spent == []


def test_pool_head_writer_passes_fence(profile_and_root, monkeypatch):
    """A writer spending the current credential-pool head passes the fence."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, {"version": 1, "providers": {}})
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {
        "xai-oauth": [{"access_token": "a-new", "refresh_token": "r-new"}]
    }})

    spent = []
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh_spy(spent))

    auth._refresh_xai_oauth_tokens(
        {"access_token": "a-new", "refresh_token": "r-new"},
        token_endpoint="https://token.x.ai/oauth2/token",
        timeout_seconds=5.0,
    )
    assert spent == ["r-new"]


def test_legacy_shared_lineage_still_spends_head_and_gets_stamped(profile_and_root, monkeypatch, caplog):
    """A legacy (pre-fix, uncontracted) shared lineage is NOT bricked: the
    writer spending the current head proceeds, emits the mixed-version
    compatibility warning, and the save stamps the write-through contract."""
    profile_path, root_path = profile_and_root
    # Pre-fix root state: no lineage_write_contract, no lineage_generation.
    _write_store(root_path, {"version": 1, "providers": {"xai-oauth": {
        "tokens": {"access_token": "a-legacy", "refresh_token": "r-legacy"},
    }}})
    _write_store(profile_path, {"version": 1, "providers": {}})

    spent = []
    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh_spy(spent))

    with caplog.at_level("WARNING", logger="hermes_cli.auth"):
        auth._refresh_xai_oauth_tokens(
            {"access_token": "a-legacy", "refresh_token": "r-legacy"},
            token_endpoint="https://token.x.ai/oauth2/token",
            timeout_seconds=5.0,
        )
    assert spent == ["r-legacy"]
    assert any("write-through contract" in rec.message for rec in caplog.records)

    root = _read_store(root_path)
    state = root["providers"]["xai-oauth"]
    assert state["tokens"]["refresh_token"] == "r-rotated"
    assert state["lineage_generation"] == 1
    assert state["lineage_write_contract"] == auth.XAI_OAUTH_LINEAGE_CONTRACT
