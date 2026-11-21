"""Regression tests for the proxy adapter's quarantine write-back.

The defect (t_98c0d6cd): ``NousPortalAdapter._save_state`` takes a stale
snapshot of Nous provider state (line 71 of ``nous_portal.py``), then on a
terminal auth error writes that snapshot back via ``_save_state`` (line 81).
``_save_state`` re-loads the auth store but then assigns
``providers["nous"] = state`` — the WHOLE stale snapshot replaces whatever is
on disk, including any fresh tokens persisted by another writer (gateway
keepalive, cron worker, ``hermes auth add``) between the snapshot read and the
save.

This file tests the fix: ``_save_state`` now re-reads the on-disk state and
only applies quarantine changes when the on-disk ``refresh_token`` and
``access_token`` still match the snapshot's values. A mismatch means another
writer already persisted fresh credentials — skip the write-back.

These tests call ``_save_state`` directly, bypassing ``_get_credential``. Since
the pre-quarantine token annotations (``_pre_quarantine_refresh_token`` /
``_pre_quarantine_access_token``) are normally set by ``_get_credential`` before
quarantine, each test that calls ``_save_state`` directly must set them
explicitly on the stale state dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from hermes_cli.auth import AuthError, _load_auth_store, _quarantine_nous_oauth_state, _save_auth_store
from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter


def _write_auth_state(tmp_path: Path, state: Dict[str, Any]) -> Path:
    """Write *state* as the ``providers.nous`` section of a fresh auth.json."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps({
            "version": 1,
            "active_provider": "nous",
            "providers": {
                "nous": {
                    **state,
                },
            },
        })
    )
    return auth_file


def _read_nous_state(tmp_path: Path) -> Dict[str, Any]:
    """Read the ``providers.nous`` section back from auth.json."""
    store = _load_auth_store(tmp_path / "auth.json")
    return dict(store.get("providers", {}).get("nous", {}))


def _make_terminal_error() -> AuthError:
    """A terminal Nous refresh error that triggers quarantine."""
    return AuthError(
        "invalid_grant: token expired or revoked",
        provider="nous",
        code="invalid_grant",
        relogin_required=True,
    )


def _set_pre_quarantine_annotations(state: Dict[str, Any]) -> None:
    """Mirror what ``_get_credential`` does before calling ``_quarantine_nous_oauth_state``.

    ``_quarantine_nous_oauth_state`` pops ``refresh_token`` and ``access_token`` from
    *state*, so ``_save_state`` can no longer compare them against the on-disk values.
    ``_get_credential`` snapshots them as transient annotations first; replicate that
    here so the CAS guard in ``_save_state`` has something to compare.
    """
    state["_pre_quarantine_refresh_token"] = state.get("refresh_token")
    state["_pre_quarantine_access_token"] = state.get("access_token")


# ── lost-update regression ─────────────────────────────────────────────────────────────────


def test_quarantine_write_back_does_not_clobber_fresh_login(
    monkeypatch, tmp_path
):
    """The lost update scenario: a concurrent writer persists fresh tokens
    between the proxy's snapshot read and its quarantine save. The fresh tokens
    must survive; the stale snapshot must NOT replace them.

    Reproduction:
      1. auth.json holds stale nous tokens ("stale_rt", "stale_at").
      2. Another process (simulated here) writes fresh tokens ("fresh_rt",
         "fresh_at") to auth.json.
      3. The proxy, still holding the stale snapshot, hits a terminal error
         and calls _save_state with the stale state.
      4. After the call, auth.json must still hold the FRESH tokens.
    """
    stale_state = {
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "test-client",
        "refresh_token": "stale_rt",
        "access_token": "stale_at",
        "expires_at": "2020-01-01T00:00:00+00:00",
        "obtained_at": "2019-12-31T00:00:00+00:00",
    }
    fresh_state = {
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "test-client",
        "refresh_token": "fresh_rt",
        "access_token": "fresh_at",
        "expires_at": "2030-01-01T00:00:00+00:00",
        "obtained_at": "2025-01-01T00:00:00+00:00",
    }
    _write_auth_state(tmp_path, stale_state)

    # Step 2: concurrent writer persists fresh tokens (simulates hermes auth add).
    _write_auth_state(tmp_path, fresh_state)

    # Step 3: the proxy calls _save_state with the STALE snapshot (as if the
    # snapshot was taken before the concurrent write). Set pre-quarantine
    # annotations so the CAS guard can compare against on-disk values.
    adapter = NousPortalAdapter()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_pre_quarantine_annotations(stale_state)

    adapter._save_state(
        stale_state,
        quarantine_error=_make_terminal_error(),
        quarantine_reason="proxy_refresh_failure",
    )

    # Step 4: the fresh tokens must survive.
    on_disk = _read_nous_state(tmp_path)
    assert on_disk.get("refresh_token") == "fresh_rt", (
        "Fresh refresh token was clobbered by the stale snapshot"
    )
    assert on_disk.get("access_token") == "fresh_at", (
        "Fresh access token was clobbered by the stale snapshot"
    )


def test_quarantine_write_back_still_quarantines_when_tokens_match(
    monkeypatch, tmp_path
):
    """When no concurrent writer interferes (tokens match between snapshot and
    on-disk), the quarantine write-back must still clear the dead tokens — the
    fix must not regress the healthy quarantine behaviour.

    Reproduction:
      1. auth.json holds stale tokens ("stale_rt", "stale_at").
      2. No concurrent write: the snapshot and on-disk state agree.
      3. The proxy hits a terminal error and calls _save_state.
      4. After the call, auth.json must have the dead tokens removed (quarantine).
    """
    stale_state = {
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "test-client",
        "refresh_token": "stale_rt",
        "access_token": "stale_at",
        "expires_at": "2020-01-01T00:00:00+00:00",
        "obtained_at": "2019-12-31T00:00:00+00:00",
    }
    _write_auth_state(tmp_path, stale_state)

    adapter = NousPortalAdapter()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_pre_quarantine_annotations(stale_state)
    # Replicate the production flow: quarantine first, then save.
    _quarantine_nous_oauth_state(stale_state, _make_terminal_error(), reason="proxy_refresh_failure")

    adapter._save_state(
        stale_state,
        quarantine_error=_make_terminal_error(),
        quarantine_reason="proxy_refresh_failure",
    )

    on_disk = _read_nous_state(tmp_path)
    # After quarantine, the dead OAuth tokens must be cleared.
    assert on_disk.get("refresh_token") is None, (
        "Quarantine must clear the stale refresh token when no concurrent write occurred"
    )
    assert on_disk.get("access_token") is None, (
        "Quarantine must clear the stale access token when no concurrent write occurred"
    )
    # The last_auth_error marker must survive (it is set by _quarantine_nous_oauth_state
    # on the snapshot before _save_state is called; the marker key is never popped).
    assert "last_auth_error" in on_disk, "Quarantine must persist a last_auth_error marker"


def test_quarantine_write_back_preserves_pool_isolation(
    monkeypatch, tmp_path
):
    """When tokens mismatch, the pool quarantine must STILL run (pool entries are
    keyed by source, not by token value, so mismatch does not protect the pool).

    This is the pessimistic-path check: even though the provider state write-back
    is skipped to preserve fresh tokens, singleton-sourced pool entries holding
    dead OAuth state must still be quarantined.
    """
    stale_state = {
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "test-client",
        "refresh_token": "stale_rt",
        "access_token": "stale_at",
        "expires_at": "2020-01-01T00:00:00+00:00",
        "obtained_at": "2019-12-31T00:00:00+00:00",
    }
    fresh_state = {
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "test-client",
        "refresh_token": "fresh_rt",
        "access_token": "fresh_at",
        "expires_at": "2030-01-01T00:00:00+00:00",
        "obtained_at": "2025-01-01T00:00:00+00:00",
    }
    _write_auth_state(tmp_path, stale_state)

    # Simulate a credential pool with a singleton-sourced dead entry.
    store = _load_auth_store(tmp_path / "auth.json")
    store["credential_pool"] = {
        "nous": [
            {
                "source": "device_code",
                "client_id": "test-client",
                "portal_base_url": "https://portal.nousresearch.com",
                "access_token": "pool_stale_at",
                "refresh_token": "pool_stale_rt",
                "expires_at": "2020-01-01T00:00:00+00:00",
            },
        ]
    }
    _save_auth_store(store, tmp_path / "auth.json")

    # Concurrent writer persists fresh provider state.
    _write_auth_state(tmp_path, fresh_state)

    adapter = NousPortalAdapter()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_pre_quarantine_annotations(stale_state)

    adapter._save_state(
        stale_state,
        quarantine_error=_make_terminal_error(),
        quarantine_reason="proxy_refresh_failure",
    )

    # Fresh provider tokens must survive.
    on_disk = _read_nous_state(tmp_path)
    assert on_disk.get("refresh_token") == "fresh_rt"
    assert on_disk.get("access_token") == "fresh_at"

    # Pool must have been quarantined (the singleton device_code entry removed).
    pool = _load_auth_store(tmp_path / "auth.json").get("credential_pool", {}).get("nous", [])
    assert len(pool) == 0, (
        "Pool quarantine must still run even when the provider write-back is skipped"
    )
