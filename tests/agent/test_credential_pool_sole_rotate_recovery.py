"""#97315: a rotation that can only hand back the entry that was just marked
exhausted is no recovery — it must return None so the turn fails.

A sole-credential ``openai-codex`` pool hitting a 429 ``usage_limit_reached``
writes a correct bench (``last_error_reset_at`` days into the future), yet the
selection path can re-admit the just-marked entry within milliseconds:
``_available_entries`` runs the auth-store sync for every exhausted codex
``device_code`` entry, and when auth.json holds a token pair that differs from
the pool entry's — e.g. the request path's own resolve refreshed the
single-use tokens — the sync adopts them and clears the bench mid-selection.
``_select_unlocked`` then returns the sole entry, ``mark_exhausted_and_rotate``
reports a successful rotation, and the caller retries the same throttled
credential forever (~2 req/s for hours; 11k+ requests against an
already-exhausted account, gateway wedged until killed by hand).

The fix mirrors the single-entry guard on the unmatched-identity branch of the
same function: if the post-mark selection returns the very entry that was just
marked, treat it as no-recovery (None) so the failure surfaces and fallback /
error propagation proceeds. A genuine re-auth still recovers on the next
selection; multi-entry pools keep rotating to a healthy sibling.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace


def _entry(idx: int, *, token: str, refresh: str) -> dict:
    return {
        "id": f"codex-{idx}",
        "label": f"codex-login-{idx}",
        "auth_type": "oauth",
        "priority": idx,
        "source": "device_code",
        "access_token": token,
        "refresh_token": refresh,
    }


def _load(tmp_path, monkeypatch, entries: list[dict]):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {"version": 1, "credential_pool": {"openai-codex": entries}}
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool("openai-codex")


def _revive_entry(pool, entry):
    """Simulate the auth-store sync adopting fresher tokens from auth.json:
    the bench (last_status / reset_at) is cleared mid-selection, exactly as
    ``_sync_codex_entry_from_auth_store`` does for a changed token pair."""
    updated = replace(
        entry,
        access_token=entry.access_token + "-adopted",
        refresh_token=(entry.refresh_token or "") + "-adopted",
        last_status=None,
        last_status_at=None,
        last_error_code=None,
        last_error_reason=None,
        last_error_message=None,
        last_error_reset_at=None,
    )
    pool._replace_entry(entry, updated)
    return updated


def test_revived_sole_entry_is_no_recovery(tmp_path, monkeypatch):
    """The auth-store sync revives the just-marked sole entry mid-selection —
    returning it reports recovery without changing the credential, so the
    caller would retry the same 429 forever. Must return None instead."""
    pool = _load(tmp_path, monkeypatch, [_entry(1, token="tok-a", refresh="rf-a")])
    pool._sync_codex_entry_from_auth_store = lambda entry: _revive_entry(pool, entry)

    nxt = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={
            "reason": "usage_limit_reached",
            "reset_at": time.time() + 95.5 * 3600,
        },
        credential_id="codex-1",
    )
    assert nxt is None
    # The failure must stay surfaced: no cursor pointing at the dead entry.
    assert pool._current_id is None


def test_benched_sole_entry_without_revival_stays_benched(tmp_path, monkeypatch):
    """Without a mid-selection revival the bench holds: the 429 reset_at keeps
    the sole entry out of rotation and rotation returns None (pre-existing
    semantics — pinned so the new guard cannot mask a broken bench)."""
    pool = _load(tmp_path, monkeypatch, [_entry(1, token="tok-a", refresh="rf-a")])
    # Tokens already in sync with auth.json: the sync path adopts nothing.
    pool._sync_codex_entry_from_auth_store = lambda entry: entry

    nxt = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={
            "reason": "usage_limit_reached",
            "reset_at": time.time() + 95.5 * 3600,
        },
        credential_id="codex-1",
    )
    assert nxt is None
    assert pool._entries[0].last_status == "exhausted"
    assert pool._entries[0].last_error_reset_at is not None


def test_multi_entry_pool_still_rotates_to_sibling(tmp_path, monkeypatch):
    """Regression guard: with a healthy sibling available, the post-mark
    rotation must still return it — the no-recovery guard only fires when the
    selection hands back the just-marked entry itself."""
    pool = _load(
        tmp_path,
        monkeypatch,
        [
            _entry(1, token="tok-a", refresh="rf-a"),
            _entry(2, token="tok-b", refresh="rf-b"),
        ],
    )
    pool._sync_codex_entry_from_auth_store = lambda entry: entry

    nxt = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={
            "reason": "usage_limit_reached",
            "reset_at": time.time() + 95.5 * 3600,
        },
        credential_id="codex-1",
    )
    assert nxt is not None
    assert nxt.id == "codex-2"
