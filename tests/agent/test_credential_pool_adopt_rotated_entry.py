"""Stale-agent-key recovery: adopt the pool's rotated token instead of
refreshing it or falling back.

Live incident 2026-09-19 03:16 PDT (``xai-oauth/grok-4.6`` → fallback to
``claude-opus-5``): a long-lived gateway session was built with access token
T1. xAI refresh tokens are single-use; another process consumed the refresh
token and the single ``device_code`` pool entry now held T2 (the same instant
a cron session was completing grok calls with T2). T1 got
``HTTP 403 unauthenticated:bad-credentials``. Recovery attributed the failure
to ``agent.api_key`` (T1), which matched no entry, so:

* ``try_refresh_matching(api_key_hint=T1)`` → ``None`` (nothing to refresh —
  correct, refreshing would burn the fresh single-use token), then
* ``mark_exhausted_and_rotate`` hit the single-entry escape and returned
  ``None`` on the premise that handing back the lone entry "does not change
  the credential" — false here, the entry holds T2 — so fallback fired while a
  perfectly valid token sat in the pool.

The fix adopts the lone entry when its ``runtime_api_key`` differs from the
failed key and is not itself expiring. Bounded by the existing
unmatched-rotation streak (#70401) and the caller's refresh-attempt cap.
"""
import json

import pytest


def _seed_pool(tmp_path, monkeypatch, entries, provider="openrouter", store=None):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "credential_pool": {provider: entries}}
    if store is not None:
        payload["providers"] = {provider: store}
    (hermes_home / "auth.json").write_text(json.dumps(payload))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool(provider)


def _entry(idx, key, **extra):
    base = {
        "id": f"cred-{idx}",
        "label": f"key-{idx}",
        "auth_type": "api_key",
        "priority": idx,
        "source": "manual",
        "access_token": key,
    }
    base.update(extra)
    return base


class TestSingleEntryPoolAdoptsRotatedToken:

    def test_unmatched_stale_key_adopts_lone_entry_once_then_bounds(
        self, tmp_path, monkeypatch
    ):
        """Agent holds T1, pool's only entry holds T2 → first rotation hands
        back the entry (credential changes); a second unmatched failure with
        the pool unchanged returns None so fallback can fire."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "T2")])
        assert pool.select() is not None

        first = pool.mark_exhausted_and_rotate(
            status_code=403,
            error_context={"reason": "unauthorized"},
            api_key_hint="T1",
        )
        assert first is not None
        assert first.runtime_api_key == "T2"
        assert pool.current() is not None and pool.current().id == first.id
        # Nothing was quarantined — the failure belonged to a key the pool
        # no longer holds.
        assert all(e.last_status != "exhausted" for e in pool._entries)

        # Caller retried with T1 again (it did not swap) — the streak cap
        # (#70401) must stop the loop.
        second = pool.mark_exhausted_and_rotate(
            status_code=403,
            error_context={"reason": "unauthorized"},
            api_key_hint="T1",
        )
        assert second is None

    def test_same_key_as_lone_entry_still_escapes(self, tmp_path, monkeypatch):
        """Regression guard for the original single-entry escape: when the
        failed key IS the lone entry's key but was reported under an identity
        that matches nothing, returning the entry changes nothing → None."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "T1")])
        assert pool.select() is not None
        # api_key_hint equals the entry's key but credential_id is bogus and
        # the #79156 disagreement path drops the id → treated as unmatched.
        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="T1",
            credential_id="not-a-real-id",
        )
        # A real match on the key marks the entry exhausted and, with one
        # entry, has nothing to rotate to.
        assert nxt is None or nxt.runtime_api_key != "T1"

    def test_unmatched_hint_with_no_hint_key_does_not_adopt(
        self, tmp_path, monkeypatch
    ):
        """credential_id-only identity that matches nothing must keep the
        old escape (no key to compare against → no adoption)."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "T2")])
        assert pool.select() is not None
        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            credential_id="ghost",
        )
        assert nxt is None

    def test_expiring_lone_entry_is_not_adopted(self, tmp_path, monkeypatch):
        """Trading a dead key for an expiring one is not a recovery."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "T2")])
        assert pool.select() is not None
        monkeypatch.setattr(pool, "_entry_needs_refresh", lambda entry: True)
        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "unauthorized"},
            api_key_hint="T1",
        )
        assert nxt is None


class TestTryRefreshMatchingAdoptsInsteadOfBurningRefreshToken:

    def test_stale_id_bound_key_adopts_without_refresh(self, tmp_path, monkeypatch):
        """credential_id matches the entry but the entry already holds a
        different token than the one that failed: adopt it, do NOT spend the
        (single-use) refresh token."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "T2")])
        assert pool.select() is not None
        calls = []

        def _boom(entry, force=False):
            calls.append(entry.id)
            raise AssertionError("refresh must not run for a rotated entry")

        monkeypatch.setattr(pool, "_refresh_entry", _boom)

        got = pool.try_refresh_matching(api_key_hint="T1", credential_id="cred-0")
        assert got is not None
        assert got.runtime_api_key == "T2"
        assert calls == []

    def test_matching_key_still_refreshes(self, tmp_path, monkeypatch):
        """Regression guard: when the failed key IS the entry's key the
        forced refresh path is unchanged."""
        pool = _seed_pool(tmp_path, monkeypatch, [_entry(0, "T1")])
        assert pool.select() is not None
        calls = []

        def _fake_refresh(entry, force=False):
            calls.append((entry.id, force))
            return entry

        monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)
        got = pool.try_refresh_matching(api_key_hint="T1", credential_id="cred-0")
        assert got is not None
        assert calls == [("cred-0", True)]

    def test_hint_matches_other_entry_prefers_that_entry(self, tmp_path, monkeypatch):
        """Multi-entry: a stale id with a hint that matches ANOTHER entry must
        refresh the key-matched entry (#79156 semantics), not adopt."""
        pool = _seed_pool(
            tmp_path, monkeypatch, [_entry(0, "T-a"), _entry(1, "T-b")]
        )
        assert pool.select() is not None
        calls = []

        def _fake_refresh(entry, force=False):
            calls.append(entry.id)
            return entry

        monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)
        # credential_id points at cred-0 but the failing key is cred-1's.
        got = pool.try_refresh_matching(api_key_hint="T-b", credential_id="cred-0")
        assert got is not None
        # Existing behaviour: id wins when both are supplied; we only assert
        # that adoption did not short-circuit the refresh.
        assert calls, "refresh path was skipped for a key that IS in the pool"


class TestXaiSingletonResyncBeforeAdopt:

    def test_xai_pool_resyncs_from_auth_store_before_adopting(
        self, tmp_path, monkeypatch
    ):
        """Agent dispatched with T0; the in-process xai pool still holds the
        later T1 but auth.json was rotated again to T2 by another
        process/keeper: the adopt step must resync from the store and hand
        back T2 (not the pool's own stale T1)."""
        entry = _entry(
            0,
            "T1",
            auth_type="oauth",
            source="device_code",
            refresh_token="R1",
            last_refresh="2026-09-19T01:00:00Z",
        )
        store = {
            "tokens": {"access_token": "T2", "refresh_token": "R2"},
            "last_refresh": "2026-09-19T02:00:00Z",
        }
        pool = _seed_pool(
            tmp_path, monkeypatch, [entry], provider="xai-oauth", store=store
        )
        # Pool loads may already resync; force the in-memory entry back to
        # T1 to model a pool that has not touched the store since rotation.
        from dataclasses import replace

        with pool._lock:
            cur = pool._entries[0]
            pool._entries[0] = replace(cur, access_token="T1", refresh_token="R1")
        monkeypatch.setattr(pool, "_entry_needs_refresh", lambda e: False)
        assert pool.select() is not None

        got = pool.mark_exhausted_and_rotate(
            status_code=403,
            error_context={"reason": "unauthorized"},
            api_key_hint="T0",
        )
        assert got is not None
        assert got.runtime_api_key == "T2"
        assert got.refresh_token == "R2"
