"""Regression tests for #106705: Codex manual:device_code pool entries must
not replay an already-consumed refresh token by re-adopting a stale singleton.

Causal chain (agent/credential_pool.py):

1. ``_sync_entry_from_auth_store`` adopts differing singleton tokens for BOTH
   ``device_code`` and ``manual:device_code`` Codex entries with no staleness
   proof.
2. A successful pool-side rotation persists the fresh chain into the
   credential-pool store, but ``_sync_device_code_entry_to_auth_store``
   deliberately skips singleton write-back for ``manual:*`` sources
   (independent-credential contract, #39236), so the singleton stays one
   rotation behind.
3. The next refresh syncs that STALE singleton over the pool's fresh entry and
   POSTs the consumed refresh token again -> ``refresh_token_reused``.

The fix gates adoption on the singleton being provably newer (``last_refresh``
comparison); these tests run the real production path (``load_pool`` ->
``_refresh_entry``) with only the HTTP transport boundary mocked.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

import pytest

from agent.credential_pool import load_pool

# Synthetic JWT-ish access tokens carrying a far-future expiry claim so
# token-expiry probes see a valid token and the refresh path is driven by
# ``force`` alone.
_FAR_FUTURE_EXP = 4102444800  # 2100-01-01


def _jwt(exp: int = _FAR_FUTURE_EXP) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part({'exp': exp})}.sig"


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# Rotation chains: old -> new1 -> new2. Timestamps strictly increase so the
# ``last_refresh`` ordering is provable.
_T_OLD = 1_800_000_000.0
_T_NEW1 = _T_OLD + 600.0

_AT_OLD, _RT_OLD = _jwt(), "rt-old"
_AT_NEW1, _RT_NEW1 = _jwt(), "rt-new1"
_AT_NEW2, _RT_NEW2 = _jwt(), "rt-new2"


def _manual_entry_payload(id: str = "manual-1", access_token: str = _AT_OLD,
                          refresh_token: str = _RT_OLD, last_refresh=None) -> dict:
    return {
        "id": id,
        "label": "manual codex grant",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "last_refresh": last_refresh,
    }


def _store(provider_state: dict, pool_entries: list) -> dict:
    return {
        "version": 1,
        "providers": {"openai-codex": provider_state},
        "credential_pool": {"openai-codex": pool_entries},
    }


def _tokens_state(access_token: str, refresh_token: str, last_refresh) -> dict:
    return {
        "tokens": {"access_token": access_token, "refresh_token": refresh_token},
        "last_refresh": last_refresh,
    }


def _write_store(tmp_path, monkeypatch, store: dict) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps(store, indent=2))
    monkeypatch.setenv("HERMES_HOME", str(home))


def _install_fake_refresh(monkeypatch, chains: dict, stamp=_T_NEW1 + 600.0):
    """Mock only the HTTP transport; record every refresh_token POSTed."""
    posted = []

    def fake_refresh(access_token, refresh_token, **kwargs):
        posted.append(refresh_token)
        if refresh_token not in chains:
            raise AssertionError(f"unexpected refresh POST for {refresh_token!r}")
        at, rt = chains[refresh_token]
        return {"access_token": at, "refresh_token": rt, "last_refresh": _iso(stamp)}

    monkeypatch.setattr(
        "agent.credential_pool.auth_mod.refresh_codex_oauth_pure", fake_refresh
    )
    return posted


class TestPoolRotationDoesNotReplayConsumedRefreshToken:
    """L1 + L6: after a pool-side rotation, the stale singleton must not win."""

    def test_second_forced_refresh_uses_rotated_chain_not_stale_singleton(self, tmp_path, monkeypatch):
        """L1 — the reporter's exact scenario.

        Entry holds the chain rotated by the pool (new1, stamped new1-time);
        the singleton still holds the consumed old chain (stamped old-time —
        write-back deliberately skipped for manual sources). A second forced
        refresh must POST the entry's own new1 refresh token — NOT re-adopt
        the stale singleton and replay old.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_OLD, _RT_OLD, _iso(_T_OLD)),
            [_manual_entry_payload(
                access_token=_AT_NEW1, refresh_token=_RT_NEW1, last_refresh=_iso(_T_NEW1),
            )],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_NEW1: (_AT_NEW2, _RT_NEW2)})

        pool = load_pool("openai-codex")
        updated = pool._refresh_entry(pool._entries[0], force=True)

        assert updated is not None
        assert updated.refresh_token == _RT_NEW2
        # The consumed old refresh token must never be re-POSTed.
        assert posted == [_RT_NEW1]

    def test_recover_path_does_not_adopt_stale_singleton_after_failed_post(self, tmp_path, monkeypatch):
        """L6 — failed-POST recovery must not regress the fresh chain.

        After the pool rotated to new1 (write-back skipped), a refresh POST
        that fails must NOT make ``_recover_failed_refresh`` adopt the stale
        old singleton as "newer tokens" — that regression is exactly the
        replay loop.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_OLD, _RT_OLD, _iso(_T_OLD)),
            [_manual_entry_payload(
                access_token=_AT_NEW1, refresh_token=_RT_NEW1, last_refresh=_iso(_T_NEW1),
            )],
        ))

        def failing_refresh(access_token, refresh_token, **kwargs):
            raise RuntimeError("synthetic network failure")

        monkeypatch.setattr(
            "agent.credential_pool.auth_mod.refresh_codex_oauth_pure", failing_refresh
        )

        pool = load_pool("openai-codex")
        pool._refresh_entry(pool._entries[0], force=True)

        # The entry must not be regressed onto the stale singleton chain.
        entry_now = next(e for e in pool._entries if e.id == "manual-1")
        assert entry_now.refresh_token == _RT_NEW1
        assert entry_now.access_token == _AT_NEW1


class TestFreshSingletonAdoptionStillWorks:
    """L2 — #70111 regression guard: a provably NEWER singleton must win."""

    def test_newer_singleton_is_adopted_over_stale_pool_entry(self, tmp_path, monkeypatch):
        """The fleet-outage fix (commit 7380b48589) must keep working.

        The singleton was rotated by another process (e.g. ``hermes model``
        re-auth); the pool entry still holds the old consumed chain. The sync
        must adopt the newer singleton and refresh with IT.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_NEW1, _RT_NEW1, _iso(_T_NEW1)),
            [_manual_entry_payload(
                access_token=_AT_OLD, refresh_token=_RT_OLD, last_refresh=_iso(_T_OLD),
            )],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_NEW1: (_AT_NEW2, _RT_NEW2)})

        pool = load_pool("openai-codex")
        updated = pool._refresh_entry(pool._entries[0], force=True)

        assert updated is not None
        assert updated.refresh_token == _RT_NEW2
        assert posted == [_RT_NEW1]

    def test_refresh_token_only_singleton_is_adopted(self, tmp_path, monkeypatch):
        """L2b — refresh_token-only singleton (the consumed-access branch).

        Another process rotated and the access_token was consumed; only the
        new refresh_token remains on disk. Adoption must still fire (newer
        ``last_refresh``) so the consumed token is not replayed.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state("", _RT_NEW1, _iso(_T_NEW1)),
            [_manual_entry_payload(
                access_token=_AT_OLD, refresh_token=_RT_OLD, last_refresh=_iso(_T_OLD),
            )],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_NEW1: (_AT_NEW2, _RT_NEW2)})

        pool = load_pool("openai-codex")
        updated = pool._refresh_entry(pool._entries[0], force=True)

        assert updated is not None
        assert updated.refresh_token == _RT_NEW2
        assert posted == [_RT_NEW1]


class TestTimestampEdgeCases:
    """L3 — missing timestamps fail open to the historical adopt-on-difference."""

    def test_missing_entry_timestamp_still_adopts_differing_singleton(self, tmp_path, monkeypatch):
        """Entry carries no ``last_refresh`` (pre-stamping pool writer).

        Old behavior (adopt on difference) must be preserved so a fresh
        re-auth is never stranded when the entry side has no timestamp.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_NEW1, _RT_NEW1, _iso(_T_NEW1)),
            [_manual_entry_payload(
                access_token=_AT_OLD, refresh_token=_RT_OLD, last_refresh=None,
            )],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_NEW1: (_AT_NEW2, _RT_NEW2)})

        pool = load_pool("openai-codex")
        updated = pool._refresh_entry(pool._entries[0], force=True)

        assert updated is not None
        assert updated.refresh_token == _RT_NEW2
        assert posted == [_RT_NEW1]

    def test_missing_singleton_timestamp_still_adopts_differing_singleton(self, tmp_path, monkeypatch):
        """Singleton carries no ``last_refresh`` (legacy auth.json writer).

        Cannot prove the singleton older -> fall back to adopt-on-difference
        so a fresh re-auth from a legacy writer is not stranded either.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_NEW1, _RT_NEW1, None),
            [_manual_entry_payload(
                access_token=_AT_OLD, refresh_token=_RT_OLD, last_refresh=_iso(_T_OLD),
            )],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_NEW1: (_AT_NEW2, _RT_NEW2)})

        pool = load_pool("openai-codex")
        updated = pool._refresh_entry(pool._entries[0], force=True)

        assert updated is not None
        assert updated.refresh_token == _RT_NEW2
        assert posted == [_RT_NEW1]


class TestDeviceCodeSeededEntries:
    """L4 — the singleton-seeded source is unaffected by the guard."""

    def test_device_code_entry_rotates_and_writeback_converges(self, tmp_path, monkeypatch):
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_OLD, _RT_OLD, _iso(_T_OLD)),
            [],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_OLD: (_AT_NEW1, _RT_NEW1)})

        pool = load_pool("openai-codex")
        seeded = [e for e in pool._entries if e.source == "device_code"]
        assert seeded, "device_code entry must be seeded from the singleton"

        updated = pool._refresh_entry(seeded[0], force=True)
        assert updated is not None
        assert updated.refresh_token == _RT_NEW1
        assert posted == [_RT_OLD]

        # Write-back must converge the singleton onto the fresh chain.
        on_disk = json.loads((tmp_path / "hermes" / "auth.json").read_text())
        synced = on_disk["providers"]["openai-codex"]["tokens"]
        assert synced["refresh_token"] == _RT_NEW1


class TestIndependentAccountNotClobbered:
    """L5 — #39236 guard: independent manual grants keep their own chain."""

    def test_independent_manual_entry_with_newer_own_timestamp_is_not_overwritten(self, tmp_path, monkeypatch):
        """An independent account's entry whose own chain is NEWER (rotated
        by this pool) must never be overwritten by the older singleton —
        same mechanism as L1, asserted as the #39236 no-clobber contract.
        """
        _write_store(tmp_path, monkeypatch, _store(
            _tokens_state(_AT_OLD, _RT_OLD, _iso(_T_OLD)),
            [_manual_entry_payload(
                id="indep-1",
                access_token=_AT_NEW1, refresh_token=_RT_NEW1, last_refresh=_iso(_T_NEW1),
            )],
        ))
        posted = _install_fake_refresh(monkeypatch, {_RT_NEW1: (_AT_NEW2, _RT_NEW2)})

        pool = load_pool("openai-codex")
        updated = pool._refresh_entry(pool._entries[0], force=True)

        assert updated is not None
        assert updated.refresh_token == _RT_NEW2
        assert _RT_OLD not in posted
