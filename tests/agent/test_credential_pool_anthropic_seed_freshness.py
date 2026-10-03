"""Anthropic ``hermes_pkce`` singleton: producer stamp + seeding freshness gate.

Third file in the series.  ``test_credential_pool_singleton_freshness.py`` covers
the ``_sync_*_entry_from_auth_store`` readers;
``test_credential_pool_seed_freshness.py`` covers the seeding-path gate for the
three ``device_code`` singleton providers.  Neither reaches Anthropic, because
its singleton could not be gated: ``~/.hermes/.anthropic_oauth.json`` carried
only ``accessToken``/``refreshToken``/``expiresAt``, and ``expiresAt`` is a token
EXPIRY, not a refresh instant -- two pairs minted minutes apart can carry the
same or an inverted ``expiresAt``.  Declaring the group against it would have
shipped an INERT gate: both sides parse to ``None``, the guard returns the
payload untouched, and a synthetic test on a stamped fixture would never notice
that nothing on the live path writes a stamp.

So the producer is fixed first.  ``_write_hermes_oauth_credentials`` -- the sole
writer of that file, and the commit step of every ``hermes_pkce`` rotation --
now stamps ``lastRefresh`` with the instant the pair was minted and returns it,
so the refreshing pool entry can carry the same instant.  Only then is
``("anthropic", "hermes_pkce")`` declared in ``_SINGLETON_FRESHNESS_GROUPS``.

``claude_code`` is deliberately NOT gated; ``test_claude_code_*`` below pins that
decision with the measurement behind it.

Everything here is synthetic: no network, no real credentials, isolated
HOME/HERMES_HOME per test.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace as dc_replace

import pytest

from agent import anthropic_credentials as AA
from agent.credential_pool import STATUS_EXHAUSTED, load_pool

_POOL_ACCESS = "synthetic-pool-access"
_POOL_REFRESH = "synthetic-pool-refresh"
_STORE_ACCESS = "synthetic-store-access"
_STORE_REFRESH = "synthetic-store-refresh"

ENTRY_TIME = "2026-09-10T04:30:00Z"
OLDER = "2026-07-22T15:28:55Z"
NEWER = "2026-09-10T05:00:00Z"

_HOUR_MS = 3_600_000

# The one place the payload key name and the table key name have to agree.  A
# rename on either side turns the gate inert, which is the whole failure mode
# this file exists to prevent.
SINGLETON_STAMP_KEY = "lastRefresh"
ENTRY_STAMP_KEY = "last_refresh"


@pytest.fixture
def anthropic_home(tmp_path, monkeypatch):
    """Real on-disk HOME/HERMES_HOME with anthropic explicitly configured."""
    import hermes_cli.auth as auth

    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True
    )
    # Never read the real macOS Keychain entry from a test.
    monkeypatch.setattr(AA, "_read_claude_code_credentials_from_keychain", lambda: None)
    assert auth._auth_file_path() == home / "auth.json"
    return home


def _pool_rows(home):
    store = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    return store.get("credential_pool", {}).get("anthropic", [])


def _write_pkce_singleton(access, refresh, *, stamp, expires_at_ms=None):
    blob = {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresAt": expires_at_ms
        if expires_at_ms is not None
        else int(time.time() * 1000) + _HOUR_MS,
    }
    if stamp is not None:
        blob[SINGLETON_STAMP_KEY] = stamp
    AA._get_hermes_oauth_file().write_text(json.dumps(blob), encoding="utf-8")


def _write_claude_code_singleton(access, refresh, *, stamp):
    oauth = {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresAt": int(time.time() * 1000) + _HOUR_MS,
    }
    if stamp is not None:
        oauth[SINGLETON_STAMP_KEY] = stamp
    path = AA.claude_code_credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"claudeAiOauth": oauth}), encoding="utf-8")


@pytest.fixture
def seeded_pkce(anthropic_home):
    """Seed a ``hermes_pkce`` row from the singleton, then rewrite and reload.

    Both halves go through the real ``load_pool`` -> ``_seed_from_singletons``
    -> ``_upsert_entry`` path -- the write path this file is about.
    """

    def build(
        *,
        entry_stamp=ENTRY_TIME,
        store_stamp=OLDER,
        entry_expires_ms=None,
        store_expires_ms=None,
        mutate_entry=None,
    ):
        _write_pkce_singleton(
            _POOL_ACCESS, _POOL_REFRESH,
            stamp=entry_stamp, expires_at_ms=entry_expires_ms,
        )
        pool = load_pool("anthropic")
        entry = next(e for e in pool.entries() if e.source == "hermes_pkce")
        assert entry.refresh_token == _POOL_REFRESH
        if mutate_entry is not None:
            updated = mutate_entry(entry)
            pool._replace_entry(entry, updated)
            pool._persist()
            entry = updated

        # Another writer regresses (or advances) the singleton.
        _write_pkce_singleton(
            _STORE_ACCESS, _STORE_REFRESH,
            stamp=store_stamp, expires_at_ms=store_expires_ms,
        )

        reloaded = load_pool("anthropic")
        row = next(e for e in reloaded.entries() if e.id == entry.id)
        disk = next(r for r in _pool_rows(anthropic_home) if r["id"] == entry.id)
        return {"entry": entry, "row": row, "disk": disk}

    return build


# ── the producer: the singleton must carry a refresh INSTANT ──────────────


def test_writer_stamps_the_singleton_with_a_refresh_instant(anthropic_home):
    """``_write_hermes_oauth_credentials`` is the choke point that must stamp.

    Without a stamp on this file nothing downstream can distinguish an older
    pair from a newer one -- ``expiresAt`` is a token expiry, not a mint time.
    """
    from agent.credential_pool import _parse_absolute_timestamp

    before = time.time()
    returned = AA._write_hermes_oauth_credentials("acc", "ref", int(before * 1000))
    after = time.time()

    on_disk = json.loads(
        AA._get_hermes_oauth_file().read_text(encoding="utf-8")
    )
    assert SINGLETON_STAMP_KEY in on_disk, (
        "the singleton must carry a refresh instant, or the freshness gate is inert"
    )
    stamped = _parse_absolute_timestamp(on_disk[SINGLETON_STAMP_KEY])
    assert stamped is not None, "the stamp must parse as an instant"
    assert before - 1 <= stamped <= after + 1

    assert returned == on_disk[SINGLETON_STAMP_KEY], (
        "the writer must hand the caller the instant it committed, so the "
        "refreshing pool entry can carry the same one"
    )


def test_writer_stamp_is_a_mint_time_not_the_token_expiry(anthropic_home):
    """Two pairs written in order must be ordered by the stamp.

    ``expiresAt`` cannot do this: a pair minted later can carry an equal or
    smaller expiry, which is exactly why a second key was needed.
    """
    from agent.credential_pool import _parse_absolute_timestamp

    far_future = int(time.time() * 1000) + 10 * _HOUR_MS
    first = AA._write_hermes_oauth_credentials("acc1", "ref1", far_future)
    time.sleep(0.01)
    # Second write is genuinely later but carries an EARLIER expiry.
    second = AA._write_hermes_oauth_credentials("acc2", "ref2", far_future - _HOUR_MS)

    assert _parse_absolute_timestamp(second) > _parse_absolute_timestamp(first)


def test_writer_preserves_the_existing_singleton_contract(anthropic_home):
    """The three historical keys must keep their meaning and their casing."""
    AA._write_hermes_oauth_credentials("acc", "ref", 1_700_000_000_000)
    on_disk = json.loads(AA._get_hermes_oauth_file().read_text(encoding="utf-8"))
    assert on_disk["accessToken"] == "acc"
    assert on_disk["refreshToken"] == "ref"
    assert on_disk["expiresAt"] == 1_700_000_000_000
    # camelCase, matching the file's existing convention rather than the
    # pool's snake_case field name.
    assert set(on_disk) == {
        "accessToken", "refreshToken", "expiresAt", SINGLETON_STAMP_KEY,
    }


# ── the gate is NOT inert: the live producer populates the key it reads ────


def test_gate_key_is_the_key_the_live_producer_writes(anthropic_home):
    """Seeding a row from a PRODUCER-WRITTEN file must stamp the entry.

    This is the anti-inertness lock.  It never hand-builds the singleton: the
    file comes from ``_write_hermes_oauth_credentials`` and the entry comes from
    ``load_pool``, so a rename on either side of the mapping fails here rather
    than passing silently on a hand-stamped fixture.
    """
    from agent.credential_pool import _SINGLETON_FRESHNESS_GROUPS

    stamp = AA._write_hermes_oauth_credentials(
        _POOL_ACCESS, _POOL_REFRESH, int(time.time() * 1000) + _HOUR_MS
    )
    entry = next(
        e for e in load_pool("anthropic").entries() if e.source == "hermes_pkce"
    )

    groups = _SINGLETON_FRESHNESS_GROUPS[("anthropic", "hermes_pkce")]
    declared_keys = {stamp_key for stamp_key, _gated in groups}
    assert declared_keys == {ENTRY_STAMP_KEY}
    for stamp_key in declared_keys:
        assert getattr(entry, stamp_key) == stamp, (
            f"the seeded entry must carry {stamp_key}; an unstamped entry makes "
            "every comparison None and the gate inert"
        )


# ── the defect: an older singleton must not clobber a fresher row ─────────


def test_seed_refuses_stale_pair(seeded_pkce):
    """Older stamp: not adopted in memory, and not persisted to disk."""
    got = seeded_pkce()
    assert got["row"].refresh_token == _POOL_REFRESH
    assert got["row"].access_token == _POOL_ACCESS
    assert got["disk"]["refresh_token"] == _POOL_REFRESH
    assert got["row"].last_refresh == ENTRY_TIME


def test_seed_refuses_stale_expiry_with_the_pair(seeded_pkce):
    """``expires_at_ms`` describes the refused access token; it goes with it."""
    fresh_ms = int(time.time() * 1000) + 4 * _HOUR_MS
    stale_ms = int(time.time() * 1000) + 9 * _HOUR_MS
    got = seeded_pkce(entry_expires_ms=fresh_ms, store_expires_ms=stale_ms)
    assert got["row"].expires_at_ms == fresh_ms, (
        "adopting the refused pair's expiry would make the kept access token "
        "look valid for longer than it is"
    )


def test_seed_refuses_stale_pair_across_timezone_offsets(seeded_pkce):
    """Instants, not ISO string ordering: +02:00 05:00 predates Z 04:00."""
    got = seeded_pkce(
        entry_stamp="2026-09-10T04:00:00Z",
        store_stamp="2026-09-10T05:00:00+02:00",
    )
    assert got["row"].refresh_token == _POOL_REFRESH
    assert got["disk"]["refresh_token"] == _POOL_REFRESH


def test_seed_refused_stale_pair_does_not_clear_exhaustion(seeded_pkce):
    """A refused pair is not a rotation, so the quarantine must survive it."""
    reset_at = time.time() + 86400
    got = seeded_pkce(mutate_entry=lambda e: dc_replace(
        e,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_reset_at=reset_at,
    ))
    assert got["row"].last_status == STATUS_EXHAUSTED
    assert got["row"].last_error_reset_at == reset_at
    assert got["disk"]["last_status"] == STATUS_EXHAUSTED


# ── preserved behavior: anything not older still lands ────────────────────


@pytest.mark.parametrize("store_stamp,entry_stamp", [
    (NEWER, ENTRY_TIME),
    (ENTRY_TIME, ENTRY_TIME),
    (None, ENTRY_TIME),
    ("not-a-timestamp", ENTRY_TIME),
    (NEWER, None),
    (None, None),
])
def test_seed_adopts_non_stale_pair(seeded_pkce, store_stamp, entry_stamp):
    """Strict older-than.

    ``(None, None)`` is the migration window: every singleton already on disk
    is unstamped, and until it is rewritten by a refresh the gate must stay
    adopt-by-default rather than freezing the row.
    """
    got = seeded_pkce(store_stamp=store_stamp, entry_stamp=entry_stamp)
    assert got["row"].refresh_token == _STORE_REFRESH
    assert got["row"].access_token == _STORE_ACCESS
    assert got["disk"]["refresh_token"] == _STORE_REFRESH


def test_seed_adopted_rotation_still_clears_exhaustion(seeded_pkce):
    """Rotation recovery: a genuinely newer pair clears the stale quarantine."""
    got = seeded_pkce(
        store_stamp=NEWER,
        mutate_entry=lambda e: dc_replace(
            e,
            last_status=STATUS_EXHAUSTED,
            last_status_at=time.time(),
            last_error_reset_at=time.time() + 86400,
        ),
    )
    assert got["row"].refresh_token == _STORE_REFRESH
    assert got["row"].last_status is None
    assert got["row"].last_error_reset_at is None


def test_seed_label_updates_despite_stale_pair(seeded_pkce):
    """Only token material is gated; the derived label keeps flowing.

    ``_upsert_entry`` leaves an already-labelled row alone, so this clears the
    label first -- otherwise the assertion would pass vacuously.
    """
    got = seeded_pkce(mutate_entry=lambda e: dc_replace(e, label=""))
    assert got["row"].label, "a refused pair must not freeze row metadata"
    assert got["row"].refresh_token == _POOL_REFRESH


# ── the real hazard loop, end to end ──────────────────────────────────────


def test_refresh_then_stale_reseed_keeps_the_rotated_pair(
    anthropic_home, monkeypatch
):
    """The defect this card exists for, driven through the production path.

    A pool-level refresh rotates the pair and commits it to the singleton.  A
    sibling process then writes an OLDER-minted pair over that file (a lost
    write-back race).  The next ``load_pool()`` must keep the rotated pair
    rather than replaying the already-consumed one.
    """
    rotated = {
        "access_token": "synthetic-rotated-access",
        "refresh_token": "synthetic-rotated-refresh",
        "expires_at_ms": int(time.time() * 1000) + _HOUR_MS,
    }
    monkeypatch.setattr(
        AA, "refresh_anthropic_oauth_pure", lambda refresh_token, **kw: rotated
    )

    _write_pkce_singleton(_POOL_ACCESS, _POOL_REFRESH, stamp=OLDER)
    pool = load_pool("anthropic")
    entry = next(e for e in pool.entries() if e.source == "hermes_pkce")

    refreshed = pool._refresh_entry(entry, force=True)
    assert refreshed is not None
    assert refreshed.refresh_token == rotated["refresh_token"]
    assert refreshed.last_refresh not in (None, "", OLDER), (
        "a refreshed entry must carry the instant its commit stamped, or the "
        "gate has nothing to compare the next singleton read against"
    )
    pool._persist()

    # Sibling process loses the write-back race and lands an older-minted pair.
    _write_pkce_singleton("synthetic-sibling-access", "synthetic-sibling-refresh",
                          stamp=OLDER)

    row = next(
        e for e in load_pool("anthropic").entries() if e.source == "hermes_pkce"
    )
    assert row.refresh_token == rotated["refresh_token"], (
        "the older-minted sibling pair replayed over a fresher rotation"
    )
    assert row.access_token == rotated["access_token"]


# ── claude_code is deliberately NOT gated (measured decision) ─────────────


def test_claude_code_is_not_declared_in_the_freshness_table():
    """Pinning the decision, not just today's behavior.

    ``claude_code`` is a BORROWED source: it is absent from
    ``credential_persistence._PERSISTABLE_PROVIDER_SOURCES``, so
    ``sanitize_borrowed_credential_payload`` strips ``access_token`` and
    ``refresh_token`` before its row reaches ``auth.json``.  Measured on this
    tree: the persisted row keeps only a ``secret_fingerprint``, and every
    ``load_pool()`` re-hydrates the live pair from the singleton.  A freshness
    gate REFUSES the singleton's token material -- which for a row that holds
    none of its own would leave it with nothing at all on a cold start.  The
    singleton is the sole authority for this source, so adopting whatever it
    holds is CORRECT, exactly as for ``minimax-oauth``.
    """
    from agent.credential_persistence import is_borrowed_credential_source
    from agent.credential_pool import _SINGLETON_FRESHNESS_GROUPS

    assert is_borrowed_credential_source("claude_code", "anthropic")
    assert not is_borrowed_credential_source("hermes_pkce", "anthropic")
    assert ("anthropic", "claude_code") not in _SINGLETON_FRESHNESS_GROUPS


def test_claude_code_still_adopts_an_older_singleton(anthropic_home):
    """The behavioral half of the decision above, through the real path."""
    _write_claude_code_singleton(_POOL_ACCESS, _POOL_REFRESH, stamp=ENTRY_TIME)
    entry = next(
        e for e in load_pool("anthropic").entries() if e.source == "claude_code"
    )
    assert entry.refresh_token == _POOL_REFRESH

    _write_claude_code_singleton(_STORE_ACCESS, _STORE_REFRESH, stamp=OLDER)
    row = next(
        e for e in load_pool("anthropic").entries() if e.source == "claude_code"
    )
    assert row.refresh_token == _STORE_REFRESH, (
        "gating claude_code would strand a row that persists no tokens of its own"
    )


def test_manual_hermes_pkce_row_is_never_gated(anthropic_home):
    """``manual:hermes_pkce`` is pool-owned with its own lifecycle.

    It has no singleton (see
    ``test_manual_hermes_pkce_refresh_does_not_create_duplicate_singleton``),
    so it must never be matched by the table.
    """
    from agent.credential_pool import (
        PooledCredential,
        _drop_stale_singleton_token_material,
    )

    existing = PooledCredential.from_dict("anthropic", {
        "id": "manual1",
        "source": "manual:hermes_pkce",
        "access_token": _POOL_ACCESS,
        "refresh_token": _POOL_REFRESH,
        ENTRY_STAMP_KEY: ENTRY_TIME,
    })
    payload = {
        "source": "manual:hermes_pkce",
        "access_token": _STORE_ACCESS,
        "refresh_token": _STORE_REFRESH,
        ENTRY_STAMP_KEY: OLDER,
    }
    kept = _drop_stale_singleton_token_material(
        existing, "anthropic", "manual:hermes_pkce", payload,
    )
    assert kept is payload

    # ...and the same payload on the seeded source IS gated, so the assertion
    # above is about the source and not about the timestamps.
    gated = _drop_stale_singleton_token_material(
        existing, "anthropic", "hermes_pkce", payload,
    )
    assert "refresh_token" not in gated
