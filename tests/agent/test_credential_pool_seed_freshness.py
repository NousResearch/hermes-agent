"""Stale-pair refusal on the SEEDING path (``load_pool`` -> ``_upsert_entry``).

Sibling of ``test_credential_pool_singleton_freshness.py``, which covers the two
``_sync_*_entry_from_auth_store`` readers.  Those guards do not sit on the write
path that ``_seed_from_singletons`` uses: every ``load_pool()`` re-reads the
auth.json singleton and hands it to ``_upsert_entry``, which adopted (and
persisted) any differing value regardless of age.  So a stale singleton written
by another process could clobber a fresher pool row on the next process start —
the same single-use-refresh-token replay the reader guards exist to prevent,
arriving through the seeding door.

The gate lives in ``_upsert_entry`` (one comparison, per-provider timestamp
groups), so all three singleton OAuth providers — and any future one that
declares its groups — inherit it.

Everything here is synthetic: no network, no real credentials, isolated
HOME/HERMES_HOME per test.
"""

from __future__ import annotations

import base64
import itertools
import json
import time

import pytest


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


_MINTED = itertools.count()


def _access(ttl: int = 4 * 3600) -> str:
    """Distinct synthetic access token; ``jti`` keeps same-TTL mints unequal."""
    return _jwt({
        "sub": "synthetic",
        "scope": "inference:invoke",
        "exp": int(time.time()) + ttl,
        "jti": f"synthetic-{next(_MINTED)}",
    })


ENTRY_TIME = "2026-09-10T04:30:00Z"
OLDER = "2026-07-22T15:28:55Z"
NEWER = "2026-09-10T05:00:00Z"

POOL_REFRESH = "synthetic-pool-refresh"
STORE_REFRESH = "synthetic-store-refresh"

SINGLETON_PROVIDERS = ["openai-codex", "xai-oauth", "nous"]


def _singleton_state(
    provider: str,
    *,
    access: str,
    refresh: str,
    stamp,
    agent_key=None,
    key_stamp=None,
    inference_base_url: str = "https://inference.nousresearch.com/v1",
    label=None,
) -> dict:
    """auth.json ``providers.<id>`` block for a singleton OAuth provider."""
    if provider == "nous":
        state = {
            "access_token": access,
            "refresh_token": refresh,
            "client_id": "hermes-cli",
            "portal_base_url": "https://portal.nousresearch.com",
            "inference_base_url": inference_base_url,
            "token_type": "Bearer",
            "scope": "inference:invoke",
            "expires_at": "2026-09-10T05:30:00+00:00",
            "agent_key": agent_key if agent_key is not None else _access(7200),
            "agent_key_expires_at": "2026-09-10T05:30:00+00:00",
        }
        if stamp is not None:
            state["obtained_at"] = stamp
        if key_stamp is not None:
            state["agent_key_obtained_at"] = key_stamp
        if label is not None:
            state["label"] = label
        return state
    state = {"tokens": {"access_token": access, "refresh_token": refresh}}
    if stamp is not None:
        state["last_refresh"] = stamp
    if label is not None:
        state["label"] = label
    return state


def _entry_stamp(entry, provider: str):
    return entry.last_refresh if provider != "nous" else entry.extra.get("obtained_at")


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """Seed a pool row from a singleton, then rewrite the singleton and reload.

    Both halves go through the real ``load_pool`` -> ``_seed_from_singletons``
    -> ``_upsert_entry`` path — the write path this file is about.
    """
    import hermes_cli.auth as auth
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert auth._auth_file_path() == tmp_path / "hermes" / "auth.json"
    assert auth._global_auth_file_path() is None
    path = auth._auth_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(provider: str, state: dict) -> None:
        raw: dict = json.loads(path.read_text()) if path.exists() else {"version": 1}
        providers: dict = raw.setdefault("providers", {})
        providers[provider] = state
        path.write_text(json.dumps(raw, indent=2))

    def build(
        provider: str,
        *,
        store_stamp=OLDER,
        entry_stamp=ENTRY_TIME,
        store_key_stamp=OLDER,
        entry_key_stamp=ENTRY_TIME,
        agent_key_mirrors_access: bool = False,
        store_url="https://inference.nousresearch.com/v1",
        store_label=None,
        mutate_entry=None,
    ):
        _write(provider, _singleton_state(
            provider,
            access=_access(),
            refresh=POOL_REFRESH,
            stamp=entry_stamp,
            key_stamp=entry_key_stamp,
        ))
        pool = load_pool(provider)
        entry = next(e for e in pool.entries() if e.source == "device_code")
        assert entry.refresh_token == POOL_REFRESH
        if mutate_entry is not None:
            updated = mutate_entry(entry)
            pool._replace_entry(entry, updated)
            pool._persist()
            entry = updated

        # Another writer regresses (or advances) the singleton.
        store_access = _access()
        store_key = store_access if agent_key_mirrors_access else _access(7200)
        _write(provider, _singleton_state(
            provider,
            access=store_access,
            refresh=STORE_REFRESH,
            stamp=store_stamp,
            agent_key=store_key,
            key_stamp=store_key_stamp,
            inference_base_url=store_url,
            label=store_label,
        ))

        reloaded = load_pool(provider)
        row = next(e for e in reloaded.entries() if e.id == entry.id)
        disk = next(
            r for r in json.loads(path.read_text())["credential_pool"][provider]
            if r["id"] == entry.id
        )
        return {
            "provider": provider,
            "entry": entry,
            "row": row,
            "disk": disk,
            "store_access": store_access,
            "store_agent_key": store_key,
            "path": path,
        }

    return build


# ── the defect: an older singleton must not clobber a fresher row ──────────


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
def test_seed_refuses_stale_pair(seeded, provider):
    """Older stamp: not adopted in memory, and not persisted to disk."""
    got = seeded(provider)
    assert got["row"].refresh_token == POOL_REFRESH
    assert got["row"].access_token == got["entry"].access_token
    assert got["disk"]["refresh_token"] == POOL_REFRESH
    assert _entry_stamp(got["row"], provider) == ENTRY_TIME


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
def test_seed_refuses_stale_pair_across_timezone_offsets(seeded, provider):
    """Instants, not ISO string ordering: +02:00 05:00 predates Z 04:00."""
    got = seeded(
        provider,
        entry_stamp="2026-09-10T04:00:00Z",
        store_stamp="2026-09-10T05:00:00+02:00",
        entry_key_stamp="2026-09-10T04:00:00Z",
        store_key_stamp="2026-09-10T05:00:00+02:00",
    )
    assert got["row"].refresh_token == POOL_REFRESH
    assert got["disk"]["refresh_token"] == POOL_REFRESH


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
def test_seed_refused_stale_pair_does_not_clear_exhaustion(seeded, provider):
    """A refused pair is not a rotation, so the quarantine must survive it."""
    from dataclasses import replace

    from agent.credential_pool import STATUS_EXHAUSTED

    reset_at = time.time() + 86400
    got = seeded(provider, mutate_entry=lambda e: replace(
        e,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_reset_at=reset_at,
    ))
    assert got["row"].last_status == STATUS_EXHAUSTED
    assert got["row"].last_error_reset_at == reset_at
    assert got["disk"]["last_status"] == STATUS_EXHAUSTED


# ── preserved behavior: anything not older still lands ────────────────────


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
@pytest.mark.parametrize("store_stamp,entry_stamp", [
    (NEWER, ENTRY_TIME),
    (ENTRY_TIME, ENTRY_TIME),
    (None, ENTRY_TIME),
    ("not-a-timestamp", ENTRY_TIME),
    (NEWER, None),
])
def test_seed_adopts_non_stale_pair(seeded, provider, store_stamp, entry_stamp):
    got = seeded(
        provider,
        store_stamp=store_stamp, entry_stamp=entry_stamp,
        store_key_stamp=store_stamp, entry_key_stamp=entry_stamp,
    )
    assert got["row"].refresh_token == STORE_REFRESH
    assert got["row"].access_token == got["store_access"]
    assert got["disk"]["refresh_token"] == STORE_REFRESH


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
def test_seed_adopted_rotation_still_clears_exhaustion(seeded, provider):
    """Rotation recovery: a genuinely newer pair clears the stale quarantine."""
    from dataclasses import replace

    from agent.credential_pool import STATUS_EXHAUSTED

    got = seeded(
        provider,
        store_stamp=NEWER, store_key_stamp=NEWER,
        mutate_entry=lambda e: replace(
            e,
            last_status=STATUS_EXHAUSTED,
            last_status_at=time.time(),
            last_error_reset_at=time.time() + 86400,
        ),
    )
    assert got["row"].refresh_token == STORE_REFRESH
    assert got["row"].last_status is None
    assert got["row"].last_error_reset_at is None


@pytest.mark.parametrize("provider", ["openai-codex", "nous"])
def test_seed_custom_label_updates_despite_stale_pair(seeded, provider):
    """Only token material is freshness-gated; a custom label keeps flowing.

    ``_upsert_entry`` leaves an already-labelled row alone, so this clears the
    label first — otherwise the assertion would pass vacuously.  xAI is absent
    because its seeding branch never carries ``state["label"]`` (pre-existing,
    unrelated to freshness); ``base_url`` covers it below.
    """
    from dataclasses import replace

    got = seeded(
        provider,
        store_label="renamed-by-user",
        mutate_entry=lambda e: replace(e, label=""),
    )
    assert got["row"].label == "renamed-by-user"
    assert got["row"].refresh_token == POOL_REFRESH


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
def test_seed_base_url_updates_despite_stale_pair(seeded, provider):
    """Routing (``base_url``) is not token material and keeps being repaired.

    Nous is absent: its seeding branch carries ``inference_base_url`` instead —
    see ``test_seed_routing_metadata_updates_despite_stale_pair``.
    """
    from dataclasses import replace

    got = seeded(provider, mutate_entry=lambda e: replace(e, base_url="https://wrong.invalid"))
    assert got["row"].base_url != "https://wrong.invalid"
    assert got["row"].refresh_token == POOL_REFRESH


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
def test_seed_manual_row_stays_independent(seeded, provider, tmp_path):
    """``manual:*`` rows are separate credentials; seeding must not touch them."""
    from dataclasses import replace

    import hermes_cli.auth as auth
    from agent.credential_pool import load_pool

    got = seeded(provider, store_stamp=NEWER, store_key_stamp=NEWER,
                 mutate_entry=lambda e: replace(e, source="manual:device_code"))
    manual = next(
        e for e in load_pool(provider).entries() if e.id == got["entry"].id
    )
    assert manual.source == "manual:device_code"
    assert manual.refresh_token == POOL_REFRESH
    assert manual.access_token == got["entry"].access_token
    assert auth._auth_file_path() == tmp_path / "hermes" / "auth.json"


@pytest.mark.parametrize("provider", SINGLETON_PROVIDERS)
def test_upsert_never_gates_a_manual_source(provider):
    """Contract on the shared helper: only singleton-seeded rows are gated.

    No seeding site emits a ``manual:*`` payload today (``_upsert_entry`` matches
    on exact source, so a ``device_code`` payload can never reach a manual row),
    which makes the source condition unreachable from
    ``_seed_from_singletons``.  It is asserted directly here so the condition is
    proven rather than merely present: a manual row is an independent credential
    with its own refresh-token lifecycle and must adopt whatever its own writer
    provides, however that writer stamps it.
    """
    from agent.credential_pool import (
        PooledCredential,
        _drop_stale_singleton_token_material,
    )

    stamp_key = "obtained_at" if provider == "nous" else "last_refresh"
    existing = PooledCredential.from_dict(provider, {
        "id": "manual1",
        "source": "manual:device_code",
        "access_token": _access(),
        "refresh_token": POOL_REFRESH,
        stamp_key: ENTRY_TIME,
    })
    payload = {
        "source": "manual:device_code",
        "access_token": _access(),
        "refresh_token": STORE_REFRESH,
        stamp_key: OLDER,
    }
    kept = _drop_stale_singleton_token_material(
        existing, provider, "manual:device_code", payload,
    )
    assert kept is payload

    # ...and the same payload on the seeded source IS gated, so the assertion
    # above is about the source and not about the timestamps.
    gated = _drop_stale_singleton_token_material(
        existing, provider, "device_code", payload,
    )
    assert "refresh_token" not in gated


# ── Nous: per-group boundary survives on the seeding path too ─────────────


def test_seed_newer_agent_key_survives_stale_oauth_pair(seeded):
    got = seeded("nous", store_stamp=OLDER, store_key_stamp=NEWER)
    assert got["row"].agent_key == got["store_agent_key"]
    assert got["row"].extra.get("agent_key_obtained_at") == NEWER
    assert got["row"].refresh_token == POOL_REFRESH
    assert got["row"].access_token == got["entry"].access_token
    assert got["disk"]["agent_key"] == got["store_agent_key"]
    assert got["disk"]["refresh_token"] == POOL_REFRESH


def test_seed_refuses_stale_agent_key(seeded):
    got = seeded("nous", store_stamp=NEWER, store_key_stamp=OLDER)
    assert got["row"].agent_key == got["entry"].agent_key
    assert got["row"].extra.get("agent_key_obtained_at") == ENTRY_TIME
    # ...while the newer OAuth pair still lands.
    assert got["row"].refresh_token == STORE_REFRESH


def test_seed_stale_access_token_cannot_ride_in_as_agent_key(seeded):
    """Invoke-JWT path mirrors agent_key onto access_token.

    A newer ``agent_key_obtained_at`` must not launder the refused stale
    access token into ``runtime_api_key``.
    """
    got = seeded(
        "nous",
        store_stamp=OLDER, store_key_stamp=NEWER,
        agent_key_mirrors_access=True,
    )
    assert got["row"].agent_key != got["store_access"]
    assert got["row"].agent_key == got["entry"].agent_key
    assert got["row"].access_token == got["entry"].access_token
    assert got["row"].runtime_api_key == got["entry"].runtime_api_key


def test_seed_routing_metadata_updates_despite_stale_pair(seeded):
    got = seeded(
        "nous",
        store_stamp=OLDER, store_key_stamp=OLDER,
        store_url="https://inference.nousresearch.com/v2",
    )
    assert got["row"].inference_base_url == "https://inference.nousresearch.com/v2"
    assert got["row"].refresh_token == POOL_REFRESH
    assert got["row"].agent_key == got["entry"].agent_key
