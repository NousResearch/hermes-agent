"""Stale-pair refusal for the xAI/Nous singleton auth.json readers.

Sibling of the Codex freshness guard: a singleton mirror must never adopt an
auth.json token pair that is *older* than the one the pool entry already
holds, and must never persist that older pair over the fresher one.  Newer,
same-time, and untimestamped stores keep today's rotation-recovery behavior.

Nous is not Codex: its provider state carries an independently-timestamped
agent key plus routing metadata.  A stale OAuth pair must not block a
legitimately newer agent-key/routing update, and a newer agent-key timestamp
must not be usable to launder the stale access token in through ``agent_key``.

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


def _access(ttl: int, *, scope: str = "inference:invoke") -> str:
    """Distinct synthetic access token; ``jti`` keeps same-TTL mints unequal."""
    return _jwt({
        "sub": "synthetic",
        "scope": scope,
        "exp": int(time.time()) + ttl,
        "jti": f"synthetic-{next(_MINTED)}",
    })


def _write_store(tmp_path, payload: dict) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps(payload, indent=2))


ENTRY_TIME = "2026-09-10T04:30:00Z"
OLDER = "2026-07-22T15:28:55Z"
NEWER = "2026-09-10T05:00:00Z"


# ── xAI OAuth ──────────────────────────────────────────────────────────────


@pytest.fixture
def xai_pool(tmp_path, monkeypatch):
    """Real load_pool seeding + real select()/refresh; only the POST is stubbed."""
    import hermes_cli.auth as auth
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert auth._auth_file_path() == tmp_path / "hermes" / "auth.json"
    assert auth._global_auth_file_path() is None
    calls: list[str] = []

    def _fake_refresh(access_token, refresh_token, **kwargs):
        calls.append(refresh_token)
        return {
            # Beyond XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS (1h), otherwise the
            # rotated token is immediately "expiring" again and re-deferred.
            "access_token": _access(4 * 3600),
            "refresh_token": "synthetic-rotated-refresh",
            "id_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
            "last_refresh": NEWER,
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)
    path = auth._auth_file_path()

    def build(*, store_time=OLDER, entry_time=ENTRY_TIME, entry_ttl=30,
              source="device_code"):
        # entry_ttl=30 puts the seeded entry inside the refresh skew, so the
        # real select() path exercises refresh; 4h keeps it leasable as-is.
        # Seed the pool from a singleton that matches the entry we want, so
        # the entry is materialised by the real seeding path.
        _write_store(tmp_path, {
            "version": 1,
            "providers": {"xai-oauth": {
                "tokens": {
                    "access_token": _access(entry_ttl),
                    "refresh_token": "synthetic-pool-refresh",
                },
                "last_refresh": entry_time,
            }},
        })
        pool = load_pool("xai-oauth")
        entry = next(e for e in pool.entries() if e.source == "device_code")
        if source != "device_code":
            from dataclasses import replace

            changed = replace(entry, source=source)
            pool._replace_entry(entry, changed)
            pool._persist()
            entry = changed
        # Another writer lands a DIFFERENT pair in the singleton.
        raw = json.loads(path.read_text())
        raw["providers"]["xai-oauth"] = {
            "tokens": {
                "access_token": _access(4 * 3600),
                "refresh_token": "synthetic-store-refresh",
            },
            "last_refresh": store_time,
        }
        path.write_text(json.dumps(raw, indent=2))
        return pool, entry, path, calls

    return build


@pytest.mark.parametrize("store_time,entry_time", [
    (OLDER, ENTRY_TIME),
    ("2026-09-10T05:00:00+02:00", "2026-09-10T04:00:00Z"),  # tz-offset, not string order
])
def test_xai_sync_refuses_stale_pair(xai_pool, store_time, entry_time):
    pool, entry, path, calls = xai_pool(store_time=store_time, entry_time=entry_time)
    before = path.read_bytes()
    assert pool._sync_entry_from_auth_store(entry) is entry
    assert path.read_bytes() == before
    assert calls == []


def test_xai_stale_store_does_not_spend_stale_refresh_on_select(xai_pool):
    """Real select() path: refusal must not strand rotation recovery."""
    pool, entry, path, calls = xai_pool()
    selected = pool.select()
    assert selected is not None
    assert calls == ["synthetic-pool-refresh"]
    assert selected.refresh_token == "synthetic-rotated-refresh"


@pytest.mark.parametrize("store_time,entry_time", [
    (NEWER, ENTRY_TIME),
    (ENTRY_TIME, ENTRY_TIME),
    (None, ENTRY_TIME),
    ("invalid", ENTRY_TIME),
    (NEWER, None),
])
def test_xai_sync_adopts_non_stale_pair(xai_pool, store_time, entry_time):
    pool, entry, path, calls = xai_pool(
        store_time=store_time, entry_time=entry_time, entry_ttl=4 * 3600,
    )
    synced = pool._sync_entry_from_auth_store(entry)
    assert synced is not entry
    assert synced.refresh_token == "synthetic-store-refresh"
    persisted = next(e for e in json.loads(path.read_text())["credential_pool"]["xai-oauth"]
                     if e["id"] == entry.id)
    assert persisted["refresh_token"] == "synthetic-store-refresh"
    assert calls == []


def test_xai_manual_row_stays_independent(xai_pool):
    """Manual rows are independent credentials; the reader never touches them."""
    pool, entry, path, calls = xai_pool(
        store_time=NEWER, entry_time=ENTRY_TIME, source="manual:device_code",
    )
    before = path.read_bytes()
    assert pool._sync_entry_from_auth_store(entry) is entry
    assert path.read_bytes() == before


# ── Nous ───────────────────────────────────────────────────────────────────


@pytest.fixture
def nous_pool(tmp_path, monkeypatch):
    import hermes_cli.auth as auth
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    assert auth._auth_file_path() == tmp_path / "hermes" / "auth.json"
    assert auth._global_auth_file_path() is None
    path = auth._auth_file_path()

    entry_access = _access(3600)

    def build(*, store_obtained_at=OLDER, entry_obtained_at=ENTRY_TIME,
              store_key_at=OLDER, entry_key_at=ENTRY_TIME,
              agent_key_mirrors_access=False, store_url=None,
              source="device_code"):
        _write_store(tmp_path, {
            "version": 1,
            "providers": {"nous": {
                "access_token": entry_access,
                "refresh_token": "synthetic-pool-refresh",
                "client_id": "hermes-cli",
                "portal_base_url": "https://portal.nousresearch.com",
                "inference_base_url": "https://inference.nousresearch.com/v1",
                "token_type": "Bearer",
                "scope": "inference:invoke",
                "obtained_at": entry_obtained_at,
                "expires_at": "2026-09-10T05:30:00+00:00",
                "agent_key": entry_access,
                "agent_key_expires_at": "2026-09-10T05:30:00+00:00",
                "agent_key_obtained_at": entry_key_at,
            }},
        })
        pool = load_pool("nous")
        entry = next(e for e in pool.entries() if e.source == "device_code")
        if source != "device_code":
            from dataclasses import replace

            changed = replace(entry, source=source)
            pool._replace_entry(entry, changed)
            pool._persist()
            entry = changed
        store_access = _access(3600)
        store_agent_key = store_access if agent_key_mirrors_access else _access(7200)
        raw = json.loads(path.read_text())
        state = raw["providers"]["nous"]
        state.update({
            "access_token": store_access,
            "refresh_token": "synthetic-store-refresh",
            "obtained_at": store_obtained_at,
            "expires_at": "2026-09-10T06:30:00+00:00",
            "agent_key": store_agent_key,
            "agent_key_expires_at": "2026-09-10T06:30:00+00:00",
            "agent_key_obtained_at": store_key_at,
        })
        if store_url:
            state["inference_base_url"] = store_url
        path.write_text(json.dumps(raw, indent=2))
        return pool, entry, path, {
            "access_token": store_access,
            "agent_key": store_agent_key,
        }

    return build


@pytest.mark.parametrize("clock", ["obtained_at", "agent_key_obtained_at"])
def test_nous_sync_preserves_timestamp_only_watermark(nous_pool, clock):
    pool, entry, path, _ = nous_pool()
    raw = json.loads(path.read_text())
    state = raw["providers"]["nous"]
    for key in ("access_token", "refresh_token", "expires_at", "agent_key",
                "agent_key_expires_at"):
        state[key] = getattr(entry, key)
    state["obtained_at"] = ENTRY_TIME
    state["agent_key_obtained_at"] = ENTRY_TIME
    state[clock] = NEWER
    path.write_text(json.dumps(raw))
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.extra[clock] == NEWER
    persisted = next(e for e in json.loads(path.read_text())["credential_pool"]["nous"]
                     if e["id"] == entry.id)
    assert persisted[clock] == NEWER
    state[clock] = ENTRY_TIME
    if clock == "obtained_at":
        state["refresh_token"] = "synthetic-intermediate-refresh"
    else:
        state["agent_key"] = "synthetic-intermediate-key"
    raw = json.loads(path.read_text())
    raw["providers"]["nous"] = state
    path.write_text(json.dumps(raw))
    retained = pool._sync_nous_entry_from_auth_store(synced)
    assert retained.extra[clock] == NEWER
    assert retained.refresh_token == entry.refresh_token
    assert retained.agent_key == entry.agent_key


def test_nous_sync_refuses_stale_pair(nous_pool):
    """Older obtained_at: neither adopted in memory nor persisted to disk."""
    pool, entry, path, store = nous_pool()
    before = path.read_bytes()
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.refresh_token == "synthetic-pool-refresh"
    assert synced.access_token == entry.access_token
    assert path.read_bytes() == before


@pytest.mark.parametrize("store_obtained_at,entry_obtained_at", [
    (NEWER, ENTRY_TIME),
    (ENTRY_TIME, ENTRY_TIME),
    (None, ENTRY_TIME),
    ("invalid", ENTRY_TIME),
    (NEWER, None),
])
def test_nous_sync_adopts_non_stale_pair(nous_pool, store_obtained_at, entry_obtained_at):
    pool, entry, path, store = nous_pool(
        store_obtained_at=store_obtained_at, entry_obtained_at=entry_obtained_at,
        store_key_at=NEWER,
    )
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.refresh_token == "synthetic-store-refresh"
    assert synced.access_token == store["access_token"]
    persisted = next(e for e in json.loads(path.read_text())["credential_pool"]["nous"]
                     if e["id"] == entry.id)
    assert persisted["refresh_token"] == "synthetic-store-refresh"


def test_nous_newer_agent_key_survives_stale_oauth_pair(nous_pool):
    """The metadata boundary: a stale pair must not block a newer agent key."""
    pool, entry, path, store = nous_pool(store_obtained_at=OLDER, store_key_at=NEWER)
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.agent_key == store["agent_key"]
    assert synced.agent_key_obtained_at == NEWER
    # ...and the stale OAuth pair is still refused.
    assert synced.refresh_token == "synthetic-pool-refresh"
    assert synced.access_token == entry.access_token
    persisted = next(e for e in json.loads(path.read_text())["credential_pool"]["nous"]
                     if e["id"] == entry.id)
    assert persisted["agent_key"] == store["agent_key"]
    assert persisted["refresh_token"] == "synthetic-pool-refresh"


def test_nous_sync_refuses_stale_agent_key(nous_pool):
    pool, entry, path, store = nous_pool(store_obtained_at=OLDER, store_key_at=OLDER)
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.agent_key == entry.agent_key
    assert synced.agent_key_obtained_at == ENTRY_TIME


def test_nous_stale_access_token_cannot_ride_in_as_agent_key(nous_pool):
    """agent_key mirrors access_token on the invoke-JWT path.

    A newer agent_key timestamp must not launder the refused stale access
    token back into the runtime key.
    """
    pool, entry, path, store = nous_pool(
        store_obtained_at=OLDER, store_key_at=NEWER, agent_key_mirrors_access=True,
    )
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.agent_key != store["access_token"]
    assert synced.agent_key == entry.agent_key
    assert synced.access_token == entry.access_token
    assert synced.runtime_api_key == entry.runtime_api_key


def test_nous_routing_metadata_updates_despite_stale_pair(nous_pool):
    """Routing metadata is not token material and keeps flowing."""
    pool, entry, path, store = nous_pool(
        store_obtained_at=OLDER, store_key_at=OLDER,
        store_url="https://inference.nousresearch.com/v2",
    )
    synced = pool._sync_nous_entry_from_auth_store(entry)
    assert synced.inference_base_url == "https://inference.nousresearch.com/v2"
    assert synced.refresh_token == "synthetic-pool-refresh"


def test_nous_manual_row_stays_independent(nous_pool):
    pool, entry, path, store = nous_pool(
        store_obtained_at=NEWER, store_key_at=NEWER, source="manual:device_code",
    )
    before = path.read_bytes()
    assert pool._sync_nous_entry_from_auth_store(entry) is entry
    assert path.read_bytes() == before


@pytest.mark.parametrize("store_time,shared_time,expected", [
    (OLDER, None, "synthetic-pool-refresh"),
    (OLDER, OLDER, "synthetic-pool-refresh"),
    (OLDER, NEWER, "synthetic-store-refresh"),
    (NEWER, None, "synthetic-store-refresh"),
    (None, None, "synthetic-store-refresh"),
    ("invalid", None, "synthetic-store-refresh"),
])
def test_nous_forced_refresh_does_not_spend_stale_refresh_token(
    nous_pool, monkeypatch, store_time, shared_time, expected,
):
    """Real try_refresh_matching() path with only the token POST stubbed."""
    pool, entry, path, store = nous_pool(store_obtained_at=store_time)
    if shared_time:
        from hermes_cli.auth import _read_shared_nous_state, _write_shared_nous_state

        shared = dict(json.loads(path.read_text())["providers"]["nous"])
        shared["obtained_at"] = shared_time
        _write_shared_nous_state(shared)
        persisted_shared = _read_shared_nous_state()
        assert persisted_shared is not None
        assert persisted_shared["obtained_at"] == shared_time
    seen: list[str] = []

    def _fake_refresh(*, refresh_token, **kwargs):
        from hermes_cli.auth import AuthError

        seen.append(refresh_token)
        if refresh_token != expected:
            raise AuthError("Consumed token", provider="nous", code="invalid_grant",
                            relogin_required=True)
        return {"access_token": _access(7200),
                "refresh_token": "synthetic-rotated-refresh", "expires_in": 7200}

    monkeypatch.setattr("hermes_cli.auth._refresh_access_token", _fake_refresh)
    updated = pool.try_refresh_matching(credential_id=entry.id)
    # Upstream already skips refresh after adopting a different usable peer
    # access token. Preserve that optimization; only the refused-stale cases
    # must POST, and they must spend exactly the retained pool refresh token.
    expected_post = [expected] if expected == "synthetic-pool-refresh" else []
    expected_result = "synthetic-rotated-refresh" if expected_post else expected
    assert seen == expected_post
    assert updated is not None
    assert updated.refresh_token == expected_result
    raw = json.loads(path.read_text())
    assert raw["providers"]["nous"]["refresh_token"] == expected_result
    persisted = next(e for e in raw["credential_pool"]["nous"] if e["id"] == entry.id)
    assert persisted["refresh_token"] == expected_result


def test_nous_refresh_rereads_newer_singleton_inside_transaction(nous_pool, monkeypatch):
    from contextlib import contextmanager

    import hermes_cli.auth as auth

    pool, entry, path, _ = nous_pool()
    transaction = auth._provider_state_transaction
    seen = []

    @contextmanager
    def concurrent_winner(provider):
        raw = json.loads(path.read_text())
        raw["providers"]["nous"].update({
            "obtained_at": NEWER,
            "refresh_token": "synthetic-winner-refresh",
        })
        path.write_text(json.dumps(raw))
        with transaction(provider) as loaded:
            yield loaded

    def refresh(*, refresh_token, **kwargs):
        seen.append(refresh_token)
        return {"access_token": _access(7200), "refresh_token": "synthetic-rotated-refresh",
                "expires_in": 7200}

    monkeypatch.setattr(auth, "_provider_state_transaction", concurrent_winner)
    monkeypatch.setattr(auth, "_refresh_access_token", refresh)
    updated = pool.try_refresh_matching(credential_id=entry.id)
    assert seen == []  # New usable peer pair wins without another POST.
    assert updated is not None
    assert updated.refresh_token == "synthetic-winner-refresh"


def test_nous_exhausted_entry_select_refuses_stale_singleton(nous_pool):
    """The recovery path that clears exhaustion must not adopt an older pair."""
    from dataclasses import replace

    from agent.credential_pool import STATUS_EXHAUSTED

    pool, entry, path, store = nous_pool()
    stale_entry = replace(
        entry,
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_reset_at=time.time() + 86400,
    )
    pool._replace_entry(entry, stale_entry)
    pool._persist()
    assert pool.select() is None
    current = next(e for e in pool.entries() if e.id == entry.id)
    assert current.refresh_token == "synthetic-pool-refresh"
    assert current.last_status == STATUS_EXHAUSTED
